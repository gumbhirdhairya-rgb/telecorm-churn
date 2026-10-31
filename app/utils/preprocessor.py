"""
app/utils/preprocessor.py
Feature engineering, schema validation, and inference-time transformation pipelines.
"""

import pandas as pd
import numpy as np

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applies verified telecom domain feature engineering from Step 4:
    1. handset_age_ratio: eqpdays / (months * 30.4375 + 1.0)
    2. cost_per_mou: revenue / (mou + 1.0)
    3. overage_burden: overage / (recchrge + 1.0)
    4. usage_drop_severe: ((changem < -20) & (mou > 50)).astype(int)
    5. network_friction_rate: dropblk / (mou + 1.0)
    
    If 'dropblk' is missing, it dynamically reconstructs dropblk = dropvce + blckvce
    (as verified in Step 4.2 audit: 96.7% exact linear sum).
    """
    df = data.copy()

    # Reconstruct dropblk if absent but dropvce and blckvce are present
    if 'dropblk' not in df.columns and 'dropvce' in df.columns and 'blckvce' in df.columns:
        df['dropblk'] = df['dropvce'] + df['blckvce']
    elif 'dropblk' not in df.columns:
        df['dropblk'] = 0.0

    # Ensure required base columns exist with safe defaults
    for col in ['eqpdays', 'months', 'revenue', 'mou', 'recchrge', 'overage', 'changem']:
        if col not in df.columns:
            df[col] = 0.0

    # 1. Handset Obsolescence Ratio
    df['handset_age_ratio'] = df['eqpdays'] / (df['months'] * 30.4375 + 1.0)

    # 2. Cost per Minute (Unit billing pressure)
    df['cost_per_mou'] = df['revenue'] / (df['mou'] + 1.0)

    # 3. Overage Burden Ratio
    df['overage_burden'] = df['overage'] / (df['recchrge'] + 1.0)

    # 4. Severe Usage Drop Indicator (Binary flag)
    df['usage_drop_severe'] = ((df['changem'] < -20) & (df['mou'] > 50)).astype(int)

    # 5. Network Quality Friction Rate
    df['network_friction_rate'] = df['dropblk'] / (df['mou'] + 1.0)

    return df

def prepare_input_dataframe(df_input: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Transforms any raw or preprocessed customer DataFrame into the exact 63-column
    feature matrix expected by best_xgboost.joblib and calibrated models.
    """
    selected_features = artifacts["selected_features"]
    expected_order = artifacts["expected_order"]
    preprocessor = artifacts["preprocessor"]
    baseline_defaults = artifacts["baseline_defaults"]

    df = df_input.copy()

    # Drop target and identifier columns if present (Zero Target Leakage)
    leakage_cols = ['churn', 'churndep', 'customer', 'calibrat']
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors='ignore')

    # Case 1: Already preprocessed 63-column matrix (e.g. X_test_final format)
    if set(expected_order).issubset(set(df.columns)):
        return df[expected_order].copy()

    # Case 2: Raw data (needs feature engineering + preprocessor transformation)
    # Populate missing secondary features with verified baseline medians/modes
    for col, default_val in baseline_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    # Apply domain feature engineering
    df_eng = engineer_features(df)

    # Select exactly the 63 features expected by preprocessor.feature_names_in_
    # Ensure all 63 columns exist
    for col in selected_features:
        if col not in df_eng.columns:
            df_eng[col] = baseline_defaults.get(col, 0.0)

    df_selected = df_eng[selected_features].copy()

    # Run through final_preprocessor ColumnTransformer
    X_proc_arr = preprocessor.transform(df_selected)

    # Reconstruct DataFrame with exact expected order (num_cols + bin_cols)
    df_transformed = pd.DataFrame(X_proc_arr, columns=expected_order, index=df_selected.index)

    return df_transformed
