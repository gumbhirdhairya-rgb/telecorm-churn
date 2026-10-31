"""
Export Final Processed Datasets with Selected Features & Serialized Pipeline
Prepares clean, leakage-free matrices for Step 5 (Model Development & Training).
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.fe_utils import engineer_telecom_features

# 1. Load raw data and split
df_raw = pd.read_csv('data/raw/cell2cell.csv')
X_raw = df_raw.drop(columns=['customer', 'calibrat', 'churndep', 'churn'])
y = df_raw['churn'].astype(int)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)

# 2. Engineer domain features on both partitions
X_train_eng = engineer_telecom_features(X_train_raw)
X_test_eng = engineer_telecom_features(X_test_raw)

# 3. Load final selected feature list
with open('data/processed/selected_features.json', 'r') as f:
    selected_meta = json.load(f)

selected_features = selected_meta['selected_features']
print(f"[+] Applying selected feature set: {len(selected_features)} features")

X_train_sel = X_train_eng[selected_features].copy()
X_test_sel = X_test_eng[selected_features].copy()

# 4. Identify continuous vs binary features within selected set
bin_cols = [c for c in selected_features if X_train_sel[c].dropna().isin([0, 1]).all()]
num_cols = [c for c in selected_features if c not in bin_cols]

print(f"[+] Continuous / Numerical: {len(num_cols)} features")
print(f"[+] Binary Categorical    : {len(bin_cols)} features")

# 5. Build ColumnTransformer pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

bin_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('passthrough', 'passthrough')
])

final_preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
        ('bin', bin_pipeline, bin_cols)
    ],
    remainder='drop'
)

# 6. Fit ONLY on X_train_sel (Zero Data Leakage)
final_preprocessor.fit(X_train_sel)

X_train_final_arr = final_preprocessor.transform(X_train_sel)
X_test_final_arr = final_preprocessor.transform(X_test_sel)

feature_order_out = num_cols + bin_cols
X_train_final = pd.DataFrame(X_train_final_arr, columns=feature_order_out, index=X_train_sel.index)
X_test_final = pd.DataFrame(X_test_final_arr, columns=feature_order_out, index=X_test_sel.index)

# 7. Save final datasets and model pipeline
PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'models'

X_train_final.to_csv(f"{PROCESSED_DIR}/X_train_final.csv", index=False)
X_test_final.to_csv(f"{PROCESSED_DIR}/X_test_final.csv", index=False)
y_train.to_csv(f"{PROCESSED_DIR}/y_train_final.csv", index=False)
y_test.to_csv(f"{PROCESSED_DIR}/y_test_final.csv", index=False)

joblib.dump(final_preprocessor, f"{MODELS_DIR}/final_preprocessor.joblib")

final_metadata = {
    "num_train_samples": len(X_train_final),
    "num_test_samples": len(X_test_final),
    "feature_count": len(feature_order_out),
    "numerical_count": len(num_cols),
    "binary_count": len(bin_cols),
    "continuous_features": num_cols,
    "binary_features": bin_cols,
    "pipeline_path": "models/final_preprocessor.joblib",
    "train_churn_ratio": float(y_train.mean()),
    "test_churn_ratio": float(y_test.mean())
}

with open(f"{PROCESSED_DIR}/final_dataset_metadata.json", "w") as f:
    json.dump(final_metadata, f, indent=4)

print(f"[+] Saved X_train_final.csv ({X_train_final.shape})")
print(f"[+] Saved X_test_final.csv  ({X_test_final.shape})")
print(f"[+] Saved final_preprocessor.joblib")
print(f"[+] Saved final_dataset_metadata.json")
