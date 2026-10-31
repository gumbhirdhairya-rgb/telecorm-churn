"""
Builds the complete, step-by-step 02_data_preprocessing.ipynb notebook
Incorporating all requirements:
- Leakage columns removed (customer, calibrat, churndep)
- Features & target separated
- Accurate feature typing (32 binary flags vs 34 numerical)
- 80:20 Stratified train-test split (stratify=y, random_state=42)
- Median imputation for numerical features
- Selective scaling (StandardScaler on continuous only; passthrough on binary flags)
- Strict leakage avoidance (fit on train only, transform train & test)
- Serializing processed datasets and pipeline artifact
"""

import json

def build_full_preprocessing_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_md(source):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        })

    def add_code(source):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
        })

    # Header Markdown
    add_md("""# 🧹 Data Cleaning & Preprocessing Pipeline
## UCS321 - Machine Learning Project | Problem Statement 9 (Telecom Customer Churn)
---

### 🎯 Pipeline Requirements & Architecture
1. **Target Leakage & Irrelevant Column Removal**:
   - `customer`: Arbitrary identifier (zero generalizable predictive signal).
   - `calibrat`: Tournament 50/50 training sample indicator (direct artificial sampling artifact).
   - `churndep`: Target duplicate with 31,047 missing values.
2. **Feature & Target Separation**:
   - Feature Matrix ($X$): 66 predictor attributes.
   - Target Vector ($y$): Binary ground truth `churn` ($0$ = Retained, $1$ = Churned).
3. **Accurate Feature Typing**:
   - **32 Binary Categorical Flags** ($0$ or $1$, representing demographic flags, service options, lifestyle).
   - **34 Continuous / Count Numerical Variables** (usage, billing amounts, tenure, device age).
4. **80:20 Stratified Train-Test Split**:
   - Preserving class proportion (~71% non-churn vs ~29% churn) across both partitions.
5. **Leakage-Free Preprocessing (`ColumnTransformer`)**:
   - Continuous Pipeline: Robust Median Imputation (`SimpleImputer(strategy='median')`) + Standardization (`StandardScaler()`).
   - Binary Pipeline: Kept unscaled (`passthrough`) to maintain sparsity and true binary semantics.
   - **Strict Fit-Transform Rule**: Preprocessors are fit **ONLY** on the training set, then applied to the test set.
6. **Artifact Persistence**:
   - Export processed training & testing datasets to `../data/processed/`.
   - Export serialized transformer pipeline to `../models/preprocessor.joblib`.
""")

    # Cell 1: Environment & Imports
    add_md("## 📦 Step 1: Environment Setup & Library Imports")
    add_code("""import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

print("[+] Essential preprocessing libraries successfully imported!")
""")

    # Cell 2: Data Ingestion
    add_md("## 📂 Step 2: Data Ingestion & Quality Audit")
    add_code("""DATA_PATH = '../data/raw/cell2cell.csv'
df_raw = pd.read_csv(DATA_PATH)

print(f"📊 Dataset Dimensions: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
print(f"🔍 Exact Duplicate Rows: {df_raw.duplicated().sum()}")
df_raw.head(3)
""")

    # Cell 3: Leakage Prevention & Target Separation
    add_md("""## 🛡️ Step 3: Prevent Target Leakage & Separate Features ($X$) and Target ($y$)

### Critical Machine Learning Safeguards:
1. `customer`: Random ID number. If kept, decision trees could split on specific customer IDs instead of learning behavioral patterns.
2. `calibrat`: A competition artifact indicating the 50/50 training subset. Using this in real-world deployment would be impossible and constitutes direct data leakage.
3. `churndep`: An incomplete duplicate of `churn` with 31,047 missing values.

We drop all three columns and isolate `churn` as the single ground truth target vector.""")
    add_code("""LEAKAGE_AND_ID_COLS = ['customer', 'calibrat', 'churndep']
TARGET_COL = 'churn'

# Verify presence
for col in LEAKAGE_AND_ID_COLS + [TARGET_COL]:
    assert col in df_raw.columns, f"Missing required column: {col}"

# Separate X and y
y = df_raw[TARGET_COL].copy().astype(int)
X = df_raw.drop(columns=LEAKAGE_AND_ID_COLS + [TARGET_COL]).copy()

print(f"[+] Target variable shape (y): {y.shape}")
print(f"[+] Feature matrix shape (X): {X.shape[0]:,} rows x {X.shape[1]} features")
print(f"[+] Removed leakage columns: {LEAKAGE_AND_ID_COLS}")
""")

    # Cell 4: Feature Typing
    add_md("""## 🔬 Step 4: Correct Feature Identification (Numerical vs Categorical Flags)

In telecom data, binary categorical indicators (e.g. `children`, `credita`, `refurb`, `webcap`, `retcall`) should **NOT** be normalized or scaled with `StandardScaler`. Standardizing $0/1$ flags to arbitrary floats:
1. Destroys interpretability in logistic regression coefficients and decision tree splits.
2. Destroys natural sparsity.

We categorize all 66 features into:
- **32 Binary Categorical Features** ($0$ or $1$)
- **34 Continuous / Count Numerical Features**""")
    add_code("""binary_cols = [c for c in X.columns if X[c].dropna().isin([0, 1]).all()]
numerical_cols = [c for c in X.columns if c not in binary_cols]

print(f"[+] Identified Binary Flags: {len(binary_cols)} features")
print(f"[+] Identified Continuous / Numerical: {len(numerical_cols)} features")
assert len(binary_cols) + len(numerical_cols) == X.shape[1], "Feature count mismatch!"

# Display categorization summary
feature_summary = pd.DataFrame({
    'Feature Type': ['Binary Categorical Flags (0/1)', 'Continuous / Numerical'],
    'Count': [len(binary_cols), len(numerical_cols)],
    'Examples': [
        ', '.join(binary_cols[:5]) + '...',
        ', '.join(numerical_cols[:5]) + '...'
    ]
})
feature_summary
""")

    # Cell 5: Stratified Train-Test Split
    add_md("""## ✂️ Step 5: 80:20 Stratified Train-Test Split

### ⚠️ Golden Rule of Machine Learning:
To prevent **Data Leakage**, the train-test split must occur **BEFORE** any imputation or scaling.
- If we imputed or scaled before splitting, information from the test set (mean, median, standard deviation) would leak into the training process.
- We use **stratified sampling (`stratify=y`)** so that both splits contain the exact same churn ratio (~71% non-churn vs ~29% churn).""")
    add_code("""X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"📊 Training Set Dimensions : {X_train.shape[0]:,} rows ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"📊 Testing Set Dimensions  : {X_test.shape[0]:,} rows ({X_test.shape[0]/len(X)*100:.1f}%)")

# Verify stratification consistency
strat_df = pd.DataFrame({
    'Original Dataset (%)': y.value_counts(normalize=True) * 100,
    'Train Split (%)': y_train.value_counts(normalize=True) * 100,
    'Test Split (%)': y_test.value_counts(normalize=True) * 100
})
strat_df.index = ['Retained (0)', 'Churned (1)']
strat_df.round(2)
""")

    # Cell 6: Preprocessing Pipeline Construction
    add_md("""## ⚙️ Step 6: Construct Scikit-Learn `ColumnTransformer` Pipeline

### Strategy:
1. **Numerical Features (34)**:
   - `SimpleImputer(strategy='median')`: Replaces missing values with the median of the feature (robust to heavy skews and outliers in telecom revenue/minutes).
   - `StandardScaler()`: Centers to mean 0 and scales to unit variance.
2. **Binary Features (32)**:
   - `passthrough`: Preserved untouched as $\{0, 1\}$ integer values to retain direct business interpretability and optimal decision-tree thresholding.""")
    add_code("""num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

bin_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Safeguard for any unseen missing flags
    ('passthrough', 'passthrough')
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numerical_cols),
        ('bin', bin_pipeline, binary_cols)
    ],
    remainder='drop'
)

print("[+] ColumnTransformer architecture configured:")
preprocessor
""")

    # Cell 7: Fit on Train and Transform
    add_md("""## 🔒 Step 7: Leakage-Free Preprocessor Fitting & Transformation

### 🛡️ Preventing Data Leakage:
- `preprocessor.fit(X_train)`: Calculates median, mean ($\mu$), and standard deviation ($\sigma$) **STRICTLY on the training partition**.
- `preprocessor.transform(X_train)`: Transforms the training set.
- `preprocessor.transform(X_test)`: Transforms the test set using the **training parameters** without learning anything from the test set.""")
    add_code("""# Fit strictly on X_train
preprocessor.fit(X_train)

# Transform both partitions
X_train_proc_arr = preprocessor.transform(X_train)
X_test_proc_arr = preprocessor.transform(X_test)

# Preserve column ordering and clean DataFrame structure
feature_names_out = numerical_cols + binary_cols

X_train_processed = pd.DataFrame(X_train_proc_arr, columns=feature_names_out, index=X_train.index)
X_test_processed = pd.DataFrame(X_test_proc_arr, columns=feature_names_out, index=X_test.index)

print(f"[+] Processed X_train shape: {X_train_processed.shape}")
print(f"[+] Processed X_test shape : {X_test_processed.shape}")
print(f"[+] Null values remaining in X_train: {X_train_processed.isnull().sum().sum()}")
print(f"[+] Null values remaining in X_test : {X_test_processed.isnull().sum().sum()}")
""")

    # Cell 8: Verify Scaling & Distribution
    add_md("""## 📊 Step 8: Preprocessing Quality & Distribution Verification
We inspect sample numerical features (now standardized) vs binary features (preserved in $\{0, 1\}$).""")
    add_code("""fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Numerical feature after scaling
sns.histplot(X_train_processed['revenue'], kde=True, ax=axes[0], color='#1f77b4', bins=40)
axes[0].set_title("Standardized Numerical Feature ('revenue')", fontweight='bold')
axes[0].set_xlabel("Standardized Value (Z-Score)")

# Binary feature preserved
sns.countplot(x=X_train_processed['webcap'], ax=axes[1], palette=['#2b5c8f', '#d95f02'])
axes[1].set_title("Preserved Binary Indicator ('webcap')", fontweight='bold')
axes[1].set_xlabel("Binary Value (0 = No, 1 = Yes)")

plt.tight_layout()
plt.show()
""")

    # Cell 9: Artifact Serialization & Export
    add_md("""## 💾 Step 9: Serialize Processed Datasets & Preprocessing Pipeline

We save all processed partitions and serialized preprocessor for seamless integration into Model Training.""")
    add_code("""PROCESSED_DIR = '../data/processed'
MODELS_DIR = '../models'
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Save processed feature matrices and target vectors
X_train_processed.to_csv(f"{PROCESSED_DIR}/X_train_processed.csv", index=False)
X_test_processed.to_csv(f"{PROCESSED_DIR}/X_test_processed.csv", index=False)
y_train.to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{PROCESSED_DIR}/y_test.csv", index=False)

# 2. Save fitted preprocessor pipeline with joblib
PIPELINE_PATH = f"{MODELS_DIR}/preprocessor.joblib"
joblib.dump(preprocessor, PIPELINE_PATH)

# 3. Save feature metadata JSON
metadata = {
    "total_samples": len(df_raw),
    "train_samples": len(X_train_processed),
    "test_samples": len(X_test_processed),
    "features_total": len(feature_names_out),
    "numerical_features_count": len(numerical_cols),
    "binary_features_count": len(binary_cols),
    "numerical_features": numerical_cols,
    "binary_features": binary_cols,
    "imputation_strategy": "median",
    "scaling_strategy": "StandardScaler (numerical only)",
    "target_variable": TARGET_COL,
    "train_churn_pct": float(y_train.mean() * 100),
    "test_churn_pct": float(y_test.mean() * 100)
}

with open(f"{PROCESSED_DIR}/preprocessing_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print(f"[+] Saved X_train_processed.csv : {os.path.getsize(f'{PROCESSED_DIR}/X_train_processed.csv') / (1024**2):.2f} MB")
print(f"[+] Saved X_test_processed.csv  : {os.path.getsize(f'{PROCESSED_DIR}/X_test_processed.csv') / (1024**2):.2f} MB")
print(f"[+] Saved y_train.csv & y_test.csv")
print(f"[+] Serialized pipeline to {PIPELINE_PATH}")
print(f"[+] Saved metadata to {PROCESSED_DIR}/preprocessing_metadata.json")
print("[+] Step 3: Data Cleaning & Preprocessing COMPLETED SUCCESSFULLY!")
""")

    # Write notebook file
    with open("notebooks/02_data_preprocessing.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("[+] Successfully generated notebooks/02_data_preprocessing.ipynb")

if __name__ == '__main__':
    build_full_preprocessing_notebook()
