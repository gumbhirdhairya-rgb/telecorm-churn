"""
Script for Step 6.6: Sampling-Aware Validation of Cell2Cell Churn
Investigating calibrat=1 (tournament balanced) vs calibrat=0 (natural holdout)
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load original raw data with calibrat preserved for indexing only
df_raw = pd.read_csv('data/raw/cell2cell.csv')

# Verify the sampling partitions
c1 = df_raw[df_raw['calibrat'] == 1]
c0 = df_raw[df_raw['calibrat'] == 0]

print("=== 1. DATASET SAMPLING STRUCTURE ===")
print(f"calibrat == 1: {len(c1):,} rows | Churners: {(c1['churn']==1).sum():,} ({(c1['churn']==1).mean()*100:.2f}%) | Retained: {(c1['churn']==0).sum():,} ({(c1['churn']==0).mean()*100:.2f}%)")
print(f"calibrat == 0: {len(c0):,} rows | Churners: {(c0['churn']==1).sum():,} ({(c0['churn']==1).mean()*100:.2f}%) | Retained: {(c0['churn']==0).sum():,} ({(c0['churn']==0).mean()*100:.2f}%)")
print(f"Total Cohort : {len(df_raw):,} rows | Churners: {(df_raw['churn']==1).sum():,} ({(df_raw['churn']==1).mean()*100:.2f}%)")

# Compare feature distributions between c1 and c0
key_features = ['revenue', 'mou', 'recchrge', 'overage', 'eqpdays', 'months']
dist_comparison = []
for f in key_features:
    dist_comparison.append({
        'Feature': f,
        'Mean_Calibrat_1': round(float(c1[f].mean()), 2),
        'Mean_Calibrat_0': round(float(c0[f].mean()), 2),
        'Median_Calibrat_1': round(float(c1[f].median()), 2),
        'Median_Calibrat_0': round(float(c0[f].median()), 2),
        'Std_Calibrat_1': round(float(c1[f].std()), 2),
        'Std_Calibrat_0': round(float(c0[f].std()), 2),
    })

dist_df = pd.DataFrame(dist_comparison)
print("\n=== FEATURE DISTRIBUTION COMPARISON (calibrat=1 vs calibrat=0) ===")
print(dist_df.to_string(index=False))

# 2. Check the test split representation
# Replicate the exact 80:20 split to identify which test samples belong to calibrat=1 vs calibrat=0
X_raw = df_raw.drop(columns=['customer', 'calibrat', 'churndep', 'churn'])
y = df_raw['churn'].astype(int)
calibrat_series = df_raw['calibrat']

X_tr_raw, X_te_raw, y_tr, y_te, cal_tr, cal_te = train_test_split(
    X_raw, y, calibrat_series, test_size=0.20, random_state=42, stratify=y
)

print("\n=== TEST SET (14,210 samples) COMPOSITION BY CALIBRAT ===")
print(f"Test samples with calibrat == 1: {(cal_te == 1).sum():,} (Churners: {((cal_te == 1) & (y_te == 1)).sum():,}, {(y_te[cal_te == 1].mean())*100:.2f}%)")
print(f"Test samples with calibrat == 0: {(cal_te == 0).sum():,} (Churners: {((cal_te == 0) & (y_te == 1)).sum():,}, {(y_te[cal_te == 0].mean())*100:.2f}%)")

# 3. Load trained Champion XGBoost and processed test features
xgb_model = joblib.load('models/best_xgboost.joblib')
X_test_final = pd.read_csv('data/processed/X_test_final.csv')
y_test_final = pd.read_csv('data/processed/y_test_final.csv')['churn']

# Predict probabilities on full test set
test_probs = xgb_model.predict_proba(X_test_final)[:, 1]

# Evaluate on Full Test Set
roc_full = roc_auc_score(y_test_final, test_probs)
pr_full = average_precision_score(y_test_final, test_probs)
y_pred_full = (test_probs >= 0.50).astype(int)
f1_full = f1_score(y_test_final, y_pred_full)
prec_full = precision_score(y_test_final, y_pred_full)
rec_full = recall_score(y_test_final, y_pred_full)

# Evaluate on calibrat == 1 subset of test set
mask_c1 = (cal_te.values == 1)
y_test_c1 = y_test_final[mask_c1]
probs_c1 = test_probs[mask_c1]
roc_c1 = roc_auc_score(y_test_c1, probs_c1)
pr_c1 = average_precision_score(y_test_c1, probs_c1)
y_pred_c1 = (probs_c1 >= 0.50).astype(int)
f1_c1 = f1_score(y_test_c1, y_pred_c1)
prec_c1 = precision_score(y_test_c1, y_pred_c1)
rec_c1 = recall_score(y_test_c1, y_pred_c1)

# Evaluate on calibrat == 0 subset of test set
mask_c0 = (cal_te.values == 0)
y_test_c0 = y_test_final[mask_c0]
probs_c0 = test_probs[mask_c0]
roc_c0 = roc_auc_score(y_test_c0, probs_c0)
pr_c0 = average_precision_score(y_test_c0, probs_c0)
y_pred_c0 = (probs_c0 >= 0.50).astype(int)
f1_c0 = f1_score(y_test_c0, y_pred_c0)
prec_c0 = precision_score(y_test_c0, y_pred_c0)
rec_c0 = recall_score(y_test_c0, y_pred_c0)

eval_summary = pd.DataFrame([
    {
        'Cohort': 'Full Combined Test Set',
        'Sample_Size': len(y_test_final),
        'Positive_Count': int(y_test_final.sum()),
        'Churn_Prevalence': f"{y_test_final.mean()*100:.2f}%",
        'ROC_AUC': round(roc_full, 4),
        'PR_AUC': round(pr_full, 4),
        'Precision (tau=0.5)': round(prec_full, 4),
        'Recall (tau=0.5)': round(rec_full, 4),
        'F1_Score (tau=0.5)': round(f1_full, 4)
    },
    {
        'Cohort': 'calibrat == 1 (Balanced Cohort)',
        'Sample_Size': int(mask_c1.sum()),
        'Positive_Count': int(y_test_c1.sum()),
        'Churn_Prevalence': f"{y_test_c1.mean()*100:.2f}%",
        'ROC_AUC': round(roc_c1, 4),
        'PR_AUC': round(pr_c1, 4),
        'Precision (tau=0.5)': round(prec_c1, 4),
        'Recall (tau=0.5)': round(rec_c1, 4),
        'F1_Score (tau=0.5)': round(f1_c1, 4)
    },
    {
        'Cohort': 'calibrat == 0 (Natural Holdout)',
        'Sample_Size': int(mask_c0.sum()),
        'Positive_Count': int(y_test_c0.sum()),
        'Churn_Prevalence': f"{y_test_c0.mean()*100:.2f}%",
        'ROC_AUC': round(roc_c0, 4),
        'PR_AUC': round(pr_c0, 4),
        'Precision (tau=0.5)': round(prec_c0, 4),
        'Recall (tau=0.5)': round(rec_c0, 4),
        'F1_Score (tau=0.5)': round(f1_c0, 4)
    }
])

print("\n=== SAMPLING-AWARE PERFORMANCE COMPARISON ===")
print(eval_summary.to_string(index=False))

eval_summary.to_csv('reports/sampling_aware_validation.csv', index=False)
dist_df.to_csv('reports/sampling_feature_distributions.csv', index=False)
print("\n[+] Saved reports/sampling_aware_validation.csv and reports/sampling_feature_distributions.csv")
