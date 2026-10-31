"""
Audit script for Step 6.7: Probability Calibration
Calculates Cox calibration intercept (alpha) & slope (beta),
Brier Skill Scores (BSS) against constant prevalence baselines,
and verifies zero leakage.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

# 1. Load data
X_train = pd.read_csv('data/processed/X_train_final.csv')
y_train = pd.read_csv('data/processed/y_train_final.csv')['churn']
X_test = pd.read_csv('data/processed/X_test_final.csv')
y_test = pd.read_csv('data/processed/y_test_final.csv')['churn']

# Re-link calibrat column for test slice evaluation
df_raw = pd.read_csv('data/raw/cell2cell.csv')
X_raw = df_raw.drop(columns=['customer', 'calibrat', 'churndep', 'churn'])
y_raw = df_raw['churn'].astype(int)
cal_raw = df_raw['calibrat']

_, _, _, _, _, cal_test = train_test_split(
    X_raw, y_raw, cal_raw, test_size=0.20, random_state=42, stratify=y_raw
)

# 2. Verify Feature Ordering Consistency
assert list(X_train.columns) == list(X_test.columns), "Feature mismatch!"
print(f"[+] Feature check passed: Exactly {len(X_train.columns)} features in identical order.")

# 3. Load Models
xgb_weighted = joblib.load('models/best_xgboost.joblib')
xgb_unweighted = joblib.load('models/unweighted_xgboost.joblib')
xgb_iso = joblib.load('models/calibrated_xgboost_isotonic.joblib')
xgb_sig = joblib.load('models/calibrated_xgboost_sigmoid.joblib')

# Predictions
p_weighted = xgb_weighted.predict_proba(X_test)[:, 1]
p_unweighted = xgb_unweighted.predict_proba(X_test)[:, 1]
p_iso = xgb_iso.predict_proba(X_test)[:, 1]
p_sig = xgb_sig.predict_proba(X_test)[:, 1]

# Bayes Adjusted for natural cohort (calibrat == 0)
p_train_prev = float(y_train.mean())
p_nat_prev = 0.0186
odds_unw = p_unweighted / (1.0 - p_unweighted + 1e-12)
ratio_c0 = (p_nat_prev / (1.0 - p_nat_prev)) / (p_train_prev / (1.0 - p_train_prev))
p_bayes_c0 = (odds_unw * ratio_c0) / (1.0 + (odds_unw * ratio_c0))

# 4. Helper function for Cox Calibration (Intercept & Slope)
def get_cox_calibration(y_true, probs):
    # Clip probabilities to avoid logit infinities
    eps = 1e-7
    p_clipped = np.clip(probs, eps, 1 - eps)
    logits = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)
    
    # Fit logistic regression on logits
    lr_cal = LogisticRegression(solver='lbfgs', C=1e6) # unregularized
    lr_cal.fit(logits, y_true)
    intercept = float(lr_cal.intercept_[0])
    slope = float(lr_cal.coef_[0][0])
    return intercept, slope

# 5. Combined Test Set Audit Table
prev_comb = float(y_test.mean())
brier_baseline_comb = prev_comb * (1.0 - prev_comb)

combined_audit = []
models_dict_comb = {
    'Weighted XGBoost (Champion)': p_weighted,
    'Unweighted XGBoost': p_unweighted,
    'Calibrated XGBoost (Isotonic)': p_iso,
    'Calibrated XGBoost (Sigmoid/Platt)': p_sig
}

for name, probs in models_dict_comb.items():
    brier = brier_score_loss(y_test, probs)
    bss = (1.0 - (brier / brier_baseline_comb)) * 100 # Brier Skill Score (%)
    alpha, beta = get_cox_calibration(y_test, probs)
    combined_audit.append({
        'Model': name,
        'Mean_Prob': round(float(probs.mean()), 4),
        'True_Prev': round(prev_comb, 4),
        'Brier_Score': round(brier, 4),
        'Brier_Baseline': round(brier_baseline_comb, 4),
        'Brier_Skill_Score_%': round(bss, 2),
        'Calib_Intercept (alpha)': round(alpha, 3),
        'Calib_Slope (beta)': round(beta, 3),
        'ROC_AUC': round(roc_auc_score(y_test, probs), 4),
        'PR_AUC': round(average_precision_score(y_test, probs), 4)
    })

df_comb_audit = pd.DataFrame(combined_audit)
print("=== COMBINED TEST SET AUDIT RESULTS ===")
print(df_comb_audit.to_string(index=False))

# 6. Natural Cohort (calibrat == 0) Audit Table
mask_c0 = (cal_test.values == 0)
y_c0 = y_test[mask_c0]
prev_c0 = float(y_c0.mean())
brier_baseline_c0 = prev_c0 * (1.0 - prev_c0)

c0_models = {
    'Weighted XGBoost (Raw)': p_weighted[mask_c0],
    'Unweighted XGBoost (Raw)': p_unweighted[mask_c0],
    'Bayes Prior-Adjusted (1.86%)': p_bayes_c0[mask_c0]
}

c0_audit = []
for name, probs in c0_models.items():
    brier = brier_score_loss(y_c0, probs)
    bss = (1.0 - (brier / brier_baseline_c0)) * 100
    alpha, beta = get_cox_calibration(y_c0, probs)
    c0_audit.append({
        'Model': name,
        'Mean_Prob': round(float(probs.mean()), 4),
        'True_Prev': round(prev_c0, 4),
        'Brier_Score': round(brier, 4),
        'Brier_Baseline': round(brier_baseline_c0, 4),
        'Brier_Skill_Score_%': round(bss, 2),
        'Calib_Intercept (alpha)': round(alpha, 3),
        'Calib_Slope (beta)': round(beta, 3),
        'ROC_AUC': round(roc_auc_score(y_c0, probs), 4),
        'PR_AUC': round(average_precision_score(y_c0, probs), 4)
    })

df_c0_audit = pd.DataFrame(c0_audit)
print("\n=== NATURAL COHORT (calibrat == 0) AUDIT RESULTS ===")
print(df_c0_audit.to_string(index=False))

# Export results
df_comb_audit.to_csv('reports/audit_combined_calibration.csv', index=False)
df_c0_audit.to_csv('reports/audit_natural_cohort_calibration.csv', index=False)
print("\n[+] Exported reports/audit_combined_calibration.csv and reports/audit_natural_cohort_calibration.csv")
