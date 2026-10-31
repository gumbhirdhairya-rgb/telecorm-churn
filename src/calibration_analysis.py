"""
Step 6.7: Probability Calibration and Deployment Readiness
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score, precision_recall_curve, f1_score
from xgboost import XGBClassifier

# 1. Load data
X_train = pd.read_csv('data/processed/X_train_final.csv')
y_train = pd.read_csv('data/processed/y_train_final.csv')['churn']
X_test = pd.read_csv('data/processed/X_test_final.csv')
y_test = pd.read_csv('data/processed/y_test_final.csv')['churn']

# Re-identify calibrat in test set for subpopulation evaluation
df_raw = pd.read_csv('data/raw/cell2cell.csv')
X_raw = df_raw.drop(columns=['customer', 'calibrat', 'churndep', 'churn'])
y_raw = df_raw['churn'].astype(int)
cal_raw = df_raw['calibrat']

_, _, _, _, _, cal_test = train_test_split(
    X_raw, y_raw, cal_raw, test_size=0.20, random_state=42, stratify=y_raw
)

# 2. Load models and train an unweighted baseline XGBoost for comparison
pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
xgb_weighted = joblib.load('models/best_xgboost.joblib')

xgb_unweighted = XGBClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=1.0,
    random_state=42, n_jobs=-1, eval_metric='logloss'
)
xgb_unweighted.fit(X_train, y_train)

# 3. Fit Calibrated Classifier on Training Data strictly via CV (5-Fold)
calibrated_xgb_iso = CalibratedClassifierCV(
    estimator=XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss'),
    method='isotonic', cv=5
)
calibrated_xgb_iso.fit(X_train, y_train)

calibrated_xgb_sig = CalibratedClassifierCV(
    estimator=XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss'),
    method='sigmoid', cv=5
)
calibrated_xgb_sig.fit(X_train, y_train)

# 4. Predict probabilities on Test Set
p_weighted = xgb_weighted.predict_proba(X_test)[:, 1]
p_unweighted = xgb_unweighted.predict_proba(X_test)[:, 1]
p_iso = calibrated_xgb_iso.predict_proba(X_test)[:, 1]
p_sig = calibrated_xgb_sig.predict_proba(X_test)[:, 1]

# 5. Prior-Probability Bayesian Recalibration for calibrat=0 (Natural Cohort)
# Training prevalence: y_train.mean() ~= 0.2901
# Natural holdout prevalence: 0.0186 (1.86%)
p_train_prev = float(y_train.mean())
p_nat_prev = 0.0186

def bayes_prevalence_adjust(probs, p_train, p_target):
    odds_train = probs / (1.0 - probs + 1e-12)
    ratio = (p_target / (1.0 - p_target)) / (p_train / (1.0 - p_train))
    odds_target = odds_train * ratio
    return odds_target / (1.0 + odds_target)

p_unw_adjusted_c0 = bayes_prevalence_adjust(p_unweighted, p_train_prev, p_nat_prev)

# 6. Evaluate Brier Scores and Calibration on Combined Test Set
models_eval = {
    'Weighted XGBoost (Champion)': p_weighted,
    'Unweighted XGBoost': p_unweighted,
    'Calibrated XGBoost (Isotonic)': p_iso,
    'Calibrated XGBoost (Sigmoid/Platt)': p_sig
}

cal_results = []
for name, p in models_eval.items():
    brier = brier_score_loss(y_test, p)
    auc = roc_auc_score(y_test, p)
    prauc = average_precision_score(y_test, p)
    mean_p = float(p.mean())
    cal_results.append({
        'Model': name,
        'Brier_Score': round(brier, 4),
        'ROC_AUC': round(auc, 4),
        'PR_AUC': round(prauc, 4),
        'Mean_Predicted_Prob': round(mean_p, 4),
        'Actual_Prevalence': round(float(y_test.mean()), 4)
    })

cal_df = pd.DataFrame(cal_results)
cal_df.to_csv('reports/probability_calibration_results.csv', index=False)
print("=== Combined Test Set Calibration Evaluation ===")
print(cal_df.to_string(index=False))

# 7. Evaluate on Natural Subpopulation (calibrat == 0)
mask_c0 = (cal_test.values == 0)
y_test_c0 = y_test[mask_c0]

c0_eval = [
    {
        'Model': 'Unweighted XGBoost (Raw)',
        'Brier_Score': round(brier_score_loss(y_test_c0, p_unweighted[mask_c0]), 4),
        'Mean_Predicted_Risk': round(float(p_unweighted[mask_c0].mean()), 4),
        'Actual_Prevalence': round(float(y_test_c0.mean()), 4),
        'ROC_AUC': round(roc_auc_score(y_test_c0, p_unweighted[mask_c0]), 4)
    },
    {
        'Model': 'Weighted XGBoost (Raw)',
        'Brier_Score': round(brier_score_loss(y_test_c0, p_weighted[mask_c0]), 4),
        'Mean_Predicted_Risk': round(float(p_weighted[mask_c0].mean()), 4),
        'Actual_Prevalence': round(float(y_test_c0.mean()), 4),
        'ROC_AUC': round(roc_auc_score(y_test_c0, p_weighted[mask_c0]), 4)
    },
    {
        'Model': 'Bayes-Adjusted for Natural Base Rate (1.96%)',
        'Brier_Score': round(brier_score_loss(y_test_c0, p_unw_adjusted_c0[mask_c0]), 4),
        'Mean_Predicted_Risk': round(float(p_unw_adjusted_c0[mask_c0].mean()), 4),
        'Actual_Prevalence': round(float(y_test_c0.mean()), 4),
        'ROC_AUC': round(roc_auc_score(y_test_c0, p_unw_adjusted_c0[mask_c0]), 4)
    }
]

c0_df = pd.DataFrame(c0_eval)
c0_df.to_csv('reports/natural_cohort_calibration.csv', index=False)
print("\n=== Natural Cohort (calibrat == 0) Calibration Evaluation ===")
print(c0_df.to_string(index=False))

# 8. Plot Reliability Diagrams / Calibration Curves
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Combined Test Set Calibration Curve
axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration (y = x)')
colors = {'Weighted XGBoost (Champion)': '#d62728', 'Unweighted XGBoost': '#1f77b4',
          'Calibrated XGBoost (Isotonic)': '#2ca02c', 'Calibrated XGBoost (Sigmoid/Platt)': '#9467bd'}

for name, p in models_eval.items():
    prob_true, prob_pred = calibration_curve(y_test, p, n_bins=10)
    axes[0].plot(prob_pred, prob_true, marker='o', label=f"{name} (Brier={brier_score_loss(y_test, p):.4f})", color=colors[name], lw=2)

axes[0].set_title("Calibration Curves on Combined Test Set (Prevalence: 29.0%)", fontweight='bold', fontsize=11)
axes[0].set_xlabel("Mean Predicted Probability")
axes[0].set_ylabel("Fraction of True Churners")
axes[0].legend(loc='upper left', fontsize=9.5)
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)

# Plot 2: Probability Distribution Comparison (Histograms)
sns.kdeplot(p_weighted, ax=axes[1], color='#d62728', label='Weighted XGBoost (Shifted High)', lw=2)
sns.kdeplot(p_unweighted, ax=axes[1], color='#1f77b4', label='Unweighted XGBoost (Centered ~29%)', lw=2)
sns.kdeplot(p_unw_adjusted_c0[mask_c0], ax=axes[1], color='#2ca02c', label='Bayes-Adjusted for Natural Base Rate (~1.96%)', lw=2)

axes[1].set_title("Probability Densities Across Calibration Methods", fontweight='bold', fontsize=11)
axes[1].set_xlabel("Predicted Probability")
axes[1].set_ylabel("Density")
axes[1].legend(loc='upper right', fontsize=9.5)
axes[1].set_xlim(0, 1)

plt.tight_layout()
fig.savefig('reports/figures/12_calibration_curves.png', dpi=300)
plt.close()
print("\n[+] Saved reports/figures/12_calibration_curves.png")

# Save calibrated estimators
joblib.dump(xgb_unweighted, 'models/unweighted_xgboost.joblib')
joblib.dump(calibrated_xgb_iso, 'models/calibrated_xgboost_isotonic.joblib')
joblib.dump(calibrated_xgb_sig, 'models/calibrated_xgboost_sigmoid.joblib')
print("[+] Serialized unweighted & calibrated models to models/")
