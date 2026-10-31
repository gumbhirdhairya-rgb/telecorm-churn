import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

X_train = pd.read_csv('data/processed/X_train_final.csv')
y_train = pd.read_csv('data/processed/y_train_final.csv')['churn']

print(f"X_train dimensions : {X_train.shape}")
print(f"y_train dimensions : {y_train.shape}")
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Imbalance Ratio (scale_pos_weight): {pos_weight:.3f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'roc_auc': 'roc_auc',
    'pr_auc': 'average_precision',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall'
}

models = {
    'Logistic Regression (Baseline)': LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs'
    ),
    'Decision Tree': DecisionTreeClassifier(
        class_weight='balanced', max_depth=6, min_samples_leaf=50, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=pos_weight,
        random_state=42, n_jobs=-1, eval_metric='logloss'
    )
}

results = {}
for name, model in models.items():
    print(f"\n[+] Running 5-Fold Stratified CV for: {name}...")
    cv_res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    
    roc_mean = cv_res['test_roc_auc'].mean()
    roc_std = cv_res['test_roc_auc'].std()
    pr_mean = cv_res['test_pr_auc'].mean()
    pr_std = cv_res['test_pr_auc'].std()
    f1_mean = cv_res['test_f1'].mean()
    f1_std = cv_res['test_f1'].std()
    prec_mean = cv_res['test_precision'].mean()
    rec_mean = cv_res['test_recall'].mean()
    
    results[name] = {
        'ROC_AUC_Mean': roc_mean,
        'ROC_AUC_Std': roc_std,
        'PR_AUC_Mean': pr_mean,
        'PR_AUC_Std': pr_std,
        'F1_Mean': f1_mean,
        'F1_Std': f1_std,
        'Precision_Mean': prec_mean,
        'Recall_Mean': rec_mean
    }
    print(f"    ROC-AUC : {roc_mean:.4f} (+/- {roc_std:.4f})")
    print(f"    PR-AUC  : {pr_mean:.4f} (+/- {pr_std:.4f})")
    print(f"    F1-Score: {f1_mean:.4f}")
    print(f"    Recall  : {rec_mean:.4f} | Precision: {prec_mean:.4f}")

res_df = pd.DataFrame(results).T
res_df.to_csv('reports/cv_model_comparison.csv')
print("\n=== Consolidated 5-Fold CV Benchmark Results ===")
print(res_df.round(4).to_string())
