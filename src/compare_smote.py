import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression

X_train = pd.read_csv('data/processed/X_train_final.csv')
y_train = pd.read_csv('data/processed/y_train_final.csv')['churn']

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Cost-Sensitive
lr_weighted = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
res_w = cross_validate(lr_weighted, X_train, y_train, cv=cv, scoring=['roc_auc', 'average_precision', 'f1'])

# SMOTE in CV pipeline (train folds only)
lr_smote_pipe = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
])
res_s = cross_validate(lr_smote_pipe, X_train, y_train, cv=cv, scoring=['roc_auc', 'average_precision', 'f1'])

print("=== Class Imbalance Benchmark: Cost-Sensitive Weights vs SMOTE ===")
print(f"Cost-Sensitive Class Weights:")
print(f"  ROC-AUC : {res_w['test_roc_auc'].mean():.4f}")
print(f"  PR-AUC  : {res_w['test_average_precision'].mean():.4f}")
print(f"  F1-Score: {res_w['test_f1'].mean():.4f}")

print(f"\nSMOTE Oversampling (Train Folds Only):")
print(f"  ROC-AUC : {res_s['test_roc_auc'].mean():.4f}")
print(f"  PR-AUC  : {res_s['test_average_precision'].mean():.4f}")
print(f"  F1-Score: {res_s['test_f1'].mean():.4f}")
