import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

X_train = pd.read_csv('data/processed/X_train_final.csv')
y_train = pd.read_csv('data/processed/y_train_final.csv')['churn']
pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

print(f"X_train shape : {X_train.shape}")
print(f"Imbalance ratio : {pos_weight:.3f}")

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Baseline models for before/after comparison
baselines = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs'),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_depth=6, min_samples_leaf=50, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=pos_weight, random_state=42, n_jobs=-1, eval_metric='logloss')
}

# 1. Tune Logistic Regression
print("\n[+] Tuning 1/4: Logistic Regression...")
lr_param_grid = {'C': [0.005, 0.01, 0.05, 0.1, 1.0, 10.0]}
grid_lr = GridSearchCV(
    LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs'),
    lr_param_grid, cv=cv, scoring='average_precision', n_jobs=-1
)
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_
print(f"    Best LR Params : {grid_lr.best_params_}")
print(f"    Best PR-AUC    : {grid_lr.best_score_:.4f}")

# 2. Tune Decision Tree
print("\n[+] Tuning 2/4: Decision Tree...")
dt_param_grid = {
    'max_depth': [4, 6, 8, 10],
    'min_samples_leaf': [25, 50, 100, 200]
}
grid_dt = GridSearchCV(
    DecisionTreeClassifier(class_weight='balanced', random_state=42),
    dt_param_grid, cv=cv, scoring='average_precision', n_jobs=-1
)
grid_dt.fit(X_train, y_train)
best_dt = grid_dt.best_estimator_
print(f"    Best DT Params : {grid_dt.best_params_}")
print(f"    Best PR-AUC    : {grid_dt.best_score_:.4f}")

# 3. Tune Random Forest
print("\n[+] Tuning 3/4: Random Forest...")
rf_param_grid = {
    'max_depth': [8, 12, 16],
    'min_samples_split': [20, 50],
    'n_estimators': [100, 150]
}
grid_rf = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    rf_param_grid, cv=cv, scoring='average_precision', n_jobs=-1
)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
print(f"    Best RF Params : {grid_rf.best_params_}")
print(f"    Best PR-AUC    : {grid_rf.best_score_:.4f}")

# 4. Tune XGBoost
print("\n[+] Tuning 4/4: XGBoost...")
xgb_param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.03, 0.07, 0.1],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'n_estimators': [100, 150]
}
grid_xgb = GridSearchCV(
    XGBClassifier(scale_pos_weight=pos_weight, random_state=42, n_jobs=-1, eval_metric='logloss'),
    xgb_param_grid, cv=cv, scoring='average_precision', n_jobs=-1
)
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
print(f"    Best XGB Params: {grid_xgb.best_params_}")
print(f"    Best PR-AUC    : {grid_xgb.best_score_:.4f}")

# 5. Evaluate Baseline vs Tuned on Full 5-Fold Stratified CV
print("\n[+] Benchmarking Baseline vs Tuned models on Stratified 5-Fold CV...")
from sklearn.model_selection import cross_val_score
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

comparison_records = []
for name, base_model, best_model, best_params in [
    ('Logistic Regression', baselines['Logistic Regression'], best_lr, grid_lr.best_params_),
    ('Decision Tree', baselines['Decision Tree'], best_dt, grid_dt.best_params_),
    ('Random Forest', baselines['Random Forest'], best_rf, grid_rf.best_params_),
    ('XGBoost (Champion)', baselines['XGBoost'], best_xgb, grid_xgb.best_params_)
]:
    # Baseline PR-AUC and ROC-AUC
    base_pr = cross_val_score(base_model, X_train, y_train, cv=cv5, scoring='average_precision', n_jobs=-1).mean()
    base_roc = cross_val_score(base_model, X_train, y_train, cv=cv5, scoring='roc_auc', n_jobs=-1).mean()
    
    # Tuned PR-AUC and ROC-AUC
    tuned_pr = cross_val_score(best_model, X_train, y_train, cv=cv5, scoring='average_precision', n_jobs=-1).mean()
    tuned_roc = cross_val_score(best_model, X_train, y_train, cv=cv5, scoring='roc_auc', n_jobs=-1).mean()
    
    comparison_records.append({
        'Model': name,
        'Baseline_PR_AUC': round(float(base_pr), 4),
        'Tuned_PR_AUC': round(float(tuned_pr), 4),
        'PR_AUC_Gain': round(float(tuned_pr - base_pr), 4),
        'Baseline_ROC_AUC': round(float(base_roc), 4),
        'Tuned_ROC_AUC': round(float(tuned_roc), 4),
        'ROC_AUC_Gain': round(float(tuned_roc - base_roc), 4),
        'Best_Parameters': str(best_params)
    })

comparison_df = pd.DataFrame(comparison_records)
comparison_df.to_csv('reports/hyperparameter_tuning_results.csv', index=False)
print("\n=== Baseline vs Tuned Hyperparameter Benchmark ===")
print(comparison_df[['Model', 'Baseline_PR_AUC', 'Tuned_PR_AUC', 'PR_AUC_Gain', 'Baseline_ROC_AUC', 'Tuned_ROC_AUC', 'ROC_AUC_Gain']].to_string(index=False))

# 6. Fit best estimators on full X_train and serialize
os.makedirs('models', exist_ok=True)
best_lr.fit(X_train, y_train)
joblib.dump(best_lr, 'models/best_logistic_regression.joblib')

best_dt.fit(X_train, y_train)
joblib.dump(best_dt, 'models/best_decision_tree.joblib')

best_rf.fit(X_train, y_train)
joblib.dump(best_rf, 'models/best_random_forest.joblib')

best_xgb.fit(X_train, y_train)
joblib.dump(best_xgb, 'models/best_xgboost.joblib')

print("\n[+] Serialized all 4 best estimators to models/ directory.")
print("[+] Tuning complete successfully!")
