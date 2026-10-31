"""
Appends Sections 5.3 & 5.4 (Hyperparameter Tuning & Final Benchmark) to notebooks/04_model_training.ipynb
"""
import json
import pandas as pd
import numpy as np

with open("notebooks/04_model_training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

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

add_md("""## 🎛️ Section 5.3: Hyperparameter Optimization (Scoring: PR-AUC)

### 🎯 Objective & Methodological Safeguards:
1. **Targeting PR-AUC (`average_precision`)**:
   - Because telecom churn is imbalanced ($29.01\\%$ positive class), models are optimized directly on Precision-Recall AUC rather than ROC-AUC or standard accuracy.
2. **Strict Test Set Isolation**:
   - Hyperparameter grids are evaluated strictly inside cross-validation folds on the training partition ($X_{\\text{train}}$, 56,837 rows). The test set remains untouched.
3. **Controlled Search Spaces**:
   - We tune regularization strength ($C$), tree depth, minimum leaf samples, and boosting parameters to balance complexity against overfitting.""")

add_code("""from sklearn.model_selection import GridSearchCV, cross_val_score

cv_tune = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

# 1. Tune Logistic Regression
print("[+] Tuning Logistic Regression...")
lr_param_grid = {'C': [0.01, 0.1, 1.0, 10.0]}
grid_lr = GridSearchCV(
    LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs'),
    lr_param_grid, cv=cv_tune, scoring='average_precision', n_jobs=-1
)
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_
print(f"    Best LR C : {grid_lr.best_params_['C']} (PR-AUC: {grid_lr.best_score_:.4f})")

# 2. Tune Decision Tree
print("[+] Tuning Decision Tree...")
dt_param_grid = {
    'max_depth': [4, 6, 8, 10],
    'min_samples_leaf': [25, 50, 100, 200]
}
grid_dt = GridSearchCV(
    DecisionTreeClassifier(class_weight='balanced', random_state=42),
    dt_param_grid, cv=cv_tune, scoring='average_precision', n_jobs=-1
)
grid_dt.fit(X_train, y_train)
best_dt = grid_dt.best_estimator_
print(f"    Best DT Params : {grid_dt.best_params_} (PR-AUC: {grid_dt.best_score_:.4f})")

# 3. Tune Random Forest
print("[+] Tuning Random Forest...")
rf_param_grid = {
    'max_depth': [8, 12, 16],
    'min_samples_split': [20, 50],
    'n_estimators': [100, 150]
}
grid_rf = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    rf_param_grid, cv=cv_tune, scoring='average_precision', n_jobs=-1
)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
print(f"    Best RF Params : {grid_rf.best_params_} (PR-AUC: {grid_rf.best_score_:.4f})")

# 4. Tune XGBoost
print("[+] Tuning XGBoost...")
xgb_param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1],
    'colsample_bytree': [0.7, 1.0],
    'n_estimators': [100, 150]
}
grid_xgb = GridSearchCV(
    XGBClassifier(scale_pos_weight=pos_weight, random_state=42, n_jobs=-1, eval_metric='logloss'),
    xgb_param_grid, cv=cv_tune, scoring='average_precision', n_jobs=-1
)
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
print(f"    Best XGB Params: {grid_xgb.best_params_} (PR-AUC: {grid_xgb.best_score_:.4f})")
""")

add_md("""## 📊 Section 5.4: Baseline vs. Tuned Benchmark & Model Serialization

We benchmark the baseline vs. tuned estimators across **Stratified 5-Fold Cross-Validation** to quantify empirical gains, then serialize the final trained models to disk.""")

add_code("""cv_eval = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

tuned_comparison = []
models_to_compare = [
    ('Logistic Regression', candidate_models['Logistic Regression (Baseline)'], best_lr, grid_lr.best_params_),
    ('Decision Tree', candidate_models['Decision Tree'], best_dt, grid_dt.best_params_),
    ('Random Forest', candidate_models['Random Forest'], best_rf, grid_rf.best_params_),
    ('XGBoost (Champion)', candidate_models['XGBoost'], best_xgb, grid_xgb.best_params_)
]

for name, base_m, tuned_m, params in models_to_compare:
    base_pr = cross_val_score(base_m, X_train, y_train, cv=cv_eval, scoring='average_precision', n_jobs=-1).mean()
    base_roc = cross_val_score(base_m, X_train, y_train, cv=cv_eval, scoring='roc_auc', n_jobs=-1).mean()
    
    tuned_pr = cross_val_score(tuned_m, X_train, y_train, cv=cv_eval, scoring='average_precision', n_jobs=-1).mean()
    tuned_roc = cross_val_score(tuned_m, X_train, y_train, cv=cv_eval, scoring='roc_auc', n_jobs=-1).mean()
    
    tuned_comparison.append({
        'Model': name,
        'Baseline_PR_AUC': round(float(base_pr), 4),
        'Tuned_PR_AUC': round(float(tuned_pr), 4),
        'PR_AUC_Gain': round(float(tuned_pr - base_pr), 4),
        'Baseline_ROC_AUC': round(float(base_roc), 4),
        'Tuned_ROC_AUC': round(float(tuned_roc), 4),
        'ROC_AUC_Gain': round(float(tuned_roc - base_roc), 4),
        'Optimal_Parameters': str(params)
    })

comparison_summary_df = pd.DataFrame(tuned_comparison)
print("=== Baseline vs. Tuned Performance Benchmark (5-Fold CV) ===")
comparison_summary_df[['Model', 'Baseline_PR_AUC', 'Tuned_PR_AUC', 'PR_AUC_Gain', 'Baseline_ROC_AUC', 'Tuned_ROC_AUC', 'ROC_AUC_Gain']]
""")

add_code("""# Visualization: Baseline vs Tuned PR-AUC and ROC-AUC
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x_idx = np.arange(len(comparison_summary_df))
bar_w = 0.35

# PR-AUC Comparison
axes[0].bar(x_idx - bar_w/2, comparison_summary_df['Baseline_PR_AUC'], width=bar_w, label='Baseline', color='#9ecae1', edgecolor='black')
axes[0].bar(x_idx + bar_w/2, comparison_summary_df['Tuned_PR_AUC'], width=bar_w, label='Tuned', color='#2171b5', edgecolor='black')
axes[0].set_xticks(x_idx)
axes[0].set_xticklabels(comparison_summary_df['Model'], rotation=15)
axes[0].set_title("PR-AUC: Baseline vs. Tuned", fontweight='bold')
axes[0].set_ylabel("PR-AUC (Average Precision)")
axes[0].set_ylim(0.35, 0.50)
axes[0].legend(loc='upper left')
for i, v in enumerate(comparison_summary_df['Tuned_PR_AUC']):
    axes[0].text(i + bar_w/2, v + 0.003, f"{v:.4f}", ha='center', fontweight='bold', fontsize=9)

# ROC-AUC Comparison
axes[1].bar(x_idx - bar_w/2, comparison_summary_df['Baseline_ROC_AUC'], width=bar_w, label='Baseline', color='#fc9272', edgecolor='black')
axes[1].bar(x_idx + bar_w/2, comparison_summary_df['Tuned_ROC_AUC'], width=bar_w, label='Tuned', color='#cb181d', edgecolor='black')
axes[1].set_xticks(x_idx)
axes[1].set_xticklabels(comparison_summary_df['Model'], rotation=15)
axes[1].set_title("ROC-AUC: Baseline vs. Tuned", fontweight='bold')
axes[1].set_ylabel("ROC-AUC Score")
axes[1].set_ylim(0.60, 0.72)
axes[1].legend(loc='upper left')
for i, v in enumerate(comparison_summary_df['Tuned_ROC_AUC']):
    axes[1].text(i + bar_w/2, v + 0.003, f"{v:.4f}", ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.show()
""")

add_code("""# Serialize best estimators fitted on the full training partition (X_train)
os.makedirs('../models', exist_ok=True)

best_lr.fit(X_train, y_train)
joblib.dump(best_lr, '../models/best_logistic_regression.joblib')

best_dt.fit(X_train, y_train)
joblib.dump(best_dt, '../models/best_decision_tree.joblib')

best_rf.fit(X_train, y_train)
joblib.dump(best_rf, '../models/best_random_forest.joblib')

best_xgb.fit(X_train, y_train)
joblib.dump(best_xgb, '../models/best_xgboost.joblib')

print("[+] Serialized 4 best estimators to models/ directory:")
print("    • models/best_logistic_regression.joblib")
print("    • models/best_decision_tree.joblib")
print("    • models/best_random_forest.joblib")
print("    • models/best_xgboost.joblib")
print("[+] Step 5: Model Development & Training COMPLETED SUCCESSFULLY!")
""")

with open("notebooks/04_model_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("[+] Successfully updated notebooks/04_model_training.ipynb with Sections 5.3 & 5.4")
