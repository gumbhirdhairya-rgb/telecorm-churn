"""
Builds notebooks/04_model_training.ipynb - Sections 5.1 & 5.2
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import pandas as pd
import numpy as np

def build_model_training_notebook_part1():
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
    add_md("""# 🤖 Step 5: Model Development & Cross-Validation
## UCS321 - Machine Learning Project | Problem Statement 9 (Telecom Customer Churn)
---

### 🎯 Objectives of Step 5
1. **Establish a Transparent Baseline**:
   - Train an interpretable **Logistic Regression** baseline with balanced class weights.
2. **Benchmark Progressive Algorithms**:
   - Evaluate non-linear **Decision Tree**, ensemble bagging **Random Forest**, and gradient boosted **XGBoost**.
3. **Rigorous Class Imbalance Strategy**:
   - Compare **Cost-Sensitive Loss Weighting** (`scale_pos_weight` and `class_weight='balanced'`) against **SMOTE**.
   - Maintain natural data geometry without introducing synthetic artifacts across discrete binary flags.
4. **Leakage-Free Stratified 5-Fold Cross-Validation**:
   - Multi-metric evaluation on the training partition ($X_{\\text{train}}$, 56,837 rows): **ROC-AUC, PR-AUC, Precision, Recall, and F1-Score**.
5. **Hyperparameter Optimization**:
   - Grid search tuning of tree depth, learning rate, subsampling, and feature subsampling strictly within cross-validation folds.
6. **Model Serialization**:
   - Export trained estimators and cross-validation performance benchmarks to `../models/` and `../reports/`.
""")

    # Section 5.1
    add_md("## 📦 Section 5.1: Setup & Class Imbalance Strategy (Weights vs. SMOTE)")
    add_code("""import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 120

# 1. Load preprocessed, feature-selected training partition (63 features)
X_train = pd.read_csv('../data/processed/X_train_final.csv')
y_train = pd.read_csv('../data/processed/y_train_final.csv')['churn']

print(f"X_train Shape : {X_train.shape[0]:,} samples x {X_train.shape[1]} features")
print(f"y_train Shape : {y_train.shape[0]:,} labels")
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class Imbalance Ratio (scale_pos_weight): {pos_weight:.3f} : 1")
""")

    add_md("""### ⚖️ Empirical Evaluation: Cost-Sensitive Class Weights vs. SMOTE
Before training, we evaluate whether synthetic oversampling (SMOTE) or loss re-weighting is superior on this dataset.

**Key Statistical Distinction:**
- **Cost-Sensitive Class Weights:** Modifies the loss function penalties ($\mathcal{L}$) directly:
  $$\mathcal{L} = - \sum_{i} w_{y_i} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$
  where $w_1 = \frac{N}{2 \cdot N_1} \approx 1.72$ and $w_0 = \frac{N}{2 \cdot N_0} \approx 0.70$.
- **SMOTE:** Synthesizes artificial observations via linear interpolation between $k$-nearest neighbors. In datasets containing 29 discrete binary flags, SMOTE's linear interpolation creates non-binary fractional artifacts unless restricted.
- *Strict Rule:* SMOTE must only be applied **inside training folds** during cross-validation.""")

    add_code("""cv_imb = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# 1. Cost-Sensitive Logistic Regression
lr_weighted = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
res_weighted = cross_validate(lr_weighted, X_train, y_train, cv=cv_imb, scoring=['roc_auc', 'average_precision', 'f1'], n_jobs=-1)

# 2. SMOTE Logistic Regression (applied strictly inside CV folds)
lr_smote_pipe = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
])
res_smote = cross_validate(lr_smote_pipe, X_train, y_train, cv=cv_imb, scoring=['roc_auc', 'average_precision', 'f1'], n_jobs=-1)

imb_comparison = pd.DataFrame({
    'Method': ['Cost-Sensitive Class Weights', 'SMOTE Oversampling (Train Folds Only)'],
    'ROC-AUC': [res_weighted['test_roc_auc'].mean(), res_smote['test_roc_auc'].mean()],
    'PR-AUC': [res_weighted['test_average_precision'].mean(), res_smote['test_average_precision'].mean()],
    'F1-Score': [res_weighted['test_f1'].mean(), res_smote['test_f1'].mean()]
})

print("=== Class Imbalance Benchmark Results ===")
imb_comparison.round(4)
""")

    # Section 5.2
    add_md("""## 🏆 Section 5.2: Multi-Model Benchmark (Stratified 5-Fold Cross-Validation)

We evaluate 4 distinct model families using **Stratified 5-Fold Cross-Validation** on the training data:
1. **Logistic Regression (Baseline):** Convex optimization with L2 regularization and balanced loss weighting.
2. **Decision Tree:** Non-linear rule-based tree with controlled depth (`max_depth=6`, `min_samples_leaf=50`).
3. **Random Forest:** Ensemble of 100 decorrelated bagged decision trees with balanced subsampling.
4. **XGBoost:** Gradient boosted tree ensemble minimizing log-loss with `scale_pos_weight=2.447`.""")

    add_code("""cv_5fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_suite = {
    'roc_auc': 'roc_auc',
    'pr_auc': 'average_precision',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall'
}

candidate_models = {
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

cv_results = {}
for name, model in candidate_models.items():
    print(f"[+] Running 5-Fold CV: {name}...")
    res = cross_validate(model, X_train, y_train, cv=cv_5fold, scoring=scoring_suite, n_jobs=-1)
    
    cv_results[name] = {
        'ROC_AUC_Mean': res['test_roc_auc'].mean(),
        'ROC_AUC_Std': res['test_roc_auc'].std(),
        'PR_AUC_Mean': res['test_pr_auc'].mean(),
        'PR_AUC_Std': res['test_pr_auc'].std(),
        'F1_Mean': res['test_f1'].mean(),
        'F1_Std': res['test_f1'].std(),
        'Precision_Mean': res['test_precision'].mean(),
        'Recall_Mean': res['test_recall'].mean()
    }

cv_df = pd.DataFrame(cv_results).T
print("\\n=== Consolidated 5-Fold Cross-Validation Performance ===")
cv_df.round(4)
""")

    add_code("""# Visualization: Model Comparison Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC-AUC with standard deviation error bars
axes[0].bar(cv_df.index, cv_df['ROC_AUC_Mean'], yerr=cv_df['ROC_AUC_Std'], capsize=5,
            color=['#7f7f7f', '#2ca02c', '#1f77b4', '#d62728'], edgecolor='black', width=0.55)
axes[0].set_title("5-Fold Cross-Validation ROC-AUC", fontweight='bold')
axes[0].set_ylabel("ROC-AUC Score")
axes[0].set_ylim(0.55, 0.72)
axes[0].tick_params(axis='x', rotation=15)
for i, v in enumerate(cv_df['ROC_AUC_Mean']):
    axes[0].text(i, v + 0.008, f"{v:.4f}", ha='center', fontweight='bold', fontsize=9.5)

# Precision vs Recall Trade-off across models
x_pos = np.arange(len(cv_df.index))
w = 0.35
axes[1].bar(x_pos - w/2, cv_df['Precision_Mean'], width=w, label='Precision', color='#1f77b4', edgecolor='black')
axes[1].bar(x_pos + w/2, cv_df['Recall_Mean'], width=w, label='Recall', color='#ff7f0e', edgecolor='black')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(cv_df.index, rotation=15)
axes[1].set_title("Precision vs. Recall Trade-off (Cost-Sensitive)", fontweight='bold')
axes[1].set_ylabel("Score")
axes[1].set_ylim(0, 0.8)
axes[1].legend(loc='upper right')

plt.tight_layout()
plt.show()
""")

    with open("notebooks/04_model_training.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("[+] Successfully generated notebooks/04_model_training.ipynb (Sections 5.1 & 5.2)")

if __name__ == '__main__':
    build_model_training_notebook_part1()
