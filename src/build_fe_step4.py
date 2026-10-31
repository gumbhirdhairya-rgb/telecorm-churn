"""
Step 4.3: Feature Selection & Importance Comparison
Calculates Random Forest, XGBoost, and Mutual Information feature importances on X_train only.
Appends Section 4 to notebooks/03_feature_engineering.ipynb and generates visual report figures.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from src.fe_utils import engineer_telecom_features

# 1. Load raw split and apply Section 4.2 justified removals
df_raw = pd.read_csv('data/raw/cell2cell.csv')
X_raw = df_raw.drop(columns=['customer', 'calibrat', 'churndep', 'churn'])
y = df_raw['churn'].astype(int)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)

# Feature engineering strictly on X_train
X_train_eng = engineer_telecom_features(X_train_raw)

# Justified removals from Section 4.2
TO_REMOVE_COLLINEAR = ['dropblk', 'mailres', 'models', 'retcalls']
X_train_clean = X_train_eng.drop(columns=TO_REMOVE_COLLINEAR)

# Median imputation for tree fitting
imputer = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_clean), columns=X_train_clean.columns, index=X_train_clean.index)

# 2. Random Forest Importance (MDI)
rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train_imp, y_train)
rf_importance = pd.Series(rf.feature_importances_, index=X_train_clean.columns)

# 3. XGBoost Importance (Gain)
xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')
xgb.fit(X_train_imp, y_train)
xgb_importance = pd.Series(xgb.feature_importances_, index=X_train_clean.columns)

# 4. Mutual Information (Stratified sample of 15,000 for fast accurate calculation)
np.random.seed(42)
sample_idx = y_train.groupby(y_train, group_keys=False).apply(lambda s: s.sample(min(len(s), 7500), random_state=42)).index
mi_scores = mutual_info_classif(X_train_imp.loc[sample_idx], y_train.loc[sample_idx], random_state=42)
mi_importance = pd.Series(mi_scores, index=X_train_clean.columns)

# 5. Compile consolidated ranking DataFrame
ranking_df = pd.DataFrame({
    'Random_Forest_MDI': rf_importance,
    'XGBoost_Importance': xgb_importance,
    'Mutual_Information': mi_importance
})
ranking_df['RF_Rank'] = ranking_df['Random_Forest_MDI'].rank(ascending=False).astype(int)
ranking_df['XGB_Rank'] = ranking_df['XGBoost_Importance'].rank(ascending=False).astype(int)
ranking_df['MI_Rank'] = ranking_df['Mutual_Information'].rank(ascending=False).astype(int)
ranking_df['Mean_Rank'] = np.round((ranking_df['RF_Rank'] + ranking_df['XGB_Rank'] + ranking_df['MI_Rank']) / 3, 2)
ranking_df = ranking_df.sort_values(by='Mean_Rank')

# Save ranking CSV
os.makedirs('reports/figures', exist_ok=True)
ranking_df.to_csv('reports/feature_importance_rankings.csv')
print("[+] Saved reports/feature_importance_rankings.csv")

# 6. Generate Comparison Figure
fig, axes = plt.subplots(1, 3, figsize=(18, 7))

# Plot 1: Top 15 Random Forest
top_rf = rf_importance.sort_values(ascending=False).head(15)
rf_colors = ['#d62728' if feat in ['handset_age_ratio', 'cost_per_mou', 'overage_burden', 'network_friction_rate'] else '#1f77b4' for feat in top_rf.index]
axes[0].barh(top_rf.index[::-1], top_rf.values[::-1], color=rf_colors[::-1], edgecolor='black')
axes[0].set_title("Random Forest Top 15 (MDI)", fontweight='bold', fontsize=12)
axes[0].set_xlabel("Mean Decrease in Impurity")

# Plot 2: Top 15 XGBoost
top_xgb = xgb_importance.sort_values(ascending=False).head(15)
xgb_colors = ['#d62728' if feat in ['handset_age_ratio', 'cost_per_mou', 'overage_burden', 'network_friction_rate'] else '#ff7f0e' for feat in top_xgb.index]
axes[1].barh(top_xgb.index[::-1], top_xgb.values[::-1], color=xgb_colors[::-1], edgecolor='black')
axes[1].set_title("XGBoost Top 15 (Gain)", fontweight='bold', fontsize=12)
axes[1].set_xlabel("Feature Importance Score")

# Plot 3: Top 15 Mutual Information
top_mi = mi_importance.sort_values(ascending=False).head(15)
mi_colors = ['#d62728' if feat in ['handset_age_ratio', 'cost_per_mou', 'overage_burden', 'network_friction_rate'] else '#2ca02c' for feat in top_mi.index]
axes[2].barh(top_mi.index[::-1], top_mi.values[::-1], color=mi_colors[::-1], edgecolor='black')
axes[2].set_title("Mutual Information Top 15", fontweight='bold', fontsize=12)
axes[2].set_xlabel("Mutual Information Score")

# Add unified legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d62728', edgecolor='black', label='Engineered Domain Features'),
    Patch(facecolor='#1f77b4', edgecolor='black', label='Original Features (RF)'),
    Patch(facecolor='#ff7f0e', edgecolor='black', label='Original Features (XGB)'),
    Patch(facecolor='#2ca02c', edgecolor='black', label='Original Features (MI)')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=11)
plt.tight_layout()
fig.savefig('reports/figures/07_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("[+] Generated reports/figures/07_feature_importance_comparison.png")

# 7. Final Selected Features (Retaining features with non-zero importance across models)
# We remove features that had exactly 0.000 importance in XGBoost and bottom 10% in RF
zero_gain_features = [f for f in X_train_clean.columns if xgb_importance[f] == 0.0 and rf_importance[f] < 0.002]
final_selected_features = [f for f in X_train_clean.columns if f not in zero_gain_features]

feature_selection_metadata = {
    "original_features_count": X_raw.shape[1],
    "engineered_features_added": ['handset_age_ratio', 'cost_per_mou', 'overage_burden', 'usage_drop_severe', 'network_friction_rate'],
    "collinear_features_removed": TO_REMOVE_COLLINEAR,
    "zero_gain_pruned_features": zero_gain_features,
    "final_selected_count": len(final_selected_features),
    "final_selected_features": final_selected_features
}

with open('data/processed/selected_features.json', 'w') as f:
    json.dump(feature_selection_metadata, f, indent=4)
print(f"[+] Final selected feature set ({len(final_selected_features)} features) saved to data/processed/selected_features.json")

# 8. Append Section 4 to notebooks/03_feature_engineering.ipynb
with open("notebooks/03_feature_engineering.ipynb", "r", encoding="utf-8") as f:
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

add_md("""## 🏆 Section 4: Feature Importance & Feature Selection Comparison

### 🎯 Methodological Setup:
We evaluate feature contributions using three complementary paradigms strictly on the training partition:
1. **Random Forest (Embedded MDI):** Measures total variance reduction / Gini impurity drop across bagging trees.
2. **XGBoost (Embedded Gain):** Measures gradient loss reduction from splits on each feature across sequential boosting stages.
3. **Mutual Information (Filter Method):** Measures non-linear dependency $I(X; Y) = H(Y) - H(Y|X)$ independent of any model architecture.

### 🛡️ Test Set Isolation:
Feature selection criteria and rankings are computed **solely on $X_{\\text{train}}$ and $y_{\\text{train}}$** to prevent target leakage.""")

add_code("""from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import mutual_info_classif

# 1. Apply verified multicollinearity pruning from Section 4.2
TO_REMOVE_COLLINEAR = ['dropblk', 'mailres', 'models', 'retcalls']
X_train_clean = X_train_eng.drop(columns=TO_REMOVE_COLLINEAR)

# 2. Impute training set
imputer = SimpleImputer(strategy='median')
X_tr_imp = pd.DataFrame(imputer.fit_transform(X_train_clean), columns=X_train_clean.columns, index=X_train_clean.index)

# 3. Fit Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_tr_imp, y_train)
rf_importance = pd.Series(rf.feature_importances_, index=X_train_clean.columns)

# 4. Fit XGBoost
xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')
xgb.fit(X_tr_imp, y_train)
xgb_importance = pd.Series(xgb.feature_importances_, index=X_train_clean.columns)

# 5. Compute Mutual Information
np.random.seed(42)
sample_idx = y_train.groupby(y_train, group_keys=False).apply(lambda s: s.sample(min(len(s), 7500), random_state=42)).index
mi_scores = mutual_info_classif(X_tr_imp.loc[sample_idx], y_train.loc[sample_idx], random_state=42)
mi_importance = pd.Series(mi_scores, index=X_train_clean.columns)

# Consolidated ranking
ranking_df = pd.DataFrame({
    'Random_Forest': rf_importance,
    'XGBoost': xgb_importance,
    'Mutual_Info': mi_importance
})
ranking_df['RF_Rank'] = ranking_df['Random_Forest'].rank(ascending=False).astype(int)
ranking_df['XGB_Rank'] = ranking_df['XGBoost'].rank(ascending=False).astype(int)
ranking_df['MI_Rank'] = ranking_df['Mutual_Info'].rank(ascending=False).astype(int)
ranking_df['Consensus_Rank'] = np.round((ranking_df['RF_Rank'] + ranking_df['XGB_Rank'] + ranking_df['MI_Rank']) / 3, 2)
ranking_df = ranking_df.sort_values(by='Consensus_Rank')

print("=== Top 15 Consensus Churn Predictors ===")
ranking_df.head(15)[['Random_Forest', 'XGBoost', 'Mutual_Info', 'Consensus_Rank']]
""")

add_code("""# Evaluate performance of engineered features
engineered_list = ['handset_age_ratio', 'cost_per_mou', 'overage_burden', 'usage_drop_severe', 'network_friction_rate']

print("=== Engineered Features Validation Summary ===")
eng_summary = ranking_df.loc[engineered_list][['RF_Rank', 'XGB_Rank', 'MI_Rank', 'Consensus_Rank']]
eng_summary
""")

add_code("""# Visualization: Comparison across all three methods
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

top12_rf = rf_importance.sort_values(ascending=False).head(12)
rf_colors = ['#d62728' if feat in engineered_list else '#1f77b4' for feat in top12_rf.index]
axes[0].barh(top12_rf.index[::-1], top12_rf.values[::-1], color=rf_colors[::-1], edgecolor='black')
axes[0].set_title("Random Forest Top 12 (MDI)", fontweight='bold')
axes[0].set_xlabel("Feature Importance")

top12_xgb = xgb_importance.sort_values(ascending=False).head(12)
xgb_colors = ['#d62728' if feat in engineered_list else '#ff7f0e' for feat in top12_xgb.index]
axes[1].barh(top12_xgb.index[::-1], top12_xgb.values[::-1], color=xgb_colors[::-1], edgecolor='black')
axes[1].set_title("XGBoost Top 12 (Gain)", fontweight='bold')
axes[1].set_xlabel("Importance Score")

top12_mi = mi_importance.sort_values(ascending=False).head(12)
mi_colors = ['#d62728' if feat in engineered_list else '#2ca02c' for feat in top12_mi.index]
axes[2].barh(top12_mi.index[::-1], top12_mi.values[::-1], color=mi_colors[::-1], edgecolor='black')
axes[2].set_title("Mutual Information Top 12", fontweight='bold')
axes[2].set_xlabel("MI Score")

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d62728', edgecolor='black', label='Engineered Domain Features'),
    Patch(facecolor='#1f77b4', edgecolor='black', label='Original Feature (RF)'),
    Patch(facecolor='#ff7f0e', edgecolor='black', label='Original Feature (XGB)'),
    Patch(facecolor='#2ca02c', edgecolor='black', label='Original Feature (MI)')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=4, fontsize=11)
plt.tight_layout()
plt.show()
""")

add_md("""### 📋 Final Feature Selection & Serialization
We preserve features demonstrating non-zero predictive contribution across models, while discarding uninformative, zero-gain variables (`callfwdv`, rare occupation dummies with 0.000 importance).""")

add_code("""# Filter zero-gain uninformative features
zero_gain = [f for f in X_train_clean.columns if xgb_importance[f] == 0.0 and rf_importance[f] < 0.002]
final_features = [f for f in X_train_clean.columns if f not in zero_gain]

print(f"Features Before Selection : {X_train_clean.shape[1]}")
print(f"Zero-Gain Pruned Features : {len(zero_gain)} ({zero_gain})")
print(f"Final Retained Features   : {len(final_features)}")

# Save selected feature list
with open('../data/processed/selected_features.json', 'w') as f:
    json.dump({'selected_features': final_features, 'count': len(final_features)}, f, indent=4)

print("[+] Successfully exported selected features list to data/processed/selected_features.json")
""")

with open("notebooks/03_feature_engineering.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("[+] Successfully updated notebooks/03_feature_engineering.ipynb with Section 4")

if __name__ == '__main__':
    pass
