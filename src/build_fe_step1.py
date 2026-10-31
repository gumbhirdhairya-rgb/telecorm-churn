"""
Creates notebooks/03_feature_engineering.ipynb - Section 1 & Section 2
"""

import json

def build_section1_and_2():
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
    add_md("""# 🔬 Step 4: Feature Engineering & Feature Selection
## UCS321 - Machine Learning Project | Problem Statement 9 (Telecom Customer Churn)
---

### 🎯 Objectives of Step 4
1. **Hypothesis-Driven Feature Engineering**:
   - Rather than generating arbitrary polynomials, construct domain-grounded telecom metrics:
     - **Handset Obsolescence Ratio** (`handset_age_ratio`): Device age relative to account tenure.
     - **Unit Cost of Usage** (`cost_per_mou`): Monthly revenue per minute of use.
     - **Overage Burden Ratio** (`overage_burden`): Proportion of fees stemming from overage penalties.
     - **Severe Usage Drop Flag** (`usage_drop_severe`): Leading indicator of customer disengagement.
     - **Network Quality Friction** (`network_friction_rate`): Dropped/blocked calls normalized by total talk time.
     - **Retention Escalation Indicator** (`retention_active`): Combined signal of customer service escalation.
2. **Preventing Target Leakage**:
   - Compute domain features strictly on raw split data ($X_{\\text{train}}$ and $X_{\\text{test}}$ independently).
   - Apply subsequent median imputation and scaling parameters derived **only from the training partition**.
3. **Multicollinearity Audit (VIF & Correlation)**:
   - Identify redundant features causing coefficient instability in linear models.
4. **Feature Importance & Selection Comparison**:
   - Compare Filter methods (Mutual Information) vs Embedded methods (Random Forest / Tree feature importances).
5. **Final Feature Serialization**:
   - Export feature-engineered, preprocessed matrices ready for model training.
""")

    # Imports
    add_md("## 📦 Section 1: Environment Setup & Data Loading")
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

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 120

print("[+] Essential libraries loaded successfully!")
""")

    add_code("""# Load original raw data to engineer features before scaling
DATA_PATH = '../data/raw/cell2cell.csv'
df_raw = pd.read_csv(DATA_PATH)

# Drop identifier & leakage columns
LEAKAGE_COLS = ['customer', 'calibrat', 'churndep']
X_raw = df_raw.drop(columns=LEAKAGE_COLS + ['churn']).copy()
y = df_raw['churn'].copy().astype(int)

# 80:20 Stratified Split to guarantee zero test leakage
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train Raw Dimensions: {X_train_raw.shape[0]:,} rows x {X_train_raw.shape[1]} features")
print(f"Test Raw Dimensions : {X_test_raw.shape[0]:,} rows x {X_test_raw.shape[1]} features")
""")

    # Feature Engineering
    add_md("""## 💡 Section 2: Hypothesis-Driven Feature Engineering

### 🎓 Domain Hypotheses (Viva-Ready Explanations):
1. **Handset Obsolescence Ratio (`handset_age_ratio`)**:
   $$\\text{handset\\_age\\_ratio} = \\frac{\\text{eqpdays}}{\\text{months} \\times 30.4375 + 1}$$
   *Rationale:* A customer whose phone age approaches or exceeds their contract tenure is carrying an outdated device. Telecom churn spikes at contract renewal when competitors offer subsidized flagship upgrades.
   
2. **Cost-per-Minute of Use (`cost_per_mou`)**:
   $$\\text{cost\\_per\\_mou} = \\frac{\\text{revenue}}{\\text{mou} + 1}$$
   *Rationale:* Measures unit cost perception. Low-usage subscribers paying high fixed recurring charges perceive poor price-to-value, elevating churn risk.

3. **Overage Burden Ratio (`overage_burden`)**:
   $$\\text{overage\\_burden} = \\frac{\\text{overage}}{\\text{recchrge} + 1}$$
   *Rationale:* Bill shock is a primary catalyst for telecom desertion. High overage relative to recurring base charges triggers churn.

4. **Severe Usage Drop Flag (`usage_drop_severe`)**:
   $$\\text{usage\\_drop\\_severe} = \\mathbb{I}(\\text{changem} < -20\\% \\land \\text{mou} > 50)$$
   *Rationale:* Customers actively switching to a secondary SIM or competitor show sharp drops in voice activity weeks before closing their account.

5. **Network Quality Friction Rate (`network_friction_rate`)**:
   $$\\text{network\\_friction\\_rate} = \\frac{\\text{dropblk}}{\\text{mou} + 1}$$
   *Rationale:* Network unreliability (dropped and blocked calls per minute) directly degrades customer satisfaction.

6. **Retention Urgency Active (`retention_active`)**:
   $$\\text{retention\\_active} = \\mathbb{I}(\\text{retcall} > 0 \\lor \\text{retcalls} > 0)$$
   *Rationale:* Combines retention escalation flags into a single high-priority alert.""")

    add_code("""def engineer_telecom_features(data: pd.DataFrame) -> pd.DataFrame:
    df_out = data.copy()
    
    # 1. Handset Obsolescence Ratio
    df_out['handset_age_ratio'] = df_out['eqpdays'] / (df_out['months'] * 30.4375 + 1.0)
    
    # 2. Cost per Minute (Unit billing pressure)
    df_out['cost_per_mou'] = df_out['revenue'] / (df_out['mou'] + 1.0)
    
    # 3. Overage Burden
    df_out['overage_burden'] = df_out['overage'] / (df_out['recchrge'] + 1.0)
    
    # 4. Severe Usage Drop Indicator
    df_out['usage_drop_severe'] = ((df_out['changem'] < -20) & (df_out['mou'] > 50)).astype(int)
    
    # 5. Network Quality Friction Rate
    df_out['network_friction_rate'] = df_out['dropblk'] / (df_out['mou'] + 1.0)
    
    # 6. Active Retention Escalation Flag
    df_out['retention_active'] = ((df_out['retcall'] > 0) | (df_out['retcalls'] > 0)).astype(int)
    
    return df_out

# Apply function to train and test splits independently
X_train_eng = engineer_telecom_features(X_train_raw)
X_test_eng = engineer_telecom_features(X_test_raw)

NEW_FEATURES = ['handset_age_ratio', 'cost_per_mou', 'overage_burden', 
                'usage_drop_severe', 'network_friction_rate', 'retention_active']

print(f"[+] Successfully engineered {len(NEW_FEATURES)} domain features.")
print(f"[+] Updated Training Dimensions: {X_train_eng.shape[1]} features (from {X_train_raw.shape[1]})")
""")

    add_code("""# Evaluate correlation of new features with churn on the training set
eng_corr = pd.concat([X_train_eng[NEW_FEATURES], y_train], axis=1).corr()['churn'].sort_values(ascending=False)

plt.figure(figsize=(9, 4.5))
colors = ['#d62728' if v > 0 else '#1f77b4' for v in eng_corr.drop('churn').values]
bars = plt.barh(eng_corr.drop('churn').index, eng_corr.drop('churn').values, color=colors, edgecolor='black')
plt.title("Correlation of Engineered Features with Customer Churn (Train Split)", fontweight='bold', fontsize=12)
plt.xlabel("Pearson Correlation (r)")
for i, v in enumerate(eng_corr.drop('churn').values):
    plt.text(v + 0.002, i, f"+{v:.4f}", va='center', fontweight='bold', fontsize=9.5)
plt.xlim(0, 0.09)
plt.tight_layout()
plt.show()

pd.DataFrame({'Feature': eng_corr.drop('churn').index, 'Correlation with Churn': eng_corr.drop('churn').values})
""")

    # Write notebook file
    with open("notebooks/03_feature_engineering.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("[+] Successfully initialized notebooks/03_feature_engineering.ipynb (Sections 1 & 2)")

if __name__ == '__main__':
    build_section1_and_2()
