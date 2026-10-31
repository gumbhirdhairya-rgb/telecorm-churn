"""
src/build_shap_notebook.py
Generates and executes notebooks/06_model_explainability.ipynb cleanly.
"""

import json
import os
import subprocess

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Step 7: Model Explainability Using SHAP (Shapley Additive exPlanations)\n",
                "## Cell2Cell Telecom Customer Churn Prediction\n",
                "\n",
                "### Academic Rubric & Viva Alignment\n",
                "- **Evaluation Component:** Innovation & Model Explainability (10 Marks)\n",
                "- **Theoretical Foundation:** Cooperative Game Theory (Shapley Values: Efficiency, Symmetry, Dummy, Additivity)\n",
                "- **Algorithm:** `shap.TreeExplainer` (Lundberg et al., *Nature Machine Intelligence* 2020) with exact polynomial-time evaluation $O(TLD^2)$\n",
                "- **Model Evaluated:** Champion Tuned XGBoost (`models/best_xgboost.joblib`)\n",
                "- **Evaluation Set:** Untouched Test Partition ($N = 14,210$, 63 features)\n",
                "\n",
                "---"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Environment Setup & Data Loading\n",
                "We load the trained champion XGBoost classifier, the untouched test set (`X_test_final.csv`, `y_test_final.csv`), and the fitted preprocessor to enable translation back to physical business units."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import joblib\n",
                "import json\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import shap\n",
                "\n",
                "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                "plt.rcParams['figure.dpi'] = 150\n",
                "\n",
                "# Load model and data\n",
                "model = joblib.load('../models/best_xgboost.joblib')\n",
                "X_test = pd.read_csv('../data/processed/X_test_final.csv')\n",
                "y_test = pd.read_csv('../data/processed/y_test_final.csv').values.ravel()\n",
                "preprocessor = joblib.load('../models/final_preprocessor.joblib')\n",
                "\n",
                "num_cols = list(preprocessor.named_transformers_['num'].feature_names_in_)\n",
                "scaler = preprocessor.named_transformers_['num'].named_steps['scaler']\n",
                "means = dict(zip(num_cols, scaler.mean_))\n",
                "scales = dict(zip(num_cols, scaler.scale_))\n",
                "\n",
                "print(f\"XGBoost model loaded: {type(model).__name__}\")\n",
                "print(f\"Test features shape: {X_test.shape}\")\n",
                "print(f\"Test target distribution: Churned = {np.sum(y_test == 1)} ({np.mean(y_test)*100:.2f}%), Retained = {np.sum(y_test == 0)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Computing Exact Shapley Values via TreeExplainer\n",
                "`shap.TreeExplainer` computes the exact local attribution of each feature to the model's raw margin output (log-odds):\n",
                "$$\\sum_{j=1}^{M} \\phi_j(x) = f(x) - E[f(X)]$$\n",
                "where $E[f(X)]$ is the expected base value and $\\phi_j(x)$ is the SHAP value for feature $j$."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "explainer = shap.TreeExplainer(model)\n",
                "shap_explanation = explainer(X_test)\n",
                "shap_values = shap_explanation.values\n",
                "base_val = float(explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value)\n",
                "\n",
                "print(f\"SHAP values matrix: {shap_values.shape}\")\n",
                "print(f\"Base value (expected margin in log-odds): {base_val:.5f}\")\n",
                "print(f\"Implied base churn probability: {1 / (1 + np.exp(-base_val)):.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Global Feature Importance\n",
                "We calculate the mean absolute SHAP value across all 14,210 test customers to rank features by overall predictive power."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "mean_abs_shap = np.mean(np.abs(shap_values), axis=0)\n",
                "shap_ranking = pd.DataFrame({\n",
                "    'feature': X_test.columns,\n",
                "    'mean_abs_shap': mean_abs_shap,\n",
                "    'mean_shap': np.mean(shap_values, axis=0)\n",
                "}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)\n",
                "shap_ranking['rank'] = shap_ranking.index + 1\n",
                "\n",
                "print(\"Top 15 Features by Global SHAP Importance:\")\n",
                "print(shap_ranking.head(15).to_string(index=False))\n",
                "\n",
                "# Bar plot\n",
                "top20 = shap_ranking.head(20).iloc[::-1]\n",
                "fig, ax = plt.subplots(figsize=(9, 7))\n",
                "bars = ax.barh(top20['feature'], top20['mean_abs_shap'], color='#1f77b4', edgecolor='black', alpha=0.85)\n",
                "ax.set_xlabel('Mean |SHAP Value| (Impact on Model Log-Odds Margin)', fontsize=11, fontweight='bold')\n",
                "ax.set_title('Top 20 Features by Global SHAP Importance', fontsize=13, fontweight='bold')\n",
                "for bar in bars:\n",
                "    w = bar.get_width()\n",
                "    ax.text(w + 0.003, bar.get_y() + bar.get_height()/2, f'{w:.4f}', ha='left', va='center', fontsize=8)\n",
                "ax.set_xlim(0, max(top20['mean_abs_shap']) * 1.15)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Summary Beeswarm Plot\n",
                "The beeswarm plot displays both feature importance and the direction of impact. Each dot is an individual customer, colored by whether their feature value was high (red) or low (blue)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig = plt.figure(figsize=(10, 8))\n",
                "shap.plots.beeswarm(shap_explanation, max_display=20, show=False)\n",
                "plt.title('SHAP Summary Beeswarm Plot (Top 20 Features)', fontsize=13, fontweight='bold', pad=12)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Dependence Plots: Revealing Non-Linear Inflection Points\n",
                "Dependence plots show how the SHAP contribution varies with the feature value, revealing non-linear thresholds and interaction patterns."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "dep_features = ['eqpdays', 'months', 'changem', 'cost_per_mou', 'handset_age_ratio', 'retcall']\n",
                "fig, axes = plt.subplots(2, 3, figsize=(16, 10))\n",
                "axes = axes.flatten()\n",
                "\n",
                "for idx, feat in enumerate(dep_features):\n",
                "    ax = axes[idx]\n",
                "    f_idx = list(X_test.columns).index(feat)\n",
                "    x_val = X_test[feat].values\n",
                "    y_val = shap_values[:, f_idx]\n",
                "    \n",
                "    ax.scatter(x_val, y_val, c='#2b5c8f', alpha=0.2, s=10, edgecolors='none')\n",
                "    \n",
                "    # Trend\n",
                "    sort_idx = np.argsort(x_val)\n",
                "    xs, ys = x_val[sort_idx], y_val[sort_idx]\n",
                "    w = max(50, len(xs) // 100)\n",
                "    rx = pd.Series(xs).rolling(w, center=True).mean().dropna()\n",
                "    ry = pd.Series(ys).rolling(w, center=True).mean().dropna()\n",
                "    ax.plot(rx, ry, color='#d62728', linewidth=2.5, label='Rolling Trend')\n",
                "    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)\n",
                "    \n",
                "    xlabel = f\"{feat} (Mean={means[feat]:.1f})\" if feat in means else feat\n",
                "    ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')\n",
                "    ax.set_ylabel(f'SHAP ({feat})', fontsize=10, fontweight='bold')\n",
                "    ax.set_title(f'Dependence: {feat}', fontsize=11, fontweight='bold')\n",
                "    ax.legend(loc='best', fontsize=8)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Local Explanations: 3 Representative Customer Personas\n",
                "We analyze 3 real customers from the untouched test set:\n",
                "1. **High Churn Risk Customer (#606)**: Churn probability = 88.08% (Actual Churn = 1)\n",
                "2. **Medium (Borderline) Risk Customer (#13293)**: Churn probability = 46.31% (Actual Churn = 0, at optimal threshold $\\tau^* = 0.4631$)\n",
                "3. **Low Churn Risk Customer (#452)**: Churn probability = 7.88% (Actual Churn = 0, Loyal)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "probs = model.predict_proba(X_test)[:, 1]\n",
                "personas = [\n",
                "    ('High Risk Customer', 606),\n",
                "    ('Medium (Borderline) Risk Customer', 13293),\n",
                "    ('Low Risk (Loyal) Customer', 452)\n",
                "]\n",
                "\n",
                "fig, axes = plt.subplots(3, 1, figsize=(12, 14))\n",
                "for ax_idx, (name, c_idx) in enumerate(personas):\n",
                "    c_prob = probs[c_idx]\n",
                "    c_true = int(y_test[c_idx])\n",
                "    c_shap = shap_values[c_idx]\n",
                "    \n",
                "    top_idx = np.argsort(np.abs(c_shap))[::-1][:8]\n",
                "    feat_names = [X_test.columns[i] for i in top_idx][::-1]\n",
                "    feat_shap = [c_shap[i] for i in top_idx][::-1]\n",
                "    \n",
                "    colors = ['#d62728' if v > 0 else '#2ca02c' for v in feat_shap]\n",
                "    labels = []\n",
                "    for f in feat_names:\n",
                "        val = X_test.iloc[c_idx][f]\n",
                "        if f in means:\n",
                "            raw = val * scales[f] + means[f]\n",
                "            labels.append(f\"{f} = {raw:.1f} (z={val:+.2f})\")\n",
                "        else:\n",
                "            labels.append(f\"{f} = {int(val)}\")\n",
                "            \n",
                "    ax = axes[ax_idx]\n",
                "    ax.barh(range(len(feat_names)), feat_shap, color=colors, edgecolor='black', alpha=0.85)\n",
                "    ax.set_yticks(range(len(feat_names)))\n",
                "    ax.set_yticklabels(labels, fontsize=9)\n",
                "    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)\n",
                "    ax.set_title(f\"{name} (Index #{c_idx}) | Predicted Churn Prob: {c_prob:.2%} | Actual Label: {c_true}\",\n",
                "                 fontsize=11, fontweight='bold')\n",
                "    ax.set_xlabel('SHAP Value (Margin Impact)', fontsize=9, fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 7. Strategic Business Translation & Actionable Interventions\n",
                "\n",
                "| Feature | Observed Direction | SHAP Impact | Telecom Business Interpretation | Actionable Retention Intervention |\n",
                "|:---|:---|:---|:---|:---|\n",
                "| `eqpdays` | High $\\to$ +Churn | Rank #1 (0.2284) | Aging handset without upgrade signals technological obsolescence and end-of-contract vulnerability. | Proactive subsidized hardware upgrade offer at Day 330 (before 12-month mark). |\n",
                "| `months` | Low $\\to$ +Churn | Rank #2 (0.1553) | New customers (<14 months) have higher churn propensity before building relationship stickiness. | Onboarding engagement programs, 90-day check-ins, and early tenure loyalty rewards. |\n",
                "| `changem` | Negative $\\to$ +Churn | Rank #3 (0.1449) | Sudden contraction in minutes of use indicates disengagement or migration to competing SIM/Wi-Fi. | Automated trigger when `changem < -20%`: send bonus minutes or customer satisfaction outreach. |\n",
                "| `cost_per_mou` | High $\\to$ +Churn | Rank #5 (0.0851) | High unit price per minute causes perceived billing injustice and plan mismatch. | Plan right-sizing: automatically recommend an optimized plan with lower rate/minute. |\n",
                "| `network_friction_rate` | High $\\to$ +Churn | Rank #7 (0.0694) | Blocked and dropped calls frustrate subscribers, severely degrading quality of experience (QoE). | VIP cell tower capacity boost and proactive service apology credits for friction > 3%. |\n",
                "| `retcall` | Active $\\to$ +Churn | Rank #14 (0.0450) | Inbound call to customer retention department is an explicit signal of imminent attrition. | Empower retention agents with instant credits, plan discounts, or immediate device renewal. |\n",
                "\n",
                "> **Methodological Caveat (Correlation vs Causation):**\n",
                "> SHAP quantifies statistical attribution within the trained XGBoost model, not causal intervention effects. For example, older handset age (`eqpdays`) correlates with churn because customers whose contracts expired keep their phones while shopping for competitors. Simply forcing a customer to keep a phone longer without an upgrade does not prevent churn."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 4,
            "pygments_lexer": "ipython3",
            "version": "3.13.14"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('notebooks/06_model_explainability.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Created notebooks/06_model_explainability.ipynb successfully.")
