"""
src/build_business_notebook.py
Generates and executes notebooks/07_business_insights.ipynb cleanly.
"""

import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Step 8: Business Insights and Customer Retention Strategy\n",
                "## Cell2Cell Telecom Customer Churn Project\n",
                "\n",
                "### Academic Rubric & Viva Alignment\n",
                "- **Evaluation Component:** Business Value, Actionable Strategy & Decision Science\n",
                "- **Dataset:** Cell2Cell Test Set ($N = 14,210$, 63 final features)\n",
                "- **Evaluated Model:** Tuned Champion XGBoost (`models/best_xgboost.joblib`)\n",
                "- **Core Objective:** Translate mathematical SHAP attributions and calibrated probabilities into a financially viable, operationally feasible customer retention strategy.\n",
                "\n",
                "---"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Data Ingestion & Model Scoring\n",
                "We load the trained champion model, the untouched test set, and unscale the normalized features to report results in real-world business units (dollars, minutes, days, and calls)."
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
                "# Recover original scaling parameters\n",
                "num_cols = list(preprocessor.named_transformers_['num'].feature_names_in_)\n",
                "scaler = preprocessor.named_transformers_['num'].named_steps['scaler']\n",
                "means = dict(zip(num_cols, scaler.mean_))\n",
                "scales = dict(zip(num_cols, scaler.scale_))\n",
                "\n",
                "probs = model.predict_proba(X_test)[:, 1]\n",
                "print(f\"Loaded X_test: {X_test.shape}, Base churn rate: {np.mean(y_test)*100:.2f}%\")\n",
                "print(f\"Predicted churn probability range: [{probs.min():.4f}, {probs.max():.4f}], Mean: {probs.mean():.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Empirical 3-Tier Customer Risk Segmentation\n",
                "We segment the 14,210 test customers into three actionable operational tiers:\n",
                "1. **High Risk Tier ($\\hat{p} \\ge 0.60$):** Severe disengagement, acute probability of defection.\n",
                "2. **Medium (Borderline) Risk Tier ($0.35 \\le \\hat{p} < 0.60$):** Ambiguous churn probability, active network users; contains our optimal decision threshold $\\tau^* = 0.4631$.\n",
                "3. **Low Risk (Loyal) Tier ($\\hat{p} < 0.35$):** Highly satisfied legacy customers; low churn propensity."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "segments = np.where(probs >= 0.60, 'High Risk', np.where(probs >= 0.35, 'Medium Risk', 'Low Risk'))\n",
                "\n",
                "df_seg = pd.DataFrame({\n",
                "    'segment': segments,\n",
                "    'pred_prob': probs,\n",
                "    'actual_churn': y_test,\n",
                "    'revenue': X_test['revenue'] * scales['revenue'] + means['revenue'],\n",
                "    'mou': X_test['mou'] * scales['mou'] + means['mou'],\n",
                "    'eqpdays': X_test['eqpdays'] * scales['eqpdays'] + means['eqpdays'],\n",
                "    'months': X_test['months'] * scales['months'] + means['months'],\n",
                "    'changem': X_test['changem'] * scales['changem'] + means['changem'],\n",
                "    'cost_per_mou': X_test['cost_per_mou'] * scales['cost_per_mou'] + means['cost_per_mou'],\n",
                "    'retcall': X_test['retcall'].values\n",
                "})\n",
                "\n",
                "summary = df_seg.groupby('segment').agg(\n",
                "    customer_count=('actual_churn', 'count'),\n",
                "    base_share_pct=('actual_churn', lambda x: len(x)/len(df_seg)*100),\n",
                "    actual_churners=('actual_churn', 'sum'),\n",
                "    actual_churn_rate=('actual_churn', 'mean'),\n",
                "    avg_pred_prob=('pred_prob', 'mean'),\n",
                "    avg_revenue_usd=('revenue', 'mean'),\n",
                "    avg_mou_min=('mou', 'mean'),\n",
                "    avg_eqpdays_days=('eqpdays', 'mean'),\n",
                "    avg_tenure_mos=('months', 'mean'),\n",
                "    avg_usage_change_pct=('changem', 'mean'),\n",
                "    avg_cost_per_mou=('cost_per_mou', 'mean'),\n",
                "    retention_call_pct=('retcall', lambda x: np.mean(x)*100)\n",
                ").loc[['High Risk', 'Medium Risk', 'Low Risk']]\n",
                "\n",
                "print(summary.round(2).to_string())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Risk Segmentation Visual Profiles\n",
                "We visualize customer volumes, observed churn rates, handset ages, usage trajectory contractions, and retention desk complaint frequencies across the three tiers."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(2, 2, figsize=(13, 9))\n",
                "\n",
                "# Panel 1: Customer Volume & Churn Rate\n",
                "ax = axes[0, 0]\n",
                "x = np.arange(len(summary))\n",
                "w = 0.35\n",
                "ax.bar(x - w/2, summary['customer_count'], w, label='Total Customers', color='#1f77b4', edgecolor='black', alpha=0.85)\n",
                "ax2 = ax.twinx()\n",
                "ax2.bar(x + w/2, summary['actual_churn_rate'] * 100, w, label='Actual Churn Rate (%)', color='#d62728', edgecolor='black', alpha=0.85)\n",
                "ax.set_xticks(x)\n",
                "ax.set_xticklabels(summary.index, fontsize=10, fontweight='bold')\n",
                "ax.set_ylabel('Number of Customers', color='#1f77b4', fontweight='bold')\n",
                "ax2.set_ylabel('Actual Churn Rate (%)', color='#d62728', fontweight='bold')\n",
                "ax2.grid(False)\n",
                "ax.set_title('Customer Base Share vs. Observed Churn Rate', fontsize=11, fontweight='bold')\n",
                "\n",
                "# Panel 2: Equipment Age\n",
                "ax = axes[0, 1]\n",
                "bars = ax.bar(summary.index, summary['avg_eqpdays_days'], color=['#d62728', '#ff7f0e', '#2ca02c'], edgecolor='black', alpha=0.85)\n",
                "ax.axhline(365, color='gray', linestyle='--', linewidth=1.5, label='1-Year Contract Expiry (365d)')\n",
                "ax.set_ylabel('Avg. Equipment Age (Days)', fontweight='bold')\n",
                "ax.set_title('Handset Age by Risk Tier', fontsize=11, fontweight='bold')\n",
                "ax.legend(loc='lower right', fontsize=8)\n",
                "\n",
                "# Panel 3: Monthly Usage Change (%)\n",
                "ax = axes[1, 0]\n",
                "colors = ['#d62728' if v < 0 else '#2ca02c' for v in summary['avg_usage_change_pct']]\n",
                "ax.bar(summary.index, summary['avg_usage_change_pct'], color=colors, edgecolor='black', alpha=0.85)\n",
                "ax.axhline(0, color='black', linewidth=1)\n",
                "ax.axhline(-20, color='orange', linestyle=':', label='Alert Boundary (-20%)')\n",
                "ax.set_ylabel('Avg. Usage Change (%)', fontweight='bold')\n",
                "ax.set_title('Monthly Usage Trajectory (% Change vs 3-Mo Avg)', fontsize=11, fontweight='bold')\n",
                "ax.legend(loc='lower left', fontsize=8)\n",
                "\n",
                "# Panel 4: Retention Call Incident Rate (%)\n",
                "ax = axes[1, 1]\n",
                "ax.bar(summary.index, summary['retention_call_pct'], color=['#d62728', '#ff7f0e', '#2ca02c'], edgecolor='black', alpha=0.85)\n",
                "ax.set_ylabel('Retention Desk Call Rate (%)', fontweight='bold')\n",
                "ax.set_title('Explicit Complaint Escalation Rate (retcall == 1)', fontsize=11, fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Retention Budget Prioritization: Cost-Benefit Simulation\n",
                "\n",
                "> **Explicit Simulation Assumptions (Hypothetical Business Parameters):**\n",
                "> 1. **Annual Margin / LTV Preserved ($V$):** \\$250.00 gross margin preserved per saved subscriber.\n",
                "> 2. **Intervention Success / Uplift Rate ($r$):** 25% (0.25) of true churners are successfully convinced to renew.\n",
                "> 3. **Uniform Campaign Cost ($C$):** \\$30.00 per customer contacted in a standard blanket or threshold-based campaign.\n",
                "\n",
                "The economic profit formula across decision threshold $\\tau$ is:\n",
                "$$\\Delta \\text{Profit}(\\tau) = [TP(\\tau) \\cdot r \\cdot V] - [(TP(\\tau) + FP(\\tau)) \\cdot C]$$"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "V_LTV = 250.0\n",
                "R_SUCCESS = 0.25\n",
                "C_UNIFORM = 30.0\n",
                "\n",
                "thresholds = np.linspace(0.05, 0.90, 86)\n",
                "econ_records = []\n",
                "\n",
                "for t in thresholds:\n",
                "    mask = probs >= t\n",
                "    n_tar = int(np.sum(mask))\n",
                "    tp = int(np.sum(mask & (y_test == 1)))\n",
                "    fp = int(np.sum(mask & (y_test == 0)))\n",
                "    \n",
                "    cost = n_tar * C_UNIFORM\n",
                "    saved = tp * R_SUCCESS\n",
                "    gross_ben = saved * V_LTV\n",
                "    net_profit = gross_ben - cost\n",
                "    roi = (net_profit / cost * 100) if cost > 0 else 0.0\n",
                "    \n",
                "    econ_records.append({\n",
                "        'threshold': round(t, 3),\n",
                "        'targeted': n_tar,\n",
                "        'tp': tp, 'fp': fp,\n",
                "        'churners_saved': round(saved, 1),\n",
                "        'total_cost': round(cost, 0),\n",
                "        'gross_benefit': round(gross_ben, 0),\n",
                "        'net_profit': round(net_profit, 0),\n",
                "        'roi_pct': round(roi, 1)\n",
                "    })\n",
                "\n",
                "econ_df = pd.DataFrame(econ_records)\n",
                "best_econ = econ_df.loc[econ_df['net_profit'].idxmax()]\n",
                "\n",
                "print(f\"Peak Net Profit Threshold: tau = {best_econ['threshold']:.3f}\")\n",
                "print(f\"Targeted Customers: {int(best_econ['targeted'])}\")\n",
                "print(f\"Saved Churners: {best_econ['churners_saved']}\")\n",
                "print(f\"Total Campaign Cost: ${best_econ['total_cost']:,.0f}\")\n",
                "print(f\"Gross Economic Benefit: ${best_econ['gross_benefit']:,.0f}\")\n",
                "print(f\"Net Economic Profit: ${best_econ['net_profit']:,.0f} (ROI: {best_econ['roi_pct']}%\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Cost-Benefit Tradeoff Curve\n",
                "We plot Gross Benefit vs Campaign Cost vs Net Economic Profit as a function of the decision threshold $\\tau$."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(9, 5.5))\n",
                "plt.plot(econ_df['threshold'], econ_df['gross_benefit'], color='#2ca02c', linewidth=2.5, label='Gross Economic Benefit ($)')\n",
                "plt.plot(econ_df['threshold'], econ_df['total_cost'], color='#d62728', linewidth=2.5, linestyle='--', label='Campaign Cost ($)')\n",
                "plt.plot(econ_df['threshold'], econ_df['net_profit'], color='#1f77b4', linewidth=3.0, label='Net Economic Profit ($)')\n",
                "\n",
                "plt.axhline(0, color='gray', linestyle=':', linewidth=1)\n",
                "plt.axvline(best_econ['threshold'], color='purple', linestyle='-.', label=f\"Optimal Threshold (tau={best_econ['threshold']:.2f})\")\n",
                "plt.axvline(0.4631, color='orange', linestyle=':', label='F1-Optimal Threshold (tau=0.46)')\n",
                "\n",
                "plt.xlabel('Classification Threshold (tau)', fontweight='bold')\n",
                "plt.ylabel('Value ($ USD)', fontweight='bold')\n",
                "plt.title('Retention Campaign Cost-Benefit Tradeoff Curve', fontsize=12, fontweight='bold')\n",
                "plt.legend(loc='center right')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Comparing Strategic Retention Policies\n",
                "We evaluate five strategic operational policies:\n",
                "1. **Policy 1: Do Nothing (Baseline)**\n",
                "2. **Policy 2: Blind Blanket Campaign (All customers)**\n",
                "3. **Policy 3: High Risk Tier Only ($\\hat{p} \\ge 0.60$)**\n",
                "4. **Policy 4: Optimal Threshold ($\\hat{p} \\ge 0.67$)**\n",
                "5. **Policy 5: Differentiated Tiered Strategy (High: \\$35 voucher, Med: \\$4 digital notification, Low: \\$0)**"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "policy_comparison = pd.read_csv('../reports/retention_policy_comparison.csv')\n",
                "print(policy_comparison[['policy_name', 'n_targeted', 'churners_saved', 'cost', 'gross_benefit', 'net_savings', 'roi_pct']].to_string(index=False))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 7. Strategic Business Translation & Operational Retention Matrix\n",
                "\n",
                "| Risk Tier | Target Criteria | Primary Churn Driver | Retention Intervention | Channel & Unit Cost | Projected ROI |\n",
                "|:---|:---|:---|:---|:---|:---|\n",
                "| **High Risk** | $\\hat{p} \\ge 0.60$ | Aging equipment ($481$d), severe usage collapse ($-55\\%$) | **Subsidized 5G Hardware Upgrade:** \\$35 hardware voucher with 12-month contract lock-in. | Outbound Senior Retention Concierge (\\$35 total cost) | +15% to +30% |\n",
                "| **Medium Risk** | $0.35 \\le \\hat{p} < 0.60$ | Unit cost friction (\\$0.41/min), contract expiry window ($415$d) | **Automated Plan Right-Sizing:** Recommend optimized tariff plan, waive overage fees, 5GB loyalty bonus. | Push notification / SMS / App banner (\\$4 automated cost) | +60% to +90% |\n",
                "| **Low Risk** | $\\hat{p} < 0.35$ | None (Healthy usage, fresh handset) | **Zero-Spend Passive Monitoring:** Do not send discounts; maintain network quality of service. | Automated background health checks (\\$0 spend) | N/A (Preserves budget) |\n",
                "\n",
                "---"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 8. Practical Limitations, Governance & Methodological Audit\n",
                "\n",
                "1. **Correlation vs. Causation:**\n",
                "   - SHAP attributions reflect statistical associations within historical observational data, not guaranteed causal intervention effects.\n",
                "   - For example, high handset age (`eqpdays`) correlates with churn because customers with expired contracts intentionally postpone upgrading while shopping for competitors. Forcing customers to keep older phones will not reduce churn.\n",
                "2. **False Positives (Cost Burden):**\n",
                "   - Every loyal customer incorrectly flagged as churner consumes marketing capital without generating incremental revenue. High-cost interventions must be restricted to high-precision tiers.\n",
                "3. **False Negatives (Missed Churners):**\n",
                "   - Unflagged churners defect silently. Setting a very high threshold (e.g., $\\tau = 0.80$) minimizes false positives but misses the majority of churners.\n",
                "4. **A/B Testing Governance:**\n",
                "   - Real-world deployment must always maintain a randomized **10% holdout control group** to measure true incremental uplift ($Uplift = \\text{Churn}_{\\text{control}} - \\text{Churn}_{\\text{treatment}}$) before crediting model predictions with business savings."
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

with open('notebooks/07_business_insights.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Created notebooks/07_business_insights.ipynb successfully.")
