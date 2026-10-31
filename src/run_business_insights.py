"""
src/run_business_insights.py
Step 8: Business Insights and Customer Retention Strategy
Computes:
1. Empirical customer risk segmentation (High, Medium, Low Risk).
2. Segment profile statistics (unscaled business metrics).
3. Cost-benefit simulation of retention policies under explicit hypothetical assumptions.
4. Threshold optimization curve for economic profit.
5. Generates Figures 17, 18, and 19.
6. Exports summary tables to reports/
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

os.makedirs('reports/figures', exist_ok=True)

print("[1/5] Loading champion model, test data, and preprocessor...")
model = joblib.load('models/best_xgboost.joblib')
X_test = pd.read_csv('data/processed/X_test_final.csv')
y_test = pd.read_csv('data/processed/y_test_final.csv').values.ravel()
probs = model.predict_proba(X_test)[:, 1]

preprocessor = joblib.load('models/final_preprocessor.joblib')
num_cols = list(preprocessor.named_transformers_['num'].feature_names_in_)
scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
means = dict(zip(num_cols, scaler.mean_))
scales = dict(zip(num_cols, scaler.scale_))

print(f"X_test shape: {X_test.shape}, Churn rate in test: {np.mean(y_test):.4f}")

# 2. Risk Segmentation
print("[2/5] Performing 3-tier empirical customer risk segmentation...")
# Risk tiers:
# High Risk: prob >= 0.60
# Medium Risk: 0.35 <= prob < 0.60 (contains optimal threshold tau* = 0.4631)
# Low Risk: prob < 0.35
segments = np.where(probs >= 0.60, 'High Risk', np.where(probs >= 0.35, 'Medium Risk', 'Low Risk'))

# Unscale key continuous features for business reporting
df_eval = pd.DataFrame()
for col in ['revenue', 'mou', 'recchrge', 'eqpdays', 'months', 'changem', 'cost_per_mou', 'network_friction_rate', 'overage_burden']:
    df_eval[col] = X_test[col] * scales[col] + means[col]

df_eval['retcall'] = X_test['retcall'].values
df_eval['creditaa'] = X_test['creditaa'].values
df_eval['pred_prob'] = probs
df_eval['actual_churn'] = y_test
df_eval['segment'] = segments

# Compute summary table by segment
seg_summary = df_eval.groupby('segment').agg(
    customer_count=('actual_churn', 'count'),
    pct_of_customer_base=('actual_churn', lambda x: len(x) / len(df_eval) * 100),
    actual_churn_count=('actual_churn', 'sum'),
    actual_churn_rate=('actual_churn', 'mean'),
    avg_predicted_prob=('pred_prob', 'mean'),
    avg_monthly_revenue=('revenue', 'mean'),
    avg_monthly_mou=('mou', 'mean'),
    avg_equipment_days=('eqpdays', 'mean'),
    avg_tenure_months=('months', 'mean'),
    avg_usage_change_pct=('changem', 'mean'),
    avg_cost_per_mou=('cost_per_mou', 'mean'),
    avg_network_friction=('network_friction_rate', 'mean'),
    retention_call_rate=('retcall', 'mean')
).loc[['High Risk', 'Medium Risk', 'Low Risk']].reset_index()

seg_summary.to_csv('reports/retention_risk_segments.csv', index=False)
print("Saved reports/retention_risk_segments.csv")
print("\nRisk Segmentation Profile:")
print(seg_summary[['segment', 'customer_count', 'actual_churn_rate', 'avg_equipment_days', 'avg_usage_change_pct', 'avg_cost_per_mou', 'retention_call_rate']].to_string(index=False))

# Plot Figure 17: Risk Segmentation Profiles
print("\n[3/5] Generating Figure 17: Customer Risk Segmentation Profiles...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Customer Count & Churn Rate
ax = axes[0, 0]
x_pos = np.arange(len(seg_summary))
width = 0.35
bars1 = ax.bar(x_pos - width/2, seg_summary['customer_count'], width, label='Total Customers', color='#1f77b4', edgecolor='black', alpha=0.85)
ax2 = ax.twinx()
bars2 = ax2.bar(x_pos + width/2, seg_summary['actual_churn_rate'] * 100, width, label='Actual Churn Rate (%)', color='#d62728', edgecolor='black', alpha=0.85)

ax.set_xticks(x_pos)
ax.set_xticklabels(seg_summary['segment'], fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Customers', fontsize=11, fontweight='bold', color='#1f77b4')
ax2.set_ylabel('Actual Churn Rate (%)', fontsize=11, fontweight='bold', color='#d62728')
ax2.grid(False)
ax.set_title('Customer Volume vs. Observed Churn Rate', fontsize=12, fontweight='bold')

# 2. Equipment Age by Segment
ax = axes[0, 1]
bars = ax.bar(seg_summary['segment'], seg_summary['avg_equipment_days'], color=['#d62728', '#ff7f0e', '#2ca02c'], edgecolor='black', alpha=0.85)
ax.axhline(365, color='gray', linestyle='--', linewidth=1.5, label='1-Year Contract Expiry (365 Days)')
ax.set_ylabel('Average Equipment Age (Days)', fontsize=11, fontweight='bold')
ax.set_title('Handset Age by Risk Tier (Non-Linear Inflection)', fontsize=12, fontweight='bold')
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 10, f"{b.get_height():.1f} d", ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_ylim(0, 550)

# 3. Usage Trajectory (changem)
ax = axes[1, 0]
colors_change = ['#d62728' if v < 0 else '#2ca02c' for v in seg_summary['avg_usage_change_pct']]
bars = ax.bar(seg_summary['segment'], seg_summary['avg_usage_change_pct'], color=colors_change, edgecolor='black', alpha=0.85)
ax.axhline(0, color='black', linewidth=1)
ax.axhline(-20, color='orange', linestyle=':', linewidth=1.5, label='Warning Boundary (-20%)')
ax.set_ylabel('Avg. Usage Change vs 3-Mo Avg (%)', fontsize=11, fontweight='bold')
ax.set_title('Monthly Voice Usage Trajectory (Disengagement Velocity)', fontsize=12, fontweight='bold')
for b in bars:
    val = b.get_height()
    y_text = val - 6 if val < 0 else val + 2
    ax.text(b.get_x() + b.get_width()/2, y_text, f"{val:+.1f}%", ha='center', va='bottom' if val >= 0 else 'top', fontsize=10, fontweight='bold')
ax.legend(loc='lower left', fontsize=9)
ax.set_ylim(-70, 30)

# 4. Retention Desk Call Rate
ax = axes[1, 1]
bars = ax.bar(seg_summary['segment'], seg_summary['retention_call_rate'] * 100, color=['#d62728', '#ff7f0e', '#2ca02c'], edgecolor='black', alpha=0.85)
ax.set_ylabel('Retention Call Incident Rate (%)', fontsize=11, fontweight='bold')
ax.set_title('Explicit Complaint Escalation Rate (retcall == 1)', fontsize=12, fontweight='bold')
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3, f"{b.get_height():.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0, 14)

plt.suptitle('Cell2Cell Test Set Empirical Risk Segmentation & Operational Indicators', fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig('reports/figures/17_risk_segmentation.png', dpi=300)
plt.close()
print("Saved reports/figures/17_risk_segmentation.png")

# 4. Cost-Benefit Economic Simulation
print("\n[4/5] Simulating retention economics and budget optimization...")

# Explicit hypothetical simulation parameters (Clearly Labeled)
# Assumption 1: Annual Margin / LTV preserved per saved customer = $250.00
# Assumption 2: Intervention success / uplift rate among true churners = 25% (0.25)
# Assumption 3: Fixed uniform intervention cost per customer = $30.00
V_LTV = 250.0
R_SUCCESS = 0.25
C_UNIFORM = 30.0

thresholds = np.linspace(0.05, 0.90, 86)
economic_curve = []

for t in thresholds:
    targeted_mask = probs >= t
    n_targeted = int(np.sum(targeted_mask))
    tp = int(np.sum(targeted_mask & (y_test == 1)))
    fp = int(np.sum(targeted_mask & (y_test == 0)))
    fn = int(np.sum((~targeted_mask) & (y_test == 1)))
    
    total_cost = n_targeted * C_UNIFORM
    churners_saved = tp * R_SUCCESS
    gross_benefit = churners_saved * V_LTV
    net_profit = gross_benefit - total_cost
    roi = (net_profit / total_cost * 100) if total_cost > 0 else 0.0
    
    economic_curve.append({
        'threshold': round(t, 3),
        'n_targeted': n_targeted,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': round(tp / n_targeted, 4) if n_targeted > 0 else 0.0,
        'recall': round(tp / np.sum(y_test == 1), 4),
        'churners_saved': round(churners_saved, 1),
        'total_cost': round(total_cost, 2),
        'gross_benefit': round(gross_benefit, 2),
        'net_profit': round(net_profit, 2),
        'roi': round(roi, 1)
    })

econ_df = pd.DataFrame(economic_curve)
opt_econ = econ_df.loc[econ_df['net_profit'].idxmax()]
print(f"Optimal Economic Threshold: tau = {opt_econ['threshold']:.3f}")
print(f"Max Net Profit: ${opt_econ['net_profit']:,.2f} | Targeted: {int(opt_econ['n_targeted'])} | Saved Churners: {opt_econ['churners_saved']:.1f} | ROI: {opt_econ['roi']:.1f}%")

# Compare discrete retention policies
policies = [
    {
        'policy_name': 'Policy 1: Do Nothing (Baseline)',
        'description': 'No retention campaigns; zero cost, zero saved churners',
        'threshold_rule': 'None',
        'n_targeted': 0,
        'tp': 0, 'fp': 0,
        'churners_saved': 0.0,
        'cost': 0.0,
        'gross_benefit': 0.0,
        'net_savings': 0.0,
        'roi_pct': 0.0
    },
    {
        'policy_name': 'Policy 2: Blind Blanket Campaign',
        'description': 'Target 100% of customers uniformly with $30 incentive',
        'threshold_rule': 'tau >= 0.00',
        'n_targeted': len(y_test),
        'tp': int(np.sum(y_test == 1)),
        'fp': int(np.sum(y_test == 0)),
        'churners_saved': round(np.sum(y_test == 1) * R_SUCCESS, 1),
        'cost': round(len(y_test) * C_UNIFORM, 0),
        'gross_benefit': round(np.sum(y_test == 1) * R_SUCCESS * V_LTV, 0),
        'net_savings': round(np.sum(y_test == 1) * R_SUCCESS * V_LTV - len(y_test) * C_UNIFORM, 0),
        'roi_pct': round(((np.sum(y_test == 1) * R_SUCCESS * V_LTV - len(y_test) * C_UNIFORM) / (len(y_test) * C_UNIFORM)) * 100, 1)
    },
    {
        'policy_name': 'Policy 3: High Risk Tier Only',
        'description': 'Target top risk cohort (pred prob >= 0.60)',
        'threshold_rule': 'tau >= 0.60',
        'n_targeted': int(np.sum(probs >= 0.60)),
        'tp': int(np.sum((probs >= 0.60) & (y_test == 1))),
        'fp': int(np.sum((probs >= 0.60) & (y_test == 0))),
        'churners_saved': round(np.sum((probs >= 0.60) & (y_test == 1)) * R_SUCCESS, 1),
        'cost': round(np.sum(probs >= 0.60) * C_UNIFORM, 0),
        'gross_benefit': round(np.sum((probs >= 0.60) & (y_test == 1)) * R_SUCCESS * V_LTV, 0),
        'net_savings': round(np.sum((probs >= 0.60) & (y_test == 1)) * R_SUCCESS * V_LTV - np.sum(probs >= 0.60) * C_UNIFORM, 0),
        'roi_pct': round(((np.sum((probs >= 0.60) & (y_test == 1)) * R_SUCCESS * V_LTV - np.sum(probs >= 0.60) * C_UNIFORM) / (np.sum(probs >= 0.60) * C_UNIFORM)) * 100, 1)
    },
    {
        'policy_name': 'Policy 4: Optimal Net Profit Cutoff',
        'description': f"Target economically optimal threshold (tau >= {opt_econ['threshold']:.2f})",
        'threshold_rule': f"tau >= {opt_econ['threshold']:.2f}",
        'n_targeted': int(opt_econ['n_targeted']),
        'tp': int(opt_econ['tp']),
        'fp': int(opt_econ['fp']),
        'churners_saved': round(opt_econ['churners_saved'], 1),
        'cost': round(opt_econ['total_cost'], 0),
        'gross_benefit': round(opt_econ['gross_benefit'], 0),
        'net_savings': round(opt_econ['net_profit'], 0),
        'roi_pct': round(opt_econ['roi'], 1)
    },
    {
        'policy_name': 'Policy 5: Differentiated Tiered Strategy',
        'description': 'High Risk: $35 hardware voucher (r=30%); Med Risk: $4 automated digital SMS (r=15%); Low Risk: $0',
        'threshold_rule': 'High >= 0.60, Med [0.35, 0.60), Low < 0.35',
        'n_targeted': int(np.sum(probs >= 0.35)),
        'tp': int(np.sum((probs >= 0.35) & (y_test == 1))),
        'fp': int(np.sum((probs >= 0.35) & (y_test == 0))),
        'churners_saved': round(int(np.sum((probs >= 0.60) & (y_test == 1))) * 0.30 + int(np.sum((probs >= 0.35) & (probs < 0.60) & (y_test == 1))) * 0.15, 1),
        'cost': round(int(np.sum(probs >= 0.60)) * 35.0 + int(np.sum((probs >= 0.35) & (probs < 0.60))) * 4.0, 0),
        'gross_benefit': round((int(np.sum((probs >= 0.60) & (y_test == 1))) * 0.30 + int(np.sum((probs >= 0.35) & (probs < 0.60) & (y_test == 1))) * 0.15) * V_LTV, 0),
        'net_savings': round((int(np.sum((probs >= 0.60) & (y_test == 1))) * 0.30 + int(np.sum((probs >= 0.35) & (probs < 0.60) & (y_test == 1))) * 0.15) * V_LTV - (int(np.sum(probs >= 0.60)) * 35.0 + int(np.sum((probs >= 0.35) & (probs < 0.60))) * 4.0), 0),
        'roi_pct': round((( (int(np.sum((probs >= 0.60) & (y_test == 1))) * 0.30 + int(np.sum((probs >= 0.35) & (probs < 0.60) & (y_test == 1))) * 0.15) * V_LTV - (int(np.sum(probs >= 0.60)) * 35.0 + int(np.sum((probs >= 0.35) & (probs < 0.60))) * 4.0) ) / (int(np.sum(probs >= 0.60)) * 35.0 + int(np.sum((probs >= 0.35) & (probs < 0.60))) * 4.0)) * 100, 1)
    }
]

policy_df = pd.DataFrame(policies)
policy_df.to_csv('reports/retention_policy_comparison.csv', index=False)
print("Saved reports/retention_policy_comparison.csv")
print("\nRetention Policy Economic Comparison:")
print(policy_df[['policy_name', 'n_targeted', 'churners_saved', 'cost', 'gross_benefit', 'net_savings', 'roi_pct']].to_string(index=False))

# Plot Figure 18: Cost-Benefit Tradeoff Curve
print("\nGenerating Figure 18: Cost-Benefit Tradeoff Curve...")
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(econ_df['threshold'], econ_df['gross_benefit'], color='#2ca02c', linewidth=2.5, label='Gross Economic Benefit ($)')
ax1.plot(econ_df['threshold'], econ_df['total_cost'], color='#d62728', linewidth=2.5, linestyle='--', label='Total Campaign Cost ($)')
ax1.plot(econ_df['threshold'], econ_df['net_profit'], color='#1f77b4', linewidth=3.0, label='Net Economic Profit ($)')

ax1.axhline(0, color='gray', linestyle=':', linewidth=1)
ax1.axvline(opt_econ['threshold'], color='purple', linestyle='-.', linewidth=1.8, label=f"Max Profit Cutoff (tau = {opt_econ['threshold']:.2f})")
ax1.axvline(0.4631, color='orange', linestyle=':', linewidth=1.5, label='F1-Optimal Cutoff (tau = 0.46)')

ax1.set_xlabel('Classification Decision Threshold (tau)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Currency Value ($ USD)', fontsize=11, fontweight='bold')
ax1.set_title('Retention Campaign Cost-Benefit Tradeoff Curve\n(Assumptions: V = $250 LTV, Cost = $30/user, Success = 25%)', fontsize=13, fontweight='bold', pad=12)

# Annotation for peak profit
max_p = opt_econ['net_profit']
max_t = opt_econ['threshold']
ax1.annotate(f"Peak Net Profit: ${max_p:,.0f}\n(Targeted: {int(opt_econ['n_targeted'])}, Saved: {opt_econ['churners_saved']:.0f})",
             xy=(max_t, max_p), xytext=(max_t + 0.08, max_p + 15000),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
             fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6))

# Blanket campaign annotation at tau = 0.05
blanket_loss = econ_df.iloc[0]['net_profit']
ax1.annotate(f"Blind Blanket Loss: -${abs(blanket_loss):,.0f}\n(Negative ROI)",
             xy=(0.05, blanket_loss), xytext=(0.08, blanket_loss + 25000),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6),
             fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ffcccc", alpha=0.8))

ax1.legend(loc='center right', frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig('reports/figures/18_retention_cost_benefit.png', dpi=300)
plt.close()
print("Saved reports/figures/18_retention_cost_benefit.png")

# 5. Figure 19: End-to-End Retention Workflow Diagram
print("[5/5] Generating Figure 19: End-to-End Operational Workflow Diagram...")
fig, ax = plt.subplots(figsize=(15, 8))
ax.axis('off')

# Diagram boxes
boxes = [
    # Layer 1: Data & Features
    {'x': 0.03, 'y': 0.70, 'w': 0.16, 'h': 0.22, 'color': '#dae8fc', 'edge': '#6c8ebf',
     'title': '1. Data Ingestion\n& Feature Store',
     'desc': '• Billing records (mou, rev)\n• Network logs (drop, blck)\n• Hardware logs (eqpdays)\n• Care logs (retcall)'},
    
    # Layer 2: Feature Engineering & Scoring Engine
    {'x': 0.24, 'y': 0.70, 'w': 0.18, 'h': 0.22, 'color': '#d5e8d4', 'edge': '#82b366',
     'title': '2. Scoring & Explain Engine\n(Champion XGBoost)',
     'desc': '• Compute 5 domain features\n• Predict churn log-odds\n• TreeExplainer SHAP attribution\n• Prior correction for prevalence'},

    # Layer 3: Risk Tiers
    {'x': 0.47, 'y': 0.78, 'w': 0.18, 'h': 0.15, 'color': '#f8cecc', 'edge': '#b85450',
     'title': 'High Risk Tier (p >= 0.60)\n[19.8% Base, 48.5% Churn]',
     'desc': 'Severe disengagement\nOld handsets, usage drop'},
     
    {'x': 0.47, 'y': 0.58, 'w': 0.18, 'h': 0.15, 'color': '#fff2cc', 'edge': '#d6b656',
     'title': 'Medium Risk (0.35 <= p < 0.60)\n[57.6% Base, 28.7% Churn]',
     'desc': 'Borderline, active users\nPrice/contract friction'},

    {'x': 0.47, 'y': 0.38, 'w': 0.18, 'h': 0.15, 'color': '#e1d5e7', 'edge': '#9673a6',
     'title': 'Low Risk Tier (p < 0.35)\n[22.6% Base, 12.7% Churn]',
     'desc': 'Stable, highly satisfied\nFresh equipment, low friction'},

    # Layer 4: Actionable Retention Actions
    {'x': 0.70, 'y': 0.78, 'w': 0.26, 'h': 0.15, 'color': '#f8cecc', 'edge': '#b85450',
     'title': 'Concierge & Hardware Upgrade',
     'desc': '• $35-$50 subsidized 5G handset offer\n• Contract renewal lock-in (12 mos)\n• Outbound call from Senior Concierge'},

    {'x': 0.70, 'y': 0.58, 'w': 0.26, 'h': 0.15, 'color': '#fff2cc', 'edge': '#d6b656',
     'title': 'Automated Digital Plan Right-Sizing',
     'desc': '• $4 automated app/SMS notification\n• Plan optimization (lower cost/mou)\n• Overage forgiveness & bonus 5GB'},

    {'x': 0.70, 'y': 0.38, 'w': 0.26, 'h': 0.15, 'color': '#e1d5e7', 'edge': '#9673a6',
     'title': 'Zero-Cost Delight & Monitoring',
     'desc': '• $0 spend; preserve marketing budget\n• Passive loyalty perk tracking\n• Continuous background health checks'},

    # Layer 5: Feedback & Causal Evaluation
    {'x': 0.24, 'y': 0.08, 'w': 0.64, 'h': 0.20, 'color': '#f5f5f5', 'edge': '#666666',
     'title': '5. Closed-Loop Uplift & Governance Monitoring (A/B Testing Framework)',
     'desc': '• 10% Holdout Control Group (measure true causal incrementality: Uplift = Churn_control - Churn_treatment)\n• Track False Positive cost burden vs True Positive LTV preservation\n• Quarterly model recalibration & drift detection'}
]

for b in boxes:
    rect = patches.FancyBboxPatch((b['x'], b['y']), b['w'], b['h'],
                                  boxstyle="round,pad=0.02",
                                  facecolor=b['color'], edgecolor=b['edge'], linewidth=2)
    ax.add_patch(rect)
    ax.text(b['x'] + b['w']/2, b['y'] + b['h'] - 0.04, b['title'],
            ha='center', va='top', fontsize=9.5, fontweight='bold')
    ax.text(b['x'] + 0.015, b['y'] + 0.02, b['desc'],
            ha='left', va='bottom', fontsize=8.5, color='#222222')

# Draw arrows
arrows = [
    # 1 -> 2
    ((0.19, 0.81), (0.24, 0.81)),
    # 2 -> 3 (High)
    ((0.42, 0.83), (0.47, 0.85)),
    # 2 -> 3 (Med)
    ((0.42, 0.79), (0.47, 0.65)),
    # 2 -> 3 (Low)
    ((0.42, 0.75), (0.47, 0.45)),
    # 3 -> 4 (High)
    ((0.65, 0.85), (0.70, 0.85)),
    # 3 -> 4 (Med)
    ((0.65, 0.65), (0.70, 0.65)),
    # 3 -> 4 (Low)
    ((0.65, 0.45), (0.70, 0.45)),
    # 4 -> 5 feedback
    ((0.83, 0.38), (0.83, 0.28)),
    ((0.24, 0.18), (0.13, 0.70))
]

for start, end in arrows:
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color='#333333', lw=1.8, mutation_scale=14))

ax.set_title('End-to-End Operational Telecom Retention Workflow Architecture',
             fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('reports/figures/19_retention_workflow.png', dpi=300)
plt.close()
print("Saved reports/figures/19_retention_workflow.png")

print("\nStep 8 computation & figure generation completed successfully!")
