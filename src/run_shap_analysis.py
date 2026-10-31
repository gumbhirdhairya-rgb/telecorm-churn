"""
src/run_shap_analysis.py
Computes Tree SHAP values on the untouched test set using the champion XGBoost model.
Generates:
1. Global Bar Plot (Top 20 features by mean |SHAP|)
2. Global Summary Beeswarm Plot (Top 20 features)
3. Dependence Plots for Key Features (showing non-linear thresholds)
4. Local Waterfall Plots for High, Medium, and Low Risk customer personas
5. Detailed CSV summaries for reporting
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

os.makedirs('reports/figures', exist_ok=True)

print("[1/6] Loading champion XGBoost model, test dataset, and preprocessor...")
model = joblib.load('models/best_xgboost.joblib')
X_test = pd.read_csv('data/processed/X_test_final.csv')
y_test = pd.read_csv('data/processed/y_test_final.csv').values.ravel()
preprocessor = joblib.load('models/final_preprocessor.joblib')

num_cols = list(preprocessor.named_transformers_['num'].feature_names_in_)
scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
means = dict(zip(num_cols, scaler.mean_))
scales = dict(zip(num_cols, scaler.scale_))

print(f"Loaded X_test: {X_test.shape}, model: {type(model).__name__}")

# TreeExplainer
print("[2/6] Computing SHAP values using TreeExplainer...")
explainer = shap.TreeExplainer(model)
shap_explanation = explainer(X_test)
shap_values = shap_explanation.values # shape: (14210, 63)
base_value = explainer.expected_value
if isinstance(base_value, np.ndarray):
    base_value = float(base_value[0])
else:
    base_value = float(base_value)

print(f"SHAP values computed: {shap_values.shape}")
print(f"Explainer base value (log-odds margin): {base_value:.5f} (Implied prior prob: {1 / (1 + np.exp(-base_value)):.4f})")

# Feature ranking
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
ranking_df = pd.DataFrame({
    'feature': X_test.columns,
    'mean_abs_shap': mean_abs_shap,
    'mean_shap': np.mean(shap_values, axis=0),
    'std_shap': np.std(shap_values, axis=0),
    'max_shap': np.max(shap_values, axis=0),
    'min_shap': np.min(shap_values, axis=0)
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

ranking_df['rank'] = ranking_df.index + 1
ranking_df.to_csv('reports/shap_feature_importance.csv', index=False)
print("Saved reports/shap_feature_importance.csv")
print("\nTop 15 features by Mean Absolute SHAP Value:")
print(ranking_df[['rank', 'feature', 'mean_abs_shap', 'mean_shap']].head(15).to_string(index=False))

# Plot 1: Global Bar Plot (Top 20)
print("\n[3/6] Generating Figure 13: SHAP Global Bar Plot...")
top_20 = ranking_df.head(20).iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(top_20['feature'], top_20['mean_abs_shap'], color='#1f77b4', edgecolor='black', alpha=0.85)
ax.set_xlabel('Mean |SHAP Value| (Average Impact on Model Log-Odds Output)', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Features by Global SHAP Importance (XGBoost Champion)', fontsize=14, fontweight='bold', pad=15)

for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.003, bar.get_y() + bar.get_height()/2, f'{w:.4f}', ha='left', va='center', fontsize=9)

ax.set_xlim(0, max(top_20['mean_abs_shap']) * 1.15)
plt.tight_layout()
plt.savefig('reports/figures/13_shap_global_bar.png', dpi=300)
plt.close()
print("Saved reports/figures/13_shap_global_bar.png")

# Plot 2: Global Beeswarm Plot (Top 20)
print("[4/6] Generating Figure 14: SHAP Summary Beeswarm Plot...")
fig = plt.figure(figsize=(11, 9))
shap.plots.beeswarm(shap_explanation, max_display=20, show=False)
plt.title('SHAP Summary Beeswarm Plot: Feature Impact on Churn Risk', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('SHAP Value (Impact on Log-Odds of Churning)', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/14_shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved reports/figures/14_shap_summary_beeswarm.png")

# Plot 3: Dependence Plots for Key Features (2x3 grid)
print("[5/6] Generating Figure 15: SHAP Dependence Plots...")
dep_features = ['eqpdays', 'months', 'changem', 'cost_per_mou', 'handset_age_ratio', 'retcall']

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for idx, feat in enumerate(dep_features):
    ax = axes[idx]
    f_idx = list(X_test.columns).index(feat)
    x_val = X_test[feat].values
    y_val = shap_values[:, f_idx]
    
    # Scatter plot
    scatter = ax.scatter(x_val, y_val, c='#2b5c8f', alpha=0.25, s=12, edgecolors='none')
    
    # Lowess or moving average trend line
    sort_order = np.argsort(x_val)
    x_sorted = x_val[sort_order]
    y_sorted = y_val[sort_order]
    
    # Compute rolling median to show non-linear threshold clearly
    window = max(50, len(x_sorted) // 100)
    rolling_x = pd.Series(x_sorted).rolling(window, center=True).mean().dropna()
    rolling_y = pd.Series(y_sorted).rolling(window, center=True).mean().dropna()
    ax.plot(rolling_x, rolling_y, color='#d62728', linewidth=2.5, label='Trend (Rolling Mean)')
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    
    # If continuous, show unscaled zero point
    if feat in means:
        raw_zero = means[feat]
        ax.set_xlabel(f'{feat} (Standardized, Mean={raw_zero:.1f})', fontsize=10, fontweight='bold')
    else:
        ax.set_xlabel(f'{feat}', fontsize=10, fontweight='bold')
        
    ax.set_ylabel(f'SHAP Value for {feat}', fontsize=10, fontweight='bold')
    ax.set_title(f'SHAP Dependence: {feat}', fontsize=12, fontweight='bold')
    ax.legend(loc='best', frameon=True, fontsize=8)

plt.suptitle('Non-Linear Feature Relationships and Inflection Points (XGBoost)', fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig('reports/figures/15_shap_dependence_top5.png', dpi=300)
plt.close()
print("Saved reports/figures/15_shap_dependence_top5.png")

# Plot 4: Local Waterfall Plots for 3 Representative Personas
print("[6/6] Generating Figure 16: Local Waterfall Plots for 3 Customer Personas...")
probs = model.predict_proba(X_test)[:, 1]

# Candidates:
# High risk: prob > 0.85, true churn = 1
high_idx = int(np.where((probs > 0.85) & (y_test == 1))[0][0])
# Medium risk: closest to tau* = 0.4631
med_idx = int(np.argmin(np.abs(probs - 0.4631)))
# Low risk: prob < 0.10, true churn = 0
low_idx = int(np.where((probs < 0.10) & (y_test == 0))[0][0])

customer_personas = [
    ('High Churn Risk Customer', high_idx, '#d62728'),
    ('Medium (Borderline) Risk Customer', med_idx, '#ff7f0e'),
    ('Low Churn Risk (Loyal) Customer', low_idx, '#2ca02c')
]

local_records = []

fig, axes = plt.subplots(3, 1, figsize=(13, 16))

for ax_idx, (persona_name, c_idx, bar_color) in enumerate(customer_personas):
    c_prob = probs[c_idx]
    c_true = int(y_test[c_idx])
    c_shap = shap_values[c_idx]
    c_base = base_value
    c_margin = c_base + np.sum(c_shap)
    
    # Top 10 features by absolute contribution
    top_c_indices = np.argsort(np.abs(c_shap))[::-1][:10]
    top_c_features = [X_test.columns[i] for i in top_c_indices][::-1]
    top_c_values = [c_shap[i] for i in top_c_indices][::-1]
    
    colors = ['#d62728' if v > 0 else '#2ca02c' for v in top_c_values]
    
    ax = axes[ax_idx]
    y_pos = np.arange(len(top_c_features))
    ax.barh(y_pos, top_c_values, color=colors, edgecolor='black', alpha=0.85)
    ax.set_yticks(y_pos)
    
    # Format labels with original unscaled values where applicable
    ytick_labels = []
    for f in top_c_features:
        f_val = X_test.iloc[c_idx][f]
        if f in means:
            raw_val = f_val * scales[f] + means[f]
            ytick_labels.append(f"{f} = {raw_val:.1f} (z={f_val:+.2f})")
        else:
            ytick_labels.append(f"{f} = {int(f_val)}")
    
    ax.set_yticklabels(ytick_labels, fontsize=9)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('SHAP Value (Contribution to Churn Log-Odds)', fontsize=10, fontweight='bold')
    ax.set_title(f"{persona_name} (Index #{c_idx}) | Predicted Churn Prob: {c_prob:.2%} | Actual Label: {c_true} (Base={c_base:.3f}, Margin={c_margin:.3f})",
                 fontsize=12, fontweight='bold', pad=10)
    
    # Record for CSV
    for rank, i in enumerate(top_c_indices, start=1):
        f = X_test.columns[i]
        f_val = X_test.iloc[c_idx][f]
        raw_val = f_val * scales[f] + means[f] if f in means else f_val
        local_records.append({
            'persona': persona_name,
            'customer_index': c_idx,
            'predicted_probability': round(c_prob, 4),
            'actual_churn': c_true,
            'rank': rank,
            'feature': f,
            'standardized_value': round(f_val, 4),
            'unscaled_raw_value': round(raw_val, 2),
            'shap_value': round(c_shap[i], 4),
            'direction': 'Increases Churn' if c_shap[i] > 0 else 'Reduces Churn'
        })

plt.suptitle('Local SHAP Feature Attributions for 3 Representative Customer Personas', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('reports/figures/16_shap_local_waterfalls.png', dpi=300)
plt.close()
print("Saved reports/figures/16_shap_local_waterfalls.png")

# Save customer details to CSV
pd.DataFrame(local_records).to_csv('reports/shap_local_customers.csv', index=False)
print("Saved reports/shap_local_customers.csv")
print("\nSHAP Analysis completed successfully!")
