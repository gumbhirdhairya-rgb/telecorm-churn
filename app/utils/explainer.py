"""
app/utils/explainer.py
Cached TreeExplainer and dynamic SHAP local waterfall visualization generator.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import shap

@st.cache_resource(show_spinner=False)
def get_tree_explainer(_model):
    """
    Caches shap.TreeExplainer instance on the champion XGBoost model.
    Leading underscore in '_model' tells Streamlit not to hash the XGBClassifier object.
    """
    return shap.TreeExplainer(_model)

def compute_local_shap(model, sample_row_df: pd.DataFrame, artifacts: dict):
    """
    Computes exact local Shapley attributions for a single customer.
    Returns:
    - base_value: expected margin log-odds
    - total_margin: model output margin (base_value + sum(shap))
    - shap_series: pd.Series of attributions
    - top_positive: top features pushing towards churn
    - top_negative: top features pulling towards loyalty
    """
    explainer = get_tree_explainer(model)
    shap_explanation = explainer(sample_row_df)
    shap_values = shap_explanation.values[0]

    base_val = explainer.expected_value
    if isinstance(base_val, np.ndarray):
        base_val = float(base_val[0])
    else:
        base_val = float(base_val)

    shap_series = pd.Series(shap_values, index=sample_row_df.columns)
    total_margin = base_val + np.sum(shap_values)

    # Top drivers
    top_positive = shap_series[shap_series > 0].sort_values(ascending=False).head(5)
    top_negative = shap_series[shap_series < 0].sort_values(ascending=True).head(5)

    return {
        "base_value": base_val,
        "total_margin": total_margin,
        "shap_series": shap_series,
        "top_positive": top_positive,
        "top_negative": top_negative
    }

def render_local_shap_plot(shap_dict: dict, sample_row_df: pd.DataFrame, artifacts: dict, title: str = "Local SHAP Feature Attributions"):
    """
    Renders a clear, high-resolution horizontal bar plot of the top 10 local SHAP drivers
    with unscaled physical units wherever applicable.
    """
    shap_series = shap_dict["shap_series"]
    base_val = shap_dict["base_value"]
    margin = shap_dict["total_margin"]

    means = artifacts.get("means", {})
    scales = artifacts.get("scales", {})

    # Select top 10 features by absolute attribution
    top_indices = np.argsort(np.abs(shap_series.values))[::-1][:10]
    top_features = [sample_row_df.columns[i] for i in top_indices][::-1]
    top_attributions = [shap_series.iloc[i] for i in top_indices][::-1]

    colors = ['#d62728' if v > 0 else '#2ca02c' for v in top_attributions]

    # Build meaningful labels with unscaled physical values
    labels = []
    for f in top_features:
        val = sample_row_df.iloc[0][f]
        if f in means and f in scales:
            raw_val = val * scales[f] + means[f]
            labels.append(f"{f} = {raw_val:.1f} (z={val:+.2f})")
        else:
            labels.append(f"{f} = {int(val)}")

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_attributions, color=colors, edgecolor='black', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=0.9)

    ax.set_xlabel('SHAP Attribution (Impact on Model Log-Odds Margin)', fontsize=10, fontweight='bold')
    ax.set_title(f"{title}\n(Base Log-Odds: {base_val:.4f} | Total Margin: {margin:.4f})", fontsize=11, fontweight='bold', pad=10)

    # Add text labels on bars
    for i, v in enumerate(top_attributions):
        ha = 'left' if v >= 0 else 'right'
        offset = 0.01 if v >= 0 else -0.01
        ax.text(v + offset, i, f"{v:+.3f}", ha=ha, va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    return fig
