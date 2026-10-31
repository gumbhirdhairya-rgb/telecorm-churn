"""
Dataset Inspection & Foundation Analysis Script
Dataset: Cell2Cell Telecom Customer Churn (71,047 rows x 70 columns)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_inspection(csv_path="data/raw/cell2cell.csv", output_dir="reports"):
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    
    print(f"[+] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    total_rows, total_cols = df.shape
    print(f"[+] Dataset Shape: {total_rows:,} rows, {total_cols} columns")
    
    # 1. Duplicates
    dup_count = int(df.duplicated().sum())
    print(f"[+] Exact duplicate rows: {dup_count}")
    
    # 2. Target Variable Analysis
    churn_counts = df['churn'].value_counts()
    churn_pct = df['churn'].value_counts(normalize=True) * 100
    print("\n[+] Target Distribution ('churn'):")
    print(f"    - Stayed (0): {churn_counts.get(0, 0):,} ({churn_pct.get(0, 0):.2f}%)")
    print(f"    - Churned (1): {churn_counts.get(1, 0):,} ({churn_pct.get(1, 0):.2f}%)")
    
    # Check calibrat & churndep
    calibrat_dist = df['calibrat'].value_counts().to_dict()
    churndep_missing = int(df['churndep'].isnull().sum())
    
    # 3. Missing Value Analysis
    missing_series = df.isnull().sum()
    missing_pct = (missing_series / len(df)) * 100
    missing_df = pd.DataFrame({
        'Feature': missing_series.index,
        'Missing_Count': missing_series.values,
        'Missing_Percentage': missing_pct.values
    }).sort_values(by='Missing_Count', ascending=False)
    
    features_with_missing = missing_df[missing_df['Missing_Count'] > 0]
    print(f"\n[+] Features with missing values: {len(features_with_missing)}")
    for _, row in features_with_missing.iterrows():
        print(f"    - {row['Feature']:12s}: {int(row['Missing_Count']):,} ({row['Missing_Percentage']:.2f}%)")

    # 4. Feature Taxonomy according to Problem Statement 9
    taxonomy = {
        "Demographic Details": [
            "age1", "age2", "children", "credita", "creditaa", "prizmrur", "prizmub", 
            "prizmtwn", "ownrent", "marryun", "marryyes", "income", "incmiss", 
            "occprof", "occcler", "occcrft", "occstud", "occhmkr", "occret", "occself"
        ],
        "Usage Patterns": [
            "mou", "overage", "roam", "changem", "dropvce", "blckvce", "unansvce", 
            "custcare", "threeway", "mourec", "outcalls", "incalls", "peakvce", 
            "opeakvce", "dropblk", "callfwdv", "callwait"
        ],
        "Billing Information": [
            "revenue", "recchrge", "directas", "changer", "months", "uniqsubs", 
            "actvsubs", "phones", "models", "eqpdays", "setprcm", "setprc", "refurb", "webcap"
        ],
        "Service Feedback & Retention": [
            "retcalls", "retaccpt", "retcall", "refer"
        ],
        "Lifestyle & Behavioral Attributes": [
            "mailord", "mailres", "mailflag", "travel", "pcown", "creditcd", 
            "newcelly", "newcelln", "mcycle", "truck", "rv"
        ],
        "Metadata / Administrative": [
            "customer", "calibrat", "churndep"
        ]
    }

    # Verify all columns mapped
    mapped_cols = sum(taxonomy.values(), [])
    unmapped = [c for c in df.columns if c not in mapped_cols]
    print(f"\n[+] Taxonomy mapping: {len(mapped_cols)} categorized, unmapped: {unmapped}")

    # 5. Numerical and Binary Categorical breakdown
    binary_cols = [c for c in df.columns if df[c].dropna().isin([0, 1]).all() and c not in ['customer', 'calibrat', 'churndep', 'churn']]
    continuous_cols = [c for c in df.columns if c not in binary_cols and c not in ['customer', 'calibrat', 'churndep', 'churn']]
    print(f"[+] Feature Types: {len(binary_cols)} binary flag features, {len(continuous_cols)} continuous/numeric features")

    # 6. Correlations with Churn
    numeric_df = df.drop(columns=['customer', 'churndep', 'calibrat'], errors='ignore')
    corr_churn = numeric_df.corr()['churn'].sort_values()
    print("\n[+] Top 5 Negative Correlations with Churn:")
    for feat, val in corr_churn.head(5).items():
        print(f"    - {feat:15s}: {val:.4f}")
    print("[+] Top 5 Positive Correlations with Churn:")
    for feat, val in corr_churn.tail(6).iloc[:-1].items():  # exclude churn itself
        print(f"    - {feat:15s}: {val:.4f}")

    # 7. Generate Key Exploratory Figures
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Figure 1: Target Variable Distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    churn_labels = ['Retained (0)', 'Churned (1)']
    colors = ['#2b5c8f', '#d95f02']
    axes[0].bar(churn_labels, churn_counts.values, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
    for i, v in enumerate(churn_counts.values):
        axes[0].text(i, v + 800, f"{v:,}\n({churn_pct.values[i]:.1f}%)", ha='center', fontweight='bold')
    axes[0].set_title("Customer Churn Class Distribution (Counts)", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Number of Customers", fontsize=11)
    axes[0].set_ylim(0, max(churn_counts.values) * 1.15)
    
    axes[1].pie(churn_counts.values, labels=churn_labels, autopct='%1.1f%%', colors=colors, startangle=140, 
                explode=(0, 0.08), shadow=True, textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[1].set_title("Churn Ratio (Imbalance: ~71% vs ~29%)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f"{output_dir}/figures/01_churn_distribution.png", dpi=300)
    plt.close()
    print(f"[+] Saved figure: {output_dir}/figures/01_churn_distribution.png")

    # Figure 2: Missing Values Bar Chart
    if len(features_with_missing) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Exclude churndep from display since it is a calibration artifact
        plot_missing = features_with_missing[features_with_missing['Feature'] != 'churndep']
        sns.barplot(data=plot_missing, x='Missing_Percentage', y='Feature', ax=ax, palette='Blues_r', edgecolor='black')
        ax.set_title("Missing Values Percentage by Feature (Excluding Calibration Artifact)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Missing Percentage (%)", fontsize=11)
        ax.set_ylabel("Feature Name", fontsize=11)
        for i, (cnt, pct) in enumerate(zip(plot_missing['Missing_Count'], plot_missing['Missing_Percentage'])):
            ax.text(pct + 0.05, i, f"{cnt:,} ({pct:.2f}%)", va='center', fontsize=9, fontweight='bold')
        plt.tight_layout()
        fig.savefig(f"{output_dir}/figures/02_missing_values.png", dpi=300)
        plt.close()
        print(f"[+] Saved figure: {output_dir}/figures/02_missing_values.png")

    # Figure 3: Key Numerical Distributions by Churn Status
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    key_features = [
        ('eqpdays', 'Equipment Days (Age of Device)', 'Days'),
        ('months', 'Customer Tenure (Months in Service)', 'Months'),
        ('revenue', 'Monthly Revenue ($)', 'USD'),
        ('mou', 'Total Monthly Minutes of Use', 'Minutes')
    ]
    for idx, (col, title, unit) in enumerate(key_features):
        ax = axes[idx // 2, idx % 2]
        # Filter extreme outliers for clean visualization
        p99 = df[col].quantile(0.99)
        sample_data = df[df[col] <= p99]
        sns.kdeplot(data=sample_data, x=col, hue='churn', common_norm=False, ax=ax, palette=['#2b5c8f', '#d95f02'], fill=True, alpha=0.3)
        ax.set_title(f"Distribution of {title}", fontsize=12, fontweight='bold')
        ax.set_xlabel(f"{col} ({unit})", fontsize=10)
        ax.legend(['Churned (1)', 'Retained (0)'], title="Churn")
    plt.tight_layout()
    fig.savefig(f"{output_dir}/figures/03_key_features_by_churn.png", dpi=300)
    plt.close()
    print(f"[+] Saved figure: {output_dir}/figures/03_key_features_by_churn.png")

    # Figure 4: Top Correlated Features with Churn
    top_corr = pd.concat([corr_churn.head(8), corr_churn.tail(9).iloc[:-1]]) # top 8 negative, top 8 positive
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#2b5c8f' if v < 0 else '#d95f02' for v in top_corr.values]
    ax.barh(top_corr.index, top_corr.values, color=colors, edgecolor='black', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title("Top 16 Features Correlated with Customer Churn (Pearson)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Correlation Coefficient (r)", fontsize=11)
    for i, v in enumerate(top_corr.values):
        align = 'right' if v < 0 else 'left'
        offset = -0.005 if v < 0 else 0.005
        ax.text(v + offset, i, f"{v:+.3f}", va='center', ha=align, fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f"{output_dir}/figures/04_top_correlations.png", dpi=300)
    plt.close()
    print(f"[+] Saved figure: {output_dir}/figures/04_top_correlations.png")

    # Generate JSON summary for reuse in reports & notebooks
    summary = {
        "dataset_name": "Cell2Cell Telecom Customer Churn",
        "total_records": total_rows,
        "total_columns": total_cols,
        "duplicate_rows": dup_count,
        "target_variable": "churn",
        "class_distribution": {
            "retained_0": int(churn_counts.get(0, 0)),
            "retained_pct": float(churn_pct.get(0, 0)),
            "churned_1": int(churn_counts.get(1, 0)),
            "churned_pct": float(churn_pct.get(1, 0)),
            "imbalance_ratio": f"{churn_pct.get(0, 0)/churn_pct.get(1, 0):.2f}:1"
        },
        "calibration_notes": {
            "calibrat_column_present": True,
            "calibrat_1_balanced_subsample": calibrat_dist.get(1, 0),
            "calibrat_0_evaluation_sample": calibrat_dist.get(0, 0),
            "churndep_missing_count": churndep_missing,
            "recommendation": "Use 'churn' as the true ground truth label across all 71,047 rows. Drop 'calibrat' and 'churndep' prior to model training to prevent data leakage."
        },
        "taxonomy": taxonomy,
        "features_with_missing": [
            {"feature": r['Feature'], "missing_count": int(r['Missing_Count']), "percentage": round(float(r['Missing_Percentage']), 3)}
            for _, r in features_with_missing.iterrows()
        ]
    }
    
    with open(f"{output_dir}/dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print(f"[+] Saved dataset summary JSON to {output_dir}/dataset_summary.json")
    print("[+] Dataset Inspection completed successfully!\n")

if __name__ == "__main__":
    run_inspection()
