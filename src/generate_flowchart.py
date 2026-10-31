"""
Generate End-to-End ML Pipeline Flowchart for UCS321 Project Rubrics
Requirement: "Flow diagram must be given with Pre-processing and visualization steps clearly documented."
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color palette
    colors = {
        'header': '#1A365D',
        'data': '#2B6CB0',
        'eda': '#319795',
        'prep': '#DD6B20',
        'feat': '#D69E2E',
        'model': '#805AD5',
        'eval': '#38A169',
        'biz': '#E53E3E',
        'box_bg': '#F7FAFC',
        'border': '#CBD5E0'
    }

    # Title Banner
    ax.text(50, 96, "End-to-End Machine Learning System Architecture: Telecom Customer Churn Prediction",
            ha='center', va='center', fontsize=16, fontweight='bold', color=colors['header'])
    ax.text(50, 92.5, "Problem 9 | Dataset: Cell2Cell (71,047 records x 70 features) | Aligned with Project Evaluation Rubrics",
            ha='center', va='center', fontsize=11, color='#4A5568')

    # Define steps
    steps = [
        {
            "id": 1,
            "title": "1. Data Ingestion & Audit",
            "color": colors['data'],
            "x": 4, "y": 70, "w": 20, "h": 17,
            "items": [
                "• Ingest Cell2Cell dataset (71,047 x 70)",
                "• Check duplicate rows (0 found)",
                "• Identify metadata ('customer')",
                "• Audit calibration columns ('calibrat', 'churndep')",
                "• Target ground truth: 'churn' (71% / 29%)"
            ]
        },
        {
            "id": 2,
            "title": "2. Exploratory Data Analysis (EDA)",
            "color": colors['eda'],
            "x": 28, "y": 70, "w": 20, "h": 17,
            "items": [
                "• Feature Taxonomy Categorization:",
                "  - Demographics (age, income, job)",
                "  - Usage (mou, dropvce, roam, calls)",
                "  - Billing (revenue, recchrge, months)",
                "  - Retention Feedback (retcalls, offers)",
                "• Class distribution & imbalance ratio",
                "• Correlation matrix & KDE density plots"
            ]
        },
        {
            "id": 3,
            "title": "3. Data Cleaning & Pre-processing",
            "color": colors['prep'],
            "x": 52, "y": 70, "w": 20, "h": 17,
            "items": [
                "• Drop IDs & leakage ('customer', 'churndep')",
                "• Missing value imputation strategy:",
                "  - Median imputation for skewed numeric",
                "  - Mode/Flag imputation for categorical",
                "• Outlier capping (IQR / 99th percentile)",
                "• Robust scaling / standard scaling"
            ]
        },
        {
            "id": 4,
            "title": "4. Feature Engineering & Selection",
            "color": colors['feat'],
            "x": 76, "y": 70, "w": 20, "h": 17,
            "items": [
                "• Domain Features engineered:",
                "  - Usage drop ratio (changem / mou)",
                "  - Revenue per minute (revenue / mou)",
                "  - Device obsolescence ratio (eqpdays/months)",
                "  - Retention escalation index",
                "• Multicollinearity filter (VIF)",
                "• Feature importance ranking"
            ]
        },
        {
            "id": 5,
            "title": "5. Class Balancing Strategy",
            "color": colors['model'],
            "x": 16, "y": 38, "w": 20, "h": 17,
            "items": [
                "• Address 71:29 Class Imbalance:",
                "  - Baseline: Cost-sensitive weights",
                "  - SMOTE (Synthetic Oversampling)",
                "  - Random Under-sampling benchmark",
                "• Strict train-only resampling",
                "  (zero data leakage to test split)"
            ]
        },
        {
            "id": 6,
            "title": "6. Model Development & Tuning",
            "color": colors['model'],
            "x": 40, "y": 38, "w": 20, "h": 17,
            "items": [
                "• Stratified 5-Fold Cross Validation",
                "• Candidate Classification Algorithms:",
                "  - Logistic Regression (Baseline)",
                "  - Random Forest Classifier",
                "  - XGBoost Classifier (Champion)",
                "  - LightGBM / Gradient Boosting",
                "• Hyperparameter tuning via Optuna/Grid"
            ]
        },
        {
            "id": 7,
            "title": "7. Comprehensive Evaluation",
            "color": colors['eval'],
            "x": 64, "y": 38, "w": 20, "h": 17,
            "items": [
                "• Multi-metric Evaluation Suite:",
                "  - ROC-AUC & PR-AUC curves",
                "  - Precision, Recall & F1-Score (Macro)",
                "  - Confusion Matrix & Misclassification cost",
                "  - Cumulative Gains & Decile Lift chart",
                "  - Calibration Curve (Brier score)"
            ]
        },
        {
            "id": 8,
            "title": "8. Model Explainability & Business ROI (Innovation)",
            "color": colors['biz'],
            "x": 28, "y": 6, "w": 44, "h": 18,
            "items": [
                "• Explainable AI (XAI) using SHAP (Shapley Additive exPlanations): Global & Local customer risk factors",
                "• Top Churn Drivers: Equipment age, billing increases, drop in usage, unmet customer care calls",
                "• Actionable Telecom Retention Strategy: High-risk customer prioritization & expected revenue saved",
                "• Interactive Streamlit / Dashboard Prototype for Customer Success Teams"
            ]
        }
    ]

    # Draw step cards
    for s in steps:
        rect = patches.FancyBboxPatch((s['x'], s['y']), s['w'], s['h'],
                                      boxstyle="round,pad=0.6",
                                      linewidth=2, edgecolor=s['color'], facecolor='#FFFFFF')
        ax.add_patch(rect)
        
        # Header strip inside box
        header_rect = patches.FancyBboxPatch((s['x'], s['y'] + s['h'] - 3.5), s['w'], 3.5,
                                             boxstyle="round,pad=0.2",
                                             linewidth=0, facecolor=s['color'])
        ax.add_patch(header_rect)
        ax.text(s['x'] + s['w']/2, s['y'] + s['h'] - 1.8, s['title'],
                ha='center', va='center', fontsize=9.5, fontweight='bold', color='#FFFFFF')

        # Items
        cur_y = s['y'] + s['h'] - 5.5
        for item in s['items']:
            ax.text(s['x'] + 0.8, cur_y, item, ha='left', va='center', fontsize=8, color='#2D3748')
            cur_y -= 2.2

    # Draw connector arrows
    # Row 1: 1 -> 2 -> 3 -> 4
    arrow_style = dict(arrowstyle="->", lw=2, color='#4A5568')
    ax.annotate('', xy=(28, 78.5), xytext=(24, 78.5), arrowprops=arrow_style)
    ax.annotate('', xy=(52, 78.5), xytext=(48, 78.5), arrowprops=arrow_style)
    ax.annotate('', xy=(76, 78.5), xytext=(72, 78.5), arrowprops=arrow_style)

    # From 4 to Row 2: 4 -> down to 5
    ax.annotate('', xy=(86, 60), xytext=(86, 70), arrowprops=arrow_style)
    ax.plot([86, 26], [60, 60], color='#4A5568', lw=2, linestyle='--')
    ax.annotate('', xy=(26, 55), xytext=(26, 60), arrowprops=arrow_style)

    # Row 2: 5 -> 6 -> 7
    ax.annotate('', xy=(40, 46.5), xytext=(36, 46.5), arrowprops=arrow_style)
    ax.annotate('', xy=(64, 46.5), xytext=(60, 46.5), arrowprops=arrow_style)

    # Row 2 to Row 3: 7 -> 8
    ax.annotate('', xy=(74, 28), xytext=(74, 38), arrowprops=arrow_style)
    ax.plot([74, 50], [28, 28], color='#4A5568', lw=2, linestyle='--')
    ax.annotate('', xy=(50, 24), xytext=(50, 28), arrowprops=arrow_style)

    plt.tight_layout()
    fig.savefig('reports/figures/00_ml_pipeline_flowchart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Successfully generated reports/figures/00_ml_pipeline_flowchart.png")

if __name__ == '__main__':
    draw_flowchart()
