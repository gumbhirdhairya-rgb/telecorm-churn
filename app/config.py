"""
app/config.py
Configuration constants, paths, risk thresholds, and theme styling for Streamlit UI.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Artifact Paths (Production Cell2Cell Models with safe backward-compatible fallback)
PROD_CELL2CELL_DIR = MODELS_DIR / "production_cell2cell"
PATH_XGB_MODEL = (PROD_CELL2CELL_DIR / "best_xgboost.joblib") if (PROD_CELL2CELL_DIR / "best_xgboost.joblib").exists() else (MODELS_DIR / "best_xgboost.joblib")
PATH_PREPROCESSOR = (PROD_CELL2CELL_DIR / "final_preprocessor.joblib") if (PROD_CELL2CELL_DIR / "final_preprocessor.joblib").exists() else (MODELS_DIR / "final_preprocessor.joblib")
PATH_ISOTONIC_MODEL = (PROD_CELL2CELL_DIR / "calibrated_xgboost_isotonic.joblib") if (PROD_CELL2CELL_DIR / "calibrated_xgboost_isotonic.joblib").exists() else (MODELS_DIR / "calibrated_xgboost_isotonic.joblib")
PATH_SIGMOID_MODEL = (PROD_CELL2CELL_DIR / "calibrated_xgboost_sigmoid.joblib") if (PROD_CELL2CELL_DIR / "calibrated_xgboost_sigmoid.joblib").exists() else (MODELS_DIR / "calibrated_xgboost_sigmoid.joblib")
PATH_UNWEIGHTED_MODEL = (PROD_CELL2CELL_DIR / "unweighted_xgboost.joblib") if (PROD_CELL2CELL_DIR / "unweighted_xgboost.joblib").exists() else (MODELS_DIR / "unweighted_xgboost.joblib")
PATH_SELECTED_FEATURES = DATA_DIR / "processed" / "selected_features.json"

# CSV Data Paths
PATH_CSV_TEST_EVAL = REPORTS_DIR / "final_test_evaluation.csv"
PATH_CSV_CV_COMPARISON = REPORTS_DIR / "cv_model_comparison.csv"
PATH_CSV_TUNING = REPORTS_DIR / "hyperparameter_tuning_results.csv"
PATH_CSV_SHAP_RANKINGS = REPORTS_DIR / "shap_feature_importance.csv"
PATH_CSV_RISK_SEGMENTS = REPORTS_DIR / "retention_risk_segments.csv"
PATH_CSV_POLICIES = REPORTS_DIR / "retention_policy_comparison.csv"

# Figure Paths
FIG_FLOWCHART = FIGURES_DIR / "00_ml_pipeline_flowchart.png"
FIG_ROC_PR = FIGURES_DIR / "09_test_roc_pr_curves.png"
FIG_CONF_MAT = FIGURES_DIR / "10_confusion_matrices.png"
FIG_SHAP_BAR = FIGURES_DIR / "13_shap_global_bar.png"
FIG_SHAP_BEESWARM = FIGURES_DIR / "14_shap_summary_beeswarm.png"
FIG_SHAP_DEP = FIGURES_DIR / "15_shap_dependence_top5.png"
FIG_SHAP_WATERFALL = FIGURES_DIR / "16_shap_local_waterfalls.png"
FIG_SEGMENTATION = FIGURES_DIR / "17_risk_segmentation.png"
FIG_COST_BENEFIT = FIGURES_DIR / "18_retention_cost_benefit.png"
FIG_WORKFLOW = FIGURES_DIR / "19_retention_workflow.png"

# Thresholds & Economics
DEFAULT_THRESHOLD = 0.5000
OPTIMAL_THRESHOLD = 0.4631
HIGH_RISK_THRESHOLD = 0.60
MED_RISK_THRESHOLD = 0.35

NATURAL_PREVALENCE = 0.01962
TRAIN_PREVALENCE = 0.29014

# UI Color Palette
PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"
SUCCESS_COLOR = "#2ca02c"
DANGER_COLOR = "#d62728"
BG_LIGHT = "#f8f9fa"
BORDER_COLOR = "#e0e0e0"
