"""
app/utils/model_loader.py
Cached artifact loader for serialized models, preprocessors, and feature schemas.
"""

import json
import joblib
import pandas as pd
import streamlit as st
from app.config import (
    PATH_XGB_MODEL,
    PATH_PREPROCESSOR,
    PATH_ISOTONIC_MODEL,
    PATH_SIGMOID_MODEL,
    PATH_SELECTED_FEATURES,
    MODELS_DIR
)

@st.cache_resource(show_spinner=False)
def load_models_and_preprocessor():
    """
    Loads and caches:
    1. Champion XGBoost Model (best_xgboost.joblib)
    2. Final Preprocessor Pipeline (final_preprocessor.joblib)
    3. Isotonic Calibrated Model (calibrated_xgboost_isotonic.joblib)
    4. Sigmoid Calibrated Model (calibrated_xgboost_sigmoid.joblib)
    5. Unweighted XGBoost Model (unweighted_xgboost.joblib, if exists)
    6. Selected 63 feature schema metadata
    """
    if not PATH_XGB_MODEL.exists():
        raise FileNotFoundError(f"Champion XGBoost model not found at {PATH_XGB_MODEL}")
    if not PATH_PREPROCESSOR.exists():
        raise FileNotFoundError(f"Final preprocessor not found at {PATH_PREPROCESSOR}")

    xgb_model = joblib.load(PATH_XGB_MODEL)
    preprocessor = joblib.load(PATH_PREPROCESSOR)

    iso_model = joblib.load(PATH_ISOTONIC_MODEL) if PATH_ISOTONIC_MODEL.exists() else None
    sig_model = joblib.load(PATH_SIGMOID_MODEL) if PATH_SIGMOID_MODEL.exists() else None

    unw_path = MODELS_DIR / "unweighted_xgboost.joblib"
    unw_model = joblib.load(unw_path) if unw_path.exists() else None

    # Load selected features schema
    with open(PATH_SELECTED_FEATURES, "r") as f:
        selected_meta = json.load(f)
    selected_features = selected_meta["selected_features"]

    # Extract column names and scaling parameters
    num_cols = list(preprocessor.named_transformers_["num"].feature_names_in_)
    bin_cols = list(preprocessor.named_transformers_["bin"].feature_names_in_)
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    num_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    bin_imputer = preprocessor.named_transformers_["bin"].named_steps["imputer"]

    means = dict(zip(num_cols, scaler.mean_))
    scales = dict(zip(num_cols, scaler.scale_))
    num_medians = dict(zip(num_cols, num_imputer.statistics_))
    bin_modes = dict(zip(bin_cols, bin_imputer.statistics_))

    # Combine reference medians/modes for baseline imputation
    baseline_defaults = {**num_medians, **bin_modes}

    return {
        "xgb_model": xgb_model,
        "preprocessor": preprocessor,
        "iso_model": iso_model,
        "sig_model": sig_model,
        "unw_model": unw_model,
        "selected_features": selected_features,
        "num_cols": num_cols,
        "bin_cols": bin_cols,
        "expected_order": num_cols + bin_cols,
        "means": means,
        "scales": scales,
        "baseline_defaults": baseline_defaults
    }
