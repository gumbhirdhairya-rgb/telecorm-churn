import sys
import os
from pathlib import Path

# Ensure project root is at sys.path[0] and remove app directory from sys.path
# to prevent Python from confusing the folder 'app/' with 'app.py' (circular import)
project_root = str(Path(__file__).resolve().parent.parent)
app_dir = str(Path(__file__).resolve().parent)

if app_dir in sys.path:
    sys.path.remove(app_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
else:
    sys.path.remove(project_root)
    sys.path.insert(0, project_root)

import streamlit as st
from app.views.page_overview import render_overview
from app.views.page_individual import render_individual_prediction
from app.views.page_bulk import render_bulk_analysis
from app.views.page_performance import render_model_performance
from app.views.page_shap import render_shap_explainability
from app.views.page_segmentation import render_risk_segmentation
from app.views.page_strategy import render_business_strategy

# 1. Page Configuration
st.set_page_config(
    page_title="Cell2Cell Telecom Churn Intelligence",
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern Enterprise Styling (CSS Injection)
st.markdown(
    """
    <style>
    /* Metric Card Enhancements */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #1f77b4;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar Enhancements */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    
    /* Table & Dataframe Styling */
    .dataframe {
        font-size: 0.9rem !important;
    }
    
    /* Card Padding & Headers */
    h1, h2, h3 {
        color: #212529;
    }
    
    /* Custom Badge */
    .badge-champion {
        background-color: #2ca02c;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 3. Sidebar Navigation & Global Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/cell-phone.png", width=64)
    st.title("Telecom Churn AI")
    st.caption("Cell2Cell Enterprise Retention Platform")
    
    st.markdown("---")
    st.subheader("Navigation")
    
    selected_page = st.radio(
        "Select Platform Module:",
        options=[
            "Executive Overview",
            "Individual Prediction",
            "Bulk Portfolio Analysis",
            "Model Performance",
            "SHAP Explainability",
            "Risk Segmentation",
            "Business Strategy & ROI"
        ],
        index=0
    )
    
    st.markdown("---")
    
    # Model Metadata Sidebar Card
    st.markdown("#### System Specifications")
    st.info(
        """
        - **Model:** Tuned XGBoost
        - **Features:** 63 (4 Filtered, 5 Engineered)
        - **Optimal Cutoff:** $\\tau^* = 0.4631$
        - **Test ROC-AUC:** 0.6786
        - **Test PR-AUC:** 0.4586
        - **Evaluation:** UCS321 Rubric Aligned
        """
    )
    
    st.caption("© 2026 Telecom Customer Retention System. Built for academic viva and enterprise deployment.")

# 4. View Router
if selected_page == "Executive Overview":
    render_overview()
elif selected_page == "Individual Prediction":
    render_individual_prediction()
elif selected_page == "Bulk Portfolio Analysis":
    render_bulk_analysis()
elif selected_page == "Model Performance":
    render_model_performance()
elif selected_page == "SHAP Explainability":
    render_shap_explainability()
elif selected_page == "Risk Segmentation":
    render_risk_segmentation()
elif selected_page == "Business Strategy & ROI":
    render_business_strategy()
