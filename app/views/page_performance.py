"""
app/views/page_performance.py
Shared Page: Model Performance & Benchmark Evaluation
Displays empirical test metrics, ROC/PR curves, confusion matrices, and CV tuning results.
"""

import streamlit as st
import pandas as pd
from app.config import (
    PATH_CSV_TEST_EVAL, 
    PATH_CSV_CV_COMPARISON, 
    PATH_CSV_TUNING,
    FIG_ROC_PR, 
    FIG_CONF_MAT
)

def render_model_performance():
    st.title("Model Performance & Benchmark Evaluation")
    st.markdown("**Rigorous evaluation across 4 candidate algorithms on the untouched 14,210-customer test partition.**")
    
    st.markdown("---")
    
    # 1. Final Test Set Evaluation Table
    st.subheader("1. Untouched Test Set Evaluation Summary")
    st.caption("All models evaluated strictly on identical test data with zero target leakage or retraining:")
    
    if PATH_CSV_TEST_EVAL.exists():
        df_test = pd.read_csv(PATH_CSV_TEST_EVAL)
        # Format numeric columns cleanly
        display_df = df_test.copy()
        for col in ['ROC_AUC', 'PR_AUC', 'Accuracy', 'Precision', 'Recall', 'F1_Score']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("Final test evaluation CSV not found.")

    st.markdown("---")
    
    # 2. ROC and Precision-Recall Visualizations
    st.subheader("2. Discrimination & Tradeoff Curves")
    col_fig1, col_fig2 = st.columns([1.1, 0.9])
    
    with col_fig1:
        st.markdown("#### ROC and Precision-Recall Curves")
        if FIG_ROC_PR.exists():
            st.image(str(FIG_ROC_PR), caption="Test ROC (AUC=0.6786) and PR Curves (AUC=0.4586)", use_container_width=True)
        else:
            st.warning("ROC/PR curves image not found.")
            
    with col_fig2:
        st.markdown("#### Confusion Matrices (Default vs Tuned Cutoff)")
        if FIG_CONF_MAT.exists():
            st.image(str(FIG_CONF_MAT), caption="Confusion Matrices at tau=0.50 vs tau*=0.4631", use_container_width=True)
        else:
            st.warning("Confusion matrix figure not found.")

    st.markdown("---")
    
    # 3. Decision Boundary Analysis
    st.subheader("3. Cost-Sensitive Threshold Optimization Analysis")
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.info(
            """
            **Default Threshold (tau = 0.5000):**
            - **Recall:** 63.73% (Catches 2,627 churners)
            - **Missed Churners (FN):** 1,495 subscribers
            - **Precision:** 40.42% (TN=6,216, FP=3,872)
            - **F1 Score:** 0.4947 (Accuracy: 62.23%)
            - *Telecom Impact:* Misses over 36% of all churners because standard 0.50 cutoff assumes symmetric error costs.
            """
        )
    with col_t2:
        st.success(
            """
            **Optimal Tuned Threshold (tau* = 0.4631):**
            - **Recall:** 72.78% (Catches 3,000 churners)
            - **Missed Churners (FN):** 1,122 subscribers
            - **Precision:** 38.30% (TN=5,256, FP=4,832)
            - **Saved Churners Delta:** **+373 additional churners detected**
            - **F1 Score:** **0.5019** (Peak out-of-fold F1)
            - *Telecom Impact:* Captures nearly 3 out of every 4 churners, substantially lowering missed defection risk.
            """
        )

    st.markdown("---")
    
    # 4. Cross-Validation and Hyperparameter Search Details
    st.subheader("4. 5-Fold Cross-Validation & Hyperparameter Tuning")
    tab_cv, tab_tune = st.tabs(["📊 Stratified 5-Fold CV Benchmark", "⚙️ Hyperparameter Grid Search"])
    
    with tab_cv:
        if PATH_CSV_CV_COMPARISON.exists():
            df_cv = pd.read_csv(PATH_CSV_CV_COMPARISON)
            st.dataframe(df_cv, use_container_width=True)
        else:
            st.info("CV comparison file not found.")
            
    with tab_tune:
        if PATH_CSV_TUNING.exists():
            df_tune = pd.read_csv(PATH_CSV_TUNING)
            st.dataframe(df_tune, use_container_width=True)
        else:
            st.info("Tuning results file not found.")
