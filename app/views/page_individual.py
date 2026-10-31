"""
app/views/page_individual.py
Primary Mode 1: Individual Customer Prediction UI
Live model inference, dynamic TreeExplainer SHAP attribution, and tailored retention action plans.
"""

import streamlit as st
import pandas as pd
import numpy as np

from app.config import OPTIMAL_THRESHOLD, DEFAULT_THRESHOLD
from app.utils.model_loader import load_models_and_preprocessor
from app.utils.preprocessor import prepare_input_dataframe
from app.utils.explainer import compute_local_shap, render_local_shap_plot
from app.utils.business_logic import (
    assign_risk_tier, 
    apply_bayes_prior_correction, 
    get_retention_recommendation
)

def render_individual_prediction():
    st.title("Individual Customer Churn Prediction")
    st.markdown("**Real-time customer risk assessment, explainable SHAP diagnostics, and tailored retention actions.**")
    
    # Load serialized artifacts
    try:
        artifacts = load_models_and_preprocessor()
    except Exception as e:
        st.error(f"❌ Failed to load model artifacts: {str(e)}")
        return

    st.markdown("---")
    
    # 1. Preset Personas Section
    st.subheader("1. Quick-Load Customer Persona Presets")
    st.caption("Select a real test cohort persona to populate the input attributes:")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    # Initialize session state for form values if not present
    if 'eqpdays' not in st.session_state:
        st.session_state.eqpdays = 380
        st.session_state.mou = 520.0
        st.session_state.revenue = 58.0
        st.session_state.recchrge = 45.0
        st.session_state.months = 18
        st.session_state.changem = -5.0
        st.session_state.overage = 15.0
        st.session_state.dropvce = 2.0
        st.session_state.blckvce = 1.0
        st.session_state.retcall = 0
        st.session_state.creditaa = 1
        st.session_state.age1 = 45
        st.session_state.uniqsubs = 1
        st.session_state.refurb = 0

    with col_p1:
        if st.button("🔴 Load High Risk Persona (#606)", use_container_width=True):
            st.session_state.eqpdays = 374
            st.session_state.mou = 0.0
            st.session_state.revenue = 56.0
            st.session_state.recchrge = 50.0
            st.session_state.months = 18
            st.session_state.changem = -60.0
            st.session_state.overage = 0.0
            st.session_state.dropvce = 0.0
            st.session_state.blckvce = 0.0
            st.session_state.retcall = 1
            st.session_state.creditaa = 0
            st.session_state.age1 = 30
            st.session_state.uniqsubs = 2
            st.session_state.refurb = 1
            st.rerun()

    with col_p2:
        if st.button("🟠 Load Medium (Borderline) Persona (#13293)", use_container_width=True):
            st.session_state.eqpdays = 382
            st.session_state.mou = 530.0
            st.session_state.revenue = 59.0
            st.session_state.recchrge = 46.0
            st.session_state.months = 13
            st.session_state.changem = 27.0
            st.session_state.overage = 45.0
            st.session_state.dropvce = 1.0
            st.session_state.blckvce = 0.0
            st.session_state.retcall = 0
            st.session_state.creditaa = 1
            st.session_state.age1 = 42
            st.session_state.uniqsubs = 2
            st.session_state.refurb = 0
            st.rerun()

    with col_p3:
        if st.button("🟢 Load Low Risk (Loyal) Persona (#452)", use_container_width=True):
            st.session_state.eqpdays = 69
            st.session_state.mou = 650.0
            st.session_state.revenue = 62.0
            st.session_state.recchrge = 35.0
            st.session_state.months = 59
            st.session_state.changem = 18.0
            st.session_state.overage = 0.0
            st.session_state.dropvce = 0.0
            st.session_state.blckvce = 0.0
            st.session_state.retcall = 0
            st.session_state.creditaa = 1
            st.session_state.age1 = 56
            st.session_state.uniqsubs = 1
            st.session_state.refurb = 0
            st.rerun()

    st.markdown("---")
    
    # 2. Input Configuration Form
    st.subheader("2. Customer Attribute Parameters")
    
    with st.form("customer_input_form"):
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📱 Hardware & Equipment", 
            "⏱️ Usage & Activity", 
            "💳 Billing & Revenue", 
            "📡 Network & Service Desk", 
            "👤 Account & Demographics"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                f_eqpdays = st.number_input(
                    "Equipment Age (Days)", min_value=0, max_value=2500, 
                    value=int(st.session_state.eqpdays), 
                    help="Days with current handset. Inflection boundary is 365 days."
                )
            with col2:
                f_refurb = st.selectbox(
                    "Refurbished Handset?", options=[0, 1], 
                    format_func=lambda x: "Yes (Refurbished)" if x == 1 else "No (Brand New)",
                    index=int(st.session_state.refurb)
                )
                
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                f_mou = st.number_input(
                    "Monthly Minutes of Use (MOU)", min_value=0.0, max_value=10000.0,
                    value=float(st.session_state.mou),
                    help="Total voice minutes in the billing period."
                )
            with col2:
                f_changem = st.number_input(
                    "% Change in Usage (vs 3-Mo Avg)", min_value=-100.0, max_value=500.0,
                    value=float(st.session_state.changem),
                    help="Percentage usage change. Severe drops below -20% signal silent churn."
                )

        with tab3:
            col1, col2, col3 = st.columns(3)
            with col1:
                f_revenue = st.number_input(
                    "Total Monthly Revenue ($)", min_value=0.0, max_value=2000.0,
                    value=float(st.session_state.revenue)
                )
            with col2:
                f_recchrge = st.number_input(
                    "Base Recurring Charge ($)", min_value=0.0, max_value=500.0,
                    value=float(st.session_state.recchrge)
                )
            with col3:
                f_overage = st.number_input(
                    "Overage Fee Charged ($)", min_value=0.0, max_value=1000.0,
                    value=float(st.session_state.overage)
                )

        with tab4:
            col1, col2, col3 = st.columns(3)
            with col1:
                f_dropvce = st.number_input(
                    "Dropped Voice Calls", min_value=0.0, max_value=200.0,
                    value=float(st.session_state.dropvce)
                )
            with col2:
                f_blckvce = st.number_input(
                    "Blocked Voice Calls", min_value=0.0, max_value=200.0,
                    value=float(st.session_state.blckvce)
                )
            with col3:
                f_retcall = st.selectbox(
                    "Retention Call Made?", options=[0, 1],
                    format_func=lambda x: "Yes (Called Retention Desk)" if x == 1 else "No",
                    index=int(st.session_state.retcall)
                )

        with tab5:
            col1, col2, col3 = st.columns(3)
            with col1:
                f_months = st.number_input(
                    "Contract Tenure (Months)", min_value=1, max_value=100,
                    value=int(st.session_state.months)
                )
            with col2:
                f_age1 = st.number_input(
                    "Subscriber Age (Years)", min_value=18, max_value=100,
                    value=int(st.session_state.age1)
                )
            with col3:
                f_creditaa = st.selectbox(
                    "Credit Rating AA (Top Tier)?", options=[0, 1],
                    format_func=lambda x: "Yes (Tier AA - Prime)" if x == 1 else "No (Standard/Subprime)",
                    index=int(st.session_state.creditaa)
                )

        st.caption("ℹ️ *Note: Secondary demographic features are automatically pre-populated with verified training-set medians/modes from the serialized imputer pipeline.*")
        
        # Scoring Configuration
        st.markdown("#### Model Scoring Engine Configuration")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            engine_mode = st.selectbox(
                "Prediction Engine Mode",
                options=[
                    "Decision Mode: Champion Weighted XGBoost (Cost-Sensitive, tau*=0.4631)",
                    "Calibrated Mode: Isotonic Calibrated Probability",
                    "Prior-Adjusted Mode: Bayes Prior-Odds Corrected (1.96% Prevalence)"
                ],
                index=0
            )
        with col_m2:
            threshold_choice = st.slider(
                "Decision Classification Threshold (tau)",
                min_value=0.10, max_value=0.90, value=OPTIMAL_THRESHOLD, step=0.01,
                help=f"Default = {DEFAULT_THRESHOLD:.2f}, F1-Optimal = {OPTIMAL_THRESHOLD:.4f}"
            )

        submit_btn = st.form_submit_button("🔍 Run Churn Prediction & Diagnostic Analysis", type="primary", use_container_width=True)

    # 3. Dynamic Prediction & Explainability Output
    st.markdown("---")
    st.subheader("3. Live Prediction Diagnostics & Action Plan")
    
    if submit_btn:
        # Build raw single-row DataFrame
        raw_input = {
            'eqpdays': f_eqpdays,
            'refurb': f_refurb,
            'mou': f_mou,
            'changem': f_changem,
            'revenue': f_revenue,
            'recchrge': f_recchrge,
            'overage': f_overage,
            'dropvce': f_dropvce,
            'blckvce': f_blckvce,
            'retcall': f_retcall,
            'months': f_months,
            'age1': f_age1,
            'creditaa': f_creditaa,
            'uniqsubs': int(st.session_state.uniqsubs)
        }
        df_single_raw = pd.DataFrame([raw_input])

        # Preprocess through verified pipeline
        try:
            X_proc = prepare_input_dataframe(df_single_raw, artifacts)
            
            # Predict probability based on selected engine mode
            if "Decision Mode" in engine_mode:
                model = artifacts["xgb_model"]
                pred_prob = float(model.predict_proba(X_proc)[0, 1])
                engine_note = f"Score produced by Cost-Sensitive Weighted XGBoost (evaluated against decision boundary tau* = {threshold_choice:.4f})."
            elif "Calibrated Mode" in engine_mode:
                iso_model = artifacts["iso_model"]
                if iso_model is not None:
                    pred_prob = float(iso_model.predict_proba(X_proc)[0, 1])
                else:
                    pred_prob = float(artifacts["xgb_model"].predict_proba(X_proc)[0, 1])
                engine_note = "Score produced by Isotonic Calibrated Classifier (empirically calibrated for combined population)."
            else: # Prior-Adjusted Mode
                raw_prob = float(artifacts["xgb_model"].predict_proba(X_proc)[0, 1])
                pred_prob = apply_bayes_prior_correction(raw_prob)
                engine_note = "Experimental Bayes prior-odds adjusted probability reflecting true 1.96% natural population churn prevalence."

            risk_tier = assign_risk_tier(pred_prob)
            is_churn_flagged = pred_prob >= threshold_choice
            recommendation = get_retention_recommendation(risk_tier)

            # Display KPI Cards
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric(
                    label="Predicted Churn Probability",
                    value=f"{pred_prob:.2%}",
                    delta=f"{'FLAGGED CHURN' if is_churn_flagged else 'RETAINED'} (tau={threshold_choice:.2f})"
                )
            with res_col2:
                st.metric(
                    label="Assigned Risk Tier",
                    value=risk_tier,
                    delta=recommendation["tier_badge"]
                )
            with res_col3:
                st.metric(
                    label="Intervention Budget",
                    value=recommendation["estimated_cost"],
                    delta=recommendation["channel"]
                )

            st.caption(f"ℹ️ *{engine_note}*")

            # Dynamic SHAP Explainability Plot
            st.markdown("---")
            st.markdown("#### Local SHAP Feature Attribution Diagnostics")
            
            with st.spinner("Computing exact Shapley attributions via TreeExplainer..."):
                shap_result = compute_local_shap(artifacts["xgb_model"], X_proc, artifacts)
                fig_shap = render_local_shap_plot(
                    shap_result, X_proc, artifacts, 
                    title=f"Customer Diagnostic SHAP Waterfall (Pred Prob: {pred_prob:.1%})"
                )
                st.pyplot(fig_shap)

            # Drivers Breakdown
            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.markdown("##### 🚨 Top Factors Increasing Churn Risk (+)")
                top_pos = shap_result["top_positive"]
                if len(top_pos) > 0:
                    for f, val in top_pos.items():
                        st.write(f"- **`{f}`**: `{val:+.3f}` log-odds")
                else:
                    st.write("None detected.")

            with col_neg:
                st.markdown("##### 🛡️ Top Factors Reinforcing Loyalty (-)")
                top_neg = shap_result["top_negative"]
                if len(top_neg) > 0:
                    for f, val in top_neg.items():
                        st.write(f"- **`{f}`**: `{val:+.3f}` log-odds")
                else:
                    st.write("None detected.")

            st.warning("⚠️ **Methodological Notice:** SHAP attributions reflect observational statistical associations in the model, not real-world causation. For example, high equipment age (`eqpdays`) correlates with churn because customers with expiring contracts delay purchasing new handsets, rather than the phone physically causing dissatisfaction.")

            # Prescribed Retention Action Card
            st.markdown("---")
            st.markdown("#### Prescribed Operational Retention Action Plan")
            st.success(
                f"""
                ### {recommendation['action_title']}
                - **Primary Outreach Channel:** {recommendation['channel']}
                - **Intervention Unit Cost:** {recommendation['estimated_cost']}
                - **Projected Effectiveness:** {recommendation['expected_uplift']}
                
                **Strategic Business Rationale:**  
                {recommendation['business_rationale']}
                """
            )

        except Exception as e:
            st.error(f"❌ Inference pipeline error: {str(e)}")
    else:
        st.info("👆 Adjust parameters above and click **'Run Churn Prediction & Diagnostic Analysis'** to generate real-time model predictions and SHAP explanations.")
