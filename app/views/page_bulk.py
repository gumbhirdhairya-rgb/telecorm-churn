"""
app/views/page_bulk.py
Primary Mode 2: Bulk Customer Analysis UI
Provides live batch inference on raw or preprocessed CSV uploads, schema validation,
risk tier breakdown, high-risk customer worklists, and enriched CSV export.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from app.config import DATA_DIR, OPTIMAL_THRESHOLD, DEFAULT_THRESHOLD
from app.utils.model_loader import load_models_and_preprocessor
from app.utils.preprocessor import prepare_input_dataframe
from app.utils.business_logic import (
    assign_risk_tier, 
    apply_bayes_prior_correction, 
    get_retention_recommendation
)

def render_bulk_analysis():
    st.title("Bulk Customer Portfolio Analysis")
    st.markdown("**High-throughput batch scoring, automated risk tier segmentation, and priority worklist generation.**")
    
    # Load serialized artifacts
    try:
        artifacts = load_models_and_preprocessor()
    except Exception as e:
        st.error(f"❌ Failed to load model artifacts: {str(e)}")
        return

    st.markdown("---")
    
    # 1. Ingestion & Download Sample Section
    st.subheader("1. Ingest Customer Data File")
    
    col_up, col_dl = st.columns([1.8, 1.2])
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Customer Records CSV", 
            type=["csv"],
            help="Accepts either raw customer attributes (cell2cell format) or 63-column preprocessed features."
        )
    with col_dl:
        st.markdown("#### Test Sample Files")
        st.caption("Download sample test batches to test the ingestion pipeline:")
        
        sample_raw_path = DATA_DIR / "sample_raw_customers.csv"
        if sample_raw_path.exists():
            with open(sample_raw_path, "rb") as f:
                st.download_button(
                    label="📥 Download Raw Customers Sample (50 rows)",
                    data=f.read(),
                    file_name="sample_raw_customers.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        test_file_path = DATA_DIR / "processed" / "X_test_final.csv"
        if test_file_path.exists():
            with open(test_file_path, "rb") as f:
                st.download_button(
                    label="📥 Download Preprocessed Sample (X_test_final)",
                    data=f.read(),
                    file_name="sample_preprocessed_customers.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # 2. Schema Validation & Input Preview
    if uploaded_file is not None:
        st.markdown("---")
        st.subheader("2. File Inspection & Validation")
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f" File parsed successfully: **{len(df_upload):,} rows** x **{len(df_upload.columns)} columns**.")
            
            # Validation checks
            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1:
                st.metric("Total Rows Ingested", f"{len(df_upload):,}")
            with col_v2:
                # Target leakage detection
                target_cols = [c for c in ['churn', 'churndep', 'customer', 'calibrat'] if c in df_upload.columns]
                if target_cols:
                    st.warning(f"⚠️ Target/ID columns detected: `{target_cols}` (Will be stripped to prevent data leakage).")
                else:
                    st.info(" Zero target leakage detected.")
            with col_v3:
                missing_vals = df_upload.isna().sum().sum()
                st.metric("Missing Cell Values", f"{missing_vals:,}", delta="Handled by Imputer Pipeline")

            st.markdown("#### Input Data Preview (First 5 Rows)")
            st.dataframe(df_upload.head(5), use_container_width=True)

            # 3. Batch Scoring Configuration
            st.markdown("---")
            st.subheader("3. Batch Scoring Configuration & Execution")
            
            col_cfg1, col_cfg2 = st.columns(2)
            with col_cfg1:
                bulk_engine_mode = st.selectbox(
                    "Prediction Engine Mode",
                    options=[
                        "Decision Mode: Champion Weighted XGBoost (Cost-Sensitive)",
                        "Calibrated Mode: Isotonic Calibrated Probability",
                        "Prior-Adjusted Mode: Bayes Prior-Odds Corrected (1.96% Prevalence)"
                    ],
                    index=0
                )
            with col_cfg2:
                bulk_threshold = st.slider(
                    "Decision Classification Threshold (tau)",
                    min_value=0.10, max_value=0.90, value=OPTIMAL_THRESHOLD, step=0.01,
                    help=f"Default = {DEFAULT_THRESHOLD:.2f}, F1-Optimal = {OPTIMAL_THRESHOLD:.4f}"
                )

            run_batch_btn = st.button("🚀 Execute Batch Churn Scoring Pipeline", type="primary", use_container_width=True)
            
            # 4. Batch Prediction Execution
            if run_batch_btn:
                with st.spinner(f"Processing and scoring {len(df_upload):,} customer records..."):
                    try:
                        # Preprocess input DataFrame
                        X_proc = prepare_input_dataframe(df_upload, artifacts)

                        # Generate probabilities
                        if "Decision Mode" in bulk_engine_mode:
                            probs = artifacts["xgb_model"].predict_proba(X_proc)[:, 1]
                        elif "Calibrated Mode" in bulk_engine_mode:
                            if artifacts["iso_model"] is not None:
                                probs = artifacts["iso_model"].predict_proba(X_proc)[:, 1]
                            else:
                                probs = artifacts["xgb_model"].predict_proba(X_proc)[:, 1]
                        else: # Prior-Adjusted Mode
                            raw_probs = artifacts["xgb_model"].predict_proba(X_proc)[:, 1]
                            probs = np.array([apply_bayes_prior_correction(p) for p in raw_probs])

                        # Build enriched results DataFrame
                        df_results = df_upload.copy()
                        df_results["Predicted_Churn_Prob"] = np.round(probs, 4)
                        df_results["Assigned_Risk_Tier"] = [assign_risk_tier(p) for p in probs]
                        df_results["Churn_Flag"] = (probs >= bulk_threshold).astype(int)

                        # Map recommended actions
                        actions = []
                        costs = []
                        channels = []
                        for tier in df_results["Assigned_Risk_Tier"]:
                            rec = get_retention_recommendation(tier)
                            actions.append(rec["action_title"])
                            costs.append(rec["estimated_cost"])
                            channels.append(rec["channel"])

                        df_results["Recommended_Action"] = actions
                        df_results["Intervention_Cost"] = costs
                        df_results["Outreach_Channel"] = channels

                        # 5. Display KPI Summary
                        st.markdown("---")
                        st.subheader("4. Portfolio Scoring Summary & Risk Distribution")
                        
                        total_scored = len(df_results)
                        high_risk_count = int(np.sum(df_results["Assigned_Risk_Tier"] == "High Risk"))
                        med_risk_count = int(np.sum(df_results["Assigned_Risk_Tier"] == "Medium Risk"))
                        low_risk_count = int(np.sum(df_results["Assigned_Risk_Tier"] == "Low Risk"))
                        flagged_churn_count = int(np.sum(df_results["Churn_Flag"] == 1))
                        avg_prob = float(np.mean(probs))

                        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                        with kpi1:
                            st.metric("Total Customers Scored", f"{total_scored:,}")
                        with kpi2:
                            st.metric("Flagged as Churners", f"{flagged_churn_count:,}", delta=f"{flagged_churn_count/total_scored:.1%} of Base")
                        with kpi3:
                            st.metric("High-Risk Cohort", f"{high_risk_count:,}", delta=f"{high_risk_count/total_scored:.1%} (p >= 0.60)")
                        with kpi4:
                            st.metric("Portfolio Churn Density", f"{avg_prob:.2%}", delta="Mean Probability")

                        # Risk Distribution Visualization
                        col_chart, col_stats = st.columns([1.2, 1.0])
                        with col_chart:
                            tier_counts = pd.Series({
                                "High Risk (p >= 0.60)": high_risk_count,
                                "Medium Risk (0.35 <= p < 0.60)": med_risk_count,
                                "Low Risk (p < 0.35)": low_risk_count
                            })
                            fig, ax = plt.subplots(figsize=(6, 3.5))
                            bars = ax.bar(tier_counts.index, tier_counts.values, color=['#d62728', '#ff7f0e', '#2ca02c'], edgecolor='black', alpha=0.85)
                            ax.set_ylabel("Customer Count", fontsize=9, fontweight='bold')
                            ax.set_title("Portfolio Risk Tier Distribution", fontsize=10, fontweight='bold')
                            for b in bars:
                                h = b.get_height()
                                ax.text(b.get_x() + b.get_width()/2, h + max(1, h*0.02), f"{h:,} ({h/total_scored:.1%})", ha='center', va='bottom', fontsize=8, fontweight='bold')
                            plt.xticks(rotation=15, ha='right', fontsize=8)
                            plt.tight_layout()
                            st.pyplot(fig)

                        with col_stats:
                            st.markdown("#### Operational Breakdown")
                            st.write(f"- **High Risk Priority:** `{high_risk_count:,}` accounts require senior concierge hardware renewal.")
                            st.write(f"- **Medium Risk Optimization:** `{med_risk_count:,}` accounts eligible for automated digital plan right-sizing.")
                            st.write(f"- **Low Risk Satisfied:** `{low_risk_count:,}` accounts require zero retention spend ($0 cost).")
                            
                            est_total_budget = high_risk_count * 35.0 + med_risk_count * 4.0
                            st.metric("Total Estimated Retention Budget", f"${est_total_budget:,.2f}", delta="Tiered Strategy")

                        # 6. Filterable High-Risk Priority Worklist
                        st.markdown("---")
                        st.subheader("5. Operational Priority Worklist (High Risk Customers)")
                        
                        filter_tier = st.selectbox(
                            "Filter Worklist by Risk Tier:",
                            options=["All Tiers", "High Risk", "Medium Risk", "Low Risk"],
                            index=1
                        )
                        
                        if filter_tier != "All Tiers":
                            df_display = df_results[df_results["Assigned_Risk_Tier"] == filter_tier]
                        else:
                            df_display = df_results

                        st.dataframe(
                            df_display.sort_values("Predicted_Churn_Prob", ascending=False),
                            use_container_width=True
                        )

                        # 7. CSV Export Button
                        st.markdown("---")
                        st.download_button(
                            label=f"📥 Download Scored Customer Portfolio CSV ({len(df_results):,} rows)",
                            data=df_results.to_csv(index=False),
                            file_name="telecom_churn_scored_portfolio.csv",
                            mime="text/csv",
                            type="primary",
                            use_container_width=True
                        )

                    except Exception as e:
                        st.error(f"❌ Batch scoring failed: {str(e)}")

        except Exception as e:
            st.error(f"❌ Error reading uploaded CSV: {str(e)}")
    else:
        st.info("👆 Please upload a customer CSV file above or download a sample test batch to begin batch analysis.")
