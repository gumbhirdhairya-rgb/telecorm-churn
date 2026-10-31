"""
app/views/page_strategy.py
Shared Page: Business Strategy & Interactive ROI Simulator
Provides policy comparison tables, cost-benefit trade-off curves, interactive ROI simulation sliders,
operational workflow architecture, and risk governance guidelines.
"""

import streamlit as st
import pandas as pd
import numpy as np
from app.config import (
    PATH_CSV_POLICIES, 
    FIG_COST_BENEFIT, 
    FIG_WORKFLOW
)

def render_business_strategy():
    st.title("Business Insights & Customer Retention Strategy")
    st.markdown("**Evidence-based decision science, budget prioritization, and interactive ROI simulation.**")
    
    st.markdown("---")
    
    # 1. Five Strategic Policies Comparison
    st.subheader("1. Retention Strategy Policy Benchmark")
    st.caption("Evaluated across the 14,210 test customers under baseline assumptions (LTV=$250, Uplift=25%, Cost=$30):")
    
    if PATH_CSV_POLICIES.exists():
        df_pol = pd.read_csv(PATH_CSV_POLICIES)
        display_pol = df_pol.copy()
        
        display_pol['n_targeted'] = display_pol['n_targeted'].apply(lambda x: f"{int(x):,}")
        display_pol['churners_saved'] = display_pol['churners_saved'].apply(lambda x: f"{x:.1f}")
        display_pol['cost'] = display_pol['cost'].apply(lambda x: f"${int(x):,}")
        display_pol['gross_benefit'] = display_pol['gross_benefit'].apply(lambda x: f"${int(x):,}")
        display_pol['net_savings'] = display_pol['net_savings'].apply(lambda x: f"${int(x):,}")
        display_pol['roi_pct'] = display_pol['roi_pct'].apply(lambda x: f"{x:+.1f}%")
        
        st.dataframe(display_pol[['policy_name', 'n_targeted', 'churners_saved', 'cost', 'gross_benefit', 'net_savings', 'roi_pct']], use_container_width=True)
    else:
        st.warning("Policy comparison CSV not found.")

    st.markdown("---")
    
    # 2. Cost-Benefit Tradeoff Curve
    st.subheader("2. Threshold Optimization & Tradeoff Curve")
    col_cb_img, col_cb_text = st.columns([1.2, 1.0])
    
    with col_cb_img:
        if FIG_COST_BENEFIT.exists():
            st.image(str(FIG_COST_BENEFIT), caption="Figure 18: Retention Campaign Cost-Benefit Tradeoff Curve", use_container_width=True)
        else:
            st.warning("Cost-benefit figure not found.")
            
    with col_cb_text:
        st.info(
            r"""
            **Key Economic Takeaways:**
            - **Blanket Campaign Destruction:** Contacting all 14,210 customers uniformly costs $\$426,300$ and destroys **$-\$168,675$ in capital (-$39.6\%$ ROI)** due to $\$302,640$ wasted on $10,088$ loyal non-churners.
            - **Peak Single Cutoff ($\tau = 0.67$):** Concentrating on ultra-high risk ($1,174$ subscribers) achieves **+$20.7\%$ ROI** (Net: $+\$7,280$).
            - **Differentiated Tiered Strategy (Policy 5):** Deploying high-touch $\$35$ hardware vouchers for High Risk paired with $\$4$ automated digital plan right-sizing for Medium Risk delivers **+$59,199$ in net profit** with a **+45.1% ROI**.
            """
        )

    st.markdown("---")
    
    # 3. Interactive ROI & Budget Simulator
    st.subheader("3. Interactive Retention ROI Simulator")
    st.caption("Adjust the business parameters below to simulate campaign economics dynamically across the 14,210 test customers:")
    
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        v_ltv = st.slider("Preserved Annual LTV / Gross Margin ($)", min_value=100.0, max_value=600.0, value=250.0, step=25.0)
    with sim_col2:
        r_uplift = st.slider("Retention Uplift / Success Rate (%)", min_value=5.0, max_value=50.0, value=25.0, step=1.0) / 100.0
    with sim_col3:
        c_cost = st.slider("Uniform Campaign Cost per Customer ($)", min_value=5.0, max_value=100.0, value=30.0, step=5.0)

    # Dynamic calculation across test set
    # Using real test set counts: Total=14,210, Churners=4,122
    # At High Risk (tau >= 0.60): N=2,814, TP=1,365, FP=1,449
    # At Optimal (tau >= 0.4631): N=7,834, TP=3,000, FP=4,834
    # At Blanket (tau >= 0.00): N=14,210, TP=4,122, FP=10,088
    
    calc_high_cost = 2814 * c_cost
    calc_high_saved = 1365 * r_uplift
    calc_high_ben = calc_high_saved * v_ltv
    calc_high_net = calc_high_ben - calc_high_cost
    calc_high_roi = (calc_high_net / calc_high_cost * 100) if calc_high_cost > 0 else 0
    
    calc_opt_cost = 7834 * c_cost
    calc_opt_saved = 3000 * r_uplift
    calc_opt_ben = calc_opt_saved * v_ltv
    calc_opt_net = calc_opt_ben - calc_opt_cost
    calc_opt_roi = (calc_opt_net / calc_opt_cost * 100) if calc_opt_cost > 0 else 0
    
    calc_all_cost = 14210 * c_cost
    calc_all_saved = 4122 * r_uplift
    calc_all_ben = calc_all_saved * v_ltv
    calc_all_net = calc_all_ben - calc_all_cost
    calc_all_roi = (calc_all_net / calc_all_cost * 100) if calc_all_cost > 0 else 0

    st.markdown("#### Real-Time Simulation Results on Test Base (14,210 Subscribers):")
    res1, res2, res3 = st.columns(3)
    
    with res1:
        st.markdown("**Blind Blanket Campaign (All)**")
        st.metric("Total Cost", f"${int(calc_all_cost):,}")
        st.metric("Saved Churners", f"{calc_all_saved:.1f}")
        st.metric("Net Savings", f"${int(calc_all_net):,}", delta=f"{calc_all_roi:+.1f}% ROI")
        
    with res2:
        st.markdown("**High Risk Only (tau >= 0.60)**")
        st.metric("Total Cost", f"${int(calc_high_cost):,}")
        st.metric("Saved Churners", f"{calc_high_saved:.1f}")
        st.metric("Net Savings", f"${int(calc_high_net):,}", delta=f"{calc_high_roi:+.1f}% ROI")

    with res3:
        st.markdown("**F1-Optimal Cutoff (tau = 0.46)**")
        st.metric("Total Cost", f"${int(calc_opt_cost):,}")
        st.metric("Saved Churners", f"{calc_opt_saved:.1f}")
        st.metric("Net Savings", f"${int(calc_opt_net):,}", delta=f"{calc_opt_roi:+.1f}% ROI")

    st.markdown("---")
    
    # 4. End-to-End Operational Workflow Diagram
    st.subheader("4. Operational Retention Architecture")
    if FIG_WORKFLOW.exists():
        st.image(str(FIG_WORKFLOW), caption="Figure 19: End-to-End Operational Closed-Loop Retention Pipeline Architecture", use_container_width=True)
    else:
        st.warning("Operational workflow figure not found.")

    st.markdown("---")
    
    # 5. Methodological Limitations & Governance
    st.subheader("5. Risk Governance & Practical Limitations")
    st.warning(
        r"""
        **1. Correlation vs. Causation:** SHAP values represent statistical attribution in observational data, not causal 
        guarantees. Retention interventions must address customer motivation (e.g. offering subsidized upgrades to secure 
        contract extensions) rather than simply manipulating proxy variables like `eqpdays`.
        
        **2. False Positives vs. False Negatives:** Spending retention money on loyal non-churners directly erodes ARPU. 
        Costly human interventions must be reserved for high-probability cohorts ($\hat{p} \ge 0.60$).
        
        **3. Probability Calibration in Production:** The training cohort is case-control oversampled (29.0% churn) compared 
        to the natural cohort (1.96% churn). Production outlays must apply Bayes prior-odds correction before committing capital.
        
        **4. Mandatory Holdout Control Groups:** A randomized 10% holdout control group must always be preserved during live 
        campaigns to measure true causal incremental lift:  
        $$\text{Incremental Lift} = \text{Churn Rate}_{\text{Control}} - \text{Churn Rate}_{\text{Treated}}$$
        """
    )
