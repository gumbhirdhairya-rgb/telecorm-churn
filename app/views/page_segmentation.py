"""
app/views/page_segmentation.py
Shared Page: Customer Risk Segmentation
Displays empirical 3-tier customer risk segmentation, unscaled business profiles,
and operational metrics from Step 8.
"""

import streamlit as st
import pandas as pd
from app.config import PATH_CSV_RISK_SEGMENTS, FIG_SEGMENTATION

def render_risk_segmentation():
    st.title("Customer Risk Segmentation & Empirical Profiles")
    st.markdown("**Evidence-based portfolio tiering across the 14,210-customer untouched test partition.**")
    
    st.markdown("---")
    
    # 1. Summary Cards for 3 Risk Tiers
    st.subheader("1. Empirical Risk Tier Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.error(
            r"""
            ### 🔴 High Risk Tier
            **Cutoff Rule:** $\hat{p} \ge 0.60$  
            - **Customer Share:** 2,814 (19.8%)  
            - **Observed Churn Rate:** **48.51%**  
            - **Avg Handset Age:** **481.4 days**  
            - **Avg Usage Contraction:** **-55.10%**  
            - **Retention Calls:** **10.95%**  
            *Action:* Subsidized 5G hardware upgrade + 12-mo contract renewal.
            """
        )
    with col2:
        st.warning(
            r"""
            ### 🟠 Medium (Borderline) Tier
            **Cutoff Rule:** $0.35 \le \hat{p} < 0.60$  
            - **Customer Share:** 8,184 (57.6%)  
            - **Observed Churn Rate:** **28.69%**  
            - **Avg Handset Age:** **414.8 days**  
            - **Avg Usage Contraction:** **-4.32%**  
            - **Retention Calls:** **1.91%**  
            *Action:* Automated digital plan right-sizing + 5GB loyalty bonus.
            """
        )
    with col3:
        st.success(
            r"""
            ### 🟢 Low Risk (Loyal) Tier
            **Cutoff Rule:** $\hat{p} < 0.35$  
            - **Customer Share:** 3,212 (22.6%)  
            - **Observed Churn Rate:** **12.73%**  
            - **Avg Handset Age:** **201.7 days**  
            - **Avg Usage Growth:** **+18.37%**  
            - **Retention Calls:** **0.56%**  
            *Action:* Zero marketing spend; preserve baseline profit margins.
            """
        )

    st.markdown("---")
    
    # 2. Risk Profiles Visual Comparison
    st.subheader("2. Segmentation Visual Profiles")
    if FIG_SEGMENTATION.exists():
        st.image(str(FIG_SEGMENTATION), caption="Figure 17: Empirical Risk Segmentation Metrics and Operational Indicators", use_container_width=True)
    else:
        st.warning("Segmentation figure not found.")

    st.markdown("---")
    
    # 3. Empirical Test Data Table
    st.subheader("3. Unscaled Attribute Profiles by Risk Tier")
    st.caption("All quantities computed from physical business units (dollars, minutes, days, and calls):")
    
    if PATH_CSV_RISK_SEGMENTS.exists():
        df_seg = pd.read_csv(PATH_CSV_RISK_SEGMENTS)
        display_df = df_seg.copy()
        
        # Format columns for display
        format_dict = {
            'customer_count': '{:,}',
            'pct_of_customer_base': '{:.1f}%',
            'actual_churn_count': '{:,}',
            'actual_churn_rate': '{:.2%}',
            'avg_predicted_prob': '{:.2%}',
            'avg_monthly_revenue': '${:.2f}',
            'avg_monthly_mou': '{:.1f} min',
            'avg_equipment_days': '{:.1f} days',
            'avg_tenure_months': '{:.1f} mos',
            'avg_usage_change_pct': '{:+.1f}%',
            'avg_cost_per_mou': '${:.2f}/min',
            'avg_network_friction': '{:.2%}',
            'retention_call_rate': '{:.2%}'
        }
        for col, fmt in format_dict.items():
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: fmt.format(x) if pd.notnull(x) else "")
                
        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("Risk segmentation CSV not found.")
