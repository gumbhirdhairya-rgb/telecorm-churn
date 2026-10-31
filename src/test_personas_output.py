"""
src/test_personas_output.py
Executes live inference on High, Medium, and Low risk personas
and displays the exact outputs shown in the Streamlit application.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from app.utils.model_loader import load_models_and_preprocessor
from app.utils.preprocessor import prepare_input_dataframe
from app.utils.explainer import compute_local_shap
from app.utils.business_logic import assign_risk_tier, apply_bayes_prior_correction, get_retention_recommendation

artifacts = load_models_and_preprocessor()

personas = [
    ("High Risk Persona (#606)", {
        "eqpdays": 374, "refurb": 1, "mou": 0.0, "changem": -60.0, "revenue": 56.0,
        "recchrge": 50.0, "overage": 0.0, "dropvce": 0.0, "blckvce": 0.0, "retcall": 1,
        "months": 18, "age1": 30, "creditaa": 0, "uniqsubs": 2
    }),
    ("Medium (Borderline) Persona (#13293)", {
        "eqpdays": 382, "refurb": 0, "mou": 530.0, "changem": 27.0, "revenue": 59.0,
        "recchrge": 46.0, "overage": 45.0, "dropvce": 1.0, "blckvce": 0.0, "retcall": 0,
        "months": 13, "age1": 42, "creditaa": 1, "uniqsubs": 2
    }),
    ("Low Risk (Loyal) Persona (#452)", {
        "eqpdays": 69, "refurb": 0, "mou": 650.0, "changem": 18.0, "revenue": 62.0,
        "recchrge": 35.0, "overage": 0.0, "dropvce": 0.0, "blckvce": 0.0, "retcall": 0,
        "months": 59, "age1": 56, "creditaa": 1, "uniqsubs": 1
    })
]

for name, data in personas:
    df_raw = pd.DataFrame([data])
    X_proc = prepare_input_dataframe(df_raw, artifacts)
    
    # 1. Decision Mode (Weighted XGBoost)
    p_decision = float(artifacts["xgb_model"].predict_proba(X_proc)[0, 1])
    # 2. Calibrated Mode (Isotonic)
    p_calibrated = float(artifacts["iso_model"].predict_proba(X_proc)[0, 1])
    # 3. Prior-adjusted Mode (Bayes 1.96%)
    p_prior = apply_bayes_prior_correction(p_decision)
    
    tier = assign_risk_tier(p_decision)
    rec = get_retention_recommendation(tier)
    shap_res = compute_local_shap(artifacts["xgb_model"], X_proc, artifacts)
    
    pos_drivers = [(k, round(float(v), 3)) for k, v in list(shap_res["top_positive"].items())[:3]]
    neg_drivers = [(k, round(float(v), 3)) for k, v in list(shap_res["top_negative"].items())[:3]]
    
    print(f"=== {name} ===")
    print(f"  - Decision Mode Churn Prob (tau*=0.4631): {p_decision:.2%}")
    print(f"  - Calibrated Mode Churn Prob:            {p_calibrated:.2%}")
    print(f"  - Prior-Adjusted Mode Churn Prob:        {p_prior:.2%}")
    print(f"  - Assigned Risk Category:                 {tier} ({rec['tier_badge']})")
    print(f"  - Top 3 Churn Pushers (+):                {pos_drivers}")
    print(f"  - Top 3 Loyalty Pullers (-):              {neg_drivers}")
    print(f"  - Prescribed Action:                      {rec['action_title']}")
    print(f"  - Outreach Channel & Est. Cost:           {rec['channel']} ({rec['estimated_cost']})")
    print()
