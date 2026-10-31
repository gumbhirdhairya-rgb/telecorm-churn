"""
app/utils/business_logic.py
Business retention rules, risk tiering, Bayes prior correction, and action plans.
"""

import numpy as np
from app.config import (
    HIGH_RISK_THRESHOLD, 
    MED_RISK_THRESHOLD, 
    NATURAL_PREVALENCE, 
    TRAIN_PREVALENCE
)

def assign_risk_tier(prob: float) -> str:
    """
    Categorizes predicted probability into standardized 3-tier risk categories:
    - High Risk: prob >= 0.60
    - Medium Risk: 0.35 <= prob < 0.60
    - Low Risk: prob < 0.35
    """
    if prob >= HIGH_RISK_THRESHOLD:
        return "High Risk"
    elif prob >= MED_RISK_THRESHOLD:
        return "Medium Risk"
    else:
        return "Low Risk"

def apply_bayes_prior_correction(p_model: float, p_train: float = TRAIN_PREVALENCE, p_nat: float = NATURAL_PREVALENCE) -> float:
    """
    Applies Bayes prior-odds adjustment to correct for case-control oversampling (Step 6.7 audit):
    logit(p_nat) = logit(p_model) + logit(p_nat_prior) - logit(p_train_prior)
    """
    p_clipped = np.clip(p_model, 1e-7, 1.0 - 1e-7)
    logit_model = np.log(p_clipped / (1.0 - p_clipped))
    logit_train = np.log(p_train / (1.0 - p_train))
    logit_nat = np.log(p_nat / (1.0 - p_nat))

    logit_adjusted = logit_model + logit_nat - logit_train
    p_natural = 1.0 / (1.0 + np.exp(-logit_adjusted))
    return float(p_natural)

def get_retention_recommendation(risk_tier: str, top_driver: str = None) -> dict:
    """
    Returns tailored operational retention action plan based on risk tier and primary churn driver.
    """
    if risk_tier == "High Risk":
        return {
            "tier_badge": "🔴 HIGH RISK",
            "tier_color": "#d62728",
            "action_title": "Subsidized 5G Hardware Upgrade Voucher + 12-Month Contract Renewal",
            "channel": "Outbound Senior Retention Concierge Call",
            "estimated_cost": "$35.00",
            "expected_uplift": "25% - 30% Churn Reduction",
            "business_rationale": "Customer exhibits acute disengagement or hardware obsolescence (>365 days). Offering an upgraded 5G device subsidized by carrier while securing a mandatory 1-year contract extension eliminates the hardware driver and legally locks in tenure."
        }
    elif risk_tier == "Medium Risk":
        return {
            "tier_badge": "🟠 MEDIUM RISK",
            "tier_color": "#ff7f0e",
            "action_title": "Automated Plan Right-Sizing & Overage Forgiveness Waiver",
            "channel": "Automated Mobile App Notification & Personalized SMS",
            "estimated_cost": "$4.00",
            "expected_uplift": "15% - 20% Churn Reduction",
            "business_rationale": "Customer is actively using the network (healthy MOU) but experiences unit pricing friction or occasional overage burden. Proactively offering an optimized plan right-sizing and 5GB loyalty bonus preserves the customer relationship at low digital marketing cost."
        }
    else:
        return {
            "tier_badge": "🟢 LOW RISK",
            "tier_color": "#2ca02c",
            "action_title": "Passive Service Assurance & Margin Protection ($0 Spend)",
            "channel": "Automated Background Network Quality-of-Service Monitoring",
            "estimated_cost": "$0.00",
            "expected_uplift": "Baseline Loyalty Maintained",
            "business_rationale": "Customer has fresh equipment, expanding voice usage, and zero complaints. Avoid unsolicited retention discounts which erode ARPU and introduce sleeping-dog churn risk."
        }
