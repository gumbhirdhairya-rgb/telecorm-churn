# Step 6.7 Audit Report: Probability Calibration & Deployment Readiness

**Project:** UCS321 Telecom Customer Churn Prediction (Cell2Cell Dataset)  
**Audit Scope:** Verification of calibration fitting integrity, Cox logistic calibration parameters, Brier Skill Scores against constant-prevalence baselines, Bayes prior-correction formula assumptions, and deployment readiness criteria.

---

### 1. Executive Summary & Core Audit Findings

| Audit Check | Verification Status | Key Evidence / Findings |
|---|:---:|---|
| **1. Leakage-Free Calibration Fitting** | **VERIFIED PASS** | `CalibratedClassifierCV` (Isotonic & Sigmoid) was fitted strictly using **5-Fold internal cross-validation on $X_{\text{train}}$**. The test set ($X_{\text{test}}$, 14,210 samples) was completely excluded from fitting and tuning. |
| **2. Feature Schema & Order Consistency** | **VERIFIED PASS** | Exactly 63 features match between training and test sets in identical columnar sequence. |
| **3. Brier Score vs. Constant Baseline** | **VERIFIED PASS** | **Weighted XGBoost fails the baseline** ($\text{BSS} = -8.28\%$). **Unweighted and Calibrated models demonstrate positive skill** ($\text{BSS} = +8.37\%$ to $+8.61\%$). |
| **4. Cox Calibration Parameters ($\alpha, \beta$)** | **VERIFIED PASS** | Weighted model has severe intercept bias ($\alpha = -0.858$). Isotonic calibrated model achieves near-ideal parameters ($\alpha = 0.004, \beta = 1.004$). |
| **5. Bayes Prior Correction on Natural Cohort** | **VERIFIED PASS** | Reduces Brier score from $0.2193 \to 0.0181$, shifting mean predicted risk from $44.72\% \to 1.84\%$ (matching natural ground truth $1.86\%$). |
| **6. Correction of Unsupported Claims** | **CORRECTED** | The Weighted XGBoost model is **not** suitable for raw probability deployment; it is strictly an outreach ranking tool. Probability-driven financial modeling requires the calibrated or Bayes-adjusted model. |

---

### 2. Verification of Calibration Fitting & Test Set Isolation

To confirm that no data leakage occurred during Step 6.7:
1. **Training Isolation:** The calibration estimators were instantiated as:
   ```python
   calibrated_xgb_iso = CalibratedClassifierCV(
       estimator=XGBClassifier(...),
       method='isotonic',
       cv=5
   )
   calibrated_xgb_iso.fit(X_train, y_train)
   ```
   `CalibratedClassifierCV(cv=5)` splits $X_{\text{train}}$ into 5 internal folds, trains the base XGBoost on 4 folds, generates out-of-fold probability estimates on the held-out fold, and fits the isotonic step-function strictly on out-of-fold probabilities.
2. **Untouched Test Evaluation:** Predictions were evaluated on $X_{\text{test}}$ only after calibration was finalized. Test labels were never provided to any fitting function.

---

### 3. Empirical Calibration Audit: Combined Test Set ($N=14,210$)

In probability verification, a model must be compared against the **Constant-Prevalence Brier Baseline**:
$$\text{Brier}_{\text{baseline}} = \bar{y}(1 - \bar{y})$$
For the combined test set with actual churn prevalence $\bar{y} = 0.2901$:
$$\text{Brier}_{\text{baseline}} = 0.2901 \times (1 - 0.2901) = \mathbf{0.2059}$$

The **Brier Skill Score (BSS)** quantifies percentage improvement over this baseline:
$$\text{BSS} = \left( 1 - \frac{\text{Brier}}{\text{Brier}_{\text{baseline}}} \right) \times 100\%$$
- $\text{BSS} > 0\%$: Genuine probabilistic predictive skill.
- $\text{BSS} \le 0\%$: Inferior to simply guessing the constant base rate for every customer.

#### Cox Logistic Calibration Framework:
To evaluate calibration intercept ($\alpha$) and slope ($\beta$), we fit a logistic regression on the log-odds:
$$\text{logit}(P(Y=1)) = \alpha + \beta \cdot \text{logit}(\hat{p})$$
- **$\alpha$ (Calibration-in-the-large):** Ideal $= 0$. $\alpha < 0$ denotes systematic over-prediction; $\alpha > 0$ denotes under-prediction.
- **$\beta$ (Calibration Slope):** Ideal $= 1$. $\beta < 1$ denotes over-confidence / excessive spread; $\beta > 1$ denotes under-confidence.

#### Combined Test Set Performance Table:

| Model | Mean Predicted $\hat{p}$ | True Prevalence ($\bar{y}$) | Brier Score | Baseline Brier | Brier Skill Score (BSS) | Cox Intercept ($\alpha$) | Cox Slope ($\beta$) | ROC-AUC | PR-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Weighted XGBoost (Champion)** | **47.25%** | 29.01% | 0.2230 | 0.2059 | **-8.28% (Negative Skill)** | **-0.858** | 1.062 | 0.6786 | 0.4586 |
| **Unweighted XGBoost** | **29.01%** | 29.01% | 0.1887 | 0.2059 | **+8.37% (Positive Skill)** | +0.041 | 1.050 | 0.6804 | 0.4578 |
| **Calibrated XGBoost (Isotonic)** | **29.00%** | 29.01% | **0.1882** | 0.2059 | **+8.61% (Best Skill)** | **+0.004** | **1.004** | **0.6816** | **0.4615** |
| **Calibrated XGBoost (Sigmoid)** | **29.01%** | 29.01% | 0.1883 | 0.2059 | **+8.55%** | +0.034 | 1.041 | 0.6818 | 0.4617 |

---

### 4. Empirical Calibration Audit: Natural Cohort (`calibrat == 0`, $N=6,292$)

In the natural holdout cohort, actual churn prevalence is $\bar{y} = 0.018595$ ($1.86\%$).
The constant-prevalence baseline is:
$$\text{Brier}_{\text{baseline}} = 0.0186 \times (1 - 0.0186) = \mathbf{0.01825}$$

#### Natural Cohort Performance Table:

| Model Configuration | Mean Predicted Risk | Actual Prevalence | Brier Score | Baseline Brier | Brier Skill Score (BSS) | Cox Intercept ($\alpha$) | Cox Slope ($\beta$) | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Weighted XGBoost (Raw)** | 44.72% | 1.86% | 0.2193 | 0.0182 | **-1101.72%** | -3.926 | 1.076 | 0.6814 |
| **Unweighted XGBoost (Raw)** | 26.75% | 1.86% | 0.0908 | 0.0182 | **-397.41%** | -3.056 | 1.002 | 0.6729 |
| **Bayes Prior-Adjusted (1.86%)** | **1.84%** | **1.86%** | **0.0181** | 0.0182 | **+0.96% (Positive Skill)** | **-0.221** | **0.940** | **0.6729** |

---

### 5. Mathematical Verification of the Bayes Prior-Correction Formula

#### The Mathematical Formulation:
Given training prevalence $\bar{y}_{\text{train}}$ and target natural prevalence $\pi_{\text{nat}}$, Bayes' rule states:
$$\text{Odds}_{\text{nat}} = \text{Odds}_{\text{train}} \times \left( \frac{\pi_{\text{nat}}}{1 - \pi_{\text{nat}}} \right) \div \left( \frac{\bar{y}_{\text{train}}}{1 - \bar{y}_{\text{train}}} \right)$$

$$\hat{p}_{\text{nat}} = \frac{\text{Odds}_{\text{nat}}}{1 + \text{Odds}_{\text{nat}}}$$

#### Assumptions of this Formula (Viva Key):
1. **Covariate Invariance across Classes ($P(X \mid Y=y)$ is invariant):**  
   The conditional distribution of features given churn status ($Y=1$) and given non-churn ($Y=0$) must be the same in the training sample and the target deployment population. This assumption was empirically verified in Step 6.6 (feature distributions of revenue, mou, and eqpdays are virtually identical between `calibrat=1` and `calibrat=0`).
2. **Accurate Specification of Target Base Rate $\pi_{\text{nat}}$:**  
   The natural population prevalence must be known or accurately estimated from uncalibrated historical accounting logs ($1.86\% - 2.0\%$).
3. **Monotonicity Preservation:**  
   Because the Bayesian adjustment is a strictly monotonic function of the input probability, **ROC-AUC is completely invariant** to this transformation ($0.6729 \to 0.6729$).

---

### 6. Corrected Deployment Readiness Assessment

We correct the generalized "Production Ready" claim from the previous summary:

```
                               OPERATIONAL USE CASES
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     [Case A: Outreach Ranking Queue]               [Case B: Expected-Loss Financial Decisioning]
     - Capacity: "Call top 500 customers"           - Rule: "Offer promo if E[Loss] > $25"
     - Metric needed: ROC-AUC / PR-AUC              - Metric needed: Calibrated Probability (Brier)
     - Champion: Weighted XGBoost                   - Champion: Isotonic Calibrated / Bayes Adjusted
     - STATUS: PRODUCTION READY                     - STATUS: PRODUCTION READY WITH RECALIBRATION ONLY
```

1. **Outreach Ranking / Priority Queues:**  
   **READY.** The Weighted XGBoost model ($\text{ROC-AUC} = 0.6786, \text{PR-AUC} = 0.4586$) reliably identifies the riskiest deciles. Call centers can sort subscribers by score and work from top to bottom.
2. **Automated Financial Decision Systems:**  
   **NOT READY using Raw Weighted Scores.** Applying raw weighted scores will calculate expected churn losses $\sim 2.4\times$ too high, causing over-spending on subsidies.
   **READY using Isotonic Calibrated (Combined $29\%$ cohort) or Bayes-Adjusted Probabilities (Natural $1.86\%$ cohort).**

---

### 7. Saved Artifacts Summary

- **Audit Report:** [`reports/06_calibration_audit.md`](file:///d:/mst%20project/reports/06_calibration_audit.md)
- **Combined Calibration CSV:** [`reports/audit_combined_calibration.csv`](file:///d:/mst%20project/reports/audit_combined_calibration.csv)
- **Natural Cohort Calibration CSV:** [`reports/audit_natural_cohort_calibration.csv`](file:///d:/mst%20project/reports/audit_natural_cohort_calibration.csv)
- **Calibration Curves & Histograms:** [`reports/figures/12_calibration_curves.png`](file:///d:/mst%20project/reports/figures/12_calibration_curves.png)
- **Serialized Calibrated Models:** [`models/calibrated_xgboost_isotonic.joblib`](file:///d:/mst%20project/models/calibrated_xgboost_isotonic.joblib), [`models/calibrated_xgboost_sigmoid.joblib`](file:///d:/mst%20project/models/calibrated_xgboost_sigmoid.joblib), [`models/unweighted_xgboost.joblib`](file:///d:/mst%20project/models/unweighted_xgboost.joblib)
