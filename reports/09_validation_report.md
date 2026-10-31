# Step 9C: End-to-End Frontend and Model Validation Report
**Project Title:** Telecom Customer Churn Prediction & Decision Science Platform  
**Evaluated Application:** Streamlit Frontend (`app/app.py` on `http://localhost:8501`)  
**Evaluated Artifacts:** `models/best_xgboost.joblib`, `models/final_preprocessor.joblib`, `models/calibrated_xgboost_isotonic.joblib`  
**Test Suite:** `src/run_step9c_validation.py`  
**Evaluation Status:** **100% PASS (9 out of 9 Validation Tasks Passed)**

---

## 1. Executive Summary & Validation Matrix

This report provides the rigorous end-to-end technical validation of the Streamlit application and its underlying machine learning pipelines, ensuring complete numerical consistency with historical training and evaluation notebooks (Steps 1–8).

```
========================================================================================================================
                                     STEP 9C VALIDATION SUMMARY TABLE
========================================================================================================================
Task ID   Validation Component                       Status   Key Verified Findings
------------------------------------------------------------------------------------------------------------------------
Task 1    Model Performance Page Metrics             PASS     Exact match: ROC-AUC=0.6786, PR-AUC=0.4586, F1=0.4947
Task 2    Confusion Matrices (0.50 vs 0.4631)        PASS     tau=0.50: [6216, 3872, 1495, 2627], tau*=0.4631: [5256, 4832, 1122, 3000]
Task 3    Individual vs Bulk Consistency             PASS     Identical probabilities across pipelines (Max diff = 0.0000e+00)
Task 4    Three Customer Personas Evaluation         PASS     High (#606): 85.36%, Med (#13293): 47.69%, Low (#452): 10.92%
Task 5    50-Row Raw CSV Batch & Download Export     PASS     Processed 50 rows (66 cols -> 63 features), zero NaNs, valid export
Task 6    Calibrated & Prior-Adjusted Modes          PASS     Isotonic verified; Bayes prior baseline (29.01% -> 1.962%) verified
Task 7    SHAP Additivity & Retention Playbook       PASS     Implied margin sigmoid reconstructed model prob to 10^-4 exactness
Task 8    Robustness & Edge-Case Guardrails          PASS     Passed: NaN imputer fallback, auto-strip target leakage, dropblk recovery
Task 9    Model Artifact Checksum Integrity          PASS     Zero artifacts modified on disk (MD5 and byte sizes 100% preserved)
========================================================================================================================
```

---

## 2. In-Depth Task-by-Task Verification

### Task 1: Model Performance Page Metrics Verification
The performance metrics rendered on the **Model Performance** page were verified against [`reports/final_test_evaluation.csv`](file:///d:/mst%20project/reports/final_test_evaluation.csv):

```
+--------------------------------------------------------------------------------------------------+
| Metric                 Reported in CSV    Streamlit Display    Verification Status    Difference |
+--------------------------------------------------------------------------------------------------+
| Test ROC-AUC               0.6786              0.6786                PASS               0.0000   |
| Test PR-AUC                0.4586              0.4586                PASS               0.0000   |
| Overall Accuracy           0.6223              0.6223                PASS               0.0000   |
| Precision (at 0.50)        0.4042              0.4042                PASS               0.0000   |
| Recall (at 0.50)           0.6373              0.6373                PASS               0.0000   |
| F1 Score (at 0.50)         0.4947              0.4947                PASS               0.0000   |
+--------------------------------------------------------------------------------------------------+
```
*Conclusion:* Zero discrepancy between UI tables and serialized evaluation metrics.

---

### Task 2: Confusion Matrices (0.50 vs 0.4631)
We verified the exact confusion matrix counts for both default and cost-sensitive thresholds across the **14,210 test customers**:

```
Table 2: Confusion Matrix Verification
-------------------------------------------------------------------------------------------------------
Threshold      True Negatives (TN)   False Positives (FP)   False Negatives (FN)   True Positives (TP)
-------------------------------------------------------------------------------------------------------
tau = 0.5000         6,216                  3,872                  1,495                  2,627
tau* = 0.4631        5,256                  4,832                  1,122                  3,000
-------------------------------------------------------------------------------------------------------
Net Delta            -960                   +960                   -373                   +373
-------------------------------------------------------------------------------------------------------
```
*Key Finding:* Lowering the threshold to $\tau^* = 0.4631$ captures **$+373$ additional true churners** (reducing False Negatives from $1,495 \to 1,122$, and increasing Recall from $63.73\% \to 72.78\%$). Sum of all 4 cells equals $14,210$ exactly.

---

### Task 3: Individual vs Bulk Prediction Consistency
To guarantee pipeline idempotency and prevent logic divergence between form inputs and file uploads, 5 identical customer profiles were passed through both pipelines:
- **Individual Pipeline:** Form dictionary $\to$ `prepare_input_dataframe()` $\to$ `xgb_model.predict_proba()`
- **Bulk Pipeline:** DataFrame upload $\to$ `prepare_input_dataframe()` $\to$ `xgb_model.predict_proba()`
- **Max Absolute Difference:** $\mathbf{0.00000000 \times 10^{-6}}$ (Strict mathematical equality).

---

### Task 4: Three Customer Personas Evaluation
The three representative customer personas were scored through the live application engine:

```
Table 3: Live Persona Scoring Output
-------------------------------------------------------------------------------------------------------------
Persona Name           Decision Mode (tau*=0.46)   Calibrated Mode   Prior-Adjusted Mode (1.96%)   Risk Tier
-------------------------------------------------------------------------------------------------------------
High Risk (#606)                85.36%                 89.62%                  22.21%              High Risk
Medium Risk (#13293)            47.69%                 27.43%                   4.27%              Medium Risk
Low Risk (#452)                 10.92%                  2.89%                   0.60%              Low Risk
-------------------------------------------------------------------------------------------------------------
```
*Key Diagnostic Insights:*
- **Persona #606:** Handset age $374$ days, $0.0$ MOU, $-60\%$ usage drop $\to$ Decision prob $85.36\%$ (High Risk).
- **Persona #13293:** Handset age $382$ days, $\$45$ overage fee $\to$ Decision prob $47.69\%$. Since $47.69\% > 46.31\%$, it correctly triggers a retention intervention.
- **Persona #452:** Handset age $69$ days, $59$ months tenure $\to$ Decision prob $10.92\%$ (Low Risk).

---

### Task 5: 50-Row Raw CSV Batch & Download Export
- Ingested [`data/sample_raw_customers.csv`](file:///d:/mst%20project/data/sample_raw_customers.csv) containing 50 raw unscaled customer records (66 raw columns).
- Pipeline successfully executed:
  1. Identified raw schema and generated the 5 interaction features.
  2. Aligned with the 63 final features in exact order.
  3. Applied `final_preprocessor.transform()`.
  4. Scored all 50 records in **0.04 seconds**.
  5. Mean predicted churn probability: **$38.44\%$**; High Risk count: **5 accounts**.
  6. Verified that the exported CSV contains all original features plus `Predicted_Churn_Prob`, `Assigned_Risk_Tier`, `Churn_Flag`, `Recommended_Action`, `Intervention_Cost`, and `Outreach_Channel`.

---

### Task 6: Calibrated & Bayes Prior-Adjusted Modes
- **Decision Mode ($\tau^* = 0.4631$):** Uses class-weighted XGBoost (`scale_pos_weight=2.447`) to maximize separation for cost-sensitive ranking.
- **Calibrated Mode:** Uses `calibrated_xgboost_isotonic.joblib` to produce empirical probabilities reflecting the combined test distribution.
- **Prior-Adjusted Mode:** Implements the Bayes prior-odds formula:
  $$\text{logit}(p_{\text{nat}}) = \text{logit}(p_{\text{model}}) + \ln\left(\frac{0.01962}{1 - 0.01962}\right) - \ln\left(\frac{0.29014}{1 - 0.29014}\right) = \text{logit}(p_{\text{model}}) - 2.8250$$
  - Baseline conversion test: An input of $29.014\%$ converts to **$1.9620\%$** exactly.
  - Persona #606 ($85.36\%$) converts to **$22.21\%$**, accurately reflecting true risk density in a low-prevalence population.

---

### Task 7: SHAP Additivity & Retention Playbook
- **TreeExplainer Additivity Verification:**
  $$E[f(X)] = +0.00003 \text{ log-odds}$$
  $$\sum_{j=1}^{63} \phi_j(x) = +1.7632 \text{ log-odds}$$
  $$\text{Margin } f(x) = +1.76323 \implies \hat{p} = \frac{1}{1 + e^{-1.76323}} = 85.36\% \equiv \text{model.predict\_proba}(x)$$
- Primary positive drivers: `cost_per_mou` ($+0.508$), `mou` ($+0.503$), `retcall` ($+0.457$).
- Recommendation mappings verified:
  - **High Risk:** Subsidized 5G Hardware Voucher + 1-Yr Contract Renewal ($\$35.00$).
  - **Medium Risk:** Automated Plan Right-Sizing & Overage Waiver ($\$4.00$).
  - **Low Risk:** Passive Quality Assurance ($\$0.00$).

---

### Task 8: Robustness & Edge-Case Guardrails
We stress-tested the preprocessing pipeline with edge cases:
1. **Missing Data / NaNs:** Passed a row of pure NaNs for continuous features. Imputer cleanly substituted training set medians/modes. Output probability was valid ($0.0 \le \hat{p} \le 1.0$) with zero exceptions.
2. **Target Leakage:** Injected columns `churn`, `churndep`, `customer`, and `calibrat`. Pipeline automatically stripped all target/ID columns before feeding into `ColumnTransformer`.
3. **Missing `dropblk`:** Injected a row with `dropvce = 5.0` and `blckvce = 3.0` without `dropblk`. Pipeline dynamically computed `dropblk = dropvce + blckvce = 8.0` and produced `network_friction_rate = 8.0 / (mou + 1.0)` with $100\%$ accuracy.

---

### Task 9: Model Artifact Checksum Integrity
To guarantee that no model parameters or preprocessing weights were altered during frontend development, cryptographic MD5 hashes and byte counts were audited against Step 5:

```
Table 4: Serialized Model Integrity Audit
--------------------------------------------------------------------------------------------------
Artifact Filename                       File Size (Bytes)    MD5 Checksum Prefix    Integrity Status
--------------------------------------------------------------------------------------------------
models/best_xgboost.joblib                   280,592             fc68a58fbe09          UNMODIFIED
models/final_preprocessor.joblib               5,883             740bcb72366f          UNMODIFIED
models/calibrated_xgboost_isotonic.joblib  1,388,764             54dd65860760          UNMODIFIED
models/calibrated_xgboost_sigmoid.joblib   1,384,956             c0df7b0fd70b          UNMODIFIED
--------------------------------------------------------------------------------------------------
```
*Conclusion:* All serialized model artifacts are completely intact, unmodified, and verified.

---

## 3. Discrepancies Identified & Rectified

During this audit, one minor cosmetic discrepancy was identified and immediately corrected:
- **Discrepancy in `page_performance.py` Text:** The explanatory narrative card in Section 3 stated Precision as $44.88\%$ and F1 as $0.4908$ at $\tau = 0.50$ (values from an earlier exploratory run).
- **Rectification:** Updated [`app/views/page_performance.py`](file:///d:/mst%20project/app/views/page_performance.py) to state the exact test set metrics from [`reports/final_test_evaluation.csv`](file:///d:/mst%20project/reports/final_test_evaluation.csv): **Precision = 40.42%**, **F1 Score = 0.4947**, and **Accuracy = 62.23%**.

---

## 4. Final Verdict

**Validation Status: APPROVED FOR PRODUCTION & VIVA DEFENSE**
- All 9 verification tasks passed with zero failures.
- Streamlit application is running live at **`http://localhost:8501`**.
