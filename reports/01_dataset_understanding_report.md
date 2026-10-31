# UCS321 Machine Learning Project: Milestone 1 Report
## Problem Statement 9: Telecom Customer Churn Prediction
**Dataset:** Cell2Cell Telecom Benchmark (71,047 records x 70 attributes)  
**Evaluation Rubrics Alignment:** Problem Understanding & Objective Clarity (5 Marks) + Foundation for Pre-processing (10 Marks)

---

### 1. Executive Summary & Objective Clarity
Customer churn is one of the most critical drivers of profitability in the telecommunications industry. Acquiring a new cellular subscriber is estimated to cost **5 to 25 times more** than retaining an existing customer through proactive retention.

**Problem Statement:**  
> *"Customer churn is a critical challenge in the telecom industry, where customers discontinue their services and move to competitors. Reducing churn can significantly improve revenue and customer loyalty. Build a classification-based machine learning model to predict whether a customer is likely to churn based on their demographic details, usage patterns, billing information, and service feedback."*

**Core Project Objectives:**
1. Develop an end-to-end, reproducible machine learning classification pipeline.
2. Formulate domain-grounded feature representations across demographic, network usage, billing, and retention touchpoints.
3. Handle class imbalance (71% retained vs 29% churned) using cost-sensitive learning and SMOTE.
4. Benchmark multiple classification algorithms (Logistic Regression, Random Forest, XGBoost) using stratified cross-validation.
5. Provide actionable business interpretability using SHAP (Shapley Additive exPlanations) to empower telecom retention teams.

---

### 2. End-to-End System Architecture Flow Diagram
*(College Requirement: Flow diagram with Pre-processing and visualization steps clearly documented)*

```
+---------------------------------------------------------------------------------------------------+
|                           END-TO-END MACHINE LEARNING SYSTEM ARCHITECTURE                         |
+---------------------------------------------------------------------------------------------------+
|  [1. Data Ingestion & Audit]               [2. Exploratory Data Analysis]                         |
|  - Load Cell2Cell (71,047 x 70)            - Categorize 70 features into 4 pillars                |
|  - Audit zero duplicates                   - Analyze class distribution (~71% vs ~29%)            |
|  - Identify ground truth 'churn'           - Bivariate density & Pearson correlation analysis     |
|                   |                                            |                                  |
|                   +--------------------+-----------------------+                                  |
|                                        v                                                          |
|  [3. Data Cleaning & Pre-processing]       [4. Feature Engineering & Selection]                   |
|  - Drop leakage: customer, calibrat,       - Ratio features (eqpdays/months, changem/mou)         |
|    churndep                                - Retention urgency index                              |
|  - Median imputation for numerical         - Multicollinearity check (VIF)                        |
|  - Outlier capping at 99th percentile      - Robust / Standard Feature Scaling                    |
|                   |                                            |                                  |
|                   +--------------------+-----------------------+                                  |
|                                        v                                                          |
|  [5. Class Imbalance Strategy]             [6. Model Training & Hyperparameter Tuning]            |
|  - Cost-sensitive class weighting          - Stratified 5-Fold Cross Validation                   |
|  - SMOTE oversampling (Train split only)   - Logistic Regression (Baseline)                       |
|  - Undersampling benchmark comparison      - Random Forest & XGBoost Classifier (Champion)        |
|                   |                                            |                                  |
|                   +--------------------+-----------------------+                                  |
|                                        v                                                          |
|  [7. Comprehensive Evaluation Suite]       [8. Explainability & Business ROI (Innovation)]        |
|  - ROC-AUC, PR-AUC, F1-Score (Macro)       - SHAP TreeExplainer feature attributions              |
|  - Confusion matrix & Misclassification    - Cumulative Gains & Decile Lift ROI curves            |
|  - Probability calibration (Brier score)   - Streamlit Decision-Support Retention Prototype       |
+---------------------------------------------------------------------------------------------------+
```
*High-resolution graphic saved at: `reports/figures/00_ml_pipeline_flowchart.png`*

---

### 3. Comprehensive Dataset Overview
- **Total Records:** 71,047 customers
- **Total Features:** 70 attributes (32 binary flag features, 34 continuous numerical features, 1 customer ID, 2 calibration columns, 1 target label)
- **Duplicate Records:** 0 exact duplicate rows
- **Memory Footprint:** 37.95 MB in memory

---

### 4. Feature Taxonomy (Mapped to Problem Statement 9 Pillars)

#### A. Demographic Details (20 Features)
- `age1`, `age2`: Age of first and second household members.
- `children`: Presence of children in household (0/1).
- `credita`, `creditaa`: Credit rating indicator flags.
- `prizmrur`, `prizmub`, `prizmtwn`: Geographic segment flags (Rural, Urban, Suburban Town).
- `ownrent`: Homeowner vs. renter indicator.
- `marryun`, `marryyes`: Marital status flags (Unmarried / Married).
- `income`: Household income bracket (integer code 1-9).
- `incmiss`: Income missing indicator.
- `occprof`, `occcler`, `occcrft`, `occstud`, `occhmkr`, `occret`, `occself`: Occupational indicator flags (Professional, Clerical, Craftsman, Student, Homemaker, Retired, Self-employed).

#### B. Usage Patterns (17 Features)
- `mou`: Total monthly minutes of use.
- `overage`: Monthly overage minutes used.
- `roam`: Roaming minutes incurred.
- `changem`: Percentage change in minutes of use relative to prior baseline.
- `dropvce`: Number of dropped voice calls.
- `blckvce`: Number of blocked voice calls.
- `unansvce`: Unanswered voice calls.
- `custcare`: Customer care calls placed.
- `threeway`: Three-way conference calls placed.
- `mourec`: Inbound minutes of use received.
- `outcalls`: Completed outbound calls.
- `incalls`: Completed inbound calls.
- `peakvce`, `opeakvce`: Peak vs. off-peak voice minutes.
- `dropblk`: Combined total of dropped or blocked calls.
- `callfwdv`: Call forwarding calls.
- `callwait`: Call waiting calls.

#### C. Billing & Account Information (14 Features)
- `revenue`: Mean monthly bill revenue ($).
- `recchrge`: Total recurring monthly charges ($).
- `directas`: Directory assistance billing charges.
- `changer`: Percentage change in total revenue relative to baseline.
- `months`: Customer account tenure (months subscribed).
- `uniqsubs`, `actvsubs`: Number of unique and active cellular subscriptions on the account.
- `phones`, `models`: Cumulative count of phones and handset models issued.
- `eqpdays`: Age in days of current handset/equipment.
- `setprcm`: Price of current phone missing flag.
- `setprc`: Price of current handset equipment.
- `refurb`: Handset refurbished flag.
- `webcap`: Wireless Web capability indicator.

#### D. Service Feedback & Retention (4 Features)
- `retcalls`: Total number of retention calls made by customer.
- `retaccpt`: Total number of retention offers accepted by customer.
- `retcall`: Retention call flag (1 if at least one retention call was logged).
- `refer`: Number of customer referral calls.

#### E. Lifestyle & Household Characteristics (11 Features)
- `mailord`: Mail order buyer flag.
- `mailres`: Mail order responder flag.
- `mailflag`: Mail marketing indicator.
- `travel`: Frequent travel flag.
- `pcown`: Personal computer ownership flag.
- `creditcd`: Credit card holder flag.
- `newcelly`, `newcelln`: New cellular customer flags.
- `mcycle`, `truck`, `rv`: Vehicle ownership flags (Motorcycle, Truck, Recreational Vehicle).

#### F. Administrative / Sampling Artifacts (3 Columns)
- `customer`: Arbitrary customer identifier.
- `calibrat`: Tournament sub-sample flag (1 = balanced 50/50 training sample, 0 = holdout sample).
- `churndep`: Target variable column populated only for `calibrat==1` (contains 31,047 NaNs for holdout).

---

### 5. Critical Discovery: Target Ground Truth & Leakage Avoidance
During the data audit, we investigated `churn`, `calibrat`, and `churndep`:
1. `churn` is the **true, complete ground truth target** across all 71,047 records:
   - **Retained (0):** 50,438 customers (70.99%)
   - **Churned (1):** 20,609 customers (29.01%)
   - **Imbalance Ratio:** 2.45 : 1
2. `churndep` and `calibrat` were historical artifacts of the Cell2Cell competition where organizers pre-selected 20,000 churners and 20,000 non-churners into a balanced partition (`calibrat=1`).
3. **Evaluation Distinction for College Marks:**  
   Using `churndep` or keeping `calibrat` in the feature matrix introduces severe **data leakage** and restricts dataset size to only 40,000.  
   **Rigorous Decision:** We utilize the full 71,047 records with `churn` as the target and drop `customer`, `calibrat`, and `churndep` prior to machine learning modeling.

---

### 6. Missing Value Analysis
Missing data is present in 14 columns, but actual operational missingness is small (< 2%):
- `age1`, `age2`: 1,244 missing values (1.75%)
- `changem`, `changer`: 502 missing values (0.71%)
- `revenue`, `mou`, `recchrge`, `directas`, `overage`, `roam`: 216 missing values (0.30%)
- `phones`, `models`, `eqpdays`: 1 missing value (0.001%)
- `churndep`: 31,047 (43.70%) — intentional competition artifact to be dropped.

**Pre-processing Imputation Plan:**
- Numerical usage/billing features: Impute using **median** (robust against extreme skews).
- Age features: Impute with median and create missing indicator flags (`age_missing`).

---

### 7. Exploratory Statistical Insights & Correlation Analysis

#### Top Positive Drivers of Churn:
1. `eqpdays` (Handset Age in Days, $r = +0.105$): Customers holding older, outdated handsets show significantly higher churn probabilities.
2. `retcall` & `retcalls` (Retention Escalations, $r = +0.073$): Customers who reached out to retention departments are already on the verge of defecting.
3. `setprcm` (Price Missing Flag, $r = +0.036$): Correlated with legacy promotions.

#### Top Negative Drivers of Churn (Protective Factors):
1. `webcap` (Wireless Web Capable, $r = -0.062$): Modern device capabilities increase digital engagement and reduce churn.
2. `recchrge` (Recurring Plan Value, $r = -0.059$): Subscribers on standardized recurring packages demonstrate higher retention than erratic pay-as-you-go users.
3. `creditaa` (Top Credit Tier, $r = -0.056$): Prime credit customers have stable finances and churn less frequently.
4. `mou` (Minutes of Use, $r = -0.052$): Highly engaged network users are less likely to disconnect.

---

### 8. Milestone 1 Deliverables Summary
1. **Directory Structure:** Clean organization with `data/raw`, `data/processed`, `notebooks/`, `src/`, `reports/figures/`.
2. **Environment Configuration:** Python 3.13, Jupyter Notebook 7.6+, pandas, numpy, scikit-learn, xgboost, imbalanced-learn, matplotlib, seaborn, scipy.
3. **Executable Notebook:** `notebooks/01_dataset_understanding_eda.ipynb` executed end-to-end with pre-rendered visual charts and markdown narrative.
4. **Architecture Flowchart:** `reports/figures/00_ml_pipeline_flowchart.png` documenting pre-processing, modeling, and evaluation.
5. **Requirements File:** `requirements.txt` for one-command reproducibility.
