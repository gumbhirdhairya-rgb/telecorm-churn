# Telecom Customer Churn Prediction Using Machine Learning

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/Model-CatBoost-yellow.svg)](https://catboost.ai/)
[![SHAP](https://img.shields.io/badge/Explainability-TreeExplainer-green.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> A comparative machine learning study evaluating gradient boosting, tree ensembles, and linear models across two distinct telecommunications datasets: **Cell2Cell (Customer Usage, Billing & Service Records, 71,047 records)** and **IBM Telco (Contractual Subscriptions, 7,043 records)**. Includes TreeExplainer SHAP interpretability, retention risk segmentation, and a local interactive Streamlit web application.

---

## 1. Project Introduction

Customer churn represents significant lost revenue for telecommunications carriers. Predicting churn requires machine learning systems that can handle class imbalance, mixed data modalities (continuous usage metrics alongside discrete categorical indicators), and complex decision boundaries.

This project investigates churn prediction as a **two-dataset comparative machine learning study**:
1. **Dataset 1 (Cell2Cell):** Focuses on customer-level cellular usage, billing trends, network friction, and equipment records. Evaluated primarily for **recall optimization, probability calibration, and simulated retention economics**.
2. **Dataset 2 (IBM Telco):** Focuses on structured contractual service relationships (tenure, contract duration, payment method, and internet service add-ons), where explicit contract commitments enable **higher raw classification accuracy (~81%)**.

Rather than treating churn prediction as a one-size-fits-all classification task, this study demonstrates how **dataset characteristics, feature types, and operational objectives govern model selection, threshold calibration, and evaluation metrics**.

---

## 2. Dataset 1 – Cell2Cell Customer Usage & Service Records

### 2.1 Dataset Overview & Feature Engineering
- **Total Records:** **71,047 customers** (70 raw attributes).
- **Target Distribution:** 50,438 non-churners (70.99%) vs. 20,609 churners (**29.01% churn prevalence**).
- **Feature Space:** Customer call volume (`mou`, `peakvce`, `opeakvce`), billing patterns (`revenue`, `overage`, `recchrge`), customer care interactions (`custcare`), equipment age (`eqpdays`), and demographic/credit indicators.
- **Engineered Domain Features:** 5 domain features were engineered to capture service and billing friction:
  - `cost_per_mou`: Unit billing pressure ($\text{revenue} / (\text{mou} + 1.0)$).
  - `network_friction_rate`: Service quality failure rate ($\text{dropblk} / (\text{mou} + 1.0)$), using total dropped and blocked calls (`dropblk`).
  - `handset_age_ratio`: Relative equipment obsolescence ($\text{eqpdays} / (\text{months} \times 30.4375 + 1.0)$).
  - `overage_burden`: Bill shock indicator ($\text{overage} / (\text{recchrge} + 1.0)$).
  - `usage_drop_severe`: Severe usage drop flag ($(\text{changem} < -20) \land (\text{mou} > 50)$).
- **Final Feature Matrix:** **63 selected features** (34 numerical, 29 binary indicators) with zero target or customer identifier leakage.

### 2.2 Independent Holdout Test Performance (14,210 Customers)
Evaluated on an independent, untouched 20% holdout test partition (`data/processed/X_test_final.csv`, 4,122 actual churners):

| Model | Decision Threshold ($\tau$) | Accuracy | Balanced Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1-Score | True Positives ($TP$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.5000 | 59.60% | 61.27% | 0.6252 | 0.3898 | 37.21% | 57.13% | 0.4507 | 2,355 |
| **Decision Tree** | 0.5000 | 58.54% | 61.16% | 0.6427 | 0.4080 | 37.28% | 62.91% | 0.4681 | 2,593 |
| **Random Forest** | 0.5000 | 62.22% | 62.30% | 0.6695 | 0.4439 | 40.01% | 60.53% | 0.4818 | 2,495 |
| **XGBoost (Champion)** | 0.5000 | **62.23%** | 62.88% | **0.6786** | **0.4586** | **40.42%** | 63.73% | 0.4947 | 2,627 |
| **XGBoost (Tuned Boundary)** | **0.4631** | 58.10% | **62.88%** | **0.6786** | **0.4586** | 38.30% | **72.78%** | **0.5019** | **3,000** |

*Data Source:* Verified against [`reports/final_test_evaluation.csv`](reports/final_test_evaluation.csv).

### 2.3 Threshold Tuning Rationale ($\tau^* = 0.4631$) & Precision-Recall Trade-Off
- **Derivation (Zero Test Leakage):** The optimal decision threshold $\tau^* = 0.4631$ was identified **strictly on training set out-of-fold cross-validation predictions** to balance class cost sensitivity, without using holdout test labels.
- **The Operational Trade-Off:** 
  - At the standard threshold $\tau = 0.50$, the champion model identifies 2,627 churners (63.73% recall) with 40.42% precision.
  - Lowering the decision threshold to $\tau^* = 0.4631$ increases **Recall to 72.78%** (+373 additional churners caught, $TP=3,000$).
  - The trade-off is a slight decline in precision from 40.42% to 38.30% (False Positives increase from 3,872 to 4,832). In a telecom retention setting, failing to flag an impending churner ($FN$) is significantly more expensive than conducting an automated digital intervention for a loyal subscriber ($FP$).

### 2.4 Model Improvement Experiments (Leakage-Free 5-Fold Stratified Cross-Validation)
To benchmark alternative gradient boosting paradigms on Cell2Cell, CatBoost and LightGBM were evaluated across the 56,837 development records using **Stratified 5-Fold Cross-Validation** with in-fold preprocessing (imputer and scaler fitted exclusively inside each training fold):

| Model Architecture | 5-Fold CV ROC-AUC | 5-Fold CV PR-AUC | 5-Fold CV Accuracy | 5-Fold CV Recall | Total 5-Fold CV Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Re-evaluated In-Fold)** | 0.6800 ± 0.0068 | 0.4590 ± 0.0081 | 61.96% | **64.74%** | 3.0s |
| **CatBoost (Tuned/Regularized)** | 0.6821 ± 0.0059 | 0.4660 ± 0.0086 | 63.57% | 61.48% | 35.9s |
| **LightGBM (Depth-Constrained)** | **0.6833 ± 0.0069** | 0.4656 ± 0.0105 | 62.83% | 63.10% | **6.9s** ⚡ |
| **LightGBM (Tuned/Regularized)** | 0.6830 ± 0.0067 | **0.4670 ± 0.0099** | **64.16%** | 59.46% | 13.6s |

*Data Source:* Verified against [`reports/model_improvement/`](reports/model_improvement/) cross-validation CSVs.

### 2.5 SHAP Explainability & Top 10 Predictors
Feature attributions were computed using polynomial-time `shap.TreeExplainer` on the 14,210-customer test partition. **4 of the 5 engineered features rank in the Global Top 10** by mean absolute SHAP value:

```
Rank   Feature                  Mean |SHAP|   Feature Type   Statistical Interpretation
---------------------------------------------------------------------------------------------------------
 1     eqpdays                   0.2284       Continuous     Handset age in days; high equipment age correlates strongly with churn risk.
 2     months                    0.1553       Continuous     Customer tenure; highest churn risk observed in first 14 months of service.
 3     changem                   0.1449       Continuous     Usage velocity; sharp drop (>20%) signals impending disengagement.
 4     uniqsubs                  0.0874       Continuous     Number of unique subscriptions on the customer account.
 5     cost_per_mou [ENGINEERED] 0.0851       Engineered     Effective unit rate ($/min); 2.4x higher SHAP contribution than raw revenue.
 6     age1                      0.0776       Continuous     Primary subscriber age; younger demographics exhibit higher churn propensity.
 7     network_friction_rate     0.0694       Engineered     dropblk / (mou + 1.0); technical friction from dropped/blocked call rates.
 8     recchrge                  0.0656       Continuous     Base recurring monthly charge level.
 9     handset_age_ratio         0.0602       Engineered     eqpdays / (months * 30.4375 + 1.0); equipment decay relative to tenure.
 10    overage_burden            0.0560       Engineered     overage / (recchrge + 1.0); bill shock intensity indicator.
```

> **Statistical Attribution vs. Causal Mechanism:** SHAP values represent statistical attribution toward model log-odds, not causal proof. For instance, high equipment age (`eqpdays`) correlates with churn because subscribers with older devices are more likely to seek upgrades or switch providers. Artificially forcing a subscriber to keep an older device would not reduce churn; addressing the underlying need via an upgrade incentive is the appropriate intervention.

### 2.6 Retention Risk Segmentation & Simulated Economics
Hypothetical retention policies were simulated across the 14,210 test customers under explicit business assumptions (assumed Customer Lifetime Value $V_{\text{LTV}} = \$250$, assumed intervention success uplift $r = 25\%$):

| Retention Policy | Target Population | True Churners ($TP$) | Churners Saved | Simulated Cost | Simulated Gross LTV | Simulated Net Benefit | Estimated ROI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Do Nothing** | 0 | 0 | 0.0 | $\$0$ | $\$0$ | **$\$0$** | 0.0% |
| **2. Blanket Campaign** | 14,210 | 4,122 | 1,030.5 | $\$426,300$ | $\$257,625$ | **$-\$168,675$** | -39.6% |
| **3. High Risk Tier ($\ge 0.60$)** | 2,814 | 1,365 | 341.2 | $\$84,420$ | $\$85,312$ | **$+\$892$** | +1.1% |
| **4. Max-Profit Cutoff ($\ge 0.67$)**| 1,174 | 680 | 170.0 | $\$35,220$ | $\$42,500$ | **$+\$7,280$** | +20.7% |
| **5. Differentiated Tiered Strategy**| **10,998** | **3,713** | **761.7** | **$\$131,226$** | **$\$190,425$** | **$+\$59,199$** | **+45.1%** |

*Note: All monetary figures are hypothetical simulations based on stated modeling assumptions, not actual audited accounting returns.*

---

## 3. Dataset 2 – IBM Telco Contractual Benchmark

### 3.1 Dataset Overview
- **Total Records:** **7,043 customers** (21 attributes).
- **Target Distribution:** 5,174 non-churners (73.46%) vs. 1,869 churners (**26.54% churn prevalence**).
- **Feature Space:** Contract terms (`Month-to-month`, `One year`, `Two year`), tenure (0–72 months), monthly charges, total charges, internet service type (`DSL`, `Fiber optic`, `No`), payment methods (`Electronic check`, `Mailed check`, bank transfer, credit card), and service add-ons (`TechSupport`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`).
- **Data Partitions:** 80% development train split (**5,634 customers**) and 20% holdout test split (**1,409 customers**).

### 3.2 Independent Holdout Test Performance (1,409 Customers)
Evaluated on an independent, untouched 20% holdout test split (374 actual churners, 1,035 non-churners):

| Model Architecture | Test Accuracy | Test ROC-AUC | Test PR-AUC | Test Precision | Test Recall | Test F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | **80.98%** | **0.8473** | **0.6675** | **68.28%** | 52.94% | **0.5964** |
| **XGBoost Classifier** | **80.55%** | **0.8448** | 0.6564 | 66.89% | 52.94% | 0.5910 |
| **Random Forest** | **80.34%** | **0.8457** | 0.6594 | 66.78% | 51.60% | 0.5822 |
| **CatBoost Classifier** | **80.20%** | **0.8425** | 0.6485 | 66.21% | 51.87% | 0.5817 |
| **LightGBM Classifier** | **80.20%** | **0.8424** | 0.6490 | 65.68% | **53.21%** | 0.5879 |
| **Blend Ensemble (XGB+LGB+Cat)** | **80.13%** | **0.8445** | 0.6563 | 65.56% | 52.94% | 0.5858 |

*Data Source:* Verified against [`reports/ibm_telco/ibm_test_evaluation.csv`](reports/ibm_telco/ibm_test_evaluation.csv).

### 3.3 Evaluation Insights: The Accuracy Paradox in IBM Telco
- While Logistic Regression achieves a strong headline accuracy of **80.98%**, its **Recall is only 52.94%** (198 True Positives vs. 176 False Negatives).
- This means the model **misses 47.06% of actual churners** in the test set.
- The high accuracy is largely driven by correctly classifying non-churners ($TN = 943$ out of $1,035$), illustrating that headline accuracy alone is insufficient for evaluating churn models on imbalanced data.

---

## 4. Cross-Dataset Comparative Analysis

The table below summarizes the two datasets using **their respective holdout test set evaluations**:

| Evaluation Dimension | Dataset 1: Cell2Cell Wireless | Dataset 2: IBM Telco |
| :--- | :--- | :--- |
| **Total Record Count** | **71,047 customers** (Large-scale usage telemetry) | **7,043 customers** (Contractual subscription) |
| **Holdout Test Set Size** | **14,210 customers** (20% split) | **1,409 customers** (20% split) |
| **Churn Prevalence** | **29.01%** (20,609 churners / 71,047) | **26.54%** (1,869 churners / 7,043) |
| **Key Predictive Signals** | Usage velocity (`changem`), equipment age (`eqpdays`), overage | Contract duration (`Month-to-month`), tenure, payment method |
| **Best Model Evaluated** | Cost-Sensitive XGBoost (Threshold $\tau^*=0.4631$) | Regularized Logistic Regression / XGBoost |
| **Test Accuracy** | **58.10%** (Tuned $\tau^*$) / **62.23%** (Default 0.50) | **80.98%** (Logistic Regression) / **80.55%** (XGBoost) |
| **Test ROC-AUC** | **0.6786** | **0.8473** |
| **Test PR-AUC** | **0.4586** | **0.6675** |
| **Test Precision** | **38.30%** (Tuned $\tau^*$) / **40.42%** (Default 0.50) | **68.28%** |
| **Test Recall** | **72.78%** (Tuned $\tau^*$) / **63.73%** (Default 0.50) | **52.94%** |
| **Test F1-Score** | **0.5019** (Tuned $\tau^*$) / **0.4947** (Default 0.50) | **0.5964** |
| **Primary Operational Focus** | Prioritizing high churn recall to prevent lost subscribers | High-precision classification based on contract boundaries |

> **Methodological Context:** The metrics above reflect fundamentally different data-generating processes. Cell2Cell contains continuous, noisy telemetry with unrecorded external drivers (resulting in an empirical ceiling near 0.70 ROC-AUC), whereas IBM Telco features explicit contract terms and payment methods that provide clean, separable signals.

---

## 5. Current Model Directory Structure

The trained model artifacts in [`models/`](models/) are organized into four functional directories:

```
models/
├── production_cell2cell/                  # Actively loaded by the Streamlit application
│   ├── best_xgboost.joblib                # Primary Champion XGBoost model
│   ├── final_preprocessor.joblib          # 63-Feature ColumnTransformer preprocessor
│   ├── calibrated_xgboost_isotonic.joblib # Isotonic probability calibrator
│   ├── calibrated_xgboost_sigmoid.joblib  # Platt sigmoid probability calibrator
│   └── unweighted_xgboost.joblib          # Unweighted reference model
│
├── production_ibm_telco/                  # Gradient-boosted models for the IBM Telco cohort
│   ├── best_ibm_xgboost.joblib            # XGBoost model (80.55% test accuracy)
│   ├── best_ibm_lightgbm.joblib           # LightGBM model (80.20% test accuracy)
│   ├── best_ibm_catboost.joblib           # CatBoost model (80.20% test accuracy)
│   └── ibm_preprocessor.joblib            # ColumnTransformer preprocessor for IBM Telco
│
├── model_improvement/                     # 5-Fold in-fold cross-validation experiments
│   ├── catboost/                          # CatBoost classifier & preprocessor
│   ├── lightgbm/                          # LightGBM classifier & preprocessor
│   └── xgboost/                           # Re-evaluated in-fold XGBoost & preprocessor
│
└── experimental_baselines/                # Historical benchmark comparison models
    ├── best_random_forest.joblib          # Random Forest benchmark (23.8 MB)
    ├── best_decision_tree.joblib          # Decision Tree benchmark
    ├── best_logistic_regression.joblib    # Linear baseline
    └── preprocessor.joblib                # Step 2 preliminary preprocessor
```

### Explanatory Note on IBM Production Artifacts (Point 3)
* While Logistic Regression achieved the highest linear holdout accuracy on IBM Telco (80.98%), the serialized artifacts in `models/production_ibm_telco/` store the **gradient-boosted tree models (XGBoost, LightGBM, CatBoost)** and their shared preprocessor.
* Gradient-boosted models were prioritized for serialization because they capture non-linear feature interactions, support threshold calibration, and enable TreeExplainer SHAP interpretability. Logistic Regression metrics remain fully documented in [`reports/ibm_telco/ibm_test_evaluation.csv`](reports/ibm_telco/ibm_test_evaluation.csv).

---

## 6. Interactive Streamlit Web Application

The interactive web dashboard is located in [`app/`](app/) and runs locally via Streamlit.

### Application Scope & Loaded Models
- **Supported Dataset:** The application **strictly serves Dataset 1 (Cell2Cell)**. It ingests the 63-feature cellular telemetry schema and processes inputs through `models/production_cell2cell/`.
- **Inference Models Loaded:**
  - `best_xgboost.joblib` (Primary inference model)
  - `final_preprocessor.joblib` (Numerical median imputation + standard scaling + binary passthrough)
  - `calibrated_xgboost_isotonic.joblib` (Calibrated probability mode)
  - `calibrated_xgboost_sigmoid.joblib` (Platt scaling mode)
- **Supported Prediction Modes:**
  1. *Decision Mode:* Uses the cost-sensitive threshold $\tau^* = 0.4631$ to prioritize churn recall.
  2. *Calibrated Mode:* Outputs empirical isotonic-calibrated probabilities.
  3. *Prior-Adjusted Mode:* Re-scales probabilities to natural industry prevalence ($1.96\%$) via Bayes' rule.

*(Note: The IBM Telco pipeline is maintained as a separate research module in `models/production_ibm_telco/` and `src/ibm_telco/`; it is not currently rendered in the Streamlit UI).*

---

## 7. Technology Stack

- **Core Runtime:** Python 3.13
- **Machine Learning & Preprocessing:** Scikit-Learn 1.6, XGBoost 2.1, LightGBM 4.7, CatBoost 1.2
- **Model Interpretability:** SHAP 0.46 (TreeExplainer)
- **Data Engineering:** Pandas 3.0, NumPy 2.5
- **Interactive UI:** Streamlit 1.45+
- **Data Visualization:** Matplotlib 3.11, Seaborn 0.13

---

## 8. Installation & Run Instructions

### 8.1 Clone the Repository
```bash
git clone https://github.com/gumbhirdhairya-rgb/telecorm-churn.git
cd telecorm-churn
```

### 8.2 Create & Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 8.3 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 8.4 Launch the Interactive Streamlit Application
```bash
python -m streamlit run app/app.py
```
Open **`http://localhost:8501`** in your browser.

### 8.5 Reproducing Pipeline Experiments
To independently run any training pipeline or verification suite:

```bash
# Run Step 1: CatBoost 5-Fold Cross-Validation
python src/train_catboost_cv.py

# Run Step 2: LightGBM 5-Fold Cross-Validation
python src/train_lightgbm_cv.py

# Run Step 3: Fair XGBoost 5-Fold Cross-Validation
python src/train_xgboost_cv.py

# Run Dataset 2: IBM Telco Training & Holdout Test Evaluation
python src/ibm_telco/train_ibm_telco_pipeline.py

# Run End-to-End Persona Inference Tests
python src/test_personas_output.py
```

---

## 9. Academic Integrity & Limitations

1. **Dataset-Specific Performance:** Model metrics are bounded by the underlying data-generating process. A model achieving 81% accuracy on IBM Telco cannot be assumed to achieve 81% on Cell2Cell without corresponding contractual features.
2. **Attribution vs. Causation:** SHAP values identify statistical contributions to the model's log-odds output. They do **not** establish real-world causal mechanisms. Interventions must address root causes rather than targeting observational symptoms.
3. **Simulation Assumptions:** Financial ROI and retention savings are hypothetical simulations based on stated modeling assumptions ($V_{\text{LTV}} = \$250$, intervention uplift $r = 25\%$, intervention cost $C = \$30$). Real-world enterprise results depend on marketing campaign execution, competitive behavior, and offer acceptance rates.
4. **Generalization Constraints:** Cross-validation and holdout validation protect against data leakage within each dataset partition, but they do not guarantee generalization to future time periods or international carrier markets.

---

## 10. Future Scope

1. **Multi-Cohort UI Integration:** Adding an IBM Telco inference tab to the Streamlit application to allow real-time contractual churn scoring alongside the existing Cell2Cell telemetry scoring.
2. **Dynamic LTV-Weighted Thresholds:** Extending the decision boundary from a fixed threshold ($\tau^* = 0.4631$) to customer-specific thresholds calculated dynamically from real-time customer lifetime value estimates.
3. **Deep Tabular Architectures:** Benchmarking attention-based tabular architectures (e.g., TabNet, FT-Transformer) against the gradient-boosted tree models.
4. **Causal Uplift Modeling:** Transitioning from observational churn probability estimation to uplift modeling (two-model or causal forest approach) to predict which subscribers will alter their behavior when targeted by a retention incentive.

---

## 11. License & Citation

Distributed under the MIT License. Developed as an academic machine learning research project for telecommunications customer retention analytics.
