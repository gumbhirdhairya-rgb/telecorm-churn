# Telecom Customer Churn Prediction Using Machine Learning

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/Model-CatBoost-yellow.svg)](https://catboost.ai/)
[![SHAP](https://img.shields.io/badge/Explainability-TreeExplainer-green.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> A comparative machine learning study evaluating gradient boosting, tree ensembles, and linear models across two distinct telecommunications datasets: **Cell2Cell (Large-Scale Enterprise Telemetry, 71,047 records)** and **IBM Telco (Contractual Subscriptions, 7,043 records)**. Includes TreeExplainer SHAP interpretability, retention risk segmentation, and a deployed Streamlit web application.

---

## 1. Project Introduction

Customer churn represents significant lost revenue for telecommunications carriers. Predicting churn requires machine learning systems that can handle class imbalance, mixed data modalities (numerical usage metrics alongside discrete categorical indicators), and complex decision boundaries.

This project investigates churn prediction as a **two-dataset comparative machine learning study**:
1. **Dataset 1 (Cell2Cell):** Focuses on uncurated, large-scale wireless network telemetry, where customer churn is heavily driven by usage friction, dropped calls, and billing changes. Evaluated primarily for **cost-benefit retention economics and recall optimization**.
2. **Dataset 2 (IBM Telco):** Focuses on structured contractual service relationships (tenure, contract duration, payment method, add-on internet services), where clear contract boundaries enable **high-accuracy discrimination (80%+)**.

Rather than treating churn prediction as a one-size-fits-all classification task, this study demonstrates how **dataset characteristics and business objectives govern model selection, threshold calibration, and evaluation metrics**.

---

## 2. Dataset 1 – Cell2Cell Wireless Telemetry

### 2.1 Dataset Overview & Feature Engineering
- **Total Records:** **71,047 customers** (70 raw attributes).
- **Target Distribution:** 50,438 non-churners (70.99%) vs. 20,609 churners (**29.01% churn prevalence**).
- **Feature Space:** Cellular usage volume (`mou`, `peakvce`, `opeakvce`), billing patterns (`revenue`, `overage`, `recchrge`), customer care interactions (`custcare`), equipment age (`eqpdays`), and demographic/credit indicators.
- **Engineered Domain Features:** 5 non-linear features were engineered to capture service decay:
  - `cost_per_mou`: Unit cost of voice consumption ($\text{revenue} / (\text{mou} + 1)$).
  - `network_friction_rate`: Service quality failure rate ($(\text{dropvce} + \text{blckvce}) / (\text{mou} + 1)$).
  - `handset_age_ratio`: Relative equipment obsolescence ($\text{eqpdays} / (\text{months} \times 30.4375 + 1)$).
  - `overage_burden`: Bill shock indicator ($\text{overage} / (\text{recchrge} + 1)$).
  - `usage_drop_severe`: Indicator for steep usage velocity decrease ($\text{changem} < -0.20$).
- **Final Feature Matrix:** **63 selected features** (34 numerical, 29 binary indicators) with zero target/identifier leakage.

### 2.2 Verified Model Performance (14,210 Holdout Test Records)
Evaluated on an independent, untouched 20% holdout test partition (`data/processed/X_test_final.csv`, 4,122 actual churners):

| Model | Decision Threshold ($\tau$) | Accuracy | Balanced Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1-Score | True Positives ($TP$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.5000 | 59.60% | 61.27% | 0.6252 | 0.3898 | 37.21% | 57.13% | 0.4507 | 2,355 |
| **Decision Tree** | 0.5000 | 58.54% | 61.16% | 0.6427 | 0.4080 | 37.28% | 62.91% | 0.4681 | 2,593 |
| **Random Forest** | 0.5000 | 62.22% | 62.30% | 0.6695 | 0.4439 | 40.01% | 60.53% | 0.4818 | 2,495 |
| **XGBoost (Champion)** | 0.5000 | **62.23%** | 62.88% | **0.6786** | **0.4586** | **40.42%** | 63.73% | 0.4947 | 2,627 |
| **XGBoost (Tuned Boundary)** | **0.4631** | 58.10% | **62.88%** | **0.6786** | **0.4586** | 38.30% | **72.78%** | **0.5019** | **3,000** |

*Data Source:* Verified against `reports/final_test_evaluation.csv`.

### 2.3 Model Improvement Experiments (Leakage-Free 5-Fold Stratified CV)
To test modern gradient boosting architectures on Cell2Cell, CatBoost and LightGBM were evaluated under strict in-fold preprocessing (median imputation and scaling fitted exclusively inside each training fold):

| Model Architecture | 5-Fold CV ROC-AUC | 5-Fold CV PR-AUC | 5-Fold CV Accuracy | 5-Fold CV Recall | Total 5-Fold Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Re-evaluated In-Fold)** | 0.6800 ± 0.0068 | 0.4590 ± 0.0081 | 61.96% | **64.74%** | 3.0s |
| **CatBoost (Tuned/Regularized)** | 0.6821 ± 0.0059 | 0.4660 ± 0.0086 | 63.57% | 61.48% | 35.9s |
| **LightGBM (Depth-Constrained)** | **0.6833 ± 0.0069** | 0.4656 ± 0.0105 | 62.83% | 63.10% | **6.9s** ⚡ |
| **LightGBM (Tuned/Regularized)** | 0.6830 ± 0.0067 | **0.4670 ± 0.0099** | **64.16%** | 59.46% | 13.6s |

*Data Source:* Verified against `reports/model_improvement/xgboost/xgboost_cv_results.csv`, `catboost_cv_results.csv`, and `lightgbm_cv_results.csv`.

### 2.4 SHAP Explainability & Top 10 Predictors
Feature attributions were computed using TreeExplainer on the test partition. **4 of our 5 engineered features rank in the Global Top 10** by mean absolute SHAP value:

```
Rank   Feature                  Mean |SHAP|   Feature Type   Business Interpretation
---------------------------------------------------------------------------------------------------------
 1     eqpdays                   0.2284       Continuous     Handset age; risk spikes near 365 days (contract end).
 2     months                    0.1553       Continuous     Customer tenure; highest churn risk in first 14 months.
 3     changem                   0.1449       Continuous     Usage velocity; sharp drop (>20%) signals impending churn.
 4     uniqsubs                  0.0874       Continuous     Number of unique subscriptions on the account.
 5     cost_per_mou [ENGINEERED] 0.0851       Engineered     Effective unit rate ($/min); 2.4x more predictive than revenue.
 6     age1                      0.0776       Continuous     Primary subscriber age; younger users exhibit higher turnover.
 7     network_friction_rate     0.0694       Engineered     (dropvce + blckvce)/(mou + 1); technical quality dissatisfaction.
 8     recchrge                  0.0656       Continuous     Base recurring monthly charge level.
 9     handset_age_ratio         0.0602       Engineered     eqpdays / (months * 30.4375 + 1); equipment decay relative to tenure.
 10    overage_burden            0.0560       Engineered     overage / (recchrge + 1); bill shock intensity indicator.
```

### 2.5 Retention Risk Segmentation & Economics
Simulated on 14,210 test customers under explicit assumptions (Customer Lifetime Value $V_{\text{LTV}} = \$250$, intervention success uplift $r = 25\%$):

| Retention Policy | Target Population | True Churners ($TP$) | Churners Saved | Campaign Cost | Gross LTV Preserved | Net Economic Benefit | Estimated ROI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Do Nothing** | 0 | 0 | 0.0 | $\$0$ | $\$0$ | **$\$0$** | 0.0% |
| **2. Blanket Campaign** | 14,210 | 4,122 | 1,030.5 | $\$426,300$ | $\$257,625$ | **$-\$168,675$** | -39.6% |
| **3. High Risk Tier ($\ge 0.60$)** | 2,814 | 1,365 | 341.2 | $\$84,420$ | $\$85,312$ | **$+\$892$** | +1.1% |
| **4. Max-Profit Cutoff ($\ge 0.67$)**| 1,174 | 680 | 170.0 | $\$35,220$ | $\$42,500$ | **$+\$7,280$** | +20.7% |
| **5. Differentiated Tiered Strategy**| **10,998** | **3,713** | **761.7** | **$\$131,226$** | **$\$190,425$** | **$+\$59,199$** | **+45.1%** |

---

## 3. Dataset 2 – IBM Telco Contractual Benchmark

### 3.1 Dataset Overview
- **Total Records:** **7,043 customers** (21 attributes).
- **Target Distribution:** 5,174 non-churners (73.46%) vs. 1,869 churners (**26.54% churn prevalence**).
- **Feature Space:** Contract commitment (`Month-to-month`, `One year`, `Two year`), tenure (0–72 months), monthly charges, total charges, internet service type (`DSL`, `Fiber optic`, `No`), payment methods (`Electronic check`, `Mailed check`, bank transfer, credit card), and security add-ons (`TechSupport`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`).
- **Data Partitions:** 80% development train split (**5,634 customers**) and 20% holdout test split (**1,409 customers**).

### 3.2 Verified Test Performance (1,409 Separate Holdout Records)
Evaluated on an independent, untouched 20% holdout test split:

| Model Architecture | Test Accuracy | Test ROC-AUC | Test PR-AUC | Test Precision | Test Recall | Test F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | **80.98%** | **0.8473** | **0.6675** | **68.28%** | 52.94% | **0.5964** |
| **XGBoost Classifier** | **80.55%** | **0.8448** | 0.6564 | 66.89% | 52.94% | 0.5910 |
| **Random Forest** | **80.34%** | **0.8457** | 0.6594 | 66.78% | 51.60% | 0.5822 |
| **CatBoost Classifier** | **80.20%** | **0.8425** | 0.6485 | 66.21% | 51.87% | 0.5817 |
| **LightGBM Classifier** | **80.20%** | **0.8424** | 0.6490 | 65.68% | **53.21%** | 0.5879 |
| **Blend Ensemble (XGB+LGB+Cat)** | **80.13%** | **0.8445** | 0.6563 | 65.56% | 52.94% | 0.5858 |

*Data Source:* Verified against `reports/ibm_telco/ibm_test_evaluation.csv`.

---

## 4. Cross-Dataset Comparative Analysis

| Dimension | Dataset 1: Cell2Cell Wireless | Dataset 2: IBM Telco |
| :--- | :--- | :--- |
| **Total Record Count** | **71,047 customers** (Large enterprise telemetry) | **7,043 customers** (Contractual subscription) |
| **Churn Prevalence** | **29.01%** (20,609 / 71,047) | **26.54%** (1,869 / 7,043) |
| **Dominant Predictors** | Usage drops (`changem`), equipment age (`eqpdays`), overage | Contract type (`Month-to-month`), tenure, payment method |
| **Best Model Architecture** | Cost-Sensitive XGBoost / LightGBM | Logistic Regression / XGBoost Classifier |
| **Test Accuracy** | **62.23%** (at $\tau=0.50$) / **58.10%** (at $\tau^*=0.4631$) | **80.98%** (Logistic Regression) / **80.55%** (XGBoost) |
| **Test ROC-AUC** | **0.6786** | **0.8473** |
| **Test PR-AUC** | **0.4586** | **0.6675** |
| **Test Precision** | **40.42%** | **68.28%** |
| **Test Recall** | **63.73%** (0.50) / **72.78%** ($\tau^*=0.4631$) | **52.94%** |
| **Test F1-Score** | **0.4947** (0.50) / **0.5019** ($\tau^*=0.4631$) | **0.5964** |
| **Primary Practical Objective** | Maximizing retention recall and net financial profit | High-precision screening of month-to-month subscribers |

> **Important Methodological Note:** Cell2Cell and IBM Telco feature different feature sets, temporal aggregations, and sampling methodologies. Therefore, the metric differences between them **reflect domain characteristics rather than an algorithmic superiority of one model over another**.

---

## 5. Current Model Directory Structure

The trained model artifacts in [`models/`](models/) are organized into four dedicated directories:

```
models/
├── production_cell2cell/                  # Actively loaded by the Streamlit application
│   ├── best_xgboost.joblib                # Production Champion XGBoost model
│   ├── final_preprocessor.joblib          # 63-Feature ColumnTransformer preprocessor
│   ├── calibrated_xgboost_isotonic.joblib # Isotonic probability calibrator
│   ├── calibrated_xgboost_sigmoid.joblib  # Platt sigmoid probability calibrator
│   └── unweighted_xgboost.joblib          # Unweighted reference model
│
├── production_ibm_telco/                  # Trained models for the IBM Telco cohort
│   ├── best_ibm_xgboost.joblib            # XGBoost model (80.55% test accuracy)
│   ├── best_ibm_lightgbm.joblib           # LightGBM model (80.20% test accuracy)
│   ├── best_ibm_catboost.joblib           # CatBoost model (80.20% test accuracy)
│   └── ibm_preprocessor.joblib            # Preprocessing pipeline for IBM Telco
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

---

## 6. Streamlit Web Application

The interactive web dashboard is located in [`app/`](app/) and can be launched locally.

### Application Scope & Loaded Models
- **Supported Dataset:** The application **strictly supports Dataset 1 (Cell2Cell)**. It loads the 63-feature telemetry schema and processes customer attributes through `models/production_cell2cell/`.
- **Inference Models Loaded:**
  - `best_xgboost.joblib` (Primary inference model)
  - `final_preprocessor.joblib` (Numerical median imputation + standard scaling + binary passthrough)
  - `calibrated_xgboost_isotonic.joblib` (Calibrated probability mode)
  - `calibrated_xgboost_sigmoid.joblib` (Platt scaling mode)
- **Prediction Modes:**
  1. *Decision Mode:* Uses the cost-sensitive threshold $\tau^* = 0.4631$ to prioritize churn recall.
  2. *Calibrated Mode:* Outputs empirical isotonic-calibrated probabilities.
  3. *Prior-Adjusted Mode:* Mathematically maps probabilities to natural industry prevalence ($1.96\%$) using Bayes' rule.

*(Note: The IBM Telco models are currently standalone research pipelines located in `models/production_ibm_telco/` and `src/ibm_telco/` and are not currently rendered in the Streamlit UI).*

---

## 7. Installation & Run Instructions

### 7.1 Clone the Repository
```bash
git clone https://github.com/gumbhirdhairya-rgb/telecorm-churn.git
cd telecorm-churn
```

### 7.2 Create & Activate Virtual Environment

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

### 7.3 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 7.4 Launch the Streamlit Web Application
```bash
python -m streamlit run app/app.py
```
Open **`http://localhost:8501`** in your browser.

### 7.5 Executing Pipeline Scripts
To reproduce any cross-validation experiments or pipelines independently:

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

## 8. Academic Integrity & Limitations

1. **Dataset Dependence:** Machine learning metrics are inherently bounded by the data-generating process. A model achieving 81% accuracy on IBM Telco cannot be assumed to achieve 81% on Cell2Cell; conversely, models optimized for Cell2Cell cannot be transferred directly to IBM Telco without retraining.
2. **Attribution vs. Causation:** SHAP values identify statistical contributions of features toward the model's log-odds output. They do **not** represent causal mechanisms. For example, high equipment age (`eqpdays`) correlates with churn because customers near contract expiry delay upgrading; forcing customers to keep an older device would not reduce churn.
3. **Simulation Assumptions:** Financial ROI and retention savings are based on explicit simulated parameters ($V_{\text{LTV}} = \$250$, intervention uplift $r = 25\%$, intervention cost $C = \$30$). Real-world enterprise results depend on marketing campaign execution, competitive behavior, and offer acceptance rates.
4. **Generalization Constraints:** Cross-validation and holdout validation prevent data leakage within each dataset partition, but they do not guarantee generalization to future time periods or international carrier markets.

---

## 9. License & Citation

Distributed under the MIT License. Developed as a comparative machine learning research project for telecommunications customer retention analytics.
