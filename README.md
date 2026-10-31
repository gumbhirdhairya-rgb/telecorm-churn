# 📶 Telecom Customer Churn Prediction & Enterprise Retention Platform

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20Champion-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-TreeExplainer-green.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit%20v1.65-red.svg)](https://streamlit.io/)
[![Evaluation](https://img.shields.io/badge/Academic-UCS321%20Aligned-purple.svg)]()

> **Problem Statement 9:** An end-to-end Machine Learning, Explainability, and Decision Science solution for telecommunications customer churn prediction on the **Cell2Cell Wireless Dataset (71,047 subscribers)**.

---

## 🌟 Executive Summary & Key Results

- **Champion Architecture:** Tuned Cost-Sensitive XGBoost Classifier (`max_depth=5, lr=0.1, n_estimators=100, scale_pos_weight=2.447`).
- **Test Performance (14,210 Holdout Records):**
  - **ROC-AUC:** **`0.6786`** (Matches cross-validation within 0.002, zero leakage/overfitting).
  - **PR-AUC:** **`0.4586`** ($+6.2\%$ over baseline Logistic Regression).
  - **Decision Boundary Optimization:** Tuned cost-sensitive threshold $\tau^* = \mathbf{0.4631}$ elevates Recall from $63.73\% \to \mathbf{72.78\%}$ (F1 = **$0.5019$**), saving **$+373$ additional true churners** over the symmetric $\tau = 0.50$ cutoff.
- **Empirical Feature Engineering:** Out of 63 retained features, **4 of our 5 engineered features rank in the Global Top 10** by mean absolute SHAP value (`cost_per_mou` #5, `network_friction_rate` #7, `handset_age_ratio` #9, `overage_burden` #10).
- **Decision Science & ROI:** Proved that a naive "blind blanket" retention campaign destroys **$-\$168,675$ (-$39.6\%$ ROI)** due to False Positive waste, whereas our machine-learning-driven **Differentiated Tiered Strategy** delivers **$+\$59,199$ in net economic profit** with a **$+45.1\%$ ROI** (saving $761.7$ high-value churners).
- **Interactive Streamlit Platform:** Modular multi-page application with real-time individual customer scoring, dynamic TreeExplainer SHAP waterfall charts, high-throughput CSV bulk scoring, and downloadable priority worklists.

---

## 📁 Repository Structure

```
├── app/                                # Streamlit Enterprise Web Application
│   ├── app.py                          # Main entrypoint & navigation router
│   ├── config.py                       # Global paths, thresholds, and theme styling
│   ├── utils/                          # Core inference & decision science utilities
│   │   ├── model_loader.py             # Cached model & preprocessor loading (@st.cache_resource)
│   │   ├── preprocessor.py             # Domain feature engineering & 63-feature pipeline
│   │   ├── explainer.py                # Cached TreeExplainer & dynamic SHAP waterfall generator
│   │   └── business_logic.py           # 3-tier risk logic, Bayes prior correction, retention rules
│   └── views/                          # Modular frontend page components
│       ├── page_overview.py            # Executive KPI dashboard & ML pipeline flowchart
│       ├── page_individual.py          # Primary Mode 1: Individual Customer Scoring & SHAP
│       ├── page_bulk.py                # Primary Mode 2: Bulk CSV Ingestion & Batch Intelligence
│       ├── page_performance.py         # Test benchmark curves, ROC/PR, confusion matrices
│       ├── page_shap.py                # Global SHAP beeswarm, dependence plots, thresholds
│       ├── page_segmentation.py        # 3-tier risk profiles & unscaled empirical metrics
│       └── page_strategy.py            # Business strategy, policy matrices, ROI simulator
├── data/
│   ├── raw/                            # Original Cell2Cell dataset (71,047 rows x 70 cols)
│   ├── processed/                      # Leakage-free train/test final matrices & feature schemas
│   │   ├── selected_features.json      # 63 final features metadata & order
│   │   ├── X_train_final.csv           # 56,837 rows x 63 features
│   │   ├── X_test_final.csv            # 14,210 rows x 63 features
│   │   └── y_test_final.csv            # 14,210 test ground truth labels
│   └── sample_raw_customers.csv        # 50-row raw sample batch for testing ingestion
├── models/                             # Serialized Scikit-Learn & XGBoost Artifacts
│   ├── best_xgboost.joblib             # Champion Tuned Cost-Sensitive XGBoost (280 KB)
│   ├── final_preprocessor.joblib       # Fitted ColumnTransformer (34 num + 29 bin) (5.8 KB)
│   ├── calibrated_xgboost_isotonic.joblib  # Isotonic Calibrated Classifier (1.3 MB)
│   ├── calibrated_xgboost_sigmoid.joblib   # Sigmoid Platt Calibrated Classifier (1.3 MB)
│   ├── best_random_forest.joblib       # Tuned Random Forest Classifier (23.8 MB)
│   ├── best_decision_tree.joblib       # Tuned Decision Tree Classifier (24 KB)
│   └── best_logistic_regression.joblib # Baseline L2 Logistic Regression (2.2 KB)
├── notebooks/                          # Interactive Jupyter Notebooks (Steps 1–8)
│   ├── 01_dataset_understanding_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   ├── 06_model_explainability.ipynb
│   └── 07_business_insights.ipynb
├── reports/                            # Exhaustive Academic & Strategic Reports
│   ├── 01_dataset_understanding_report.md
│   ├── 06_calibration_audit.md
│   ├── 07_shap_analysis.md
│   ├── 08_business_insights.md
│   ├── 09_validation_report.md
│   ├── final_test_evaluation.csv
│   ├── cv_model_comparison.csv
│   ├── shap_feature_importance.csv
│   ├── retention_risk_segments.csv
│   ├── retention_policy_comparison.csv
│   └── figures/                        # High-Resolution Publication Figures (300 DPI)
├── src/                                # Reusable Python Scripts & Execution Pipelines
│   ├── fe_utils.py                     # Verified 5-feature domain engineering functions
│   ├── run_shap_analysis.py            # Global SHAP & local waterfall generator
│   ├── run_business_insights.py        # Segmentation & economic simulation engine
│   └── run_step9c_validation.py        # Comprehensive 9-task automated test suite
├── requirements.txt                    # Project dependency manifest
├── .gitignore                          # Standard git exclusion rules
└── README.md                           # Master project documentation
```

---

## 🚀 Quickstart: Launching the Streamlit Platform

### 1. Environment Setup
```bash
git clone https://github.com/<your-username>/telecom-customer-churn-ai.git
cd telecom-customer-churn-ai

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Application
```bash
streamlit run app/app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 📊 Benchmark Model Performance (14,210 Test Customers)

| Model Name | Threshold ($\tau$) | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1-Score | True Positives ($TP$) | Saved Churners vs 0.50 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 0.5000 | 0.6255 | 0.3968 | 61.34% | 37.10% | 57.13% | 0.4501 | 2,355 | — |
| **Decision Tree** | 0.5000 | 0.6369 | 0.4060 | 61.82% | 38.01% | 62.91% | 0.4740 | 2,593 | — |
| **Random Forest** | 0.5000 | 0.6706 | 0.4471 | 63.45% | 39.50% | 60.53% | 0.4782 | 2,495 | — |
| **XGBoost (Champion)** | 0.5000 | **0.6786** | **0.4586** | 62.23% | 40.42% | 63.73% | 0.4947 | 2,627 | Baseline |
| **XGBoost (Tuned Boundary)** | **0.4631** | **0.6786** | **0.4586** | 58.10% | 38.30% | **72.78%** | **0.5019** | **3,000** | **+373 Churners** |

---

## 🧠 SHAP Explainability & Top 10 Predictors

Computed via `shap.TreeExplainer` in polynomial time $\mathcal{O}(TLD^2)$:

```
Top 10 Global Features by Mean Absolute SHAP Value:
Rank   Feature                  Mean |SHAP|   Type          Business Interpretation
-------------------------------------------------------------------------------------------------------
 1     eqpdays                   0.2284       Continuous    Equipment age; surges at 365d (contract end).
 2     months                    0.1553       Continuous    Tenure; high vulnerability for <14 months.
 3     changem                   0.1449       Continuous    Usage velocity; drops below -20% signal churn.
 4     uniqsubs                  0.0874       Continuous    Account line complexity and friction.
 5     cost_per_mou [ENGINEERED] 0.0851       Engineered    Unit cost ($/min); beats raw revenue by 2.4x.
 6     age1                      0.0776       Continuous    Subscriber age; younger users switch faster.
 7     network_friction_rate     0.0694       Engineered    (dropvce + blckvce) / (mou + 1). Quality failure.
 8     recchrge                  0.0656       Continuous    Base recurring charge level.
 9     handset_age_ratio         0.0602       Engineered    eqpdays / (months * 30.4375 + 1). Relative decay.
 10    overage_burden            0.0560       Engineered    overage / (recchrge + 1). Bill shock indicator.
```

---

## 💼 Retention Economics & Policy Comparison

Simulated on 14,210 test customers under explicit assumptions ($V_{\text{LTV}}=\$250$, Uplift $r=25\%$, Cost $C=\$30$):

| Retention Policy | Targeted Base | True Churners ($TP$) | Churners Saved | Total Cost | Gross LTV Benefit | Net Economic Savings | ROI (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Do Nothing** | 0 | 0 | 0.0 | $\$0$ | $\$0$ | **$\$0$** | 0.0% |
| **2. Blind Blanket Campaign** | 14,210 | 4,122 | 1,030.5 | $\$426,300$ | $\$257,625$ | **-$168,675** | **-39.6%** |
| **3. High Risk Tier ($\ge 0.60$)**| 2,814 | 1,365 | 341.2 | $\$84,420$ | $\$85,312$ | **+$892** | +1.1% |
| **4. Max-Profit Cutoff ($\ge 0.67$)**| 1,174 | 680 | 170.0 | $\$35,220$ | $\$42,500$ | **+$7,280** | **+20.7%** |
| **5. Differentiated Tiered Strategy**| **10,998** | **3,713** | **761.7** | **$\$131,226$** | **$\$190,425$** | **+$59,199** | **+45.1%** |

*Key Lesson:* Aligning intervention cost with risk density (High: $\$35$ hardware voucher; Med: $\$4$ digital plan right-sizing; Low: $\$0$) prevents False Positive budget waste and maximizes preserved enterprise value.

---

## 🎓 Academic Viva Voce Quick Defense

1. **Why use cost-sensitive learning over SMOTE?**  
   *Answer:* The dataset features 29 binary flags. Synthetic oversampling methods like SMOTE interpolate across continuous space, generating ungrounded fractional values for discrete binary indicators. Cost-sensitive loss weighting (`scale_pos_weight=2.447`) adjusts gradient penalties without corrupting the natural feature geometry.
2. **Why does TreeExplainer compute log-odds margins instead of probabilities?**  
   *Answer:* Gradient boosted trees add leaf outputs in log-odds space ($\ln(p / (1 - p))$). Shapley additivity requires that individual feature contributions sum to the total margin: $\sum \phi_j(x) = f(x) - E[f(X)]$. The final calibrated probability is recovered via logistic sigmoid: $\hat{p} = 1 / (1 + e^{-(E[f] + \sum \phi_j)})$.
3. **What is the difference between SHAP attribution and causal churn intervention?**  
   *Answer:* SHAP quantifies observational statistical attribution, not real-world causation. For example, high equipment age (`eqpdays`) correlates with churn because customers with expiring contracts delay purchasing new handsets. Artificially forcing a customer to keep an older phone would accelerate churn. An effective intervention requires addressing the root cause—offering a subsidized hardware upgrade to secure a 12-month contract extension.

---

## 📄 License & Attribution

Developed as part of the UCS321 Machine Learning Curriculum. All models, code, and reports are distributed under the MIT License.
