# Model Artifact Audit & Repository Organization Plan

**Date:** November 4, 2026  
**Auditor:** Antigravity AI Assistant  
**Status:** Audit Complete — Awaiting Approval Before Moving Files  

---

## 1. Executive Summary

This audit catalogs all 19 machine learning model and preprocessor artifacts currently present in the repository (`models/`). It identifies:
1. Which artifacts are **actively loaded and executed by the Streamlit application**.
2. Which artifacts belong to the **Cohort 2 (IBM Telco) high-accuracy pipeline**.
3. Which artifacts are **experimental benchmarks and intermediate milestones**.
4. How each model is paired with its corresponding preprocessor and calibration models.
5. A proposed, non-destructive reorganization plan to keep the repository clean and maintainable.

> **CRITICAL GUARANTEE:** No files have been deleted, moved, or overwritten. The existing Streamlit production application continues to point to its verified paths.

---

## 2. Complete Inventory of All Model Artifacts

| Artifact Name | Path | Size | Role / Description |
| :--- | :--- | :---: | :--- |
| `best_xgboost.joblib` | `models/` | 280.6 KB | **Production Champion Model** (Cell2Cell, weighted threshold 0.4631) |
| `final_preprocessor.joblib` | `models/` | 5.9 KB | **Production Preprocessor** (63 features: 34 numerical + 29 binary) |
| `calibrated_xgboost_isotonic.joblib` | `models/` | 1.39 MB | **Production Calibration Model** (Isotonic regression) |
| `calibrated_xgboost_sigmoid.joblib` | `models/` | 1.38 MB | **Production Calibration Model** (Platt scaling) |
| `unweighted_xgboost.joblib` | `models/` | 280.9 KB | **Baseline Unweighted Model** (72.05% raw accuracy benchmark) |
| `best_random_forest.joblib` | `models/` | 23.8 MB | **Benchmark Model** (Step 4/5 comparison, 100 trees, depth 12) |
| `best_decision_tree.joblib` | `models/` | 24.8 KB | **Benchmark Model** (Step 4/5 comparison, depth 6) |
| `best_logistic_regression.joblib` | `models/` | 2.3 KB | **Baseline Model** (Step 4/5 comparison, linear benchmark) |
| `preprocessor.joblib` | `models/` | 5.8 KB | **Intermediate Artifact** (Step 2 raw preprocessor before 5 domain features) |
| `best_catboost_classifier.joblib` | `models/model_improvement/catboost/` | 907.8 KB | **Improvement Model** (Step 1 CatBoost, 800 trees, 63.57% acc, 0.6821 AUC) |
| `catboost_preprocessor.joblib` | `models/model_improvement/catboost/` | 5.9 KB | **In-Fold Preprocessor** for CatBoost experiment |
| `best_lightgbm_classifier.joblib` | `models/model_improvement/lightgbm/` | 1.02 MB | **Improvement Model** (Step 2 LightGBM, 64.16% acc, 0.6833 AUC) |
| `lightgbm_preprocessor.joblib` | `models/model_improvement/lightgbm/` | 5.9 KB | **In-Fold Preprocessor** for LightGBM experiment |
| `best_xgboost_reevaluated.joblib` | `models/model_improvement/xgboost/` | 267.3 KB | **Improvement Model** (Step 3 Fair In-Fold XGBoost, 0.6800 AUC) |
| `xgboost_preprocessor.joblib` | `models/model_improvement/xgboost/` | 5.9 KB | **In-Fold Preprocessor** for Fair XGBoost experiment |
| `best_ibm_xgboost.joblib` | `models/ibm_telco/` | 248.7 KB | **Cohort 2 Model** (IBM Telco XGBoost, 80.55% test accuracy) |
| `best_ibm_lightgbm.joblib` | `models/ibm_telco/` | 512.9 KB | **Cohort 2 Model** (IBM Telco LightGBM, 80.20% test accuracy) |
| `best_ibm_catboost.joblib` | `models/ibm_telco/` | 187.9 KB | **Cohort 2 Model** (IBM Telco CatBoost, 80.20% test accuracy) |
| `ibm_preprocessor.joblib` | `models/ibm_telco/` | 8.8 KB | **Cohort 2 Preprocessor** (Imputer + Scaler + OHE for IBM Telco) |

---

## 3. Usage Classification

### Group A: Actively Loaded by the Streamlit Application (DO NOT TOUCH)
These files are hard-coded in `app/config.py` and loaded dynamically into Streamlit cache via `app/utils/model_loader.py`:
1. `models/best_xgboost.joblib` &rarr; Used by `page_individual.py`, `page_bulk.py`, `page_strategy.py`, `page_shap.py`.
2. `models/final_preprocessor.joblib` &rarr; Used to transform raw user inputs into the exact 63-feature matrix.
3. `models/calibrated_xgboost_isotonic.joblib` &rarr; Used when the user selects "Calibrated Mode".
4. `models/calibrated_xgboost_sigmoid.joblib` &rarr; Used as alternative calibration mode.
5. `models/unweighted_xgboost.joblib` &rarr; Loaded as optional reference by `model_loader.py`.

### Group B: High-Accuracy Cohort 2 Models (IBM Telco - 81% Accuracy)
These files power the newly added contractual subscription cohort:
1. `models/ibm_telco/best_ibm_xgboost.joblib`
2. `models/ibm_telco/best_ibm_lightgbm.joblib`
3. `models/ibm_telco/best_ibm_catboost.joblib`
4. `models/ibm_telco/ibm_preprocessor.joblib`

### Group C: Step 1–3 Model Improvement Experiments (Cell2Cell In-Fold Benchmarks)
These files store the results of our leakage-free 5-Fold cross-validation experiments comparing CatBoost, LightGBM, and XGBoost:
1. `models/model_improvement/catboost/best_catboost_classifier.joblib` + preprocessor
2. `models/model_improvement/lightgbm/best_lightgbm_classifier.joblib` + preprocessor
3. `models/model_improvement/xgboost/best_xgboost_reevaluated.joblib` + preprocessor

### Group D: Static Benchmark Artifacts & Historical Milestones
These models were trained in Notebooks 04 & 05 (`notebooks/04_model_training.ipynb`) to create the baseline comparison table in the report figures:
1. `models/best_random_forest.joblib` (23.8 MB) — Static baseline for Step 5 evaluation figures.
2. `models/best_decision_tree.joblib` (24.8 KB) — Static baseline for Step 5 evaluation figures.
3. `models/best_logistic_regression.joblib` (2.3 KB) — Static baseline for Step 5 evaluation figures.
4. `models/preprocessor.joblib` (5.8 KB) — Precursor to `final_preprocessor.joblib` created in Step 2 before feature engineering was added.

---

## 4. Model & Preprocessor Pairing Verification

Every prediction pipeline requires a strict match between the trained estimator and the feature space produced by its preprocessor:

| Prediction Pipeline | Estimator Artifact | Required Preprocessor | Verification Status |
| :--- | :--- | :--- | :--- |
| **Cell2Cell Production UI** | `best_xgboost.joblib` | `final_preprocessor.joblib` | **100% Verified** (63 features, correct order) |
| **Cell2Cell Isotonic Mode** | `calibrated_xgboost_isotonic.joblib` | `final_preprocessor.joblib` | **100% Verified** |
| **Cell2Cell Sigmoid Mode** | `calibrated_xgboost_sigmoid.joblib` | `final_preprocessor.joblib` | **100% Verified** |
| **Cell2Cell CatBoost Experiment** | `best_catboost_classifier.joblib` | `catboost_preprocessor.joblib` | **100% Verified** (63 features in-fold) |
| **Cell2Cell LightGBM Experiment** | `best_lightgbm_classifier.joblib` | `lightgbm_preprocessor.joblib` | **100% Verified** (63 features in-fold) |
| **Cell2Cell XGBoost Fair Re-eval** | `best_xgboost_reevaluated.joblib` | `xgboost_preprocessor.joblib` | **100% Verified** (63 features in-fold) |
| **IBM Telco Pipeline** | `best_ibm_*.joblib` | `ibm_preprocessor.joblib` | **100% Verified** (29 OHE features) |

---

## 5. Redundancy & Observations

1. **`models/preprocessor.joblib` vs `models/final_preprocessor.joblib`:**
   * `preprocessor.joblib` was created in Step 2 on the raw 67 columns (without the 5 domain features).
   * `final_preprocessor.joblib` was created in Step 4 on the selected 63 columns (including the 5 domain features).
   * *Recommendation:* Keep `preprocessor.joblib` archived in an `experimental_baselines/` folder so it does not cause confusion with `final_preprocessor.joblib`.
2. **`models/best_random_forest.joblib` Size (23.8 MB):**
   * This is by far the largest file in the repo. It is not loaded by the web app, but it is referenced by `notebooks/05_model_evaluation.ipynb`. Keeping it in a dedicated benchmark subfolder will clean up the root `models/` directory.

---

## 6. Proposed Clean Repository Structure

To make your GitHub repository instantly understandable to professors, reviewers, and interviewers, here is the proposed non-destructive reorganization plan:

```
models/
├── production_cell2cell/                  # [Cohort 1: Production Web App Models]
│   ├── best_xgboost.joblib                # Primary Champion Model
│   ├── final_preprocessor.joblib          # Production Preprocessor (63 features)
│   ├── calibrated_xgboost_isotonic.joblib # Isotonic Calibration
│   ├── calibrated_xgboost_sigmoid.joblib  # Sigmoid Calibration
│   └── unweighted_xgboost.joblib          # 72.05% Accuracy Reference Model
│
├── production_ibm_telco/                  # [Cohort 2: High-Accuracy Models (81% Acc)]
│   ├── best_ibm_xgboost.joblib            # 80.55% Test Accuracy
│   ├── best_ibm_lightgbm.joblib           # 80.20% Test Accuracy
│   ├── best_ibm_catboost.joblib           # 80.20% Test Accuracy
│   └── ibm_preprocessor.joblib            # Preprocessor for IBM Telco
│
├── model_improvement/                     # [Cohort 1: 5-Fold In-Fold Experiments]
│   ├── catboost/
│   │   ├── best_catboost_classifier.joblib
│   │   └── catboost_preprocessor.joblib
│   ├── lightgbm/
│   │   ├── best_lightgbm_classifier.joblib
│   │   └── lightgbm_preprocessor.joblib
│   └── xgboost/
│       ├── best_xgboost_reevaluated.joblib
│       └── xgboost_preprocessor.joblib
│
└── experimental_baselines/                # [Cohort 1: Step 4/5 Benchmark Baselines]
    ├── best_random_forest.joblib          # 23.8 MB Benchmark Model
    ├── best_decision_tree.joblib          # Benchmark Model
    ├── best_logistic_regression.joblib    # Baseline Linear Model
    └── preprocessor.joblib                # Step 2 Raw Preprocessor
```

---

## 7. Migration Safety Plan (If Approved)

If you approve this proposed structure:
1. **Zero-Downtime App Compatibility:**
   * We will update `app/config.py` to point to `models/production_cell2cell/`.
   * Alternatively, we can use symlinks or keep the 5 production files at the root of `models/` while moving the benchmarks into subfolders.
2. **Notebook Compatibility:**
   * Notebooks 04 & 05 paths will be updated to reflect the new subfolder locations.
3. **Git History Preservation:**
   * File movements will be done via `git mv` so full commit history is preserved on GitHub.

---

**Next Action:** Awaiting your review and approval. No files will be moved until you confirm.
