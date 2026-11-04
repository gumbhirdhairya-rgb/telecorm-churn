# Step 1: CatBoost Classifier Evaluation Report

## 1. Overview & Experimental Setup
- **Model**: CatBoost Classifier (Symmetric Oblivious Trees)
- **Dataset**: Cell2Cell Telecom Churn Dataset (56,838 training samples, 80% split)
- **Holdout Test Set**: 14,210 samples (20% split) - **Completely Untouched** to prevent data snooping
- **Validation Method**: 5-Fold Stratified Cross-Validation (`random_state=42`)
- **Data Leakage Prevention**: Imputation and feature scaling were fitted **strictly inside each cross-validation fold**
- **Evaluation Threshold**: 0.50 (Initial Standard Decision Threshold)

## 2. CatBoost 5-Fold Cross-Validation Results

| Model | ROC-AUC (Mean ± Std) | PR-AUC (Mean ± Std) | Accuracy | Balanced Accuracy | Precision | Recall | F1-Score | Total CV Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost (Baseline)** | 0.6815 ± 0.0083 | 0.4648 ± 0.0102 | 0.6290 | 0.6284 | 0.4091 | 0.6270 | 0.4951 | 22.1s |
| **CatBoost (Tuned/Regularized)** | 0.6821 ± 0.0059 | 0.4660 ± 0.0086 | 0.6357 | 0.6295 | 0.4139 | 0.6148 | 0.4947 | 35.9s |

## 3. Side-by-Side Comparison: CatBoost vs XGBoost Baseline

| Metric | XGBoost (Baseline) | CatBoost (Baseline) | CatBoost (Tuned) | Delta (Best CatBoost vs XGBoost) |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.6803 | 0.6815 | 0.6821 | +0.0018 |
| **PR-AUC** | 0.4608 | 0.4648 | 0.4660 | +0.0053 |
| **F1-Score** | 0.4961 | 0.4951 | 0.4947 | -0.0014 |
| **Recall (Churn Caught)** | 0.6364 | 0.6270 | 0.6148 | -0.0215 |
| **Precision** | 0.4066 | 0.4091 | 0.4139 | +0.0073 |
| **Accuracy** | ~0.6223 | 0.6290 | 0.6357 | +0.0134 |
| **Balanced Accuracy** | ~0.6265 | 0.6284 | 0.6295 | +0.0030 |

## 4. Consolidated Out-of-Fold (OOF) Confusion Matrix (Threshold = 0.50)

**Best CatBoost (CatBoost (Tuned/Regularized))** on 56,838 Out-of-Fold Training Predictions:

| Actual \ Predicted | Predicted Non-Churn (0) | Predicted Churn (1) |
| :--- | :---: | :---: |
| **Actual Non-Churn (0)** | **25993** (True Negatives) | **14357** (False Positives) |
| **Actual Churn (1)** | **6350** (False Negatives) | **10137** (True Positives) |
