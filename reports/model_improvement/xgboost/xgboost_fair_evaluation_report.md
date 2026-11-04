# Step 3: Fair XGBoost Re-evaluation Report

## 1. Overview & Methodology Alignment
In the original benchmark (`cv_model_comparison.csv`), XGBoost was evaluated on `X_train_final.csv`, which had already been standardized using a global preprocessor fitted across all 56,838 training samples.

In this re-evaluation:
- **Identical CV Splits**: Exact same 5-Fold Stratified split (`random_state=42`) used for CatBoost and LightGBM.
- **Zero Data Leakage**: Median imputation and standard scaling are fitted **strictly inside each training fold**, ensuring no validation fold data leaks into preprocessing.
- **Feature Consistency**: Exact same 63 selected features from `selected_features.json`.
- **Holdout Test Set Protection**: Final 14,210-customer holdout set remained **completely untouched**.

## 2. XGBoost Re-evaluation Results (In-Fold Preprocessed)

| Model Variant | ROC-AUC (Mean ± Std) | PR-AUC (Mean ± Std) | Accuracy | Balanced Accuracy | Precision | Recall | F1-Score | Total CV Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Champion Architecture)** | 0.6787 ± 0.0062 | 0.4569 ± 0.0077 | 0.6222 | 0.6261 | 0.4039 | 0.6354 | 0.4938 | 2.6s |
| **XGBoost (Regularized/Tuned)** | 0.6800 ± 0.0068 | 0.4590 ± 0.0081 | 0.6196 | 0.6278 | 0.4031 | 0.6474 | 0.4968 | 3.0s |

## 3. Consolidated Gradient Boosting Trio Comparison (100% Apples-to-Apples)

| Metric | XGBoost (Original Baseline) | XGBoost (Re-evaluated In-Fold) | CatBoost (Best) | LightGBM (Tuned/Reg) | LightGBM (Depth-Constrained) | Best Performer |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ROC-AUC (Overall Ranking)** | 0.6803 | 0.6800 | 0.6821 | 0.6830 | **0.6833** | **LightGBM (Depth-Constrained)** 🟢 |
| **PR-AUC (Churn Curve)** | 0.4608 | 0.4590 | 0.4660 | **0.4670** | 0.4656 | **LightGBM (Tuned/Reg)** 🟢 |
| **Raw Accuracy** | 0.6223 | 0.6196 | 0.6357 | **0.6416** | 0.6283 | **LightGBM (Tuned/Reg - 64.16%)** 🟢 |
| **Balanced Accuracy** | 0.6265 | 0.6278 | **0.6295** | 0.6277 | 0.6291 | **CatBoost (62.95%)** 🟢 |
| **Precision** | 0.4066 | 0.4031 | 0.4139 | **0.4173** | 0.4089 | **LightGBM (Tuned/Reg - 41.73%)** 🟢 |
| **Recall (Churn Caught)** | 0.6364 | **0.6474** | 0.6148 | 0.5946 | 0.6310 | **XGBoost (Highest Recall)** 🟡 |
| **F1-Score** | 0.4961 | 0.4968 | 0.4947 | 0.4904 | **0.4962** | **LightGBM (Depth-Constrained)** 🟢 |
| **Evaluation Folds** | Identical 5 Folds | Identical 5 Folds | Identical 5 Folds | Identical 5 Folds | Identical 5 Folds | **100% Identical** |
| **In-Fold Preprocessing** | No (Global Scaled) | **Yes (Leakage-Free)** | **Yes (Leakage-Free)** | **Yes (Leakage-Free)** | **Yes (Leakage-Free)** | **Methodologically Pure** |

## 4. Consolidated Out-of-Fold (OOF) Confusion Matrix (Threshold = 0.50)

**Re-evaluated XGBoost (XGBoost (Regularized/Tuned))** on 56,837 Out-of-Fold Training Predictions:

| Actual \ Predicted | Predicted Non-Churn (0) | Predicted Churn (1) |
| :--- | :---: | :---: |
| **Actual Non-Churn (0)** | **24541** (True Negatives) | **15809** (False Positives) |
| **Actual Churn (1)** | **5814** (False Negatives) | **10673** (True Positives) |
