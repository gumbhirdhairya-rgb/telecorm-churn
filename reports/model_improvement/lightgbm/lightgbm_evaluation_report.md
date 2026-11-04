# Step 2: LightGBM Classifier Evaluation Report

## 1. Overview & Experimental Setup
- **Model**: LightGBM Classifier (Leaf-Wise Histogram Gradient Boosting)
- **Dataset**: Cell2Cell Telecom Churn Dataset (56,837 training samples, 80% split)
- **Holdout Test Set**: 14,210 samples (20% split) - **Completely Untouched / Locked**
- **Validation Method**: Stratified 5-Fold Cross-Validation (`random_state=42`) with identical fold indices
- **Data Leakage Prevention**: Preprocessing (imputation & scaling) fitted **strictly inside each cross-validation fold**
- **Evaluation Threshold**: 0.50 (Initial Standard Decision Threshold)

## 2. LightGBM Hyperparameter Tuning & Cross-Validation Results

| Model Variant | ROC-AUC (Mean ± Std) | PR-AUC (Mean ± Std) | Accuracy | Balanced Accuracy | Precision | Recall | F1-Score | Total CV Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Baseline)** | 0.6823 ± 0.0061 | 0.4662 ± 0.0088 | 0.6362 | 0.6281 | 0.4137 | 0.6086 | 0.4926 | 7.9s |
| **LightGBM (Tuned/Regularized)** | 0.6830 ± 0.0067 | 0.4670 ± 0.0099 | 0.6416 | 0.6277 | 0.4173 | 0.5946 | 0.4904 | 13.6s |
| **LightGBM (Depth-Constrained)** | 0.6833 ± 0.0069 | 0.4656 ± 0.0105 | 0.6283 | 0.6291 | 0.4089 | 0.6310 | 0.4962 | 6.9s |

## 3. Consolidated Gradient Boosting Trio Comparison: XGBoost vs CatBoost vs LightGBM

| Metric | XGBoost (Baseline) | CatBoost (Best) | LightGBM (Best) | Best Performer |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC (Overall Ranking)** | 0.6803 | 0.6821 | **0.6833** | LightGBM |
| **PR-AUC (Churn Curve)** | 0.4608 | 0.4660 | **0.4656** | CatBoost |
| **Accuracy** | 0.6223 | 0.6357 | **0.6283** | CatBoost |
| **Balanced Accuracy** | 0.6265 | 0.6295 | **0.6291** | CatBoost |
| **Precision** | 0.4066 | 0.4139 | **0.4089** | CatBoost |
| **Recall (Churn Caught)** | **0.6364** | 0.6148 | 0.6310 | XGBoost |
| **F1-Score** | **0.4961** | 0.4947 | 0.4962 | LightGBM |
| **Evaluation Folds** | Identical 5 Folds | Identical 5 Folds | Identical 5 Folds | **100% Consistent** |
| **Preprocessing Leakage** | Global Train Scaled | In-Fold Fitted | In-Fold Fitted | **Zero Leakage** |

## 4. Consolidated Out-of-Fold (OOF) Confusion Matrix (Threshold = 0.50)

**Best LightGBM (LightGBM (Depth-Constrained))** on 56,837 Out-of-Fold Training Predictions:

| Actual \ Predicted | Predicted Non-Churn (0) | Predicted Churn (1) |
| :--- | :---: | :---: |
| **Actual Non-Churn (0)** | **25304** (True Negatives) | **15046** (False Positives) |
| **Actual Churn (1)** | **6083** (False Negatives) | **10404** (True Positives) |
