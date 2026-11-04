"""
Step 1: CatBoost Classifier - Leakage-Free Stratified 5-Fold Cross-Validation
Telecom Customer Churn AI Platform

Trains and evaluates CatBoost Classifier using exact 63 features, with preprocessing
fitted strictly INSIDE each cross-validation fold to guarantee zero data leakage.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import time
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    average_precision_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from catboost import CatBoostClassifier

from src.fe_utils import engineer_telecom_features

def run_catboost_experiment():
    print("=" * 75)
    print("Step 1: CatBoost Classifier Evaluation (Leakage-Free 5-Fold Stratified CV)")
    print("=" * 75)

    # 1. Output directories
    MODELS_DIR = "models/model_improvement/catboost"
    REPORTS_DIR = "reports/model_improvement/catboost"
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 2. Load raw Cell2Cell dataset and perform identical 80/20 stratified split
    print("\n[1/6] Loading Cell2Cell dataset and creating initial 80/20 train/holdout split...")
    df_raw = pd.read_csv("data/raw/cell2cell.csv")
    X_raw = df_raw.drop(columns=['customer', 'calibrat', 'churndep', 'churn'])
    y = df_raw['churn'].astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Training set size : {X_train_raw.shape[0]} rows (used for 5-Fold CV)")
    print(f"      Holdout test size : {X_test_raw.shape[0]} rows (UNTOUCHED / LOCKED)")

    # 3. Apply feature engineering on training set
    print("\n[2/6] Engineering the 5 domain features on training partition...")
    X_train_eng = engineer_telecom_features(X_train_raw)

    # 4. Load exact 63 selected features
    with open("data/processed/selected_features.json", "r") as f:
        selected_meta = json.load(f)
    selected_features = selected_meta['selected_features']
    print(f"      Selected features : {len(selected_features)} features (exact XGBoost alignment)")

    X_train_sel = X_train_eng[selected_features].copy()
    y_train = y_train.reset_index(drop=True)
    X_train_sel = X_train_sel.reset_index(drop=True)

    # Identify continuous vs binary columns
    bin_cols = [c for c in selected_features if X_train_sel[c].dropna().isin([0, 1]).all()]
    num_cols = [c for c in selected_features if c not in bin_cols]
    print(f"      Continuous features: {len(num_cols)} | Binary features: {len(bin_cols)}")

    # Class imbalance weight
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"      Imbalance Ratio (scale_pos_weight): {pos_weight:.3f}")

    # 5. Define CatBoost configurations
    configs = {
        'CatBoost (Baseline)': {
            'iterations': 500,
            'learning_rate': 0.08,
            'depth': 5,
            'l2_leaf_reg': 3.0,
            'scale_pos_weight': pos_weight,
            'random_seed': 42,
            'verbose': False,
            'thread_count': -1
        },
        'CatBoost (Tuned/Regularized)': {
            'iterations': 800,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 5.0,
            'scale_pos_weight': pos_weight,
            'random_seed': 42,
            'verbose': False,
            'thread_count': -1
        }
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    overall_results = []
    oof_predictions = pd.DataFrame({'true_churn': y_train})

    print("\n[3/6] Starting Leakage-Free 5-Fold Stratified Cross-Validation...")

    best_model_obj = None
    best_preprocessor_obj = None
    best_config_name = None
    best_roc_auc = -1.0

    for model_name, params in configs.items():
        print(f"\n---> Evaluating: {model_name}...")
        fold_metrics = []
        oof_probs = np.zeros(len(y_train))
        start_total_time = time.time()

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_sel, y_train), 1):
            fold_start_time = time.time()

            # Split fold
            X_tr, y_tr = X_train_sel.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train_sel.iloc[val_idx], y_train.iloc[val_idx]

            # Build and fit ColumnTransformer ONLY on fold training data (Zero Data Leakage)
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            bin_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('passthrough', 'passthrough')
            ])
            fold_preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, num_cols),
                    ('bin', bin_pipeline, bin_cols)
                ],
                remainder='drop'
            )

            # Fit strictly on fold train
            fold_preprocessor.fit(X_tr)
            feature_order_out = num_cols + bin_cols

            X_tr_trans = pd.DataFrame(
                fold_preprocessor.transform(X_tr),
                columns=feature_order_out,
                index=X_tr.index
            )
            X_va_trans = pd.DataFrame(
                fold_preprocessor.transform(X_va),
                columns=feature_order_out,
                index=X_va.index
            )

            # Train CatBoost
            model = CatBoostClassifier(**params)
            model.fit(X_tr_trans, y_tr)

            fold_time = time.time() - fold_start_time

            # Predictions at threshold 0.50
            va_probs = model.predict_proba(X_va_trans)[:, 1]
            va_preds = (va_probs >= 0.50).astype(int)
            oof_probs[val_idx] = va_probs

            # Compute metrics for this fold
            roc = roc_auc_score(y_va, va_probs)
            pr = average_precision_score(y_va, va_probs)
            acc = accuracy_score(y_va, va_preds)
            bal_acc = balanced_accuracy_score(y_va, va_preds)
            prec = precision_score(y_va, va_preds, zero_division=0)
            rec = recall_score(y_va, va_preds, zero_division=0)
            f1 = f1_score(y_va, va_preds, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_va, va_preds).ravel()

            fold_metrics.append({
                'Fold': fold,
                'ROC_AUC': roc,
                'PR_AUC': pr,
                'Accuracy': acc,
                'Balanced_Accuracy': bal_acc,
                'Precision': prec,
                'Recall': rec,
                'F1_Score': f1,
                'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
                'Fold_Time_Sec': fold_time
            })
            print(f"      Fold {fold}: ROC-AUC={roc:.4f} | PR-AUC={pr:.4f} | BalAcc={bal_acc:.4f} | Rec={rec:.4f} | Prec={prec:.4f} | F1={f1:.4f} ({fold_time:.1f}s)")

        total_time = time.time() - start_total_time
        metrics_df = pd.DataFrame(fold_metrics)

        # Consolidated OOF metrics
        oof_preds = (oof_probs >= 0.50).astype(int)
        oof_roc = roc_auc_score(y_train, oof_probs)
        oof_pr = average_precision_score(y_train, oof_probs)
        oof_acc = accuracy_score(y_train, oof_preds)
        oof_bal_acc = balanced_accuracy_score(y_train, oof_preds)
        oof_prec = precision_score(y_train, oof_preds, zero_division=0)
        oof_rec = recall_score(y_train, oof_preds, zero_division=0)
        oof_f1 = f1_score(y_train, oof_preds, zero_division=0)
        oof_tn, oof_fp, oof_fn, oof_tp = confusion_matrix(y_train, oof_preds).ravel()

        col_oof_name = f"{model_name.replace(' ', '_').lower()}_prob"
        oof_predictions[col_oof_name] = oof_probs

        summary_row = {
            'Model': model_name,
            'ROC_AUC_Mean': metrics_df['ROC_AUC'].mean(),
            'ROC_AUC_Std': metrics_df['ROC_AUC'].std(),
            'PR_AUC_Mean': metrics_df['PR_AUC'].mean(),
            'PR_AUC_Std': metrics_df['PR_AUC'].std(),
            'Accuracy_Mean': metrics_df['Accuracy'].mean(),
            'Accuracy_Std': metrics_df['Accuracy'].std(),
            'Balanced_Accuracy_Mean': metrics_df['Balanced_Accuracy'].mean(),
            'Balanced_Accuracy_Std': metrics_df['Balanced_Accuracy'].std(),
            'Precision_Mean': metrics_df['Precision'].mean(),
            'Precision_Std': metrics_df['Precision'].std(),
            'Recall_Mean': metrics_df['Recall'].mean(),
            'Recall_Std': metrics_df['Recall'].std(),
            'F1_Mean': metrics_df['F1_Score'].mean(),
            'F1_Std': metrics_df['F1_Score'].std(),
            'OOF_ROC_AUC': oof_roc,
            'OOF_PR_AUC': oof_pr,
            'OOF_Accuracy': oof_acc,
            'OOF_Balanced_Accuracy': oof_bal_acc,
            'OOF_Precision': oof_prec,
            'OOF_Recall': oof_rec,
            'OOF_F1': oof_f1,
            'OOF_TN': oof_tn, 'OOF_FP': oof_fp, 'OOF_FN': oof_fn, 'OOF_TP': oof_tp,
            'Total_CV_Time_Sec': total_time
        }
        overall_results.append(summary_row)

        print(f"\n      [+] {model_name} Summary:")
        print(f"          ROC-AUC : {summary_row['ROC_AUC_Mean']:.4f} (+/- {summary_row['ROC_AUC_Std']:.4f})")
        print(f"          PR-AUC  : {summary_row['PR_AUC_Mean']:.4f} (+/- {summary_row['PR_AUC_Std']:.4f})")
        print(f"          Accuracy: {summary_row['Accuracy_Mean']:.4f} | Balanced Acc: {summary_row['Balanced_Accuracy_Mean']:.4f}")
        print(f"          F1-Score: {summary_row['F1_Mean']:.4f} | Recall: {summary_row['Recall_Mean']:.4f} | Precision: {summary_row['Precision_Mean']:.4f}")
        print(f"          Total 5-Fold Training Time: {total_time:.1f}s")

        if summary_row['ROC_AUC_Mean'] > best_roc_auc:
            best_roc_auc = summary_row['ROC_AUC_Mean']
            best_config_name = model_name

    # 6. Fit Final Model on Full Training Set & Save
    print(f"\n[4/6] Fitting Best Candidate ({best_config_name}) on full 56,838 training rows...")
    final_num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    final_bin_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('passthrough', 'passthrough')
    ])
    full_preprocessor = ColumnTransformer(
        transformers=[
            ('num', final_num_pipe, num_cols),
            ('bin', final_bin_pipe, bin_cols)
        ],
        remainder='drop'
    )
    full_preprocessor.fit(X_train_sel)
    feature_order_out = num_cols + bin_cols

    X_train_full_trans = pd.DataFrame(
        full_preprocessor.transform(X_train_sel),
        columns=feature_order_out,
        index=X_train_sel.index
    )

    best_params = configs[best_config_name]
    final_catboost = CatBoostClassifier(**best_params)
    final_catboost.fit(X_train_full_trans, y_train)

    # Save artifacts in dedicated model_improvement directories
    print("\n[5/6] Saving CatBoost models and experiment artifacts...")
    catboost_model_path = os.path.join(MODELS_DIR, "best_catboost_classifier.joblib")
    catboost_preprocessor_path = os.path.join(MODELS_DIR, "catboost_preprocessor.joblib")
    joblib.dump(final_catboost, catboost_model_path)
    joblib.dump(full_preprocessor, catboost_preprocessor_path)

    # Save CV results and OOF predictions
    cv_results_df = pd.DataFrame(overall_results)
    cv_results_path = os.path.join(REPORTS_DIR, "catboost_cv_results.csv")
    oof_preds_path = os.path.join(REPORTS_DIR, "catboost_oof_predictions.csv")
    cv_results_df.to_csv(cv_results_path, index=False)
    oof_predictions.to_csv(oof_preds_path, index=False)

    print(f"      Saved model       : {catboost_model_path}")
    print(f"      Saved preprocessor: {catboost_preprocessor_path}")
    print(f"      Saved CV results  : {cv_results_path}")
    print(f"      Saved OOF preds   : {oof_preds_path}")

    # 7. Load baseline XGBoost metrics and build comparison report
    print("\n[6/6] Generating Side-by-Side Comparison against XGBoost Baseline...")
    xgb_row = None
    if os.path.exists("reports/cv_model_comparison.csv"):
        existing_cv = pd.read_csv("reports/cv_model_comparison.csv")
        first_col = existing_cv.columns[0]
        xgb_matches = existing_cv[existing_cv[first_col].str.contains("XGBoost", case=False, na=False)]
        if not xgb_matches.empty:
            xgb_row = xgb_matches.iloc[0].to_dict()

    # Markdown Report Generation
    report_md_path = os.path.join(REPORTS_DIR, "catboost_evaluation_report.md")
    with open(report_md_path, "w") as f:
        f.write("# Step 1: CatBoost Classifier Evaluation Report\n\n")
        f.write("## 1. Overview & Experimental Setup\n")
        f.write("- **Model**: CatBoost Classifier (Symmetric Oblivious Trees)\n")
        f.write("- **Dataset**: Cell2Cell Telecom Churn Dataset (56,838 training samples, 80% split)\n")
        f.write("- **Holdout Test Set**: 14,210 samples (20% split) - **Completely Untouched** to prevent data snooping\n")
        f.write("- **Validation Method**: 5-Fold Stratified Cross-Validation (`random_state=42`)\n")
        f.write("- **Data Leakage Prevention**: Imputation and feature scaling were fitted **strictly inside each cross-validation fold**\n")
        f.write("- **Evaluation Threshold**: 0.50 (Initial Standard Decision Threshold)\n\n")
        
        f.write("## 2. CatBoost 5-Fold Cross-Validation Results\n\n")
        f.write("| Model | ROC-AUC (Mean ± Std) | PR-AUC (Mean ± Std) | Accuracy | Balanced Accuracy | Precision | Recall | F1-Score | Total CV Time |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in overall_results:
            f.write(f"| **{r['Model']}** | {r['ROC_AUC_Mean']:.4f} ± {r['ROC_AUC_Std']:.4f} | {r['PR_AUC_Mean']:.4f} ± {r['PR_AUC_Std']:.4f} | {r['Accuracy_Mean']:.4f} | {r['Balanced_Accuracy_Mean']:.4f} | {r['Precision_Mean']:.4f} | {r['Recall_Mean']:.4f} | {r['F1_Mean']:.4f} | {r['Total_CV_Time_Sec']:.1f}s |\n")

        f.write("\n## 3. Side-by-Side Comparison: CatBoost vs XGBoost Baseline\n\n")
        f.write("| Metric | XGBoost (Baseline) | CatBoost (Baseline) | CatBoost (Tuned) | Delta (Best CatBoost vs XGBoost) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        
        best_cat = cv_results_df.loc[cv_results_df['ROC_AUC_Mean'].idxmax()]
        
        xgb_roc = xgb_row.get('ROC_AUC_Mean', 0.6803) if xgb_row else 0.6803
        xgb_pr = xgb_row.get('PR_AUC_Mean', 0.4608) if xgb_row else 0.4608
        xgb_f1 = xgb_row.get('F1_Mean', 0.4961) if xgb_row else 0.4961
        xgb_rec = xgb_row.get('Recall_Mean', 0.6364) if xgb_row else 0.6364
        xgb_prec = xgb_row.get('Precision_Mean', 0.4066) if xgb_row else 0.4066
        
        base_cat = cv_results_df[cv_results_df['Model'] == 'CatBoost (Baseline)'].iloc[0]
        tune_cat = cv_results_df[cv_results_df['Model'] == 'CatBoost (Tuned/Regularized)'].iloc[0]

        f.write(f"| **ROC-AUC** | {xgb_roc:.4f} | {base_cat['ROC_AUC_Mean']:.4f} | {tune_cat['ROC_AUC_Mean']:.4f} | {tune_cat['ROC_AUC_Mean'] - xgb_roc:+.4f} |\n")
        f.write(f"| **PR-AUC** | {xgb_pr:.4f} | {base_cat['PR_AUC_Mean']:.4f} | {tune_cat['PR_AUC_Mean']:.4f} | {tune_cat['PR_AUC_Mean'] - xgb_pr:+.4f} |\n")
        f.write(f"| **F1-Score** | {xgb_f1:.4f} | {base_cat['F1_Mean']:.4f} | {tune_cat['F1_Mean']:.4f} | {tune_cat['F1_Mean'] - xgb_f1:+.4f} |\n")
        f.write(f"| **Recall (Churn Caught)** | {xgb_rec:.4f} | {base_cat['Recall_Mean']:.4f} | {tune_cat['Recall_Mean']:.4f} | {tune_cat['Recall_Mean'] - xgb_rec:+.4f} |\n")
        f.write(f"| **Precision** | {xgb_prec:.4f} | {base_cat['Precision_Mean']:.4f} | {tune_cat['Precision_Mean']:.4f} | {tune_cat['Precision_Mean'] - xgb_prec:+.4f} |\n")
        f.write(f"| **Accuracy** | ~0.6223 | {base_cat['Accuracy_Mean']:.4f} | {tune_cat['Accuracy_Mean']:.4f} | {tune_cat['Accuracy_Mean'] - 0.6223:+.4f} |\n")
        f.write(f"| **Balanced Accuracy** | ~0.6265 | {base_cat['Balanced_Accuracy_Mean']:.4f} | {tune_cat['Balanced_Accuracy_Mean']:.4f} | {tune_cat['Balanced_Accuracy_Mean'] - 0.6265:+.4f} |\n")

        f.write("\n## 4. Consolidated Out-of-Fold (OOF) Confusion Matrix (Threshold = 0.50)\n\n")
        f.write(f"**Best CatBoost ({best_config_name})** on 56,838 Out-of-Fold Training Predictions:\n\n")
        f.write("| Actual \\ Predicted | Predicted Non-Churn (0) | Predicted Churn (1) |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| **Actual Non-Churn (0)** | **{best_cat['OOF_TN']}** (True Negatives) | **{best_cat['OOF_FP']}** (False Positives) |\n")
        f.write(f"| **Actual Churn (1)** | **{best_cat['OOF_FN']}** (False Negatives) | **{best_cat['OOF_TP']}** (True Positives) |\n")

    print(f"      Saved comprehensive markdown report: {report_md_path}")
    print("\n[+] Step 1 CatBoost Cross-Validation Complete!")
    return cv_results_df

if __name__ == "__main__":
    run_catboost_experiment()
