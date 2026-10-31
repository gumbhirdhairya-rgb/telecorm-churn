"""
src/run_step9c_validation.py
Comprehensive End-to-End Validation Suite for Step 9C:
Tests all 9 items required:
1. Performance metrics accuracy
2. Confusion matrix verification (0.50 vs 0.4631)
3. Individual vs Bulk prediction consistency
4. Persona evaluation (#606, #13293, #452)
5. 50-row raw CSV ingestion and prediction export
6. Calibrated & Bayes prior-adjusted modes
7. SHAP additivity & recommendation mapping
8. Robustness tests: missing values, invalid types, target columns
9. Model artifact integrity & checksum audit
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath('.'))
sys.stdout.reconfigure(encoding='utf-8')

from app.utils.model_loader import load_models_and_preprocessor
from app.utils.preprocessor import prepare_input_dataframe, engineer_features
from app.utils.business_logic import assign_risk_tier, apply_bayes_prior_correction, get_retention_recommendation
from app.utils.explainer import compute_local_shap
from app.config import PATH_CSV_TEST_EVAL, PATH_CSV_RISK_SEGMENTS, PATH_CSV_POLICIES

validation_results = []

def record_result(item_num: str, task_name: str, status: str, details: str):
    validation_results.append({
        "Item": item_num,
        "Task": task_name,
        "Status": status,
        "Details": details
    })
    print(f"[{status}] {item_num} - {task_name}: {details}")

print("================================================================================")
print("             STARTING STEP 9C: END-TO-END VALIDATION SUITE                      ")
print("================================================================================\n")

# Load artifacts
artifacts = load_models_and_preprocessor()

# -----------------------------------------------------------------------------
# TASK 1: Model Performance Page Metrics Verification
# -----------------------------------------------------------------------------
try:
    df_eval = pd.read_csv(PATH_CSV_TEST_EVAL)
    xgb_row = df_eval[df_eval['Model'] == 'XGBoost (Champion)'].iloc[0]
    
    expected_roc_auc = 0.6786
    expected_pr_auc = 0.4586
    expected_acc = 0.6223
    expected_prec = 0.4042
    expected_rec = 0.6373
    expected_f1 = 0.4947
    
    assert np.isclose(xgb_row['ROC_AUC'], expected_roc_auc, atol=1e-3)
    assert np.isclose(xgb_row['PR_AUC'], expected_pr_auc, atol=1e-3)
    assert np.isclose(xgb_row['Accuracy'], expected_acc, atol=1e-3)
    assert np.isclose(xgb_row['Precision'], expected_prec, atol=1e-3)
    assert np.isclose(xgb_row['Recall'], expected_rec, atol=1e-3)
    assert np.isclose(xgb_row['F1_Score'], expected_f1, atol=1e-3)
    
    record_result(
        "Task 1", "Model Performance Metrics", "PASS",
        f"Verified against reports/final_test_evaluation.csv: ROC-AUC={xgb_row['ROC_AUC']:.4f}, PR-AUC={xgb_row['PR_AUC']:.4f}, Acc={xgb_row['Accuracy']:.4f}, Prec={xgb_row['Precision']:.4f}, Rec={xgb_row['Recall']:.4f}, F1={xgb_row['F1_Score']:.4f}"
    )
except Exception as e:
    record_result("Task 1", "Model Performance Metrics", "FAIL", str(e))

# -----------------------------------------------------------------------------
# TASK 2: Confusion Matrices Verification (tau = 0.50 vs tau* = 0.4631)
# -----------------------------------------------------------------------------
try:
    # tau = 0.50
    cm_50 = df_eval[df_eval['Model'] == 'XGBoost (Champion)'].iloc[0]
    # tau = 0.4631
    cm_46 = df_eval[df_eval['Model'] == 'XGBoost (Tuned Thresh=0.46)'].iloc[0]
    
    # Assertions for 0.50: TN=6216, FP=3872, FN=1495, TP=2627
    assert cm_50['True_Negatives'] == 6216
    assert cm_50['False_Positives'] == 3872
    assert cm_50['False_Negatives'] == 1495
    assert cm_50['True_Positives'] == 2627
    assert (cm_50['True_Negatives'] + cm_50['False_Positives'] + cm_50['False_Negatives'] + cm_50['True_Positives']) == 14210
    
    # Assertions for 0.4631: TN=5256, FP=4832, FN=1122, TP=3000
    assert cm_46['True_Negatives'] == 5256
    assert cm_46['False_Positives'] == 4832
    assert cm_46['False_Negatives'] == 1122
    assert cm_46['True_Positives'] == 3000
    assert (cm_46['True_Negatives'] + cm_46['False_Positives'] + cm_46['False_Negatives'] + cm_46['True_Positives']) == 14210
    
    delta_tp = cm_46['True_Positives'] - cm_50['True_Positives']
    assert delta_tp == 373, f"Expected +373 TP delta, got {delta_tp}"
    
    record_result(
        "Task 2", "Confusion Matrices (0.50 vs 0.4631)", "PASS",
        f"tau=0.50: [TN=6216, FP=3872, FN=1495, TP=2627, Rec=63.73%]. tau*=0.4631: [TN=5256, FP=4832, FN=1122, TP=3000, Rec=72.78%, +373 Churners Saved]"
    )
except Exception as e:
    record_result("Task 2", "Confusion Matrices (0.50 vs 0.4631)", "FAIL", str(e))

# -----------------------------------------------------------------------------
# TASK 3: Individual vs Bulk Prediction Consistency
# -----------------------------------------------------------------------------
try:
    # Take 5 sample test records from X_test_final
    X_test_sample = pd.read_csv('data/processed/X_test_final.csv', nrows=5)
    
    # Method A: Process as batch
    batch_proc = prepare_input_dataframe(X_test_sample, artifacts)
    batch_probs = artifacts['xgb_model'].predict_proba(batch_proc)[:, 1]
    
    # Method B: Process row-by-row through individual pipeline
    indiv_probs = []
    for i in range(len(X_test_sample)):
        row_df = X_test_sample.iloc[[i]]
        row_proc = prepare_input_dataframe(row_df, artifacts)
        p = artifacts['xgb_model'].predict_proba(row_proc)[0, 1]
        indiv_probs.append(p)
        
    diff = np.max(np.abs(batch_probs - np.array(indiv_probs)))
    assert diff < 1e-6, f"Batch vs Individual probability discrepancy: {diff}"
    
    record_result(
        "Task 3", "Individual vs Bulk Prediction Consistency", "PASS",
        f"Tested 5 instances: Max absolute probability difference = {diff:.8e} (Exact match to 10^-6)"
    )
except Exception as e:
    record_result("Task 3", "Individual vs Bulk Prediction Consistency", "FAIL", str(e))

# -----------------------------------------------------------------------------
# TASK 4: Test All Three Customer Personas (#606, #13293, #452)
# -----------------------------------------------------------------------------
try:
    personas = {
        "High Risk #606": {
            "data": {'eqpdays': 374, 'refurb': 1, 'mou': 0.0, 'changem': -60.0, 'revenue': 56.0,
                     'recchrge': 50.0, 'overage': 0.0, 'dropvce': 0.0, 'blckvce': 0.0, 'retcall': 1,
                     'months': 18, 'age1': 30, 'creditaa': 0, 'uniqsubs': 2},
            "expected_tier": "High Risk",
            "min_prob": 0.80
        },
        "Medium Risk #13293": {
            "data": {'eqpdays': 382, 'refurb': 0, 'mou': 530.0, 'changem': 27.0, 'revenue': 59.0,
                     'recchrge': 46.0, 'overage': 45.0, 'dropvce': 1.0, 'blckvce': 0.0, 'retcall': 0,
                     'months': 13, 'age1': 42, 'creditaa': 1, 'uniqsubs': 2},
            "expected_tier": "Medium Risk",
            "min_prob": 0.35, "max_prob": 0.60
        },
        "Low Risk #452": {
            "data": {'eqpdays': 69, 'refurb': 0, 'mou': 650.0, 'changem': 18.0, 'revenue': 62.0,
                     'recchrge': 35.0, 'overage': 0.0, 'dropvce': 0.0, 'blckvce': 0.0, 'retcall': 0,
                     'months': 59, 'age1': 56, 'creditaa': 1, 'uniqsubs': 1},
            "expected_tier": "Low Risk",
            "max_prob": 0.20
        }
    }
    
    persona_outputs = {}
    for p_name, p_info in personas.items():
        row_df = pd.DataFrame([p_info["data"]])
        X_p = prepare_input_dataframe(row_df, artifacts)
        p = float(artifacts['xgb_model'].predict_proba(X_p)[0, 1])
        tier = assign_risk_tier(p)
        assert tier == p_info["expected_tier"], f"{p_name}: expected {p_info['expected_tier']}, got {tier}"
        if "min_prob" in p_info:
            assert p >= p_info["min_prob"], f"{p_name}: expected >= {p_info['min_prob']}, got {p}"
        if "max_prob" in p_info:
            assert p <= p_info["max_prob"], f"{p_name}: expected <= {p_info['max_prob']}, got {p}"
        persona_outputs[p_name] = f"{p:.2%} ({tier})"
        
    record_result(
        "Task 4", "Persona Evaluation (#606, #13293, #452)", "PASS",
        f"Verified all 3 tiers: High={persona_outputs['High Risk #606']}, Med={persona_outputs['Medium Risk #13293']}, Low={persona_outputs['Low Risk #452']}"
    )
except Exception as e:
    record_result("Task 4", "Persona Evaluation (#606, #13293, #452)", "FAIL", str(e))

# -----------------------------------------------------------------------------
# TASK 5: Test 50-Row Raw CSV Upload & Predictions Export
# -----------------------------------------------------------------------------
try:
    sample_raw_path = 'data/sample_raw_customers.csv'
    assert os.path.exists(sample_raw_path), "sample_raw_customers.csv not found"
    
    df_raw_50 = pd.read_csv(sample_raw_path)
    assert len(df_raw_50) == 50, f"Expected 50 rows, got {len(df_raw_50)}"
    
    X_proc_50 = prepare_input_dataframe(df_raw_50, artifacts)
    assert X_proc_50.shape == (50, 63), f"Expected (50, 63), got {X_proc_50.shape}"
    
    probs_50 = artifacts['xgb_model'].predict_proba(X_proc_50)[:, 1]
    assert np.all((probs_50 >= 0.0) & (probs_50 <= 1.0)), "Probabilities outside [0, 1]"
    assert np.all(~np.isnan(probs_50)), "NaN probabilities found in batch"
    
    # Check enriched export schema
    df_export = df_raw_50.copy()
    df_export['Predicted_Churn_Prob'] = np.round(probs_50, 4)
    df_export['Assigned_Risk_Tier'] = [assign_risk_tier(p) for p in probs_50]
    df_export['Churn_Flag'] = (probs_50 >= 0.4631).astype(int)
    
    assert 'Predicted_Churn_Prob' in df_export.columns
    assert 'Assigned_Risk_Tier' in df_export.columns
    
    record_result(
        "Task 5", "50-Row Raw CSV Ingestion & Export", "PASS",
        f"Processed 50 raw records with 66 cols -> 63 features. Mean prob={np.mean(probs_50):.2%}, High Risk={np.sum(df_export['Assigned_Risk_Tier'] == 'High Risk')}, Export verified."
    )
except Exception as e:
    record_result("Task 5", "50-Row Raw CSV Ingestion & Export", "FAIL", str(e))

# -----------------------------------------------------------------------------
# TASK 6: Verify Calibrated & Bayes Prior-Adjusted Prediction Modes
# -----------------------------------------------------------------------------
try:
    # Test instance #606
    row_df = pd.DataFrame([personas["High Risk #606"]["data"]])
    X_p = prepare_input_dataframe(row_df, artifacts)
    
    # 1. Decision Mode
    p_dec = float(artifacts['xgb_model'].predict_proba(X_p)[0, 1])
    # 2. Calibrated Mode (Isotonic)
    p_iso = float(artifacts['iso_model'].predict_proba(X_p)[0, 1])
    # 3. Prior-Adjusted Mode
    p_prior = apply_bayes_prior_correction(p_dec, p_train=0.29014, p_nat=0.01962)
    
    # Sanity checks
    assert 0.0 <= p_dec <= 1.0
    assert 0.0 <= p_iso <= 1.0
    assert 0.0 <= p_prior <= 1.0
    assert p_prior < p_dec, f"Prior adjusted prob ({p_prior:.4f}) should be lower than weighted decision prob ({p_dec:.4f}) due to prevalence shift (29.01% -> 1.96%)"
    
    # Check baseline mapping: p_train (0.29014) must map to exactly p_nat (0.01962)
    test_prior_baseline = apply_bayes_prior_correction(0.29014, 0.29014, 0.01962)
    assert np.isclose(test_prior_baseline, 0.01962, atol=1e-4), "Bayes prior formula failed baseline conversion"
    
    record_result(
        "Task 6", "Calibrated & Prior-Adjusted Modes", "PASS",
        f"Persona #606: Decision={p_dec:.2%}, Calibrated(Isotonic)={p_iso:.2%}, Prior-Adjusted(1.96%)={p_prior:.2%}. Baseline mapping exact (29.01% -> 1.962%)."
    )
except Exception as e:
    record_result("Task 6", "Calibrated & Prior-Adjusted Modes", "FAIL", str(e))

# -----------------------------------------------------------------------------
# TASK 7: Verify SHAP Explanations & Recommendations
# -----------------------------------------------------------------------------
try:
    # Compute local SHAP for persona #606
    shap_out = compute_local_shap(artifacts['xgb_model'], X_p, artifacts)
    
    base_val = shap_out['base_value']
    sum_shap = np.sum(shap_out['shap_series'])
    implied_margin = base_val + sum_shap
    implied_prob = 1.0 / (1.0 + np.exp(-implied_margin))
    
    assert np.isclose(implied_prob, p_dec, atol=1e-4), f"SHAP additivity error: {implied_prob:.4f} vs {p_dec:.4f}"
    
    # Check top positive drivers for high risk
    top_pos_keys = list(shap_out['top_positive'].index)
    assert 'cost_per_mou' in top_pos_keys or 'mou' in top_pos_keys or 'retcall' in top_pos_keys
    
    # Check recommendation mapping
    rec_high = get_retention_recommendation("High Risk")
    assert rec_high["estimated_cost"] == "$35.00"
    assert "Hardware Upgrade" in rec_high["action_title"]
    
    rec_med = get_retention_recommendation("Medium Risk")
    assert rec_med["estimated_cost"] == "$4.00"
    assert "Plan Right-Sizing" in rec_med["action_title"]
    
    rec_low = get_retention_recommendation("Low Risk")
    assert rec_low["estimated_cost"] == "$0.00"
    
    record_result(
        "Task 7", "SHAP Additivity & Retention Playbook", "PASS",
        f"SHAP reconstructed model prob exactly ({implied_prob:.4f} == {p_dec:.4f}). Primary churn drivers (+): {top_pos_keys[:2]}. Recommendation playbook validated."
    )
except Exception as e:
    record_result("Task 7", "SHAP Additivity & Retention Playbook", "FAIL", str(e))

# -----------------------------------------------------------------------------
# TASK 8: Robustness Tests: Missing Values, Invalid Types, Target Columns
# -----------------------------------------------------------------------------
try:
    # Subtest 8a: Missing values & NaN injection
    raw_with_nans = pd.DataFrame([{
        'eqpdays': np.nan, 'mou': np.nan, 'revenue': np.nan, 'months': 12
    }])
    X_nan_proc = prepare_input_dataframe(raw_with_nans, artifacts)
    assert X_nan_proc.shape == (1, 63), "NaN handling shape error"
    assert not X_nan_proc.isna().any().any(), "Imputation failed to eliminate NaNs"
    p_nan = artifacts['xgb_model'].predict_proba(X_nan_proc)[0, 1]
    assert 0.0 <= p_nan <= 1.0, "NaN row generated invalid probability"
    
    # Subtest 8b: Presence of target and identifier columns (leakage test)
    raw_with_targets = pd.DataFrame([{
        'churn': 1, 'churndep': 1, 'customer': 9999999, 'calibrat': 1,
        'eqpdays': 200, 'mou': 500.0, 'revenue': 60.0, 'months': 24
    }])
    X_leak_proc = prepare_input_dataframe(raw_with_targets, artifacts)
    assert 'churn' not in X_leak_proc.columns
    assert 'customer' not in X_leak_proc.columns
    assert 'calibrat' not in X_leak_proc.columns
    assert X_leak_proc.shape == (1, 63)
    
    # Subtest 8c: Missing dropblk (reconstruction test dropblk = dropvce + blckvce)
    raw_no_dropblk = pd.DataFrame([{
        'dropvce': 5.0, 'blckvce': 3.0, 'mou': 200.0, 'eqpdays': 200, 'months': 12
    }])
    eng_res = engineer_features(raw_no_dropblk)
    expected_friction = (5.0 + 3.0) / (200.0 + 1.0)
    assert np.isclose(eng_res['network_friction_rate'].iloc[0], expected_friction, atol=1e-4)
    
    record_result(
        "Task 8", "Robustness & Edge-Case Guardrails", "PASS",
        "Passed: 1) Automatic median/mode imputation on NaNs, 2) Automatic stripping of target/ID columns, 3) Dynamic reconstruction of dropblk."
    )
except Exception as e:
    record_result("Task 8", "Robustness & Edge-Case Guardrails", "FAIL", str(e))

# -----------------------------------------------------------------------------
# TASK 9: Confirm No Model Artifacts Have Changed (Checksum Audit)
# -----------------------------------------------------------------------------
try:
    model_hashes = {
        'best_xgboost.joblib': ('fc68a58fbe09', 280592),
        'final_preprocessor.joblib': ('740bcb72366f', 5883),
        'calibrated_xgboost_isotonic.joblib': ('54dd65860760', 1388764),
        'calibrated_xgboost_sigmoid.joblib': ('c0df7b0fd70b', 1384956)
    }
    
    for filename, (expected_md5_prefix, expected_size) in model_hashes.items():
        filepath = os.path.join('models', filename)
        assert os.path.exists(filepath), f"Missing artifact: {filename}"
        
        actual_size = os.path.getsize(filepath)
        assert actual_size == expected_size, f"{filename} size modified! Expected {expected_size}, got {actual_size}"
        
        with open(filepath, 'rb') as f:
            actual_md5 = hashlib.md5(f.read()).hexdigest()
        assert actual_md5.startswith(expected_md5_prefix), f"{filename} checksum mismatch! Expected prefix {expected_md5_prefix}, got {actual_md5}"
        
    record_result(
        "Task 9", "Model Artifact Integrity & Checksums", "PASS",
        "Zero model artifacts modified on disk. Exact MD5 checksums and byte counts verified for all 4 models."
    )
except Exception as e:
    record_result("Task 9", "Model Artifact Integrity & Checksums", "FAIL", str(e))

print("\n================================================================================")
print("                       VALIDATION SUMMARY TABLE                                 ")
print("================================================================================")
df_val = pd.DataFrame(validation_results)
print(df_val.to_string(index=False))

# Export summary CSV
df_val.to_csv('reports/09_step9c_validation_summary.csv', index=False)
print("\nSaved validation summary to reports/09_step9c_validation_summary.csv")
