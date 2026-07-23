import os
import json
import ast
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
import scipy.stats as stats

results_dir = "results"
runs = sorted(glob.glob(os.path.join(results_dir, "run_*")))
if not runs:
    print("No run directories found in results/")
    exit(1)

print(f"Found {len(runs)} run directories: {[os.path.basename(r) for r in runs]}")

# ==========================================
# PROMPT 1: SURROGATE PREDICTIVE QUALITY
# ==========================================
print("\n" + "="*60)
print("PROMPT 1: SURROGATE PREDICTIVE QUALITY")
print("="*60)

surrogate_logs = []
candidate_dfs = []

for r in runs:
    s_log = os.path.join(r, "surrogate_log.csv")
    if os.path.exists(s_log):
        df_s = pd.read_csv(s_log)
        df_s["run"] = os.path.basename(r)
        surrogate_logs.append(df_s)
        
    c_csv = os.path.join(r, "policy_memory_candidates.csv")
    if os.path.exists(c_csv):
        df_c = pd.read_csv(c_csv)
        df_c["run"] = os.path.basename(r)
        candidate_dfs.append(df_c)

if surrogate_logs:
    all_s = pd.concat(surrogate_logs, ignore_index=True)
    print("\n[Surrogate Log Audit Across Runs]")
    print(all_s.groupby("Chunk_ID")[["R2", "MAE", "Sample_Count"]].mean().to_string())
    
    trusted_count = (all_s["Is_Trusted"] == True).sum()
    untrusted_count = (all_s["Is_Trusted"] == False).sum()
    total_chunks = len(all_s)
    print(f"\nSurrogate Trusted: {trusted_count}/{total_chunks} ({trusted_count/total_chunks*100:.1f}%)")
    print(f"Surrogate Untrusted/Fallback: {untrusted_count}/{total_chunks} ({untrusted_count/total_chunks*100:.1f}%)")
    print(f"Max R2 across all chunk evaluations: {all_s['R2'].max():.4f}")
    print(f"Min R2 across all chunk evaluations: {all_s['R2'].min():.4f}")
    print(f"Mean R2 across all chunk evaluations: {all_s['R2'].mean():.4f}")
    print(f"Mean MAE across all chunk evaluations: {all_s['MAE'].mean():.4f}")

# Cross-Validation on combined candidates
if candidate_dfs:
    all_c = pd.concat(candidate_dfs, ignore_index=True)
    print(f"\nTotal candidate data points collected across all runs: {len(all_c)}")
    
    # Parse Alpha vectors and FACI vectors
    def parse_arr(val):
        if isinstance(val, str):
            val_clean = val.replace("np.float64(", "").replace(")", "")
            try:
                return ast.literal_eval(val_clean)
            except:
                return []
        return val

    X_list = []
    y_list = []
    for _, row in all_c.iterrows():
        alpha = parse_arr(row["Alpha"])
        faci = parse_arr(row["FACI_Vector"])
        f1 = float(row["Macro_F1"])
        if len(alpha) == 8 and len(faci) == 8:
            X_list.append(alpha + faci)
            y_list.append(f1)
            
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Parsed valid (Alpha + FACI -> Macro_F1) samples: N = {len(y)}")
    
    if len(y) > 10:
        rf = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_scores = []
        mae_scores = []
        for train_idx, test_idx in kf.split(X):
            rf.fit(X[train_idx], y[train_idx])
            preds = rf.predict(X[test_idx])
            r2_scores.append(r2_score(y[test_idx], preds))
            mae_scores.append(mean_absolute_error(y[test_idx], preds))
            
        print(f"5-Fold CV Surrogate R^2: {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})")
        print(f"5-Fold CV Surrogate MAE: {np.mean(mae_scores):.4f} (+/- {np.std(mae_scores):.4f})")

# ==========================================
# PROMPT 2: GA-GWO SELECTION & BUDGET AUDIT
# ==========================================
print("\n" + "="*60)
print("PROMPT 2: GA-GWO SELECTION & BUDGET COMPARISON")
print("="*60)

metrics_dfs = []
for r in runs:
    m_csv = os.path.join(r, "metrics.csv")
    if os.path.exists(m_csv):
        df_m = pd.read_csv(m_csv)
        metrics_dfs.append(df_m)

if metrics_dfs:
    all_m = pd.concat(metrics_dfs, ignore_index=True)
    
    print("\n[Strategy Selection Breakdown for Proposed Hybrid GA-GWO]")
    hybrid_m = all_m[all_m["baseline"] == "Proposed Hybrid GA-GWO"]
    strat_counts = hybrid_m["strategy"].value_counts()
    print(strat_counts.to_string())
    print("\nStrategy Percentages:")
    print((strat_counts / len(hybrid_m) * 100).to_string())
    
    print("\n[Average Budget Assigned by Baseline]")
    budget_stats = all_m.groupby("baseline")["budget"].agg(["mean", "min", "max", "std"])
    print(budget_stats.to_string())
    
    print("\n[Average Macro F1 by Baseline Across Chunks]")
    f1_stats = all_m.groupby("baseline")["macro_f1"].agg(["mean", "max", "std"])
    print(f1_stats.to_string())

# ==========================================
# PROMPT 3: PER-CLASS PERFORMANCE ANALYSIS
# ==========================================
print("\n" + "="*60)
print("PROMPT 3: PER-CLASS PERFORMANCE BREAKDOWN")
print("="*60)

for r in runs:
    run_name = os.path.basename(r)
    rpt_path = os.path.join(r, "classification_report.txt")
    if os.path.exists(rpt_path):
        print(f"\n--- {run_name} Classification Report ---")
        with open(rpt_path, "r") as f:
            print(f.read().strip())

# ==========================================
# PROMPT 4: FEATURE SUFFICIENCY CHECK
# ==========================================
print("\n" + "="*60)
print("PROMPT 4: FEATURE SUFFICIENCY CHECK")
print("="*60)

if candidate_dfs and len(y) > 10:
    feature_names = [
        "alpha_bt", "alpha_bert", "alpha_budget", "alpha_mask", "alpha_sem", "alpha_ent", "alpha_pri", "alpha_lr",
        "faci_comp", "faci_ent_dens", "faci_fraud_dens", "faci_risk", "faci_amb", "faci_entropy", "faci_rare", "faci_redact"
    ]
    
    # Random Forest Feature Importance
    rf_full = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_full.fit(X, y)
    importances = rf_full.feature_importances_
    
    # Feature Correlations with Macro F1
    correlations = []
    for i in range(X.shape[1]):
        r_pearson, p_pearson = stats.pearsonr(X[:, i], y)
        r_spearman, p_spearman = stats.spearmanr(X[:, i], y)
        correlations.append({
            "Feature": feature_names[i] if i < len(feature_names) else f"F_{i}",
            "RF_Importance": round(importances[i], 4),
            "Pearson_r": round(r_pearson, 4),
            "Pearson_p": round(p_pearson, 4),
            "Spearman_r": round(r_spearman, 4),
            "Spearman_p": round(p_spearman, 4),
        })
        
    df_feat = pd.DataFrame(correlations)
    print("\nFeature Importance and Correlation Analysis:")
    print(df_feat.to_string(index=False))

# ==========================================
# PROMPT 5: DATA VOLUME SUFFICIENCY CHECK
# ==========================================
print("\n" + "="*60)
print("PROMPT 5: DATA VOLUME SUFFICIENCY CHECK")
print("="*60)

if candidate_dfs:
    total_candidates = len(all_c)
    unique_chunks = len(all_c['Chunk_ID'].unique()) if 'Chunk_ID' in all_c.columns else 0
    runs_count = len(candidate_dfs)
    print(f"Total evaluated policy candidate samples across {runs_count} runs: {total_candidates}")
    print(f"Average samples collected per run: {total_candidates / runs_count:.1f}")
    print(f"Number of input features for surrogate model: 16 (8 Alpha policy + 8 FACI vector)")
    print(f"Sample-to-feature ratio: {total_candidates / 16:.1f} samples per feature")
    
    if total_candidates < 200:
        print("DIAGNOSIS: Data volume is extremely small for learning complex non-linear policy mapping!")
    else:
        print("DIAGNOSIS: Data volume is adequate for 16-feature regression.")

# ==========================================
# PROMPT 6: CONTROL RULE / TRIVIAL ALTERNATIVE
# ==========================================
print("\n" + "="*60)
print("PROMPT 6: TRIVIAL ALTERNATIVE / CONTROL RULE CHECK")
print("="*60)

if metrics_dfs:
    print("\n[Baseline Comparison Summary Across All Runs]")
    for b in all_m["baseline"].unique():
        sub = all_m[all_m["baseline"] == b]
        f1_mean = sub["macro_f1"].mean()
        f1_max = sub["macro_f1"].max()
        print(f"Baseline: {b:<35s} | Mean Macro F1: {f1_mean:.4f} | Max Macro F1: {f1_max:.4f}")

print("\nDiagnostic complete.")
