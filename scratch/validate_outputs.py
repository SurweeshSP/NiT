import os
import json
import pandas as pd
import glob

results_dir = "results"
print("==========================================")
print("OUTPUT ARTIFACT VALIDATION REPORT")
print("==========================================")

# 1. Key CSV Files Check
csv_files = [
    "final_summary.csv",
    "statistical_analysis.csv",
    "faci_scores.csv",
    "policy_memory.csv",
    "surrogate_log.csv",
    "validator_sweep.csv"
]

print("\n--- 1. Primary CSV Files Integrity Check ---")
for f in csv_files:
    fpath = os.path.join(results_dir, f)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        try:
            df = pd.read_csv(fpath)
            print(f"[OK] {f:<25s} | Size: {size/1024:.1f} KB | Rows: {len(df)} | Columns: {len(df.columns)}")
        except Exception as e:
            print(f"[FAIL] {f:<25s} | Read Error: {e}")
    else:
        print(f"[MISSING] {f:<25s}")

# Check per-run CSVs
print("\n--- 1b. Per-Run CSV Files Check ---")
for i in range(1, 6):
    r_dir = os.path.join(results_dir, f"run_{i}")
    m_path = os.path.join(r_dir, "metrics.csv")
    if os.path.exists(m_path):
        df_m = pd.read_csv(m_path)
        print(f"[OK] run_{i}/metrics.csv           | Size: {os.path.getsize(m_path)/1024:.1f} KB | Rows: {len(df_m)}")

# 2. Key Report Markdown Files
md_files = [
    "final_report.md",
    "paper.md",
    "conclusion.md",
    "data_profile.md"
]

print("\n--- 2. Markdown Report Files Check ---")
for f in md_files:
    fpath = os.path.join(results_dir, f)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        with open(fpath, "r", encoding="utf-8", errors="replace") as file:
            lines = file.readlines()
        print(f"[OK] {f:<25s} | Size: {size/1024:.1f} KB | Lines: {len(lines)}")
    else:
        print(f"[MISSING] {f:<25s}")

# 3. JSON Files
json_files = [
    "label_mapping.json"
]

print("\n--- 3. JSON Configuration Check ---")
for f in json_files:
    fpath = os.path.join(results_dir, f)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        with open(fpath, "r", encoding="utf-8") as file:
            data = json.load(file)
        print(f"[OK] {f:<25s} | Size: {size} bytes | Keys: {list(data.keys())}")
    else:
        print(f"[MISSING] {f:<25s}")

# 4. Multi-Run Directory Validation
print("\n--- 4. Multi-Run Subdirectories Check ---")
run_dirs = sorted(glob.glob(os.path.join(results_dir, "run_*")))
print(f"Found {len(run_dirs)} run subdirectories:")
for rd in run_dirs:
    r_name = os.path.basename(rd)
    r_summary = os.path.join(rd, "run_summary.csv")
    r_repro = os.path.join(rd, "reproducibility.json")
    if os.path.exists(r_summary) and os.path.exists(r_repro):
        print(f"  [OK] {r_name:<10s} -> run_summary.csv and reproducibility.json present.")
    else:
        print(f"  [INCOMPLETE] {r_name:<10s}")

print("\n--- 5. Empirical Results Metrics Summary ---")
summary_path = os.path.join(results_dir, "final_summary.csv")
if os.path.exists(summary_path):
    df_s = pd.read_csv(summary_path)
    print(f"Total Runs Evaluated: {len(df_s)}")
    print(f"Macro F1 Mean: {df_s['Macro F1'].mean():.4f} +/- {df_s['Macro F1'].std():.4f}")
    print(f"Accuracy Mean: {df_s['Accuracy'].mean():.4f} +/- {df_s['Accuracy'].std():.4f}")
    print(f"Precision Mean: {df_s['Precision'].mean():.4f} +/- {df_s['Precision'].std():.4f}")
    print(f"Recall Mean: {df_s['Recall'].mean():.4f} +/- {df_s['Recall'].std():.4f}")

print("\nValidation complete.")
