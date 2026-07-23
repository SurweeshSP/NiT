import os
import pandas as pd
import numpy as np

OUTPUT_DIR = "results/multi_seed"
SEEDS = [42, 43, 44, 505050]

all_metrics = []
all_summaries = []

for s in SEEDS:
    metrics_path = os.path.join(OUTPUT_DIR, f"seed_{s}", "metrics.csv")
    summary_path = os.path.join(OUTPUT_DIR, f"seed_{s}", "run_summary.csv")
    
    if os.path.exists(metrics_path):
        all_metrics.append(pd.read_csv(metrics_path))
    if os.path.exists(summary_path):
        all_summaries.append(pd.read_csv(summary_path))

if all_metrics:
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    # Aggregate baseline comparison
    agg = (
        combined_metrics.groupby("baseline")["macro_f1"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    agg.columns = ["Baseline", "Macro F1 Mean", "Macro F1 Std", "Min", "Max"]
    agg["Macro F1 (mean±std)"] = agg.apply(
        lambda r: f"{r['Macro F1 Mean']:.4f} ± {r['Macro F1 Std']:.4f}", axis=1
    )
    agg.to_csv(os.path.join(OUTPUT_DIR, "baseline_comparison.csv"), index=False)
    print("baseline_comparison.csv written.")

if all_summaries:
    combined_summaries = pd.concat(all_summaries, ignore_index=True)
    combined_summaries.to_csv(os.path.join(OUTPUT_DIR, "final_summary.csv"), index=False)
    print("final_summary.csv written.")
    
    # Statistical analysis: simple mean/std over run_summary stats
    stat_df = pd.DataFrame({
        "Metric": ["Accuracy", "Macro F1", "Weighted F1", "Precision", "Recall"],
        "Mean": [
            combined_summaries["accuracy"].mean(),
            combined_summaries["macro_f1"].mean(),
            combined_summaries["weighted_f1"].mean(),
            combined_summaries["precision"].mean(),
            combined_summaries["recall"].mean(),
        ],
        "Std": [
            combined_summaries["accuracy"].std(),
            combined_summaries["macro_f1"].std(),
            combined_summaries["weighted_f1"].std(),
            combined_summaries["precision"].std(),
            combined_summaries["recall"].std(),
        ]
    })
    stat_df.to_csv(os.path.join(OUTPUT_DIR, "statistical_analysis.csv"), index=False)
    print("statistical_analysis.csv written.")
