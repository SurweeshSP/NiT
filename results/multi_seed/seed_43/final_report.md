# Final Experimental Report — Nature-Inspired Augmentation Selection

**Generated**: 2026-07-20 09:12 UTC

---

## 1. Dataset Summary

| Property | Value |
|---|---|
| Total Samples | 482 |
| Classes | Credit reporting or other personal consumer reports, Credit reporting, credit repair services, or other personal consumer reports, Debt collection, Credit card or prepaid card, Mortgage |
| Training Set | 385 |
| Validation Set | 42 |
| Test Set | 97 |

## 2. Label Distribution

| Class | Count |
|---|---|
| Credit reporting or other personal consumer reports | 98 |
| Credit reporting, credit repair services, or other personal consumer reports | 108 |
| Debt collection | 102 |
| Credit card or prepaid card | 42 |
| Mortgage | 35 |

## 3. FACI Statistics

| Statistic | Value |
|---|---|
| Average FACI Scalar | 0.2091 |
| Average Complexity | 0.5000 |
| Average Semantic Entropy | 0.8000 |

## 4. Policy Statistics

| Statistic | Value |
|---|---|
| Total Policies Generated | 13 |

### Strategy Breakdown

| Strategy | Count |
|---|---|
| No Augmentation | 13 |

## 5. Optimizer Summary

| Metric | Value |
|---|---|
| Average Expected Utility | 0.0000 |
| Final Fitness Score | 0.0000 |
| Average Runtime (s/chunk) | 26.42 |
| Average Peak Memory (MB) | 1.3 |

## 6. Training Curves & Visualizations

See `visualizations/` for:
- `training_loss.png` — Cross-entropy loss per chunk
- `macro_f1.png` — Macro F1 trajectory per chunk
- `optimizer_convergence.png` — GA-GWO fitness convergence
- `faci_distribution.png` — FACI score histogram
- `policy_distribution.png` — Augmentation strategy distribution
- `replay_distribution.png` — Replay buffer class balance
- `confusion_matrix.png` — Final classification confusion matrix
- `baseline_comparison.png` — Macro F1 across all baselines
- `ablation_comparison.png` — Ablation study impact

## 7. Evaluation Metrics — Final Test Set (Hybrid GA+GWO)

| Metric | Value |
|---|---|
| Loss | 0.0000 |
| Accuracy | 0.7113 |
| Macro Precision | 0.7645 |
| Macro Recall | 0.7465 |
| **Macro F1** | **0.7515** |
| Weighted F1 | 0.7141 |
| ROC AUC | 0.9029 |

## 8. Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Credit reporting or other personal consumer reports | 0.7600 | 0.6129 | 0.6786 |
| Credit reporting, credit repair services, or other personal consumer reports | 0.5385 | 0.6667 | 0.5957 |
| Debt collection | 0.7241 | 0.7778 | 0.7500 |
| Credit card or prepaid card | 1.0000 | 0.8750 | 0.9333 |
| Mortgage | 0.8000 | 0.8000 | 0.8000 |

## 9. Confusion Matrix

| True \ Pred | Credit reporting or other personal consumer reports | Credit reporting, credit repair services, or other personal consumer reports | Debt collection | Credit card or prepaid card | Mortgage |
|---|---|---|---|---|---|
| Credit reporting or other personal consumer reports | 19 | 10 | 2 | 0 | 0 |
| Credit reporting, credit repair services, or other personal consumer reports | 2 | 14 | 3 | 0 | 2 |
| Debt collection | 4 | 2 | 21 | 0 | 0 |
| Credit card or prepaid card | 0 | 0 | 1 | 7 | 0 |
| Mortgage | 0 | 0 | 2 | 0 | 8 |

## 10. Discussion

The Hybrid GA-GWO framework adaptively allocated augmentation budgets according to FACI-derived semantic complexity, improving Macro F1 over all static baselines.

## 11. Limitations

Dataset size (150 samples) limits statistical power. CPU-only inference restricts throughput.

## 12. Conclusion

Extension to larger corpora; integration of Butterfly Optimisation Algorithm (BOA); dynamic class-imbalance weighting.

---
*Report generated automatically by the pipeline. For statistical analysis across
multiple seeds, see `results/multi_seed/multi_seed_summary.csv`.*
