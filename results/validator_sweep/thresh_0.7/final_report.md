# Final Experimental Report — Nature-Inspired Augmentation Selection

**Generated**: 2026-07-21 06:54 UTC

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
| No Augmentation | 2 |
| BERT Contextual | 1 |
| Back Translation | 1 |
| Hybrid | 9 |

## 5. Optimizer Summary

| Metric | Value |
|---|---|
| Average Expected Utility | 0.6330 |
| Final Fitness Score | 0.7245 |
| Average Runtime (s/chunk) | 17.30 |
| Average Peak Memory (MB) | 1.1 |

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
| Accuracy | 0.5773 |
| Macro Precision | 0.6048 |
| Macro Recall | 0.6375 |
| **Macro F1** | **0.6048** |
| Weighted F1 | 0.5759 |
| ROC AUC | 0.8332 |

## 8. Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Credit reporting or other personal consumer reports | 0.5135 | 0.6129 | 0.5588 |
| Credit reporting, credit repair services, or other personal consumer reports | 0.4211 | 0.3810 | 0.4000 |
| Debt collection | 0.8235 | 0.5185 | 0.6364 |
| Credit card or prepaid card | 0.5385 | 0.8750 | 0.6667 |
| Mortgage | 0.7273 | 0.8000 | 0.7619 |

## 9. Confusion Matrix

| True \ Pred | Credit reporting or other personal consumer reports | Credit reporting, credit repair services, or other personal consumer reports | Debt collection | Credit card or prepaid card | Mortgage |
|---|---|---|---|---|---|
| Credit reporting or other personal consumer reports | 19 | 5 | 2 | 3 | 2 |
| Credit reporting, credit repair services, or other personal consumer reports | 10 | 8 | 1 | 2 | 0 |
| Debt collection | 8 | 4 | 14 | 1 | 0 |
| Credit card or prepaid card | 0 | 0 | 0 | 7 | 1 |
| Mortgage | 0 | 2 | 0 | 0 | 8 |

## 10. Discussion

The Hybrid GA-GWO framework adaptively allocated augmentation budgets according to FACI-derived semantic complexity, improving Macro F1 over all static baselines.

## 11. Limitations

Dataset size (150 samples) limits statistical power. CPU-only inference restricts throughput.

## 12. Conclusion

Extension to larger corpora; integration of Butterfly Optimisation Algorithm (BOA); dynamic class-imbalance weighting.

---
*Report generated automatically by the pipeline. For statistical analysis across
multiple seeds, see `results/multi_seed/multi_seed_summary.csv`.*
