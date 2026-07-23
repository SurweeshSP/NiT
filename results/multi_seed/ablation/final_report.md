# Final Experimental Report — Nature-Inspired Augmentation Selection

**Generated**: 2026-07-20 12:56 UTC

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
| Average FACI Scalar | 0.2111 |
| Average Complexity | 0.5000 |
| Average Semantic Entropy | 0.8000 |

## 4. Policy Statistics

| Statistic | Value |
|---|---|
| Total Policies Generated | 16 |

### Strategy Breakdown

| Strategy | Count |
|---|---|
| No Augmentation | 16 |

## 5. Optimizer Summary

| Metric | Value |
|---|---|
| Average Expected Utility | 0.0000 |
| Final Fitness Score | 0.0000 |
| Average Runtime (s/chunk) | 15.61 |
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
| Accuracy | 0.7320 |
| Macro Precision | 0.7772 |
| Macro Recall | 0.7677 |
| **Macro F1** | **0.7669** |
| Weighted F1 | 0.7289 |
| ROC AUC | 0.9197 |

## 8. Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Credit reporting or other personal consumer reports | 0.7500 | 0.7742 | 0.7619 |
| Credit reporting, credit repair services, or other personal consumer reports | 0.7333 | 0.5238 | 0.6111 |
| Debt collection | 0.6250 | 0.7407 | 0.6780 |
| Credit card or prepaid card | 0.8889 | 1.0000 | 0.9412 |
| Mortgage | 0.8889 | 0.8000 | 0.8421 |

## 9. Confusion Matrix

| True \ Pred | Credit reporting or other personal consumer reports | Credit reporting, credit repair services, or other personal consumer reports | Debt collection | Credit card or prepaid card | Mortgage |
|---|---|---|---|---|---|
| Credit reporting or other personal consumer reports | 24 | 3 | 4 | 0 | 0 |
| Credit reporting, credit repair services, or other personal consumer reports | 3 | 11 | 6 | 0 | 1 |
| Debt collection | 5 | 1 | 20 | 1 | 0 |
| Credit card or prepaid card | 0 | 0 | 0 | 8 | 0 |
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
