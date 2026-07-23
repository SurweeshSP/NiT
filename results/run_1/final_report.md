# Final Experimental Report — Nature-Inspired Augmentation Selection

**Generated**: 2026-07-21 06:32 UTC

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
| Average FACI Scalar | 0.2068 |
| Average Complexity | 0.5000 |
| Average Semantic Entropy | 0.8000 |

## 4. Policy Statistics

| Statistic | Value |
|---|---|
| Total Policies Generated | 10 |

### Strategy Breakdown

| Strategy | Count |
|---|---|
| No Augmentation | 1 |
| BERT Contextual | 3 |
| Back Translation | 3 |
| Hybrid | 3 |

## 5. Optimizer Summary

| Metric | Value |
|---|---|
| Average Expected Utility | 0.5603 |
| Final Fitness Score | 0.6888 |
| Average Runtime (s/chunk) | 9.55 |
| Average Peak Memory (MB) | 6.6 |

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
| Accuracy | 0.5670 |
| Macro Precision | 0.6103 |
| Macro Recall | 0.5928 |
| **Macro F1** | **0.5938** |
| Weighted F1 | 0.5563 |
| ROC AUC | 0.8310 |

## 8. Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Credit reporting or other personal consumer reports | 0.5366 | 0.7097 | 0.6111 |
| Credit reporting, credit repair services, or other personal consumer reports | 0.4615 | 0.2857 | 0.3529 |
| Debt collection | 0.5600 | 0.5185 | 0.5385 |
| Credit card or prepaid card | 0.8571 | 0.7500 | 0.8000 |
| Mortgage | 0.6364 | 0.7000 | 0.6667 |

## 9. Confusion Matrix

| True \ Pred | Credit reporting or other personal consumer reports | Credit reporting, credit repair services, or other personal consumer reports | Debt collection | Credit card or prepaid card | Mortgage |
|---|---|---|---|---|---|
| Credit reporting or other personal consumer reports | 22 | 4 | 5 | 0 | 0 |
| Credit reporting, credit repair services, or other personal consumer reports | 9 | 6 | 4 | 1 | 1 |
| Debt collection | 9 | 3 | 14 | 0 | 1 |
| Credit card or prepaid card | 0 | 0 | 0 | 6 | 2 |
| Mortgage | 1 | 0 | 2 | 0 | 7 |

## 10. Discussion

The Hybrid GA-GWO framework adaptively allocated augmentation budgets according to FACI-derived semantic complexity, improving Macro F1 over all static baselines.

## 11. Limitations

Dataset size (150 samples) limits statistical power. CPU-only inference restricts throughput.

## 12. Conclusion

Extension to larger corpora; integration of Butterfly Optimisation Algorithm (BOA); dynamic class-imbalance weighting.

---
*Report generated automatically by the pipeline. For statistical analysis across
multiple seeds, see `results/multi_seed/multi_seed_summary.csv`.*
