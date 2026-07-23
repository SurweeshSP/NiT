# Final Experimental Report — Nature-Inspired Augmentation Selection

**Generated**: 2026-07-21 06:44 UTC

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
| Average FACI Scalar | 0.2073 |
| Average Complexity | 0.5000 |
| Average Semantic Entropy | 0.8000 |

## 4. Policy Statistics

| Statistic | Value |
|---|---|
| Total Policies Generated | 9 |

### Strategy Breakdown

| Strategy | Count |
|---|---|
| No Augmentation | 2 |
| Back Translation | 2 |
| BERT Contextual | 2 |
| Hybrid | 3 |

## 5. Optimizer Summary

| Metric | Value |
|---|---|
| Average Expected Utility | 0.5517 |
| Final Fitness Score | 0.7452 |
| Average Runtime (s/chunk) | 13.73 |
| Average Peak Memory (MB) | 1.0 |

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
| Accuracy | 0.4845 |
| Macro Precision | 0.5835 |
| Macro Recall | 0.5221 |
| **Macro F1** | **0.5446** |
| Weighted F1 | 0.4896 |
| ROC AUC | 0.7809 |

## 8. Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Credit reporting or other personal consumer reports | 0.5000 | 0.3871 | 0.4364 |
| Credit reporting, credit repair services, or other personal consumer reports | 0.3077 | 0.3810 | 0.3404 |
| Debt collection | 0.4848 | 0.5926 | 0.5333 |
| Credit card or prepaid card | 1.0000 | 0.7500 | 0.8571 |
| Mortgage | 0.6250 | 0.5000 | 0.5556 |

## 9. Confusion Matrix

| True \ Pred | Credit reporting or other personal consumer reports | Credit reporting, credit repair services, or other personal consumer reports | Debt collection | Credit card or prepaid card | Mortgage |
|---|---|---|---|---|---|
| Credit reporting or other personal consumer reports | 12 | 12 | 7 | 0 | 0 |
| Credit reporting, credit repair services, or other personal consumer reports | 6 | 8 | 6 | 0 | 1 |
| Debt collection | 6 | 5 | 16 | 0 | 0 |
| Credit card or prepaid card | 0 | 0 | 0 | 6 | 2 |
| Mortgage | 0 | 1 | 4 | 0 | 5 |

## 10. Discussion

The Hybrid GA-GWO framework adaptively allocated augmentation budgets according to FACI-derived semantic complexity, improving Macro F1 over all static baselines.

## 11. Limitations

Dataset size (150 samples) limits statistical power. CPU-only inference restricts throughput.

## 12. Conclusion

Extension to larger corpora; integration of Butterfly Optimisation Algorithm (BOA); dynamic class-imbalance weighting.

---
*Report generated automatically by the pipeline. For statistical analysis across
multiple seeds, see `results/multi_seed/multi_seed_summary.csv`.*
