# Final Experimental Report — Nature-Inspired Augmentation Selection

**Generated**: 2026-07-21 06:50 UTC

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
| Average FACI Scalar | 0.2075 |
| Average Complexity | 0.5000 |
| Average Semantic Entropy | 0.8000 |

## 4. Policy Statistics

| Statistic | Value |
|---|---|
| Total Policies Generated | 12 |

### Strategy Breakdown

| Strategy | Count |
|---|---|
| No Augmentation | 2 |
| BERT Contextual | 1 |
| Back Translation | 1 |
| Hybrid | 8 |

## 5. Optimizer Summary

| Metric | Value |
|---|---|
| Average Expected Utility | 0.6274 |
| Final Fitness Score | 0.7599 |
| Average Runtime (s/chunk) | 14.13 |
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
| Accuracy | 0.5876 |
| Macro Precision | 0.6330 |
| Macro Recall | 0.6249 |
| **Macro F1** | **0.6263** |
| Weighted F1 | 0.5890 |
| ROC AUC | 0.8185 |

## 8. Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Credit reporting or other personal consumer reports | 0.5625 | 0.5806 | 0.5714 |
| Credit reporting, credit repair services, or other personal consumer reports | 0.4348 | 0.4762 | 0.4545 |
| Debt collection | 0.6400 | 0.5926 | 0.6154 |
| Credit card or prepaid card | 0.7778 | 0.8750 | 0.8235 |
| Mortgage | 0.7500 | 0.6000 | 0.6667 |

## 9. Confusion Matrix

| True \ Pred | Credit reporting or other personal consumer reports | Credit reporting, credit repair services, or other personal consumer reports | Debt collection | Credit card or prepaid card | Mortgage |
|---|---|---|---|---|---|
| Credit reporting or other personal consumer reports | 18 | 8 | 5 | 0 | 0 |
| Credit reporting, credit repair services, or other personal consumer reports | 6 | 10 | 3 | 1 | 1 |
| Debt collection | 7 | 4 | 16 | 0 | 0 |
| Credit card or prepaid card | 0 | 0 | 0 | 7 | 1 |
| Mortgage | 1 | 1 | 1 | 1 | 6 |

## 10. Discussion

The Hybrid GA-GWO framework adaptively allocated augmentation budgets according to FACI-derived semantic complexity, improving Macro F1 over all static baselines.

## 11. Limitations

Dataset size (150 samples) limits statistical power. CPU-only inference restricts throughput.

## 12. Conclusion

Extension to larger corpora; integration of Butterfly Optimisation Algorithm (BOA); dynamic class-imbalance weighting.

---
*Report generated automatically by the pipeline. For statistical analysis across
multiple seeds, see `results/multi_seed/multi_seed_summary.csv`.*
