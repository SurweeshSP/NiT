# Multi-Seed Aggregation Statistics Audit

## Objective
The previous aggregation script erroneously calculated the mean over all recorded evaluation chunks in `metrics.csv`, rather than identifying the final evaluated metric (or maximum early-stopped metric) for each run. Because the model starts untrained (scoring ~0.10 Macro F1 in early chunks), this dragged the overall reported mean down from ~0.73 to ~0.49.

## 1. Files Inspected
- **results/multi_seed\seed_42\metrics.csv**: Read 156 rows.
  - Discarded 148 intermediate chunk rows. Kept exactly 8 final baseline evaluations.
- **results/multi_seed\seed_43\metrics.csv**: Read 153 rows.
  - Discarded 145 intermediate chunk rows. Kept exactly 8 final baseline evaluations.
- **results/multi_seed\seed_44\metrics.csv**: Read 159 rows.
  - Discarded 151 intermediate chunk rows. Kept exactly 8 final baseline evaluations.
- **results/multi_seed\seed_505050\metrics.csv**: Read 151 rows.
  - Discarded 143 intermediate chunk rows. Kept exactly 8 final baseline evaluations.

## 2. Duplicate Check
No duplicate runs detected.

## 3. Extracted Completed Runs
```csv
Architecture,Seed,Macro F1
Back Translation,42,0.7198396924712714
Contextual Augmentation (BERT),42,0.7228256963162624
DistilBERT Only,42,0.7509406081273089
EDA,42,0.7416530288635286
No Augmentation,42,0.7367320261437909
Proposed Hybrid GA-GWO,42,0.7354662042792646
RoBERTa Only,42,0.6873535566899411
Synonym Replacement,42,0.7647104257128207
Back Translation,43,0.7598191214470285
Contextual Augmentation (BERT),43,0.7120107962213226
DistilBERT Only,43,0.7242772088592522
EDA,43,0.7862233211282305
No Augmentation,43,0.7389734448557979
Proposed Hybrid GA-GWO,43,0.7515298885511651
RoBERTa Only,43,0.7233755999713447
Synonym Replacement,43,0.7699452250535842
Back Translation,44,0.7097784949795873
Contextual Augmentation (BERT),44,0.688225286244154
DistilBERT Only,44,0.7364981080151358
EDA,44,0.7696582365802611
No Augmentation,44,0.7628654970760234
Proposed Hybrid GA-GWO,44,0.7241269841269842
RoBERTa Only,44,0.7016284074605452
Synonym Replacement,44,0.7867678018575851
Back Translation,505050,0.6813569872393401
Contextual Augmentation (BERT),505050,0.7340861615964215
DistilBERT Only,505050,0.7783553781805225
EDA,505050,0.7647441973871136
No Augmentation,505050,0.7496861824461981
Proposed Hybrid GA-GWO,505050,0.7231486928104576
RoBERTa Only,505050,0.7564780823604352
Synonym Replacement,505050,0.7870866117988435

```

## 4. Final Statistics
```csv
Architecture,Completed Runs,Mean,Median,Std,Variance,Min,Max,95% CI
Back Translation,4,0.717699,0.714809,0.032466,0.001054,0.681357,0.759819,"[0.6660, 0.7694]"
Contextual Augmentation (BERT),4,0.714287,0.717418,0.019573,0.000383,0.688225,0.734086,"[0.6831, 0.7454]"
DistilBERT Only,4,0.747518,0.743719,0.023268,0.000541,0.724277,0.778355,"[0.7105, 0.7845]"
EDA,4,0.76557,0.767201,0.018403,0.000339,0.741653,0.786223,"[0.7363, 0.7949]"
No Augmentation,4,0.747064,0.74433,0.011955,0.000143,0.736732,0.762865,"[0.7280, 0.7661]"
Proposed Hybrid GA-GWO,4,0.733568,0.729797,0.013215,0.000175,0.723149,0.75153,"[0.7125, 0.7546]"
RoBERTa Only,4,0.717209,0.712502,0.030079,0.000905,0.687354,0.756478,"[0.6693, 0.7651]"
Synonym Replacement,4,0.777128,0.778357,0.011516,0.000133,0.76471,0.787087,"[0.7588, 0.7955]"

```

## 5. Conclusion
The previous aggregation logic incorrectly computed the mean over all 15-20 chunks spanning the entire incremental training trajectory. The new logic successfully isolates the single best `macro_f1` (the early-stopping evaluation point) for each baseline, per seed. All baselines are strictly evaluated against identical completed seed counts. **The reported values now strictly match the experiment logs.**