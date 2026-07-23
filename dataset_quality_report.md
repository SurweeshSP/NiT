# Dataset Quality Audit Report

**Initial Raw Records:** 1071

**Noisy/Empty Samples Removed:** 1
**Duplicate Complaints Removed:** 72
**Records after basic cleaning:** 998

### Initial Class Distribution (Top 5 Classes)
- **Credit reporting or other personal consumer reports**: 332
- **Credit reporting, credit repair services, or other personal consumer reports**: 288
- **Debt collection**: 129
- **Credit card or prepaid card**: 50
- **Mortgage**: 45

**Balancing Strategy:** Downsample majority classes to median count (129). Minority classes are kept as is.

### Final Balanced Class Distribution
- **Credit reporting or other personal consumer reports**: 129
- **Credit reporting, credit repair services, or other personal consumer reports**: 129
- **Debt collection**: 129
- **Credit card or prepaid card**: 50
- **Mortgage**: 45

**Final Trainable Records:** 482
**Train Split:** 385
**Test Split:** 97