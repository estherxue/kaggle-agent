# S6E6 EDA Summary

## Data

| File | Rows | Cols |
|------|------|------|
| train.csv | 577,347 | 12 |
| test.csv | 247,435 | 11 |

## Target (`class`)

| Class | Share |
|-------|-------|
| GALAXY | 65.4% |
| QSO | 20.3% |
| STAR | 14.3% (minority) |

## LB metric

**Balanced accuracy** (per-class recall average). Submit `id, class` labels.

## Baseline v1 (LGB+XGB, logloss选模)

| | OOF BA | Public LB |
|--|--------|-----------|
| baseline | 0.9652 | 0.96592 |

## Blend v1 (LGB+XGB+CAT + weights + class bias)

| Model | OOF BA (argmax) |
|-------|-----------------|
| LGB | 0.96478 |
| XGB | 0.96427 |
| CAT | 0.95947 |

**Blend weights**: LGB 0.588, XGB 0.408, CAT 0.004  
**Class bias**: GALAXY +0.20, QSO +0.29, STAR +0.43  

| | OOF BA | per-class recall (G/Q/S) |
|--|--------|---------------------------|
| before bias | 0.96496 | 0.954 / 0.975 / 0.966 |
| **after bias** | **0.96501** | 0.950 / 0.975 / **0.971** |

Submission: `submissions/blend_v1.csv` — see Kaggle for LB score.

## Next steps (not yet done)

- Target OOF BA ≥ 0.9675: need stronger features or binary-chain
- CatBoost underperformed OOF; weight near zero
