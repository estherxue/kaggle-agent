# Playground Series S6E6 — Cursor Agent Notes

Competition: **Predicting Stellar Class** (`playground-series-s6e6`)

## Paths

| Item | Path |
|------|------|
| Data | `data/train.csv`, `data/test.csv`, `data/sample_submission.csv` |
| Experiments (kagent) | `experiments/<id>/` — code uses `../data/` |
| Agent tasks | `agent_tasks/task_XXXX.md` → write `task_XXXX_response.md` |
| Submissions | `submissions/` |

## Modeling rules

- **LB metric**: multiclass log loss — submit class probabilities
- **Training**: multiclass logloss + `compute_sample_weight(class_weight="balanced")`
- **Local CV**: log_loss + balanced_accuracy + macro_f1 + macro_recall + confusion matrix
- **Watch**: Quasar (likely minority class) per-class recall
- **Features**: spectral/brightness first; ra/dec interactions optional (ablation)

## Response format (for agent_tasks)

```markdown
## HYPOTHESIS
One sentence.

## CODE
```python
# runnable code
```
```

## Commands

```bash
kagent run playground-series-s6e6
kagent task playground-series-s6e6
kagent continue playground-series-s6e6
python baseline.py
```
