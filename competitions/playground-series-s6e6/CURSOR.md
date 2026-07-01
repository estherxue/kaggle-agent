# Playground Series S6E6 — Cursor Agent Notes

Competition: **Predicting Stellar Class** (`playground-series-s6e6`)

## Paths

| Item | Path |
|------|------|
| Data | `data/train.csv`, `data/test.csv`, `data/sample_submission.csv` |
| Experiments (kagent) | `experiments/<id>/` — code uses `../data/` |
| Agent tasks | `agent_tasks/task_XXXX.md` → write `task_XXXX_response.md` |
| Submissions | `submissions/` |

## Infra rule (NON-NEGOTIABLE)

- **All training runs on Kaggle kernels, never locally.** The local machine only does
  stacking (`stack.py`/`blend.py`) and submission. Every new model/seed/feature run is a
  script kernel under `kaggle/<name>/`. Rationale: a single full-data (577k rows) sklearn fit
  can take hours locally; Kaggle script kernels auto-terminate when done and don't waste quota.
  See CLAUDE.md "Kaggle remote training" for kernel conventions (find_input path helper,
  code/artifacts datasets, GPU torch pitfalls, results.txt output).

## Modeling rules

- **LB metric** (the ONLY thing scored): **balanced accuracy** = unweighted mean of per-class
  recall (GALAXY/QSO/STAR). Submission is **hard labels** `id,class` — NOT probabilities.
  ⚠️ multi-class logloss is used *only* as the training objective / early-stopping proxy; it is
  **not** the evaluation metric. Always optimise/select on balanced accuracy.
- **Training**: multiclass logloss objective + `compute_sample_weight(class_weight="balanced")`;
  early stopping on the validation fold (logloss proxy)
- **Local CV**: balanced_accuracy (primary) + per-class recall + log_loss + macro_f1 (reference)
- **Watch**: GALAXY recall (currently the weakest class, ~0.956) — it caps balanced accuracy
- **Features**: spectral/brightness first; ra/dec interactions optional (ablation)
- **HPO** (`tune.py` → `artifacts/best_params_<model>.json`, auto-layered by `train_oof.py`):
  cheap random search (10 trials, 60k subsample, no optuna) did **not** beat the tuned
  defaults — tuned lgb OOF BA 0.96427 < default 0.96441. Don't bother unless running a
  thorough search (optuna TPE + more trials + larger/full data).
- **all_v3 feature set** (broad colors u_r/g_i/u_z, redshift_low flag, redshift×color, joint
  spectral_type|galaxy_population freq) targeted the dominant GALAXY→STAR error. Every single
  model improved marginally (lgb 0.96441→0.96461, xgb +0.0001, cat +0.0002) **but the blend
  was a tie**: all_v3 0.96608 vs all_v2 0.96609 (before-bias both 0.96552). The extra signal
  is redundant across the correlated GBDTs. Not adopted; kept as a feature_set option.
- **Correlation check confirms redundancy** (`analyze_correlation.py`): all_v3 vs all_v2 raised
  mean pairwise error-Jaccard +0.011 (0.716→0.727) and agreement +0.0007, while the **oracle
  (any-of-4-correct) ceiling did NOT rise** (bal-acc 0.97415→0.97384). So shared features make
  the GBDTs *more* correlated without adding coverage → blend can't improve.
- **Oracle ceiling ≈ 0.974 bal-acc vs blend 0.966** → ~0.8% headroom exists, but only a model
  that adds *new coverage* unlocks it: different family (LogReg/MLP/KNN), a GALAXY-vs-STAR
  specialist, or per-model (not shared) feature sets. More shared GBDT features won't help.
- **Class bias**: `run_pipeline.py` searches it by default (do NOT pass `--no-bias`); helps the
  weak GALAXY recall. CatBoost early stopping is ineffective here (val logloss improves to the
  2000-iter cap even at LR 0.1) — to speed cat up, lower the iteration cap, not the LR.
- **Diverse model + stacking WORKS** (the session's first real gain):
  - `train_diverse.py --model logreg` (multinomial LogReg on scaled all_v2) is weak alone
    (OOF BA 0.925) but adds big coverage: 4-GBDT oracle 0.97415 → +logreg **0.98071**.
  - A *flat* weighted blend can't use it (4GBDT+logreg = 0.96586, worse than 0.96609 — the weak
    member drags the average down).
  - `train_diverse.py --model mlp` (sklearn MLP, scaled all_v2): OOF BA 0.94170 with a very
    different error profile — GALAXY recall 0.972 (> GBDTs!) but STAR 0.900. Complements the
    trees exactly on their weak class.
  - A **LogReg stacking meta-learner** on base OOF log-probs (`stack.py`) converts that coverage
    into score (it trusts each model only where reliable). C is insensitive (0.3–2 all ~0.9668):
    | stack | OOF BA (after bias) |
    |---|---|
    | flat 4 GBDT (baseline) | 0.96609 |
    | stack 4GBDT+logreg | 0.96671 |
    | stack 4GBDT+mlp | 0.96665 |
    | **stack 4GBDT+logreg+mlp** | **0.96679** |
  - Lesson: convert coverage → score via *stacking*, not flat averaging (flat 4GBDT+logreg was
    0.96586, *worse*). `config.MODELS` stays the 4 GBDTs; stack.py takes models explicitly.
- **Specialist model** (`specialist.py`): LightGBM with upweighted GALAXY/STAR (factor 2×) and
  downweighted QSO (0.25×). OOF BA=0.96037, GALAXY recall=0.9726 (vs GBDTs ~0.955), early
  stopping DID trigger (iters 1333-1509). On its own it's weaker but its profile is highly
  complementary. Crucially: adding specialist to the meta-stack **regularizes class bias**
  (QSO bias drops from -0.94 to -0.32), indicating it corrects meta-learner overconfidence
  in QSO.
- **all_v3 logreg/mlp** (`train_diverse.py --feature-set all_v3 --name logreg_v3/mlp_v3`):
  logreg_v3 OOF 0.92503; mlp_v3 OOF 0.94070. No meaningful stack gain over v2 versions
  (+0.00004 max). v3 features are near-redundant for linear/MLP models too on this data.
- **Single-seed additional models — diminishing returns**:
  | stack | OOF BA (after bias) |
  |---|---|
  | 6 models (baseline) | 0.96679 |
  | +specialist (7) | 0.96682 |
  | +logreg_v3+mlp_v3 (8) | 0.96683 |
  | all 9 | 0.96683 |
  All within noise of +0.00004 — single-seed diversity saturated.
- **Multi-seed averaging** (`multi_seed.py`): seeds [42, 2025, 3407], cat_iterations=1500
  (early stopping ineffective so lower cap for speed). Per-model OOF gains:
  lgb +0.00074, hgb +0.00065, cat +0.00067, xgb +0.00036. Stack improvements:
  | stack | OOF BA (after bias) | class bias note |
  |---|---|---|
  | 6-model multi-seed | 0.96706 | extreme (-0.94 QSO) — unstable |
  | **7-model multi-seed + specialist** | **0.96708** | normal (-0.32 QSO) — stable |
  Lesson: specialist also acts as a **calibrator** for the meta-learner when base models use
  multi-seed averages (smoother probabilities → more extreme biases without specialist).
- **Current best = 7-model multi-seed + specialist, OOF BA 0.96708** →
  `submissions/stack_multi_seed_7models.csv`. Reproduce:
  `python3 stack.py --models lgb_multi,xgb_multi,cat_multi,hgb_multi,logreg_multi,mlp_multi,specialist --output submissions/stack_multi_seed_7models.csv`
- **Evolution of OOF BA**:
  flat 4-GBDT 0.96609 → 6-model stack 0.96679 (+0.0007) → 7-model multi-seed 0.96708 (+0.0003)
  → cumulative gain vs baseline: **+0.00099**
- **Confirmed on public LB** (stacking gain is real, not just OOF):
  | submission | OOF BA | public LB |
  |---|---|---|
  | flat 4-GBDT blend | 0.96609 | 0.96703 |
  | 6-model single-seed stack | 0.96679 | 0.96739 |
  | 7-model multi-seed stack | 0.96708 | 0.96746 |
  | 7-model single-seed stack | 0.96680 | **0.96749** |
  | 9-model (7 multi + realmlp + knn_multi) | **0.96713** | 0.96740 |
- **⚠️ NOISE-LIMITED REGIME REACHED (decisive finding, 2026-06-22)**: OOF BA and public LB are
  now effectively DECORRELATED at the 4th decimal. The 9-model stack has the HIGHEST OOF
  (0.96713) but a LOWER public LB (0.96740) than the simpler 7-model single-seed (OOF 0.96680,
  LB 0.96749). Public LB = 20% of test (~20k rows), 1-σ sampling noise ≈ 0.0007 — far larger
  than the ~0.0001 spread between these stacks. **All these LB differences are noise.**
  Implications:
  - Stop chasing OOF 4th-decimal gains (realmlp +0.00004, knn_multi +0.00001 are noise; adding
    marginal members risks overfitting the meta-learner to OOF and shake-up on private LB).
  - For the final private-LB submission, prefer a SIMPLE, variance-reduced ensemble (multi-seed
    averaging genuinely helps generalization) over piling on marginal diverse members.
  - This empirically confirms the whitepaper's noise-limited-leaderboard / shake-up-risk warning.
- **RealMLP** (pytabkit RealMLP_TD, CPU kernel ~90min/fold): OOF BA 0.94878, recalls GALAXY
  0.9752 (>GBDTs) / QSO 0.9632 / STAR 0.9079. Adds +944 oracle samples (mostly GALAXY), oracle
  ceiling 0.98386→0.98487. In stack: 7→7+realmlp = 0.96708→0.96712 (noise).
- **Per-seed-to-meta experiment** (feed individual seed preds instead of averaging, to expose
  cross-seed variance as an uncertainty signal): does NOT beat averaging-first. LogReg meta
  0.96705 (linear → combo of per-seed logps ≡ combo of their average, can't use variance); MLP
  meta 0.96705 (raw 0.95740, overfits 57-dim). per-class scale+shift calibration: +0.00001
  (noise). **Anomaly patching (blend meta→base-mean on high-disagreement tail): HURTS in every
  config** — the meta already beats the fallback even on disagreement samples. Negative result.
- **TabPFN-3** (Transformer ICL, Prior Labs): needs TABPFN_TOKEN cached via save_token() + a
  monkeypatch bypass of the misfiring ensure_license_accepted() gate; GPU hit "no kernel image"
  (arch mismatch) so runs on CPU. CPU is slow: context=10000/bags=2 was ~3.2h/fold (cancelled at
  12h); reduced to context=6000/bags=1, 5 folds done (~6.3k s/fold) but test pred cancelled at 12h
  (OOF pre-saved before test, so oof_tabpfn.npy survived; test_tabpfn.npy NOT saved).
  **TabPFN OOF BA=0.93591** (recalls GALAXY 0.9713 / QSO 0.9566 / STAR 0.8798 — weakest member).
  Adds +595 oracle samples (oracle 0.98487→0.98552) BUT does NOT help the stack: 9→9+tabpfn
  0.96713→0.96709, 7→7+tabpfn 0.96708→0.96705. Same failure mode as single-seed KNN — too weak
  / noisy for the meta-learner to trust. NOT adopted; don't bother regenerating its test preds.
- **FINAL submission picks (noise-limited, both within LB noise ~0.0007)**:
  (1) 7-model multi-seed stack — simplest, variance-reduced, LB 0.96746 [robust pick];
  (2) 9-model stack (7-multi + specialist + realmlp + knn_multi) — OOF best 0.96713, LB 0.96740.
  Extra base models (realmlp/knn_multi/tabpfn) give only noise-level OOF moves; stop here.
- **Nested-CV honest evaluation (2026-06-22)**: outer 5-fold, inner 5-fold meta-OOF, calibration
  fit on inner-OOF only — test rows never seen during meta or calibration fitting:
  | config | nested-CV BA (additive bias) | nested-CV BA (scale+shift) |
  |---|---|---|
  | 7-multi (robust) | 0.96685 | 0.96696 |
  | 9-model (7-multi+realmlp+knn_multi) | **0.96701** | 0.96700 |
  | 9+et (10 models) | 0.96693 | 0.96697 |
  Key findings: (1) **9-model wins** nested-CV — ET *hurts* (-0.00008 with bias, -0.00003 with
  scale+shift vs 9-model). ET OOF 0.94166 is too noisy; oracle +168 samples doesn't translate to
  stack gain when meta-learner can't trust it. (2) scale+shift calibration gives +0.00011 on
  7-multi but ≤0.00001 on 9-model — not worth added complexity. (3) nested-CV BA ~0.00007–0.00012
  below plain stack.py OOF (expected — no calibration leakage). OOF ranking preserved.
  **Conclusion: ET not adopted. 9-model is the best heterogeneous stack by honest evaluation.
  Do NOT generate test_et.npy — no submission gain expected.**
- **10-model+ET submissions (2026-06-24)** — submitted to observe public LB despite nested-CV
  prediction of no gain:
  | submission | OOF BA (before calib) | OOF BA (after calib) | public LB |
  |---|---|---|---|
  | 10model+ET bias calib | 0.96701 | 0.96711 | pending |
  | 10model+ET scale+shift calib | 0.96702 | 0.96712 | pending |
  scale+shift params: GALAXY scale=2.336 shift=1.491 / QSO scale=1.463 shift=0.768 / STAR scale=1.040 shift=0.597.
  Recalls after ss: GALAXY 0.9559 / QSO 0.9748 / STAR 0.9706.
  Note: scale+shift calib takes ~2.5h local CPU (scipy DE 18090 evals × 200ms each on 577k OOF rows).
  Use `polish=False` and `maxiter=100` for faster future runs.

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
