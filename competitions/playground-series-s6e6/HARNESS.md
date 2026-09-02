# S6E6 — What actually made the agent effective

Harness retrospective for **Playground Series S6E6 (Predicting Stellar Class)**.
Result: **4th / 2817**, private 0.97060, honest nested-CV 0.97045.

Scope: the scaffolding around the modeling work, not the modeling itself. Every entry is
here because it demonstrably changed the outcome. The anti-inventory at the end lists what
was measured and did *not* work, so it is not rebuilt.

Companion docs in this directory: `CURSOR.md` (operating rules), `LEADER_GAP_ANALYSIS.md`
(timestamped experiment log), `eda_report.md`.

> The one-line thesis: **the leverage was never in the model code.** It was a contract that
> let independently-produced artifacts compose, a remote executor that turned a laptop into a
> fleet, and an honest metric that refused to be fooled by the public leaderboard.

---

## 1. The alignment contract — the highest-leverage engineering here

**Where:** `config.py`, `kaggle/code_ds/kernel_common.py` (419 lines).

One frozen contract:

```
StratifiedKFold(5, shuffle=True, random_state=42) on integer y, in train-CSV row order
GALAXY=0 / QSO=1 / STAR=2
OOF (577347, 3) float32   |   test (247435, 3) float32
artifacts/oof_<name>.npy  +  artifacts/test_<name>.npy
```

The split depends **only on `y` and the seed** — not on features, not on the model, not on
who ran it. Consequence: any OOF array produced by any kernel, on any day, by any model
family, aligns row-for-row with every other one, for free.

What that bought:

- `stack.py --models a,b,c,...` — compose any subset with a CLI flag. **167 `.npy` artifacts,
  30 kernels, a final 48-model pool.**
- **Third-party OOFs drop straight in.** 19 published OOF arrays from a Kaggle GM were stacked
  without a line of glue code — worth **+0.00075 honest CV** (BASE24 linear 0.96954 → pool44
  0.97029), the largest late-stage gain.
- Shape verification becomes mechanical: a wrong-shaped `.npy` is a caught error, not a
  silently-misaligned stack.

`kernel_common.py` is the contract made executable: `stratkfold()` is the single source of
truth for the split, `save_oof_test()` the single writer, `reconstruct_sdss_cats()` the shared
categorical reconstruction. Kernels `from kernel_common import ...` instead of copy-pasting
boilerplate into 21 `run.py` files.

> **Rule:** before generating the *second* artifact of any kind, freeze the contract that lets
> artifact #1 and #47 combine. Retrofitting alignment costs far more than declaring it.

---

## 2. Remote execution — laptop as orchestrator, cloud as compute

**Where:** `kaggle/` (30 kernel dirs, `sync_code.sh`, `run_remote.sh`),
`src/kaggle_agent/tools/kernel_fleet.py` (1000 lines, repo root).

```
sync_code.sh      →  copy 13 source files into code_ds/ → `kaggle datasets version`
                     (code ships as a DATASET, not duplicated per kernel)
kaggle/<name>/    →  one experiment = one dir + kernel-metadata.json + run.py
run_remote.sh     →  push → poll status → on complete, `kernels output` pulls .npy
local             →  stack/blend (pure numpy, seconds) + submit
```

The hard rule that made this necessary: **never train locally.** A single sklearn fit on
577k rows takes hours on the laptop; Kaggle script kernels auto-terminate and don't waste
quota.

### `kernel_fleet.py` — the part worth keeping

Bash (`run_new.sh`, `poll_all.sh`) got the job done but died on every edge case. The Python
driver encodes each edge case as tested logic:

| Component | Problem it solves |
|---|---|
| `gpu_cap_admits()` | Kaggle allows **2** concurrent GPU batch sessions; a 3rd push fails. Pure predicate, so the queue is testable without credentials. CPU kernels bypass the cap. |
| `classify_push_output()` | Distinguishes "GPU cap hit, retry later" from "slug is dead" from a real error. |
| `bump_slug()` | **Slug poisoning**: a kernel whose *first* push failed returns "Notebook not found" forever. Only fix is a new slug — automated. |
| `pull_and_verify()` | Kernel outputs can be truncated. Every expected `.npy` is `np.load`ed and shape-checked; re-pull up to 3×. |
| `diagnose_log()` | 6 known failure signatures → the actual fix text. |
| `run_fleet()` | Pending queue + poll + pull-on-complete + diagnose-on-fail. **One bad kernel never takes down the fleet.** |

The six diagnoses, each of which cost failed GPU kernels to learn:

| signature | cause | fix |
|---|---|---|
| `no_kernel_image` | Kaggle GPU is a **P100 (sm_60)**; stock torch ≥2.10+cu128 ships no sm_60 image | `pip install torch==2.4.1` from the cu121 index **before** `import torch`; some libs then need `device='cuda:0'`, not bare `'cuda'` |
| `cudf_pascal_dropped` | cuDF/RAPIDS also dropped Pascal → `invalid device ordinal` | all feature engineering in pandas/numpy. CatBoost-GPU and XGBoost-GPU still work on P100 |
| `slug_poisoned` | first push failed → slug half-registers | re-push under a new hyphenated slug |
| `catboost_float_cat_features` | float-typed categorical columns | cast to int, or let CatBoost infer |
| `bad_oof_shape` | array saved with wrong shape | `reshape(-1, 3)` before save; enforce the contract in §1 |
| `oom` | memory | reduce batch/dtype, or chunk |

Validated end-to-end on real Kaggle: 6 kernels including GPU-cap=2 staging and a
transient-CLI-error auto-retry. A mass-FE kernel hit the 12h wall and was cancelled — its OOF
had been saved first, so it survived and pulled cleanly.

> **Rule:** the remote executor's job is not to run jobs, it is to **survive them failing**.
> Budget the same effort for the failure catalogue as for the happy path.

**Known gap:** the fleet driver ran as a foreground process and was killed on an overnight
session restart. Kernels survived (they run on Kaggle) and their OOFs had already been pulled,
but the driver should be a tracked background job.

---

## 3. Honest evaluation — scored zero points, decided the competition

**Where:** `nested_cv.py`, `select_final.py`, `metrics.py`, `hillclimb.py`.

`stack.py` fits class calibration on the full meta-OOF then reports BA on that same data —
optimistic. `nested_cv.py` holds every test row out of **both** the meta fit and the
calibration fit. The gap it exposed:

| Ensembler | plain OOF | honest nested-CV | optimism gap |
|---|---|---|---|
| linear meta (bias-calibrated) | 0.97040 | 0.96954 | **0.00090** |
| GBDT meta (heavy reg) | 0.97039 | 0.97022 | 0.00017 |
| hill-climb (bagged Caruana) | 0.97012 | 0.97002 | 0.00011 |
| gbdt_meta_pool48 | 0.97043 | 0.97037 | 0.00006 |

Then the private leaderboard settled it:

| submission | honest CV | public | **private** | drop |
|---|---|---|---|---|
| strat_multilevel_pool48 | **0.97045** ← best | 0.97079 | **0.97060** ← best | −0.00019 |
| stack_gbdtmeta_pool48 | 0.97037 | **0.97104** ← best | 0.97047 | −0.00057 |
| stack_gbdtmeta (24) | 0.97022 | 0.97050 | 0.97047 | −0.00003 |
| stack_pool44 (linear) | 0.97029 | 0.97072 | 0.97046 | −0.00026 |
| stack_24_final (linear) | 0.96954 | 0.97077 | 0.97035 | −0.00042 |
| stack_hillclimb | 0.97002 | 0.97055 | 0.97024 | −0.00031 |

Three findings that generalize:

1. **The highest honest-CV model was the highest private model.** Public LB — a single noisy
   draw on ~49k rows, σ ≈ 0.00087 — ranked them wrong.
2. **Shakedown was proportional to public-minus-honest-CV excess.** +0.00034 excess → fell
   0.00019; +0.00067 → fell 0.00057; +0.00123 → fell 0.00042. The optimism gap is a
   *predictor*, not just a diagnostic.
3. **Kaggle scores `max(selected)` on private.** Picking by honest CV *and* keeping one
   decorrelated hedge secured 4th. Chasing the best public model alone → 0.97047, worse rank.

### The anti-hack rule

The top public cluster (0.9724–0.9728) ran **"Ridge Flip + Probability Consensus"**: Ridge
fitted directly onto past submissions' *public scores* to reverse-engineer which per-row label
flips raise the public 20%, iterating on live LB feedback. Ruled out as public-LB probing. The
whole board shook down on private; that refusal is *how* an honest 0.97060 reached 4th.

> **Rule:** when optimizing against a metric you can query repeatedly, build the honest
> estimator *before* you need it, and write down in advance that final selection is by honest
> CV. The temptation to pick the pretty public number arrives late, under time pressure.

---

## 4. The rules layer — `CLAUDE.md` / `CURSOR.md`

Not documentation. The only text *guaranteed* to be in context at the start of every session,
so it is where constraints go that must survive a `/compact`, a crash, or a fresh session days
later.

| Kind | Example | Why it must be a rule, not a memory |
|---|---|---|
| Irreversible-risk gate | "Never commit an active competition's solution" (`.gitignore` = `*`) | The remote is public. One `git push` leaks the approach. |
| Resource-shape rule | "Never train locally — all training on Kaggle kernels" | One sklearn fit on 577k rows = hours locally, minutes remotely. |
| Metric lock | "The metric is balanced accuracy, NOT logloss" | Cost a real bug: early-stopping on logloss while selecting on BA. |
| Environment landmine | the six `diagnose_log` signatures above | Each cost multiple failed GPU kernels before it was written down. |

> **Rule:** any fact that cost more than one failed run to learn belongs in `CLAUDE.md` the
> same day. The test is not "is this interesting" but **"would a fresh session repeat this
> mistake?"**

---

## 5. Adversarial pre-review of untestable code

Because no kernel could be run locally, a kernel's first execution was on Kaggle, burning
quota. Mitigation: **author, then adversarially review each kernel before pushing** — a second
agent whose only job is to refute it.

Caught, in code that `py_compile` and `ast.parse` both passed:
- a logloss-vs-balanced-accuracy early-stopping bug,
- a dropped-function `NameError`,
- a false ceiling claim.

> **Rule:** when the feedback loop costs money or hours, insert an adversarial reviewer
> *before* execution. Syntax checks do not catch semantic bugs, and semantic bugs are exactly
> what an expensive loop punishes.

---

## Anti-inventory: measured, and did not work

### Modeling dead ends

| Attempt | Standalone OOF | Contribution | Note |
|---|---|---|---|
| Mass feature engineering (243 feats, 76 groupby aggs) | 0.96539 | **+0** | 0.984 argmax-agreement with the best base; also hit the 12h CPU wall |
| Optuna HPO (60–100 trials) | 0.96506 / 0.96562 | **+0** | Lost to hand-tuned catv3 0.96877. A cheap 10-trial search was *negative*: 0.96427 < 0.96441 default |
| AutoGluon (good_quality, bagged) | 0.95646 | **+0** | |
| DAE representation learning (swap-noise) | 0.96444 | **+0** | 8 clean features: nothing to learn |
| Hierarchical cascade (STAR-vs-rest → GALAXY-vs-QSO) | 0.96335 | **net loss** | Moves recalls around, adds no coverage |
| Pseudo-labeling (transductive, conf ≥ 0.995) | 0.96561 / 0.96538 | **+0** | Identical to plain GBDTs |
| ICL family (TabPFN-3 / TabICL) | 0.93591 / 0.95797 | **negative** | 30 models 0.97041 → 31 models 0.97040 |
| Calibration & decision rules | — | +0.00004 / +0.00001 / **−0.00065** | scale+shift, temperature, equal-recall. Additive bias was already optimal — BA-optimal recalls are *unequal* (G 0.959 < Q 0.975 < S 0.977) |
| Per-seed features into the meta | 0.96705 | −0.00003 | Provably redundant: a linear meta over per-seed log-probs ≡ over their average |
| Anomaly patching / probability re-vote | — | **hurts in every config** | The meta already beats the fallback on high-disagreement rows |
| ExtraTrees | 0.94166 | −0.00008 | Rejected by nested-CV *before* submission |
| CatBoost early stopping | — | **never fires** | Val logloss falls to the 2000-iter cap even at LR 0.1. The only speed lever is a lower iteration cap, not a lower LR |

Two structural lessons underneath the table:

- **More correlated features make the ensemble worse, not better.** The `all_v3` feature set
  lifted every single model (lgb +0.0002) but the blend was a *tie* (0.96608 vs 0.96609).
  `analyze_correlation.py` gave the mechanism: pairwise error-Jaccard rose 0.716 → 0.727 while
  the oracle ceiling *fell* 0.97415 → 0.97384. **Track the oracle ceiling, not the
  single-model score, when deciding whether to add a member.**
- **The honest ceiling is provable and worth proving.** 24 own models + a GM's 19 published
  OOFs + a GBDT meta + a transformer-ICL model all landed at OOF ~0.9704. At that point the
  correct action is to report the ceiling, not to keep grinding.

### What *did* move the score, for contrast

| Lever | Gain |
|---|---|
| Original SDSS17 augmentation (wt ~0.1, train folds only, never validation) | **+0.0015 … +0.004 per model** — the single biggest lever |
| Strong deep models (RealMLP R2-103, DCN, TabM) | single model 0.96908 ≈ the entire prior stack; the old broken RealMLP was 0.94878 |
| Logit multinomial-LR stacking (not flat blending, not log-probs) | +0.0007 — and a flat blend with the same members was *worse* (0.96586) |
| Broadening the pool to 48 (GM's published OOFs) | +0.00075 honest CV |
| Heavily-regularized GBDT meta (leaves 8, min_child 1000, λ 20, 5-seed) | +0.0007 honest CV on the 24-pool; best public on the 48-pool |

### Process failures

- **Noise-limited regime was detected on 06-22 and then ignored for days.** The 9-model stack
  had the best OOF (0.96713) but a *worse* public LB (0.96740) than a 7-model stack (0.96680 →
  0.96749). Every subsequent "add one more marginal model" was inside the noise.
  **Once 4th-decimal OOF stops tracking LB, stop adding members; switch to variance reduction
  and honest selection.**
- **The best public submission was almost never generated.** `gbdt_meta_pool48` existed only
  as a *level-1* component inside a level-2 meta-of-metas. Its standalone honest CV (0.97037)
  was within 8e-5 of the ensemble containing it (0.97045) — the number was seen and not acted
  on, assuming the ensemble subsumed it. It did not: standalone scored public 0.97104, beating
  the ensemble that contained it by +0.00025.
  **Rule: whenever a sub-component's honest score is within ~1σ of the ensemble built on top
  of it, materialize and submit it standalone.**
  Compounding cause: the file was produced by parallel work after the last `submissions/` scan
  and a context compaction hid it. **Rule: re-inventory `submissions/` and `artifacts/` before
  declaring a sweep exhausted — a sweep is only as complete as the last directory listing.**
