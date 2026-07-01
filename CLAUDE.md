# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⛔ Git policy for ACTIVE competitions (HARD RULE)

**Never commit or push an active competition's solution to git until the competition has ended.**
This prevents leaking the approach to competitors (the repo has a public GitHub remote `origin`).

- The whole active-competition directory is gitignored. For S6E6 the file
  `competitions/playground-series-s6e6/.gitignore` contains `*` (ignore everything in that dir),
  which overrides the parent `competitions/.gitignore` rule that would otherwise track `*.py`/`*.md`.
- The previously-tracked solution files have been **untracked** (`git rm --cached`, kept on disk).
- Do **not** run `git add` / `git commit` / `git push` on anything under an active competition dir.
  Do not "helpfully" re-add them. Only AFTER the competition closes: relax/delete that `.gitignore`,
  re-add the files, and commit.
- Note: an early baseline of S6E6 was already pushed to `origin/main` (commit 6120a08) before this rule.
  Leave it unless the user explicitly asks to scrub it from the remote.

## Commands

```bash
# Install (editable)
pip install -e .

# Run agent
kagent run <competition-slug>          # start fresh
kagent run <competition-slug> --resume  # resume from saved state
kagent continue <competition-slug>      # resume after Cursor writes a task response
kagent guide <competition-slug> "..."  # inject guidance while running
kagent status <competition-slug>
kagent task <competition-slug>         # show current pending Cursor task file

# Tests
pytest                                  # all tests
pytest tests/unit/                      # unit only
pytest tests/unit/test_config.py        # single file

# Lint / format
ruff check .
black .
```

## Architecture

This repo has two distinct layers that rarely overlap:

### 1. Agent Framework (`src/kaggle_agent/`)

An autonomous Kaggle competition agent. The lifecycle is driven by `Orchestrator`, which advances a `CompetitionState` through enum phases: `INITIALIZING → UNDERSTANDING → LOADING_KNOWLEDGE → EDA → EXPERIMENTING → SUBMITTING → COMPLETED`. State is persisted as `competitions/<slug>/state.json` so runs are resumable.

**LLM integration — Cursor file handoff protocol** (the main pattern in use):  
Instead of making API calls, `CursorAgentProvider` writes a task prompt to `competitions/<slug>/agent_tasks/task_XXXX.md` and raises `CursorTaskPending`. The CLI catches this and exits, telling the user to have the Cursor agent write `task_XXXX_response.md`. `kagent continue` re-enters the orchestrator and reads the response file. This is the primary "LLM provider" in `config.yaml`; other providers (`litellm`, `placeholder`) exist but are disabled by default.

**Data flow through key classes:**
- `LLMRouter` (in `llm/__init__.py`) routes per-role (planner/coder/reviewer) to the configured provider
- `CodeExecutor` runs generated Python in a subprocess with timeout, collects artifacts (`*.csv`, `*.pkl`, etc.) from the working dir
- `ExperimentTracker` writes experiment records, code, and artifacts under `competitions/<slug>/experiments/<id>/`
- `PlaybookManager` / `SkillManager` load from `knowledge/playbooks/` and `knowledge/skills/` and inject relevant text into LLM prompts
- `ReflectionEngine` calls the LLM post-experiment to update knowledge

### 2. Active Competition (`competitions/playground-series-s6e6/`)

A **hand-built ML pipeline** for the current active competition (Predicting Stellar Class — balanced accuracy on GALAXY/QSO/STAR). This is independent of the agent framework; it is a set of standalone scripts run directly.

**Script responsibilities:**
| Script | Role |
|---|---|
| `train_oof.py` | 5-fold OOF for 4 GBDTs (lgb/xgb/cat/hgb); writes `artifacts/oof_<m>.npy` + `artifacts/test_<m>.npy` |
| `train_diverse.py` | Non-GBDT models (logreg/mlp/knn/rf/et); same artifact format |
| `specialist.py` | LightGBM with GALAXY/STAR upweighting; acts as meta-learner calibrator |
| `multi_seed.py` | Trains all models across seeds [42, 2025, 3407], averages into `oof_<m>_multi.npy` |
| `stack.py` | LogReg meta-learner on concatenated log-prob features; `--models` selects which OOF arrays to use |
| `blend.py` | Flat weighted blend (alternate to stacking) |
| `config.py` | Single source of truth: `ARTIFACTS`, `DATA`, `MODELS`, `CLASS_ORDER`, `N_SPLITS` |
| `features.py` | All feature engineering; `prepare_lgb_xgb()` is the main entry point |

**Feature sets** (passed as `--feature-set`): `base`, `all_v2` (current default, includes color/redshift interactions), `all_v3` (tested, not adopted — more correlated).

**Kaggle remote training** (`kaggle/` directory):  
⚠️ **HARD RULE: never train locally — all model training runs on Kaggle kernels.** Local machine is for orchestration, stacking (`stack.py`/`blend.py`, pure numpy, seconds), and submission only. Any new model/seed/feature experiment becomes a kernel under `kaggle/<name>/` (script kernel, auto-terminates on completion). This holds even for "quick" runs — a single sklearn fit on the full 577k rows can take hours locally.

Kernel conventions (learned the hard way, keep them):
- Datasets mount at `/kaggle/input/datasets/<owner>/<slug>` (nested) on current Kaggle, but older kernels saw the flat `/kaggle/input/<slug>`. Use a `find_input()` helper that tries both. Competition data is at `/kaggle/input/competitions/<slug>/`.
- Code is shared via the `s6e6-pipeline-code` dataset; OOF/test arrays via the `s6e6-artifacts` dataset. After editing training code, `bash kaggle/sync_code.sh "msg"` (re-uploads the code dataset) **and wait for it to finish processing** before pushing kernels that depend on it.
- **GPU = P100 (sm_60 Pascal); modern stacks dropped Pascal** (learned 2026-06, cost many failed kernels):
  - **Stock Kaggle torch is 2.10+cu128 with NO sm_60 kernels** → any plain-torch kernel dies on the first GPU
    op with `CUDA error: no kernel image is available`. Fix: BEFORE `import torch`, run
    `pip install -q torch==2.4.1 --extra-index-url https://download.pytorch.org/whl/cu121` (cu121 covers sm_60
    P100 AND sm_75 T4); set `enable_internet:true`. With this torch, some libs need an INDEXED device —
    e.g. TabICL must use `device='cuda:0'`, not `'cuda'` (torch 2.4 `mem_get_info` rejects bare `'cuda'`).
  - **cuDF/RAPIDS also dropped Pascal** → `cudf.read_csv` crashes `cudaErrorInvalidDevice: invalid device
    ordinal` on P100. Fix: rewrite the feature engineering in pure pandas/numpy. CatBoost-GPU and XGBoost-GPU
    themselves DO run on P100 (their own CUDA); only the cuDF FE layer must go.
  - CatBoost GPU guard: `task_type='GPU'` only errors at `.fit()`, not construction — detect up front via
    `catboost.utils.get_gpu_device_count()`; use `devices='0'` (NEVER `'0:1'`, that's the T4×2 layout).
- **GPU batch-session cap = 2**: a 3rd/4th GPU push fails with "Maximum batch GPU session count of 2 reached".
  Run cheap models on CPU (`enable_gpu:false`) to dodge it and the torch wall at once; stage GPU launches.
- **Slug poisoning**: if a kernel's FIRST push fails (e.g. the session-cap error), the slug half-registers and
  every later push returns "Notebook not found" — fix by pushing under a NEW slug. Kaggle slugs use hyphens,
  never underscores (dir may use `_`, but metadata `id` must be hyphenated, e.g. dir `gbdt_orig` → id `s6e6-gbdtorig`).
- **Third-party `dataset_sources` can fail to attach** → mirror into your own account
  (`kaggle datasets create`) and reference that; own datasets are reliable. (SDSS17 mirror: `cindyxue1122/s6e6-original-sdss17`.)
- Print results to a `results.txt` in `/kaggle/working` — `kaggle kernels output` reliably pulls output files, but does not always pull the console log for completed kernels.

Workflow:
```bash
# After changing training code:
bash kaggle/sync_code.sh "description"   # updates cindyxue1122/s6e6-pipeline-code dataset

# Trigger training (oof + diverse run in parallel on Kaggle):
bash kaggle/run_remote.sh                # polls until done, then pulls artifacts/*.npy

# Stack locally (fast):
python stack.py --models lgb_multi,xgb_multi,cat_multi,hgb_multi,logreg_multi,mlp_multi,specialist \
  --output submissions/stack.csv
kaggle competitions submit -c playground-series-s6e6 -f submissions/stack.csv -m "description"
```

Competition metric is **balanced accuracy** (NOT logloss). Always evaluate on `balanced_accuracy_score`, never logloss. See `competitions/playground-series-s6e6/CURSOR.md` and `LEADER_GAP_ANALYSIS.md` for the full experiment history, oracle analysis, and modeling rules.

## Competition strategy lessons (hard-won; reusable across Playground comps)

- **The #1 Playground lever is ORIGINAL-DATA augmentation.** These synthetic comps are generated from a real
  source dataset — append it to each fold's TRAIN pool at low weight (~0.06–0.1), NEVER into validation (so OOF
  stays honest). If the synthetic data has categorical cols the original lacks, RECONSTRUCT them from raw
  features (S6E6: `spectral_type=cut(r-g,[-1,-0.5,0])`, `galaxy_population=cut(u-r,2.2)` — verified 100% exact,
  they're just binned colors). This was the single biggest gain (~+0.003–0.005/model) and our gap to the field.
- **A dense top-of-LB cluster usually = a shared public recipe** — pull it first (`kaggle kernels list
  --sort-by voteCount`, `kaggle kernels pull`) before grinding. For S6E6 it was **"Ridge Flip + Probability
  Consensus"**, which fits per-row label flips DIRECTLY to the public-LB score (encodes past subs' public
  scores, Ridge-regresses which flips raise it). That is **public-LB probing/overfitting** — it inflates the
  public 20% and SHAKES DOWN on the private 80%. **Never do it** (it hurts the real, private result), and don't
  mistake such inflated public scores for honest model quality.
- **Trust CV, not the public LB.** Public-LB σ ≈ 0.001–0.002 (~49k rows); 4th-decimal OOF gains don't
  translate. Pick final submissions by CV / nested-CV, never by "which scored highest on the public LB".
  PROVEN by S6E6's final private LB (4th/2817): the highest-honest-nested-CV model (meta² 0.97045) was the
  highest PRIVATE model (0.97060); the best-PUBLIC model (0.97104, even with a tiny 6e-5 optimism gap) was NOT
  best-private (0.97047) — its public lead reversed on private. Public LB is a single noisy draw; 5-fold nested-CV
  is lower-variance and ranked the private winner correctly. A model's public-minus-honest-CV excess predicted its
  private shakedown (bigger excess → bigger fall). Select BOTH final picks by honest CV + one decorrelated hedge:
  Kaggle scores max(selected on private), so the honest #1 being in the pair is what secures the rank.
- **The honest ceiling is real and provable.** When OOF plateaus across many diverse strong models, prove it:
  test against the BEST pool available — incl. a GM's *published* OOFs (legit ensembling, `dataset_sources`)
  and a different model FAMILY (e.g. ICL/TabICL). If even those don't move it, the ceiling is intrinsic to the
  data → stop adding correlated models and report it honestly. (S6E6: my 24 models, +cdeotte's 6, +GBDT meta,
  +TabICL all = OOF ~0.9704 / LB ~0.9707; public 0.972 was unreachable honestly.)
- **Stacking that works**: meta = multinomial LogReg on per-model **logits** `log(p/(1-p))` (clipped), not
  log-probs; StandardScale the meta-features + C≈3; average the meta over ~5 seeds. A GBDT meta has a better
  honest nested-CV but doesn't beat linear on LB (noise-limited). Beware: a bias-calibrated linear meta's
  PLAIN OOF is optimistic by ~+0.0009 vs nested-CV (calibration leakage) — use **nested-CV** for honest numbers.
  OvR (one-vs-rest) GBDTs add real decomposition diversity (`ovrcat` was strong + complementary).
  UPDATE (S6E6, broad pool): the "GBDT meta doesn't beat linear on LB" rule only held on the small 24-base set —
  on a BROAD diverse pool (48 models incl. ~19 GM-published OOFs) a heavily-regularized 5-seed GBDT meta
  (`leaves=8, min_child=1000, λ=20`) won BOTH honest nested-CV (0.97037) AND public LB (0.97104), beating the
  linear meta and the level-2 meta-of-metas. More diverse bases → more cross-model interactions for the GBDT to exploit.
- **Materialize strong sub-components standalone — don't assume the ensemble subsumes them.** A level-1 meta
  (or single base) whose standalone honest nested-CV is within ~1σ of the ensemble built on top of it MUST be
  generated + submitted on its own. Averaging a strong member with weaker siblings only cuts variance; on a single
  draw the pure member can WIN (S6E6: the pool48 GBDT meta at honest 0.97037 scored public 0.97104, beating the
  meta-of-metas that *contained* it at 0.97079). Corollary: a "sweep" is only as complete as the last directory
  scan — re-inventory `submissions/` + `artifacts/` before declaring the honest method-space exhausted.
- **Stack-alignment contract** (so every base model's OOF/test combine): OOF from
  `StratifiedKFold(5, shuffle=True, random_state=42)` on integer labels in CSV order (the split depends only on
  y+seed, so this auto-aligns OOFs across all models); test rows in `sample_submission` order; GALAXY=0/QSO=1/STAR=2.
- **For kernels you can't test locally** (hard rule: no local training): author + adversarially review each
  before spending GPU quota — review caught a logloss-vs-balacc early-stop bug and a dropped-function NameError
  that `py_compile`/`ast.parse` could not.

## Configuration

`config.yaml` at the repo root is the agent's runtime config. Key sections: `llm.providers` (add API keys via env vars named in `api_key_env`), `budget` (max experiments, max LLM cost), `paths` (relative to CWD where `kagent` is invoked). `Config.load()` looks for `config.yaml` in CWD by default.

## Knowledge Base

`knowledge/playbooks/` — markdown files (`general.md`, `tabular.md`, `techniques/`) injected as LLM context.  
`knowledge/skills/` — reusable code snippets with YAML metadata, also injected into LLM prompts.  
Both are live-editable and loaded at runtime by `PlaybookManager` / `SkillManager`. The `ReflectionEngine` can update them after experiments.
