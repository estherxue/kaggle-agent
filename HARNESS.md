# HARNESS.md — What actually made the agent effective

A cross-competition inventory of the **harness**: the scaffolding around the modeling
work, not the modeling work itself. Every entry below is here because it demonstrably
changed an outcome on a real competition. The final section lists what did *not* work,
so it does not get rebuilt.

Evidence base: Playground S6E6 (Predicting Stellar Class — **4th / 2817**, private
0.97060), NeuroGolf 2026, Multi-Step Tool Attacks (MSTA).

> The one-line thesis: **the leverage was never in the model code.** It was in rules that
> survive a context reset, a product contract that lets independent runs compose, a
> remote executor that turns a laptop into a fleet, and an honest metric that refuses to
> be fooled by the public leaderboard.

---

## L0 — Rules layer: `CLAUDE.md` as an enforced context

**Where:** repo-root `CLAUDE.md`, per-competition `CURSOR.md`.

Not documentation. It is the only part of the harness that is *guaranteed* to be in
context at the start of every session, so it is where constraints go that must survive a
`/compact`, a crash, or a fresh session days later.

What earns a line in it:

| Kind | Example (S6E6) | Why it must be a rule, not a memory |
|---|---|---|
| Irreversible-risk gate | "Never commit an active competition's solution" (`.gitignore` = `*`) | The remote is public. One `git push` leaks the approach. |
| Resource-shape rule | "Never train locally — all training on Kaggle kernels" | One sklearn fit on 577k rows = hours locally, minutes remotely. |
| Metric lock | "The metric is balanced accuracy, NOT logloss" | Cost a real bug: early-stopping on logloss while selecting on BA. |
| Environment landmine | P100 = sm_60; stock torch ships no sm_60 image | Each one cost multiple failed GPU kernels before it was written down. |

**Reusable rule:** any fact that cost more than one failed run to learn belongs in
`CLAUDE.md` the same day. The test is not "is this interesting" but **"would a fresh
session repeat this mistake?"**

---

## L1 — Contract layer: make independent runs compose

**Where:** `competitions/playground-series-s6e6/config.py`,
`kaggle/code_ds/kernel_common.py` (419 lines).

The single highest-leverage piece of engineering in the whole campaign. One frozen
contract:

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

- `stack.py --models a,b,c,...` — compose any subset with a CLI flag. 167 `.npy`
  artifacts, 30 kernels, a final 48-model pool.
- **Third-party OOFs drop straight in.** Chris Deotte's 19 published OOF arrays were
  stacked without a line of glue code. That single move was worth **+0.00075 honest CV**
  (BASE24 linear 0.96954 → pool44 0.97029) — the largest late-stage gain.
- Shape verification becomes mechanical: a wrong-shaped `.npy` is a caught error, not a
  silently-misaligned stack.

`kernel_common.py` is the contract made executable and importable — `stratkfold()` is the
single source of truth for the split, `save_oof_test()` the single writer. Kernels
`from kernel_common import ...` instead of copy-pasting boilerplate into 21 `run.py`
files.

**Reusable rule:** before generating the second artifact of any kind, freeze the contract
that lets artifact #1 and #47 be combined. Retrofitting alignment is far more expensive
than declaring it.

---

## L2 — Execution layer: laptop as orchestrator, cloud as compute

**Where:** `competitions/playground-series-s6e6/kaggle/` (30 kernel dirs, `sync_code.sh`,
`run_remote.sh`), `src/kaggle_agent/tools/kernel_fleet.py` (1000 lines).

```
sync_code.sh      →  copy 13 source files into code_ds/ → `kaggle datasets version`
                     (code ships as a DATASET, not duplicated per kernel)
kaggle/<name>/    →  one experiment = one dir + kernel-metadata.json + run.py
run_remote.sh     →  push → poll status → on complete, `kernels output` pulls .npy
local             →  stack/blend (pure numpy, seconds) + submit
```

### `kernel_fleet.py` — the part worth keeping

Bash (`run_new.sh`, `poll_all.sh`) got the job done but died on every edge case. The
Python driver encodes each edge case as tested logic:

| Component | Problem it solves |
|---|---|
| `gpu_cap_admits()` | Kaggle allows **2** concurrent GPU batch sessions; a 3rd push fails. Pure predicate, so the queue is testable without credentials. CPU kernels bypass the cap. |
| `classify_push_output()` | Distinguishes "GPU cap hit, retry later" from "slug is dead" from a real error. |
| `bump_slug()` | **Slug poisoning**: a kernel whose *first* push failed returns "Notebook not found" forever. Only fix is a new slug — automated. |
| `pull_and_verify()` | Kernel outputs can be truncated. Every expected `.npy` is `np.load`ed and shape-checked; re-pull up to 3×. |
| `diagnose_log()` | 6 known failure signatures → the actual fix text (`no_kernel_image`, `cudf_pascal_dropped`, `slug_poisoned`, `catboost_float_cat_features`, `bad_oof_shape`, `oom`). |
| `run_fleet()` | Pending queue + poll + pull-on-complete + diagnose-on-fail. **One bad kernel never takes down the fleet.** |

Validated end-to-end on real Kaggle: 6 kernels including GPU-cap=2 staging and a
transient-CLI-error auto-retry. A mass-FE kernel hit the 12h wall and was cancelled — its
OOF had been saved first, so it survived and pulled cleanly.

**Reusable rule:** the remote executor's job is not to run jobs, it is to **survive them
failing**. Budget the same effort for the failure catalogue as for the happy path.

**Known gap:** the fleet driver ran as a foreground process and was killed on an overnight
session restart. Kernels survived (they run on Kaggle), but the driver should be a tracked
background job.

---

## L3 — Judgment layer: the metric that refuses to be fooled

**Where:** `nested_cv.py`, `select_final.py`, `metrics.py`, `hillclimb.py`.

This layer scored **zero points** and decided the competition.

`stack.py` fits class calibration on the full meta-OOF and then reports BA on that same
data — optimistic. `nested_cv.py` holds every test row out of **both** the meta fit and the
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
| stack_24_final | 0.96954 | 0.97077 | 0.97035 | −0.00042 |

Three findings that generalize:

1. **The highest honest-CV model was the highest private model.** Public LB — a single
   noisy draw on ~49k rows, σ ≈ 0.00087 — ranked them wrong.
2. **Shakedown was proportional to public-minus-honest-CV excess.** +0.00034 excess → fell
   0.00019; +0.00067 → fell 0.00057; +0.00123 → fell 0.00042. The optimism gap is a
   *predictor*, not just a diagnostic.
3. **Kaggle scores `max(selected)` on private.** Picking by honest CV *and* keeping one
   decorrelated hedge is what secured 4th. Chasing the best public model alone → 0.97047,
   a worse rank.

**Reusable rule:** when optimizing against a metric you can query repeatedly, build the
honest estimator *before* you need it. And write down, in advance, that final selection is
by honest CV — the temptation to pick the pretty public number arrives late, under time
pressure.

### Corollary: the anti-hack rule

The top public cluster (0.9724–0.9728) ran "Ridge Flip + Probability Consensus" — Ridge
regression fitted directly onto past submissions' *public scores* to reverse-engineer which
label flips raise the public 20%. It was ruled out as public-LB probing. The whole board
shook down on private; that refusal is *how* an honest 0.97060 reached 4th.

---

## L4 — Protocol layer: how to run agent fleets

**Where:** `.claude/skills/agent-field-lessons/SKILL.md`,
`.claude/skills/deli-auto-research/SKILL.md`.

Two skills, auto-loaded by description match:

- **`agent-field-lessons`** — five protocol lessons from a ~10M-token subagent fleet
  campaign (ONNX code-golf, 400 independent targets). Covers: re-measure every subagent's
  claimed win yourself; a fleet reporting "infeasible" from a proxy environment is
  reporting on the proxy; one bad member can zero a bundled artifact.
- **`deli-auto-research`** — anti-stall / anti-loop conventions for long-horizon
  autonomous runs: fresh-context iterations, file-persisted state, quantitative stall
  detection with *forced* structural pivots, guardian/worker separation.

### Adversarial pre-review of untestable code

Hard rule on S6E6: no local training. So a kernel's first execution is on Kaggle, burning
quota. Mitigation: **author, then adversarially review each kernel before pushing** — a
second agent whose only job is to refute it.

Caught, in code that `py_compile` and `ast.parse` both passed:
- a logloss-vs-balanced-accuracy early-stopping bug,
- a dropped-function `NameError`,
- a false ceiling claim.

**Reusable rule:** when the feedback loop costs money or hours, insert an adversarial
reviewer *before* execution. Syntax checks do not catch semantic bugs, and semantic bugs
are exactly what an expensive loop punishes.

---

## Cross-competition reusables

Harness pieces that generalize beyond the competition they were built for.

### `competitions/neurogolf-2026/harness/`

- **`research_harness.py`** — parallel research fan-out: N researcher subagents (each owns
  one sub-question, uses server-side `web_search`) → one synthesizer that cross-verifies
  and flags contradictions. Mission-driven via a JSON file, so it retargets to any topic by
  editing `missions/*.json`. Researchers are told to mark low-confidence claims explicitly
  rather than paper over gaps.
- **`verify_scoring.py`** — executes the competition's *real* scoring path locally instead
  of trusting the documented formula. On a competition whose score is a non-obvious
  function (here `max(1, 25 − log(memory + params))`), reverse-engineering the true
  objective is often the single biggest lever.

### `harness/` — distilled from NeuroGolf 2026

Root-level, domain-stripped tooling from the 400-target ONNX code-golf campaign
(score 247 → 7625.77 under a hard no-exploit constraint). See `harness/README.md`.

- **`experience_db.py`** — an evidence-graded ledger + posterior-reward scheduler. Two
  ideas carry it:
  - **Every number carries an evidence level**, and the levels never conflate:
    `estimated` < `local_pass` < `holdout_clean` < `authority_ok` < `measured`.
    Nothing below `authority_ok` ships; nothing at `estimated` may be quoted as fact.
    This exists because an *estimated* score sat in a column that looked identical to
    measured rows, an agent read it as fact, a "fix" shipped, and the real metric went
    down by exactly that amount.
  - **Store refutations, not just wins.** A proven-dead direction stops the next agent
    spending a whole wave re-refuting it. Retrieval is by *structure* (`--bottleneck`,
    `--sig`), so an agent starting on a new target finds prior work that hit the same wall.
  - Corollary it encodes: **no static estimator survives contact.** Four successive
    "remaining headroom" estimates read 425.83 → 44.85 → 36.70 → 1.15 as each was actually
    checked — a 370× over-promise. `sched` therefore starts from a heavily damped prior and
    lets measured reward take over.

- **The four-stage audit pipeline** (documented in `harness/README.md`, not shipped as an
  abstract framework — the code is inseparable from ONNX/ARC, the *shape* is what
  transfers). For absorbing any third-party artifact set:
  `SCAN → GATE → DIFFERENTIAL PREFLIGHT → SUBMIT`.
  - *Scan*: record `loadfail`/`runfail` as **UNKNOWN**, never as "no improvement" —
    40/400 members were unrunnable locally; treating them as failures would have discarded
    8 that were both correct and cheaper.
  - *Gate*: **any example shipped with the problem is a training set** — the artifact's
    author saw it too. Promote only what survives freshly generated data on an independent
    seed. This stage rejected 19 members that were exact on every provided example and
    dirty on fresh data, one at 14.75% error.
  - *Differential preflight*: never read a checker failure in isolation — run the identical
    check on a **known-good control** and diff. A local checker called 94 members "fatal";
    the control showed 96 of the same, proving all 94 were local artifacts. Without it,
    14 good models would have been reverted.
  - *Submit*: state the expected delta first, then compare. Predicted +191.94, actual
    +180.86 — and the 11-point gap is what revealed the authority computes memory as
    `max(static, runtime)` while the local probe measured runtime only.

This is the same "build an honest local replica of the remote scorer" instinct as L3, plus
the discipline for what to do when the artifacts come from **someone else**.

### `competitions/multi-step-tool-attacks/harness/`

- **`eval_replay.py`** — offline evaluator that replays candidates exactly like the
  hosted gateway does and scores them with the real scoring function. No GPU, no target
  LLM. Turns a slow, rate-limited, remote submission loop into a local one.
- **`FINDINGS.md`** — the scoring/predicate/guardrail mechanics reverse-engineered from
  the SDK, written down before optimizing against them.

**The shared pattern across all four campaigns: build a local, honest replica of the
remote scorer first.** `nested_cv.py`, `verify_scoring.py`, and `eval_replay.py` are the
same idea in three domains, and each was the highest-value tool in its competition. The
audit pipeline in `harness/README.md` is the fourth: what to do when the replica and the
authority disagree.

### Repo-level conventions worth keeping

- `experiment_log.py` → append-only `artifacts/experiments.csv` + `.jsonl`. Cheap, and the
  only reason a 6-week campaign can be audited after the fact.
- `LEADER_GAP_ANALYSIS.md` — timestamped running log of every experiment, including the
  ones that failed and the conclusions that were later *revised*. The revisions are the
  valuable part; it records being wrong about the ceiling and what test settled it.
- Competition dirs gitignored by default (`competitions/.gitignore` tracks only `*.py` /
  `*.md`), with the solution untracked while the competition is live.

---

## Anti-inventory: proven not to work

Do not rebuild these. Each was measured on S6E6 (balanced accuracy, 577k rows,
8 raw astronomical features).

### Modeling dead ends

| Attempt | Standalone OOF | Contribution | Note |
|---|---|---|---|
| Mass feature engineering (243 feats, 76 groupby aggs) | 0.96539 | **+0** | 0.984 argmax-agreement with the best base; also hit the 12h CPU wall |
| Optuna HPO (60–100 trials) | 0.96506 / 0.96562 | **+0** | Lost to hand-tuned 0.96877. Cheap 10-trial search was *negative*: 0.96427 < 0.96441 default |
| AutoGluon (good_quality, bagged) | 0.95646 | **+0** | |
| DAE representation learning (swap-noise) | 0.96444 | **+0** | 8 clean features: nothing to learn |
| Hierarchical cascade (STAR-vs-rest → GALAXY-vs-QSO) | 0.96335 | **net loss** | Moves recalls around, doesn't add coverage |
| Pseudo-labeling (transductive, conf ≥ 0.995) | 0.96561 / 0.96538 | **+0** | Identical to plain GBDTs |
| ICL family (TabPFN-3 / TabICL) | 0.93591 / 0.95797 | **negative** | 30 models 0.97041 → 31 models 0.97040 |
| Calibration & decision rules | — | +0.00004 / +0.00001 / **−0.00065** | scale+shift, temperature, equal-recall. Additive bias was already optimal |
| Per-seed features into the meta | 0.96705 | −0.00003 | Provably redundant: a linear meta over per-seed log-probs ≡ over their average |
| Anomaly patching / probability re-vote | — | **hurts in every config** | The meta already beats the fallback on high-disagreement rows |
| ExtraTrees | 0.94166 | −0.00008 | Rejected by nested-CV before submission |

Two structural lessons underneath the table:

- **More correlated features make the ensemble worse, not better.** The `all_v3` feature
  set lifted every single model (lgb +0.0002) but the blend was a *tie* (0.96608 vs
  0.96609). `analyze_correlation.py` gave the mechanism: pairwise error-Jaccard rose
  0.716 → 0.727 while the oracle ceiling *fell* 0.97415 → 0.97384. **Track the oracle
  ceiling, not the single-model score, when adding a member.**
- **The honest ceiling is provable and worth proving.** 24 own models + a GM's 19 published
  OOFs + a GBDT meta + a transformer-ICL model all landed at OOF ~0.9704. At that point
  the correct action is to report the ceiling, not to keep grinding.

### Process failures

- **Noise-limited regime was detected on 06-22 and then ignored for days.** The 9-model
  stack had the best OOF (0.96713) but a *worse* public LB (0.96740) than a 7-model stack
  (0.96680 → 0.96749). Every subsequent "add one more marginal model" was inside the noise.
  **Once 4th-decimal OOF stops tracking LB, stop adding members and switch to variance
  reduction and honest selection.**
- **The best public submission was almost never generated.** `gbdt_meta_pool48` existed
  only as a *level-1* component inside a level-2 meta-of-metas. Its standalone honest CV
  (0.97037) was within 8e-5 of the ensemble containing it (0.97045) — the number was seen
  and not acted on, on the assumption that the ensemble subsumed it. It did not: standalone
  scored public 0.97104, beating the ensemble that contained it by +0.00025.
  **Rule: whenever a sub-component's honest score is within ~1σ of the ensemble built on
  top of it, materialize and submit it standalone.**
  Compounding cause: the file was produced by parallel work after the last `submissions/`
  scan, and a context compaction hid it. **Rule: re-inventory `submissions/` and
  `artifacts/` before declaring a sweep exhausted — a sweep is only as complete as the last
  directory listing.**

### The framework that was never used

`src/kaggle_agent/` implements an autonomous competition agent: an `Orchestrator` state
machine (`INITIALIZING → UNDERSTANDING → LOADING_KNOWLEDGE → EDA → EXPERIMENTING →
SUBMITTING → COMPLETED`), a Cursor file-handoff LLM protocol, `PlaybookManager` /
`SkillManager` / `ReflectionEngine`.

**No competition used it.** S6E6, NeuroGolf, and MSTA all ran with empty `agent_tasks/`,
empty `experiments/`, and no `state.json`. The effective agent loop was Claude Code driving
the scripts directly, steered by `CLAUDE.md`.

The one piece that flowed *back* into the framework from real competition use is
`tools/kernel_fleet.py` — and it was built only after the bash it replaces had already
failed in production.

**Lesson: the harness that worked was written in response to a specific failure that had
already happened.** The one designed up-front, from a spec, went unused.
