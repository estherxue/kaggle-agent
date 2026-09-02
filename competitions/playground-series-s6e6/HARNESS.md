# S6E6 — how the harness carried the campaign

Playground Series S6E6 (Predicting Stellar Class), 2026-06-13 → 07-01.
**4th / 2817**, private 0.97060.

This is not a description of the harness. It is an account of *where the harness intervened
during the run*, what it changed at that moment, and what it failed to do. Each section is
one phase of the campaign: the situation, the intervention, the delta.

The arc, in one line: **public 0.96703 → 0.97104, private 0.97060.** Two-thirds of that came
on a single day — the day the harness stopped being a convenience and became the thing doing
the work.

Source records: `LEADER_GAP_ANALYSIS.md` (timestamped experiment log), `CURSOR.md` (operating
rules), `artifacts/experiments.csv`.

---

## Phase 0 · 06-13 → 06-15 — the harness that did not exist yet

**Situation.** A fresh repo, an autonomous-agent framework (`src/kaggle_agent/`), and a
competition to enter.

**What actually happened.** The framework was never invoked. `agent_tasks/` stayed empty,
`experiments/` stayed empty, no `state.json` was ever written. Work proceeded as Claude Code
running scripts directly.

**Why that matters here.** The first thing the campaign taught was that a harness designed
up-front against a spec does not survive contact — and that the pieces which *did* end up
carrying the run were all written later, each in response to a specific failure that had
already happened. Every section below is an instance of that pattern.

---

## Phase 1 · 06-15 → 06-20 — a contract written before anyone needed it

**Situation.** Four GBDTs, one machine, flat blending. OOF 0.96609 → public **0.96703**.

**Intervention.** Before generating the second artifact, the fold contract was frozen in
`config.py`:

```
StratifiedKFold(5, shuffle=True, random_state=42) on integer y, in train-CSV row order
GALAXY=0 / QSO=1 / STAR=2   |   OOF (577347,3) float32   |   test (247435,3) float32
artifacts/oof_<name>.npy    +   artifacts/test_<name>.npy
```

At the time this looked like tidiness. It was the single highest-leverage decision of the
campaign, and the reason is structural: **the split depends only on `y` and the seed** — not on
features, not on the model, not on who ran it or when.

**What it changed, later.** Every subsequent phase cashed this in:

- `stack.py --models a,b,c,…` let any subset be composed with a CLI flag. The pool grew from
  4 to **48 models across 30 kernels and 167 `.npy` artifacts** without one line of
  integration code being written.
- On 06-28 a Kaggle GM's **19 published OOF arrays** were dropped straight into the pool.
  They aligned row-for-row on arrival. That was worth **+0.00075 honest CV** (BASE24 linear
  0.96954 → pool44 0.97029) — the largest single gain of the late campaign, and it cost
  nothing but a download.
- A wrong-shaped array became a caught error rather than a silently misaligned stack.

**Also this phase — the discovery that reframed the modeling.** `train_diverse.py` produced a
logistic regression at OOF 0.925 and an MLP at 0.942, both far below the GBDTs. A flat blend
including them got *worse* (0.96586 vs 0.96609). Stacking them made it *better* (0.96679).
The harness lesson: a weak member drags an average down but gives a meta-learner coverage —
convert coverage to score with a meta-learner, never with a mean.

---

## Phase 2 · 06-20 → 06-24 — the harness's first job was to say "stop"

**Situation.** Multi-seed averaging and a specialist model had pushed OOF to 0.96708. More
members kept being added.

**Intervention 1 — noise detection (06-22).** Comparing OOF against public LB across stacks
produced a result nobody wanted:

| stack | OOF | public LB |
|---|---|---|
| 7-model single-seed | 0.96680 | **0.96749** |
| 7-model multi-seed | 0.96708 | 0.96746 |
| 9-model | **0.96713** | 0.96740 |

The best OOF had the *worst* LB. Public LB is ~49k rows, 1σ ≈ 0.0007–0.002; the spread between
these stacks was ~0.0001. **The two metrics had decorrelated at the 4th decimal.** This was
written into `CURSOR.md` the same day as a decisive finding.

**Intervention 2 — nested-CV (06-22).** `nested_cv.py` was written because `stack.py` fits
class calibration on the full meta-OOF and then scores on that same data. The honest version
holds every row out of *both* the meta fit and the calibration fit.

Its first act was to **reject a model before it was ever submitted**: ExtraTrees, OOF 0.94166,
scored −0.00008 in nested-CV. `CURSOR.md` records the verdict — *"ET not adopted. Do NOT
generate test_et.npy — no submission gain expected."*

**What it changed.** It should have ended the "add one more member" phase. It did not — ET was
submitted anyway on 06-24 to watch the LB, and the harness's prediction was correct: no gain.
Days were spent inside the noise band before the campaign pivoted. **The harness produced the
right signal on 06-22 and it was acted on properly only on 06-25.**

---

## Phase 3 · 06-24 → 06-25 — the day the harness earned everything

**Situation.** Public LB 0.96750, stuck. A dense cluster of ~20 teams sat at 0.9724–0.9728 —
a **0.005 gap, far outside LB noise**. A tight cluster like that means a shared public recipe.

**Intervention 1 — the gap analysis.** `LEADER_GAP_ANALYSIS.md` was opened to split the gap
into two parts, and this split is what made the rest of the campaign possible:

- **(A) Public-LB overfitting.** The cluster recipe was *"Ridge Flip + Probability
  Consensus"*: Ridge regression fitted onto past submissions' **public scores** to
  reverse-engineer which per-row label flips raise the public 20%, iterated on live LB
  feedback. Classified as probing, forbidden, and — critically — predicted to **shake down on
  private**.
- **(B) A real model-strength gap.** The leaders' *single* models matched our whole stack.
  Two levers were identified and neither was exotic: appending the **original SDSS17 dataset**
  to each fold's training pool at low weight (never into validation, so OOF stays honest), and
  **actually strong deep models** (RealMLP R2-103, a DCN, TabM).

That separation is the whole campaign in one page: it told us what to copy and what to refuse.

**Intervention 2 — the infra rule that forced the pivot.** `CLAUDE.md` already carried
*"never train locally."* A single sklearn fit on 577k rows takes hours on the laptop. So the
new models could not be tried incrementally at home — they had to become Kaggle kernels, which
meant building the remote path. That constraint is what turned a slow local loop into a fleet.

**Intervention 3 — four infra walls in a few hours, each written down as it fell.** Every one
of these killed kernels before it was understood:

| wall | symptom | fix, once found |
|---|---|---|
| P100 is **sm_60** | stock torch ≥2.10+cu128 → `CUDA error: no kernel image is available`; killed realmlp5 *and* tabm | `pip install torch==2.4.1` from the cu121 index **before** `import torch`; some libs then need `device='cuda:0'`, not bare `'cuda'` |
| cuDF dropped Pascal | `cudaErrorInvalidDevice` at `cudf.read_csv`; killed cat-v3 and xgb-v5 | rewrite the entire feature engineering in pandas/numpy — CatBoost-GPU and XGBoost-GPU still run fine on P100 |
| GPU batch cap = **2** | 3rd concurrent GPU push refused | stage launches; run cheap models on CPU to dodge both the cap and the torch wall |
| slug poisoning | a kernel whose *first* push failed returns "Notebook not found" **forever** | re-push under a new hyphenated slug (`gbdt_orig` → `s6e6-gbdtorig`) |

Each went into `CLAUDE.md` the same day. That is the rules layer working as designed: these
are facts that no amount of reasoning recovers, and a fresh session would have burned the same
quota rediscovering them.

**Intervention 4 — adversarial pre-review.** Because nothing could be run locally, a kernel's
first execution was on Kaggle, burning quota. So each kernel was authored, then handed to a
second agent whose only job was to refute it. In code that `py_compile` and `ast.parse` both
passed, this caught **a logloss-vs-balanced-accuracy early-stopping bug** and **a
dropped-function `NameError`** — either of which would have wasted a full kernel run.

**The delta.** In roughly one day: public **0.96750 → 0.97003**. The single-model results tell
the story — `realmlp5` alone scored OOF 0.96908, *approximately equal to the entire previous
stack*, and the original-data GBDTs lifted GALAXY recall (the bottleneck class) from 0.955 to
0.961.

---

## Phase 4 · 06-25 → 06-27 — bash breaks, and the fleet driver gets written

**Situation.** Wave after wave of kernels: catv3, xgbv5, ovrcat, ovrxgb, realmlp5b/5c, nn2b.
The shell scripts (`run_new.sh`, `poll_all.sh`, `poll_wave2.sh`) worked on the happy path and
died on every edge case — a GPU-capped push was indistinguishable from a dead slug, a
truncated `.npy` looked like a success, one bad kernel stalled the batch.

**Intervention.** `kernel_fleet.py` (1000 lines) replaced them, encoding each failure already
survived as tested logic: `gpu_cap_admits()` as a pure schedulable predicate (CPU kernels
bypass the cap), `classify_push_output()` to tell "retry later" from "slug is dead",
`bump_slug()` to auto-recover from poisoning, `pull_and_verify()` to `np.load` and shape-check
every pulled array with up to 3 re-pulls, and `diagnose_log()` mapping six known signatures
to their actual fix text.

**What it changed.** It let the campaign run kernels unattended. Concretely, on 06-29 it
pushed and managed three concurrent CPU kernels with auto-pull and shape verification, and a
mass-FE kernel that hit the 12h wall was **cancelled without losing its work** — the OOF had
been written before the test predictions, so it survived the cancel and pulled cleanly.

**Also this phase — the plateau is confirmed, honestly.** Adding OvR models raised OOF
+0.0003 but LB stayed flat (0.97066 → 0.97061). Rather than reading that as noise to push
through, it was recorded as *"honest expected LB has plateaued at ~0.9706."*

---

## Phase 5 · 06-26 → 06-29 — using the harness to prove a negative

**Situation.** The stack sat at OOF ~0.9704 and would not move. The question was whether that
was a wall or a lack of imagination.

**Intervention.** The contract from Phase 1 made "test the ceiling exhaustively" cheap, because
any new model was just one more `oof_*.npy`. So it was tested against everything available:

| test | result |
|---|---|
| our 24 models | OOF 0.97040 |
| + a GM's 6 published OOFs (30 models) | 0.97041 — **no change** |
| + TabICL, a genuinely different inductive bias (31) | 0.97040 — **negative** |
| + mass-FE (243 feats, 76 groupby aggs) | +0, 0.984 argmax-agreement with the best base |
| + Optuna HPO, AutoGluon, DAE, pseudo-labels, hierarchical cascade | all ~0.965, all +0 |

**What it changed.** The conclusion flipped from "we are not trying hard enough" to **"the
~0.9704 ceiling is intrinsic to the data"** — 8 clean astronomical features, signal-saturated.
That is a harness result, not a modeling one: without a contract that makes adding a model
nearly free, this would have been an opinion instead of a measurement.

`analyze_correlation.py` supplied the mechanism for why more features stopped helping: the
`all_v3` feature set lifted *every individual model* (lgb +0.0002) yet the blend was a tie
(0.96608 vs 0.96609), because pairwise error-Jaccard rose 0.716 → 0.727 while **the oracle
ceiling fell** 0.97415 → 0.97384. More shared features made the models more correlated without
adding coverage.

**And a correction the harness forced on itself.** The first ceiling claim was over-stated —
the gap to #1 was 2.4σ, statistically real, and the "even a GM's pool plateaus" test had used
only 6 shared models, excluding exactly the decorrelated ICL family that might have helped. So
ICL was actually tested (and ruled out). `LEADER_GAP_ANALYSIS.md` records the claim, the
objection, and the revision — which is why that file is worth more than a clean summary would
have been.

---

## Phase 6 · 06-29 — where the harness did *not* save us

**Situation.** A 48-model pool, several meta-learners, and a level-2 meta-of-metas as the
headline result at honest 0.97045.

**What went wrong.** `gbdt_meta_pool48`, a *level-1* component inside that meta-of-metas, had
a standalone honest nested-CV of 0.97037 — within 8e-5 of the ensemble containing it. **That
number was seen and not acted on**, on the assumption that the ensemble subsumed it. It did
not: generated standalone, it scored public **0.97104**, beating the ensemble built on top of
it by +0.00025 and becoming the best public submission of the campaign.

It was nearly missed entirely: the file was produced by parallel work *after* the last
`submissions/` scan, and a context compaction hid it from view.

**The two rules that came out of it**, now in `CLAUDE.md`:
- Whenever a sub-component's honest score is within ~1σ of the ensemble built on top of it,
  **materialize and submit it standalone** — averaging a strong member with weaker siblings
  only cuts variance, and on a single draw the pure member can win.
- **Re-inventory `submissions/` and `artifacts/` before declaring a sweep exhausted.** A sweep
  is only as complete as the last directory listing.

---

## Phase 7 · 07-01 — the verdict

The private leaderboard is the only place the harness's central claim could be tested.

| submission | honest CV | public | **private** | drop |
|---|---|---|---|---|
| strat_multilevel_pool48 | **0.97045** ← best | 0.97079 | **0.97060** ← best | −0.00019 |
| stack_gbdtmeta_pool48 | 0.97037 | **0.97104** ← best | 0.97047 | −0.00057 |
| stack_gbdtmeta (24) | 0.97022 | 0.97050 | 0.97047 | −0.00003 |
| stack_pool44 (linear) | 0.97029 | 0.97072 | 0.97046 | −0.00026 |
| stack_24_final (linear) | 0.96954 | 0.97077 | 0.97035 | −0.00042 |
| stack_hillclimb | 0.97002 | 0.97055 | 0.97024 | −0.00031 |

1. **The highest honest-CV model was the highest private model.** `nested_cv.py` — which never
   added a single point — ranked the private winner correctly where public LB did not.
2. **The public model won the public and lost the private.** Its +0.00025 public lead over the
   honest pick reversed into a −0.00013 private deficit.
3. **Shakedown tracked public-minus-honest excess almost monotonically**: +0.00034 excess fell
   0.00019; +0.00067 fell 0.00057; +0.00123 fell 0.00042. The optimism gap was a *predictor*,
   not just a diagnostic.
4. **Kaggle scores `max(selected)` on private.** Selecting both by honest CV plus one
   decorrelated hedge is what secured 4th. Chasing the 0.97104 public alone would have
   yielded 0.97047 — a worse rank.
5. **The Phase-3 forecast held.** The 0.972+ public cluster that used Ridge-Flip probing
   collapsed on private. Refusing it is *how* an honest 0.97060 reached 4th.

---

## Ledger — which harness piece moved which number

| harness piece | when it paid | measured effect |
|---|---|---|
| fold-alignment contract (`config.py`, `kernel_common.py`) | 06-28 | 19 third-party OOFs composed on arrival → **+0.00075 honest CV** |
| `CLAUDE.md` infra rules (P100 / cuDF / GPU cap / slugs) | 06-25 | unblocked the 4 kernels carrying **+0.0025 in one day** |
| "never train locally" rule | 06-24 | forced the remote pivot that made that day possible |
| adversarial pre-review | 06-25 | caught a metric bug + a `NameError` before they burned GPU quota |
| `nested_cv.py` | 06-22, 07-01 | rejected ET pre-submission; **picked the private winner** |
| noise detection | 06-22 | correctly ended the marginal-member phase (acted on 3 days late) |
| the contract, again | 06-26→29 | made "prove the ceiling" a measurement, not an opinion |
| `kernel_fleet.py` | 06-27→29 | unattended fleets; salvaged a 12h-wall kernel's OOF |
| `LEADER_GAP_ANALYSIS.md` | throughout | recorded a wrong ceiling claim *and its revision* |
| `experiment_log.py` | throughout | append-only CSV/JSONL; the only reason this account can be reconstructed |

## What the harness failed to do

- **It signalled the noise-limited regime on 06-22 and was overruled for three days.** A
  detector nobody acts on is worth nothing; the rule now lives in `CLAUDE.md` as a stop
  condition, not an observation.
- **It had no inventory step.** The best public submission was nearly lost because nothing
  re-scanned `submissions/` after parallel work wrote into it.
- **The fleet driver was a foreground process.** An overnight session restart killed it. The
  kernels survived (they run on Kaggle) and their OOFs had already been pulled, but the driver
  should be a tracked background job.
- **`kernel_fleet.py` still has no tests**, despite its docstring claiming its pure logic was
  split out to be testable without credentials.
