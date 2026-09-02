# HARNESS.md — What actually made the agent effective

A cross-campaign inventory of the **harness**: the scaffolding around the work, not the
work itself. Every entry is here because it demonstrably changed an outcome. The later
sections list what did *not* work, so it does not get rebuilt.

**Evidence base**

| campaign | result | what it tested |
|---|---|---|
| Playground S6E6 — Predicting Stellar Class | **4th / 2817**, private 0.97060 | a 6-week ensembling campaign against a noisy public leaderboard |
| NeuroGolf 2026 — ONNX code golf | 247 → **7625.77**, 400 targets | a long-horizon agent-fleet campaign under no-overfit *and* no-exploit constraints |
| Multi-Step Tool Attacks (MSTA) | red-team harness | reverse-engineering a hosted scorer |

> The one-line thesis: **the leverage was never in the model code.** It was in rules that
> survive a context reset, a contract that lets independent runs compose, a remote executor
> that turns a laptop into a fleet, and an honest local replica of the scorer that refuses
> to be fooled.

---

# Part I — The five layers

## L0 — Rules layer: `CLAUDE.md` as enforced context

**Where:** repo-root `CLAUDE.md`, per-competition `CURSOR.md`, `neurogolf-26/METHODS.md`
(236 lines, 42 rules).

Not documentation. It is the only part of the harness *guaranteed* to be in context at the
start of every session, so it is where constraints go that must survive a `/compact`, a
crash, or a fresh session days later.

What earns a line in it:

| Kind | Example (S6E6) | Why it must be a rule, not a memory |
|---|---|---|
| Irreversible-risk gate | "Never commit an active competition's solution" (`.gitignore` = `*`) | The remote is public. One `git push` leaks the approach. |
| Resource-shape rule | "Never train locally — all training on Kaggle kernels" | One sklearn fit on 577k rows = hours locally, minutes remotely. |
| Metric lock | "The metric is balanced accuracy, NOT logloss" | Cost a real bug: early-stopping on logloss while selecting on BA. |
| Environment landmine | P100 = sm_60; stock torch ships no sm_60 image | Each one cost multiple failed GPU kernels before it was written down. |

**Reusable rule:** any fact that cost more than one failed run to learn belongs in
`CLAUDE.md` the same day. The test is not "is this interesting" but **"would a fresh session
repeat this mistake?"**

---

## L1 — Contract layer: make independent runs compose

**Where:** `competitions/playground-series-s6e6/config.py`,
`kaggle/code_ds/kernel_common.py` (419 lines).

The highest-leverage piece of engineering in the S6E6 campaign. One frozen contract:

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

- `stack.py --models a,b,c,...` — compose any subset with a CLI flag. 167 `.npy` artifacts,
  30 kernels, a final 48-model pool.
- **Third-party OOFs drop straight in.** 19 published OOF arrays from a Kaggle GM were
  stacked without a line of glue code — worth **+0.00075 honest CV** (BASE24 linear 0.96954
  → pool44 0.97029), the largest late-stage gain.
- Shape verification becomes mechanical: a wrong-shaped `.npy` is a caught error, not a
  silently-misaligned stack.

`kernel_common.py` is the contract made executable: `stratkfold()` is the single source of
truth for the split, `save_oof_test()` the single writer. Kernels `from kernel_common import
...` instead of copy-pasting boilerplate into 21 `run.py` files.

**Reusable rule:** before generating the *second* artifact of any kind, freeze the contract
that lets artifact #1 and #47 combine. Retrofitting alignment costs far more than declaring
it.

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

Bash (`run_new.sh`, `poll_all.sh`) got the job done but died on every edge case. The Python
driver encodes each edge case as tested logic:

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
background job. On NeuroGolf the analogous fix was `_tools/pin1.py` (17 lines) — single-thread
runtime pinning, without which load average hit **257**.

---

## L3 — Judgment layer: an honest local replica of the remote scorer

The single most valuable pattern in this document. It appears in all three campaigns and was
the highest-value tool in each.

### Verify the objective *before* optimizing it — `verify_scoring.py` (71 lines)

The highest-leverage artifact in the whole repo, per line written. It imports the **real**
`neurogolf_utils` and executes its scoring path instead of trusting the documented formula.

It **falsified the cost function the campaign was about to optimize against**: the research
brief asserted `cost = params + memory + MACs`; execution proved MACs had been removed —
**compute is free**. That flipped the entire target to "don't materialize intermediates,"
which is where essentially every later gain came from (one target 16.23 → 22.70; another
20.62 → 25.00).

The companion `findings/VERIFICATION.md` also caught, by the same method: I/O is one-hot
`FLOAT[1,10,30,30]` not integer; graph input/output are excluded from cost (⇒ a single-node
graph costs **0** memory); `Compress` is banned; and there are **400 targets, not 199** — a
file-listing API had silently truncated at 199 and caused a wrong "correction".

**Reusable rule:** the scoring function is the specification. Execute it before you optimize
against it. A documented formula, a research summary, and an API listing are all hearsay.

### Refuse to be fooled by the metric you can query — `nested_cv.py`

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
| stack_24_final | 0.96954 | 0.97077 | 0.97035 | −0.00042 |

Three findings that generalize:

1. **The highest honest-CV model was the highest private model.** Public LB — a single noisy
   draw on ~49k rows, σ ≈ 0.00087 — ranked them wrong.
2. **Shakedown was proportional to public-minus-honest-CV excess.** +0.00034 excess → fell
   0.00019; +0.00067 → fell 0.00057; +0.00123 → fell 0.00042. The optimism gap is a
   *predictor*, not just a diagnostic.
3. **Kaggle scores `max(selected)` on private.** Picking by honest CV *and* keeping one
   decorrelated hedge secured 4th. Chasing the best public model alone → 0.97047, worse rank.

**Corollary — the anti-hack rule.** The top public cluster (0.9724–0.9728) ran "Ridge Flip +
Probability Consensus": Ridge fitted directly onto past submissions' *public scores* to
reverse-engineer which label flips raise the public 20%. Ruled out as LB probing. The board
shook down on private; that refusal is *how* an honest 0.97060 reached 4th.

### Audit anything you did not produce — `harness/README.md`

Absorbing a third-party artifact set (a public solution pack, another team's models, an
LLM-generated batch) is the highest-leverage and highest-risk move available. Four stages, in
order; each caught what the previous could not:

```
1. SCAN                  per member: candidate vs incumbent, cost + correctness
     ↓                   record UNKNOWN explicitly — never silently skip
2. GATE                  fresh / held-out data, independent seed
     ↓                   the provided examples are a TRAINING SET
3. DIFFERENTIAL          run the authoritative checker on the candidate bundle
   PREFLIGHT             AND on a known-good control, then diff
     ↓
4. SUBMIT                predict the delta first, then compare predicted vs actual
```

- *Scan* — record `loadfail`/`runfail` as **UNKNOWN**, never as "no improvement". 40/400
  members were unrunnable locally; filing them as failures would have discarded 8 that were
  both correct and cheaper (+2.81 on the real metric).
- *Gate* — **any example shipped with the problem is a training set**; the artifact's author
  saw it too. Promote only what survives freshly generated data on an independent seed.
  Rejected **19** members that were exact on every provided example and dirty on fresh data,
  worst at 14.75% error. Two refinements that decided calls: measure the *incumbent on the
  same sample* (equally dirty ⇒ inherent task ambiguity, not a defect), and **match sample
  sizes** (a slow candidate's `0/27` is consistent with a 0.25% error rate; comparing it to
  `1/400` is not a verdict).
- *Differential preflight* — never read a checker failure in isolation. A local checker
  called **94** members fatal; the same check on a control that had already scored cleanly in
  production flagged **96** of the same kind — and since one unscorable member voids the
  batch, a nonzero production score *proves* all members were scorable. All 94 were local
  artifacts. Without the control, 14 good models would have been reverted.
- *Submit* — state the expected delta first. Predicted +191.94, actual +180.86; that 11-point
  gap is what revealed the authority computes memory as `max(static, runtime)` while the local
  probe measured runtime only. **A prediction you never compare against teaches nothing.**

### Grade every number by how it was obtained — `harness/experience_db.py` (323 lines)

An evidence-graded ledger plus a posterior-reward scheduler. Two ideas carry it:

- **Every number carries an evidence level**, and the levels never conflate:
  `estimated` < `local_pass` < `holdout_clean` < `authority_ok` < `measured`.
  Nothing below `authority_ok` ships; nothing at `estimated` may be quoted as fact. This
  exists because an *estimated* score sat in a column that looked identical to measured rows,
  an agent read it as fact, a "fix" shipped, and the real metric fell by exactly that amount.
  A measurement taken *through* a semantics-altering shim is recorded with `instrument=` and
  is not valid evidence at any level.
- **Store refutations, not just wins.** A proven-dead direction stops the next agent spending
  a whole wave re-refuting it. Retrieval is by **structure** (`--bottleneck`, `--sig`), not
  identity, so an agent starting on a new target finds prior work that hit the same wall.
- Corollary it encodes: **no static estimator survives contact.** Four successive "remaining
  headroom" estimates read 425.83 → 44.85 → 36.70 → **1.15** as each was actually checked — a
  370× over-promise. `sched` therefore starts from a heavily damped prior and lets measured
  reward dominate as attempts accumulate. Log the reward of *every* attempt including 0 and
  truncated episodes: after ~10 waves the original ledger held only **4** records carrying a
  realized reward. Outcomes had been recorded; the signal scheduling needs had not.

---

## L4 — Protocol layer: how to run agent fleets

**Where:** `.claude/skills/agent-field-lessons/SKILL.md` (11 lessons),
`.claude/skills/deli-auto-research/SKILL.md`.

- **`agent-field-lessons`** — 11 protocol lessons from the 400-target fleet campaign. Each
  cost real score. Highlights beyond L3's audit pipeline: subagents overclaim ~10× (verify
  against the live baseline, never their reported delta); a compatibility shim is part of the
  measuring instrument; **"unexecutable" is not "defective"**; and *your lower bound is a bound
  on your current frame, not on the problem* — three times a floor was written down and broken
  by changing representation.
- **`deli-auto-research`** — anti-stall / anti-loop conventions for long-horizon autonomous
  runs: fresh-context iterations, file-persisted state, quantitative stall detection with
  *forced* structural pivots, guardian/worker separation. Four pivots run under it returned
  clean negatives — which is a real result: it closed a vein instead of leaving it open.

### Adversarial pre-review of untestable code

Hard rule on S6E6: no local training. A kernel's first execution is on Kaggle, burning quota.
Mitigation: **author, then adversarially review each kernel before pushing** — a second agent
whose only job is to refute it.

Caught, in code that `py_compile` and `ast.parse` both passed: a logloss-vs-balanced-accuracy
early-stopping bug, a dropped-function `NameError`, and a false ceiling claim.

**Reusable rule:** when the feedback loop costs money or hours, insert an adversarial reviewer
*before* execution. Syntax checks do not catch semantic bugs, and semantic bugs are exactly
what an expensive loop punishes.

### Exploit detection, when the rules forbid exploits

Exploits come in at least two classes, and **checking only the one you thought of finds only
that one**:

| class | example | how it presents |
|---|---|---|
| undefined semantics | pooling with a **negative stride** | fails spec validation; works only via undefined runtime behaviour |
| accounting gap | 577,192 elements of parameters parked in `TfIdfVectorizer` attributes (22/400 models) — `calculate_params()` enumerates initializers and *Constant-node* attributes only | **passes** spec validation — the gap is in the scorer, not the format |

The first was found by hypothesis (cost ~15.3 points to decline); the second only by a
multi-lens audit that included a "cost-accounting evasion" lens (cost 22.20). Both sat in
the same third-party pack — one whose provenance was airtight (immutable Kaggle result JSON;
the shipped `.zip.bin` hashed to the sha256 its own manifest claimed). **Provenance quality
does not predict method quality.** Declining both was deliberate: 7647.97 → **7625.77**.

"Fails the local checker" is **not** the test — legitimate-but-exotic constructs fail it too
(negative *pads* have well-defined crop semantics and appear in every pack). The usable test:
(1) does the spec define behaviour for this construct at all, and (2) is it an isolated
outlier or a widespread idiom? The exploit appeared in 5/400 of one pack and 0/400 of two
others; the legitimate construct was the norm everywhere. And do not condemn a whole pack for
containing one: of 9 flagged artifacts, 5 were exploits and one was a legitimate single-node
construction worth +4.38.

---

# Part II — Inventory: where each piece lives

## A. Generic, and now in this repo

| what | evidence it helped |
|---|---|
| `harness/experience_db.py` (323L) | evidence grading + posterior-reward scheduler — see L3 |
| `harness/README.md` | the four-stage audit pipeline — see L3 |
| `.claude/skills/agent-field-lessons` (L1–L11) | L1–L6 written 2026-07-25 then froze while 9 more rules accumulated in the competition repo; L7–L11 are that backport |
| `src/kaggle_agent/tools/kernel_fleet.py` (1000L) | the only module that flowed *back* from real competition use — see L2 |

## B. Competition-specific, but the shape transfers

| what | why keep it |
|---|---|
| `competitions/neurogolf-2026/harness/verify_scoring.py` (71L) | the objective-falsifier — highest leverage per line in the repo |
| `competitions/neurogolf-2026/findings/VERIFICATION.md` | claim-by-claim check against the authoritative source |
| `competitions/multi-step-tool-attacks/harness/eval_replay.py` | offline evaluator that replays candidates exactly like the hosted gateway and scores them with the real scorer — no GPU, no target LLM |
| `competitions/multi-step-tool-attacks/FINDINGS.md` | scoring/predicate/guardrail mechanics reverse-engineered *before* optimizing against them |
| `competitions/playground-series-s6e6/{nested_cv,select_final,hillclimb}.py` | the honest-selection toolkit — see L3 |
| `experiment_log.py` → append-only `experiments.csv` + `.jsonl` | cheap, and the only reason a 6-week campaign can be audited afterwards |
| `LEADER_GAP_ANALYSIS.md` | timestamped log including failed experiments and **revised** conclusions. The revisions are the valuable part |

## C. Domain-specific → stays in `neurogolf-26`

Porting these would be copying ONNX/ARC internals under a generic name.

| tool | lines | role |
|---|---:|---|
| `_tools/nghar.py` | 381 | cost probe, exactness, fresh-data gate, merge — the workhorse |
| `_tools/ngkb.py` | 378 | the domain instance of `experience_db` (456 records) |
| `_tools/gate.py` | 116 | fresh-data gate, time-bounded, reports actual n |
| `_tools/preflight.py` | 128 | official-scorer full pass |
| `_tools/blockers.py` | 82 | official-scorer blocker check over a bundle |
| `_tools/scan_rmz.py` | 108 | full-transparency pool scan (records UNKNOWNs) |
| `_tools/pin1.py` | 17 | single-thread runtime pinning (load average hit 257 without it) |
| `_tools/ngbuild.py` / `ngpatterns.py` / `ngtemplates.py` | 940 | ONNX construction primitives |
| `METHODS.md` | 236 / 42 rules | the domain rulebook — the actual carrier of the campaign |

---

# Part III — Anti-inventory: proven not to work

Do not rebuild these.

## Modeling dead ends (S6E6: balanced accuracy, 577k rows, 8 raw features)

| Attempt | Standalone OOF | Contribution | Note |
|---|---|---|---|
| Mass feature engineering (243 feats, 76 groupby aggs) | 0.96539 | **+0** | 0.984 argmax-agreement with the best base; also hit the 12h CPU wall |
| Optuna HPO (60–100 trials) | 0.96506 / 0.96562 | **+0** | Lost to hand-tuned 0.96877. A cheap 10-trial search was *negative*: 0.96427 < 0.96441 default |
| AutoGluon (good_quality, bagged) | 0.95646 | **+0** | |
| DAE representation learning (swap-noise) | 0.96444 | **+0** | 8 clean features: nothing to learn |
| Hierarchical cascade (STAR-vs-rest → GALAXY-vs-QSO) | 0.96335 | **net loss** | Moves recalls around, adds no coverage |
| Pseudo-labeling (transductive, conf ≥ 0.995) | 0.96561 / 0.96538 | **+0** | Identical to plain GBDTs |
| ICL family (TabPFN-3 / TabICL) | 0.93591 / 0.95797 | **negative** | 30 models 0.97041 → 31 models 0.97040 |
| Calibration & decision rules | — | +0.00004 / +0.00001 / **−0.00065** | scale+shift, temperature, equal-recall. Additive bias was already optimal |
| Per-seed features into the meta | 0.96705 | −0.00003 | Provably redundant: a linear meta over per-seed log-probs ≡ over their average |
| Anomaly patching / probability re-vote | — | **hurts in every config** | The meta already beats the fallback on high-disagreement rows |
| ExtraTrees | 0.94166 | −0.00008 | Rejected by nested-CV before submission |

Two structural lessons underneath the table:

- **More correlated features make the ensemble worse, not better.** The `all_v3` feature set
  lifted every single model (lgb +0.0002) but the blend was a *tie* (0.96608 vs 0.96609).
  `analyze_correlation.py` gave the mechanism: pairwise error-Jaccard rose 0.716 → 0.727 while
  the oracle ceiling *fell* 0.97415 → 0.97384. **Track the oracle ceiling, not the
  single-model score, when adding a member.**
- **The honest ceiling is provable and worth proving.** 24 own models + a GM's 19 published
  OOFs + a GBDT meta + a transformer-ICL model all landed at OOF ~0.9704. At that point the
  correct action is to report the ceiling, not to keep grinding.

## Process failures

- **Noise-limited regime was detected on 06-22 and then ignored for days.** The 9-model stack
  had the best OOF (0.96713) but a *worse* public LB (0.96740) than a 7-model stack (0.96680 →
  0.96749). Every subsequent "add one more marginal model" was inside the noise. **Once
  4th-decimal OOF stops tracking LB, stop adding members; switch to variance reduction and
  honest selection.**
- **The best public submission was almost never generated.** `gbdt_meta_pool48` existed only
  as a *level-1* component inside a level-2 meta-of-metas. Its standalone honest CV (0.97037)
  was within 8e-5 of the ensemble containing it (0.97045) — the number was seen and not acted
  on, assuming the ensemble subsumed it. It did not: standalone scored public 0.97104, beating
  the ensemble that contained it by +0.00025. **Rule: whenever a sub-component's honest score
  is within ~1σ of the ensemble built on top of it, materialize and submit it standalone.**
  Compounding cause: the file was produced by parallel work after the last `submissions/` scan
  and a context compaction hid it. **Rule: re-inventory `submissions/` and `artifacts/` before
  declaring a sweep exhausted — a sweep is only as complete as the last directory listing.**
- **The lessons file was written, then was unreachable from where the work happened.**
  `agent-field-lessons` (L1–L6) lived in this repo's *project-local* `.claude/skills/`. The
  NeuroGolf campaign ran in a different repo, so the skill **was never loadable during the
  campaign it was distilled from** — and L1-class errors were re-derived three times: a
  validator false-positive nearly reverted 14 good artifacts; a wiped scratchpad silently
  turned all 400 targets into phantom failures; and three verdicts nearly rested on comparing
  `0/27` against `1/400`. It also froze on 2026-07-25 while nine more rules accumulated in
  the other repo (backported here as L7–L11, five weeks late).
  **Rule: knowledge must live where the work happens — global `~/.claude/skills/`, not
  project-local — and backporting is a scheduled step, not a hope.**

## Built, but never used

**Retention rule:** anything not written for the campaign that exposed it is *listed, not
deleted* — "unused by me" is not "worthless". Only dead weight written **for** a campaign is
removed, and then only with its evidence recorded here.

| what | status |
|---|---|
| **`src/kaggle_agent/`** — `orchestrator.py` (657L), `cli.py`, `interaction.py`, `llm/`, `knowledge/` | The repo's nominal purpose: a state machine `INITIALIZING → … → COMPLETED`, a Cursor file-handoff LLM protocol, `PlaybookManager` / `SkillManager` / `ReflectionEngine`. **Zero references from any competition directory** — imported only by its own tests. All three campaigns ran with empty `agent_tasks/`, empty `experiments/`, no `state.json`. The effective loop was Claude Code driving scripts directly, steered by `CLAUDE.md`. |
| **`competitions/neurogolf-2026/harness/research_harness.py`** (165L) + `missions/neurogolf.json` + `requirements.txt` | **DELETED 2026-09-02** (in git history). Parallel research fan-out (N researchers + 1 synthesizer). Written *for* this competition and **never actually run** for it — `findings/` contains no `research_*.md` or `synthesis.md`, and its own README conceded the pages "were researched by hand this run". Worse, its product was wrong exactly where it mattered: the mission asserted `cost` included MACs, and `verify_scoring.py` had to falsify it. |

**The pattern:** every harness piece that worked was written *in response to a failure that had
already happened.* The two designed up-front from a spec — the agent framework and the research
orchestrator — went unused, and the one that did produce output produced a wrong objective.

---

# Part IV — Open structural problem

**The skills are project-local.** `agent-field-lessons` and `deli-auto-research` live in
`kaggle-agent/.claude/skills/`. Working in the `neurogolf-26` repo they were **not loadable at
all** — so even the correctly-written L1–L6 did nothing during the campaign, and L1-class
errors were re-derived three times from scratch: a false-positive validator reading, a wiped
scratchpad silently turning 400 targets into phantom failures, and an unmatched-sample-size
comparison.

Moving them to `~/.claude/skills/` would make them global. **Not done — awaiting a decision.**

---

## The one-line version

**Reusable:** an evidence-graded ledger, a four-stage audit pipeline, a fault-tolerant remote
fleet driver, and 11 protocol lessons.
**Decisive:** an honest local replica of the scorer — 71 lines that falsified the objective on
NeuroGolf, and a nested-CV that picked the private winner on S6E6.
**Unused:** the agent framework this repo is named after.
