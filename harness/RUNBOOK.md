# RUNBOOK — running a competition through this harness

The operating procedure distilled from three campaigns: Playground S6E6 (private
**4th / 2817**), NeuroGolf 2026 (**247 → 7625.77**), and MSTA. It is ordered the way a
campaign actually runs, and every step names the competition where skipping it cost
something real.

Read it as a checklist, not a philosophy. The two per-competition records
([S6E6](../competitions/playground-series-s6e6/HARNESS.md),
[NeuroGolf](../competitions/neurogolf-2026/HARNESS.md)) are the evidence behind each step.

> **The single most transferable finding:** every harness piece that worked was written in
> response to a failure that had *already happened*. The pieces designed up-front from a spec
> went unused. So: do Phase 0–3 up front because they are cheap and their absence is
> catastrophic, then let the rest accrete from real failures.

---

## Phase 0 — Verify the objective before optimizing it

**Do this first. It is the highest-leverage hour of the campaign.**

Do not trust the competition description, a research summary, or an API listing. Import the
**real scorer** and execute its path on a toy input.

```bash
# NeuroGolf: 71 lines that imported the authoritative neurogolf_utils and ran its scoring path
python competitions/neurogolf-2026/harness/verify_scoring.py
```

What that one script found:

| claim | reality |
|---|---|
| `cost = params + memory + MACs` | **MACs had been removed — compute is free** |
| I/O is integer | one-hot `FLOAT[1,10,30,30]` |
| graph input/output count toward cost | excluded ⇒ a single-node graph costs params alone |
| 199 tasks | **400** — the file-listing API had silently truncated |

The first row reversed the entire optimization target, from "reduce arithmetic" to "never
materialize an intermediate". Every large gain in that campaign is downstream of it.

**Checklist**
- [ ] Locate the authoritative scorer/checker (a package, a public kernel, the eval harness).
- [ ] Execute it locally on a trivial input; print every intermediate quantity it computes.
- [ ] Write the verified mechanics into `findings/VERIFICATION.md` — claim by claim, each
      marked confirmed or **falsified**.
- [ ] Reconcile against the task/row count you *think* you have. Truncated listings are silent.

---

## Phase 1 — Freeze the artifact contract, before the second artifact exists

Every campaign becomes a pile of independently produced artifacts. Decide *now* how any two
of them combine, or pay to retrofit it later.

The S6E6 contract, which never changed:

```
StratifiedKFold(5, shuffle=True, random_state=42) on integer y, in train-CSV row order
GALAXY=0 / QSO=1 / STAR=2   |   OOF (577347,3) float32   |   test (247435,3) float32
artifacts/oof_<name>.npy    +   artifacts/test_<name>.npy
```

The property that matters: **the split depends only on `y` and the seed** — not on features,
not on the model, not on who ran it. So anything anyone produces aligns row-for-row on arrival.

What it paid: 167 arrays from 30 kernels composed via a CLI flag, and 13 days later a Kaggle
GM's **19 published OOF arrays dropped in with zero glue code** — worth **+0.00075 honest CV**,
the largest late-campaign gain, for the price of a download.

**Checklist**
- [ ] One `config.py` as the single source of truth for split, seed, label order, shapes.
- [ ] Make the split a pure function of `(y, seed)`. Never of features or model.
- [ ] One writer function for artifacts; one naming scheme.
- [ ] State the expected shape/dtype so a mismatch is a caught error, not a silent misalignment.

---

## Phase 2 — Put every expensive fact into `CLAUDE.md` the day you learn it

`CLAUDE.md` is the only text guaranteed to be in context at the start of every session. It is
where facts go that no amount of reasoning recovers.

The test is not "is this interesting" but **"would a fresh session repeat this mistake?"**

Four categories, with the S6E6 instances:

| kind | instance | what it prevented |
|---|---|---|
| irreversible-risk gate | never commit an active competition's solution (the remote is public) | leaking the approach |
| resource-shape rule | **never train locally** — 577k rows × one sklearn fit = hours | the rule that *forced* the remote pivot |
| metric lock | the metric is balanced accuracy, **not** logloss | a real early-stopping bug |
| environment landmine | P100 is sm_60; cuDF dropped Pascal; GPU cap = 2; slug poisoning | four walls that each killed kernels |

**Checklist**
- [ ] Add the fact the same day, with the symptom *and* the fix.
- [ ] Prefer a stop condition over an observation ("stop adding members when X" beats "X is true").
- [ ] Note: skills in a project-local `.claude/skills/` are **not loadable from another repo**.
      On NeuroGolf that cost three re-derivations of the same class of error. Put cross-project
      lessons in `~/.claude/skills/`.

---

## Phase 3 — Build the remote execution path

If training cannot be local, build the remote loop before you need it under time pressure.

```bash
# S6E6 shape — code ships as a DATASET, not duplicated per kernel
bash kaggle/sync_code.sh "description"   # → `kaggle datasets version`
bash kaggle/run_remote.sh                # push → poll → pull artifacts/*.npy
# local: stack/blend (pure numpy, seconds) + submit
```

One experiment = one directory with `kernel-metadata.json` + `run.py`. The `run.py` imports the
shared code from the dataset, so 21 kernels share one copy.

Once the shell scripts start dying on edge cases, replace them with
[`kernel_fleet.py`](../src/kaggle_agent/tools/kernel_fleet.py):

| function | edge case it survives |
|---|---|
| `gpu_cap_admits()` | 2 concurrent GPU sessions max; CPU kernels bypass the cap |
| `classify_push_output()` | "retry later" vs "slug is dead" vs a real error |
| `bump_slug()` | slug poisoning — a failed *first* push kills a slug permanently |
| `pull_and_verify()` | truncated outputs; `np.load` + shape-check, up to 3 re-pulls |
| `diagnose_log()` | 6 known failure signatures → the actual fix text |
| `run_fleet()` | one bad kernel never takes down the batch |

**Checklist**
- [ ] Kernels must save the *most valuable* artifact first. A mass-FE kernel hit the 12h wall
      and was cancelled — its OOF survived because it was written before the test predictions.
- [ ] Run the fleet driver as a **tracked background job**. A foreground driver was killed by an
      overnight session restart.
- [ ] For code you cannot test locally: author it, then have a **second agent adversarially
      review it** before spending quota. That caught a metric bug and a `NameError` that
      `py_compile` and `ast.parse` both passed.

---

## Phase 4 — Build the honest estimator before you need it

Any metric you can query repeatedly is a training set. So is any example shipped with the
problem — the artifact's author saw it too.

Two forms, same idea:

- **S6E6 — `nested_cv.py`.** The plain stack fits calibration on the full meta-OOF and scores
  on that same data. Nested-CV holds every row out of both the meta fit *and* the calibration
  fit. It exposed a **0.00090 optimism gap** in the linear meta versus 0.00006 in the winner.
- **NeuroGolf — the fresh-data gate.** Promote nothing that is only exact on the provided
  examples; require freshly generated data on a seed independent of whatever built the
  candidate. It rejected **19 members** that were clean on every provided example and dirty on
  fresh data, one at 14.75% error.

Two refinements that decided real calls:
- Measure the **incumbent on the same sample**. Equally dirty ⇒ inherent task ambiguity, not a
  candidate defect.
- **Match sample sizes.** A slow candidate's `0/27` is consistent with a 0.25% error rate;
  comparing it against `1/400` is not a verdict.

**Why this is the step that pays:** on S6E6 nested-CV added zero points and picked the winner.
The highest honest-CV model was the highest *private* model; the best *public* model lost.
Shakedown tracked public-minus-honest excess almost monotonically (+0.00034 → fell 0.00019;
+0.00067 → fell 0.00057; +0.00123 → fell 0.00042).

**Checklist**
- [ ] Write down, in advance, that final selection is by honest CV. The temptation to pick the
      pretty public number arrives late, under time pressure.
- [ ] Track the **oracle ceiling** (any-model-correct), not single-model scores, when deciding
      whether to add a member. On S6E6 a feature set lifted every model yet the blend tied —
      pairwise error-Jaccard rose 0.716 → 0.727 while the oracle ceiling *fell*.

---

## Phase 5 — Scale to a fleet, and give it a memory

Once multiple agents work in parallel, the bottleneck stops being compute and becomes
**re-deriving what another wave already knew**.

```bash
export EXPDB_DIR=./_expdb

# BEFORE dispatching an agent — paste this into its prompt
python harness/experience_db.py brief --target 17

# AFTER every attempt, including reward 0 and truncated episodes
python harness/experience_db.py attempt '{"target":17,"direction":"<structural frame>","reward":0.0}'

# what to work on next
python harness/experience_db.py sched --top 20
```

Two rules the tool enforces:

1. **Store refutations, not just wins.** A dead direction stops the next agent burning a wave.
   Retrieval is by *structure* (`--bottleneck`, `--sig`), so an agent on a new target finds
   whoever hit the same wall.
2. **Every number carries an evidence level**, and they never conflate:
   `estimated` < `local_pass` < `holdout_clean` < `authority_ok` < `measured`.
   **Nothing below `authority_ok` ships. Nothing at `estimated` may be quoted as fact.**
   A number measured *through* a semantics-altering shim records `instrument=` and is not valid
   evidence at any level.

This exists because an estimated score sat in a column that looked identical to measured rows;
an agent read it as fact, a "fix" shipped, and the authoritative metric fell by exactly that
amount. The incumbents had been fine.

**And distrust every static estimator.** Four successive "remaining headroom" estimates read
425.83 → 44.85 → 36.70 → **1.15** as each was actually checked — a **370× over-promise**. Treat
a headroom number as a screen for what to measure, never as headroom. `sched` therefore starts
from a damped prior and lets measured reward take over.

**Checklist**
- [ ] Log the reward of **every** attempt, including 0 and truncated ones. After ~10 waves one
      ledger held only 4 records with a realized reward — conclusions had been stored, signal
      had not.
- [ ] Prune **directions**, not targets. A failed method is evidence about the method.
- [ ] Re-measure any subagent's claimed win against the live baseline. They overclaim ~10×.

---

## Phase 6 — Auditing a third-party artifact set

The highest-leverage *and* highest-risk move available. Four stages, in order — each catches
what the previous cannot. Full detail in [`README.md`](README.md).

```
1. SCAN                per member: candidate vs incumbent, cost + correctness
     ↓                 record UNKNOWN explicitly — never `try/except: continue`
2. GATE                fresh / held-out data, independent seed
     ↓                 the provided examples are a TRAINING SET
3. DIFFERENTIAL        run the authoritative checker on the candidate bundle
   PREFLIGHT           AND on a known-good control, then diff
     ↓
4. SUBMIT              predict the delta first, then compare predicted vs actual
```

The two that are least obvious and paid the most:

- **Scan records UNKNOWN.** 40 of 400 members could not run locally. Filed as unknown rather
  than failed, 8 were later escalated to the authority and **8/8 proved correct and cheaper: +2.81**.
- **Preflight needs a control.** The local checker called **94** members fatal; the identical
  check on a bundle that had already scored cleanly in production flagged **96** of the same.
  Since one unscorable member voids the batch, a nonzero production score *proves* all members
  were scorable — so all 94 were local artifacts. Without the control, 14 good models would
  have been reverted.

**If the rules forbid exploits**, note that exploits come in classes and *checking only the
class you hypothesised finds only that class*. NeuroGolf's pack contained both an undefined-
semantics exploit (negative-stride pooling, fails validation) and an accounting-gap exploit
(577,192 parameters parked in `TfIdfVectorizer` attributes — **passes** every spec check,
because the scorer enumerates initializers and Constant-node attributes only). Declining both
was deliberate: 7647.97 → **7625.77**. And the pack's provenance was airtight —
**provenance quality does not predict method quality**.

---

## Phase 7 — Choosing what to submit

- [ ] Select by **honest CV**, not the public leaderboard.
- [ ] Keep one **decorrelated hedge** as the second pick. Kaggle scores `max(selected)` on
      private; on S6E6 the hedge is what secured 4th, and picking the best-public model alone
      would have scored worse.
- [ ] **Materialize strong sub-components standalone.** If a level-1 meta or single base has an
      honest score within ~1σ of the ensemble built on top of it, submit it on its own.
      Averaging a strong member with weaker siblings only cuts variance; on a single draw the
      pure member can win. On S6E6 the level-1 GBDT meta scored public 0.97104, beating the
      ensemble that *contained* it — and was nearly never generated.
- [ ] **Re-inventory `submissions/` and `artifacts/` before declaring the sweep exhausted.**
      A sweep is only as complete as the last directory listing. That submission was produced by
      parallel work after the last scan and a context compaction hid it.

---

## Stop conditions

Recognising these late is the most common way to waste a week.

| signal | what it means | what to do |
|---|---|---|
| 4th-decimal CV gains stop tracking the leaderboard | **noise-limited**: LB σ (~0.001 on 49k rows) exceeds the spread you are chasing | stop adding members; switch to variance reduction and honest selection |
| adding a strong new family moves the ensemble by ~0 | the ceiling may be **intrinsic to the data** | prove it — test against the best pool available, including published OOFs from a GM and a different model *family*. If those do not move it, report the ceiling |
| a dense cluster sits far above you on the public LB | usually a **shared public recipe**, often leaderboard probing | pull it and read it before grinding. If it fits the public score directly, refuse it — it shakes down on private |
| four estimates of remaining headroom in a row | every static estimator over-promises | measure, do not estimate |

On S6E6 the noise-limited signal fired on 06-22 and was overruled for three days. **A detector
nobody acts on is worth nothing** — which is why these belong in `CLAUDE.md` as stop conditions.

---

## Using `neurogolf-26` as the reference implementation

The NeuroGolf solver repo is attached as a submodule at
`competitions/neurogolf-2026/solver/`, so the fully-worked domain instance of everything above
is readable alongside the generic version here.

```bash
git submodule update --init competitions/neurogolf-2026/solver
```

| generic (here) | worked instance (submodule) |
|---|---|
| `harness/experience_db.py` — evidence ladder + scheduler | `neurogolf_solver/agent_kit/_tools/ngkb.py` (456 records; levels `estimated < official_exact < gen_clean < scorer_ok < lb_confirmed`) |
| `harness/README.md` §Gate | `_tools/gate.py` — fresh-generation gate, time-bounded, reports the achieved `n` |
| `harness/README.md` §Differential preflight | `_tools/preflight.py`, `_tools/blockers.py` — official-scorer passes over a whole bundle |
| `harness/README.md` §Scan | `_tools/scan_rmz.py` — pool scan that records UNKNOWNs |
| Phase 0 verification | `competitions/neurogolf-2026/harness/verify_scoring.py` |
| Phase 2 rules layer | `neurogolf_solver/agent_kit/METHODS.md` — 42 domain rules |
| — | `_tools/pin1.py` — single-thread pinning; load average hit **257** without it |

**Two caveats before running anything in there.** The submodule is a reference, not a
turn-key kit:

1. **Hardcoded absolute paths.** Nine tracked `.py` files (`_tools/gate.py`,
   `preflight.py`, `blockers.py`, `scan_rmz.py`, and five `_cands_w41/build*.py`) still
   `sys.path.insert` an absolute `/Users/.../Documents/coding/neurogolf-26/...`. They will not
   run from the submodule path unedited.
2. **The venv it names is ephemeral.** `METHODS.md` points at
   `/private/tmp/claude-501/ngvenv/bin/python` (onnx 1.22 / ort 1.23.2). That path does not
   survive a reboot; recreate the environment rather than expecting it.

Read them for the *shape*. Porting the code would mean copying ONNX/ARC internals under a
generic name, which is why only `ngkb.py` was generalized.
