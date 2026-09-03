# NeuroGolf 2026 — how the harness changed the outcome

**An account of what the harness actually did during the run** — where it intervened, what it
caught, and what would have shipped without it.

**Result: 247 → 7625.77**, 400 targets, under hard no-overfit *and* no-exploit constraints.

The honest thesis: **the harness's job was to catch me being wrong, and it had to do that
repeatedly.** Almost every score gain below is downstream of a moment where a mechanism
overruled my judgement. The ledger records 12 refuted directions; most were my own
conclusions.

---

## The interventions, in order

### 1. Before any optimization: the objective itself was wrong

I was about to golf `cost = params + memory + MACs`. Running `verify_scoring.py` — 71 lines
that import the *real* scorer and execute its path — showed MACs had been removed from the
objective. **Compute is free.**

*What it changed:* the target flipped from "reduce arithmetic" to "never materialize an
intermediate tensor". Every large gain in the entire campaign came from that reframe — one
target 16.23 → 22.70, another 20.62 → 25.00 via a single 50-operand Einsum.
*Without it:* six weeks optimizing a quantity that costs nothing.

It also corrected the task count from 199 to **400** — the file-listing API had silently
truncated, and I had already "corrected" the number in the wrong direction.

### 2. Absorbing a third-party pack: the scan refused to guess

The pack scan classifies each member as `ok / inexact / loadfail / runfail / same` rather
than `try/except: continue`. 40 of 400 members could not be executed locally.

*What it changed:* those were filed as **UNKNOWN**, not as failures. Later, 8 of them were
escalated to the authoritative environment and **8/8 proved correct and cheaper: +2.81**.
*Without it:* a `continue` would have silently discarded all 40.

### 3. The gate overruled "it passes every provided example"

Every candidate had to be clean on freshly generated data with a seed independent of anything
used to build it. Across three audits this rejected **19 members that were exact on every
provided example and dirty on fresh data** — the worst at 14.75% error.

*What it changed:* 19 overfit artifacts did not ship.
*Without it:* they all looked perfect by the only check I would otherwise have run.

Two refinements earned their place mid-run. Measuring the *incumbent* on the same sample
separated "this candidate is broken" from "this task is inherently ambiguous" — for four
targets every clean alternative showed the identical dirt rate, which is a property of the
generator, not of any model. And matching sample sizes mattered: three verdicts nearly rested
on comparing a slow incumbent truncated at `0/27` against a candidate at `1/400`, which is no
comparison at all. Re-running the slow side at n=216–400 confirmed those verdicts on real
evidence.

### 4. The differential preflight stopped me reverting 14 good models

The local official-scorer pass reported **94 members as fatal**. I was ready to act on it.

Running the identical check on a control that had already scored cleanly in production showed
**96 of the same class**. And since one unscorable member voids the whole zip, a nonzero
production score *proves* every member was scorable — so all 94 were local toolchain
artifacts (a stricter `onnx` version, missing runtime kernels).

*What it changed:* 14 good artifacts stayed in.
*Without it:* I would have reverted them on a confident, wrong reading.

### 5. Predict-then-compare exposed a hidden cost rule

Stated before submitting: **+191.94**. Actual: **+180.86**.

*What it changed:* the 11-point gap was the finding — the authority computes memory as
`max(static_shape, runtime_profile)` while my probe measured runtime only.
*Without it:* a win, unexamined, and a wrong cost model still in use.

### 6. The ledger overturned my own "impossible"

I had recorded one target as **refuted**: "any 0-dirt construction needs ≥2 large contractions,
which exceeds budget." Because the refutation was written down *with its reasoning*, when a
counterexample appeared I could see immediately which premise died — the winning construction
indexed its intermediate by *hypothesis* rather than by grid position, so no large contraction
was needed.

*What it changed:* that target went 16.23 → 19.41 → 22.70.
*Without it:* a permanent "impossible" tag on a solvable target.

The same error recurred twice more (a "80 bytes is the floor" that fell to cost 1; a headroom
bound off by 370×). The pattern the ledger surfaced: **my lower bounds were bounds on the
frame I was in, not on the problem.**

### 7. Evidence grading stopped a wrong number propagating twice

An *estimated* score once sat in a column that looked identical to measured rows. An agent
read it as fact, I shipped a "fix", and the real metric fell by exactly that amount (−2.70).

*What it changed:* the ledger now refuses to store a number without how it was obtained
(`estimated < local_pass < holdout_clean < authority_ok < measured`), and the importer
auto-downgrades rows whose provenance says "fallback". On re-import, 24 rows were caught.
*Without it:* the same conflation, again.

### 8. The scheduler stopped a theoretical jackpot holding the fleet hostage

One target advertised a theoretical +4.91 and had already absorbed two full waves for zero
findings. Under posterior-reward scheduling, **one logged zero-reward attempt moved it from
rank 2 to rank 72.**

*What it changed:* effort moved to unmeasured arms.
*Without it:* "highest headroom" ranking would have kept sending agents back.

This is also what four successive headroom estimates taught: `425.83 → 44.85 → 36.70 → 1.15`
as each was actually checked. **No static estimator survived contact**, so allocation had to
run on measured reward.

### 9. The multi-lens audit found the exploit my hypothesis missed

I checked for negative strides — my hypothesis — and found 5 models. An audit run along
several independent lenses, including one aimed at the *scoring code* rather than the artifact
format, found a second and entirely different class: **577,192 elements of parameters parked
in `TfIdfVectorizer` attributes across 22 models**, free because `calculate_params()`
enumerates initializers and Constant-node attributes only.

*What it changed:* both were excluded, deliberately, for **−37.5 points** (7647.97 → 7625.77).
*Without it:* the second class would have shipped, undetected, inside an "honest" bundle.

The separating test, since "fails the local checker" is not one (legitimate negative *pads*
fail it too): does the spec define the construct at all, and is it an outlier or an idiom?
Negative strides appeared in 5/400 of one pack and 0/400 of two others.

---

## Where the harness was absent, and it cost me

- **The lessons file was unreachable from the work.** `agent-field-lessons` lived in
  project-local `.claude/skills/` in *this* repo while the campaign ran in another, so it was
  never loadable during the campaign it was distilled from. L1-class errors were re-derived
  three times: the 94-false-positive reading above, a wiped scratchpad silently turning all
  400 targets into phantom failures, and the `0/27` vs `1/400` comparison.
- **State lived in a scratchpad that gets wiped mid-session.** Twice this destroyed completed
  measurements; once it made an entire pack look broken. Fixed by falling back to in-repo data
  and moving artifacts to `_measure/`.
- **Concurrency had no protocol.** A parallel process committed my working tree mid-edit, and
  8 seconds later I overwrote a 355-line document with an 80-line one. Recovered via
  `git checkout`, then merged as `+36/-0` — but only because I checked before assuming.

## What this record is not

The tooling that ran the campaign is **not in this repo** — it is
`neurogolf-26/neurogolf_solver/agent_kit/` (42 rules in `METHODS.md`, ~2,200 lines of tools).
It was not ported because porting it would mean copying ONNX/ARC internals under a generic
name. The parts that *do* generalize were distilled into `/harness/` and into L1–L11 of
`agent-field-lessons`; this file is the account of what they did here.
