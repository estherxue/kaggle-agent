# harness — the parts that actually carried a campaign

Reusable pieces distilled from `neurogolf-2026` (400 independent optimization targets,
score 247 → 7625.77, hard no-exploit / no-overfit constraint). Everything here earned its
place by catching a real error; nothing is included because it seemed like good practice.

> **Looking for the operating procedure?** [`RUNBOOK.md`](RUNBOOK.md) is the phase-ordered
> checklist for running a competition through this harness. This file is the deep dive on one
> of its phases — auditing third-party artifacts (RUNBOOK Phase 6).

## What is here

| file | what it is |
|---|---|
| `RUNBOOK.md` | the phase-ordered operating procedure for running a competition |
| `experience_db.py` | evidence-graded ledger + posterior-reward scheduler (the one genuinely generic tool) |
| `../competitions/neurogolf-2026/harness/verify_scoring.py` | imports the REAL scorer and runs its path — the ancestor of everything below |

The audit pipeline itself is documented below rather than shipped as an abstract framework:
its code is inseparable from ONNX/ARC, and a hook-based "generic" version would be an empty
shell. The *shape* is what transfers.

---

## The audit pipeline (pattern)

Absorbing a third-party artifact set — a public solution pack, another team's models, an
LLM-generated batch — is the highest-leverage and highest-risk move available. Four stages,
in this order. Each caught something the previous one could not.

```
  1. SCAN          per member: candidate vs incumbent, cost + correctness
       ↓           record UNKNOWN explicitly — never silently skip
  2. GATE          fresh / held-out data, independent seed
       ↓           the provided examples are a TRAINING SET
  3. DIFFERENTIAL  run the authoritative checker on the candidate bundle
     PREFLIGHT     AND on a known-good control, then diff
       ↓
  4. SUBMIT        predict first, then compare predicted vs actual
```

### 1. Scan — record unknowns as unknowns

The naive scan does `try: measure() except: continue`, which files "I could not run this"
under the same heading as "this is not an improvement". Use five explicit states:

`ok` / `inexact` / `loadfail` / `runfail` / `same`

`loadfail` and `runfail` mean **exactness is UNKNOWN**, not false. In one audit 40 of 400
members could not be executed locally; treating those as failures would have discarded 8
models that later proved both correct and cheaper (+2.81 on the real metric).

### 2. Gate — the provided examples are a training set

Any metric you can query repeatedly is a training set, and so are any examples shipped with
the problem: the artifact's author could see them too. Promote only what also survives
freshly generated / held-out data, drawn with a seed independent of anything used to build it.

Concretely, across three audits this stage rejected **19 members that were exact on every
provided example and dirty on fresh data** — including one at 14.75% error. Without it they
would all have shipped.

Two refinements that matter:
- Measure the **incumbent on the same sample**. If it is equally dirty, the dirt is inherent
  task ambiguity, not a defect of the candidate. That distinction decided several calls.
- **Match the sample sizes.** A slow candidate hitting a time budget yields `0/27`, which is
  perfectly consistent with a 0.25% error rate — comparing it against `1/400` is not a
  verdict. Re-run the slow side with a longer budget before believing the comparison.

### 3. Differential preflight — never read a checker failure in isolation

Run the authoritative checker over **every** member; one unscorable member can void an
entire batch. But its failures are unreadable on their own: a local toolchain one version
behind the authority produces failures that are artifacts, not defects.

So run the identical check on a **control that is already known-good in production**, and
diff. Only failure classes ABSENT from the control are candidate blockers.

In the source campaign the local checker reported 94 "fatal" members. The control — a bundle
that had scored cleanly in production — showed 96 of the same. Since one unscorable member
voids the batch, a nonzero production score PROVES every member was scorable, so all 94 were
local artifacts. Without the control, 14 good models would have been reverted.

### 4. Submit — predict, then compare

State the expected delta before submitting. Predicted +191.94, actual +180.86; the 11-point
gap is what surfaced that the authority computes memory as `max(static, runtime)` while the
local probe measured runtime only. A prediction you never compare against teaches nothing.

---

## Exploit detection

If the objective has a "no exploits" constraint, note that exploits come in at least two
classes and **checking only the one you thought of finds only that one**.

| class | example | how it presents |
|---|---|---|
| undefined semantics | pooling with a NEGATIVE stride | fails spec validation; works only via undefined runtime behaviour |
| accounting gap | parameters parked in an op attribute the cost function never enumerates | **passes** spec validation; the gap is in the scorer, not the format |

The first was found by hypothesis and cost ~15 points to decline. The second was found only
by a multi-lens audit that included a "cost-accounting evasion" lens, and cost 22.20. Both
were present in the same third-party pack.

**"Fails the local checker" is not the test** — legitimate-but-exotic constructs fail it too
(negative *pads* have well-defined crop semantics and appear in every pack). The usable test:

1. does the spec define behaviour for this construct at all, and
2. is it an isolated outlier or a widespread idiom? (the exploit appeared in 5/400 of one
   pack and 0/400 of two others; the legitimate construct was the norm everywhere)

And do not condemn a whole pack for containing an exploit: from the same 9 flagged artifacts,
5 were exploits and one was a legitimate single-node construction worth +4.38.

---

## Using experience_db

```bash
export EXPDB_DIR=./_expdb

# BEFORE dispatching an agent — paste the output into its prompt
python harness/experience_db.py brief --target 17

# AFTER every attempt, including reward 0 and truncated episodes
python harness/experience_db.py attempt '{"target":17,"direction":"<structural frame>","reward":0.0}'

# pick what to work on next
python harness/experience_db.py sched --top 20
```

The failure mode it exists to prevent: after ~10 waves the original ledger held **4** records
carrying a realized reward. Outcomes were recorded; the reward-per-unit-effort signal that
scheduling actually needs was not. Log the signal, not just the conclusion.
