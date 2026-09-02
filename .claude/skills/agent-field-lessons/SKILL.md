---
name: agent-field-lessons
description: >-
  Hard-won protocol lessons for orchestrating fleets of construction/optimization subagents against a
  measurable objective. Invoke when: a subagent fleet reports wins you have not independently
  re-measured; agents conclude a direction is "infeasible" based on a local/proxy environment; a
  rewrite-style fleet returns zero usable results; you are optimizing against a metric you can query
  repeatedly (public leaderboard, eval endpoint, judge model); a bundle/batch artifact can be zeroed
  by one bad member; or repeated "different" directions keep hitting the same blocker. Complements
  deli-auto-research (stall detection & structural pivots) with what breaks in practice at fleet scale.
type: Agent Framework
tags: multi-agent, orchestration, verification, overfitting, benchmarking, fleet, subagent, evaluation-harness
---

# Agent field lessons

Eleven protocol-level lessons from a long-horizon agent-fleet campaign (an ONNX code-golf
competition: 400 independent optimization targets, score 247 → 7625.77 under hard no-overfit AND
no-exploit constraints). None is domain-specific; each cost real score to learn. They sit downstream of
[deli-auto-research](../deli-auto-research/SKILL.md): that framework tells you *when* to pivot, these
tell you *what fails* once you actually run fleets against a metric.

---

## L1 — The measurement environment is itself a structural constraint

The most-missed structural constraint is **the harness you evaluate with**.

Three agent waves on one target produced *correct* solutions that were all rejected as "too
expensive" — by a **local runtime one minor version behind the authoritative one**. The agents'
infeasibility arguments all bottomed out in a tool error (`NOT_IMPLEMENTED` on an integer kernel),
not in a property of the problem, so we escalated the decisive measurement to the real scorer.

**The escalation falsified our own hypothesis, and that is the point.** Costs in the authoritative
environment came back *identical* to the proxy (9441 vs 9441, 4729 vs 4729): the wall was real, the
"one version behind" story was wrong. What the run bought was not a win but the **retirement of a
live hypothesis** — three waves' worth of "maybe the sandbox is lying" was closed for the price of one
kernel run. It also surfaced something no local check could: two *incumbent* artifacts that the
authoritative scorer **refuses to score at all** (`memory = None`) while every local tool passed them.

So the rule is not "the proxy always lies". It is: **a verdict that rests on a tool error is not a
verdict yet**, and the authoritative environment is the only thing that can convert it into one —
whichever way it lands.

**Rules**

- Before accepting "this direction is infeasible", ask: *has this been measured in the authoritative
  environment, or only in the proxy?*
- A local `NOT_IMPLEMENTED` / unsupported-op / OOM / timeout is **not** evidence about the candidate.
  It is evidence about your sandbox. Escalate to a real-environment check before it hardens into a
  conclusion.
- Version-pin the proxy to the authority. Where that is impossible, keep a **one-command path** to
  the authoritative environment. If truth is expensive to obtain, you will rationalise proxy results
  instead of getting it.

**Smell test**: your infeasibility argument names a tool, a version, or an error string rather than a
property of the problem.

---

## L2 — Subagents overclaim ~10×; verify against the live baseline, never their reported delta

Across three waves, self-reported gains exceeded independently-verified gains by roughly an order of
magnitude (+4.5 claimed → +0.42 real). One wave's headline "+5.01 win" was measured against a **stale
baseline file** and was actually a *loss*.

This is not dishonesty. Each fresh-context agent re-derives its own baseline, and baselines drift as
the main line advances — so every agent silently benchmarks against a different, usually convenient,
reference.

**Rules**

- **The orchestrator re-measures every candidate against the current live baseline.** Never merge on
  a subagent's number. (This is "separate execution from evaluation" applied to *metrics*, not just to
  progress judgements.)
- Pin the baseline **by path + content hash** and inject it into the prompt. An agent that picks its
  own reference picks a favourable one.
- Re-run the acceptance gate with a **different seed / different split** than the agent used.
  Same-seed re-runs validate nothing.

---

## L3 — Construction agents need a two-phase objective: find-correct, then shrink

A wave of 48 agents told to "build something better than the incumbent" returned **zero** usable
results. The same population, on comparable targets, told to run two explicit phases, returned real
wins.

1. **Phase 1 — find a correct new structure. Worse than the incumbent on the final metric is
   acceptable and expected here.**
2. **Phase 2 — only now optimize it** (drop carried state and recompute it, fuse steps, collapse to
   the minimal form).

Single-objective agents kill a promising architecture the moment it scores worse, and never reach the
optimization pass that would have won. Post-mortem against the winners confirmed the same split: their
*architectural rewrites* averaged ~10× the gain of *local polish*.

**Rules**

- Whenever the deliverable is a **rewrite**, state the two phases explicitly and say Phase-1
  regressions are expected and must not trigger abandonment.
- Ask for both artifacts: the Phase-1 correct-but-costly version *and* the Phase-2 shrunk version.
  The former is reusable evidence even when the latter fails.

---

## L4 — When the queried metric is a training set, discipline beats greed

The public metric was silently recomputed mid-campaign. A line built by **local surgery tuned to that
metric** dropped **−36** overnight. The line built only from candidates that passed a
**freshly-generated-data gate** (≥5000 unseen samples, zero errors) did not move at all.

**Rules**

- Any metric you can query repeatedly is a training set. Hold an **acceptance gate the metric cannot
  see** — freshly generated or held-out data — and promote only what passes it, *even when the queried
  metric says the cheap thing is better*.
- Prefer **rewrites** over **polish**: polish is what overfits the queried metric, and it is what
  evaporates on recomputation.
- Log the gate result next to every promoted artifact. When a recomputation lands, that log tells you
  immediately which line to trust.

---

## L5 — Bookkeeping that decides outcomes

- **One bad artifact can zero an entire batch.** Where the deliverable is a bundle, validate *every*
  member in the authoritative environment before shipping. Two whole submissions were lost to a single
  bad file before this became a mandatory step.
- **Group-test bisection for authority-only failures.** Split the changed set, resubmit halves, diff
  the scores. This isolated a member that scored fine alone but zeroed a *different* member when
  bundled — invisible to every local check.
- **`directions_tried` must record the structural frame, not the label.** We logged three "different"
  directions that shared one frame and one blocker; recording them as distinct hid the fact that we had
  not actually pivoted since the first attempt. Log the *constraint each direction accepts*, and a new
  direction must relax a constraint no previous one relaxed.

---

## L6 — A compatibility shim is part of the measuring instrument

Our harness auto-rewrote two operators the local runtime could not execute (a semantics-*altering*
rewrite) so that models would at least run. Months later a sweep reported that **five shipped
artifacts were wrong on the official examples** — one scoring 28/265 — implying ~81 points of phantom
credit. Two independent local re-verifications reproduced the exact failure counts, so we shipped a
"fix".

**The authoritative metric went *down* by exactly the difference.** The incumbent had always been
correct and scoring full credit; the failures were produced by our own shim. A second agent's
"independent replication" had simply re-implemented the same faulty rewrite.

**Rules**

- Exactness measured **through** a compatibility shim is not exactness. Tag every artifact whose
  evaluation path touched a shim, and treat its correctness as **unknown**, not as failed.
- Reproduction by a second agent using the same shim is **not** independent evidence. Independence
  means a different *path*, not a different author.
- Before "fixing" a suspected-broken artifact, price the downside: replacing a *working* artifact with
  a worse-but-verifiable one costs you the difference. Test the cheapest single swap first and let the
  authoritative metric adjudicate.
- Note the pattern: L1 and L6 are the **same instrument** fooling us twice, in opposite directions
  (once hiding a real wall, once inventing a fake defect). Instrument-caused verdicts are the single
  most expensive class of error in this work.

---

## L7 — Never read a validator failure in isolation; diff it against a known-good control

Running the authoritative checker over a batch is mandatory when one bad member can void the
whole batch. But its output is **unreadable on its own**, because a local toolchain that
trails the authoritative one produces failures that are artifacts, not defects.

Our local checker flagged 94 members of a candidate batch as fatal. Running the identical
check on a **control that had already scored cleanly in production** flagged 96 of the same
kind. And since one unscorable member voids the entire batch, a nonzero production score
*proves* every member was scorable — so all 94 were local artifacts.

Without that control we would have reverted 14 good artifacts.

**Rules**
- Keep a known-good control and run every batch-level check on it too. Only failure classes
  ABSENT from the control are candidate blockers.
- Look for an invariant that converts a production observation into a proof about members
  ("one bad member voids the batch" ⇒ "a nonzero score proves all members were valid").
- A validator failure names *your* toolchain until you have shown it names the artifact.

---

## L8 — Every static estimator over-promises; allocate by measured reward

We built four successive estimates of "how much headroom remains". Each was refuted by the
next once actually checked:

| estimator | claimed | what checking found |
|---|---:|---|
| total recoverable memory | 425.83 | probed the top 4 targets: 4/4 already optimal |
| narrowable-by-value slack | 44.85 | top arms were type-locked and could not be narrowed |
| + validity-aware | 36.70 | top arms still unrealizable |
| + producer/consumer constraints | **1.15** | measured |

A **370× over-promise** from the first estimate. The pattern was not one bad estimator; it was
that every estimator omits a constraint class you have not thought of yet.

**Rules**
- Treat any static headroom number as a **screen for what to measure**, never as headroom.
- Rank work by **posterior expected reward**: start from a heavily damped prior, let measured
  reward dominate as attempts accumulate, add a UCB exploration term. One zero-reward attempt
  should visibly demote a target (in ours, rank 2 → rank 72).
- Log the reward of **every** attempt, including 0 and truncated episodes. After ~10 waves our
  ledger held only 4 records with a realized reward — we had recorded conclusions, not signal.
- Prune **directions**, not targets. A failed method is evidence about the method; two targets
  we had logged as refuted later fell to a different frame entirely.

---

## L9 — Your "lower bound" is a bound on your current frame, not on the problem

Three times we wrote down a floor and were wrong, always the same way — the bound was a
property of the representation we happened to be using.

- "This target needs ≥2 large contractions, which exceeds budget" → the winning construction
  indexed its intermediate by *hypothesis* instead of by position, so no large contraction was
  needed at all.
- "80 bytes is optimal here; both intermediates are irreducible" → a construction that fed the
  input into a single contraction 24 times materialised nothing, reaching cost 1.
- "425 points of memory are recoverable" → see L8.

**Rule:** before recording a bound, ask *is this a property of the problem, or of the frame I
am currently in?* Record the frame alongside the bound, so the next agent knows what to vary.
A refuted bound is worth writing down precisely because it names the frame that failed.

---

## L10 — "Exploit" vs "merely exotic" needs a test, and exploits come in classes

When the objective carries a no-exploit constraint, you will meet constructs that are legal,
effective, and questionable. Intuition is not a criterion, and neither is "the validator
rejects it" — legitimate-but-exotic constructs get rejected by strict local validators too.

The test that worked, both parts required:
1. **Does the spec define behaviour for this construct at all?** (negative *pads* have
   well-defined crop semantics; a negative *stride* has none)
2. **Is it an isolated outlier or a widespread idiom?** (the exploit appeared in 5/400 members
   of one artifact set and 0/400 of two others; the legitimate construct was the norm in all)

**Exploits come in at least two classes, and checking only the class you thought of finds only
that class:**

| class | presents as |
|---|---|
| undefined semantics | fails spec validation; works only via undefined runtime behaviour |
| accounting gap | **passes** spec validation — the gap is in the *scorer*, not the format |

We found the first by hypothesis. The second — parameters parked in an operator attribute the
cost function never enumerates — was found only by a multi-lens audit that happened to include
a "cost-accounting evasion" lens. Both were in the same artifact set.

**Rules**
- Audit for exploits along *several independent lenses*, not just your hypothesis. Include one
  lens aimed squarely at the scoring/accounting code rather than at the artifact format.
- Do not condemn a whole source for containing an exploit: of 9 flagged artifacts from one
  pack, 5 were exploits and 1 was a legitimate construction worth adopting.
- Price the exclusion and say it out loud. Ours cost ~37.5 points, deliberately.

---

## L11 — Unexecutable is not defective (a refinement of L1)

L1 says a tool error is evidence about your sandbox. The operational corollary is that
"cannot run locally" and "ran and produced a wrong answer" must be handled **differently**:

- **No adverse evidence** (never executed) → escalate to the authoritative environment. We
  tested 8 such artifacts and 8/8 proved correct and cheaper (+2.81).
- **Adverse evidence** (ran, was wrong) → do NOT escalate on a hunch. There the upside was
  bounded by a small cost delta while the downside was a whole target's score. Negative
  expected value; we declined.

**Rule:** before escalating an unknown to an expensive authority, ask which of the two you
have. And when comparing a slow artifact against a fast one, **match the sample sizes** — a
slow candidate truncated at `0/27` is perfectly consistent with the `1/400` you are comparing
it against, and three of our verdicts nearly rested on that mismatch.

---

## Quick checklist

Before accepting a fleet's output:

- [ ] Re-measured every candidate myself, against a hash-pinned current baseline?
- [ ] Acceptance gate re-run with a fresh seed / unseen data?
- [ ] Any "infeasible" verdict traced to a real-environment check, not a proxy error?
- [ ] Rewrite fleets given an explicit two-phase objective?
- [ ] Every bundle member validated in the authoritative environment before shipping?
- [ ] Does the new direction relax a constraint that no previous direction relaxed?
- [ ] Did the evaluation path touch a compatibility shim? (If so: correctness is *unknown*, not failed.)
- [ ] Is the "independent replication" actually a different path, or the same shim twice?
- [ ] Diffed every batch-level validator failure against a KNOWN-GOOD control? (L7)
- [ ] Treating static headroom numbers as a screen, not as headroom? (L8)
- [ ] Logged the reward of every attempt — including 0 and truncated ones? (L8)
- [ ] Is that "lower bound" a property of the problem, or of my current frame? (L9)
- [ ] Audited for exploits along several lenses, including one aimed at the SCORER? (L10)
- [ ] For an unknown: no-adverse-evidence (escalate) or adverse-evidence (don't)? (L11)
- [ ] Sample sizes matched before believing a slow-vs-fast comparison? (L11)
