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

Five protocol-level lessons from a long-horizon agent-fleet campaign (an ONNX code-golf competition:
400 independent optimization targets, ~10M subagent tokens, score 7268 → 7372.65 under a hard
no-overfit constraint). None is domain-specific; each cost real score to learn. They sit downstream of
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
