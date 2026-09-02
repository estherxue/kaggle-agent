# NeuroGolf 2026 — harness record

Competition-scoped companion to the cross-competition `/HARNESS.md` at the repo root.
That file holds the layered framework and the lessons that generalize; this one holds what
was true *of this competition*: the tooling that carried it, the tooling that did not, and
the facts that had to be learned the expensive way.

**Result: 247 → 7625.77**, under hard no-overfit *and* no-exploit constraints.

| bundle | LB | what changed |
|---|---:|---|
| blend36 | 7372.66 | the honest line as of the deadline |
| blend38 | 7464.30 | +199 gated models from a 7477.38 public pack |
| blend39 | 7467.11 | +8 artifacts local ORT could not execute, escalated to the authority |
| blend40 | 7647.97 | +202 gated models from a 7672.16 pack, minus 5 negative-stride exploits |
| **blend41** | **7625.77** | **−22 models hiding parameters in operator attributes** (a deliberate drop) |

The last row is the point: the final move was *downward*, by choice.

---

## What this repo contributed

**`harness/verify_scoring.py` (71 lines)** — the highest-leverage artifact here by a wide
margin. It imports the real `neurogolf_utils.py` and runs its scoring path, and it
**falsified the cost formula the campaign was about to optimize against**: the research brief
asserted `cost = params + memory + MACs`; the authority showed MACs had been removed
(changelog 2026-05-04). Compute is free.

Everything downstream follows from that one correction. If you pay for *intermediate tensors*
rather than compute, the objective becomes "never materialize an intermediate" — which is
where essentially every later gain came from.

**`findings/VERIFICATION.md`** — the claim-by-claim scorecard. Also caught: I/O is one-hot
`FLOAT[1,10,30,30]` (not integer); graph `input`/`output` are excluded from memory, so a
single-node graph costs 0 memory; `Compress` is banned too; and there are **400** tasks, not
199 — the file-listing API had silently truncated at 199 and produced a confident wrong
"correction".

## What did not contribute

**`harness/research_harness.py` + `missions/neurogolf.json` — deleted 2026-09-02.** Written
for this competition and never run for it (`findings/` holds no `research_<id>.md` or
`synthesis.md`), and its product was wrong exactly where it mattered — the mission is where
the MACs claim came from. In git history if the fan-out pattern is ever wanted.

## What actually carried the campaign — and it is not here

The working tooling lives in the competition repo, `neurogolf-26/neurogolf_solver/agent_kit/`:

| | lines | role |
|---|---:|---|
| `METHODS.md` | 236 / **42 rules** | the domain rulebook — the real carrier |
| `_tools/nghar.py` | 381 | cost probe, exactness, fresh-data gate, merge |
| `_tools/ngkb.py` | 378 | experience ledger (456 records); generalized here as `/harness/experience_db.py` |
| `_tools/gate.py` | 116 | fresh-ARC-GEN gate, time-bounded, reports the achieved `n` |
| `_tools/preflight.py` / `blockers.py` | 210 | official-scorer passes over a whole bundle |
| `_tools/scan_rmz.py` | 108 | pool scan that records UNKNOWNs instead of hiding them |
| `_tools/pin1.py` | 17 | single-thread runtime pinning (load average hit 257 without it) |
| `_tools/ngbuild/ngpatterns/ngtemplates.py` | 940 | ONNX construction primitives |

Not ported: porting these would be copying ONNX/ARC internals under a generic name.

---

## Scorer mechanics that cost real score to learn

- `cost = params + memory`; memory is per-named-tensor `max(static_shape, runtime_profile)`.
  **Graph `input`/`output` and MACs are free** ⇒ a single-node graph costs *params alone*.
- The grader runs ORT 1.24.4 with `ORT_DISABLE_ALL` — no folding, no fusion.
- Output dtype is free (the grader thresholds `> 0`).
- **One unscorable member voids the entire zip** — which also means a *nonzero* score proves
  every member was scorable. That inference is the whole basis of the differential preflight.
- The local toolchain is not the grader: local `onnx` 1.22 rejects negative Conv pads the
  grader accepts, and local ORT 1.23.2 lacks kernels it has. **ORT 1.24.4 is not on PyPI for
  this platform**, so the proxy cannot be version-pinned — escalate to the leaderboard instead.

## The two exploits found, and declined

Both sat in the *same* pack, whose provenance was airtight (immutable Kaggle result JSON;
the shipped `.zip.bin` hashed to the sha256 its own manifest claimed). **Provenance quality
does not predict method quality.**

| class | what | how it presents | cost to decline |
|---|---|---|---:|
| undefined semantics | `MaxPool` with a **negative stride** (5/400) | fails `onnx.checker(full_check)` *and* strict shape inference, yet solves 266/266 | ~15.3 |
| accounting gap | 577,192 elements of parameters in `TfIdfVectorizer` attributes (22/400) | **passes** every spec check — `calculate_params()` enumerates initializers and *Constant-node* attributes only | 22.20 |

"Fails the local checker" is **not** the test — negative *pads* fail it too and are legitimate
(well-defined crop semantics, present in every pack, shipped cleanly). The test that works:
(1) does the spec define behaviour for the construct at all, and (2) outlier or idiom?
Negative strides: 5/400 in one pack, 0/400 in two others.

**Checking only the class you hypothesised finds only that class.** The first came from a
hypothesis; the second only from an audit lens aimed at the *scoring code*.

**Do not condemn a pack for containing an exploit**: of the 9 flagged full-score artifacts,
5 were exploits and one was a legitimate 50-operand single-node Einsum worth +4.38.

## Where the headroom was not

Four successive estimates of remaining headroom, each refuted by the next once actually
checked:

`425.83` → `44.85` → `36.70` → **`1.15`**  — a **370× over-promise**.

The root cause of the last collapse: a tensor's dtype is fixed by its *producer*, and nearly
every tensor here descends from the f32 graph input, so 91.2% of "narrowable" bytes cannot be
narrowed without inserting a cast that costs more than it saves. Treat any static headroom
number as a screen for what to measure, never as headroom.

## Pointers

- Methodology: `neurogolf-26/neurogolf_solver/agent_kit/METHODS.md` (42 rules)
- Generic lessons: `/.claude/skills/agent-field-lessons/SKILL.md` (L1–L11)
- Reusable tooling: `/harness/` (`experience_db.py`, the four-stage audit pipeline)
- Cross-competition synthesis: `/HARNESS.md`
