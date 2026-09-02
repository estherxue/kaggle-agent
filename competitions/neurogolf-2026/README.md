# NeuroGolf 2026 — what this competition contributed

Build the smallest ONNX network that *exactly* solves each of 400 ARC-AGI tasks.
Final honest score: **7625.77** (no exploits, no overfitting).

What survives here is small on purpose: the campaign's working tooling lives in the
competition repo (`neurogolf-26/neurogolf_solver/agent_kit/`), and the generic parts were
distilled into `../../harness/`. What remains is the piece that made the difference at the
start, plus the verified facts it produced.

## `harness/verify_scoring.py`

Imports the **real** `neurogolf_utils.py` and executes its scoring path, so local points
equal leaderboard points.

Seventy-one lines, and the single highest-leverage artifact in this repo: it **falsified the
cost formula the campaign was about to optimize against**. The research brief asserted
`cost = params + memory + MACs`; running the authoritative scorer proved MACs had been
removed (utils changelog 2026-05-04) — compute is free.

That correction re-pointed everything. If you pay for *intermediate tensors* rather than
compute, the target becomes "never materialize an intermediate", which is where essentially
every later gain came from (one task 16.23 → 22.70; another 20.62 → 25.00 via a single
50-operand Einsum).

```bash
kaggle competitions download -f neurogolf_utils/neurogolf_utils.py neurogolf-2026 -p ng_data
kaggle competitions download -f task001.json neurogolf-2026 -p ng_data
pip install onnx onnxruntime onnx-tool ipython matplotlib numpy
python harness/verify_scoring.py
```

## `findings/`

- **`VERIFICATION.md`** — claim-by-claim scorecard of the research against the authoritative
  source. Besides the MACs correction it caught: I/O is one-hot `FLOAT[1,10,30,30]` (not
  integer); graph `input`/`output` are excluded from the memory cost, so a single-node graph
  costs 0 memory; `Compress` is banned too; and there are **400** tasks, not 199 — the
  file-listing API silently truncated at 199 and produced a wrong "correction".
- **`competition_brief.md`** — the verified rules, scoring and constraints.
- **`solution_approach.md`** — the corrected strategy that followed from the above.

## Removed: the research orchestrator

`harness/research_harness.py` (165L), `harness/missions/neurogolf.json` and
`requirements.txt` were deleted 2026-09-02. The orchestrator was a parallel-research →
synthesis fan-out, but:

- it was **never actually run for this competition** — `findings/` contains no
  `research_<id>.md` or `synthesis.md`, and this README previously conceded the pages "were
  researched by hand this run";
- its product was **wrong exactly where it mattered**: the mission asserted MACs were part
  of the cost, which `verify_scoring.py` then had to falsify.

Kept in git history if the pattern is ever wanted. The transferable lesson is recorded in
`../../HARNESS.md`: *the harness that worked was written in response to a failure that had
already happened; the one designed up front, from a spec, went unused.*
