# NeuroGolf 2026 — Competition Brief (VERIFIED)

> Status: **verified against the authoritative `neurogolf_utils.py`** (downloaded from the
> competition via the Kaggle API on 2026-06-27) plus a sample task file and the rules page.
> Every load-bearing fact below was checked against source code or executed empirically
> (see `VERIFICATION.md`). Lines marked ⚠️ are inferred, not source-proven.

## TL;DR
"Neural-network golf": for each of **400** ARC-AGI tasks, hand-build the **smallest** ONNX
network that exactly reproduces the transformation. Smaller = lower `cost` = more points.

## Objective & dataset ✅
- **400 tasks**: `task001.json … task400.json` (the full ARC-AGI-1 training set).
  (NOTE: the `kaggle competitions files` API listing silently capped at 199; the actual
  dataset zip contains all 400 — verified by downloading + counting, no gaps.)
- Each task JSON has three splits: `train`, `test`, and **`arc-gen`** (procedurally generated
  variants — e.g. task001 ships 5 train / 1 test / **262 arc-gen** examples).
- For each task you submit one ONNX network: `task001.onnx … task400.onnx`.

## Scoring ✅ (executed end-to-end)
```
points_per_task = max(1.0, 25.0 - ln( max(1.0, memory + params) ))
cost            = memory + params          #  <-- NO MACs term
```
- **`params`** = total element count across all `initializer` / `sparse_initializer` tensors
  and `Constant`-node values (scalars cost 1).
- **`memory`** = sum over *intermediate* tensors of `static_num_elements × dtype_itemsize`
  (FLOAT = 4 bytes), taken as the **max across runs** via the ONNX Runtime profiler.
  The fixed `input` and `output` tensors are **excluded** from memory.
- **Zero-cost networks score the full 25** (the inner `max(1.0, …)` makes `ln(1)=0`).
- Empirical anchors (real utils): identity **1×1 conv → 100 params, 0 memory → 20.395 pts**;
  **3×3 conv → 900 params → 18.198 pts**. A parameter-free op graph (e.g. a pure `Transpose`)
  → cost 0 → **25 pts**.
- Total leaderboard score = sum over the (up to 400) tasks you solve. Ceiling ≈ 400×25 =
  10000; leaders sit ~7900 (i.e. nearly all 400 solved at avg ~20 pts).

## I/O contract ✅ (this is fixed — important)
- Input tensor **`"input"`**: `FLOAT[1, 10, 30, 30]` — a **one-hot** encoding, 10 color
  channels × 30×30. Grids are placed top-left and **zero-padded** to 30×30.
- Output tensor **`"output"`**: `FLOAT[1, 10, 30, 30]`, same one-hot layout.
- Decision rule: the grader takes **`(raw_output > 0.0)`** and compares it for **exact
  equality** (`np.array_equal`) to the one-hot target. So each active cell must have its
  correct channel `> 0` and all other channels `≤ 0`; padding cells must be `≤ 0` everywhere.
- Tests with grids larger than 30×30 are ignored. Internally you may use any shapes/channels;
  only the boundary `input`/`output` tensors are pinned (and they don't count toward memory).

## Correctness gate ✅ (hard)
A network earns points only if it is **strictly correct on every `train` + `test` + `arc-gen`
example** for that task (the local checker requires `wrong == 0` across all splits before it
prints "READY for submission"). ⚠️ The official leaderboard almost certainly re-grades against
a larger held-out ARC-GEN corpus (and possibly a private set) — so the network must implement
the *true generalizing rule*, not memorize the shipped examples.

## Constraints on the ONNX graph ✅
- **Opset / IR = 10**; operator domain must be `""` or `"ai.onnx"` (no custom domains).
- **Static shapes only**: every dim must be a positive `dim_value` (no `dim_param`, no ≤0).
- **Exactly one input and one output**; `input`/`output` names reserved; no name collisions
  between initializers and I/O tensors.
- **Banned op types**: `Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function`, **`Compress`**
  (7 total — verified in `_EXCLUDED_OP_TYPES`), plus any **`Sequence*`** op, plus subgraph
  attributes (`Graph`/`Graphs`), functions, and sequence-typed tensors.
- **File size ≤ 1.44 MB** (`1,509,949` bytes) per `.onnx`.

## Timeline ✅ (today = 2026-06-27)
- Start: 2026-04-15 · **Entry deadline = Team-merger deadline: 2026-07-08** ·
  **Final submission: 2026-07-15** (all 23:59 UTC).

## Prizes & venue ✅/⚠️
- **$50,000** total pool ✅ (Kaggle API: reward "50,000 Usd", 2532 teams). ⚠️ Additional
  "top student team" and "longest leader" awards reported by the overview/press, not re-proven
  from code. Part of the **IJCAI-ECAI 2026** Competitions Track (Bremen, Germany).

## Canonical minimal solver (provided by the organizers)
`neurogolf_utils.single_layer_conv2d_network(weight_fn, kernel_size)` builds a **single `Conv`**
with weight shape `[10, 10, k, k]` mapping the one-hot input straight to the one-hot output.
This is the intended "one tiny layer = one program" template: recolor via `k=1` (100 params),
local spatial rules via larger kernels.

## Sources
- Authoritative: `neurogolf_utils.py` (competition file, Apache-2.0, Google LLC) — scoring
  ground truth; fetch with `kaggle competitions download -f neurogolf_utils/neurogolf_utils.py neurogolf-2026`.
- Kaggle competition: https://www.kaggle.com/competitions/neurogolf-2026 (overview / rules / data)
- Kaggle API: `kaggle competitions list -s neurogolf` → deadline 2026-07-15, 50,000 USD, 2532 teams
- IJCAI-ECAI 2026 competitions — https://2026.ijcai.org/competitions/
