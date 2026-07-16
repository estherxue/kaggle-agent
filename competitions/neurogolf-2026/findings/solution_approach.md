# NeuroGolf 2026 — Solution Approach (完成思路, VERIFIED)

Program synthesis disguised as network design: each ONNX file is a tiny, fixed, feed-forward
"program" over a one-hot grid tensor. Win = solve as many of the **400** tasks as possible
*correctly* (the gate), then golf each one's `cost = params + intermediate_memory`.

## The golf objective — what actually counts (corrected)
`points = max(1, 25 − ln(params + memory))`. Two consequences that drive every design choice:

1. **Compute is FREE.** MACs/FLOPs were removed from the objective (utils v2026-05-04). You may
   use as many multiply-adds / unrolled steps as you like — they cost nothing.
2. **What you pay for is `params` + `intermediate-tensor memory`.** Each *intermediate* tensor
   adds `num_elements × dtype_bytes` (a full `[1,10,30,30]` float map = **36,000 bytes**); the
   pinned `input`/`output` tensors are free. So the levers are:
   - **Fewer & smaller intermediate tensors** → prefer 1–2 node graphs; crop/reduce early;
     use fewer channels or smaller H×W internally (only the boundary is pinned at 10×30×30).
   - **Fewer params** → parameter-free ops where possible; tiny/sparse/quantized initializers.

### Cost ladder (verified anchors)
| Construction                                  | params | memory | points |
|-----------------------------------------------|-------:|-------:|-------:|
| Parameter-free op graph (e.g. pure `Transpose`)|     0  |    0   | **25.0** |
| Single `1×1 Conv` recolor (10→10)             |   100  |    0   | 20.39  |
| Single `3×3 Conv`                             |   900  |    0   | 18.20  |
| +1 extra `[1,10,30,30]` intermediate (f32)    |   +0   | +36000 | −~10.5 |

→ **Parameter-free geometric tasks are the max-value targets (≈25 pts).** Opset-10 puts shape
args of `Transpose`/`Pad`/etc. in *attributes* (not initializers), so reflect/rotate/transpose/
translate can hit cost 0. Recolor needs a 10×10 conv (~20 pts) unless expressible param-free.

## I/O reality (fixed — design to it)
- `input`,`output` are both `FLOAT[1,10,30,30]` **one-hot** (channel = color 0–9), grid placed
  top-left, zero-padded. Grader compares `(output > 0)` for exact equality to the one-hot target.
- So: at each real cell, drive the correct channel `>0` and all others `≤0`; in the padding
  region keep all channels `≤0`. (Earlier "[1,1,H,W] integer" idea was wrong — the boundary is
  one-hot 10-channel.)

## Per-family ONNX realizations (all opset-10, static)
| Task family                    | Cheapest realization                          | ~cost |
|--------------------------------|-----------------------------------------------|------:|
| Reflect / rotate / transpose   | `Transpose` (perm attr)                       | 0 params |
| Translate / shift              | `Pad` + `Slice` (attrs in opset≤10)           | 0 params |
| Crop / extract region          | `Slice` (static)                              | small |
| Tile / repeat                  | `Tile` / `Concat`                             | small |
| Recolor / color permutation    | `1×1 Conv` `[10,10,1,1]` (or `Gather` table)  | ≤100 params |
| Local neighborhood / CA rule   | fixed `k×k Conv` + `Relu`/`Clip` threshold    | 100·k² params |
| Symmetry completion / fill     | `Max`/`Add` of reflected copies               | 0–small |
| Counting / majority            | `ReduceSum` + compare (tiny linear)           | small |
- **Cellular-automaton / propagation** (flood fill, gravity, growth): `Loop` is banned, so
  **unroll a fixed number of conv passes** (grid ≤30 ⇒ ≤~30 steps cover any distance). Compute
  is free, but **each pass's output tensor costs ~36 KB of memory** — so reuse the same small
  kernel across passes (params shared) and keep intermediate channel counts minimal. Memory, not
  compute, is the budget for unrolled solvers.

## Pipeline (what to build)
1. **Triage the 400 tasks** by transformation family; auto-detect the free wins first
   (pure geometry / pure recolor / constant-shape).
2. **Per-family ONNX builders** (`build_transpose`, `build_translate(dx,dy)`, `build_recolor(map)`,
   `build_ca(kernel, steps)`), each emitting a minimal static opset-10 graph.
3. **Rule inference** from the train pairs → instantiate the right template (the color map, axis,
   shift, kernel). Classic ARC DSL search, but the output is an ONNX template, not Python.
4. **Local validation oracle** = the highest-value infra: run each candidate via `onnxruntime`
   against `train`+`test`+**`arc-gen`** and a self-generated stress set (fuzz grid size, colors,
   positions within the family). Ship only nets that pass *generalization*, mirroring the hidden
   grader. Reuse `neurogolf_utils.verify_*` / `score_network` directly so local points == LB points.
5. **Golf** passing nets: drop intermediate tensors, fuse nodes, shrink internal shapes/channels,
   quantize/sparsify initializers, move shape args into attributes. Keep < 1.44 MB, no banned ops.
6. **Package** `submission.zip` of `taskNNN.onnx` for every solved task.

## Constraint linter (build it in)
- [ ] opset/IR = 10, domain ∈ {"", "ai.onnx"} · single input + single output
- [ ] static positive dims everywhere · no `input`/`output` name collisions
- [ ] no `Loop/Scan/NonZero/Unique/Script/Function/Compress`, no `Sequence*`, no subgraphs
- [ ] file ≤ 1,509,949 bytes
- [ ] passes train+test+arc-gen with `wrong == 0` AND a self-generated generalization stress set

## Strategy
- **Coverage ≫ micro-golf.** Solving one more task ≈ +18–25 pts; shaving a solved task from 100→10
  params ≈ +2.3 pts. Maximize *tasks solved* first; golf later.
- **Free 25s first.** Sweep for tasks solvable by a single parameter-free op (transpose/reflect/
  translate/crop) — each is a clean 25 with near-zero effort.
- **The local arc-gen oracle is the real judge.** Build the fuzzer early.
- ~11 days to entry (7/8), ~18 to final (7/15): bank the geometric/recolor families now.

## How this was produced / verified
`../harness/` runs N parallel research subagents → one synthesizer (the flow the user asked to
codify). The competition facts here were then **verified against the authoritative
`neurogolf_utils.py`** and by executing its real scoring path — see `VERIFICATION.md`.
