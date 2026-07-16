# Verification log — earlier conclusions vs. authoritative source

On 2026-06-27 the Kaggle API became usable, so the hand-research conclusions were checked
against the **authoritative `neurogolf_utils.py`** (competition file), a sample task JSON, the
rules page, and an **empirical run of the real scoring path** (`verify_scoring.py`, executed
with onnx/onnxruntime/onnx-tool installed).

## Scorecard
| # | Earlier claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Score = `max(1, 25 − ln(cost))` | ✅ **Correct** | utils L514; ran it: 1×1 conv → `25−ln(100)=20.395` |
| 2 | `cost = params + memory + **MACs**` | ❌ **Wrong** | cost = `params + memory` only. MACs removed (utils changelog 2026-05-04). Compute is free. |
| 3 | Strict correctness on ARC-AGI + ARC-GEN (+ private) | �» **Mostly** | Local gate = `train+test+arc-gen` must be 0-wrong (L499–509). Private/held-out set is ⚠️ inferred, not in utils. |
| 4 | "~400 tasks" | ✅ **Right after all** | **400 tasks** (`task001…task400.json`, no gaps) in the downloaded zip. The `kaggle competitions files` API listing silently capped at 199, which caused a wrong intermediate "correction" to 199 — the full data restored the original ~400. |
| 5 | ARC-AGI-1 training subset | ✅ | 400 = the full ARC-AGI-1 training set; each task has train/test/arc-gen. |
| 6 | submission.zip of `taskNNN.onnx` | ✅ | filename `task{:03d}.onnx` (L484); zip per instructions. |
| 7 | Per-file ≤ 1.44 MB | ✅ | `_FILESIZE_LIMIT_IN_BYTES = 1.44*1024*1024 = 1,509,949` (L109). |
| 8 | Static shapes required | ✅ | strictly enforced in `calculate_memory` (rejects `dim_param`/≤0). |
| 9 | Banned ops = Loop/Scan/NonZero/Unique/Script/Function | �» **Incomplete** | Also **`Compress`** (in `_EXCLUDED_OP_TYPES`), plus `Sequence*` ops, subgraphs, functions, custom domains. |
| 10 | Entry 7/8, Final 7/15, $50k | ✅ | rules page + Kaggle API (`deadline 2026-07-15`, `50,000 Usd`, 2532 teams). |

## New facts the research missed (now incorporated)
- **Fixed one-hot I/O**: `input`/`output` are `FLOAT[1,10,30,30]`; grids placed top-left,
  zero-padded; grader thresholds `(output > 0)` and compares exactly. (Earlier "[1,1,H,W]
  integer" I/O idea was wrong.)
- **`input`/`output` tensors are excluded from the memory cost** — only intermediate
  activations + params count. A single Conv input→output has **0 memory**.
- **Opset/IR = 10**; domain must be `""`/`"ai.onnx"`; exactly one input + one output.
- **Organizer-provided template**: `single_layer_conv2d_network(weight_fn, k)` → single `Conv`
  with weight `[10,10,k,k]`. The intended "one tiny layer = one program" pattern.
- **Strategic flip from removing MACs**: golf target is params + intermediate memory; unrolled
  CA solvers are free in compute but pay ~36 KB per `[1,10,30,30]` intermediate tensor.

## Empirical run (real `neurogolf_utils`)
```
GRID_SHAPE : [1, 10, 30, 30]
EXCLUDED   : LOOP SCAN NONZERO UNIQUE SCRIPT FUNCTION COMPRESS
FILESIZE   : 1,509,949 bytes (=1.44*1024*1024)   OPSET/IR: 10
identity 1×1 conv → params=100  memory=0  cost=100  → 20.3948 pts   (ARC-AGI 4/0, ARC-GEN 1/0)
identity 3×3 conv → params=900  memory=0  cost=900  → 18.1976 pts
NonZero graph     → score_network = (None, None)  [rejected]
```

## Reproduce
```bash
kaggle competitions download -f neurogolf_utils/neurogolf_utils.py neurogolf-2026 -p ng_data
kaggle competitions download -f task001.json neurogolf-2026 -p ng_data
pip install onnx onnxruntime onnx-tool ipython matplotlib numpy
python ../harness/verify_scoring.py     # prints the table above
```
