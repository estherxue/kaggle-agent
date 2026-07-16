# kaggle-agent · research harness

A small, reusable harness that codifies the flow we ran for NeuroGolf 2026:

```
            ┌──────────────┐
   mission  │  researcher  │  web_search ─┐
  (topic +  ├──────────────┤              │
  N sub-Qs) │  researcher  │  web_search ─┼──►  ┌──────────────┐
       ──►  ├──────────────┤              │     │ synthesizer  │ ──► synthesis.md
            │  researcher  │  web_search ─┘     └──────────────┘
            └──────────────┘
            N agents in PARALLEL            one agent CONSOLIDATES
```

- **Parallel fan-out:** each researcher owns one sub-question and gathers cited facts via
  the Anthropic server-side `web_search` tool. They run concurrently (`asyncio.gather`).
- **Single synthesis:** one agent merges the reports, flags contradictions, marks
  load-bearing facts, and lists what still needs primary-source confirmation.

## Layout
```
kaggle-agent/
├── README.md                     # this file
├── requirements.txt              # anthropic SDK
├── harness/
│   ├── research_harness.py       # the orchestrator (parallel research -> synthesis)
│   └── missions/neurogolf.json   # the NeuroGolf research mission (edit to reuse)
└── findings/                     # outputs (committed: the brief + approach we produced)
    ├── competition_brief.md      # NeuroGolf rules/scoring/constraints (verified)
    └── solution_approach.md      # how to actually solve the competition
```

## Run
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
cd harness
python research_harness.py --mission missions/neurogolf.json --out ../findings
# -> ../findings/research_<id>.md per researcher + ../findings/synthesis.md
```

Reuse for any topic: copy `missions/neurogolf.json`, change `topic` + `questions`, rerun.
Models are overridable via `NG_RESEARCHER_MODEL` / `NG_SYNTHESIZER_MODEL`.

## What we found (NeuroGolf 2026) — verified
See [`findings/competition_brief.md`](findings/competition_brief.md) for the verified rules,
[`findings/solution_approach.md`](findings/solution_approach.md) for the strategy, and
[`findings/VERIFICATION.md`](findings/VERIFICATION.md) for the claim-by-claim check against the
authoritative `neurogolf_utils.py`.

Headline (all source-verified): per-task score `max(1, 25 - ln(cost))` with
**`cost = params + intermediate_memory`** (**no MACs** — compute is free). Fixed one-hot I/O
`FLOAT[1,10,30,30]`; submit one `taskNNN.onnx` for each of **400** tasks (`task001..task400`),
≤1.44 MB each, opset 10, static shapes only, no
`Loop/Scan/NonZero/Unique/Script/Function/Compress`/`Sequence*`. Must be 0-wrong on
`train+test+arc-gen`. **Coverage of tasks beats micro-golfing any one**; parameter-free
geometric ops (e.g. a pure `Transpose`) score the max 25.

## Two flows in this harness
- `harness/research_harness.py` — the **parallel-research → synthesis** orchestrator (the flow
  the user asked to codify). Auth-gated pages were researched by hand this run; the orchestrator
  is the productized version for future non-gated missions.
- `harness/verify_scoring.py` — a **ground-truth verifier** that imports the real
  `neurogolf_utils` and executes its scoring path, so local points == leaderboard points.
  Download `neurogolf_utils.py` + a task JSON into `harness/ng_data/` first (see VERIFICATION.md).
