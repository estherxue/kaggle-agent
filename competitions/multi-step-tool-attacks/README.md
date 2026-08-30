# Multi-Step Tool Attacks — harness

Reusable, **non-solution** tooling for the Kaggle competition
`ai-agent-security-multi-step-tool-attacks` (red-team / attack-only). The actual attack
solution (`attack.py`, experiments, venv) lives OUTSIDE this repo, in
`/Users/xingyuanxue1122/Documents/coding/multi-step-tool-attacks`, because the
competition is active and the kaggle-agent remote is public (see repo `CLAUDE.md`).

- **`FINDINGS.md`** — how scoring, predicates, the guardrail, and the gateway work
  (reverse-engineered from `aicomp_sdk` 3.1.2). Read this first.
- **`PLAN.md`** — baseline status + roadmap to a guardrail-beating attack.
- **`harness/eval_replay.py`** — offline local evaluator: run an `attack.py` against any
  agent×guardrail, replay its candidates like the gateway, print the real score. No GPU.
- **`kaggle/build_notebook.py`** + **`kaggle/push_submission.sh`** — package the
  canonical `attack.py` into the 2-cell submission notebook and push it.

## Layout of the two directories

```
kaggle-agent/ (this repo, worktree)          multi-step-tool-attacks/ (separate repo)
└─ competitions/multi-step-tool-attacks/      ├─ .venv/            aicomp-sdk + CLI
   ├─ FINDINGS.md  PLAN.md  README.md         ├─ attack/attack.py  ← the solution
   ├─ harness/eval_replay.py                  ├─ experiments/
   └─ kaggle/{build_notebook.py,              └─ submissions/
              push_submission.sh,
              submission/kernel-metadata.json}
```

## Quick start

```bash
OP=/Users/xingyuanxue1122/Documents/coding/multi-step-tool-attacks
PY=$OP/.venv/bin/python

# 1) plumbing smoke test — proves the attack + scoring pipeline works (RAW > 0)
$PY harness/eval_replay.py --attack $OP/attack/attack.py --agent deterministic --guardrail none --budget-s 15

# 2) public-guardrail parity (deterministic agent scores ~0 — that's the real challenge)
$PY harness/eval_replay.py --attack $OP/attack/attack.py --agent deterministic --guardrail optimal --budget-s 15

# 3) build + push the Kaggle submission notebook
bash kaggle/push_submission.sh $OP/attack/attack.py
# then click "Submit to Competition" on the pushed notebook
```

## Local install (already done in the operation dir)

```bash
python3.11 -m venv .venv && .venv/bin/pip install aicomp-sdk   # provides `aicomp` CLI
```

`aicomp evaluate redteam attack.py --agent deterministic --env gym` is the SDK's own
scorer-style runner (writes `evaluation_artifacts/report.json`); `eval_replay.py` is our
thinner, guardrail-selectable equivalent for the dev loop.
