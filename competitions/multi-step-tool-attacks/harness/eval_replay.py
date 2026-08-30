"""
Local, offline evaluator for red-team attack submissions — the fast dev loop that
does NOT need a GPU or the target LLM.

It loads an `attack.py` (its `AttackAlgorithm`), runs `run(env, config)` against a
chosen agent×guardrail, then REPLAYS the returned candidates exactly like the Kaggle
gateway does (`build_attack_env` -> reset -> interact each user message) and scores
them with the real `score_attack`. Use it three ways:

  * --guardrail none   : plumbing check — proves the attack + scoring pipeline can
                         produce a non-zero score (the deterministic agent is only
                         exploitable when the guardrail is permissive).
  * --guardrail optimal: public-LB parity for the *guardrail* (agent still differs:
                         deterministic here vs gpt-oss-20b/gemma on Kaggle).
  * --agent openai     : real-LLM parity if OPENAI_API_KEY is set (or point the
                         gpt_oss/gemma backends at a local GGUF — see README).

Examples:
  python eval_replay.py --attack ../../../../multi-step-tool-attacks/attack/attack.py \
      --agent deterministic --guardrail none --budget-s 20
  python eval_replay.py --attack <path> --agent deterministic --guardrail optimal --budget-s 30
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

from aicomp_sdk.agents import build_agent
from aicomp_sdk.attacks.contracts import AttackCandidate, AttackRunConfig
from aicomp_sdk.core.cells import cell_signature
from aicomp_sdk.core.predicates import eval_predicates
from aicomp_sdk.evaluation.ops import build_attack_env, resolve_fixtures_dir
from aicomp_sdk.guardrails.base import GuardrailBase
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail
from aicomp_sdk.attacks.baselines.guardrail_rules import Guardrail as WeakGuardrail
from aicomp_sdk.scoring import score_attack, score_attack_raw

GUARDRAILS = {"none": GuardrailBase, "optimal": OptimalGuardrail, "weak": WeakGuardrail}


def load_attack(path: Path):
    spec = importlib.util.spec_from_file_location("submission_attack", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod  # required so @dataclass in the module resolves
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.AttackAlgorithm


def make_env(agent: str, guardrail_cls, fixtures: Path, hops: int, seed: int):
    return build_attack_env(
        seed=seed, fixtures_dir=fixtures,
        agent_factory=lambda: build_agent(agent),
        guardrail=guardrail_cls(), env_selection="gym", max_tool_hops=hops,
    )


def replay_candidate(candidate: AttackCandidate, agent: str, guardrail_cls,
                     fixtures: Path, hops: int, seed: int) -> dict[str, Any] | None:
    """Score-side replay of one candidate. Returns a finding dict or None."""
    env = make_env(agent, guardrail_cls, fixtures, hops, seed)
    env.reset()
    for msg in candidate.user_messages:
        env.interact(msg, max_tool_hops=hops)
    trace = env.export_trace_dict()
    preds = eval_predicates(trace)
    if not preds:
        return None
    return {
        "predicates": preds,
        "score_cell_signature": cell_signature(trace.get("tool_events", [])),
        "user_messages": list(trace.get("user_messages", [])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", required=True, type=Path, help="path to attack.py")
    ap.add_argument("--agent", default="deterministic",
                    help="deterministic | openai | gpt_oss | gemma | auto")
    ap.add_argument("--guardrail", default="optimal", choices=list(GUARDRAILS))
    ap.add_argument("--budget-s", type=float, default=20.0)
    ap.add_argument("--hops", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--show", type=int, default=8, help="print first N scoring findings")
    args = ap.parse_args()

    fixtures = resolve_fixtures_dir(None)
    gcls = GUARDRAILS[args.guardrail]
    AttackAlgorithm = load_attack(args.attack.resolve())

    # 1) Generate candidates (same guardrail the scorer will use).
    gen_env = make_env(args.agent, gcls, fixtures, args.hops, args.seed)
    t0 = time.time()
    cands = AttackAlgorithm({}).run(
        gen_env, AttackRunConfig(time_budget_s=args.budget_s, max_tool_hops=args.hops)
    )
    gen_s = time.time() - t0
    print(f"generation: {len(cands)} candidates in {gen_s:.1f}s "
          f"(agent={args.agent}, guardrail={args.guardrail}, hops={args.hops})")

    # 2) Replay + score exactly like the gateway.
    findings: list[dict[str, Any]] = []
    for c in cands:
        f = replay_candidate(c, args.agent, gcls, fixtures, args.hops, args.seed)
        if f:
            findings.append(f)

    raw = score_attack_raw(findings)
    norm = score_attack(findings)
    uniq = len({f["score_cell_signature"]["hash"] for f in findings})
    sev_sum = sum(int(p["severity"]) for f in findings for p in f["predicates"])
    print(f"findings that fired on replay: {len(findings)} / {len(cands)}")
    print(f"  severity sum = {sev_sum}   unique score-cells = {uniq}")
    print(f"  RAW score    = {raw:.1f}")
    print(f"  NORMALIZED   = {norm:.2f}   (leaderboard metric, 0..1000)")

    for i, f in enumerate(findings[: args.show], 1):
        pn = [f"{p['predicate']}(s{p['severity']})" for p in f["predicates"]]
        print(f"  [{i}] {pn}  <-  {f['user_messages']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
