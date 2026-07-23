"""
Offline end-to-end validation of the single-post attack, using a COMPLIANT-LLM stand-in.

The deterministic SDK agent ignores url/data in the user message, so it can't exercise
the single-post attack. `CompliantPostAgent` simulates a fully-compliant target: it parses
`url=… data=…` (or `http.post(url="…", data="…")`) from the user message and emits exactly
that http.post. Running the real attack.py against it + OptimalGuardrail validates the
whole pipeline offline (calibration -> budget sizing -> verify-and-keep -> replay scoring).
It is an UPPER BOUND on the real models (which refuse some fraction); the point is to prove
the machinery is correct and the returned candidates actually fire on replay.

Run: python test_attack_offline.py --attack ../../../../multi-step-tool-attacks/attack/attack.py --budget-s 20
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import time
from pathlib import Path
from typing import Any

from aicomp_sdk.agents.types import FinalResponseDecision, ToolCall, ToolCallDecision
from aicomp_sdk.core.predicates import eval_predicates
from aicomp_sdk.core.cells import cell_signature
from aicomp_sdk.evaluation.ops import build_attack_env, resolve_fixtures_dir
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail
from aicomp_sdk.scoring import score_attack, score_attack_raw

FIX = resolve_fixtures_dir(None)
_URL = re.compile(r'url\s*=\s*"?([^\s",)]+)')
_DATA = re.compile(r'data\s*=\s*"?([^\s",)]+(?:[^\s",)]*)?)')


class CompliantPostAgent:
    """A maximally-compliant target: does the http.post the user message asks for, once."""

    def __init__(self):
        self._done = False

    def next_action(self, *, history: Any, tools: Any):
        # find the latest user message
        msg = ""
        for ev in reversed(list(getattr(history, "events", lambda: [])() if callable(getattr(history, "events", None)) else getattr(history, "events", []))):
            text = getattr(ev, "text", None)
            if text and type(ev).__name__ == "UserMessageEvent":
                msg = text
                break
        if self._done or not msg:
            return FinalResponseDecision(text="OK")
        mu, md = _URL.search(msg), _DATA.search(msg)
        if not mu:
            return FinalResponseDecision(text="OK")
        self._done = True
        return ToolCallDecision(call=ToolCall(call_id="c1", tool_name="http.post",
                                              arguments={"url": mu.group(1),
                                                         "data": md.group(1) if md else "SECRET_MARKER"}))

    def reset_state(self):
        self._done = False

    def snapshot_state(self):
        return {"version": 1, "backend": "compliant", "data": {"done": self._done}}

    def restore_state(self, s):
        self._done = bool(s.get("data", {}).get("done", False))


def load_attack(path: Path):
    spec = importlib.util.spec_from_file_location("subm_attack", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m.AttackAlgorithm


def make_env(hops: int):
    return build_attack_env(seed=123, fixtures_dir=FIX,
                            agent_factory=CompliantPostAgent, guardrail=OptimalGuardrail(),
                            env_selection="gym", max_tool_hops=hops)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", required=True, type=Path)
    ap.add_argument("--budget-s", type=float, default=20.0)
    ap.add_argument("--hops", type=int, default=8)
    args = ap.parse_args()

    from aicomp_sdk.attacks.contracts import AttackRunConfig
    AttackAlgorithm = load_attack(args.attack.resolve())

    t0 = time.time()
    cands = AttackAlgorithm({}).run(make_env(args.hops),
                                    AttackRunConfig(time_budget_s=args.budget_s, max_tool_hops=args.hops))
    gen_s = time.time() - t0
    print(f"generation: {len(cands)} candidates in {gen_s:.1f}s (compliant stand-in)")

    # replay every candidate exactly like the gateway, score.
    findings = []
    for c in cands:
        env = make_env(args.hops)
        env.reset()
        for msg in c.user_messages:
            env.interact(msg, max_tool_hops=args.hops)
        tr = env.export_trace_dict()
        preds = eval_predicates(tr)
        if preds:
            findings.append({"predicates": preds,
                             "score_cell_signature": cell_signature(tr.get("tool_events", []))})
    uniq = len({f["score_cell_signature"]["hash"] for f in findings})
    print(f"fired on replay: {len(findings)}/{len(cands)}  | unique cells: {uniq}")
    print(f"RAW = {score_attack_raw(findings):.1f}   NORMALIZED = {score_attack(findings):.2f} / 1000")
    print(f"per-candidate replay latency ~ {gen_s/max(1,len(cands)):.3f}s  "
          f"(=> ~{int(0.85*9000/max(0.01,gen_s/max(1,len(cands))))} candidates fit a real 9000s budget)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
