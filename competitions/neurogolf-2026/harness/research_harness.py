#!/usr/bin/env python3
"""
Parallel research harness for the kaggle-agent.

Flow (the one we just ran by hand, now codified):
  1. Fan out N **researcher** subagents in parallel — each owns one sub-question and uses
     the Anthropic server-side `web_search` tool to gather + cite facts.
  2. One **synthesizer** subagent consolidates all researcher reports into a single
     cross-verified brief, flagging contradictions and low-confidence claims.

Outputs are written to an output dir: one markdown file per researcher plus a final
`synthesis.md`. Designed to be dropped into the kaggle-agent and reused for any topic
(not just NeuroGolf) by editing the mission file.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python research_harness.py --mission missions/neurogolf.json --out ../findings

Mission file (JSON):
    {
      "topic": "The 2026 NeuroGolf Championship (Kaggle / IJCAI-ECAI 2026)",
      "questions": [
        {"id": "mechanics", "q": "Exact scoring formula, correctness gate, ..."},
        {"id": "tiny_onnx", "q": "How to build smallest ONNX nets that solve ARC ..."},
        {"id": "prior_art", "q": "Public notebooks & baselines and what they do ..."}
      ]
    }

Dependencies: `pip install anthropic` (>= 0.40).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from anthropic import AsyncAnthropic
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency: pip install anthropic")

# --- Model tiers -------------------------------------------------------------
# Researchers do web-grounded fact gathering (Sonnet is the sweet spot); the
# synthesizer does the harder cross-verification + judgement (Opus).
RESEARCHER_MODEL = os.environ.get("NG_RESEARCHER_MODEL", "claude-sonnet-4-6")
SYNTHESIZER_MODEL = os.environ.get("NG_SYNTHESIZER_MODEL", "claude-opus-4-8")

WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 8}

RESEARCHER_SYSTEM = """You are a meticulous research subagent. You own exactly ONE \
sub-question. Use web_search aggressively to gather primary-source facts. Rules:
- Prefer official / primary sources; cross-check any number, formula, date, or constraint \
against at least two independent sources before stating it as fact.
- If sources disagree or a page is auth-gated/unreachable, SAY SO explicitly and mark the \
claim low-confidence — never paper over a gap.
- Output tight Markdown: a short factual report, then a '## Sources' list of every URL used.
- Your output IS data for a downstream synthesizer, not a chat reply. No preamble."""

SYNTHESIZER_SYSTEM = """You are the synthesis subagent. You receive several researcher \
reports (each already cited). Produce ONE consolidated Markdown brief that:
- Merges the findings into a clean, skimmable structure.
- Explicitly flags any contradictions between researchers and picks the better-supported \
claim (explain why).
- Marks load-bearing facts (formulas, hard constraints, deadlines) and notes which still \
need primary-source confirmation.
- Ends with a '## Open questions / to verify' section.
No preamble — output the brief directly."""


@dataclass
class Question:
    id: str
    q: str


async def run_researcher(client: AsyncAnthropic, topic: str, question: Question) -> str:
    """One parallel researcher: server-side web_search loop, returns cited Markdown."""
    prompt = (
        f"Research topic: {topic}\n\n"
        f"Your sub-question (focus ONLY on this):\n{question.q}\n\n"
        "Gather and cross-verify the facts, then write your cited report."
    )
    resp = await client.messages.create(
        model=RESEARCHER_MODEL,
        max_tokens=4096,
        system=RESEARCHER_SYSTEM,
        tools=[WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": prompt}],
    )
    return _text_of(resp)


async def run_synthesizer(client: AsyncAnthropic, topic: str, reports: dict[str, str]) -> str:
    """The single consolidation agent."""
    joined = "\n\n".join(f"### Researcher: {qid}\n{rep}" for qid, rep in reports.items())
    prompt = (
        f"Research topic: {topic}\n\n"
        f"Here are the {len(reports)} researcher reports:\n\n{joined}\n\n"
        "Now produce the consolidated brief."
    )
    resp = await client.messages.create(
        model=SYNTHESIZER_MODEL,
        max_tokens=8192,
        system=SYNTHESIZER_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return _text_of(resp)


def _text_of(resp) -> str:
    """Concatenate the text blocks of a Messages response (ignoring tool-use blocks)."""
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()


async def main_async(mission_path: Path, out_dir: Path) -> None:
    mission = json.loads(mission_path.read_text())
    topic: str = mission["topic"]
    questions = [Question(**q) for q in mission["questions"]]
    out_dir.mkdir(parents=True, exist_ok=True)

    client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY

    print(f"[harness] topic: {topic}")
    print(f"[harness] fanning out {len(questions)} parallel researchers "
          f"({RESEARCHER_MODEL}) ...")

    # 1) Parallel fan-out — all researchers run concurrently.
    results = await asyncio.gather(
        *(run_researcher(client, topic, q) for q in questions),
        return_exceptions=True,
    )

    reports: dict[str, str] = {}
    for q, res in zip(questions, results):
        if isinstance(res, Exception):
            print(f"[harness]   ! researcher '{q.id}' failed: {res}")
            reports[q.id] = f"_Researcher failed: {res}_"
        else:
            reports[q.id] = res
        (out_dir / f"research_{q.id}.md").write_text(reports[q.id])
        print(f"[harness]   - wrote research_{q.id}.md")

    # 2) Single synthesizer consolidates everything.
    print(f"[harness] synthesizing ({SYNTHESIZER_MODEL}) ...")
    synthesis = await run_synthesizer(client, topic, reports)
    (out_dir / "synthesis.md").write_text(synthesis)
    print(f"[harness] done -> {out_dir / 'synthesis.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel research -> synthesis harness")
    ap.add_argument("--mission", required=True, type=Path,
                    help="JSON mission file (topic + questions)")
    ap.add_argument("--out", default=Path("findings"), type=Path,
                    help="output directory for reports")
    args = ap.parse_args()
    asyncio.run(main_async(args.mission, args.out))


if __name__ == "__main__":
    main()
