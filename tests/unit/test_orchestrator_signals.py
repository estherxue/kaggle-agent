"""Tests for cross-process STOP and GUIDANCE signal handling.

These cover two regression fixes in the orchestrator:

1. STOP was a no-op: ``kagent stop`` writes a STOP_REQUESTED file via
   ``CompetitionInterface`` in a *separate* process, but the experiment loop
   only ever consulted an in-memory flag. The loop must read the same
   file-backed signal.
2. GUIDANCE was dropped: ``kagent guide`` writes ``guidance_queue.json`` via
   ``GuidanceQueue``, but ``_consume_guidance`` read a different in-memory
   store. Both producer and consumer must share one queue.

The orchestrator pulls in heavy LLM / knowledge / executor dependencies in
``__init__``, so the loop-level tests build a bare instance via
``object.__new__`` and attach only the attributes the methods under test
touch. This exercises the *real* ``Orchestrator._experiment_loop`` and
``Orchestrator._consume_guidance`` code paths.
"""

from types import SimpleNamespace

import pytest

from kaggle_agent.interaction import CompetitionInterface
from kaggle_agent.orchestrator import (
    CompetitionPhase,
    CompetitionState,
    Orchestrator,
)


# ---------------------------------------------------------------------------
# Cross-process contract: a signal written by one interface (the CLI process)
# is visible to a second interface at the same path (the orchestrator process).
# ---------------------------------------------------------------------------

def test_stop_signal_crosses_processes(tmp_path):
    """request_stop() on one interface is seen by another at the same path."""
    cli_side = CompetitionInterface("comp", tmp_path)
    agent_side = CompetitionInterface("comp", tmp_path)

    assert agent_side.get_stop_requested() is False
    cli_side.request_stop()
    assert agent_side.get_stop_requested() is True

    agent_side.clear_stop_request()
    assert agent_side.get_stop_requested() is False
    # And the fresh CLI-side view also reflects the clear.
    assert CompetitionInterface("comp", tmp_path).get_stop_requested() is False


def test_guidance_crosses_processes(tmp_path):
    """add_guidance() on one interface is consumed by another at same path."""
    cli_side = CompetitionInterface("comp", tmp_path)
    cli_side.add_guidance("try target encoding", source="user")

    agent_side = CompetitionInterface("comp", tmp_path)
    consumed = agent_side.guidance.consume()
    assert consumed is not None
    assert consumed.content == "try target encoding"

    agent_side.guidance.mark_processed(consumed.id, adopted=True)
    # Queue is now empty for any subsequent reader.
    assert CompetitionInterface("comp", tmp_path).guidance.consume() is None


# ---------------------------------------------------------------------------
# Helpers to build a minimal, dependency-free orchestrator for loop tests.
# ---------------------------------------------------------------------------

def _bare_orchestrator(tmp_path, max_experiments=5):
    """Build an Orchestrator with only the attrs the loop methods use."""
    orch = object.__new__(Orchestrator)
    orch.competition = "comp"
    orch.interface = CompetitionInterface("comp", tmp_path)
    orch.state = CompetitionState(competition="comp")
    orch._should_stop = False
    orch.config = SimpleNamespace(
        budget=SimpleNamespace(
            max_experiments_per_competition=max_experiments,
            max_llm_cost_usd=1000.0,
        )
    )
    # No-op persistence; the real loop calls _save_state each iteration.
    orch._save_state = lambda: None
    return orch


# ---------------------------------------------------------------------------
# Fix 1: STOP is honored by the experiment loop.
# ---------------------------------------------------------------------------

def test_experiment_loop_honors_stop_before_first_experiment(tmp_path):
    """A stop requested before the loop starts prevents any experiment."""
    orch = _bare_orchestrator(tmp_path)

    ran = []
    orch._run_experiment = lambda guidance=None: ran.append(guidance)

    # CLI process requests stop.
    CompetitionInterface("comp", tmp_path).request_stop()

    orch._experiment_loop()

    assert ran == []  # never entered the body
    assert orch.state.phase == CompetitionPhase.STOPPED
    # Signal consumed so a resumed run is not immediately re-halted.
    assert orch.interface.get_stop_requested() is False
    assert any("Stop requested" in n for n in orch.state.notes)


def test_experiment_loop_honors_stop_midway(tmp_path):
    """Stop requested during a run halts the loop on the next iteration."""
    orch = _bare_orchestrator(tmp_path, max_experiments=10)

    calls = {"n": 0}

    def fake_run(guidance=None):
        calls["n"] += 1
        orch.state.experiment_count += 1
        # After the first experiment, the user requests a stop.
        if calls["n"] == 1:
            CompetitionInterface("comp", tmp_path).request_stop()

    orch._run_experiment = fake_run

    orch._experiment_loop()

    assert calls["n"] == 1  # second iteration short-circuited by stop check
    assert orch.state.phase == CompetitionPhase.STOPPED
    assert orch.interface.get_stop_requested() is False


def test_experiment_loop_completes_without_stop(tmp_path):
    """Without a stop signal the loop runs to the budget and submits."""
    orch = _bare_orchestrator(tmp_path, max_experiments=3)

    def fake_run(guidance=None):
        orch.state.experiment_count += 1

    orch._run_experiment = fake_run

    orch._experiment_loop()

    assert orch.state.experiment_count == 3
    assert orch.state.phase == CompetitionPhase.SUBMITTING


# ---------------------------------------------------------------------------
# Fix 2: GUIDANCE written via the shared queue is consumed by the loop.
# ---------------------------------------------------------------------------

def test_experiment_loop_consumes_shared_guidance(tmp_path):
    """Guidance from `kagent guide` reaches _run_experiment via the loop."""
    orch = _bare_orchestrator(tmp_path, max_experiments=2)

    # CLI process injects guidance before the loop runs.
    CompetitionInterface("comp", tmp_path).add_guidance("use 5-fold CV")

    seen = []

    def fake_run(guidance=None):
        seen.append(guidance)
        orch.state.experiment_count += 1

    orch._run_experiment = fake_run

    orch._experiment_loop()

    # First experiment gets the guidance; second gets None (queue drained).
    assert seen == ["use 5-fold CV", None]
    # Guidance was marked processed/adopted, not left pending.
    stats = orch.interface.guidance.get_stats()
    assert stats["pending_count"] == 0
    assert stats["adopted_count"] == 1


def test_consume_guidance_returns_none_when_empty(tmp_path):
    """_consume_guidance returns None on an empty shared queue."""
    orch = _bare_orchestrator(tmp_path)
    assert orch._consume_guidance() is None


def test_consume_guidance_fifo_and_marks_processed(tmp_path):
    """_consume_guidance drains FIFO and records each as processed."""
    orch = _bare_orchestrator(tmp_path)
    orch.add_guidance("first")
    orch.add_guidance("second")

    assert orch._consume_guidance() == "first"
    assert orch._consume_guidance() == "second"
    assert orch._consume_guidance() is None

    stats = orch.interface.guidance.get_stats()
    assert stats["pending_count"] == 0
    assert stats["processed_count"] == 2
