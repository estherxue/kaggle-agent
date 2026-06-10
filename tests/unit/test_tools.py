"""Tests for tools layer."""

import pytest
import tempfile
from pathlib import Path

from kaggle_agent.tools import (
    CodeExecutor,
    ExperimentTracker,
    MockKaggleClient,
)


def test_code_executor_success():
    """Test successful code execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = CodeExecutor(timeout_sec=10)

        code = """
import math
result = math.sqrt(16)
print(f"Result: {result}")
"""
        result = executor.execute(
            code=code,
            working_dir=Path(tmpdir),
        )

        assert result.success is True
        assert result.return_code == 0
        assert "Result: 4.0" in result.stdout
        assert result.execution_time_sec > 0


def test_code_executor_failure():
    """Test failed code execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = CodeExecutor(timeout_sec=10)

        code = "raise ValueError('Test error')"
        result = executor.execute(
            code=code,
            working_dir=Path(tmpdir),
        )

        assert result.success is False
        assert result.return_code != 0
        assert "ValueError" in result.stderr


def test_code_executor_timeout():
    """Test code execution timeout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = CodeExecutor(timeout_sec=1)

        code = "import time; time.sleep(10)"
        result = executor.execute(
            code=code,
            working_dir=Path(tmpdir),
        )

        assert result.timed_out is True
        assert result.success is False


def test_code_executor_artifact_collection():
    """Test artifact collection after execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = CodeExecutor(timeout_sec=10)

        code = '''
with open("output.csv", "w") as f:
    f.write("id,value\\n")
    f.write("1,100\\n")

with open("model.pkl", "w") as f:
    f.write("fake pickle")
'''
        result = executor.execute(
            code=code,
            working_dir=Path(tmpdir),
            artifacts_to_collect=["*.csv", "*.pkl"],
        )

        assert result.success is True
        assert len(result.artifacts) == 2
        assert "output.csv" in [p.name for p in result.artifacts.values()]


def test_experiment_tracker():
    """Test experiment tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker("test-comp", Path(tmpdir))

        # Create experiment
        exp = tracker.create_experiment(
            hypothesis="Test hypothesis",
        )

        assert exp.id == "exp_001"
        assert exp.competition == "test-comp"
        assert exp.hypothesis == "Test hypothesis"

        # Save code
        code_path = tracker.get_code_path(exp.id)
        code_path.write_text("# Test code")
        exp.code_file = code_path

        # Update metrics
        from kaggle_agent.tools.tracker import ExperimentMetrics
        exp.metrics = ExperimentMetrics(cv_score=0.85)

        # Save experiment
        tracker.save_experiment(exp)

        # Load experiment
        loaded = tracker.load_experiment(exp.id)
        assert loaded.hypothesis == exp.hypothesis
        assert loaded.metrics.cv_score == 0.85


def test_experiment_tracker_best():
    """Test tracking best experiment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker("test-comp", Path(tmpdir))

        # Create experiments with different scores
        exp1 = tracker.create_experiment(hypothesis="First")
        exp1.metrics.cv_score = 0.80
        tracker.save_experiment(exp1)
        tracker.update_best(exp1.id, 0.80)

        exp2 = tracker.create_experiment(hypothesis="Second")
        exp2.metrics.cv_score = 0.85
        tracker.save_experiment(exp2)
        is_best = tracker.update_best(exp2.id, 0.85)

        assert is_best is True
        best = tracker.get_best_experiment()
        assert best is not None
        assert best.id == exp2.id

        # Try worse score
        exp3 = tracker.create_experiment(hypothesis="Third")
        exp3.metrics.cv_score = 0.82
        tracker.save_experiment(exp3)
        is_best = tracker.update_best(exp3.id, 0.82)
        assert is_best is False


def test_experiment_tracker_summary():
    """Test experiment summary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker("test-comp", Path(tmpdir))

        # Create some experiments
        for i in range(3):
            exp = tracker.create_experiment(hypothesis=f"Test {i}")
            exp.metrics.cv_score = 0.80 + i * 0.01
            exp.is_successful = True
            tracker.save_experiment(exp)

        summary = tracker.get_summary()
        assert summary["competition"] == "test-comp"
        assert summary["total_experiments"] == 3
        assert len(summary["experiments"]) == 3


def test_mock_kaggle_client():
    """Test MockKaggleClient."""
    client = MockKaggleClient()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download competition
        dest = Path(tmpdir)
        result = client.download_competition("test-slug", dest)

        assert result == dest
        assert "test-slug" in client.downloaded_competitions
        assert (dest / "train.csv").exists()
        assert (dest / "test.csv").exists()
        assert (dest / "sample_submission.csv").exists()

        # Submit
        sub_path = dest / "submission.csv"
        sub_path.write_text("id,target\n1,0.5\n")
        sub_id = client.submit("test-slug", sub_path, "Test submission")

        assert sub_id.startswith("mock-submission")
        assert len(client.submissions) == 1

        # Get submissions
        subs = client.get_my_submissions("test-slug")
        assert len(subs) == 1
        assert subs[0].score == 0.85
