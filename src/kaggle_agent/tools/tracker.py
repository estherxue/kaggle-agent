"""Experiment tracking for Kaggle Agent.

Manages experiment history, metrics, and artifacts for a competition.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ExperimentMetrics:
    """Metrics for an experiment."""

    # Cross-validation metrics
    cv_score: Optional[float] = None
    cv_std: Optional[float] = None
    cv_folds: List[float] = field(default_factory=list)

    # Leaderboard score (after submission)
    lb_score: Optional[float] = None

    # Training metrics
    training_time_sec: Optional[float] = None
    n_samples: Optional[int] = None
    n_features: Optional[int] = None

    # Custom metrics
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    model_type: str = "unknown"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    preprocessing: List[str] = field(default_factory=list)


@dataclass
class Experiment:
    """A single experiment run."""

    id: str
    timestamp: datetime
    competition: str

    # What was attempted
    hypothesis: str
    config: ExperimentConfig

    # Code and files
    code_file: Optional[Path] = None
    config_file: Optional[Path] = None

    # Results
    metrics: ExperimentMetrics = field(default_factory=ExperimentMetrics)
    execution_result: Optional[Dict[str, Any]] = None

    # Reflection
    reflection: str = ""
    is_successful: bool = False
    parent_experiment: Optional[str] = None  # For tracking experiment lineage

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "competition": self.competition,
            "hypothesis": self.hypothesis,
            "config": asdict(self.config),
            "metrics": asdict(self.metrics),
            "code_file": str(self.code_file) if self.code_file else None,
            "config_file": str(self.config_file) if self.config_file else None,
            "execution_result": self.execution_result,
            "reflection": self.reflection,
            "is_successful": self.is_successful,
            "parent_experiment": self.parent_experiment,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            competition=data["competition"],
            hypothesis=data["hypothesis"],
            config=ExperimentConfig(**data.get("config", {})),
            metrics=ExperimentMetrics(**data.get("metrics", {})),
            code_file=Path(data["code_file"]) if data.get("code_file") else None,
            config_file=Path(data["config_file"]) if data.get("config_file") else None,
            execution_result=data.get("execution_result"),
            reflection=data.get("reflection", ""),
            is_successful=data.get("is_successful", False),
            parent_experiment=data.get("parent_experiment"),
        )


@dataclass
class ExperimentManifest:
    """Manifest of all experiments for a competition."""

    competition: str
    experiments: List[str] = field(default_factory=list)
    current_experiment: Optional[str] = None
    best_experiment: Optional[str] = None
    best_cv_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "competition": self.competition,
            "experiments": self.experiments,
            "current_experiment": self.current_experiment,
            "best_experiment": self.best_experiment,
            "best_cv_score": self.best_cv_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentManifest":
        return cls(
            competition=data["competition"],
            experiments=data.get("experiments", []),
            current_experiment=data.get("current_experiment"),
            best_experiment=data.get("best_experiment"),
            best_cv_score=data.get("best_cv_score"),
        )


class ExperimentTracker:
    """Tracks experiments for a competition.

    Manages experiment directory structure:
    competitions/<slug>/
        manifest.json
        experiments/
            exp_001/
                experiment.json
                code.py
                config.yaml
                reflection.md
                artifacts/
            exp_002/
                ...
    """

    def __init__(self, competition_slug: str, base_path: Path):
        """Initialize tracker.

        Args:
            competition_slug: Competition identifier
            base_path: Base directory for competitions
        """
        self.competition = competition_slug
        self.base_path = Path(base_path)
        self.competition_path = self.base_path / competition_slug
        self.experiments_path = self.competition_path / "experiments"
        self.manifest_path = self.competition_path / "manifest.json"

        # Ensure directories exist
        self.experiments_path.mkdir(parents=True, exist_ok=True)

        # Load or create manifest
        self._manifest = self._load_manifest()
        self._counter = len(self._manifest.experiments) + 1

    def _load_manifest(self) -> ExperimentManifest:
        """Load experiment manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                data = json.load(f)
            return ExperimentManifest.from_dict(data)
        return ExperimentManifest(competition=self.competition)

    def _save_manifest(self) -> None:
        """Save experiment manifest."""
        with open(self.manifest_path, "w") as f:
            json.dump(self._manifest.to_dict(), f, indent=2)

    def create_experiment(
        self,
        hypothesis: str,
        config: Optional[ExperimentConfig] = None,
        parent_id: Optional[str] = None,
    ) -> Experiment:
        """Create a new experiment.

        Args:
            hypothesis: What we're trying to test
            config: Experiment configuration
            parent_id: Parent experiment for tracking lineage

        Returns:
            New Experiment object
        """
        exp_id = f"exp_{self._counter:03d}"
        self._counter += 1

        exp = Experiment(
            id=exp_id,
            timestamp=datetime.now(),
            competition=self.competition,
            hypothesis=hypothesis,
            config=config or ExperimentConfig(),
            parent_experiment=parent_id,
        )

        # Create experiment directory
        exp_dir = self.experiments_path / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "artifacts").mkdir(exist_ok=True)

        # Update manifest
        self._manifest.experiments.append(exp_id)
        self._manifest.current_experiment = exp_id
        self._save_manifest()

        return exp

    def save_experiment(self, experiment: Experiment) -> None:
        """Save experiment to disk.

        Args:
            experiment: Experiment to save
        """
        exp_dir = self.experiments_path / experiment.id

        # Save experiment metadata
        exp_file = exp_dir / "experiment.json"
        with open(exp_file, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2, default=str)

        # Save code if exists
        if experiment.code_file and experiment.code_file.exists():
            # Code already written elsewhere, just update the reference
            pass

        # Save config as YAML
        if experiment.config:
            config_file = exp_dir / "config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(asdict(experiment.config), f)
            experiment.config_file = config_file

        # Save reflection as markdown
        if experiment.reflection:
            reflection_file = exp_dir / "reflection.md"
            reflection_file.write_text(experiment.reflection)

    def load_experiment(self, exp_id: str) -> Experiment:
        """Load an experiment from disk.

        Args:
            exp_id: Experiment ID

        Returns:
            Loaded Experiment
        """
        exp_file = self.experiments_path / exp_id / "experiment.json"
        with open(exp_file, "r") as f:
            data = json.load(f)
        return Experiment.from_dict(data)

    def get_code_path(self, exp_id: str) -> Path:
        """Get path for experiment code file.

        Args:
            exp_id: Experiment ID

        Returns:
            Path to code.py
        """
        return self.experiments_path / exp_id / "code.py"

    def get_artifacts_path(self, exp_id: str) -> Path:
        """Get artifacts directory for an experiment.

        Args:
            exp_id: Experiment ID

        Returns:
            Path to artifacts directory
        """
        return self.experiments_path / exp_id / "artifacts"

    def update_best(self, exp_id: str, cv_score: float) -> bool:
        """Update best experiment if this one is better.

        Args:
            exp_id: Experiment ID
            cv_score: Cross-validation score

        Returns:
            True if this is the new best
        """
        is_best = False

        if self._manifest.best_cv_score is None:
            is_best = True
        else:
            # Higher is better for now (can be configurable per metric)
            # TODO: Handle metrics where lower is better
            is_best = cv_score > self._manifest.best_cv_score

        if is_best:
            self._manifest.best_experiment = exp_id
            self._manifest.best_cv_score = cv_score
            self._save_manifest()

        return is_best

    def get_best_experiment(self) -> Optional[Experiment]:
        """Get the current best experiment.

        Returns:
            Best experiment or None
        """
        if self._manifest.best_experiment:
            return self.load_experiment(self._manifest.best_experiment)
        return None

    def list_experiments(self) -> List[str]:
        """List all experiment IDs.

        Returns:
            List of experiment IDs
        """
        return self._manifest.experiments.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments.

        Returns:
            Dict with summary statistics
        """
        experiments = []
        for exp_id in self._manifest.experiments:
            try:
                exp = self.load_experiment(exp_id)
                experiments.append({
                    "id": exp.id,
                    "hypothesis": exp.hypothesis[:50] + "..." if len(exp.hypothesis) > 50 else exp.hypothesis,
                    "cv_score": exp.metrics.cv_score,
                    "lb_score": exp.metrics.lb_score,
                    "is_successful": exp.is_successful,
                })
            except Exception:
                pass

        return {
            "competition": self.competition,
            "total_experiments": len(experiments),
            "best_experiment": self._manifest.best_experiment,
            "best_cv_score": self._manifest.best_cv_score,
            "experiments": experiments,
        }

    def get_manifest(self) -> ExperimentManifest:
        """Get the current manifest.

        Returns:
            ExperimentManifest
        """
        return self._manifest

    def export_results(self, output_path: Path) -> None:
        """Export all results to a file.

        Args:
            output_path: Path to write results
        """
        summary = self.get_summary()
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
