"""Orchestrator for Kaggle Agent competition lifecycle.

Manages the full lifecycle of a competition:
1. Understanding: Load competition info
2. Knowledge Loading: Load relevant playbooks and skills
3. EDA: Generate and execute exploratory analysis
4. Experiment Loop: Iterate on hypotheses
5. Submit: Select and submit best solution
6. Retrospective: Reflect and update knowledge
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config
from .knowledge import PlaybookManager, ReflectionEngine, SkillManager
from .llm import ChatMessage, LLMRouter
from .tools import (
    CodeExecutor,
    Experiment,
    ExperimentConfig,
    ExperimentMetrics,
    ExperimentTracker,
    KaggleClient,
    MockKaggleClient,
)


class CompetitionPhase(Enum):
    """Phases of the competition lifecycle."""

    INITIALIZING = auto()
    UNDERSTANDING = auto()
    LOADING_KNOWLEDGE = auto()
    EDA = auto()
    EXPERIMENTING = auto()
    SUBMITTING = auto()
    RETROSPECTIVE = auto()
    COMPLETED = auto()
    ERROR = auto()
    STOPPED = auto()


@dataclass
class CompetitionState:
    """Current state of a competition run."""

    competition: str
    phase: CompetitionPhase = CompetitionPhase.INITIALIZING
    competition_type: str = "unknown"
    started_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    current_experiment: Optional[str] = None
    best_cv_score: Optional[float] = None
    experiment_count: int = 0
    total_llm_cost: float = 0.0
    guidance_queue: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "competition": self.competition,
            "phase": self.phase.name,
            "competition_type": self.competition_type,
            "started_at": self.started_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "current_experiment": self.current_experiment,
            "best_cv_score": self.best_cv_score,
            "experiment_count": self.experiment_count,
            "total_llm_cost": self.total_llm_cost,
            "guidance_queue": self.guidance_queue,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompetitionState":
        return cls(
            competition=data["competition"],
            phase=CompetitionPhase[data.get("phase", "INITIALIZING")],
            competition_type=data.get("competition_type", "unknown"),
            started_at=datetime.fromisoformat(data["started_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            current_experiment=data.get("current_experiment"),
            best_cv_score=data.get("best_cv_score"),
            experiment_count=data.get("experiment_count", 0),
            total_llm_cost=data.get("total_llm_cost", 0.0),
            guidance_queue=data.get("guidance_queue", []),
            notes=data.get("notes", []),
        )


class Orchestrator:
    """Main orchestrator for Kaggle Agent.

    Coordinates all components through the competition lifecycle.
    """

    def __init__(
        self,
        config: Config,
        competition: str,
        llm_router: LLMRouter,
        kaggle_client: Optional[KaggleClient] = None,
        resume: bool = False,
    ):
        """Initialize orchestrator.

        Args:
            config: Agent configuration
            competition: Competition slug
            llm_router: LLM router for generation
            kaggle_client: Kaggle client (or MockKaggleClient for testing)
            resume: Whether to resume from previous state
        """
        self.config = config
        self.competition = competition
        self.llm = llm_router

        # Initialize clients
        self.kaggle = kaggle_client or KaggleClient(
            dry_run=config.kaggle.dry_run
        )
        self.executor = CodeExecutor(
            timeout_sec=config.execution.timeout_sec,
            allow_network=config.execution.allow_network,
            python_path=config.execution.python_path,
        )

        # Initialize knowledge systems
        knowledge_path = config.resolve_path("knowledge")
        self.playbooks = PlaybookManager(knowledge_path)
        self.skills = SkillManager(knowledge_path)
        self.reflection = ReflectionEngine(
            llm_router=llm_router,
            playbook_manager=self.playbooks,
            skill_manager=self.skills,
        )

        # Initialize experiment tracking
        competitions_path = config.resolve_path("competitions")
        self.tracker = ExperimentTracker(competition, competitions_path)

        # State management
        self.state_path = competitions_path / competition / "state.json"
        if resume and self.state_path.exists():
            self.state = self._load_state()
        else:
            self.state = CompetitionState(competition=competition)

        # Control flags
        self._should_stop = False

    def _load_state(self) -> CompetitionState:
        """Load state from disk."""
        with open(self.state_path, "r") as f:
            data = json.load(f)
        return CompetitionState.from_dict(data)

    def _save_state(self) -> None:
        """Save state to disk."""
        self.state.last_updated = datetime.now()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def stop(self) -> None:
        """Signal the orchestrator to stop after current experiment."""
        self._should_stop = True
        self.state.phase = CompetitionPhase.STOPPED
        self._save_state()

    def add_guidance(self, guidance: str) -> None:
        """Add guidance to the queue.

        Args:
            guidance: Guidance text
        """
        self.state.guidance_queue.append(guidance)
        self._save_state()

    def _consume_guidance(self) -> Optional[str]:
        """Consume guidance from queue."""
        if self.state.guidance_queue:
            return self.state.guidance_queue.pop(0)
        return None

    def run(self) -> None:
        """Run the full competition lifecycle."""
        try:
            # Phase 1: Initialize
            self._initialize()

            # Phase 2: Understand competition
            if self.state.phase in [CompetitionPhase.INITIALIZING, CompetitionPhase.UNDERSTANDING]:
                self._understand_competition()

            # Phase 3: Load knowledge
            if self.state.phase == CompetitionPhase.LOADING_KNOWLEDGE:
                self._load_knowledge()

            # Phase 4: EDA
            if self.state.phase == CompetitionPhase.EDA:
                self._run_eda()

            # Phase 5: Experiment loop
            if self.state.phase == CompetitionPhase.EXPERIMENTING:
                self._experiment_loop()

            # Phase 6: Submit
            if self.state.phase == CompetitionPhase.SUBMITTING:
                self._submit_best()

            # Phase 7: Retrospective
            if self.state.phase == CompetitionPhase.COMPLETED:
                self._retrospective()

        except Exception as e:
            self.state.phase = CompetitionPhase.ERROR
            self.state.notes.append(f"Error: {str(e)}")
            self._save_state()
            raise

    def _initialize(self) -> None:
        """Initialize competition workspace."""
        self.state.phase = CompetitionPhase.UNDERSTANDING
        self._save_state()

        # Download competition data
        comp_path = self.config.resolve_path("competitions") / self.competition
        self.kaggle.download_competition(self.competition, comp_path / "data")

        self.state.notes.append(f"Downloaded competition data to {comp_path}")
        self._save_state()

    def _understand_competition(self) -> None:
        """Understand competition overview and metrics."""
        # Get competition info from Kaggle
        info = self.kaggle.get_competition_info(self.competition)

        # Determine competition type (simplified - could use LLM)
        self.state.competition_type = self._infer_competition_type(info)

        self.state.notes.append(f"Competition type: {self.state.competition_type}")
        self.state.phase = CompetitionPhase.LOADING_KNOWLEDGE
        self._save_state()

    def _infer_competition_type(self, info: Any) -> str:
        """Infer competition type from info."""
        # Simple heuristic based on description
        description_lower = info.description.lower() if info.description else ""

        if any(word in description_lower for word in ["image", "photo", "picture", "vision"]):
            return "cv"
        elif any(word in description_lower for word in ["text", "nlp", "language", "sentence"]):
            return "nlp"
        else:
            # Default to tabular
            return "tabular"

    def _load_knowledge(self) -> None:
        """Load relevant knowledge for the competition."""
        # Knowledge is loaded implicitly when LLM context is built
        # This phase can be used to pre-cache or validate

        playbooks = self.playbooks.get_relevant_playbooks(self.state.competition_type)
        self.state.notes.append(f"Loaded {len(playbooks)} relevant playbooks")

        skills = self.skills.get_skills_for_context(
            self.state.competition_type, max_skills=5
        )
        self.state.notes.append(f"Loaded {len(skills)} relevant skills")

        self.state.phase = CompetitionPhase.EDA
        self._save_state()

    def _run_eda(self) -> None:
        """Run exploratory data analysis."""
        # Build EDA prompt with knowledge context
        context = self._build_llm_context()

        prompt = f"""{context}

## Task

Generate Python code for exploratory data analysis (EDA) of this {self.state.competition_type} competition.

Your code should:
1. Load the data files
2. Show basic statistics
3. Analyze feature distributions
4. Check for missing values
5. Understand the target variable
6. Identify any data quality issues

Write the complete, runnable Python code."""

        # Generate EDA code
        response = self.llm.chat(
            role="coder",
            messages=[
                ChatMessage(role="system", content="You are a data scientist writing EDA code."),
                ChatMessage(role="user", content=prompt),
            ],
            max_tokens=2000,
        )

        # Execute EDA code
        code = self._extract_code(response.content)
        comp_path = self.config.resolve_path("competitions") / self.competition
        result = self.executor.execute(
            code=code,
            working_dir=comp_path / "eda",
            artifacts_to_collect=["*.png", "*.csv", "eda_report.md"],
        )

        # Save EDA results
        eda_report_path = comp_path / "eda_report.md"
        eda_report_path.write_text(
            f"# EDA Report\n\n## Execution\nSuccess: {result.success}\n\n"
            f"## Output\n```\n{result.stdout[:5000]}\n```\n\n"
            f"## Errors\n```\n{result.stderr[:2000]}\n```\n"
        )

        self.state.notes.append(f"EDA completed: success={result.success}")
        self.state.phase = CompetitionPhase.EXPERIMENTING
        self._save_state()

    def _experiment_loop(self) -> None:
        """Main experiment iteration loop."""
        budget = self.config.budget

        while (
            self.state.experiment_count < budget.max_experiments_per_competition
            and not self._should_stop
        ):
            # Check budget
            if self.state.total_llm_cost >= budget.max_llm_cost_usd:
                self.state.notes.append("LLM budget exhausted")
                break

            # Check for guidance
            guidance = self._consume_guidance()

            # Run single experiment
            self._run_experiment(guidance)

            # Update cost tracking
            self.state.total_llm_cost = self.llm.get_total_cost()
            self._save_state()

        if self._should_stop:
            self.state.phase = CompetitionPhase.STOPPED
        else:
            self.state.phase = CompetitionPhase.SUBMITTING
        self._save_state()

    def _run_experiment(self, guidance: Optional[str] = None) -> None:
        """Run a single experiment."""
        self.state.experiment_count += 1

        # Build context
        context = self._build_llm_context()

        # Get previous experiments for context
        prev_exps = self.tracker.get_summary()["experiments"][-5:]

        # Build hypothesis and code prompt
        guidance_text = f"\n## User Guidance\n{guidance}\n" if guidance else ""

        prompt = f"""{context}

## Previous Experiments (last 5)
{self._format_previous_experiments(prev_exps)}
{guidance_text}

## Task

Propose a hypothesis and write Python code to test it for experiment #{self.state.experiment_count}.

Your response should include:
1. HYPOTHESIS: What you're testing (1 sentence)
2. CODE: Complete, runnable Python code

The code should:
- Load data from ../data/
- Train a model
- Output CV score
- Save predictions to submission.csv
"""

        # Generate experiment
        response = self.llm.chat(
            role="coder",
            messages=[
                ChatMessage(role="system", content="You are an ML engineer running experiments."),
                ChatMessage(role="user", content=prompt),
            ],
            max_tokens=2500,
        )

        # Parse hypothesis and code
        content = response.content
        hypothesis = self._extract_hypothesis(content)
        code = self._extract_code(content)

        # Create experiment record
        exp = self.tracker.create_experiment(
            hypothesis=hypothesis or f"Experiment {self.state.experiment_count}",
        )
        self.state.current_experiment = exp.id

        # Save code
        code_path = self.tracker.get_code_path(exp.id)
        code_path.write_text(code)

        # Execute experiment
        comp_path = self.config.resolve_path("competitions") / self.competition
        result = self.executor.execute(
            code=code,
            working_dir=comp_path / "experiments" / exp.id,
            artifacts_to_collect=["submission.csv", "*.pkl", "*.json"],
        )

        # Parse results
        metrics = self._parse_experiment_results(result)
        exp.metrics = ExperimentMetrics(
            cv_score=metrics.get("cv_score"),
            training_time_sec=result.execution_time_sec,
        )
        exp.is_successful = result.success and metrics.get("cv_score") is not None

        # Generate reflection
        if exp.is_successful:
            reflection = self.reflection.reflect_on_experiment(
                hypothesis=hypothesis,
                code=code,
                metrics=metrics,
                execution_result={
                    "success": result.success,
                    "timed_out": result.timed_out,
                    "has_errors": not result.success,
                },
                previous_experiments=prev_exps,
            )
            exp.reflection = reflection

        # Save experiment
        self.tracker.save_experiment(exp)

        # Update best if applicable
        if exp.metrics.cv_score:
            is_best = self.tracker.update_best(exp.id, exp.metrics.cv_score)
            if is_best:
                self.state.best_cv_score = exp.metrics.cv_score
                self.state.notes.append(f"New best CV: {exp.metrics.cv_score:.4f} (exp {exp.id})")

        self._save_state()

    def _submit_best(self) -> None:
        """Submit the best experiment to Kaggle."""
        best = self.tracker.get_best_experiment()
        if not best:
            self.state.notes.append("No successful experiments to submit")
            self.state.phase = CompetitionPhase.COMPLETED
            self._save_state()
            return

        # Get submission file
        comp_path = self.config.resolve_path("competitions") / self.competition
        artifacts_path = self.tracker.get_artifacts_path(best.id)
        submission_file = artifacts_path / "submission.csv"

        if not submission_file.exists():
            self.state.notes.append(f"No submission file for {best.id}")
            self.state.phase = CompetitionPhase.COMPLETED
            self._save_state()
            return

        # Submit
        if self.config.kaggle.auto_submit:
            submission_id = self.kaggle.submit(
                slug=self.competition,
                file_path=submission_file,
                message=f"Kaggle Agent submission (exp {best.id}, CV: {best.metrics.cv_score:.4f})",
            )
            self.state.notes.append(f"Submitted: {submission_id}")

            # Get LB score (may not be immediately available)
            submissions = self.kaggle.get_my_submissions(self.competition, limit=1)
            if submissions:
                best.metrics.lb_score = submissions[0].score
                self.tracker.save_experiment(best)

        self.state.phase = CompetitionPhase.COMPLETED
        self._save_state()

    def _retrospective(self) -> None:
        """Generate retrospective and update knowledge."""
        summary = self.tracker.get_summary()

        retrospective = self.reflection.generate_retrospective(
            competition=self.competition,
            competition_type=self.state.competition_type,
            experiments=summary["experiments"],
            final_score=summary.get("best_cv_score"),
        )

        # Save retrospective
        comp_path = self.config.resolve_path("competitions") / self.competition
        self.reflection.save_retrospective(
            competition=self.competition,
            retrospective=retrospective,
            output_path=comp_path / "retrospective.md",
        )

        self.state.notes.append("Retrospective completed")
        self._save_state()

    def _build_llm_context(self) -> str:
        """Build context string for LLM prompts."""
        parts = []

        # Add relevant playbooks
        playbook_context = self.playbooks.get_context_for_llm(
            self.state.competition_type, max_techniques=5
        )
        parts.append(playbook_context)

        # Add relevant skills
        skill_context = self.skills.get_context_for_llm(
            self.state.competition_type, max_skills=5
        )
        parts.append(skill_context)

        return "\n\n".join(parts)

    @staticmethod
    def _extract_code(content: str) -> str:
        """Extract code from LLM response."""
        # Look for code blocks
        import re

        code_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
        if code_blocks:
            return code_blocks[-1]  # Return last code block

        # If no code blocks, return content after "CODE:"
        if "CODE:" in content:
            return content.split("CODE:")[-1].strip()

        return content

    @staticmethod
    def _extract_hypothesis(content: str) -> Optional[str]:
        """Extract hypothesis from LLM response."""
        import re

        match = re.search(r"HYPOTHESIS[:\s]*(.*?)(?:\n|CODE:|```)", content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _parse_experiment_results(result: Any) -> Dict[str, Any]:
        """Parse metrics from experiment execution."""
        metrics = {}

        # Try to extract CV score from stdout
        import re

        # Look for common patterns
        cv_match = re.search(r"CV score[:\s]+([\d.]+)", result.stdout, re.IGNORECASE)
        if cv_match:
            metrics["cv_score"] = float(cv_match.group(1))

        # Alternative patterns
        if "cv_score" not in metrics:
            score_match = re.search(r"score[:\s]+([\d.]+)", result.stdout, re.IGNORECASE)
            if score_match:
                metrics["cv_score"] = float(score_match.group(1))

        return metrics

    @staticmethod
    def _format_previous_experiments(experiments: List[Dict[str, Any]]) -> str:
        """Format previous experiments for context."""
        if not experiments:
            return "No previous experiments."

        lines = []
        for exp in experiments:
            lines.append(f"- {exp.get('id', '?'')}: CV={exp.get('cv_score', 'N/A')}, "
                        f"Success={exp.get('is_successful', False)}")
        return "\n".join(lines)

    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "competition": self.competition,
            "phase": self.state.phase.name,
            "experiment_count": self.state.experiment_count,
            "best_cv_score": self.state.best_cv_score,
            "llm_cost": self.state.total_llm_cost,
            "guidance_queue_length": len(self.state.guidance_queue),
            "notes": self.state.notes[-5:],  # Last 5 notes
        }
