"""Skills management for code-based knowledge.

Skills are reusable code modules with metadata about their
applicability and validation history.
"""

import ast
import importlib.util
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import yaml


@dataclass
class SkillMetadata:
    """Metadata for a skill.

    Embedded as docstring or separate YAML file next to the skill.
    """

    name: str
    description: str
    applicable_types: List[str]  # ["tabular", "cv", "nlp"]
    applicable_conditions: str
    inputs: List[str]  # Expected input variable names/types
    outputs: List[str]  # Output variable names/types
    verified_in: List[str] = field(default_factory=list)  # Competition slugs
    avg_cv_improvement: Optional[float] = None
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_yaml(cls, path: Path) -> "SkillMetadata":
        """Load from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.__dict__)


@dataclass
class Skill:
    """A reusable code skill.

    Skills are Python modules with metadata. They can be:
    - Functions for feature engineering
    - Classes for CV splitters
    - Complete pipeline templates
    """

    name: str
    module_path: Path
    metadata: SkillMetadata
    source_code: str = ""
    is_validated: bool = False

    @classmethod
    def from_file(cls, module_path: Path, metadata_path: Optional[Path] = None) -> "Skill":
        """Load a skill from file.

        Args:
            module_path: Path to Python module
            metadata_path: Path to metadata YAML (optional)

        Returns:
            Skill object
        """
        # Load source code
        source_code = module_path.read_text()

        # Extract name from module filename
        name = module_path.stem

        # Load or create metadata
        if metadata_path is None:
            metadata_path = module_path.with_suffix(".yaml")

        if metadata_path.exists():
            metadata = SkillMetadata.from_yaml(metadata_path)
        else:
            # Infer from docstring
            metadata = cls._infer_metadata(name, source_code)

        # Check if validated
        is_validated = len(metadata.verified_in) >= 3

        return cls(
            name=name,
            module_path=module_path,
            metadata=metadata,
            source_code=source_code,
            is_validated=is_validated,
        )

    @staticmethod
    def _infer_metadata(name: str, source_code: str) -> SkillMetadata:
        """Infer metadata from source code docstring."""
        # Try to extract docstring
        try:
            tree = ast.parse(source_code)
            docstring = ast.get_docstring(tree)
        except SyntaxError:
            docstring = ""

        return SkillMetadata(
            name=name,
            description=docstring or f"Skill: {name}",
            applicable_types=["tabular"],  # Default assumption
            applicable_conditions="See source code",
            inputs=[],
            outputs=[],
        )

    def load_module(self) -> Any:
        """Dynamically load the skill as a Python module.

        Returns:
            Loaded module
        """
        spec = importlib.util.spec_from_file_location(self.name, self.module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load skill from {self.module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def get_function(self, func_name: str) -> Any:
        """Get a specific function from the skill module.

        Args:
            func_name: Function name

        Returns:
            Function object
        """
        module = self.load_module()
        return getattr(module, func_name)

    def update_validation(
        self,
        competition: str,
        cv_improvement: Optional[float] = None,
    ) -> None:
        """Update validation history.

        Args:
            competition: Competition slug
            cv_improvement: CV score improvement
        """
        if competition not in self.metadata.verified_in:
            self.metadata.verified_in.append(competition)

        # Update average improvement
        if cv_improvement is not None:
            if self.metadata.avg_cv_improvement is None:
                self.metadata.avg_cv_improvement = cv_improvement
            else:
                # Simple moving average
                n = len(self.metadata.verified_in)
                self.metadata.avg_cv_improvement = (
                    (self.metadata.avg_cv_improvement * (n - 1) + cv_improvement) / n
                )

        self.metadata.last_updated = datetime.now().isoformat()
        self.is_validated = len(self.metadata.verified_in) >= 3

        # Save metadata
        self.save_metadata()

    def save_metadata(self) -> None:
        """Save metadata to YAML file."""
        metadata_path = self.module_path.with_suffix(".yaml")
        with open(metadata_path, "w") as f:
            f.write(self.metadata.to_yaml())

    def to_usage_string(self) -> str:
        """Generate a usage code snippet.

        Returns:
            Python code string for using this skill
        """
        lines = [
            f"# Skill: {self.name}",
            f"# {self.metadata.description[:100]}",
            f"from skills.{self.module_path.parent.name}.{self.name} import main_function",
            "",
            "# Usage:",
        ]

        if self.metadata.inputs:
            lines.append(f"# Inputs: {', '.join(self.metadata.inputs)}")
        if self.metadata.outputs:
            lines.append(f"# Outputs: {', '.join(self.metadata.outputs)}")

        lines.append(f"# result = main_function({', '.join(self.metadata.inputs)})")

        return "\n".join(lines)


class SkillManager:
    """Manages code skills library.

    Directory structure:
    knowledge/skills/
        __init__.py
        base.py
        feature_engineering/
            target_encoding.py
            target_encoding.yaml
            count_encoding.py
            count_encoding.yaml
        cross_validation/
            stratified_kfold.py
            stratified_kfold.yaml
        models/
            lgbm_classifier.py
            lgbm_classifier.yaml
        pipelines/
            tabular_baseline.py
    """

    def __init__(self, knowledge_path: Path):
        """Initialize skill manager.

        Args:
            knowledge_path: Path to knowledge directory
        """
        self.skills_path = Path(knowledge_path) / "skills"

        # Ensure directories exist
        for category in ["feature_engineering", "cross_validation", "models", "pipelines"]:
            (self.skills_path / category).mkdir(parents=True, exist_ok=True)

    def _discover_skills(self, category: Optional[str] = None) -> List[Path]:
        """Discover all skill module files.

        Args:
            category: Optional category to limit search

        Returns:
            List of module paths
        """
        if category:
            search_path = self.skills_path / category
            if not search_path.exists():
                return []
            return list(search_path.glob("*.py"))
        else:
            # Search all categories
            skills = []
            for cat_dir in self.skills_path.iterdir():
                if cat_dir.is_dir() and cat_dir.name != "__pycache__":
                    skills.extend(cat_dir.glob("*.py"))
            return skills

    def load_skill(self, name: str, category: Optional[str] = None) -> Optional[Skill]:
        """Load a specific skill.

        Args:
            name: Skill name (module name)
            category: Optional category to search in

        Returns:
            Skill or None if not found
        """
        if category:
            module_path = self.skills_path / category / f"{name}.py"
            if module_path.exists():
                return Skill.from_file(module_path)
            return None
        else:
            # Search all categories
            for cat_dir in self.skills_path.iterdir():
                if cat_dir.is_dir():
                    module_path = cat_dir / f"{name}.py"
                    if module_path.exists():
                        return Skill.from_file(module_path)
            return None

    def list_skills(
        self,
        category: Optional[str] = None,
        competition_type: Optional[str] = None,
        validated_only: bool = False,
    ) -> List[Skill]:
        """List skills with optional filtering.

        Args:
            category: Filter by category
            competition_type: Filter by applicable competition type
            validated_only: Only return validated skills

        Returns:
            List of Skill objects
        """
        skills = []

        for module_path in self._discover_skills(category):
            try:
                skill = Skill.from_file(module_path)

                # Apply filters
                if competition_type and competition_type not in skill.metadata.applicable_types:
                    continue
                if validated_only and not skill.is_validated:
                    continue

                skills.append(skill)
            except Exception:
                # Skip broken skills
                continue

        return skills

    def get_skills_for_context(
        self,
        competition_type: str,
        max_skills: int = 10,
        prefer_validated: bool = True,
    ) -> List[Skill]:
        """Get skills relevant for LLM context.

        Args:
            competition_type: Type of competition
            max_skills: Maximum number of skills
            prefer_validated: Prefer validated skills

        Returns:
            List of relevant skills
        """
        skills = self.list_skills(competition_type=competition_type)

        # Sort: validated first, then by average improvement
        def sort_key(skill: Skill) -> tuple:
            validated = 1 if skill.is_validated else 0
            improvement = skill.metadata.avg_cv_improvement or 0
            return (validated, improvement)

        skills.sort(key=sort_key, reverse=True)

        return skills[:max_skills]

    def create_skill(
        self,
        name: str,
        category: str,
        source_code: str,
        metadata: SkillMetadata,
    ) -> Skill:
        """Create a new skill.

        Args:
            name: Skill name
            category: Skill category
            source_code: Python source code
            metadata: Skill metadata

        Returns:
            Created Skill
        """
        # Write source file
        module_path = self.skills_path / category / f"{name}.py"
        module_path.write_text(source_code)

        # Write metadata
        metadata_path = module_path.with_suffix(".yaml")
        with open(metadata_path, "w") as f:
            f.write(metadata.to_yaml())

        return Skill.from_file(module_path, metadata_path)

    def refine_skill_from_experiment(
        self,
        experiment_code: str,
        experiment_result: Dict[str, Any],
    ) -> Optional[Skill]:
        """Extract a new skill from successful experiment code.

        This is a placeholder for automatic skill extraction.
        In practice, this would use LLM to identify and extract
        reusable code patterns from successful experiments.

        Args:
            experiment_code: Experiment source code
            experiment_result: Experiment results

        Returns:
            New Skill or None if extraction fails
        """
        # TODO: Implement LLM-based skill extraction
        # For now, return None
        return None

    def get_context_for_llm(
        self,
        competition_type: str,
        max_skills: int = 10,
    ) -> str:
        """Get formatted skill context for LLM.

        Args:
            competition_type: Type of competition
            max_skills: Maximum skills to include

        Returns:
            Formatted string for LLM context
        """
        skills = self.get_skills_for_context(competition_type, max_skills)

        if not skills:
            return "# AVAILABLE SKILLS\n\nNo skills available yet."

        parts = [f"# AVAILABLE SKILLS ({len(skills)} shown)\n"]

        for skill in skills:
            parts.append(f"\n## {skill.name}")
            parts.append(f"Category: {skill.module_path.parent.name}")
            parts.append(f"Description: {skill.metadata.description}")
            parts.append(f"Validated in: {', '.join(skill.metadata.verified_in[:3])}")
            if skill.metadata.avg_cv_improvement:
                parts.append(f"Avg improvement: {skill.metadata.avg_cv_improvement:+.4f}")

            # Show signature from source
            parts.append(f"\n```python")
            parts.append(skill.source_code[:1000])  # Truncated
            parts.append("```")

        return "\n".join(parts)
