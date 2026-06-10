"""Playbook management for text-based knowledge.

Playbooks are hierarchical markdown files containing:
- General methodology (applicable to all competitions)
- Competition-type specific strategies (tabular, CV, NLP)
- Technique cards (specific tactics with validation records)
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml


@dataclass
class ValidationRecord:
    """Record of a technique's validation in a competition."""

    competition: str
    date: str
    cv_improvement: Optional[float] = None
    lb_improvement: Optional[float] = None
    notes: str = ""


@dataclass
class TechniqueCard:
    """A specific technique with usage conditions and validation.

    Stored as markdown files in knowledge/playbooks/techniques/
    """

    name: str
    applicable_types: List[str]  # ["tabular", "cv", "nlp"]
    applicable_conditions: str  # Markdown description
    usage_code: str  # Code example
    validations: List[ValidationRecord] = field(default_factory=list)
    source_file: Optional[Path] = None

    @classmethod
    def from_markdown(cls, path: Path) -> "TechniqueCard":
        """Parse a technique card from markdown file.

        Expected format:
        ---
        name: Target Encoding
        applicable_types: [tabular]
        ---

        ## Applicable Conditions
        ...

        ## Usage
        ```python
        ...
        ```

        ## Validation
        - competition: house-prices
          date: 2024-01-15
          cv_improvement: +0.002
        """
        content = path.read_text()

        # Parse frontmatter
        frontmatter_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
        if not frontmatter_match:
            raise ValueError(f"No frontmatter found in {path}")

        meta = yaml.safe_load(frontmatter_match.group(1))

        # Extract sections
        body = content[frontmatter_match.end():]

        # Find Usage section
        usage_match = re.search(
            r"## Usage\s*\n```python\n(.*?)\n```", body, re.DOTALL
        )
        usage_code = usage_match.group(1) if usage_match else ""

        # Find Applicable Conditions
        conditions_match = re.search(
            r"## Applicable Conditions\s*\n(.*?)(?=##|$)", body, re.DOTALL
        )
        conditions = conditions_match.group(1).strip() if conditions_match else ""

        # Parse Validation section
        validations = []
        validation_match = re.search(r"## Validation\s*\n(.*?)$", body, re.DOTALL)
        if validation_match:
            # Parse list items as YAML
            val_yaml = validation_match.group(1)
            try:
                val_list = yaml.safe_load(val_yaml)
                if isinstance(val_list, list):
                    for v in val_list:
                        validations.append(ValidationRecord(
                            competition=v.get("competition", ""),
                            date=v.get("date", ""),
                            cv_improvement=v.get("cv_improvement"),
                            lb_improvement=v.get("lb_improvement"),
                            notes=v.get("notes", ""),
                        ))
            except yaml.YAMLError:
                pass

        return cls(
            name=meta.get("name", path.stem),
            applicable_types=meta.get("applicable_types", []),
            applicable_conditions=conditions,
            usage_code=usage_code,
            validations=validations,
            source_file=path,
        )

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        meta = {
            "name": self.name,
            "applicable_types": self.applicable_types,
        }

        lines = [
            "---",
            yaml.dump(meta).strip(),
            "---",
            "",
            "## Applicable Conditions",
            "",
            self.applicable_conditions,
            "",
            "## Usage",
            "",
            "```python",
            self.usage_code,
            "```",
            "",
            "## Validation",
            "",
        ]

        for v in self.validations:
            lines.append(f"- competition: {v.competition}")
            lines.append(f"  date: {v.date}")
            if v.cv_improvement is not None:
                lines.append(f"  cv_improvement: {v.cv_improvement:+.4f}")
            if v.lb_improvement is not None:
                lines.append(f"  lb_improvement: {v.lb_improvement:+.4f}")
            if v.notes:
                lines.append(f"  notes: {v.notes}")
            lines.append("")

        return "\n".join(lines)

    def save(self, path: Optional[Path] = None) -> None:
        """Save to markdown file."""
        path = path or self.source_file
        if path is None:
            raise ValueError("No path specified")
        path.write_text(self.to_markdown())


@dataclass
class PlaybookEntry:
    """An entry in a playbook (general or type-specific)."""

    title: str
    content: str
    source_file: Path


class PlaybookManager:
    """Manages playbook knowledge base.

    Directory structure:
    knowledge/playbooks/
        general.md          # General methodology
        tabular.md          # Tabular competition strategies
        cv.md               # Computer vision strategies
        nlp.md              # NLP strategies
        techniques/         # Individual technique cards
            target-encoding.md
            kfold-stratified.md
            ...
    """

    def __init__(self, knowledge_path: Path):
        """Initialize playbook manager.

        Args:
            knowledge_path: Path to knowledge directory
        """
        self.playbooks_path = Path(knowledge_path) / "playbooks"
        self.techniques_path = self.playbooks_path / "techniques"

        # Ensure directories exist
        self.playbooks_path.mkdir(parents=True, exist_ok=True)
        self.techniques_path.mkdir(parents=True, exist_ok=True)

    def load_playbook(self, name: str) -> str:
        """Load a playbook file (general, tabular, cv, nlp).

        Args:
            name: Playbook name without extension

        Returns:
            Playbook content
        """
        path = self.playbooks_path / f"{name}.md"
        if not path.exists():
            return ""
        return path.read_text()

    def load_all_playbooks(self) -> Dict[str, str]:
        """Load all main playbook files.

        Returns:
            Dict mapping playbook name to content
        """
        playbooks = {}
        for playbook_file in ["general", "tabular", "cv", "nlp"]:
            content = self.load_playbook(playbook_file)
            if content:
                playbooks[playbook_file] = content
        return playbooks

    def get_relevant_playbooks(self, competition_type: str) -> Dict[str, str]:
        """Get playbooks relevant to a competition type.

        Args:
            competition_type: Type of competition (tabular, cv, nlp)

        Returns:
            Dict of relevant playbook contents
        """
        result = {}

        # Always include general
        general = self.load_playbook("general")
        if general:
            result["general"] = general

        # Include type-specific
        specific = self.load_playbook(competition_type)
        if specific:
            result[competition_type] = specific

        return result

    def load_technique(self, name: str) -> Optional[TechniqueCard]:
        """Load a specific technique card.

        Args:
            name: Technique name (file stem)

        Returns:
            TechniqueCard or None if not found
        """
        path = self.techniques_path / f"{name}.md"
        if not path.exists():
            return None
        return TechniqueCard.from_markdown(path)

    def list_techniques(self) -> List[str]:
        """List all available technique names.

        Returns:
            List of technique names (file stems)
        """
        if not self.techniques_path.exists():
            return []
        return [
            f.stem for f in self.techniques_path.glob("*.md")
        ]

    def find_techniques_for_type(self, competition_type: str) -> List[TechniqueCard]:
        """Find all techniques applicable to a competition type.

        Args:
            competition_type: Type of competition

        Returns:
            List of applicable TechniqueCards
        """
        techniques = []
        for name in self.list_techniques():
            tech = self.load_technique(name)
            if tech and competition_type in tech.applicable_types:
                techniques.append(tech)
        return techniques

    def find_techniques_for_condition(
        self,
        competition_type: str,
        condition_keywords: List[str],
    ) -> List[TechniqueCard]:
        """Find techniques matching condition keywords.

        Args:
            competition_type: Type of competition
            condition_keywords: Keywords to search for

        Returns:
            List of matching TechniqueCards
        """
        matching = []
        for tech in self.find_techniques_for_type(competition_type):
            conditions = tech.applicable_conditions.lower()
            if any(kw.lower() in conditions for kw in condition_keywords):
                matching.append(tech)
        return matching

    def add_validation(
        self,
        technique_name: str,
        validation: ValidationRecord,
    ) -> bool:
        """Add a validation record to a technique.

        Args:
            technique_name: Name of technique
            validation: Validation record to add

        Returns:
            True if added successfully
        """
        tech = self.load_technique(technique_name)
        if tech is None:
            return False

        tech.validations.append(validation)
        tech.save()
        return True

    def create_technique(self, card: TechniqueCard) -> None:
        """Create a new technique card.

        Args:
            card: Technique card to create
        """
        # Generate filename from name
        filename = card.name.lower().replace(" ", "-") + ".md"
        path = self.techniques_path / filename
        card.source_file = path
        card.save(path)

    def update_playbook(self, name: str, new_content: str) -> None:
        """Update a playbook file.

        Args:
            name: Playbook name
            new_content: New content to write
        """
        path = self.playbooks_path / f"{name}.md"
        path.write_text(new_content)

    def get_context_for_llm(
        self,
        competition_type: str,
        max_techniques: int = 10,
    ) -> str:
        """Get formatted context for LLM consumption.

        Args:
            competition_type: Type of competition
            max_techniques: Maximum techniques to include

        Returns:
            Formatted string for LLM context
        """
        parts = []

        # Add relevant playbooks
        playbooks = self.get_relevant_playbooks(competition_type)
        for name, content in playbooks.items():
            parts.append(f"## {name.upper()} PLAYBOOK")
            parts.append(content)
            parts.append("")

        # Add relevant techniques
        techniques = self.find_techniques_for_type(competition_type)
        if techniques:
            parts.append(f"## RELEVANT TECHNIQUES ({len(techniques)} available)")
            for tech in techniques[:max_techniques]:
                parts.append(f"\n### {tech.name}")
                parts.append(f"Applicable when: {tech.applicable_conditions[:200]}...")
                if tech.validations:
                    v = tech.validations[-1]  # Most recent
                    parts.append(f"Last validated: {v.competition} (CV: {v.cv_improvement:+.4f})")
                parts.append(f"```python\n{tech.usage_code[:500]}...\n```")

        return "\n".join(parts)
