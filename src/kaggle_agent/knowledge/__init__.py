"""Knowledge management for Kaggle Agent.

Provides:
- Playbook management (text-based experience)
- Skills management (code-based experience)
- Reflection and knowledge update mechanisms
"""

from .playbooks import PlaybookManager, PlaybookEntry, TechniqueCard
from .skills import SkillManager, Skill, SkillMetadata
from .reflection import ReflectionEngine, ReflectionResult

__all__ = [
    "PlaybookManager",
    "PlaybookEntry",
    "TechniqueCard",
    "SkillManager",
    "Skill",
    "SkillMetadata",
    "ReflectionEngine",
    "ReflectionResult",
]
