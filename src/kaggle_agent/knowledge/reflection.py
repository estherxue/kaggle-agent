"""Reflection engine for knowledge updates.

Handles:
- Experiment-level reflection (quick notes after each experiment)
- Competition-level retrospective (comprehensive analysis after competition)
- Knowledge update (proposing changes to playbooks and skills)
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..llm import ChatMessage, LLMRouter
from .playbooks import PlaybookManager, TechniqueCard, ValidationRecord
from .skills import SkillManager, Skill, SkillMetadata


@dataclass
class ReflectionResult:
    """Result of reflection process."""

    summary: str
    key_insights: List[str]
    proposed_playbook_updates: List[Dict[str, Any]]
    proposed_new_techniques: List[Dict[str, Any]]
    proposed_skill_updates: List[Dict[str, Any]]
    success_rating: float  # 0.0 to 1.0


class ReflectionEngine:
    """Engine for reflection and knowledge updates.

    Uses LLM to:
    1. Analyze experiment results and generate reflections
    2. Summarize competition experiences
    3. Propose updates to knowledge base
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        playbook_manager: PlaybookManager,
        skill_manager: SkillManager,
    ):
        """Initialize reflection engine.

        Args:
            llm_router: LLM router for generation
            playbook_manager: Playbook manager
            skill_manager: Skill manager
        """
        self.llm = llm_router
        self.playbooks = playbook_manager
        self.skills = skill_manager

    def reflect_on_experiment(
        self,
        hypothesis: str,
        code: str,
        metrics: Dict[str, Any],
        execution_result: Dict[str, Any],
        previous_experiments: List[Dict[str, Any]],
    ) -> str:
        """Generate reflection for a single experiment.

        Args:
            hypothesis: What was tried
            code: Code that was run
            metrics: Result metrics (CV, training time, etc.)
            execution_result: Execution details
            previous_experiments: Context from previous experiments

        Returns:
            Reflection text
        """
        prompt = f"""You are analyzing a machine learning experiment for a Kaggle competition.

## Experiment

**Hypothesis:** {hypothesis}

**Code:**
```python
{code[:2000]}...
```

**Results:**
- CV Score: {metrics.get('cv_score', 'N/A')}
- LB Score: {metrics.get('lb_score', 'N/A')}
- Training Time: {metrics.get('training_time_sec', 'N/A')}s
- Success: {execution_result.get('success', False)}

**Execution:**
- Timed out: {execution_result.get('timed_out', False)}
- Has errors: {execution_result.get('has_errors', False)}

## Previous Context

{self._format_previous_experiments(previous_experiments)}

## Your Task

Write a brief reflection (3-5 sentences) on this experiment:
1. Did it work as expected?
2. What worked or didn't work?
3. What should be tried next?

Be specific and actionable."""

        response = self.llm.chat(
            role="reviewer",
            messages=[
                ChatMessage(role="system", content="You are a machine learning engineer analyzing experiments."),
                ChatMessage(role="user", content=prompt),
            ],
            max_tokens=500,
        )

        return response.content.strip()

    def generate_retrospective(
        self,
        competition: str,
        competition_type: str,
        experiments: List[Dict[str, Any]],
        final_rank: Optional[int] = None,
        final_score: Optional[float] = None,
    ) -> ReflectionResult:
        """Generate comprehensive retrospective for a competition.

        Args:
            competition: Competition slug
            competition_type: Type of competition
            experiments: All experiments from the competition
            final_rank: Final leaderboard rank
            final_score: Final score

        Returns:
            ReflectionResult with insights and proposed updates
        """
        # Get relevant playbooks for context
        playbook_context = self.playbooks.get_context_for_llm(competition_type)
        skill_context = self.skills.get_context_for_llm(competition_type)

        prompt = f"""You are conducting a post-mortem analysis of a Kaggle competition.

## Competition: {competition}
Type: {competition_type}
Final Score: {final_score}
Final Rank: {final_rank}

## Current Knowledge

{playbook_context}

{skill_context}

## Experiment History

{self._format_experiment_history(experiments)}

## Your Task

Write a comprehensive retrospective and propose knowledge updates:

1. **Summary**: Overall performance and approach
2. **Key Insights** (3-5 bullet points): What actually worked
3. **What Didn't Work**: Failed attempts and why
4. **CV-LB Analysis**: Was local validation reliable?

## Proposed Knowledge Updates

Based on this competition, what should be added to our knowledge base?

### New Techniques to Document
- Name, applicable conditions, and key code pattern

### Playbook Updates
- General principles that emerged
- Type-specific learnings

### Skill Refinements
- Existing skills that should be updated based on results

Format your response as a structured analysis."""

        response = self.llm.chat(
            role="reviewer",
            messages=[
                ChatMessage(role="system", content="You are a senior ML engineer conducting competition retrospectives."),
                ChatMessage(role="user", content=prompt),
            ],
            max_tokens=2000,
        )

        # Parse response into structured result
        # For now, return a simplified structure
        # TODO: Implement proper parsing of structured output
        content = response.content

        return ReflectionResult(
            summary=content[:500],
            key_insights=["Key insight 1", "Key insight 2"],  # TODO: Parse from content
            proposed_playbook_updates=[],
            proposed_new_techniques=[],
            proposed_skill_updates=[],
            success_rating=0.5 if final_score else 0.0,
        )

    def propose_playbook_updates(
        self,
        retrospective: ReflectionResult,
        competition_type: str,
    ) -> List[Dict[str, Any]]:
        """Propose specific updates to playbooks.

        Args:
            retrospective: Retrospective analysis
            competition_type: Type of competition

        Returns:
            List of proposed updates with rationale
        """
        # Get current playbook
        current_playbook = self.playbooks.load_playbook(competition_type)

        prompt = f"""Review the retrospective and propose specific updates to the {competition_type} playbook.

## Current Playbook

{current_playbook[:2000]}...

## Retrospective Insights

{retrospective.summary}

Key insights:
{chr(10).join(f"- {i}" for i in retrospective.key_insights)}

## Task

Propose 2-3 specific additions or modifications to the playbook.
For each proposal, provide:
1. Section to update (or "new section")
2. Proposed content (concise)
3. Rationale

Format as a list of structured proposals."""

        response = self.llm.chat(
            role="summarizer",
            messages=[
                ChatMessage(role="system", content="You are updating ML playbooks based on empirical results."),
                ChatMessage(role="user", content=prompt),
            ],
            max_tokens=1500,
        )

        # TODO: Parse structured proposals from response
        return []

    def extract_new_technique(
        self,
        experiment_code: str,
        experiment_results: Dict[str, Any],
        competition_type: str,
    ) -> Optional[TechniqueCard]:
        """Extract a new technique card from successful experiment.

        Args:
            experiment_code: Code from successful experiment
            experiment_results: Experiment metrics
            competition_type: Type of competition

        Returns:
            New TechniqueCard or None if not suitable
        """
        # Check if result is good enough
        if not experiment_results.get("is_successful", False):
            return None

        prompt = f"""Analyze this successful experiment code and extract a reusable technique.

## Experiment Results
- CV Score: {experiment_results.get('cv_score')}
- Improvement: {experiment_results.get('improvement', 'N/A')}

## Code

```python
{experiment_code[:3000]}...
```

## Task

If this contains a generalizable technique, describe:
1. Name of the technique
2. When it applies (conditions)
3. Core code pattern (the reusable part)

If not generalizable, reply with "NOT_GENERALIZABLE".
"""

        response = self.llm.chat(
            role="coder",
            messages=[
                ChatMessage(role="system", content="You extract reusable ML techniques from code."),
                ChatMessage(role="user", content=prompt),
            ],
            max_tokens=1000,
        )

        content = response.content.strip()

        if "NOT_GENERALIZABLE" in content:
            return None

        # Try to parse technique from response
        # TODO: Implement proper parsing
        name = "extracted_technique"
        conditions = "See documentation"
        code_pattern = experiment_code[:500]

        return TechniqueCard(
            name=name,
            applicable_types=[competition_type],
            applicable_conditions=conditions,
            usage_code=code_pattern,
            validations=[
                ValidationRecord(
                    competition=experiment_results.get('competition', 'unknown'),
                    date=datetime.now().isoformat(),
                    cv_improvement=experiment_results.get('improvement'),
                )
            ],
        )

    @staticmethod
    def _format_previous_experiments(experiments: List[Dict[str, Any]]) -> str:
        """Format previous experiments for context."""
        if not experiments:
            return "No previous experiments."

        lines = []
        for i, exp in enumerate(experiments[-5:]):  # Last 5
            lines.append(f"{i+1}. {exp.get('hypothesis', 'Unknown')[:60]}...")
            lines.append(f"   CV: {exp.get('cv_score', 'N/A')}, Success: {exp.get('is_successful')}")
        return "\n".join(lines)

    @staticmethod
    def _format_experiment_history(experiments: List[Dict[str, Any]]) -> str:
        """Format full experiment history."""
        if not experiments:
            return "No experiments."

        lines = []
        for exp in experiments:
            lines.append(f"### {exp.get('id', 'Unknown')}")
            lines.append(f"Hypothesis: {exp.get('hypothesis', 'Unknown')}")
            lines.append(f"CV Score: {exp.get('cv_score', 'N/A')}")
            lines.append(f"Success: {exp.get('is_successful', False)}")
            lines.append("")
        return "\n".join(lines)

    def save_retrospective(
        self,
        competition: str,
        retrospective: ReflectionResult,
        output_path: Path,
    ) -> None:
        """Save retrospective to markdown file.

        Args:
            competition: Competition slug
            retrospective: Retrospective result
            output_path: Path to write file
        """
        content = f"""# Retrospective: {competition}

**Date:** {datetime.now().isoformat()}
**Success Rating:** {retrospective.success_rating:.2f}/1.0

## Summary

{retrospective.summary}

## Key Insights

{chr(10).join(f"- {insight}" for insight in retrospective.key_insights)}

## Proposed Knowledge Updates

### New Techniques
{chr(10).join(str(t) for t in retrospective.proposed_new_techniques) or "None proposed"}

### Playbook Updates
{chr(10).join(str(u) for u in retrospective.proposed_playbook_updates) or "None proposed"}

### Skill Updates
{chr(10).join(str(u) for u in retrospective.proposed_skill_updates) or "None proposed"}

---
*Auto-generated by Kaggle Agent Reflection Engine*
"""
        output_path.write_text(content)
