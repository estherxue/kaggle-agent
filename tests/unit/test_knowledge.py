"""Tests for knowledge system."""

import pytest
import tempfile
from pathlib import Path

from kaggle_agent.knowledge import (
    PlaybookManager,
    TechniqueCard,
    ValidationRecord,
    SkillManager,
    Skill,
    SkillMetadata,
)


def test_playbook_manager_load():
    """Test loading playbooks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create playbook files
        knowledge_path = Path(tmpdir)
        playbooks_path = knowledge_path / "playbooks"
        playbooks_path.mkdir(parents=True)

        (playbooks_path / "general.md").write_text("# General\nTest content")
        (playbooks_path / "tabular.md").write_text("# Tabular\nTabular content")

        manager = PlaybookManager(knowledge_path)

        general = manager.load_playbook("general")
        assert "General" in general

        tabular = manager.load_playbook("tabular")
        assert "Tabular" in tabular

        all_playbooks = manager.load_all_playbooks()
        assert "general" in all_playbooks
        assert "tabular" in all_playbooks


def test_technique_card_persistence():
    """Test technique card saving and loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        knowledge_path = Path(tmpdir)
        techniques_path = knowledge_path / "playbooks" / "techniques"
        techniques_path.mkdir(parents=True)

        # Create a technique card
        card = TechniqueCard(
            name="Test Technique",
            applicable_types=["tabular"],
            applicable_conditions="Use when you have numeric features",
            usage_code="def transform(x): return x * 2",
            validations=[
                ValidationRecord(
                    competition="test",
                    date="2024-01-01",
                    cv_improvement=0.01,
                )
            ],
        )

        # Save
        path = techniques_path / "test-technique.md"
        card.source_file = path
        card.save()

        # Load
        loaded = TechniqueCard.from_markdown(path)
        assert loaded.name == "Test Technique"
        assert loaded.applicable_types == ["tabular"]
        assert len(loaded.validations) == 1
        assert loaded.validations[0].cv_improvement == 0.01


def test_skill_manager():
    """Test skill management."""
    with tempfile.TemporaryDirectory() as tmpdir:
        knowledge_path = Path(tmpdir)
        skills_path = knowledge_path / "skills"
        (skills_path / "feature_engineering").mkdir(parents=True)

        # Create a skill
        module_path = skills_path / "feature_engineering" / "test_skill.py"
        module_path.write_text('"""Test skill."""\n\ndef transform(x):\n    return x * 2\n')

        # Create metadata
        metadata_path = module_path.with_suffix(".yaml")
        metadata = SkillMetadata(
            name="test_skill",
            description="Test skill",
            applicable_types=["tabular"],
            applicable_conditions="Test",
            inputs=["x"],
            outputs=["result"],
        )
        metadata_path.write_text(metadata.to_yaml())

        manager = SkillManager(knowledge_path)

        # Load skill
        skill = manager.load_skill("test_skill", category="feature_engineering")
        assert skill is not None
        assert skill.name == "test_skill"
        assert skill.metadata.description == "Test skill"


def test_skill_listing():
    """Test listing skills."""
    with tempfile.TemporaryDirectory() as tmpdir:
        knowledge_path = Path(tmpdir)
        skills_path = knowledge_path / "skills"

        # Create skills in different categories
        for cat in ["feature_engineering", "models"]:
            (skills_path / cat).mkdir(parents=True)
            (skills_path / cat / "skill1.py").write_text("# skill")
            (skills_path / cat / "skill1.yaml").write_text(
                "name: skill1\ndescription: Test\n"
                "applicable_types: [tabular]\napplicable_conditions: Test\n"
                "inputs: []\noutputs: []"
            )

        manager = SkillManager(knowledge_path)

        # List all skills
        skills = manager.list_skills()
        assert len(skills) == 2

        # List by category
        fe_skills = manager.list_skills(category="feature_engineering")
        assert len(fe_skills) == 1


def test_skill_validation_update():
    """Test updating skill validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        knowledge_path = Path(tmpdir)
        skills_path = knowledge_path / "skills" / "feature_engineering"
        skills_path.mkdir(parents=True)

        # Create skill
        module_path = skills_path / "test_skill.py"
        module_path.write_text('"""Test."""')

        metadata = SkillMetadata(
            name="test_skill",
            description="Test",
            applicable_types=["tabular"],
            applicable_conditions="Test",
            inputs=[],
            outputs=[],
        )
        metadata_path = module_path.with_suffix(".yaml")
        metadata_path.write_text(metadata.to_yaml())

        skill = Skill.from_file(module_path, metadata_path)

        # Initially not validated
        assert skill.is_validated is False

        # Update validation
        skill.update_validation("comp1", 0.01)
        skill.update_validation("comp2", 0.02)
        skill.update_validation("comp3", 0.015)

        # Now validated (>= 3 competitions)
        assert skill.is_validated is True
        assert skill.metadata.avg_cv_improvement is not None
