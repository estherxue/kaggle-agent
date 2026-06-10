"""Tests for configuration module."""

import pytest
import tempfile
from pathlib import Path

from kaggle_agent.config import Config


def test_config_load(tmp_path):
    """Test loading configuration from file."""
    config_content = """
llm:
  providers:
    - name: "test"
      type: "placeholder"
      api_key_env: "TEST_API_KEY"
      model: "gpt-4o-mini"
      cost_per_1k_prompt: 0.15
      cost_per_1k_completion: 0.60
  roles:
    planner: "test"
    coder: "test"
    reviewer: "test"
    summarizer: "test"
budget:
  max_experiments_per_competition: 10
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    config = Config.load(config_file)

    assert len(config.llm.providers) == 1
    assert config.llm.providers[0].name == "test"
    assert config.llm.roles.planner == "test"
    assert config.budget.max_experiments_per_competition == 10


def test_get_provider():
    """Test getting provider by name."""
    from kaggle_agent.config import LLMProviderConfig, LLMConfig

    provider = LLMProviderConfig(
        name="test",
        type="placeholder",
        api_key_env="TEST_KEY",
        model="gpt-4",
        cost_per_1k_prompt=0.0,
        cost_per_1k_completion=0.0,
    )

    config = Config(
        llm=LLMConfig(
            providers=[provider],
            roles={},
        )
    )

    found = config.get_provider("test")
    assert found is not None
    assert found.name == "test"

    not_found = config.get_provider("nonexistent")
    assert not_found is None


def test_resolve_path(tmp_path):
    """Test path resolution."""
    from kaggle_agent.config import LLMConfig, LLMRolesConfig, LLMDefaultConfig
    config = Config(
        llm=LLMConfig(
            providers=[],
            roles=LLMRolesConfig(planner="", coder="", reviewer="", summarizer=""),
            default=LLMDefaultConfig(),
        ),
        paths={"knowledge": str(tmp_path / "knowledge"), "competitions": str(tmp_path / "comps")},
    )

    knowledge_path = config.resolve_path("knowledge")
    assert knowledge_path == tmp_path / "knowledge"

    competitions_path = config.resolve_path("competitions")
    assert competitions_path == tmp_path / "comps"
