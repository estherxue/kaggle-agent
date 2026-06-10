"""Configuration loading and management."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    name: str
    type: str  # "placeholder", "litellm", "openrouter", "ollama"
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    model: str
    cost_per_1k_prompt: float = 0.0
    cost_per_1k_completion: float = 0.0


class LLMRolesConfig(BaseModel):
    """Mapping of agent roles to provider names."""

    planner: str = "openai"
    coder: str = "openai"
    reviewer: str = "openai"
    summarizer: str = "openai"


class LLMDefaultConfig(BaseModel):
    """Default parameters for LLM calls."""

    temperature: float = 0.7
    max_tokens: int = 4096


class LLMConfig(BaseModel):
    """Complete LLM configuration."""

    providers: List[LLMProviderConfig]
    roles: LLMRolesConfig
    default: LLMDefaultConfig = Field(default_factory=LLMDefaultConfig)


class BudgetConfig(BaseModel):
    """Budget and safety limits."""

    max_experiments_per_competition: int = 50
    max_llm_cost_usd: float = 50.0
    max_execution_time_per_exp_sec: int = 600
    max_submissions_per_day: int = 5
    min_cv_improvement_threshold: float = 0.0


class PathsConfig(BaseModel):
    """Path configuration."""

    knowledge: str = "knowledge"
    competitions: str = "competitions"


class ExecutionConfig(BaseModel):
    """Code execution settings."""

    timeout_sec: int = 300
    allow_network: bool = False
    python_path: Optional[str] = None


class KaggleConfig(BaseModel):
    """Kaggle-specific settings."""

    auto_submit: bool = True
    dry_run: bool = False


class Config(BaseModel):
    """Complete Kaggle Agent configuration."""

    llm: LLMConfig
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    kaggle: KaggleConfig = Field(default_factory=KaggleConfig)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        """Load configuration from YAML file.

        Args:
            path: Path to config file. If None, looks for config.yaml in current dir.

        Returns:
            Parsed Config object.
        """
        if path is None:
            path = Path("config.yaml")

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls.model_validate(data)

    def get_provider(self, name: str) -> Optional[LLMProviderConfig]:
        """Get provider configuration by name."""
        for provider in self.llm.providers:
            if provider.name == name:
                return provider
        return None

    def get_api_key(self, provider_name: str) -> Optional[str]:
        """Get API key for a provider from environment."""
        provider = self.get_provider(provider_name)
        if provider is None or provider.api_key_env is None:
            return None
        return os.environ.get(provider.api_key_env)

    def resolve_path(self, path_type: str) -> Path:
        """Resolve a configured path to absolute path."""
        if path_type == "knowledge":
            return Path(self.paths.knowledge).resolve()
        elif path_type == "competitions":
            return Path(self.paths.competitions).resolve()
        else:
            raise ValueError(f"Unknown path type: {path_type}")
