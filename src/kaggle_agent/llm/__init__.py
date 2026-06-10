"""LLM layer for Kaggle Agent.

Provides unified interface to multiple LLM providers with role-based routing
and cost tracking.

Usage:
    from kaggle_agent.llm import LLMRouter, ChatMessage

    router = LLMRouter.from_config(config)
    response = router.chat(
        role="coder",
        messages=[ChatMessage(role="user", content="Write a Python function...")]
    )
    print(f"Cost: ${response.cost_usd:.4f}")
"""

from typing import Dict, List, Optional

from ..config import Config, LLMProviderConfig
from .base import ChatMessage, ChatResponse, LLMProvider
from .placeholder import MockProvider, PlaceholderProvider

# Provider type registry
PROVIDER_TYPES = {
    "placeholder": PlaceholderProvider,
    "mock": MockProvider,
    # Future providers can be registered here:
    # "litellm": LiteLLMProvider,
    # "openrouter": OpenRouterProvider,
    # "ollama": OllamaProvider,
}


class LLMRouter:
    """Routes LLM requests to appropriate providers based on role.

    Supports multiple providers and tracks costs across all calls.
    """

    def __init__(
        self,
        providers: Dict[str, LLMProvider],
        role_mapping: Dict[str, str],
        default_params: dict,
    ):
        self.providers = providers
        self.role_mapping = role_mapping
        self.default_params = default_params
        self._cost_tracker: Dict[str, float] = {}  # provider_name -> total cost
        self._token_tracker: Dict[str, Dict[str, int]] = {}  # provider_name -> token counts

    @classmethod
    def from_config(cls, config: Config) -> "LLMRouter":
        """Create router from configuration.

        Args:
            config: Loaded configuration

        Returns:
            Configured LLMRouter
        """
        providers: Dict[str, LLMProvider] = {}

        for provider_cfg in config.llm.providers:
            provider = cls._create_provider(provider_cfg, config)
            if provider:
                providers[provider_cfg.name] = provider

        role_mapping = {
            "planner": config.llm.roles.planner,
            "coder": config.llm.roles.coder,
            "reviewer": config.llm.roles.reviewer,
            "summarizer": config.llm.roles.summarizer,
        }

        default_params = {
            "temperature": config.llm.default.temperature,
            "max_tokens": config.llm.default.max_tokens,
        }

        return cls(providers, role_mapping, default_params)

    @staticmethod
    def _create_provider(
        provider_cfg: LLMProviderConfig, config: Config
    ) -> Optional[LLMProvider]:
        """Create a provider instance from configuration.

        Args:
            provider_cfg: Provider configuration
            config: Global config (for API key lookup)

        Returns:
            Provider instance or None if creation fails
        """
        provider_class = PROVIDER_TYPES.get(provider_cfg.type)
        if provider_class is None:
            raise ValueError(f"Unknown provider type: {provider_cfg.type}")

        # Get API key from environment
        api_key = config.get_api_key(provider_cfg.name) or "dummy-key"

        return provider_class(
            name=provider_cfg.name,
            model=provider_cfg.model,
            api_key=api_key,
            base_url=provider_cfg.base_url,
            cost_per_1k_prompt=provider_cfg.cost_per_1k_prompt,
            cost_per_1k_completion=provider_cfg.cost_per_1k_completion,
        )

    def chat(
        self,
        messages: List[ChatMessage],
        role: str = "coder",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> ChatResponse:
        """Send chat request for a specific role.

        Args:
            messages: List of chat messages
            role: Agent role (planner, coder, reviewer, summarizer)
            max_tokens: Override default max tokens
            temperature: Override default temperature

        Returns:
            ChatResponse from the LLM
        """
        provider_name = self.role_mapping.get(role)
        if not provider_name:
            raise ValueError(f"Unknown role: {role}")

        provider = self.providers.get(provider_name)
        if not provider:
            raise ValueError(f"Provider not found: {provider_name}")

        # Use default params if not overridden
        max_tokens = max_tokens or self.default_params["max_tokens"]
        temperature = temperature or self.default_params["temperature"]

        # Make the call
        response = provider.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Track cost and tokens
        self._track_usage(provider_name, response)

        return response

    def _track_usage(self, provider_name: str, response: ChatResponse) -> None:
        """Track token usage and cost for a provider."""
        # Update cost tracker
        self._cost_tracker[provider_name] = (
            self._cost_tracker.get(provider_name, 0.0) + response.cost_usd
        )

        # Update token tracker
        if provider_name not in self._token_tracker:
            self._token_tracker[provider_name] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        self._token_tracker[provider_name]["prompt_tokens"] += response.prompt_tokens
        self._token_tracker[provider_name]["completion_tokens"] += response.completion_tokens
        self._token_tracker[provider_name]["total_tokens"] += response.total_tokens

    def get_cost_summary(self) -> Dict[str, Dict]:
        """Get summary of costs and token usage.

        Returns:
            Dict mapping provider names to usage statistics
        """
        summary = {}
        for name in self.providers.keys():
            summary[name] = {
                "cost_usd": self._cost_tracker.get(name, 0.0),
                "tokens": self._token_tracker.get(
                    name,
                    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                ),
            }
        return summary

    def get_total_cost(self) -> float:
        """Get total cost across all providers."""
        return sum(self._cost_tracker.values())

    def reset_tracking(self) -> None:
        """Reset cost and token tracking."""
        self._cost_tracker.clear()
        self._token_tracker.clear()


__all__ = [
    "ChatMessage",
    "ChatResponse",
    "LLMProvider",
    "LLMRouter",
    "PlaceholderProvider",
    "MockProvider",
]
