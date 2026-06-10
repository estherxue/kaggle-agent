"""LiteLLM adapter for Kaggle Agent.

This module provides an adapter for the litellm library, which supports
many providers (OpenAI, Anthropic, Cohere, OpenRouter, etc.) through
a unified interface.

This is currently a placeholder - implement when litellm support is needed.
"""

from typing import List

from .base import ChatMessage, ChatResponse, LLMProvider


class LiteLLMProvider(LLMProvider):
    """LiteLLM provider adapter.

    This class can be implemented to use the litellm library for
    unified access to many LLM providers.

    To implement:
    1. Add litellm to dependencies
    2. Import litellm
    3. Use litellm.completion() in the chat method
    4. Map responses to ChatResponse format
    """

    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        cost_per_1k_prompt: float = 0.0,
        cost_per_1k_completion: float = 0.0,
    ):
        super().__init__(
            name=name,
            model=model,
            api_key=api_key,
            base_url=base_url,
            cost_per_1k_prompt=cost_per_1k_prompt,
            cost_per_1k_completion=cost_per_1k_completion,
        )
        # TODO: Import and configure litellm
        # import litellm
        # litellm.api_key = api_key

    def chat(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse:
        """Send chat request using litellm.

        Placeholder - implement when litellm support is needed.
        """
        raise NotImplementedError(
            "LiteLLM provider not yet implemented. "
            "Use PlaceholderProvider for now, or implement this class."
        )

    def count_tokens(self, messages: List[ChatMessage]) -> int:
        """Count tokens using litellm's tokenizer."""
        # TODO: Use litellm.token_counter()
        total_chars = sum(len(m.content) for m in messages)
        return total_chars // 4


class OpenRouterProvider(LiteLLMProvider):
    """OpenRouter-specific provider (extends LiteLLM).

    OpenRouter provides unified access to many models including Claude,
    GPT-4, Llama, etc. through a single API.
    """

    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        cost_per_1k_prompt: float = 0.0,
        cost_per_1k_completion: float = 0.0,
    ):
        # OpenRouter uses its own base URL
        super().__init__(
            name=name,
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            cost_per_1k_prompt=cost_per_1k_prompt,
            cost_per_1k_completion=cost_per_1k_completion,
        )


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider.

    For running local models like Llama, Mistral, etc.
    """

    def __init__(
        self,
        name: str,
        model: str,
        api_key: str = "ollama",  # Ollama doesn't need API key
        base_url: str = "http://localhost:11434",
        cost_per_1k_prompt: float = 0.0,
        cost_per_1k_completion: float = 0.0,
    ):
        super().__init__(
            name=name,
            model=model,
            api_key=api_key,
            base_url=base_url,
            cost_per_1k_prompt=cost_per_1k_prompt,
            cost_per_1k_completion=cost_per_1k_completion,
        )

    def chat(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse:
        """Send chat request to local Ollama server.

        Placeholder - implement when local LLM support is needed.
        """
        raise NotImplementedError(
            "Ollama provider not yet implemented. "
            "Use PlaceholderProvider for now, or implement this class."
        )

    def count_tokens(self, messages: List[ChatMessage]) -> int:
        """Count tokens - Ollama may not provide token counts."""
        # Local models are free, so rough estimate is fine
        total_chars = sum(len(m.content) for m in messages)
        return total_chars // 4
