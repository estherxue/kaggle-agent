"""Base classes for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class ChatMessage:
    """A single message in a chat conversation."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatResponse:
    """Response from an LLM provider."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    cost_usd: float


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All provider implementations must inherit from this class and implement
    the abstract methods. This allows for easy swapping between different
    LLM backends (OpenAI, Anthropic, OpenRouter, Ollama, etc.).
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
        self.name = name
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.cost_per_1k_prompt = cost_per_1k_prompt
        self.cost_per_1k_completion = cost_per_1k_completion

    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse:
        """Send a chat request to the LLM.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature

        Returns:
            ChatResponse with content and metadata
        """
        pass

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate the cost of a request.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Estimated cost in USD
        """
        return (
            (prompt_tokens / 1000.0) * self.cost_per_1k_prompt
            + (completion_tokens / 1000.0) * self.cost_per_1k_completion
        )

    @abstractmethod
    def count_tokens(self, messages: List[ChatMessage]) -> int:
        """Count tokens in messages (for cost estimation).

        Args:
            messages: List of chat messages

        Returns:
            Estimated token count
        """
        pass
