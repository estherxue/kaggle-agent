"""Placeholder LLM provider using direct API calls.

This is a simple implementation that uses the OpenAI SDK directly.
It can be replaced with litellm or other providers later.
"""

import json
from typing import List

from .base import ChatMessage, ChatResponse, LLMProvider


class PlaceholderProvider(LLMProvider):
    """Placeholder provider using direct OpenAI API calls.

    This provider uses the requests library to make direct API calls to OpenAI.
    It's intentionally simple - litellm or other adapters can be added later
    for more advanced features.
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
            base_url=base_url or "https://api.openai.com/v1",
            cost_per_1k_prompt=cost_per_1k_prompt,
            cost_per_1k_completion=cost_per_1k_completion,
        )
        self._session = None

    def _get_session(self):
        """Get or create requests session."""
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def chat(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse:
        """Send chat request to OpenAI API."""
        import requests

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = self._get_session().post(url, headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        cost = self.estimate_cost(prompt_tokens, completion_tokens)

        return ChatResponse(
            content=choice["message"]["content"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=usage.get("total_tokens", prompt_tokens + completion_tokens),
            model=data.get("model", self.model),
            cost_usd=cost,
        )

    def count_tokens(self, messages: List[ChatMessage]) -> int:
        """Estimate token count using a simple approximation.

        This is a rough estimate - 4 chars ~= 1 token for English text.
        For production, use tiktoken or similar.
        """
        total_chars = sum(len(m.content) for m in messages)
        # Rough estimate: 4 chars per token
        return total_chars // 4


class MockProvider(LLMProvider):
    """Mock provider for testing.

    Returns predetermined responses without making API calls.
    """

    def __init__(
        self,
        name: str = "mock",
        model: str = "mock-model",
        responses: dict | None = None,
    ):
        # Mock provider doesn't need API key
        super().__init__(
            name=name,
            model=model,
            api_key="mock-key",
            cost_per_1k_prompt=0.0,
            cost_per_1k_completion=0.0,
        )
        self.responses = responses or {}
        self.call_history: List[List[ChatMessage]] = []

    def chat(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse:
        """Return mock response."""
        self.call_history.append(messages)

        # Try to find a matching response based on prompt content
        prompt = messages[-1].content if messages else ""
        content = self.responses.get(prompt, "Mock response")

        # Estimate tokens
        prompt_tokens = self.count_tokens(messages)
        completion_tokens = len(content) // 4

        return ChatResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=self.model,
            cost_usd=0.0,
        )

    def count_tokens(self, messages: List[ChatMessage]) -> int:
        """Simple token count estimate."""
        total_chars = sum(len(m.content) for m in messages)
        return total_chars // 4
