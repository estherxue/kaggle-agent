"""Quick test for LLM layer."""

import os
import sys

from ..config import Config
from . import LLMRouter, ChatMessage


def main():
    """Test LLM layer with mock provider."""
    # Create a minimal config for testing
    config = Config(
        llm={
            "providers": [
                {
                    "name": "mock",
                    "type": "mock",
                    "model": "mock-model",
                    "cost_per_1k_prompt": 0.0,
                    "cost_per_1k_completion": 0.0,
                }
            ],
            "roles": {
                "planner": "mock",
                "coder": "mock",
                "reviewer": "mock",
                "summarizer": "mock",
            },
            "default": {"temperature": 0.7, "max_tokens": 1024},
        }
    )

    router = LLMRouter.from_config(config)

    # Test basic chat
    print("Testing LLM Router...")
    messages = [ChatMessage(role="user", content="Hello, are you working?")]
    response = router.chat(messages, role="coder")

    print(f"Response: {response.content}")
    print(f"Tokens: {response.total_tokens}")
    print(f"Cost: ${response.cost_usd:.4f}")

    # Test cost tracking
    summary = router.get_cost_summary()
    print(f"\nCost Summary: {summary}")
    print(f"Total Cost: ${router.get_total_cost():.4f}")

    print("\nLLM layer test passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
