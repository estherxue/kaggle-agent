"""Tests for LLM layer."""

import pytest
from kaggle_agent.llm import ChatMessage, LLMRouter, MockProvider
from kaggle_agent.config import Config, LLMConfig, LLMRolesConfig, LLMProviderConfig


def test_chat_message():
    """Test ChatMessage dataclass."""
    msg = ChatMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_mock_provider():
    """Test MockProvider."""
    provider = MockProvider(
        name="mock",
        model="mock-model",
        responses={"test": "Test response"},
    )

    response = provider.chat(
        messages=[ChatMessage(role="user", content="test")],
    )

    assert response.content == "Test response"
    assert response.model == "mock-model"
    assert response.cost_usd == 0.0

    # Test history tracking
    assert len(provider.call_history) == 1
    assert provider.call_history[0][0].content == "test"


def test_llm_router_role_mapping():
    """Test LLMRouter role-based routing."""
    mock_provider = MockProvider(name="test", responses={"code": "def test(): pass"})

    router = LLMRouter(
        providers={"test": mock_provider},
        role_mapping={
            "planner": "test",
            "coder": "test",
            "reviewer": "test",
            "summarizer": "test",
        },
        default_params={"temperature": 0.7, "max_tokens": 1024},
    )

    response = router.chat(
        role="coder",
        messages=[ChatMessage(role="user", content="code")],
    )

    assert response.content == "def test(): pass"


def test_cost_tracking():
    """Test cost tracking across multiple calls."""
    mock_provider = MockProvider(name="test")

    router = LLMRouter(
        providers={"test": mock_provider},
        role_mapping={"coder": "test"},
        default_params={"temperature": 0.7, "max_tokens": 1024},
    )

    # Make multiple calls
    for _ in range(3):
        router.chat(
            role="coder",
            messages=[ChatMessage(role="user", content="test")],
        )

    summary = router.get_cost_summary()
    assert "test" in summary
    assert summary["test"]["cost_usd"] == 0.0  # Mock is free

    total = router.get_total_cost()
    assert total == 0.0


def test_router_from_config():
    """Test creating router from config with placeholder provider."""
    config = Config(
        llm=LLMConfig(
            providers=[
                LLMProviderConfig(
                    name="placeholder",
                    type="placeholder",
                    api_key_env=None,
                    model="gpt-4o-mini",
                )
            ],
            roles=LLMRolesConfig(
                planner="placeholder",
                coder="placeholder",
                reviewer="placeholder",
                summarizer="placeholder",
            ),
            default={"temperature": 0.7, "max_tokens": 1024},
        )
    )

    router = LLMRouter.from_config(config)
    assert "placeholder" in router.providers
    assert router.role_mapping["planner"] == "placeholder"


def test_unknown_role_raises():
    """Test that unknown role raises error."""
    router = LLMRouter(
        providers={},
        role_mapping={},
        default_params={},
    )

    with pytest.raises(ValueError, match="Unknown role"):
        router.chat(role="unknown", messages=[])
