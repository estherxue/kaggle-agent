"""Cursor IDE agent provider via file handoff protocol.

Writes task prompts to disk and reads responses written by the Cursor coding agent.
No external API key required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .base import ChatMessage, ChatResponse, LLMProvider


class CursorTaskPending(Exception):
    """Raised when a task file was written and response is not yet available."""

    def __init__(self, task_path: Path, response_path: Path) -> None:
        self.task_path = task_path
        self.response_path = response_path
        super().__init__(
            f"Cursor agent task pending: {task_path}\n"
            f"Write response to: {response_path}"
        )


class CursorAgentProvider(LLMProvider):
    """Delegate LLM work to the Cursor coding agent via file handoff."""

    COUNTER_FILE = ".task_counter"

    def __init__(
        self,
        name: str,
        model: str,
        tasks_dir: Path,
        api_key: str = "",
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
        self.tasks_dir = Path(tasks_dir)

    def _load_counter(self) -> int:
        counter_path = self.tasks_dir / self.COUNTER_FILE
        if counter_path.exists():
            return int(json.loads(counter_path.read_text()).get("next", 0))
        return 0

    def _save_counter(self, value: int) -> None:
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        counter_path = self.tasks_dir / self.COUNTER_FILE
        counter_path.write_text(json.dumps({"next": value}, indent=2))

    def _format_messages(self, messages: List[ChatMessage]) -> str:
        parts = []
        for msg in messages:
            parts.append(f"## {msg.role.upper()}\n\n{msg.content}\n")
        return "\n".join(parts)

    def _next_paths(self) -> tuple[Path, Path, int]:
        task_id = self._load_counter()
        task_path = self.tasks_dir / f"task_{task_id:04d}.md"
        response_path = self.tasks_dir / f"task_{task_id:04d}_response.md"
        return task_path, response_path, task_id

    def get_pending_task(self) -> Optional[tuple[Path, Path]]:
        """Return (task_path, response_path) if a task awaits response."""
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        task_id = self._load_counter()
        if task_id == 0:
            return None
        pending_id = task_id - 1
        task_path = self.tasks_dir / f"task_{pending_id:04d}.md"
        response_path = self.tasks_dir / f"task_{pending_id:04d}_response.md"
        if task_path.exists() and not response_path.exists():
            return task_path, response_path
        return None

    def chat(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse:
        """Read response file if present; otherwise write task and pause."""
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        task_path, response_path, task_id = self._next_paths()

        if response_path.exists():
            content = response_path.read_text()
            self._save_counter(task_id + 1)
            tokens = self.count_tokens(messages) + len(content) // 4
            return ChatResponse(
                content=content,
                prompt_tokens=tokens // 2,
                completion_tokens=tokens // 2,
                total_tokens=tokens,
                model=self.model,
                cost_usd=0.0,
            )

        task_body = (
            "# Cursor Agent Task\n\n"
            f"**Model role context**: respond in markdown. Include runnable Python in a "
            f"```python code block when code is requested.\n\n"
            f"**Expected response file**: `{response_path.name}`\n\n"
            f"**Parameters**: max_tokens={max_tokens}, temperature={temperature}\n\n"
            "---\n\n"
            f"{self._format_messages(messages)}\n"
        )
        task_path.write_text(task_body)
        raise CursorTaskPending(task_path, response_path)

    def count_tokens(self, messages: List[ChatMessage]) -> int:
        text = "".join(m.content for m in messages)
        return max(1, len(text) // 4)
