"""Interaction module for Kaggle Agent.

Handles:
- Guidance queue (user input during competition runs)
- Status reporting
- Safe stopping mechanism
"""

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Guidance:
    """A single guidance entry."""

    id: str
    timestamp: datetime
    content: str
    source: str  # "user", "system", "auto"
    processed: bool = False
    adopted: Optional[bool] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "source": self.source,
            "processed": self.processed,
            "adopted": self.adopted,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Guidance":
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            content=data["content"],
            source=data["source"],
            processed=data.get("processed", False),
            adopted=data.get("adopted"),
            notes=data.get("notes", ""),
        )


class GuidanceQueue:
    """Thread-safe guidance queue for a competition.

    File-based queue for persistence and cross-process communication.
    """

    def __init__(self, competition_path: Path):
        """Initialize guidance queue.

        Args:
            competition_path: Path to competition directory
        """
        self.queue_path = competition_path / "guidance_queue.json"
        self.lock = threading.Lock()
        self._counter = 0

        # Load or initialize
        self._load()

    def _load(self) -> None:
        """Load queue from disk."""
        if self.queue_path.exists():
            with open(self.queue_path, "r") as f:
                data = json.load(f)
            self._pending = [Guidance.from_dict(g) for g in data.get("pending", [])]
            self._processed = [Guidance.from_dict(g) for g in data.get("processed", [])]
            self._counter = data.get("counter", 0)
        else:
            self._pending: List[Guidance] = []
            self._processed: List[Guidance] = []
            self._save()

    def _save(self) -> None:
        """Save queue to disk."""
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.queue_path, "w") as f:
            json.dump(
                {
                    "pending": [g.to_dict() for g in self._pending],
                    "processed": [g.to_dict() for g in self._processed],
                    "counter": self._counter,
                },
                f,
                indent=2,
            )

    def add(self, content: str, source: str = "user") -> str:
        """Add guidance to queue.

        Args:
            content: Guidance text
            source: Source of guidance (user, system, auto)

        Returns:
            Guidance ID
        """
        with self.lock:
            # Reload first so a counter bumped by another process is respected
            # and we append to the latest on-disk pending list.
            self._load()
            self._counter += 1
            guidance = Guidance(
                id=f"g{self._counter:03d}",
                timestamp=datetime.now(),
                content=content,
                source=source,
            )
            self._pending.append(guidance)
            self._save()
            return guidance.id

    def get_pending(self) -> List[Guidance]:
        """Get all pending guidance.

        Returns:
            List of pending Guidance objects
        """
        with self.lock:
            self._load()
            return self._pending.copy()

    def consume(self) -> Optional[Guidance]:
        """Consume one guidance from queue.

        Reloads from disk first so guidance written by another process (e.g.
        ``kagent guide``) after this object was constructed is visible. The
        consumed item is removed from the on-disk pending list immediately.

        Returns:
            Guidance or None if empty
        """
        with self.lock:
            self._load()
            if not self._pending:
                return None
            guidance = self._pending.pop(0)
            # Move it into the processed list immediately and persist, so a
            # crash before mark_processed() does not replay the same guidance,
            # and so a later mark_processed(id, ...) can locate it to record
            # whether it was ultimately adopted.
            guidance.processed = True
            self._processed.append(guidance)
            self._save()
            return guidance

    def mark_processed(
        self,
        guidance_id: str,
        adopted: bool,
        notes: str = "",
    ) -> None:
        """Mark guidance as processed.

        Args:
            guidance_id: ID of guidance
            adopted: Whether guidance was adopted
            notes: Additional notes
        """
        with self.lock:
            self._load()
            # Find in pending (shouldn't happen, but just in case)
            for i, g in enumerate(self._pending):
                if g.id == guidance_id:
                    g = self._pending.pop(i)
                    g.processed = True
                    g.adopted = adopted
                    g.notes = notes
                    self._processed.append(g)
                    self._save()
                    return

            # If not in pending, check processed
            for g in self._processed:
                if g.id == guidance_id:
                    g.adopted = adopted
                    g.notes = notes
                    self._save()
                    return

    def get_history(self, limit: int = 50) -> List[Guidance]:
        """Get processed guidance history.

        Args:
            limit: Maximum number to return

        Returns:
            List of processed Guidance objects
        """
        with self.lock:
            return self._processed[-limit:]

    def clear(self) -> None:
        """Clear all guidance (use with caution)."""
        with self.lock:
            self._pending.clear()
            self._processed.clear()
            self._save()

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self.lock:
            self._load()
            processed = self._processed
            adopted_count = sum(1 for g in processed if g.adopted)
            rejected_count = sum(1 for g in processed if g.adopted is False)

            return {
                "pending_count": len(self._pending),
                "processed_count": len(processed),
                "adopted_count": adopted_count,
                "rejected_count": rejected_count,
                "adoption_rate": adopted_count / len(processed) if processed else 0.0,
            }


class CompetitionInterface:
    """Interface for interacting with a running competition.

    Provides methods for:
    - Adding guidance
    - Getting status
    - Requesting stop
    """

    def __init__(self, competition_slug: str, competitions_path: Path):
        """Initialize interface.

        Args:
            competition_slug: Competition slug
            competitions_path: Path to competitions directory
        """
        self.competition = competition_slug
        self.comp_path = competitions_path / competition_slug
        self.guidance = GuidanceQueue(self.comp_path)

    def add_guidance(self, content: str, source: str = "user") -> str:
        """Add guidance.

        Args:
            content: Guidance text
            source: Source of guidance

        Returns:
            Guidance ID
        """
        return self.guidance.add(content, source)

    def get_status(self) -> Dict[str, Any]:
        """Get competition status.

        Returns:
            Status dictionary
        """
        # Load state from orchestrator
        state_path = self.comp_path / "state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                state = json.load(f)
        else:
            state = {"phase": "UNKNOWN", "competition": self.competition}

        # Add guidance stats
        stats = self.guidance.get_stats()

        return {
            "competition": self.competition,
            "phase": state.get("phase", "UNKNOWN"),
            "experiment_count": state.get("experiment_count", 0),
            "best_cv_score": state.get("best_cv_score"),
            "llm_cost_usd": state.get("total_llm_cost", 0.0),
            "guidance_pending": stats["pending_count"],
            "guidance_total": stats["processed_count"],
            "guidance_adoption_rate": f"{stats['adoption_rate']:.1%}",
            "notes": state.get("notes", [])[-5:],
        }

    def request_stop(self) -> bool:
        """Request graceful stop.

        Creates a stop signal file that orchestrator checks.

        Returns:
            True if signal created
        """
        stop_path = self.comp_path / "STOP_REQUESTED"
        stop_path.write_text(datetime.now().isoformat())
        return True

    def get_stop_requested(self) -> bool:
        """Check if stop has been requested."""
        stop_path = self.comp_path / "STOP_REQUESTED"
        return stop_path.exists()

    def clear_stop_request(self) -> None:
        """Clear stop request."""
        stop_path = self.comp_path / "STOP_REQUESTED"
        if stop_path.exists():
            stop_path.unlink()

    def get_guidance_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get guidance history.

        Args:
            limit: Maximum number to return

        Returns:
            List of guidance dicts
        """
        history = self.guidance.get_history(limit)
        return [g.to_dict() for g in history]
