"""Agent memory protocols."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["IAgentMemory"]


@runtime_checkable
class IAgentMemory(Protocol):
    """Optional state/history mechanism used by an agent or brain."""

    def reset(self) -> None:
        """Clear memory state."""
        ...

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe memory snapshot."""
        ...
