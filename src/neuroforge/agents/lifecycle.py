"""Agent lifecycle protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["IAgentLifecycle"]


@runtime_checkable
class IAgentLifecycle(Protocol):
    """Lifecycle hooks for reusable agents."""

    def reset(self) -> None:
        """Reset any episode-local state."""
        ...

    def close(self) -> None:
        """Release resources held by the agent."""
        ...
