"""Runtime lifecycle contracts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["IRuntimeComponent"]


@runtime_checkable
class IRuntimeComponent(Protocol):
    """Protocol for components with an explicit runtime lifecycle."""

    def start(self) -> None:
        """Start the component."""
        ...

    def stop(self) -> None:
        """Stop the component."""
        ...
