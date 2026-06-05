"""Button mapper protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neuroforge.agents.action import AgentAction

__all__ = ["IButtonMapper"]


@runtime_checkable
class IButtonMapper(Protocol):
    """Maps generic actions to controller button state."""

    def map_buttons(self, action: AgentAction) -> object:
        """Return implementation-defined button state."""
        ...
