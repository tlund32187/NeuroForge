"""Agent protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neuroforge.agents.action import AgentAction
    from neuroforge.agents.observation import AgentObservation

__all__ = ["IAgent"]


@runtime_checkable
class IAgent(Protocol):
    """Entity that senses, decides, remembers, and acts."""

    def act(self, observation: AgentObservation) -> AgentAction:
        """Choose an action for one observation."""
        ...

    def reset(self) -> None:
        """Reset episode-local state."""
        ...
