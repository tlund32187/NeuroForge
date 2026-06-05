"""A tiny agent using canonical agent DTOs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuroforge.agents.action import AgentAction

if TYPE_CHECKING:
    from neuroforge.agents.observation import AgentObservation


class AlwaysRightAgent:
    """Example agent that emits a simple action payload."""

    def act(self, observation: AgentObservation) -> AgentAction:
        del observation
        return AgentAction({"right": True})

    def reset(self) -> None:
        """Reset episode-local state."""
