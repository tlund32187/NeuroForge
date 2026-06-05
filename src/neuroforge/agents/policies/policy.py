"""Policy protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neuroforge.agents.action import AgentAction
    from neuroforge.agents.brain import BrainDecision
    from neuroforge.agents.observation import AgentObservation

__all__ = ["IPolicy"]


@runtime_checkable
class IPolicy(Protocol):
    """Selects actions from observations and brain decisions."""

    def select_action(
        self,
        observation: AgentObservation,
        decision: BrainDecision,
    ) -> AgentAction:
        """Select an action."""
        ...
