"""Random policy placeholder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuroforge.agents.action import AgentAction

if TYPE_CHECKING:
    from neuroforge.agents.brain import BrainDecision
    from neuroforge.agents.observation import AgentObservation

__all__ = ["RandomPolicy"]


class RandomPolicy:
    """Deterministic placeholder for a future random policy."""

    def select_action(
        self,
        observation: AgentObservation,
        decision: BrainDecision,
    ) -> AgentAction:
        return AgentAction({"observation_step": observation.step, "signals": decision.signals})
