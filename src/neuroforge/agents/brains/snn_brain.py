"""SNN brain adapter placeholder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuroforge.agents.brain import BrainDecision

if TYPE_CHECKING:
    from neuroforge.agents.observation import AgentObservation

__all__ = ["SNNBrain"]


class SNNBrain:
    """Minimal SNN brain adapter surface."""

    def decide(self, observation: AgentObservation) -> BrainDecision:
        return BrainDecision({"observation": observation})
