"""A tiny brain using canonical brain abstractions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuroforge.agents.brain import BrainDecision

if TYPE_CHECKING:
    from neuroforge.agents.observation import AgentObservation


class BiasBrain:
    """Example brain that returns fixed decision signals."""

    def decide(self, observation: AgentObservation) -> BrainDecision:
        del observation
        return BrainDecision({"right": 1.0, "jump": 0.0})
