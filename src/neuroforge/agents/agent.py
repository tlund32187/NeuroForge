"""Simple composable agent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from neuroforge.agents.base import IAgent

if TYPE_CHECKING:
    from neuroforge.agents.action import AgentAction
    from neuroforge.agents.brain import IBrain
    from neuroforge.agents.observation import AgentObservation
    from neuroforge.agents.policies.policy import IPolicy

__all__ = ["Agent"]


@dataclass(slots=True)
class Agent(IAgent):
    """Agent that delegates decision-making to a brain and policy."""

    brain: IBrain
    policy: IPolicy

    def act(self, observation: AgentObservation) -> AgentAction:
        decision = self.brain.decide(observation)
        return self.policy.select_action(observation, decision)

    def reset(self) -> None:
        """Reset episode-local state."""
        return None
