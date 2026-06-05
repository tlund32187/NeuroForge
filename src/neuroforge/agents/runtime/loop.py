"""Runtime loop primitives for agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuroforge.agents.action import AgentAction
    from neuroforge.agents.base import IAgent
    from neuroforge.agents.observation import AgentObservation

__all__ = ["step_agent"]


def step_agent(agent: IAgent, observation: AgentObservation) -> AgentAction:
    """Run one agent decision step."""
    return agent.act(observation)
