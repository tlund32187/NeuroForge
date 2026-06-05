"""A tiny environment sketch using canonical agent DTOs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuroforge.agents.observation import AgentObservation

if TYPE_CHECKING:
    from neuroforge.agents.action import AgentAction


class CounterEnvironment:
    """Example environment that increments a counter per action."""

    def __init__(self) -> None:
        self.step_count = 0

    def reset(self) -> AgentObservation:
        self.step_count = 0
        return AgentObservation({"step": self.step_count})

    def step(self, action: AgentAction) -> AgentObservation:
        del action
        self.step_count += 1
        return AgentObservation({"step": self.step_count})
