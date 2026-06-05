"""Agent evaluation result containers."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["AgentEvaluationResult"]


@dataclass(frozen=True, slots=True)
class AgentEvaluationResult:
    """Aggregate evaluation result for an agent."""

    episodes: int
    mean_reward: float
