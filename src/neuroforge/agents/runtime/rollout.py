"""Rollout result containers."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RolloutSummary"]


@dataclass(frozen=True, slots=True)
class RolloutSummary:
    """Summary of an agent rollout."""

    steps: int
    total_reward: float = 0.0
