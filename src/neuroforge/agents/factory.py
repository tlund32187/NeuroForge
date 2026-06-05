"""Agent construction helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuroforge.agents.agent import Agent

if TYPE_CHECKING:
    from neuroforge.agents.brain import IBrain
    from neuroforge.agents.policies.policy import IPolicy

__all__ = ["build_agent"]


def build_agent(*, brain: IBrain, policy: IPolicy) -> Agent:
    """Build a composable agent from a brain and policy."""
    return Agent(brain=brain, policy=policy)
