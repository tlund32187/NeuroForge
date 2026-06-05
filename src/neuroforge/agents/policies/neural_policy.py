"""Neural policy placeholder."""

from __future__ import annotations

from neuroforge.agents.policies.random_policy import RandomPolicy

__all__ = ["NeuralPolicy"]


class NeuralPolicy(RandomPolicy):
    """Policy backed by neural decision signals."""
