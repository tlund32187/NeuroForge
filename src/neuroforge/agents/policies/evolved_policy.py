"""Evolved policy placeholder."""

from __future__ import annotations

from neuroforge.agents.policies.neural_policy import NeuralPolicy

__all__ = ["EvolvedPolicy"]


class EvolvedPolicy(NeuralPolicy):
    """Policy backed by evolved brain or genome outputs."""
