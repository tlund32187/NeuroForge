"""Adaptive HyperNEAT brain adapter placeholder."""

from __future__ import annotations

from neuroforge.agents.brains.hyperneat_brain import HyperNEATBrain

__all__ = ["AdaptiveHyperNEATBrain"]


class AdaptiveHyperNEATBrain(HyperNEATBrain):
    """Brain backed by an adaptive HyperNEAT phenotype."""
