"""HyperNEAT brain adapter placeholder."""

from __future__ import annotations

from neuroforge.agents.brains.snn_brain import SNNBrain

__all__ = ["HyperNEATBrain"]


class HyperNEATBrain(SNNBrain):
    """Brain backed by a HyperNEAT phenotype."""
