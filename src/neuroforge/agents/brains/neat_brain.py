"""NEAT brain adapter placeholder."""

from __future__ import annotations

from neuroforge.agents.brains.snn_brain import SNNBrain

__all__ = ["NEATBrain"]


class NEATBrain(SNNBrain):
    """Brain backed by a NEAT phenotype."""
