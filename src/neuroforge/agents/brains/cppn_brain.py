"""CPPN brain adapter placeholder."""

from __future__ import annotations

from neuroforge.agents.brains.snn_brain import SNNBrain

__all__ = ["CPPNBrain"]


class CPPNBrain(SNNBrain):
    """Brain backed directly by a CPPN."""
