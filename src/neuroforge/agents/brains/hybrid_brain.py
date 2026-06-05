"""Hybrid brain adapter placeholder."""

from __future__ import annotations

from neuroforge.agents.brains.snn_brain import SNNBrain

__all__ = ["HybridBrain"]


class HybridBrain(SNNBrain):
    """Brain composed from multiple decision mechanisms."""
