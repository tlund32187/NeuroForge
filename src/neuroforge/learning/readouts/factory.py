"""Factory helpers for learning readouts."""

from __future__ import annotations

from neuroforge.construction.registry import Registry
from neuroforge.learning.readouts.rate_decoder import RateDecoder
from neuroforge.learning.readouts.spike_count import SpikeCountReadout

__all__ = ["build_readout_registry"]


def build_readout_registry() -> Registry:
    """Build a registry populated with built-in readouts."""
    registry = Registry("readouts")
    registry.register("rate_decoder", RateDecoder)
    registry.register("spike_count", SpikeCountReadout)
    return registry
