"""Shared neuron parameter types."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["NeuronParams"]


@dataclass(frozen=True, slots=True)
class NeuronParams:
    """Base parameter dataclass for neuron model-specific parameter sets."""
