"""Biological compartment identifiers."""

from __future__ import annotations

from enum import StrEnum

__all__ = ["Compartment"]


class Compartment(StrEnum):
    """Neuron compartment identifiers."""

    SOMA = "soma"
    DENDRITE = "dendrite"
    AXON = "axon"
