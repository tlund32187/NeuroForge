"""Shared enumerations used across contracts."""

from __future__ import annotations

from enum import StrEnum

__all__ = ["Compartment"]


class Compartment(StrEnum):
    """Neuron compartment identifiers.

    Starts with SOMA only.  DENDRITE and AXON will be added when
    multi-compartment models are implemented (Phase 9).
    """

    SOMA = "soma"
