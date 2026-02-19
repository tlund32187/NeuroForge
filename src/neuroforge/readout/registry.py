"""Readout registry — thin wrapper around :data:`DEFAULT_HUB.readouts`.

Mirrors the style of ``neuroforge.neurons.registry`` and
``neuroforge.synapses.registry``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuroforge.factories.hub import DEFAULT_HUB

if TYPE_CHECKING:
    from neuroforge.core.registry import Registry

__all__ = ["READOUTS", "create_readout"]

READOUTS: Registry = DEFAULT_HUB.readouts


def create_readout(key: str, **kwargs: object) -> object:
    """Create a readout by registry key.

    Parameters
    ----------
    key:
        Readout key (e.g. ``"rate_decoder"``).
    **kwargs:
        Forwarded to the readout constructor.
    """
    return READOUTS.create(key, **kwargs)
