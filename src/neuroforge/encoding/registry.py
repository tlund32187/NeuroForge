"""Encoder registry — thin wrapper around :data:`DEFAULT_HUB.encoders`.

Mirrors the style of ``neuroforge.neurons.registry`` and
``neuroforge.synapses.registry`` so callers can use whichever import
path they prefer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuroforge.factories.hub import DEFAULT_HUB

if TYPE_CHECKING:
    from neuroforge.core.registry import Registry

__all__ = ["ENCODERS", "create_encoder"]

ENCODERS: Registry = DEFAULT_HUB.encoders


def create_encoder(key: str, **kwargs: object) -> object:
    """Create an encoder by registry key.

    Parameters
    ----------
    key:
        Encoder key (e.g. ``"rate"``).
    **kwargs:
        Forwarded to the encoder constructor.
    """
    return ENCODERS.create(key, **kwargs)
