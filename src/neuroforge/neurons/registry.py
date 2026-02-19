"""Neuron model registry — thin wrapper around :data:`DEFAULT_HUB.neurons`.

The module-level :data:`NEURON_MODELS` is an alias for
``DEFAULT_HUB.neurons`` so every existing import keeps working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuroforge.factories.hub import DEFAULT_HUB

if TYPE_CHECKING:
    from neuroforge.core.registry import Registry

__all__ = ["NEURON_MODELS", "create_neuron_model"]

NEURON_MODELS: Registry = DEFAULT_HUB.neurons


def create_neuron_model(key: str, **kwargs: object) -> object:
    """Create a neuron model by registry key.

    Parameters
    ----------
    key:
        Model key (e.g. ``"lif"``).
    **kwargs:
        Forwarded to the model constructor.
    """
    return NEURON_MODELS.create(key, **kwargs)
