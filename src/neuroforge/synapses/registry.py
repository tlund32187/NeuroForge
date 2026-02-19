"""Synapse model registry — thin wrapper around :data:`DEFAULT_HUB.synapses`.

The module-level :data:`SYNAPSE_MODELS` is an alias for
``DEFAULT_HUB.synapses`` so every existing import keeps working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuroforge.factories.hub import DEFAULT_HUB

if TYPE_CHECKING:
    from neuroforge.core.registry import Registry

__all__ = ["SYNAPSE_MODELS", "create_synapse_model"]

SYNAPSE_MODELS: Registry = DEFAULT_HUB.synapses


def create_synapse_model(key: str, **kwargs: object) -> object:
    """Create a synapse model by registry key."""
    return SYNAPSE_MODELS.create(key, **kwargs)
