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

# Safety bootstrap: keep delayed synapse available even if a custom hub
# variant omitted the key during initial registration.
if not SYNAPSE_MODELS.has("static_delayed"):
    from neuroforge.synapses.delayed_static import DelayedStaticSynapseModel

    SYNAPSE_MODELS.register("static_delayed", DelayedStaticSynapseModel)


def create_synapse_model(key: str, **kwargs: object) -> object:
    """Create a synapse model by registry key."""
    return SYNAPSE_MODELS.create(key, **kwargs)
