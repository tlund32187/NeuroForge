"""Synapse model registry."""

from __future__ import annotations

from neuroforge.core.registry import Registry

__all__ = ["SYNAPSE_MODELS", "create_synapse_model"]

SYNAPSE_MODELS: Registry = Registry("synapses")


def _register_builtins() -> None:
    """Register built-in synapse models."""
    from neuroforge.synapses.static import StaticSynapseModel

    SYNAPSE_MODELS.register("static", StaticSynapseModel)


_register_builtins()


def create_synapse_model(key: str, **kwargs: object) -> object:
    """Create a synapse model by registry key."""
    return SYNAPSE_MODELS.create(key, **kwargs)
