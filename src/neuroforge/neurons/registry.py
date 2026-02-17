"""Neuron model registry."""

from __future__ import annotations

from neuroforge.core.registry import Registry

__all__ = ["NEURON_MODELS", "create_neuron_model"]

NEURON_MODELS: Registry = Registry("neurons")


def _register_builtins() -> None:
    """Register built-in neuron models."""
    from neuroforge.neurons.lif.model import LIFModel

    NEURON_MODELS.register("lif", LIFModel)


_register_builtins()


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
