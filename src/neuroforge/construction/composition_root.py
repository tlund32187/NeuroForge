"""Explicit composition root for built-in NeuroForge components."""

from __future__ import annotations

from neuroforge.construction.hub import FactoryHub
from neuroforge.construction.registrations import (
    applications,
    biology,
    learning,
    perception,
    plasticity,
    synapses,
)

__all__ = ["DEFAULT_HUB", "build_default_hub"]


def build_default_hub() -> FactoryHub:
    """Create a :class:`FactoryHub` populated with built-in components."""
    hub = FactoryHub()
    biology.register(hub)
    synapses.register(hub)
    plasticity.register(hub)
    learning.register(hub)
    perception.register(hub)
    applications.register(hub)
    return hub


DEFAULT_HUB: FactoryHub = build_default_hub()
