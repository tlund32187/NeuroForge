"""Seed a reusable policy genome from the canonical neuroevolution package."""

from __future__ import annotations

from neuroforge.neuroevolution import PolicyGenome


def build_seed() -> PolicyGenome:
    """Return a deterministic seed genome for customization."""
    return PolicyGenome.seed("custom_policy", randomise=False)

