"""Define a HyperNEAT substrate through the canonical neuroevolution package."""

from __future__ import annotations

from neuroforge.neuroevolution.genomes.substrate import SubstrateConfig


def build_substrate_config() -> SubstrateConfig:
    """Return a small substrate useful for example experiments."""
    return SubstrateConfig(input_shape=(1, 8, 8), hidden_shape=(4, 4))

