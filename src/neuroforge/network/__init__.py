"""NeuroForge network construction — DTOs and factory."""

from neuroforge.network.factory import (
    NetworkFactory,
    to_topology_json,
)
from neuroforge.network.gate_builder import (
    GateNetwork,
    build_dale_signs,
    build_gate_network,
    build_projection,
    init_projection_weights,
)
from neuroforge.network.specs import (
    GateNetworkSpec,
    NetworkSpec,
    PopulationSpec,
    ProjectionSpec,
)

__all__ = [
    "GateNetwork",
    "GateNetworkSpec",
    "NetworkFactory",
    "NetworkSpec",
    "PopulationSpec",
    "ProjectionSpec",
    "build_dale_signs",
    "build_gate_network",
    "build_projection",
    "init_projection_weights",
    "to_topology_json",
]
