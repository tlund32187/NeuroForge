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
    VisionBackboneSpec,
    VisionBlockSpec,
    VisionInputSpec,
)
from neuroforge.network.topology_builders import (
    build_block_sparse_topology,
    build_dense_topology,
    build_sparse_fanin_topology,
    build_sparse_fanout_topology,
    build_sparse_random_topology,
)

__all__ = [
    "GateNetwork",
    "GateNetworkSpec",
    "NetworkFactory",
    "NetworkSpec",
    "PopulationSpec",
    "ProjectionSpec",
    "VisionBackboneSpec",
    "VisionBlockSpec",
    "VisionInputSpec",
    "build_dale_signs",
    "build_gate_network",
    "build_block_sparse_topology",
    "build_dense_topology",
    "build_projection",
    "init_projection_weights",
    "build_sparse_fanin_topology",
    "build_sparse_fanout_topology",
    "build_sparse_random_topology",
    "to_topology_json",
]
