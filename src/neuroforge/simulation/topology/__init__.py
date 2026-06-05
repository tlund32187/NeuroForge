"""Topology specifications and builders for simulation construction."""

from neuroforge.simulation.topology.builders import (
    build_block_sparse_topology,
    build_dense_topology,
    build_sparse_fanin_topology,
    build_sparse_fanout_topology,
    build_sparse_random_topology,
)
from neuroforge.simulation.topology.specs import (
    GateNetworkSpec,
    NetworkSpec,
    PopulationSpec,
    ProjectionSpec,
    VisionBackboneSpec,
    VisionBlockSpec,
    VisionInputSpec,
)

__all__ = [
    "GateNetworkSpec",
    "NetworkSpec",
    "PopulationSpec",
    "ProjectionSpec",
    "VisionBackboneSpec",
    "VisionBlockSpec",
    "VisionInputSpec",
    "build_block_sparse_topology",
    "build_dense_topology",
    "build_sparse_fanin_topology",
    "build_sparse_fanout_topology",
    "build_sparse_random_topology",
]

