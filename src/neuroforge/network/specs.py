"""Network specification DTOs for declarative network construction.

These frozen dataclasses describe the *structure* of a spiking neural
network (populations, projections, topology) without performing any
allocation or I/O.  They are consumed by :class:`NetworkFactory` to
produce a fully initialised :class:`CoreEngine`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "GateNetworkSpec",
    "PopulationSpec",
    "ProjectionSpec",
    "NetworkSpec",
]


@dataclass(frozen=True, slots=True)
class PopulationSpec:
    """Specification for a neuron population.

    Attributes
    ----------
    name:
        Unique population identifier (e.g. ``"input"``, ``"hidden"``).
    n:
        Number of neurons.
    neuron_model:
        Registry key for the neuron model (e.g. ``"lif"``).
    neuron_params:
        Keyword arguments forwarded to the neuron model constructor.
    """

    name: str
    n: int
    neuron_model: str
    neuron_params: dict[str, Any] = field(default_factory=lambda: {})


@dataclass(frozen=True, slots=True)
class ProjectionSpec:
    """Specification for a synaptic projection between two populations.

    Attributes
    ----------
    name:
        Unique projection identifier.
    source:
        Name of the pre-synaptic population.
    target:
        Name of the post-synaptic population.
    synapse_model:
        Registry key for the synapse model (e.g. ``"static"``).
    synapse_params:
        Keyword arguments forwarded to the synapse model constructor.
    topology:
        Wiring / initialisation configuration.  MVP keys:

        - ``type`` — ``"dense"`` (all-to-all) or ``"mask"`` (future).
        - ``init`` — weight initialisation: ``"uniform"`` (default),
          ``"zeros"``.
        - ``low`` / ``high`` — bounds for uniform init (default ±0.3).
        - ``bias`` — ``True`` to allocate a bias vector on the target
          side of the projection.
    """

    name: str
    source: str
    target: str
    synapse_model: str
    synapse_params: dict[str, Any] = field(default_factory=lambda: {})
    topology: dict[str, Any] = field(default_factory=lambda: {})


@dataclass(frozen=True, slots=True)
class NetworkSpec:
    """Full network specification (populations + projections).

    Attributes
    ----------
    populations:
        Ordered list of population specifications.
    projections:
        Ordered list of projection specifications.
    metadata:
        Arbitrary extra data (task name, version, notes, …).
    """

    populations: list[PopulationSpec]
    projections: list[ProjectionSpec]
    metadata: dict[str, Any] = field(default_factory=lambda: {})


@dataclass(frozen=True, slots=True)
class GateNetworkSpec:
    """High-level specification for a logic-gate spiking network.

    Describes the network *shape* and initialisation policy without
    allocating any tensors.  Consumed by :func:`build_gate_network`.

    Attributes
    ----------
    input_size:
        Number of input neurons (e.g. 2 for binary gates).
    hidden_size:
        Number of hidden neurons.  Set to 0 for the no-hidden
        (input → output) case.
    output_size:
        Number of output neurons (1 for single-gate tasks).
    n_inhibitory_hidden:
        How many of the *hidden_size* neurons are inhibitory
        (Dale's Law).  Ignored when ``hidden_size == 0``.
    seed:
        Random seed for reproducible weight initialisation.
    device:
        Torch device string (``"cpu"`` or ``"cuda"``).
    dtype:
        Torch dtype string (``"float32"`` or ``"float64"``).
    init_scale:
        Symmetric uniform init range ``[-init_scale, +init_scale]``.
    p_connect:
        Connection probability per possible edge.  ``1.0`` gives
        fully-connected (dense) wiring; values < 1 produce a
        randomly-sampled sparse topology (seeded).
    neuron_model:
        Registry key for the neuron model used in all populations
        (e.g. ``"lif_surr"``).
    synapse_model:
        Registry key for the synapse model used in all projections
        (e.g. ``"static_dales"``).
    """

    input_size: int = 2
    hidden_size: int = 6
    output_size: int = 1
    n_inhibitory_hidden: int = 2
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float64"
    init_scale: float = 0.3
    p_connect: float = 1.0
    neuron_model: str = "lif_surr"
    synapse_model: str = "static_dales"
