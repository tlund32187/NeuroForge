"""Network specification DTOs for declarative network construction.

These frozen dataclasses describe the *structure* of a spiking neural
network (populations, projections, topology) without performing any
allocation or I/O.  They are consumed by :class:`NetworkFactory` to
produce a fully initialised :class:`CoreEngine`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["PopulationSpec", "ProjectionSpec", "NetworkSpec"]


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
