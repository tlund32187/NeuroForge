"""Tests for Phase 3 simulation and construction package boundaries."""

from __future__ import annotations

import importlib.util

import pytest


def _module_missing(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except ModuleNotFoundError:
        return True


@pytest.mark.unit
def test_simulation_engine_canonical_imports() -> None:
    from neuroforge.simulation.engine.bench_utils import measure_steps_per_sec
    from neuroforge.simulation.engine.core import CoreEngine, Population, Projection

    assert CoreEngine.__name__ == "CoreEngine"
    assert Population.__name__ == "Population"
    assert Projection.__name__ == "Projection"
    assert measure_steps_per_sec.__name__ == "measure_steps_per_sec"


@pytest.mark.unit
def test_simulation_topology_canonical_imports() -> None:
    from neuroforge.simulation.topology.builders import build_dense_topology
    from neuroforge.simulation.topology.specs import (
        NetworkSpec,
        PopulationSpec,
        ProjectionSpec,
    )

    assert NetworkSpec.__name__ == "NetworkSpec"
    assert PopulationSpec.__name__ == "PopulationSpec"
    assert ProjectionSpec.__name__ == "ProjectionSpec"
    assert build_dense_topology.__name__ == "build_dense_topology"


@pytest.mark.unit
def test_construction_canonical_imports() -> None:
    from neuroforge.construction.gate_builder import build_gate_network
    from neuroforge.construction.network_factory import NetworkFactory, to_topology_json

    assert NetworkFactory.__name__ == "NetworkFactory"
    assert build_gate_network.__name__ == "build_gate_network"
    assert to_topology_json.__name__ == "to_topology_json"


@pytest.mark.unit
def test_legacy_wrapper_modules_are_not_present() -> None:
    legacy_modules = [
        "neuroforge.contracts.monitors",
        "neuroforge.contracts.types",
        "neuroforge.contracts.neurons",
        "neuroforge.contracts.synapses",
        "neuroforge.monitors",
        "neuroforge.monitors.bus",
        "neuroforge.neurons.base",
        "neuroforge.neurons.lif.model",
        "neuroforge.neurons.lif.surrogate",
        "neuroforge.neurons.registry",
        "neuroforge.synapses.static",
        "neuroforge.synapses.delayed_static",
        "neuroforge.synapses.dales_static",
        "neuroforge.synapses.registry",
        "neuroforge.learning.rstdp",
        "neuroforge.learning.online_rstdp",
        "neuroforge.engine.core_engine",
        "neuroforge.engine.bench_utils",
        "neuroforge.network.topology_builders",
        "neuroforge.network.factory",
        "neuroforge.network.gate_builder",
        "neuroforge.runners.run_context",
        "neuroforge.runners",
        "neuroforge.factories",
        "neuroforge.factories.hub",
        "neuroforge.biology.neurons.registry",
        "neuroforge.biology.synapses.registry",
        "neuroforge.encoding.registry",
        "neuroforge.readout.registry",
        "neuroforge.losses.registry",
        "neuroforge.core",
        "neuroforge.dashboard",
        "neuroforge.data",
        "neuroforge.encoders",
        "neuroforge.encoding",
        "neuroforge.evolution",
        "neuroforge.game",
        "neuroforge.losses",
        "neuroforge.readout",
        "neuroforge.tasks",
        "neuroforge.training",
        "neuroforge.vision",
        "neuroforge.vision.registry",
        "neuroforge.learning.registry",
    ]

    missing = [name for name in legacy_modules if _module_missing(name)]

    assert missing == legacy_modules
