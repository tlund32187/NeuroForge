"""Verify contracts are importable and kept in the target package shape."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module_missing(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except ModuleNotFoundError:
        return True


def _assign_frozen_attr(target: object, name: str, value: object) -> None:
    setattr(target, name, value)


@pytest.mark.unit
def test_contracts_tree_contains_expected_modules() -> None:
    root = Path(__file__).parents[2] / "src" / "neuroforge" / "contracts"
    expected = {
        "__init__.py",
        "tensors.py",
        "runtime.py",
        "registries.py",
        "factories.py",
        "simulation.py",
        "messaging.py",
        "monitoring.py",
        "biology/__init__.py",
        "biology/compartments.py",
        "biology/neurons.py",
        "biology/synapses.py",
        "biology/plasticity.py",
        "biology/receptors.py",
        "biology/ion_channels.py",
        "biology/neuromodulators.py",
        "applications/__init__.py",
        "applications/tasks.py",
        "applications/games.py",
        "applications/evolution.py",
    }

    missing = [path for path in sorted(expected) if not (root / path).is_file()]
    assert missing == []


@pytest.mark.unit
def test_removed_contract_modules_are_not_present() -> None:
    legacy_modules = [
        "neuroforge.contracts.tensor",
        "neuroforge.contracts.construction",
        "neuroforge.contracts.game",
        "neuroforge.contracts.encoding",
        "neuroforge.contracts.learning",
        "neuroforge.contracts.types",
        "neuroforge.contracts.monitors",
    ]

    missing = [name for name in legacy_modules if _module_missing(name)]

    assert missing == legacy_modules


@pytest.mark.unit
def test_core_contract_imports() -> None:
    from neuroforge.contracts.factories import IFactory
    from neuroforge.contracts.registries import IRegistry
    from neuroforge.contracts.runtime import IRuntimeComponent
    from neuroforge.contracts.tensors import Tensor

    assert all(cls is not None for cls in [IFactory, IRegistry, IRuntimeComponent])
    assert Tensor is not None


@pytest.mark.unit
def test_biology_contract_imports() -> None:
    from neuroforge.contracts.biology.compartments import Compartment
    from neuroforge.contracts.biology.ion_channels import IIonChannel
    from neuroforge.contracts.biology.neuromodulators import NeuromodulatorSignal
    from neuroforge.contracts.biology.neurons import INeuronModel, StepContext
    from neuroforge.contracts.biology.plasticity import ILearningRule, LearningBatch
    from neuroforge.contracts.biology.receptors import IReceptorModel
    from neuroforge.contracts.biology.synapses import ISynapseModel, SynapseTopology

    assert Compartment.SOMA.value == "soma"
    assert all(
        cls is not None
        for cls in [
            IIonChannel,
            ILearningRule,
            INeuronModel,
            IReceptorModel,
            ISynapseModel,
            LearningBatch,
            NeuromodulatorSignal,
            StepContext,
            SynapseTopology,
        ]
    )


@pytest.mark.unit
def test_application_contract_imports() -> None:
    from neuroforge.contracts.applications.evolution import (
        FitnessResult,
        IFitnessEvaluator,
        IGenome,
    )
    from neuroforge.contracts.applications.games import (
        ControllerAction,
        GameObservation,
        IGameClient,
        ScreenFrame,
        VisionGameMetrics,
    )
    from neuroforge.contracts.applications.tasks import ILoss, IReadout, ReadoutResult

    assert all(
        cls is not None
        for cls in [
            ControllerAction,
            FitnessResult,
            GameObservation,
            IFitnessEvaluator,
            IGameClient,
            IGenome,
            ILoss,
            IReadout,
            ReadoutResult,
            ScreenFrame,
            VisionGameMetrics,
        ]
    )


@pytest.mark.unit
def test_messaging_and_monitoring_are_split() -> None:
    from neuroforge.contracts.messaging import EventTopic, IEventBus, MonitorEvent
    from neuroforge.contracts.monitoring import IMonitor

    assert all(cls is not None for cls in [EventTopic, IEventBus, IMonitor, MonitorEvent])


@pytest.mark.unit
def test_protocols_have_expected_methods() -> None:
    from neuroforge.contracts.applications.evolution import IGenome
    from neuroforge.contracts.applications.games import IGameClient
    from neuroforge.contracts.biology.neurons import INeuronModel
    from neuroforge.contracts.biology.plasticity import ILearningRule
    from neuroforge.contracts.biology.synapses import ISynapseModel
    from neuroforge.contracts.registries import IRegistry
    from neuroforge.contracts.simulation import ISimulationEngine

    expectations = {
        INeuronModel: {"init_state", "step", "reset_state", "state_tensors"},
        ISynapseModel: {"init_state", "step", "state_tensors"},
        ILearningRule: {"init_state", "step", "state_tensors"},
        ISimulationEngine: {"build", "reset", "step", "run"},
        IRegistry: {"register", "create", "list_keys", "has"},
        IGameClient: {"reset", "step", "close"},
        IGenome: {"to_dict"},
    }

    for protocol, methods in expectations.items():
        for method in methods:
            assert hasattr(protocol, method), f"{protocol.__name__} missing {method}"


@pytest.mark.unit
def test_dtos_are_frozen() -> None:
    from neuroforge.contracts.biology.neurons import StepContext
    from neuroforge.contracts.simulation import SimulationConfig, StepResult

    ctx = StepContext(step=0, dt=0.001, t=0.0)
    with pytest.raises(AttributeError):
        _assign_frozen_attr(ctx, "step", 1)

    cfg = SimulationConfig()
    with pytest.raises(AttributeError):
        _assign_frozen_attr(cfg, "dt", 0.01)

    result = StepResult(step=0, t=0.0, spikes={})
    assert result.extra == {}
