"""Verify all contracts are importable, runtime_checkable, and have expected methods."""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestContractsImportable:
    """All contract modules import without error."""

    def test_import_tensor(self) -> None:
        from neuroforge.contracts.tensor import Tensor

        assert Tensor is not None

    def test_import_types(self) -> None:
        from neuroforge.contracts.types import Compartment

        assert Compartment.SOMA.value == "soma"

    def test_import_factories(self) -> None:
        from neuroforge.contracts.factories import IRegistry

        assert IRegistry is not None

    def test_import_neurons(self) -> None:
        from neuroforge.contracts.neurons import (
            INeuronModel,
            NeuronInputs,
            NeuronStepResult,
            StepContext,
        )

        assert all(
            cls is not None
            for cls in [INeuronModel, NeuronInputs, NeuronStepResult, StepContext]
        )

    def test_import_synapses(self) -> None:
        from neuroforge.contracts.synapses import (
            ISynapseModel,
            SynapseInputs,
            SynapseStepResult,
            SynapseTopology,
        )

        assert all(
            cls is not None
            for cls in [
                ISynapseModel,
                SynapseTopology,
                SynapseInputs,
                SynapseStepResult,
            ]
        )

    def test_import_learning(self) -> None:
        from neuroforge.contracts.learning import (
            ILearningRule,
            LearningBatch,
            LearningStepResult,
        )

        assert all(
            cls is not None
            for cls in [ILearningRule, LearningBatch, LearningStepResult]
        )

    def test_import_simulation(self) -> None:
        from neuroforge.contracts.simulation import (
            ISimulationEngine,
            SimulationConfig,
            StepResult,
        )

        assert all(
            cls is not None for cls in [ISimulationEngine, SimulationConfig, StepResult]
        )


@pytest.mark.unit
class TestProtocolsAreRuntimeCheckable:
    """Protocols should be usable with isinstance() at runtime."""

    def test_ineuron_model(self) -> None:
        from neuroforge.contracts.neurons import INeuronModel

        assert hasattr(INeuronModel, "__protocol_attrs__") or hasattr(
            INeuronModel, "_is_runtime_protocol"
        )

    def test_isynapse_model(self) -> None:
        from neuroforge.contracts.synapses import ISynapseModel

        assert hasattr(ISynapseModel, "__protocol_attrs__") or hasattr(
            ISynapseModel, "_is_runtime_protocol"
        )

    def test_ilearning_rule(self) -> None:
        from neuroforge.contracts.learning import ILearningRule

        assert hasattr(ILearningRule, "__protocol_attrs__") or hasattr(
            ILearningRule, "_is_runtime_protocol"
        )

    def test_isimulation_engine(self) -> None:
        from neuroforge.contracts.simulation import ISimulationEngine

        assert hasattr(ISimulationEngine, "__protocol_attrs__") or hasattr(
            ISimulationEngine, "_is_runtime_protocol"
        )

    def test_iregistry(self) -> None:
        from neuroforge.contracts.factories import IRegistry

        assert hasattr(IRegistry, "__protocol_attrs__") or hasattr(
            IRegistry, "_is_runtime_protocol"
        )


@pytest.mark.unit
class TestProtocolsHaveExpectedMethods:
    """Each protocol declares the methods we expect."""

    def test_ineuron_model_methods(self) -> None:
        from neuroforge.contracts.neurons import INeuronModel

        expected = {"init_state", "step", "reset_state", "state_tensors"}
        # Protocol methods appear in annotations or as direct attrs
        for method in expected:
            assert hasattr(INeuronModel, method), f"Missing method: {method}"

    def test_isynapse_model_methods(self) -> None:
        from neuroforge.contracts.synapses import ISynapseModel

        expected = {"init_state", "step", "state_tensors"}
        for method in expected:
            assert hasattr(ISynapseModel, method), f"Missing method: {method}"

    def test_ilearning_rule_methods(self) -> None:
        from neuroforge.contracts.learning import ILearningRule

        expected = {"init_state", "step", "state_tensors"}
        for method in expected:
            assert hasattr(ILearningRule, method), f"Missing method: {method}"

    def test_isimulation_engine_methods(self) -> None:
        from neuroforge.contracts.simulation import ISimulationEngine

        expected = {"build", "reset", "step", "run"}
        for method in expected:
            assert hasattr(ISimulationEngine, method), f"Missing method: {method}"

    def test_iregistry_methods(self) -> None:
        from neuroforge.contracts.factories import IRegistry

        expected = {"register", "create", "list_keys", "has"}
        for method in expected:
            assert hasattr(IRegistry, method), f"Missing method: {method}"


@pytest.mark.unit
class TestDTOsAreFrozen:
    """DTOs should be immutable (frozen dataclasses)."""

    def test_step_context_frozen(self) -> None:
        from neuroforge.contracts.neurons import StepContext

        ctx = StepContext(step=0, dt=0.001, t=0.0)
        with pytest.raises(AttributeError):
            ctx.step = 1  # type: ignore[misc]

    def test_simulation_config_frozen(self) -> None:
        from neuroforge.contracts.simulation import SimulationConfig

        cfg = SimulationConfig()
        with pytest.raises(AttributeError):
            cfg.dt = 0.01  # type: ignore[misc]

    def test_simulation_config_defaults(self) -> None:
        from neuroforge.contracts.simulation import SimulationConfig

        cfg = SimulationConfig()
        assert cfg.dt == 1e-3
        assert cfg.seed == 42
        assert cfg.device == "cpu"
        assert cfg.dtype == "float32"

    def test_step_result_defaults(self) -> None:
        from neuroforge.contracts.simulation import StepResult

        result = StepResult(step=0, t=0.0, spikes={})
        assert result.extra == {}


@pytest.mark.unit
class TestEnums:
    """Enum members have expected values."""

    def test_compartment_soma(self) -> None:
        from neuroforge.contracts.types import Compartment

        assert Compartment.SOMA.value == "soma"
        assert isinstance(Compartment.SOMA, str)
