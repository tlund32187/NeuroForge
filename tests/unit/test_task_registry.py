"""Unit tests for the class-based task registry."""

from __future__ import annotations

from neuroforge.applications.tasks.registry import TASK_REGISTRY, get_task_spec


def test_known_keys_present() -> None:
    assert set(TASK_REGISTRY) == {"evolution", "logic_gate", "multi_gate", "game_training"}


def test_game_training_spec_loads_classes() -> None:
    from neuroforge.applications.tasks.game_training import GameTrainingConfig, GameTrainingTask

    spec = get_task_spec("game_training")
    assert spec is not None
    config_cls, task_cls = spec.load()
    assert config_cls is GameTrainingConfig
    assert task_cls is GameTrainingTask


def test_evolution_spec_loads_classes() -> None:
    from neuroforge.applications.tasks.evolution import EvolutionConfig, EvolutionTask

    spec = get_task_spec("evolution")
    assert spec is not None
    config_cls, task_cls = spec.load()
    assert config_cls is EvolutionConfig
    assert task_cls is EvolutionTask


def test_get_task_spec_unknown_returns_none() -> None:
    assert get_task_spec("does_not_exist") is None


def test_logic_gate_spec_loads_classes() -> None:
    from neuroforge.applications.tasks.logic_gates import LogicGateConfig, LogicGateTask

    spec = get_task_spec("logic_gate")
    assert spec is not None
    config_cls, task_cls = spec.load()
    assert config_cls is LogicGateConfig
    assert task_cls is LogicGateTask


def test_multi_gate_spec_loads_classes() -> None:
    from neuroforge.applications.tasks.multi_gate import MultiGateConfig, MultiGateTask

    spec = get_task_spec("multi_gate")
    assert spec is not None
    config_cls, task_cls = spec.load()
    assert config_cls is MultiGateConfig
    assert task_cls is MultiGateTask


def test_spec_label_is_human_readable() -> None:
    for key, spec in TASK_REGISTRY.items():
        assert spec.key == key
        assert spec.label
