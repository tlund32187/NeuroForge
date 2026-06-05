"""Unit tests for SMB3 evolution script configuration helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    from types import ModuleType


def _load_evolve_script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    del monkeypatch
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "neuroforge"
        / "applications"
        / "smb3"
        / "evolve.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evolve_smb3_test_module",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_evolve_script_reads_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "evolve_run"
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLUTION_RUN_DIR", str(run_dir))
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_POPULATION", "6")
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_GENERATIONS", "2")
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_ELITES", "1")
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_EVAL_EPISODES", "3")
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_EVAL_FRAMES", "120")
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_WORKERS", "2")
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_SEED", "99")
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_MUTATION_RATE", "0.2")
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_MUTATION_POWER", "0.75")
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_SPECIES_THRESHOLD", "0.55")
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_EVAL_REPEATS", "3")
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_PROGRESS_SCALE", "120")
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_SURVIVAL_SCALE", "2")
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_DURABLE_PROGRESS_WEIGHT", "1.5")
    monkeypatch.setenv("NEUROFORGE_SMB3_LAUNCH_EMUHAWK", "0")

    module = _load_evolve_script(monkeypatch)

    assert module._resolve_run_dir() == run_dir
    assert module.POPULATION_SIZE == 6
    assert module.GENERATIONS == 2
    assert module.ELITE_COUNT == 1
    assert module.EVAL_EPISODES == 3
    assert module.EVAL_FRAMES_PER_EPISODE == 120
    assert module.EVOLUTION_WORKERS == 2
    assert module.EVOLUTION_SEED == 99
    assert pytest.approx(0.2) == module.MUTATION_RATE
    assert pytest.approx(0.75) == module.MUTATION_POWER
    assert pytest.approx(0.55) == module.SPECIES_THRESHOLD
    assert module.EVAL_REPEATS == 3
    assert module.EVOLVE_PROFILE == "explore"
    assert module.EVOLVE_STALL_PATIENCE == 240
    assert module.EVOLVE_MIN_PROGRESS_FRAMES == 240
    assert module.EVOLVE_MAX_DECIDE_TICKS == 16
    assert pytest.approx(120.0) == module.FITNESS_PROGRESS_SCALE
    assert pytest.approx(2.0) == module.FITNESS_SURVIVAL_SCALE
    assert pytest.approx(1.5) == module.FITNESS_DURABLE_PROGRESS_WEIGHT
    assert module.LAUNCH_EMUHAWK is False


@pytest.mark.unit
def test_evolve_script_rejects_invalid_small_env_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_POPULATION", "1")
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_GENERATIONS", "0")
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_EVAL_FRAMES", "0")
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_SPECIES_THRESHOLD", "0")
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_EVAL_REPEATS", "0")

    module = _load_evolve_script(monkeypatch)

    assert module.POPULATION_SIZE == 32
    assert module.GENERATIONS == 40
    assert module.EVAL_FRAMES_PER_EPISODE == 900
    assert pytest.approx(0.5) == module.SPECIES_THRESHOLD
    assert module.EVAL_REPEATS == 1


@pytest.mark.unit
def test_evolve_script_validate_profile_uses_fuller_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_PROFILE", "validate")

    module = _load_evolve_script(monkeypatch)

    assert module.EVOLVE_PROFILE == "validate"
    assert module.EVAL_FRAMES_PER_EPISODE == 3600
    assert module.EVAL_REPEATS == 2
    assert module.EVOLVE_STALL_PATIENCE == 600
    assert module.EVOLVE_MIN_PROGRESS_FRAMES == 0
    assert module.EVOLVE_MAX_DECIDE_TICKS == 0


@pytest.mark.unit
def test_evolve_script_caps_live_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVE_WORKERS", "9")

    module = _load_evolve_script(monkeypatch)

    assert module.REQUESTED_EVOLUTION_WORKERS == 9
    assert module.EVOLUTION_WORKERS == 4
    assert module.EVOLUTION_WORKER_PORTS == (8650, 8651, 8652, 8653)


@pytest.mark.unit
def test_worker_evaluator_uses_unique_port_and_error_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_evolve_script(monkeypatch)
    captured: list[object] = []

    def fake_builder(config: object, **_kwargs: object) -> object:
        captured.append(config)
        return object()

    monkeypatch.setattr(module, "build_live_smb3_fitness_evaluator", fake_builder)
    live_cfg = module.SMB3LiveFitnessConfig(
        emuhawk_path="emu",
        rom_path="rom",
        lua_script="lua",
        port=9100,
    )

    module._build_evaluator_for_worker(live_cfg, run_dir=tmp_path, worker_index=2)

    assert captured
    cfg = cast("Any", captured[0])
    assert cfg.port == 9102
    assert str(tmp_path / "bizhawk" / "bridge_error_worker_2.log") == cfg.bridge_error_path


@pytest.mark.unit
def test_console_monitor_prints_resume_context(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_evolve_script(monkeypatch)
    monitor = module._ConsoleMonitor()

    monitor._print_resume(
        {
            "generations": 4,
            "resume": {
                "loaded": True,
                "path": "runs/evolve/evolution/checkpoint.json",
                "schema_version": 2,
                "generation": 2,
                "evaluations": 12,
                "population_size": 6,
                "config_differences": {
                    "population_size": {"checkpoint": 6, "active": 4},
                    "generations": {"checkpoint": 2, "active": 4},
                    "resume": {"checkpoint": False, "active": True},
                },
            },
        }
    )

    output = capsys.readouterr().out
    assert "resumed checkpoint: gen 3/4 | pop=6 | evals=12 | schema=v2" in output
    assert "runs/evolve/evolution/checkpoint.json" in output
    assert "active config changes: population_size 6->4, generations 2->4" in output
    assert "resume False->True" not in output
