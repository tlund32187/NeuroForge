"""Unit tests for SMB3 evolution script configuration helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

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
    assert module.EVAL_FRAMES_PER_EPISODE == 3600
    assert pytest.approx(0.5) == module.SPECIES_THRESHOLD
    assert module.EVAL_REPEATS == 2


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
