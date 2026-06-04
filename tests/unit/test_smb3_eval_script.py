"""Unit tests for SMB3 checkpoint-evaluation script helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from neuroforge.evolution import Gene, PolicyGenome

if TYPE_CHECKING:
    from types import ModuleType


def _load_eval_script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    path = scripts_dir / "evaluate_smb3_checkpoint.py"
    spec = importlib.util.spec_from_file_location("evaluate_smb3_checkpoint", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_evolution_checkpoint(path: Path, genome: PolicyGenome) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "generation": genome.generation,
                "evaluations": 1,
                "population": [genome.to_dict()],
                "best": {
                    "genome": genome.to_dict(),
                    "fitness": 12.5,
                    "metrics": {},
                    "episodes": 1,
                    "frames": 10,
                    "species_id": 0,
                    "adjusted_fitness": 12.5,
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.unit
def test_eval_script_reads_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "policy.pt"
    evolution_checkpoint = tmp_path / "evolution.json"
    monkeypatch.setenv("NEUROFORGE_SMB3_PORT", "8765")
    monkeypatch.setenv("NEUROFORGE_SMB3_FRAMESKIP", "8")
    monkeypatch.setenv("NEUROFORGE_SMB3_SPEED_PERCENT", "500")
    monkeypatch.setenv("NEUROFORGE_EVAL_EPISODES", "2")
    monkeypatch.setenv("NEUROFORGE_EVAL_FRAMES", "120")
    monkeypatch.setenv("NEUROFORGE_EVAL_TELEMETRY_EVERY", "0")
    monkeypatch.setenv("NEUROFORGE_EVAL_DETERMINISTIC", "0")
    monkeypatch.setenv("NEUROFORGE_EVAL_RANDOM_BASELINE", "1")
    monkeypatch.setenv("NEUROFORGE_SMB3_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("NEUROFORGE_SMB3_USE_EVOLVED", "1")
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLUTION_CHECKPOINT", str(evolution_checkpoint))
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVED_MODE", "full")

    module = _load_eval_script(monkeypatch)

    assert module.PORT == 8765
    assert module.FRAMESKIP == 8
    assert module.BIZHAWK_SPEED_PERCENT == 500
    assert module.MAX_EPISODES == 2
    assert module.FRAMES_PER_EPISODE == 120
    assert module.TELEMETRY_EVERY == 0
    assert module.DETERMINISTIC is False
    assert module.RUN_RANDOM_BASELINE is True
    assert checkpoint == module.CHECKPOINT
    assert module.USE_EVOLVED_GENOME is True
    assert str(evolution_checkpoint) == module.EVOLUTION_CHECKPOINT
    assert module.EVOLVED_MODE == "full"


@pytest.mark.unit
def test_full_phenotype_eval_enables_partial_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_eval_script(monkeypatch)
    replacements: dict[str, int | bool] = {
        "n_hidden_layers": 3,
        "hidden_fanin": 24,
        "input_to_motor_skip": True,
    }
    genome = PolicyGenome(
        id="deep_eval",
        generation=2,
        genes=tuple(
            Gene(gene.innovation, gene.key, replacements[gene.key])
            if gene.key in replacements
            else gene
            for gene in PolicyGenome.seed("base", randomise=False).genes
        ),
    )
    checkpoint = tmp_path / "run" / "evolution" / "checkpoint.json"
    _write_evolution_checkpoint(checkpoint, genome)

    module.USE_EVOLVED_GENOME = True
    module.EVOLUTION_CHECKPOINT = str(checkpoint)
    module.EVOLVED_MODE = "full"

    cfg = module._build_eval_config(checkpoint_path=tmp_path / "policy.pt")

    assert cfg.n_hidden_layers == 3
    assert cfg.hidden_fanin == 24
    assert cfg.input_to_motor_skip is True
    assert cfg.resume is True
    assert cfg.resume_allow_partial is True


@pytest.mark.unit
def test_full_phenotype_random_baseline_uses_evolved_architecture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_eval_script(monkeypatch)
    replacements: dict[str, int | bool] = {
        "n_hidden_layers": 2,
        "hidden_fanin": 48,
        "input_to_motor_skip": True,
    }
    genome = PolicyGenome(
        id="deep_random_eval",
        generation=2,
        genes=tuple(
            Gene(gene.innovation, gene.key, replacements[gene.key])
            if gene.key in replacements
            else gene
            for gene in PolicyGenome.seed("base", randomise=False).genes
        ),
    )
    checkpoint = tmp_path / "run" / "evolution" / "checkpoint.json"
    _write_evolution_checkpoint(checkpoint, genome)

    module.USE_EVOLVED_GENOME = True
    module.EVOLUTION_CHECKPOINT = str(checkpoint)
    module.EVOLVED_MODE = "full"

    cfg = module._build_eval_config(checkpoint_path=None)

    assert cfg.n_hidden_layers == 2
    assert cfg.hidden_fanin == 48
    assert cfg.input_to_motor_skip is True
    assert cfg.resume is False
    assert cfg.resume_allow_partial is False
