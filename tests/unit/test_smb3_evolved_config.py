"""Unit tests for shared SMB3 evolved-genome config application."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from neuroforge.evolution import Gene, PolicyGenome
from neuroforge.tasks.game_training import GameTrainingConfig

if TYPE_CHECKING:
    from types import ModuleType


def _load_evolved_config_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "smb3_evolved_config",
        scripts_dir / "smb3_evolved_config.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _genome_with(replacements: dict[str, int | float | bool]) -> PolicyGenome:
    return PolicyGenome(
        id="evolved_smb3",
        generation=3,
        genes=tuple(
            Gene(gene.innovation, gene.key, replacements[gene.key])
            if gene.key in replacements
            else gene
            for gene in PolicyGenome.seed("base", randomise=False).genes
        ),
    )


def _write_evolution_checkpoint(path: Path, genome: PolicyGenome) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "generation": genome.generation,
                "evaluations": 3,
                "population": [genome.to_dict()],
                "best": {
                    "genome": genome.to_dict(),
                    "fitness": 42.25,
                    "metrics": {},
                    "episodes": 1,
                    "frames": 12,
                    "species_id": 0,
                    "adjusted_fitness": 42.25,
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.unit
def test_apply_evolved_config_noops_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_evolved_config_module(monkeypatch)
    base = GameTrainingConfig(seed=77)

    selection = module.apply_evolved_genome_config(
        base,
        use_evolved=False,
        evolved_mode="full",
        evolution_checkpoint=None,
        runs_dir=Path("missing"),
    )

    assert selection.config is base
    assert selection.requested is False
    assert selection.applied is False
    assert module.evolved_config_status_lines(selection, action="training") == []


@pytest.mark.unit
def test_apply_evolved_config_compatible_mode_preserves_network_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_evolved_config_module(monkeypatch)
    checkpoint = tmp_path / "run" / "evolution" / "checkpoint.json"
    _write_evolution_checkpoint(
        checkpoint,
        _genome_with(
            {
                "decide_ticks": 7,
                "reward_scale": 0.08,
                "n_hidden": 96,
                "n_hidden_layers": 3,
                "hidden_fanin": 24,
                "input_to_motor_skip": True,
            }
        ),
    )
    base = GameTrainingConfig(
        seed=77,
        n_hidden=64,
        n_hidden_layers=1,
        hidden_fanin=0,
        input_to_motor_skip=False,
    )

    selection = module.apply_evolved_genome_config(
        base,
        use_evolved=True,
        evolved_mode="compatible",
        evolution_checkpoint=str(checkpoint),
        runs_dir=tmp_path,
    )

    assert selection.applied is True
    assert selection.include_network_shape is False
    assert selection.config.decide_ticks == 7
    assert selection.config.rstdp.reward_scale == pytest.approx(0.08)
    assert selection.config.n_hidden == 64
    assert selection.config.n_hidden_layers == 1
    assert selection.config.hidden_fanin == 0
    assert selection.config.input_to_motor_skip is False


@pytest.mark.unit
def test_apply_evolved_config_full_mode_enables_partial_resume_for_shape_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_evolved_config_module(monkeypatch)
    checkpoint = tmp_path / "run" / "evolution" / "checkpoint.json"
    _write_evolution_checkpoint(
        checkpoint,
        _genome_with(
            {
                "n_hidden_layers": 3,
                "hidden_fanin": 24,
                "input_to_motor_skip": True,
            }
        ),
    )
    base = GameTrainingConfig(resume=True, resume_allow_partial=False)

    selection = module.apply_evolved_genome_config(
        base,
        use_evolved=True,
        evolved_mode="full",
        evolution_checkpoint=str(checkpoint),
        runs_dir=tmp_path,
    )

    assert selection.applied is True
    assert selection.include_network_shape is True
    assert selection.partial_resume_enabled is True
    assert selection.config.n_hidden_layers == 3
    assert selection.config.hidden_fanin == 24
    assert selection.config.input_to_motor_skip is True
    assert selection.config.resume_allow_partial is True
    assert module.evolved_config_status_lines(selection, action="eval")[-1] == (
        "Evolved genome mode=full changes network shape; "
        "partial checkpoint warm-start enabled."
    )


@pytest.mark.unit
def test_apply_evolved_config_reports_missing_and_invalid_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_evolved_config_module(monkeypatch)
    base = GameTrainingConfig(seed=77, n_hidden_layers=1)

    missing = module.apply_evolved_genome_config(
        base,
        use_evolved=True,
        evolved_mode="full",
        evolution_checkpoint=None,
        runs_dir=tmp_path / "missing_runs",
    )
    assert missing.applied is False
    assert missing.config is base
    assert module.evolved_config_status_lines(missing, action="training") == [
        "WARNING: no evolution checkpoint found; using base SMB3 config."
    ]

    checkpoint = tmp_path / "run" / "evolution" / "checkpoint.json"
    _write_evolution_checkpoint(
        checkpoint,
        _genome_with({"decide_ticks": 9, "n_hidden_layers": 3}),
    )
    invalid = module.apply_evolved_genome_config(
        base,
        use_evolved=True,
        evolved_mode="wide",
        evolution_checkpoint=str(checkpoint),
        runs_dir=tmp_path,
    )

    assert invalid.applied is True
    assert invalid.include_network_shape is False
    assert invalid.config.decide_ticks == 9
    assert invalid.config.n_hidden_layers == 1
    assert invalid.warnings == ("unknown evolved mode 'wide'; using compatible mode.",)
