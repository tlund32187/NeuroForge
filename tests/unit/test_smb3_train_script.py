"""Unit tests for SMB3 training script environment configuration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


def _load_train_script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    del monkeypatch
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "neuroforge"
        / "applications"
        / "smb3"
        / "train.py"
    )
    spec = importlib.util.spec_from_file_location(
        "train_smb3_test_module",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_train_script_reads_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "policy.pt"
    monkeypatch.setenv("NEUROFORGE_SMB3_TRAIN_EPISODES", "2")
    monkeypatch.setenv("NEUROFORGE_SMB3_TRAIN_FRAMES", "120")
    monkeypatch.setenv("NEUROFORGE_SMB3_TRAIN_TELEMETRY_EVERY", "0")
    monkeypatch.setenv("NEUROFORGE_SMB3_CHECKPOINT_EVERY", "30")
    monkeypatch.setenv("NEUROFORGE_SMB3_RESUME", "0")
    monkeypatch.setenv("NEUROFORGE_SMB3_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("NEUROFORGE_SMB3_CONSOLIDATION_STRENGTH", "0.125")
    monkeypatch.setenv("NEUROFORGE_SMB3_PORT", "8765")
    monkeypatch.setenv("NEUROFORGE_SMB3_FRAMESKIP", "8")
    monkeypatch.setenv("NEUROFORGE_SMB3_SPEED_PERCENT", "500")
    monkeypatch.setenv("NEUROFORGE_SMB3_USE_EVOLVED", "1")

    module = _load_train_script(monkeypatch)

    assert module.PORT == 8765
    assert module.MAX_EPISODES == 2
    assert module.FRAMES_PER_EPISODE == 120
    assert module.TELEMETRY_EVERY == 0
    assert module.FRAMESKIP == 8
    assert module.BIZHAWK_SPEED_PERCENT == 500
    assert module.CHECKPOINT_EVERY == 30
    assert module.RESUME is False
    assert checkpoint == module.CHECKPOINT
    assert module.USE_EVOLVED_GENOME is True
    assert module.LAMARCKIAN_WRITEBACK is False
    assert pytest.approx(0.125) == module.CONSOLIDATION_STRENGTH


@pytest.mark.unit
def test_train_script_rejects_invalid_small_env_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEUROFORGE_SMB3_PORT", "0")
    monkeypatch.setenv("NEUROFORGE_SMB3_TRAIN_EPISODES", "0")
    monkeypatch.setenv("NEUROFORGE_SMB3_TRAIN_FRAMES", "0")
    monkeypatch.setenv("NEUROFORGE_SMB3_TRAIN_TELEMETRY_EVERY", "-1")
    monkeypatch.setenv("NEUROFORGE_SMB3_FRAMESKIP", "0")
    monkeypatch.setenv("NEUROFORGE_SMB3_SPEED_PERCENT", "0")
    monkeypatch.setenv("NEUROFORGE_SMB3_CHECKPOINT_EVERY", "-1")
    monkeypatch.setenv("NEUROFORGE_SMB3_CONSOLIDATION_STRENGTH", "-0.5")

    module = _load_train_script(monkeypatch)

    assert module.PORT == 8650
    assert module.MAX_EPISODES == 50
    assert module.FRAMES_PER_EPISODE == 2000
    assert module.TELEMETRY_EVERY == 30
    assert module.FRAMESKIP == 4
    assert module.BIZHAWK_SPEED_PERCENT == 400
    assert module.CHECKPOINT_EVERY == 1000
    assert pytest.approx(0.0) == module.CONSOLIDATION_STRENGTH


@pytest.mark.unit
def test_train_script_lamarckian_writeback_defaults_to_full_evolved_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEUROFORGE_SMB3_EVOLVED_MODE", "full")
    module = _load_train_script(monkeypatch)
    assert module.LAMARCKIAN_WRITEBACK is True
