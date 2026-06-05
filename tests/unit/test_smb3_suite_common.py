"""Unit tests for shared SMB3 suite helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import neuroforge.applications.smb3.suite_common as common

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
def test_base_env_prepends_local_src(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PYTHONPATH", "existing")

    env = common.base_env(tmp_path)

    assert env["PYTHONPATH"] == f"{tmp_path / 'src'}{common.os.pathsep}existing"


@pytest.mark.unit
def test_evolution_run_dir_env_is_copied(tmp_path: Path) -> None:
    original = {"PYTHONPATH": "x"}
    run_dir = tmp_path / "runs" / "evolve_suite_1"

    env = common.with_evolution_run_dir(original, run_dir)

    assert original == {"PYTHONPATH": "x"}
    assert env[common.EVOLUTION_RUN_DIR_ENV] == str(run_dir)


@pytest.mark.unit
def test_evolution_checkpoint_path_uses_suite_run_dir(tmp_path: Path) -> None:
    run_dir = common.new_evolution_run_dir(tmp_path / "runs", label="evolve_suite")

    assert run_dir.parent == tmp_path / "runs"
    assert run_dir.name.startswith("evolve_suite_")
    assert common.evolution_checkpoint_path(run_dir) == (
        run_dir / "evolution" / "checkpoint.json"
    )
