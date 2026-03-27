"""Acceptance smoke test for the vision CLI command."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.acceptance
def test_vision_cli_runs_multiple_steps(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    artifacts_dir = tmp_path / "artifacts"

    env = dict(os.environ)
    src_dir = str(repo_root / "src")
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir if not py_path else f"{src_dir}{os.pathsep}{py_path}"

    cmd = [
        sys.executable,
        "-m",
        "neuroforge.runners.cli",
        "vision",
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--dataset",
        "synthetic",
        "--steps",
        "3",
        "--batch-size",
        "4",
        "--n-classes",
        "3",
        "--image-channels",
        "1",
        "--image-h",
        "8",
        "--image-w",
        "8",
        "--backbone-time-steps",
        "2",
        "--artifacts",
        str(artifacts_dir),
    ]
    completed = subprocess.run(  # noqa: S603
        cmd,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert completed.returncode == 0, (
        "vision command failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}\n"
    )

    run_dirs = sorted(p for p in artifacts_dir.glob("run_*") if p.is_dir())
    assert run_dirs, "No run directory was created by vision command."
    run_dir = run_dirs[-1]
    summary_path = run_dir / "vision" / "vision_summary.json"
    assert summary_path.exists(), "vision_summary.json was not written."

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = summary["result"]
    assert int(result["steps"]) == 3
    assert len(result["loss_history"]) == 3
    assert len(result["accuracy_history"]) == 3

    assert (run_dir / "vision" / "metrics" / "confusion_matrix.npy").exists()
    assert (run_dir / "vision" / "metrics" / "per_class_accuracy.json").exists()
    assert (run_dir / "vision" / "metrics" / "vision_layer_stats.csv").exists()
    assert (run_dir / "vision" / "samples" / "input_grid.png").exists()
