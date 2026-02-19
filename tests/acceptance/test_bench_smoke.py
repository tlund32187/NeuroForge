"""Acceptance smoke test for the benchmark CLI command."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.acceptance
def test_bench_cli_smoke_writes_summary(tmp_path: Path) -> None:
    """`neuroforge bench` should complete and write benchmark artifacts."""
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
        "bench",
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--n-input",
        "64",
        "--n-hidden",
        "64",
        "--n-output",
        "1",
        "--topology",
        "sparse_random",
        "--p-connect",
        "0.2",
        "--steps",
        "50",
        "--warmup",
        "10",
        "--artifacts",
        str(artifacts_dir),
    ]
    completed = subprocess.run(  # noqa: S603
        cmd,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, (
        "bench command failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}\n"
    )

    run_dirs = sorted(p for p in artifacts_dir.glob("run_*") if p.is_dir())
    assert run_dirs, "No run directory was created by bench command."
    run_dir = run_dirs[-1]

    summary_path = run_dir / "bench" / "bench_summary.json"
    assert summary_path.exists(), "bench_summary.json was not written."

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert float(summary["steps_per_sec"]) > 0.0
    assert float(summary["ms_per_step"]) > 0.0

    scalars_path = run_dir / "metrics" / "scalars.csv"
    assert scalars_path.exists(), "scalars.csv was not written."
    with scalars_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        assert rows, "Expected at least one scalar row."
        assert reader.fieldnames is not None
        assert "perf.steps_per_sec" in reader.fieldnames
