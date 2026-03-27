"""Acceptance smoke tests for the vision benchmark harness CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.acceptance
def test_vision_bench_multi_seed_sweep_writes_aggregate(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    artifacts_dir = tmp_path / "artifacts"
    config_path = tmp_path / "vision_bench_config.json"
    config_path.write_text(
        json.dumps(
            {
                "artifacts": str(artifacts_dir),
                "runner": {
                    "device": "cpu",
                    "dtype": "float32",
                    "deterministic": True,
                    "benchmark": False,
                    "warn_only": True,
                    "steps": 2,
                    "batch_size": 4,
                    "n_classes": 3,
                    "dataset": "synthetic",
                    "image_channels": 1,
                    "image_h": 8,
                    "image_w": 8,
                    "backbone_time_steps": 2,
                    "backbone_encoding_mode": "rate",
                },
            }
        ) + "\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    src_dir = str(repo_root / "src")
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir if not py_path else f"{src_dir}{os.pathsep}{py_path}"

    cmd = [
        sys.executable,
        "-m",
        "neuroforge.runners.vision_bench",
        "--config",
        str(config_path),
        "--seeds",
        "0,1,2",
    ]
    completed = subprocess.run(  # noqa: S603
        cmd,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )

    assert completed.returncode == 0, (
        "vision_bench command failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}\n"
    )

    run_dirs = sorted(p for p in artifacts_dir.glob("run_*") if p.is_dir())
    assert len(run_dirs) == 3
    for run_dir in run_dirs:
        summary_path = run_dir / "reports" / "benchmark_summary.json"
        perf_path = run_dir / "reports" / "perf_summary.json"
        assert summary_path.exists(), f"Missing benchmark summary: {summary_path}"
        assert perf_path.exists(), f"Missing perf summary: {perf_path}"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        determinism = summary["determinism"]
        assert determinism["deterministic"] is True
        assert determinism["benchmark"] is False
        assert determinism["warn_only"] is True
        throughput = summary["throughput"]
        assert isinstance(throughput["steps_per_sec"], (int, float))
        assert isinstance(throughput["samples_per_sec"], (int, float))
        assert isinstance(throughput["ms_per_step"], (int, float))
        train_perf = throughput["train"]
        eval_perf = throughput["eval"]
        assert isinstance(train_perf["steps_per_sec"], (int, float))
        assert isinstance(train_perf["samples_per_sec"], (int, float))
        assert isinstance(eval_perf["steps_per_sec"], (int, float))
        assert isinstance(eval_perf["samples_per_sec"], (int, float))
        if "gpu_memory" in summary:
            gpu_memory = summary["gpu_memory"]
            assert isinstance(gpu_memory["max_allocated_bytes"], int)
            assert isinstance(gpu_memory["max_reserved_bytes"], int)

    aggregate_paths = sorted(
        p for p in artifacts_dir.glob("run_*/reports/multi_seed_aggregate.json")
        if p.is_file()
    )
    assert len(aggregate_paths) == 1

    aggregate = json.loads(aggregate_paths[0].read_text(encoding="utf-8"))
    det = aggregate["determinism"]
    assert det["deterministic"] is True
    assert det["benchmark"] is False
    assert det["warn_only"] is True
    metrics = aggregate["metrics"]
    val_stats = metrics["best_val_accuracy"]
    throughput_stats = metrics["throughput_steps_per_sec"]
    throughput_samples_stats = metrics["throughput_samples_per_sec"]
    assert isinstance(val_stats["mean"], (int, float))
    assert isinstance(val_stats["std"], (int, float))
    assert isinstance(throughput_stats["mean"], (int, float))
    assert isinstance(throughput_stats["std"], (int, float))
    assert isinstance(throughput_samples_stats["mean"], (int, float))
    assert isinstance(throughput_samples_stats["std"], (int, float))
