"""Unit tests for Phase 8 Gate B validator."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

import neuroforge.runners.vision_gate_b as gate_b

if TYPE_CHECKING:
    from pathlib import Path

    from neuroforge.runners.vision_bench import VisionBenchPlan


def _write_config(tmp_path: Path, *, dataset: str = "mnist") -> Path:
    config_path = tmp_path / "vision_gate_b_config.json"
    config_path.write_text(
        json.dumps(
            {
                "artifacts": str(tmp_path / "artifacts"),
                "runner": {
                    "seed": 5,
                    "device": "cpu",
                    "dtype": "float32",
                    "deterministic": True,
                    "benchmark": False,
                    "warn_only": True,
                    "steps": 100,
                    "batch_size": 8,
                    "n_classes": 10,
                    "dataset": dataset,
                    "dataset_root": ".cache/torchvision",
                    "dataset_download": False,
                    "dataset_val_fraction": 0.1,
                    "dataset_num_workers": 0,
                    "dataset_pin_memory": False,
                    "allow_fashion_mnist": False,
                    "dataset_normalize": True,
                    "dataset_random_crop": False,
                    "dataset_crop_padding": 2,
                    "dataset_random_horizontal_flip": False,
                    "image_channels": 1,
                    "image_h": 28,
                    "image_w": 28,
                    "lr": 1e-3,
                    "loss_fn": "bce_logits",
                    "readout": "spike_count",
                    "readout_threshold": 0.0,
                    "backbone_time_steps": 4,
                    "backbone_encoding_mode": "rate",
                },
            }
        ) + "\n",
        encoding="utf-8",
    )
    return config_path


def _fake_result(
    tmp_path: Path,
    plan: VisionBenchPlan,
    *,
    val_acc: float,
    test_acc: float | None,
    steps: int,
    run_failed: bool = False,
    include_confusion: bool = True,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    dataset_tag = str(plan.runner.dataset).strip().lower()
    for idx, seed in enumerate(plan.seeds):
        run_dir = tmp_path / "artifacts" / f"run_fake_{dataset_tag}_{idx}"
        reports_dir = run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = run_dir / "vision" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        if include_confusion:
            (metrics_dir / "confusion_matrix.json").write_text(
                json.dumps({"matrix": [[1, 0], [0, 1]]}) + "\n",
                encoding="utf-8",
            )
        runs.append(
            {
                "seed": int(seed),
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "dataset": str(plan.runner.dataset),
                "failed": bool(run_failed),
                "error": "simulated_failure" if run_failed else None,
                "steps": int(steps),
                "determinism": {
                    "seed": int(seed),
                    "deterministic": bool(plan.runner.deterministic),
                    "benchmark": bool(plan.runner.benchmark),
                    "warn_only": bool(plan.runner.warn_only),
                },
                "best_metrics": {
                    "val_accuracy": float(val_acc),
                    "test_accuracy": None if test_acc is None else float(test_acc),
                    "accuracy": float(val_acc),
                },
                "final_metrics": {
                    "accuracy": float(val_acc),
                },
                "throughput": {
                    "steps_per_sec": 11.0,
                },
            }
        )

    aggregate_path = (
        tmp_path / "artifacts" / f"run_fake_{dataset_tag}_0"
        / "reports" / "multi_seed_aggregate.json"
    )
    aggregate_path.write_text(
        json.dumps({"metrics": {"best_val_accuracy": {"mean": val_acc, "std": 0.0}}}) + "\n",
        encoding="utf-8",
    )
    return {
        "config_path": plan.config_path,
        "artifacts": plan.artifacts,
        "seeds": [int(seed) for seed in plan.seeds],
        "runs": runs,
        "aggregate_path": str(aggregate_path),
        "aggregate": {"metrics": {"best_val_accuracy": {"mean": val_acc, "std": 0.0}}},
    }


@pytest.mark.unit
def test_gate_b_validation_writes_fail_report_when_accuracy_below_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path, dataset="mnist")

    def _fake_run(plan: VisionBenchPlan) -> dict[str, Any]:
        return _fake_result(tmp_path, plan, val_acc=0.982, test_acc=0.981, steps=100)

    monkeypatch.setattr(gate_b, "run_vision_benchmark", _fake_run)

    gate_pass, report_path, report = gate_b.run_gate_b_validation(
        config_path=config_path,
        seeds_override=None,
        n_seeds=3,
        min_accuracy_pct=99.0,
        max_budget_steps=150,
        artifacts_override=None,
    )

    assert not gate_pass
    assert report["status"] == "FAIL"
    assert report_path.exists()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["summary"]["b1_pass"] is False
    assert saved["summary"]["b2_pass"] is False
    assert saved["summary"]["gate_pass"] is False
    assert saved["b1"]["summary"]["total_runs"] == 3
    assert saved["b2"]["summary"]["total_runs"] == 3
    assert "poker_dvs" in saved


@pytest.mark.unit
def test_gate_b_validation_passes_when_all_runs_meet_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path, dataset="mnist")

    def _fake_run(plan: VisionBenchPlan) -> dict[str, Any]:
        return _fake_result(tmp_path, plan, val_acc=0.992, test_acc=0.991, steps=80)

    monkeypatch.setattr(gate_b, "run_vision_benchmark", _fake_run)

    gate_pass, report_path, report = gate_b.run_gate_b_validation(
        config_path=config_path,
        seeds_override=(1, 2),
        n_seeds=3,
        min_accuracy_pct=99.0,
        max_budget_steps=100,
        artifacts_override=None,
    )

    assert gate_pass
    assert report["status"] == "PASS"
    assert report["seeds"] == [1, 2]
    assert report["summary"]["b1_pass"] is True
    assert report["summary"]["b2_pass"] is True
    assert report["b1"]["summary"]["passed_runs"] == 2
    assert report["b2"]["summary"]["passed_runs"] == 2
    assert report["poker_dvs"]["non_gating"] is True
    assert report_path.exists()


@pytest.mark.unit
def test_gate_b_main_prints_clear_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(tmp_path, dataset="mnist")

    def _fake_run(plan: VisionBenchPlan) -> dict[str, Any]:
        return _fake_result(tmp_path, plan, val_acc=0.995, test_acc=0.994, steps=90)

    monkeypatch.setattr(gate_b, "run_vision_benchmark", _fake_run)

    rc = gate_b._main([
        "--config",
        str(config_path),
        "--n-seeds",
        "2",
        "--min-accuracy-pct",
        "99.0",
        "--max-budget-steps",
        "120",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Gate B: PASS" in out
    assert "B1 (MNIST):" in out
    assert "B2 (NMNIST):" in out
    assert "Poker-DVS (non-gating):" in out
    assert "Report:" in out


@pytest.mark.unit
def test_pokerdvs_is_non_gating_even_when_runs_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path, dataset="mnist")

    def _fake_run(plan: VisionBenchPlan) -> dict[str, Any]:
        dataset = str(plan.runner.dataset).strip().lower()
        if dataset == "pokerdvs":
            return _fake_result(
                tmp_path,
                plan,
                val_acc=0.30,
                test_acc=0.28,
                steps=80,
                run_failed=True,
                include_confusion=False,
            )
        return _fake_result(tmp_path, plan, val_acc=0.995, test_acc=0.994, steps=80)

    monkeypatch.setattr(gate_b, "run_vision_benchmark", _fake_run)

    gate_pass, _report_path, report = gate_b.run_gate_b_validation(
        config_path=config_path,
        seeds_override=(0, 1, 2),
        n_seeds=3,
        min_accuracy_pct=99.0,
        max_budget_steps=100,
        artifacts_override=None,
    )

    poker_summary = report["poker_dvs"]["summary"]
    assert gate_pass
    assert report["status"] == "PASS"
    assert report["summary"]["b1_pass"] is True
    assert report["summary"]["b2_pass"] is True
    assert poker_summary["failed_runs"] == 3
    assert poker_summary["runs_with_confusion"] == 0
