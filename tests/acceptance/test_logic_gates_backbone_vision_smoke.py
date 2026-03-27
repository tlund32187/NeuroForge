"""Acceptance smoke test for logic-gates through the Phase-8 vision pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuroforge.runners.vision_bench import build_plan_from_config, run_vision_benchmark


@pytest.mark.acceptance
def test_logic_gates_backbone_tiny_preset_writes_vision_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    preset_path = repo_root / "configs" / "logic" / "logic_gates_with_backbone_tiny.json"
    assert preset_path.exists(), f"Missing preset: {preset_path}"

    raw = json.loads(preset_path.read_text(encoding="utf-8"))
    runner = dict(raw.get("runner", {}))
    runner["steps"] = 3
    runner["batch_size"] = 8
    runner["device"] = "cpu"
    runner["dataset_root"] = str(tmp_path / "dataset_cache")
    raw["runner"] = runner
    raw["artifacts"] = str(tmp_path / "artifacts")
    raw["seeds"] = [0]

    config_path = tmp_path / "logic_gates_with_backbone_tiny_smoke.json"
    config_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    plan = build_plan_from_config(config_path)
    result = run_vision_benchmark(plan)
    runs = result.get("runs", [])
    assert isinstance(runs, list) and len(runs) == 1

    run_dir = Path(str(runs[0]["run_dir"]))
    metrics_dir = run_dir / "vision" / "metrics"
    samples_dir = run_dir / "vision" / "samples"
    assert (metrics_dir / "confusion_matrix.npy").exists() or (metrics_dir / "confusion_matrix.json").exists()
    assert (metrics_dir / "vision_layer_stats.csv").exists()
    assert (samples_dir / "input_grid.png").exists()

