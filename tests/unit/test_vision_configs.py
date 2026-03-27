"""Validation tests for Phase 8 baseline vision configs."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuroforge.runners.vision_bench import build_plan_from_config


@pytest.mark.unit
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_lifconv_v1.json",
        "mnist_lifconv_v1_stronger.json",
    ],
)
def test_phase8_vision_config_parses(config_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "vision" / config_name
    assert config_path.exists(), f"Missing baseline config: {config_path}"

    plan = build_plan_from_config(config_path)
    assert plan.runner.dataset == "mnist"
    assert plan.runner.n_classes == 10
    assert plan.runner.image_channels == 1
    assert plan.runner.image_h == 28
    assert plan.runner.image_w == 28
    assert plan.runner.deterministic is True


@pytest.mark.unit
def test_logic_gates_backbone_tiny_config_parses() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "logic" / "logic_gates_with_backbone_tiny.json"
    assert config_path.exists(), f"Missing logic preset config: {config_path}"

    plan = build_plan_from_config(config_path)
    assert plan.runner.dataset == "logic_gates_pixels"
    assert plan.runner.backbone_type == "lif_convnet_v1"
    assert plan.runner.image_channels == 1
    assert plan.runner.image_h == 8
    assert plan.runner.image_w == 8
