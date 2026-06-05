"""Tests for perception package boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_perception_tree_contains_expected_modules() -> None:
    root = Path(__file__).parents[2] / "src" / "neuroforge" / "perception"
    expected = {
        "__init__.py",
        "vision/__init__.py",
        "vision/factory.py",
        "vision/registry.py",
        "vision/classification_engine.py",
        "vision/blocks/__init__.py",
        "vision/backbones/__init__.py",
        "vision/encoding/__init__.py",
    }

    missing = [path for path in sorted(expected) if not (root / path).is_file()]
    assert missing == []


@pytest.mark.unit
def test_vision_registry_contains_builtin_backbone() -> None:
    from neuroforge.perception.vision.registry import build_vision_backbone_registry

    registry = build_vision_backbone_registry()

    assert registry.has("lif_convnet_v1")
