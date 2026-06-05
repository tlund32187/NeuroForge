"""Tests for learning package boundaries."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module_missing(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except ModuleNotFoundError:
        return True


@pytest.mark.unit
def test_learning_tree_contains_expected_modules() -> None:
    root = Path(__file__).parents[2] / "src" / "neuroforge" / "learning"
    expected = {
        "__init__.py",
        "training_loop.py",
        "objectives.py",
        "losses/__init__.py",
        "losses/registry.py",
        "losses/mse.py",
        "losses/bce.py",
        "encoders/__init__.py",
        "encoders/rate.py",
        "encoders/factory.py",
        "readouts/__init__.py",
        "readouts/spike_count.py",
        "readouts/rate_decoder.py",
        "readouts/factory.py",
        "stats.py",
    }

    missing = [path for path in sorted(expected) if not (root / path).is_file()]
    assert missing == []
    assert not (root / "training").exists()


@pytest.mark.unit
def test_learning_canonical_imports() -> None:
    from neuroforge.learning.encoders.factory import build_encoder_registry
    from neuroforge.learning.encoders.rate import RateEncoder
    from neuroforge.learning.losses import BceLogitsLoss, MseCountLoss
    from neuroforge.learning.losses.registry import build_loss_registry
    from neuroforge.learning.objectives import ObjectiveResult
    from neuroforge.learning.readouts.factory import build_readout_registry
    from neuroforge.learning.readouts.rate_decoder import RateDecoder
    from neuroforge.learning.readouts.spike_count import SpikeCountReadout
    from neuroforge.learning.training_loop import OnlineRSTDPTrainer

    assert RateEncoder.__name__ == "RateEncoder"
    assert RateDecoder.__name__ == "RateDecoder"
    assert SpikeCountReadout.__name__ == "SpikeCountReadout"
    assert MseCountLoss.__name__ == "MseCountLoss"
    assert BceLogitsLoss.__name__ == "BceLogitsLoss"
    assert ObjectiveResult.__name__ == "ObjectiveResult"
    assert OnlineRSTDPTrainer.__name__ == "OnlineRSTDPTrainer"
    assert build_encoder_registry().has("rate")
    assert build_readout_registry().has("spike_count")
    assert build_loss_registry().has("mse_count")


@pytest.mark.unit
def test_removed_learning_modules_are_not_present() -> None:
    legacy_modules = [
        "neuroforge.encoding.decode",
        "neuroforge.encoding.losses",
        "neuroforge.encoding.rate",
        "neuroforge.encoding.readout",
        "neuroforge.encoding",
        "neuroforge.losses",
        "neuroforge.losses.factory",
        "neuroforge.losses.registry",
        "neuroforge.losses.spike",
        "neuroforge.readout",
        "neuroforge.readout.factory",
        "neuroforge.readout.rate",
        "neuroforge.readout.registry",
        "neuroforge.readout.spike_count",
        "neuroforge.learning.training",
        "neuroforge.learning.training.online_rstdp",
    ]

    missing = [name for name in legacy_modules if _module_missing(name)]

    assert missing == legacy_modules
