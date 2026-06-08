"""Tests for STDP competitive feature maps (Layer A1).

The core claim is *unsupervised differentiation*: presented several distinct
local patterns, the bank must learn to fire a *different* winner feature for
each (competition / lateral inhibition), and each winner must become selective
to its pattern (STDP). Plus determinism, homeostasis (no dead-unit collapse),
and that it chains onto A0's contrast output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from neuroforge.contracts.applications.games import ScreenFrame
from neuroforge.perception.vision.encoding import (
    RetinaEncoder,
    STDPFeatureConfig,
    STDPFeatureLayer,
    render_feature_atlas,
)

if TYPE_CHECKING:
    from torch import Tensor

pytest.importorskip("torch")

_PATCH = 5
_CH = 2
_DIM = _CH * _PATCH * _PATCH  # 50


def _disjoint_patterns(k: int) -> Tensor:
    """k maximally-separable patterns: each lights a disjoint block of inputs."""
    pats = torch.zeros(k, _DIM)
    chunk = _DIM // k
    for i in range(k):
        pats[i, i * chunk : (i + 1) * chunk] = 1.0
    return pats


def _train(layer: STDPFeatureLayer, patterns: Tensor, *, batches: int, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    k = patterns.shape[0]
    for _ in range(batches):
        idx = torch.randint(0, k, (32,), generator=gen)
        noise = torch.rand(32, _DIM, generator=gen) * 0.1
        layer.partial_fit((patterns[idx] + noise).clamp(0.0, 1.0))


@pytest.mark.unit
def test_distinct_patterns_get_distinct_winners() -> None:
    patterns = _disjoint_patterns(4)
    layer = STDPFeatureLayer(STDPFeatureConfig(n_features=24, patch=_PATCH, seed=1))
    _train(layer, patterns, batches=300, seed=7)

    resp = layer.responses(patterns)          # [4, F]
    winners = resp.argmax(dim=1).tolist()
    assert len(set(winners)) >= 3             # competition differentiated the features


@pytest.mark.unit
def test_winner_features_are_selective() -> None:
    patterns = _disjoint_patterns(4)
    layer = STDPFeatureLayer(STDPFeatureConfig(n_features=24, patch=_PATCH, seed=1))
    _train(layer, patterns, batches=300, seed=7)

    resp = layer.responses(patterns)          # [4, F]
    baseline = float(resp.mean())
    for i in range(4):
        winner = int(resp[i].argmax())
        # The winning feature responds to its own pattern more than to the average.
        assert float(resp[i, winner]) > baseline


@pytest.mark.unit
def test_homeostasis_keeps_multiple_features_in_use() -> None:
    patterns = _disjoint_patterns(4)
    layer = STDPFeatureLayer(
        STDPFeatureConfig(n_features=24, patch=_PATCH, conscience=0.5, seed=1),
    )
    _train(layer, patterns, batches=300, seed=7)
    used = int((layer._win_count > 0).sum())
    assert used >= 3                           # not collapsed onto a single unit


@pytest.mark.unit
def test_training_is_deterministic() -> None:
    patterns = _disjoint_patterns(4)
    a = STDPFeatureLayer(STDPFeatureConfig(n_features=16, patch=_PATCH, seed=3))
    b = STDPFeatureLayer(STDPFeatureConfig(n_features=16, patch=_PATCH, seed=3))
    _train(a, patterns, batches=50, seed=11)
    _train(b, patterns, batches=50, seed=11)
    assert torch.allclose(a.weights, b.weights)


@pytest.mark.unit
def test_feature_atlas_renders_a_grid_image() -> None:
    layer = STDPFeatureLayer(STDPFeatureConfig(n_features=16, patch=_PATCH))
    img = render_feature_atlas(layer, scale=4)
    assert img.dtype == torch.uint8
    assert img.ndim == 2
    # 16 features -> 4x4 grid; tile = 5*4 = 20; with 1px separators: 4*20 + 5 = 85.
    assert tuple(img.shape) == (85, 85)
    assert torch.equal(img, render_feature_atlas(layer, scale=4))  # deterministic


@pytest.mark.unit
def test_chains_onto_retina_contrast_output() -> None:
    # A0 -> A1: encode a frame to contrast channels, then run the feature layer.
    enc = RetinaEncoder()
    out_h, out_w = 28, 32
    raw = torch.randint(0, 256, (out_h * 4, out_w * 4, 3), dtype=torch.uint8)
    frame = ScreenFrame(
        width=out_w * 4, height=out_h * 4, channels=3, data=bytes(raw.flatten().tolist()),
    )
    drive = enc.to_drive(frame)               # [2*28*32]
    contrast = drive.reshape(2, out_h, out_w)  # ON/OFF channels

    layer = STDPFeatureLayer(STDPFeatureConfig(n_features=12, patch=_PATCH, stride=2))
    layer.train_on_contrast(contrast)
    fmap = layer.feature_map(contrast)
    assert fmap.shape[0] == 12
    assert tuple(layer.encode(contrast).shape) == (12,)
