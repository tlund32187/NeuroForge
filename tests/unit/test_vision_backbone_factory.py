"""Unit tests for vision backbone specs and FactoryHub resolution."""

from __future__ import annotations

import pytest
import torch

from neuroforge.factories.hub import build_default_hub
from neuroforge.network.specs import VisionBackboneSpec, VisionBlockSpec, VisionInputSpec
from neuroforge.vision.factory import (
    LifConvNetV1,
    LIFConvNetV1BackboneFactory,
    ResolvedVisionBackbone,
)
from neuroforge.vision.registry import create_vision_backbone


@pytest.mark.unit
def test_vision_backbone_resolves_via_factory_hub() -> None:
    """A validated VisionBackboneSpec resolves from hub registry by ``spec.type``."""
    spec = VisionBackboneSpec(
        type="lif_convnet_v1",
        input=VisionInputSpec(channels=3, height=32, width=32),
        time_steps=6,
        encoding_mode="rate",
        blocks=[
            VisionBlockSpec(type="conv", params={"out_channels": 16, "kernel_size": 3}),
            VisionBlockSpec(type="pool", params={"kernel_size": 2}),
            VisionBlockSpec(type="res", params={"channels": 16, "depth": 2}),
        ],
        output_dim=64,
    )
    hub = build_default_hub()
    factory = hub.vision_backbones.create(spec.type, spec=spec)
    assert isinstance(factory, LIFConvNetV1BackboneFactory)

    backbone = factory.build()
    assert isinstance(backbone, LifConvNetV1)
    assert isinstance(backbone, ResolvedVisionBackbone)
    assert backbone.type == "lif_convnet_v1"
    assert backbone.input.channels == 3
    assert backbone.time_steps == 6
    assert backbone.encoding_mode == "rate"
    assert backbone.output_dim == 64

    x = torch.rand(2, 3, 32, 32, dtype=torch.float32)
    features, state = backbone(x)
    assert features.shape == (2, 64)
    assert state.time_steps == 6
    assert len(state.per_layer_spike_counts) == len(backbone.block_names)


@pytest.mark.unit
def test_create_vision_backbone_helper() -> None:
    spec = VisionBackboneSpec(
        type="lif_convnet_v1",
        input=VisionInputSpec(channels=1, height=16, width=16),
        time_steps=6,
        encoding_mode="poisson",
        blocks=[VisionBlockSpec(type="conv", params={"out_channels": 8, "kernel_size": 3})],
        output_dim=32,
    )
    backbone = create_vision_backbone(spec)
    x = torch.rand(3, 1, 16, 16, dtype=torch.float32)
    features, state = backbone(x)
    assert backbone.type == "lif_convnet_v1"
    assert backbone.encoding_mode == "poisson"
    assert backbone.output_dim == 32
    assert features.shape == (3, 32)
    for count in state.per_layer_spike_counts.values():
        assert tuple(count.shape) == (3,)


@pytest.mark.unit
def test_lif_convnet_v1_accepts_prebinned_event_tensor_input() -> None:
    spec = VisionBackboneSpec(
        type="lif_convnet_v1",
        input=VisionInputSpec(channels=2, height=8, width=8),
        time_steps=4,
        encoding_mode="rate",
        blocks=[
            VisionBlockSpec(type="conv", params={"out_channels": 6, "kernel_size": 3}),
            VisionBlockSpec(type="pool", params={"kernel_size": 2, "mode": "avg"}),
        ],
        output_dim=12,
    )
    backbone = create_vision_backbone(spec)
    events = torch.rand(5, 4, 2, 8, 8, dtype=torch.float32)
    features, state = backbone(events)
    assert tuple(features.shape) == (5, 12)
    assert state.time_steps == 4


@pytest.mark.unit
def test_vision_backbone_spec_invalid_encoding_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="encoding_mode"):
        VisionBackboneSpec(
            type="lif_convnet_v1",
            input=VisionInputSpec(channels=1, height=8, width=8),
            time_steps=4,
            encoding_mode="invalid_mode",
            blocks=[VisionBlockSpec(type="conv", params={"out_channels": 4, "kernel_size": 3})],
            output_dim=8,
        )


@pytest.mark.unit
def test_vision_block_spec_missing_conv_param_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="out_channels"):
        VisionBlockSpec(type="conv", params={"kernel_size": 3})


@pytest.mark.unit
def test_lif_convnet_v1_reproducible_on_cpu_with_fixed_seed() -> None:
    spec = VisionBackboneSpec(
        type="lif_convnet_v1",
        input=VisionInputSpec(channels=2, height=12, width=12),
        time_steps=5,
        encoding_mode="poisson",
        blocks=[
            VisionBlockSpec(
                type="conv",
                params={"out_channels": 6, "kernel_size": 3, "norm": None},
            ),
            VisionBlockSpec(type="pool", params={"kernel_size": 2, "mode": "avg"}),
            VisionBlockSpec(
                type="res",
                params={"channels": 6, "depth": 1, "norm": None},
            ),
        ],
        output_dim=10,
    )

    torch.manual_seed(2026)
    x = torch.rand(4, 2, 12, 12, dtype=torch.float32)

    torch.manual_seed(100)
    model_a = create_vision_backbone(spec)
    torch.manual_seed(777)
    feat_a, state_a = model_a(x)

    torch.manual_seed(100)
    model_b = create_vision_backbone(spec)
    torch.manual_seed(777)
    feat_b, state_b = model_b(x)

    assert torch.allclose(feat_a, feat_b)
    assert state_a.time_steps == state_b.time_steps
    assert set(state_a.per_layer_spike_counts) == set(state_b.per_layer_spike_counts)
    for key in state_a.per_layer_spike_counts:
        assert torch.allclose(
            state_a.per_layer_spike_counts[key],
            state_b.per_layer_spike_counts[key],
        )
