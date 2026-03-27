"""Unit tests for minimal spiking vision block modules."""

from __future__ import annotations

import pytest
import torch

from neuroforge.vision.blocks import SpikingConvBlock, SpikingPool, SpikingResBlock

_CUDA = torch.cuda.is_available()
_skip_no_cuda = pytest.mark.skipif(not _CUDA, reason="CUDA not available")


def _make_input(
    *,
    batch: int,
    channels: int,
    height: int,
    width: int,
    device: str,
) -> torch.Tensor:
    torch.manual_seed(7)
    x = torch.randn(batch, channels, height, width, dtype=torch.float32, device=device)
    return x


@pytest.mark.unit
def test_spiking_conv_block_shape_stride2() -> None:
    block = SpikingConvBlock(
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        stride=2,
        norm=None,
        spike_threshold=0.0,
    )
    x = _make_input(batch=2, channels=3, height=16, width=16, device="cpu")
    y = block(x)
    assert y.shape == (2, 8, 8, 8)
    assert bool(((y == 0.0) | (y == 1.0)).all())


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["max", "avg"])
def test_spiking_pool_shape(mode: str) -> None:
    pool = SpikingPool(mode=mode, kernel_size=2, stride=2)
    x = _make_input(batch=2, channels=4, height=10, width=12, device="cpu")
    y = pool(x)
    assert y.shape == (2, 4, 5, 6)


@pytest.mark.unit
def test_spiking_res_block_preserves_shape() -> None:
    block = SpikingResBlock(
        in_channels=8,
        out_channels=8,
        kernel_size=3,
        stride=1,
        norm="batch",
        spike_threshold=0.0,
    )
    x = _make_input(batch=2, channels=8, height=16, width=16, device="cpu")
    y = block(x)
    assert y.shape == x.shape


@pytest.mark.unit
def test_spiking_res_block_downsample_shape() -> None:
    block = SpikingResBlock(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        stride=2,
        norm="batch",
        spike_threshold=0.0,
    )
    x = _make_input(batch=2, channels=8, height=16, width=16, device="cpu")
    y = block(x)
    assert y.shape == (2, 16, 8, 8)


@pytest.mark.unit
def test_spiking_conv_block_surrogate_grad_flow() -> None:
    block = SpikingConvBlock(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        norm=None,
        spike_threshold=0.0,
        spike_beta=5.0,
    )
    x = _make_input(batch=1, channels=3, height=8, width=8, device="cpu").requires_grad_(True)
    y = block(x)
    loss = y.mean()
    loss.backward()
    assert x.grad is not None
    assert float(x.grad.abs().sum().item()) > 0.0


@pytest.mark.unit
@pytest.mark.cuda
@_skip_no_cuda
def test_spiking_blocks_forward_cuda() -> None:
    dev = "cuda"
    x = _make_input(batch=1, channels=4, height=12, width=12, device=dev)

    conv = SpikingConvBlock(
        in_channels=4,
        out_channels=8,
        kernel_size=3,
        stride=1,
        norm="batch",
        spike_threshold=0.0,
    ).to(dev)
    pool = SpikingPool(mode="max", kernel_size=2, stride=2).to(dev)
    res = SpikingResBlock(
        in_channels=8,
        out_channels=8,
        kernel_size=3,
        stride=1,
        norm="batch",
        spike_threshold=0.0,
    ).to(dev)

    y = conv(x)
    z = pool(y)
    out = res(y)

    assert y.device.type == "cuda"
    assert z.device.type == "cuda"
    assert out.device.type == "cuda"
    assert y.shape == (1, 8, 12, 12)
    assert z.shape == (1, 8, 6, 6)
    assert out.shape == (1, 8, 12, 12)

