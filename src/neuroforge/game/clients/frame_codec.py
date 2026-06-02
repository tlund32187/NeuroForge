"""Decode emulator PNG screenshots into raw frame bytes.

BizHawk's reliable Lua capture is a PNG screenshot, so the ``png`` wire format
carries a complete PNG that this module turns into the row-major ``H×W×C``
bytes a :class:`~neuroforge.contracts.game.ScreenFrame` expects. The ``raw``
format bypasses this entirely. Torch/torchvision are imported lazily so the
rest of the bridge stays dependency-light and importable without them.
"""

from __future__ import annotations

from typing import Any

__all__ = ["decode_png_to_raw"]

_RGBA = 4
_RGB = 3
_GRAY = 1


def decode_png_to_raw(data: bytes, *, width: int, height: int, channels: int) -> bytes:
    """Decode a PNG into row-major ``height*width*channels`` bytes.

    Performs the few channel conversions a NES capture needs (RGBA→RGB,
    RGB→grayscale, gray→RGB). Raises :class:`ValueError` if the decoded image
    dimensions do not match ``(height, width)``.
    """
    from neuroforge.core.torch_utils import require_torch

    torch: Any = require_torch()
    try:
        from torchvision.io import decode_png  # pyright: ignore[reportMissingTypeStubs]
    except ImportError as exc:  # pragma: no cover - exercised only without torchvision
        msg = "PNG frame format requires torchvision (pip install '.[vision]')"
        raise RuntimeError(msg) from exc

    buffer = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    image: Any = decode_png(buffer)  # [C, H, W] uint8
    decoded_c, decoded_h, decoded_w = (
        int(image.shape[0]),
        int(image.shape[1]),
        int(image.shape[2]),
    )
    if decoded_h != height or decoded_w != width:
        msg = f"decoded PNG is {decoded_w}x{decoded_h}, expected {width}x{height}"
        raise ValueError(msg)

    converted = _convert_channels(image, decoded_c, channels, torch=torch)
    # [C, H, W] -> [H, W, C] row-major bytes.
    hwc = converted.permute(1, 2, 0).contiguous()
    return bytes(hwc.flatten().tolist())


def _convert_channels(image: Any, src: int, dst: int, *, torch: Any) -> Any:
    """Convert a ``[C,H,W]`` uint8 tensor from *src* to *dst* channels."""
    if src == dst:
        return image
    if src == _RGBA and dst == _RGB:
        return image[:3]
    if src == _RGB and dst == _GRAY:  # luminance (BT.601)
        weights = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
        gray = (image.float() * weights).sum(dim=0, keepdim=True)
        return gray.round().clamp(0, 255).to(torch.uint8)
    if src == _GRAY and dst == _RGB:
        return image.repeat(3, 1, 1)
    if src == _RGBA and dst == _GRAY:
        return _convert_channels(image[:3], _RGB, _GRAY, torch=torch)
    msg = f"unsupported channel conversion: {src} -> {dst}"
    raise ValueError(msg)
