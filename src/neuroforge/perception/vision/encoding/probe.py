"""Evaluation probe for the perception stack (Layer A4).

Perception is trained with NO labels. To answer "did it actually learn useful,
invariant structure?" *without* leaking labels into training, we use a tiny
hand-labeled probe set at *evaluation time only*: encode each frame, then measure
how separable the classes are in the representation (nearest-centroid accuracy).
High accuracy when the same content appears in different palettes is the concrete,
falsifiable signal that the encoding is invariant *and* discriminative.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neuroforge.kernel.torch_utils import require_torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from neuroforge.contracts.applications.games import ScreenFrame
    from neuroforge.perception.vision.encoding.frame_encoder import IFrameEncoder

__all__ = ["nearest_centroid_accuracy", "representation_separability"]


def nearest_centroid_accuracy(codes: Any, labels: Sequence[int]) -> float:
    """Fraction of *codes* classified correctly by nearest class centroid."""
    torch = require_torch()
    label_tensor = torch.tensor(list(labels))
    classes = torch.unique(label_tensor)
    centroids = torch.stack([codes[label_tensor == c].mean(dim=0) for c in classes])
    dists = torch.cdist(codes, centroids)            # [N, n_classes]
    predicted = classes[dists.argmin(dim=1)]
    return float((predicted == label_tensor).float().mean())


def representation_separability(
    encoder: IFrameEncoder,
    frames: Sequence[ScreenFrame],
    labels: Sequence[int],
) -> float:
    """Encode *frames* and return nearest-centroid accuracy against *labels*.

    Each frame is encoded from a reset state, so a stateful (motion) channel reads
    zero — the probe measures the *static* content representation only.
    """
    if len(frames) != len(labels):
        msg = "frames and labels must be the same length"
        raise ValueError(msg)
    torch = require_torch()
    codes: list[Any] = []
    for frame in frames:
        encoder.reset()
        codes.append(encoder.to_drive(frame).detach().reshape(-1).float().cpu())
    return nearest_centroid_accuracy(torch.stack(codes), labels)
