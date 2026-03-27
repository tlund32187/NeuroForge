"""Vision classification engine: backbone + readout head + readout decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from neuroforge.core.torch_utils import require_torch

if TYPE_CHECKING:
    from neuroforge.contracts.encoding import IReadout, ReadoutResult
    from neuroforge.vision.backbones import VisionState
    from neuroforge.vision.factory import ResolvedVisionBackbone

torch = require_torch()
nn = torch.nn

__all__ = ["VisionClassificationEngine", "VisionStepResult"]


@dataclass(frozen=True, slots=True)
class VisionStepResult:
    """Output bundle from one vision classification forward step."""

    readout: ReadoutResult
    logits: Any
    features: Any
    state: VisionState


class VisionClassificationEngine(nn.Module):
    """Owns the forward pipeline ``image -> backbone -> head -> readout``."""

    def __init__(
        self,
        *,
        backbone: ResolvedVisionBackbone,
        n_classes: int,
        readout: IReadout,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            msg = "Resolved vision backbone must be a torch.nn.Module"
            raise TypeError(msg)
        if n_classes <= 1:
            msg = "n_classes must be > 1"
            raise ValueError(msg)
        if not callable(readout):
            msg = "readout must be callable"
            raise TypeError(msg)

        self.backbone = backbone
        self.readout = readout
        self.n_classes = int(n_classes)
        self.head = nn.Linear(int(backbone.output_dim), self.n_classes)

    def forward_step(
        self,
        images: Any,
        *,
        state: VisionState | None = None,
    ) -> VisionStepResult:
        """Run one supervised batch forward step."""
        features, next_state = self.backbone(images, state=state)
        logits = self.head(features)  # [B, C]
        # Existing readout API expects time-first spikes; use a 1-step stream.
        readout = self.readout(logits.unsqueeze(0))
        return VisionStepResult(
            readout=readout,
            logits=logits,
            features=features,
            state=next_state,
        )
