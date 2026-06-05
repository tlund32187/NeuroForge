"""PerceptionStack (Layer A4 wiring): the brain's invariant visual front-end.

Composes the bio-perception stages into a single drop-in encoder for the spiking
policy (it satisfies ``IFrameEncoder``, like ``FramePreprocessor``):

* **A0 retina** - palette/brightness/contrast-invariant ON/OFF contrast (always
  computed; it is the substrate for A1);
* **A1 STDP features** *(opt-in)* - competitively-learned local feature maps;
* **A2 trace-rule object cells** *(opt-in, needs A1)* - temporally-bound invariant
  object code;
* **A3 motion figure-ground** *(opt-in)* - scroll-compensated sprite saliency.

The learned stages (A1/A2) **train online while encoding** (``learn=True``): each
frame their unsupervised rules update from what the brain just saw, so perception
adapts during play. Their weights persist across episodes (continual learning);
A2's temporal trace and A3's motion reference reset each episode. With everything
off but A0(+A3) this is identical to the earlier fixed front-end (backward
compatible). State is serialisable so it can be checkpointed alongside the policy
- essential so a resumed run's policy and perception stay aligned.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from neuroforge.kernel.torch_utils import require_torch
from neuroforge.perception.vision.encoding.motion_figure_ground import (
    MotionFigureGround,
    MotionFigureGroundConfig,
)
from neuroforge.perception.vision.encoding.retina import RetinaEncoder, RetinaEncoderConfig
from neuroforge.perception.vision.encoding.stdp_features import STDPFeatureConfig, STDPFeatureLayer
from neuroforge.perception.vision.encoding.trace_invariance import (
    TraceInvariantConfig,
    TraceInvariantLayer,
)

__all__ = ["PerceptionStack", "PerceptionStackConfig"]


@dataclass(frozen=True, slots=True)
class PerceptionStackConfig:
    """Configuration for :class:`PerceptionStack`."""

    retina: RetinaEncoderConfig = field(default_factory=RetinaEncoderConfig)
    motion: bool = True                 # append the A3 figure-ground saliency channel
    features: bool = False              # drive with A1 STDP feature maps (instead of raw A0)
    objects: bool = False               # append the A2 object code (needs features)
    feature_cfg: STDPFeatureConfig | None = None   # None -> derived from the retina grid
    n_objects: int = 16
    learn: bool = True                  # online-train A1/A2 while encoding
    amplitude: float = 10.0             # drive scale for normalized learned channels
    motion_amplitude: float = 10.0

    def __post_init__(self) -> None:
        if self.objects and not self.features:
            msg = "PerceptionStackConfig.objects requires features=True"
            raise ValueError(msg)


class PerceptionStack:
    """Bio-perception front-end (A0 + optional A1/A2/A3) as an IFrameEncoder."""

    def __init__(self, config: PerceptionStackConfig | None = None) -> None:
        self._cfg = config or PerceptionStackConfig()
        self._torch = require_torch()
        rcfg = self._cfg.retina
        self._retina = RetinaEncoder(rcfg)
        self._grid = rcfg.out_h * rcfg.out_w

        self._motion = (
            MotionFigureGround(
                MotionFigureGroundConfig(
                    out_h=rcfg.out_h, out_w=rcfg.out_w, device=rcfg.device, dtype=rcfg.dtype,
                ),
            )
            if self._cfg.motion
            else None
        )

        self._a1: STDPFeatureLayer | None = None
        self._a2: TraceInvariantLayer | None = None
        self._fh = self._fw = 0
        if self._cfg.features:
            base = self._cfg.feature_cfg or STDPFeatureConfig(n_features=16, patch=5, stride=4)
            fcfg = dataclasses.replace(
                base, in_channels=2, device=rcfg.device, dtype=rcfg.dtype,
            )
            self._a1 = STDPFeatureLayer(fcfg)
            self._fh = (rcfg.out_h - fcfg.patch) // fcfg.stride + 1
            self._fw = (rcfg.out_w - fcfg.patch) // fcfg.stride + 1
            if self._cfg.objects:
                self._a2 = TraceInvariantLayer(
                    TraceInvariantConfig(
                        n_inputs=fcfg.n_features, n_objects=self._cfg.n_objects,
                        device=rcfg.device, dtype=rcfg.dtype,
                    ),
                )

    @property
    def feature_layer(self) -> STDPFeatureLayer | None:
        """The A1 STDP feature layer (or None if features are disabled)."""
        return self._a1

    @property
    def object_layer(self) -> TraceInvariantLayer | None:
        """The A2 trace-rule object layer (or None if objects are disabled)."""
        return self._a2

    @property
    def input_size(self) -> int:
        if self._a1 is not None:
            size = self._a1.n_features * self._fh * self._fw
            if self._a2 is not None:
                size += self._a2.n_objects
        else:
            size = self._retina.input_size
        if self._motion is not None:
            size += self._grid
        return size

    def reset(self) -> None:
        """Reset per-episode state: motion reference + A2 trace. A1 weights persist."""
        self._retina.reset()
        if self._motion is not None:
            self._motion.reset()
        if self._a2 is not None:
            self._a2.reset()

    def to_drive(self, frame: Any) -> Any:
        torch = self._torch
        contrast = self._retina.to_drive(frame)            # [2*grid], ~amplitude-scaled
        parts: list[Any] = []

        if self._a1 is not None:
            chw = contrast.reshape(2, self._cfg.retina.out_h, self._cfg.retina.out_w)
            # Read the representation the policy acts on this frame *before* learning.
            fmap = self._a1.feature_map(chw).reshape(-1)
            parts.append(self._scaled(fmap))
            pooled = self._a1.encode(chw) if self._a2 is not None else None
            if self._a2 is not None:
                parts.append(self._scaled(self._a2.encode(pooled)))
            if self._cfg.learn:
                self._a1.train_on_contrast(chw)
                if self._a2 is not None:
                    self._a2.observe(pooled)
        else:
            parts.append(contrast)

        if self._motion is not None:
            saliency = self._motion.saliency(frame).reshape(-1) * self._cfg.motion_amplitude
            parts.append(saliency)

        return torch.cat([p.to(contrast.device, contrast.dtype) for p in parts])

    #

    def state_dict(self) -> dict[str, Any]:
        """Serialisable learned state of A1/A2 (empty if neither is enabled)."""
        out: dict[str, Any] = {}
        if self._a1 is not None:
            out["a1"] = self._a1.state_dict()
        if self._a2 is not None:
            out["a2"] = self._a2.state_dict()
        return out

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore A1/A2 learned state produced by :meth:`state_dict`."""
        if self._a1 is not None and isinstance(state.get("a1"), dict):
            self._a1.load_state_dict(state["a1"])
        if self._a2 is not None and isinstance(state.get("a2"), dict):
            self._a2.load_state_dict(state["a2"])

    #

    def _scaled(self, values: Any) -> Any:
        """Normalize a learned channel to ~[0, amplitude] for a firing-able drive."""
        peak = values.max()
        return values / (peak + 1e-6) * self._cfg.amplitude
