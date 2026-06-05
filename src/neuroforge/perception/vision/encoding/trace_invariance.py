"""Trace-rule invariance (Layer A2): stable object codes across frames/worlds.

The object-identity stage. Objects move and recolor continuously, so their
*identity* persists across consecutive frames even as their appearance (position,
palette, small deformation) changes. The **trace rule** (Földiák 1991; VisNet,
Wallis & Rolls) exploits exactly this: learning binds inputs that occur close
together in *time* to the same output cell, so each cell becomes invariant to the
transformations that happen within those short windows.

Concretely, a bank of "object cells" competes for the A1 feature vector each
frame (winner-take-all), but the competition is biased by a decaying **trace** of
who won recently — so a temporally-continuous trajectory of appearances is
captured by one cell, which then responds invariantly to all of them. With the
trace disabled (``trace_gain=0``) this degrades to plain competitive clustering,
which splits a morphing object across several cells — the trace is what yields
*invariance* rather than mere categorization. No labels; no prepopulated shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuroforge.kernel.torch_utils import require_torch, resolve_device_dtype

__all__ = ["TraceInvariantConfig", "TraceInvariantLayer"]


@dataclass(frozen=True, slots=True)
class TraceInvariantConfig:
    """Configuration for :class:`TraceInvariantLayer`."""

    n_inputs: int               # dim of the input feature vector (e.g. A1 features)
    n_objects: int = 16         # invariant object cells
    lr: float = 0.05
    trace_decay: float = 0.5    # trace EMA rate in (0, 1]; higher = shorter memory
    trace_gain: float = 2.0     # how strongly the trace biases competition (0 = clustering)
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.n_inputs < 1:
            msg = "TraceInvariantConfig.n_inputs must be >= 1"
            raise ValueError(msg)
        if self.n_objects < 2:
            msg = "TraceInvariantConfig.n_objects must be >= 2"
            raise ValueError(msg)
        if not 0.0 < self.trace_decay <= 1.0:
            msg = "TraceInvariantConfig.trace_decay must be in (0, 1]"
            raise ValueError(msg)
        if self.trace_gain < 0.0:
            msg = "TraceInvariantConfig.trace_gain must be >= 0"
            raise ValueError(msg)


class TraceInvariantLayer:
    """Learn temporally-invariant object cells from a stream of feature vectors."""

    def __init__(self, config: TraceInvariantConfig) -> None:
        self._cfg = config
        self._torch = require_torch()
        self._dev, self._dtype = resolve_device_dtype(config.device, config.dtype)
        gen = self._torch.Generator(device="cpu")
        gen.manual_seed(int(config.seed))
        w = self._torch.rand(config.n_objects, config.n_inputs, generator=gen, dtype=self._dtype)
        self._w = w.to(self._dev)
        self._trace = self._torch.zeros(config.n_objects, dtype=self._dtype, device=self._dev)

    @property
    def n_objects(self) -> int:
        return self._cfg.n_objects

    @property
    def weights(self) -> Any:
        return self._w.clone()

    def reset(self) -> None:
        """Clear the temporal trace (call at episode/sequence boundaries)."""
        self._trace.zero_()

    def state_dict(self) -> dict[str, Any]:
        """Serialisable learned state (weights; the trace is per-episode)."""
        return {"w": self._w.detach().cpu().clone()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore learned state produced by :meth:`state_dict`."""
        if "w" in state:
            self._w = state["w"].to(self._dev, self._dtype)

    def responses(self, x: Any) -> Any:
        """Norm-corrected match of input *x* ``[n_inputs]`` to each cell ``[n_objects]``."""
        norms = self._w.norm(dim=1) + 1e-6
        return (self._w @ x.reshape(-1).to(self._dtype)) / norms

    def winner(self, x: Any) -> int:
        """Invariant cell index for *x* (no trace bias — pure identity readout)."""
        return int(self.responses(x).argmax())

    def observe(self, x: Any) -> int:
        """Present one frame's features; learn (trace rule) and return the winner.

        The winner is chosen with the trace bias (temporal continuity), then its
        weights move toward the current input and the trace is updated.
        """
        cfg = self._cfg
        vec = x.reshape(-1).to(self._dtype).to(self._dev)
        biased = self.responses(vec) + cfg.trace_gain * self._trace
        win = int(biased.argmax())

        self._w[win] = self._w[win] + cfg.lr * (vec - self._w[win])
        self._trace.mul_(1.0 - cfg.trace_decay)
        self._trace[win] += cfg.trace_decay
        return win

    def encode(self, x: Any) -> Any:
        """Invariant code for *x* (the per-cell response vector ``[n_objects]``)."""
        return self.responses(x)
