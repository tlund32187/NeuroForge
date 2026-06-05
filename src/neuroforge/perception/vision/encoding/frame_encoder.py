"""The frame-encoder seam: pixels -> input-population drive current.

Both the simple :class:`~neuroforge.perception.vision.encoding.frame_preprocess.FramePreprocessor`
(raw rate code) and the biologically-faithful perception stack
(:class:`~neuroforge.perception.vision.encoding.PerceptionStack`: retina + motion) satisfy
this protocol structurally, so the spiking policy can be driven by either
without change. This is the seam A4 introduces so the invariant perception can
replace raw pixels as the brain's input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neuroforge.contracts.applications.games import ScreenFrame

__all__ = ["IFrameEncoder"]


@runtime_checkable
class IFrameEncoder(Protocol):
    """Turn a :class:`ScreenFrame` into a 1-D drive current for the input layer."""

    @property
    def input_size(self) -> int:
        """Number of input neurons this encoder drives."""
        ...

    def to_drive(self, frame: ScreenFrame) -> Any:
        """Return a 1-D drive tensor ``[input_size]``."""
        ...

    def reset(self) -> None:
        """Clear per-episode state (call at episode start)."""
        ...
