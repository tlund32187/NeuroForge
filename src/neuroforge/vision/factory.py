"""Vision backbone factories resolved from :class:`VisionBackboneSpec`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from neuroforge.vision.backbones import LifConvNetV1, VisionState

if TYPE_CHECKING:
    from neuroforge.network.specs import VisionBackboneSpec

__all__ = [
    "LifConvNetV1",
    "LIFConvNetV1BackboneFactory",
    "ResolvedVisionBackbone",
    "VisionBackboneFactory",
    "VisionState",
]

@runtime_checkable
class ResolvedVisionBackbone(Protocol):
    """Runtime vision backbone interface produced by factory build()."""

    type: str
    output_dim: int

    def __call__(
        self,
        x: Any,
        state: VisionState | None = None,
    ) -> tuple[Any, VisionState]:
        ...


class VisionBackboneFactory:
    """Base class for factory-hub registered vision backbone factories."""

    backbone_type: str = ""

    def __init__(self, *, spec: VisionBackboneSpec) -> None:
        if not self.backbone_type:
            msg = "VisionBackboneFactory.backbone_type must be set by subclasses"
            raise ValueError(msg)
        if spec.type != self.backbone_type:
            msg = (
                "VisionBackboneFactory received mismatched spec.type; "
                f"expected {self.backbone_type!r}, got {spec.type!r}"
            )
            raise ValueError(msg)
        self._spec = spec

    @property
    def spec(self) -> VisionBackboneSpec:
        """Original validated config spec used for this factory instance."""
        return self._spec

    def build(self) -> ResolvedVisionBackbone:
        """Build a runtime backbone module from the stored spec."""
        msg = "VisionBackboneFactory.build must be implemented by subclasses"
        raise NotImplementedError(msg)


class LIFConvNetV1BackboneFactory(VisionBackboneFactory):
    """Built-in factory for the ``lif_convnet_v1`` backbone type."""

    backbone_type = "lif_convnet_v1"

    def build(self) -> LifConvNetV1:
        return LifConvNetV1(self.spec)
