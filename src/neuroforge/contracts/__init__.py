"""Protocol and DTO contracts for NeuroForge."""

from neuroforge.contracts.factories import IFactory
from neuroforge.contracts.registries import IRegistry
from neuroforge.contracts.runtime import IRuntimeComponent
from neuroforge.contracts.tensors import Tensor

__all__ = [
    "IFactory",
    "IRegistry",
    "IRuntimeComponent",
    "Tensor",
]
