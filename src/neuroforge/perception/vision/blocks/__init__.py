"""Minimal spiking vision block building blocks."""

from neuroforge.perception.vision.blocks.conv import SpikingConvBlock, build_2d_norm
from neuroforge.perception.vision.blocks.pool import SpikingPool
from neuroforge.perception.vision.blocks.res import SpikingResBlock
from neuroforge.perception.vision.blocks.spike import SurrogateSpike

__all__ = [
    "SpikingConvBlock",
    "SpikingPool",
    "SpikingResBlock",
    "SurrogateSpike",
    "build_2d_norm",
]

