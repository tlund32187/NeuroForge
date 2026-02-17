# NeuroForge

A biologically inspired spiking **liquid brain** simulation toolkit.

## Goals

- SOLID-friendly modular architecture (contracts + implementations)
- Pluggable neuron models (LIF, GLIF, AdEx, multi-compartment)
- Synapses with receptors (AMPA/NMDA/GABA), learning (STDP, R-STDP, 3-factor, homeostasis)
- Neuromodulators, astrocytes, energy accounting
- 3D spatial wiring with myelinated axonal delays
- Online learning with eligibility traces
- CoDeepNEAT-like neuroevolution with CPPN/HyperNEAT connectivity
- Multi-agent support, working memory across games
- Pixels-only NES gameplay via BizHawk
- Benchmark tasks: logic gates, MNIST, N-MNIST, Poker-DVS

## Installation

```bash
# Core (no torch)
pip install -e .

# With PyTorch
pip install -e ".[torch]"

# Development
pip install -e ".[dev,torch]"
```

## Quick Start

```python
import neuroforge
print(neuroforge.__version__)
```

## Status

Phase 0 — Project skeleton. See CHANGELOG.md for progress.
