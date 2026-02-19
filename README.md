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

# Optional resource monitoring (CPU/RAM/GPU)
pip install -e ".[monitoring]"
```

## Quick Start

```python
import neuroforge
print(neuroforge.__version__)
```

## Status

Phase 0 - Project skeleton. See CHANGELOG.md for progress.

## Resource Monitoring (Opt-in)

Resource monitoring is disabled by default. Enable it through training config:

```json
{
  "gate": "MULTI",
  "max_trials": 1500,
  "device": "auto",
  "monitoring": {
    "resource": {
      "enabled": true,
      "every_n_steps": 10,
      "include_system": true,
      "include_process": true,
      "include_gpu": true,
      "gpu_index": 0
    }
  }
}
```

When enabled, the dashboard Resources panel shows CPU, RAM, GPU, VRAM, and torch CUDA allocator series in live mode and replay mode.

Optional packages:
- `pip install psutil` for CPU/RAM metrics
- `pip install nvidia-ml-py` for NVIDIA utilization/temperature/power

If optional dependencies are unavailable, metrics are skipped and training continues.
