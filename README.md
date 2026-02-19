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

## Phase 6: Stability & Instrumentation

Phase 6 adds opt-in instrumentation for reproducibility, trial-level diagnostics,
stability detection, and multi-seed regression checks.

### Determinism Controls

CLI runs now apply a deterministic mode entrypoint before network/task creation.

- `--seed INT` controls Python/NumPy/Torch seeding.
- `--deterministic` / `--no-deterministic` toggles deterministic algorithms.
- `--benchmark` toggles backend benchmark mode.
- `--warn-only` / `--strict-determinism` controls deterministic-op violations.

Example:

```bash
neuroforge run --task multi_gate --seed 1234 --deterministic --stability --fail-fast
```

### Trial Stats Monitor

`TrialStatsMonitor` enriches `TRAINING_TRIAL` events (and therefore `scalars.csv`)
with rate, sparsity, and convergence counters, including:

- `out_spike_count`, `rate_out_hz`
- `rate_in_mean_hz`, `rate_in_max_hz`, `sparsity_in`
- `rate_hid_mean_hz`, `rate_hid_max_hz`, `sparsity_hid`
- `conv_streak`, `conv_per_pattern_min`, `conv_per_pattern_mean`

Use:

- `--trial-stats` (default on)
- `--no-trial-stats` to disable

### Stability Monitor (+ Fail-Fast)

`StabilityMonitor` injects stability flags into scalar/trial events:

- `stab_nan_inf`
- `stab_weight_explode`
- `stab_rate_saturation`
- `stab_oscillation`
- `stab_stagnation`

Use:

- `--stability` (default on)
- `--no-stability` to disable
- `--stability-every N` to control check cadence
- `--fail-fast` to abort runs on critical instability (`NaN/Inf`, weight explosion)

### Multi-Seed Stability Harness

Use the harness to run multiple seeds and get a JSON stability summary:

```bash
neuroforge stability --task multi_gate --seeds 1,2,3,4,5
```

The report includes per-seed outcomes plus aggregates such as:

- `pass_rate`
- `median_converge`
- `p90_converge`
- `total_flag_counts`
- `critical_flag_counts_successful`

### Interpreting Dashboard + `scalars.csv`

- Live and replay dashboards include a **Stability** panel with:
  - Flag badges (NaN/Inf, weight explode, saturation, oscillation, stagnation)
  - Charts for `rate_out_hz`, `w_maxabs_ih/w_maxabs_ho`, `g_norm_ih/g_norm_ho`, and accuracy
- `artifacts/<run_id>/metrics/scalars.csv` now contains Phase 6 columns.
- Stability flags are numeric (`0` = clear, `1` = detected) for easy filtering/alerting.
- Blank cells mean a metric was not emitted for that step/cadence (not necessarily an error).

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
