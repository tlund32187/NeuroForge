"""Benchmark runner utilities for Phase 7 performance baselines.

This module provides:
- ``build_random_bench_spec`` for synthetic dense/sparse networks
- ``run_bench`` for no-grad step-throughput measurements

The benchmark path is intentionally monitor/event-friendly and performs
no file I/O directly.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

from neuroforge.contracts.types import Compartment
from neuroforge.core.torch_utils import require_torch, resolve_device_dtype
from neuroforge.factories.hub import DEFAULT_HUB
from neuroforge.network.factory import NetworkFactory, to_topology_json
from neuroforge.network.specs import NetworkSpec, PopulationSpec, ProjectionSpec

__all__ = [
    "BenchRunConfig",
    "SUPPORTED_BENCH_TOPOLOGIES",
    "SUPPORTED_BENCH_SYNAPSES",
    "build_random_bench_spec",
    "run_bench",
]

SUPPORTED_BENCH_TOPOLOGIES: tuple[str, ...] = (
    "dense",
    "sparse_random",
    "sparse_fanout",
    "sparse_fanin",
    "block_sparse",
)

SUPPORTED_BENCH_SYNAPSES: tuple[str, ...] = (
    "static",
    "static_dales",
    "static_delayed",
)


@dataclass(frozen=True, slots=True)
class BenchRunConfig:
    """Benchmark configuration for synthetic network performance runs."""

    device: str = "cpu"
    dtype: str = "float32"
    seed: int = 1234
    n_input: int = 64
    n_hidden: int = 256
    n_output: int = 1
    topology: str = "sparse_random"
    synapse_model: str = "static"
    p_connect: float = 0.1
    fanout: int = 16
    fanin: int = 16
    block_pre: int = 16
    block_post: int = 16
    p_block: float = 0.25
    max_delay: int = 3
    steps: int = 2000
    warmup: int = 200
    amplitude: float = 20.0
    sync_cuda_timing: bool = True

    def __post_init__(self) -> None:
        if self.topology not in SUPPORTED_BENCH_TOPOLOGIES:
            msg = f"Unsupported topology: {self.topology!r}"
            raise ValueError(msg)
        if self.synapse_model not in SUPPORTED_BENCH_SYNAPSES:
            msg = f"Unsupported synapse model: {self.synapse_model!r}"
            raise ValueError(msg)
        if self.n_input <= 0:
            msg = "n_input must be > 0"
            raise ValueError(msg)
        if self.n_hidden < 0:
            msg = "n_hidden must be >= 0"
            raise ValueError(msg)
        if self.n_output <= 0:
            msg = "n_output must be > 0"
            raise ValueError(msg)
        if self.steps <= 0:
            msg = "steps must be > 0"
            raise ValueError(msg)
        if self.warmup < 0:
            msg = "warmup must be >= 0"
            raise ValueError(msg)


def _topology_cfg(cfg: BenchRunConfig) -> dict[str, Any]:
    topo: dict[str, Any] = {
        "type": cfg.topology,
        "init": "uniform",
        "low": -0.3,
        "high": 0.3,
        "sort": True,
    }
    if cfg.topology == "sparse_random":
        topo["p_connect"] = float(cfg.p_connect)
    elif cfg.topology == "sparse_fanout":
        topo["fanout"] = int(cfg.fanout)
    elif cfg.topology == "sparse_fanin":
        topo["fanin"] = int(cfg.fanin)
    elif cfg.topology == "block_sparse":
        topo["block_pre"] = int(cfg.block_pre)
        topo["block_post"] = int(cfg.block_post)
        topo["p_block"] = float(cfg.p_block)

    if cfg.synapse_model == "static_delayed":
        if cfg.max_delay > 0:
            topo["delays"] = {"mode": "uniform_int", "max_delay": int(cfg.max_delay)}
        else:
            topo["delays"] = {"mode": "zeros", "max_delay": 0}
    return topo


def build_random_bench_spec(cfg: BenchRunConfig) -> NetworkSpec:
    """Build a synthetic benchmark network specification."""
    topo = _topology_cfg(cfg)

    pops = [
        PopulationSpec("input", int(cfg.n_input), "lif"),
        PopulationSpec("hidden", int(cfg.n_hidden), "lif"),
        PopulationSpec("output", int(cfg.n_output), "lif"),
    ]

    projections: list[ProjectionSpec] = []
    if cfg.n_hidden > 0:
        projections.append(
            ProjectionSpec(
                name="input_hidden",
                source="input",
                target="hidden",
                synapse_model=cfg.synapse_model,
                topology=dict(topo),
            ),
        )
        projections.append(
            ProjectionSpec(
                name="hidden_output",
                source="hidden",
                target="output",
                synapse_model=cfg.synapse_model,
                topology=dict(topo),
            ),
        )
    else:
        projections.append(
            ProjectionSpec(
                name="input_output",
                source="input",
                target="output",
                synapse_model=cfg.synapse_model,
                topology=dict(topo),
            ),
        )

    return NetworkSpec(
        populations=pops,
        projections=projections,
        metadata={
            "task": "bench",
            "seed": int(cfg.seed),
            "topology": cfg.topology,
            "synapse_model": cfg.synapse_model,
        },
    )


def _torch_memory_snapshot(*, device: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    torch = require_torch()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        allocated_mb = float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0)
        reserved_mb = float(torch.cuda.memory_reserved()) / (1024.0 * 1024.0)
        peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        out["torch_cuda_allocated_mb"] = allocated_mb
        out["torch_cuda_reserved_mb"] = reserved_mb
        out["torch_cuda_max_allocated_mb"] = peak_mb
    return out


def _host_memory_snapshot() -> dict[str, float]:
    try:
        psutil = importlib.import_module("psutil")
    except ImportError:
        return {}

    process = psutil.Process()
    vm = psutil.virtual_memory()
    return {
        "resource.ram.system_used_mb": float(vm.used) / (1024.0 * 1024.0),
        "resource.ram.system_total_mb": float(vm.total) / (1024.0 * 1024.0),
        "resource.ram.process_rss_mb": float(process.memory_info().rss) / (1024.0 * 1024.0),
    }


def run_bench(cfg: BenchRunConfig) -> dict[str, Any]:
    """Run a no-grad throughput benchmark and return a JSON-friendly summary."""
    torch = require_torch()
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    dev, tdt = resolve_device_dtype(cfg.device, cfg.dtype)
    factory = NetworkFactory(DEFAULT_HUB.neurons, DEFAULT_HUB.synapses)
    spec = build_random_bench_spec(cfg)
    engine = factory.build(spec, device=str(dev), dtype=cfg.dtype, seed=cfg.seed)

    drive = torch.empty(cfg.n_input, dtype=tdt, device=dev)

    def external_drive_fn(_step: int) -> dict[str, dict[Compartment, Any]]:
        drive.uniform_(0.0, float(cfg.amplitude))
        return {"input": {Compartment.SOMA: drive}}

    use_cuda_timing = bool(
        cfg.sync_cuda_timing
        and str(dev).startswith("cuda")
        and torch.cuda.is_available()
    )

    with torch.no_grad():
        if cfg.warmup > 0:
            engine.run_steps(cfg.warmup, external_drive_fn, collect=False)

        cuda_elapsed_ms: float | None = None
        if use_cuda_timing:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall_t0 = perf_counter()
            start_ev.record()
            engine.run_steps(cfg.steps, external_drive_fn, collect=False)
            end_ev.record()
            torch.cuda.synchronize()
            wall_elapsed_s = perf_counter() - wall_t0
            cuda_elapsed_ms = float(start_ev.elapsed_time(end_ev))
            timed_elapsed_s = cuda_elapsed_ms / 1000.0 if cuda_elapsed_ms > 0 else wall_elapsed_s
        else:
            wall_t0 = perf_counter()
            engine.run_steps(cfg.steps, external_drive_fn, collect=False)
            wall_elapsed_s = perf_counter() - wall_t0
            timed_elapsed_s = wall_elapsed_s

    steps_per_sec = 0.0 if timed_elapsed_s <= 0.0 else float(cfg.steps / timed_elapsed_s)
    ms_per_step = 0.0 if cfg.steps <= 0 else float((timed_elapsed_s * 1000.0) / cfg.steps)

    edge_counts = {
        name: int(proj.topology.weights.numel())
        for name, proj in engine.projections.items()
    }

    summary: dict[str, Any] = {
        "mode": "bench",
        "device": str(dev),
        "dtype": str(tdt).replace("torch.", ""),
        "seed": int(cfg.seed),
        "steps": int(cfg.steps),
        "warmup": int(cfg.warmup),
        "n_input": int(cfg.n_input),
        "n_hidden": int(cfg.n_hidden),
        "n_output": int(cfg.n_output),
        "topology": str(cfg.topology),
        "synapse_model": str(cfg.synapse_model),
        "timed_elapsed_s": float(timed_elapsed_s),
        "wall_elapsed_s": float(wall_elapsed_s),
        "steps_per_sec": float(steps_per_sec),
        "ms_per_step": float(ms_per_step),
        "projection_edge_counts": edge_counts,
        "edge_count_total": int(sum(edge_counts.values())),
        "topology_json": to_topology_json(engine),
    }
    if cuda_elapsed_ms is not None:
        summary["cuda_elapsed_ms"] = float(cuda_elapsed_ms)

    summary.update(_torch_memory_snapshot(device=dev))
    summary.update(_host_memory_snapshot())
    summary["config"] = asdict(cfg)
    return summary
