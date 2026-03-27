# pyright: basic, reportMissingImports=false
"""CUDA smoke tests — skipped automatically on CPU-only systems.

Run explicitly with::

    pytest -q -m cuda
"""

from __future__ import annotations

import pytest
import torch

_CUDA = torch.cuda.is_available()
_skip_no_cuda = pytest.mark.skipif(not _CUDA, reason="CUDA not available")


@pytest.mark.cuda
@_skip_no_cuda
class TestCudaEngine:
    """Verify that CoreEngine runs on CUDA and tensors remain on-device."""

    def test_engine_step_on_cuda(self) -> None:
        """Build a tiny 2→1 engine on cuda, step it, and assert device."""
        from neuroforge.contracts.simulation import SimulationConfig
        from neuroforge.contracts.synapses import SynapseTopology
        from neuroforge.contracts.types import Compartment
        from neuroforge.engine.core_engine import CoreEngine, Population, Projection
        from neuroforge.neurons.lif.model import LIFModel, LIFParams
        from neuroforge.synapses.static import StaticSynapseModel

        cfg = SimulationConfig(dt=1e-3, seed=1, device="cuda", dtype="float32")
        lif = LIFModel(LIFParams(tau_mem=20e-3, v_thresh=1.0))

        engine = CoreEngine(cfg)
        engine.add_population(Population(name="inp", model=lif, n=2))
        engine.add_population(Population(name="out", model=lif, n=1))

        topo = SynapseTopology(
            pre_idx=torch.tensor([0, 1], dtype=torch.long, device="cuda"),
            post_idx=torch.tensor([0, 0], dtype=torch.long, device="cuda"),
            weights=torch.tensor([0.5, 0.5], dtype=torch.float32, device="cuda"),
            delays=torch.zeros(2, dtype=torch.long, device="cuda"),
            n_pre=2,
            n_post=1,
        )
        engine.add_projection(
            Projection(
                name="inp->out",
                model=StaticSynapseModel(),
                source="inp",
                target="out",
                topology=topo,
            )
        )
        engine.build()

        # Drive input neurons with strong current to provoke spikes.
        drive = {
            "inp": {
                Compartment.SOMA: torch.tensor(
                    [60.0, 60.0], dtype=torch.float32, device="cuda",
                ),
            },
        }

        for _ in range(5):
            result = engine.step(external_drive=drive)

        # Assert spike tensors are on CUDA.
        for pop_name, spk in result.spikes.items():
            assert spk.device.type == "cuda", f"{pop_name} spikes not on CUDA"

        # Assert CUDA memory was actually used.
        assert torch.cuda.memory_allocated() > 0

    def test_cuda_memory_stats(self) -> None:
        """Verify torch.cuda memory functions return non-zero values."""
        # Allocate a small tensor on CUDA to exercise the allocator.
        t = torch.zeros(1024, device="cuda")
        assert torch.cuda.memory_allocated() > 0
        assert torch.cuda.max_memory_allocated() > 0
        del t


@pytest.mark.cuda
@_skip_no_cuda
class TestCudaMetricsMonitor:
    """Verify that CudaMetricsMonitor enriches events."""

    def test_monitor_injects_memory_stats(self) -> None:
        from neuroforge.contracts.monitors import EventTopic, MonitorEvent
        from neuroforge.monitors.cuda_monitor import CudaMetricsMonitor

        mon = CudaMetricsMonitor(enabled=True)

        # Allocate something on CUDA so memory > 0.
        _x = torch.zeros(1024, device="cuda")

        event = MonitorEvent(
            topic=EventTopic.SCALAR,
            step=0,
            t=0.0,
            source="test",
            data={"trial": 0, "accuracy": 0.5},
        )
        mon.on_event(event)

        assert "cuda_mem_allocated" in event.data
        assert event.data["cuda_mem_allocated"] > 0
        assert "cuda_mem_peak" in event.data
        del _x

    def test_monitor_noop_on_non_scalar(self) -> None:
        from neuroforge.contracts.monitors import EventTopic, MonitorEvent
        from neuroforge.monitors.cuda_monitor import CudaMetricsMonitor

        mon = CudaMetricsMonitor(enabled=True)
        event = MonitorEvent(
            topic=EventTopic.TOPOLOGY,
            step=0,
            t=0.0,
            source="test",
            data={"layers": []},
        )
        mon.on_event(event)
        assert "cuda_mem_allocated" not in event.data
