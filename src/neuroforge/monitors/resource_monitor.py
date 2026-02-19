"""ResourceMonitor - enrich scalar events with host/GPU utilization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neuroforge.contracts.monitors import EventTopic, MonitorEvent

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ["ResourceMonitor"]

_LOGGER = logging.getLogger(__name__)


class ResourceMonitor:
    """Attach resource-utilization metrics to scalar-like monitor events.

    The monitor is opt-in and best-effort:
    - host CPU/RAM via psutil (optional dependency)
    - NVIDIA utilization/VRAM via NVML (optional dependency)
    - torch CUDA allocator metrics when CUDA is available
    """

    _warned_psutil_missing = False
    _warned_nvml_missing = False
    _warned_nvml_init = False

    def __init__(
        self,
        *,
        enabled: bool = False,
        every_n_steps: int = 10,
        include_process: bool = True,
        include_system: bool = True,
        include_gpu: bool = True,
        gpu_index: int = 0,
    ) -> None:
        self.enabled = enabled
        self._every_n_steps = max(1, int(every_n_steps))
        self._include_process = include_process
        self._include_system = include_system
        self._include_gpu = include_gpu
        self._gpu_index = max(0, int(gpu_index))

        self._psutil: ModuleType | None = None
        self._process: Any = None
        self._pynvml: ModuleType | None = None
        self._nvml_handle: Any = None
        self._nvml_ready = False
        self._first_cpu_sample = True
        self._last_payload: dict[str, float] = {}

        self._init_psutil()
        self._init_nvml()

    def on_event(self, event: MonitorEvent) -> None:
        """Enrich scalar/trial events at the configured cadence."""
        if not self.enabled:
            return
        if event.topic not in (EventTopic.SCALAR, EventTopic.TRAINING_TRIAL):
            return
        if event.step % self._every_n_steps != 0:
            return

        payload = self._collect_payload()
        if not payload:
            return

        event.data.update(payload)
        self._last_payload = payload

    def reset(self) -> None:
        """Reset short-lived monitor state."""
        self._first_cpu_sample = True
        self._last_payload = {}

    def snapshot(self) -> dict[str, Any]:
        """Return monitor status and the most recent metric payload."""
        return {
            "enabled": self.enabled,
            "every_n_steps": self._every_n_steps,
            "include_process": self._include_process,
            "include_system": self._include_system,
            "include_gpu": self._include_gpu,
            "gpu_index": self._gpu_index,
            "psutil_available": self._psutil is not None,
            "nvml_available": self._nvml_ready,
            "last_payload": dict(self._last_payload),
        }

    def _init_psutil(self) -> None:
        if not (self._include_process or self._include_system):
            return
        try:
            import psutil
        except ImportError:
            if not ResourceMonitor._warned_psutil_missing:
                _LOGGER.warning(
                    "ResourceMonitor: psutil is not installed; CPU/RAM metrics disabled.",
                )
                ResourceMonitor._warned_psutil_missing = True
            self._include_process = False
            self._include_system = False
            return

        self._psutil = psutil
        self._process = psutil.Process()

        # Prime CPU counters so the first real sample is meaningful.
        with suppress_exceptions():
            if self._include_system:
                psutil.cpu_percent(interval=None)
            if self._include_process:
                self._process.cpu_percent(interval=None)

    def _init_nvml(self) -> None:
        if not self._include_gpu:
            return
        try:
            import pynvml
        except ImportError:
            if not ResourceMonitor._warned_nvml_missing:
                _LOGGER.warning(
                    "ResourceMonitor: pynvml not installed; NVIDIA utilization metrics disabled.",
                )
                ResourceMonitor._warned_nvml_missing = True
            return

        self._pynvml = pynvml
        try:
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            self._nvml_ready = True
        except Exception:  # pragma: no cover - dependent on host GPU runtime
            if not ResourceMonitor._warned_nvml_init:
                _LOGGER.warning(
                    "ResourceMonitor: NVML init failed; NVIDIA utilization metrics disabled.",
                )
                ResourceMonitor._warned_nvml_init = True
            self._nvml_handle = None
            self._nvml_ready = False

    def _collect_payload(self) -> dict[str, float]:
        payload: dict[str, float] = {}
        payload.update(self._collect_cpu_ram())
        payload.update(self._collect_nvml_gpu())
        payload.update(self._collect_torch_cuda())
        return payload

    def _collect_cpu_ram(self) -> dict[str, float]:
        if self._psutil is None:
            return {}

        out: dict[str, float] = {}
        psutil = self._psutil

        if self._include_system:
            with suppress_exceptions():
                out["resource.cpu.system_percent"] = float(psutil.cpu_percent(interval=None))
            with suppress_exceptions():
                vm = psutil.virtual_memory()
                out["resource.ram.system_used_mb"] = float(vm.used) / (1024.0 * 1024.0)
                out["resource.ram.system_total_mb"] = float(vm.total) / (1024.0 * 1024.0)

        if self._include_process and self._process is not None:
            with suppress_exceptions():
                out["resource.cpu.process_percent"] = float(
                    self._process.cpu_percent(interval=None),
                )
            with suppress_exceptions():
                mi = self._process.memory_info()
                out["resource.ram.process_rss_mb"] = float(mi.rss) / (1024.0 * 1024.0)

        return out

    def _collect_nvml_gpu(self) -> dict[str, float]:
        if not self._include_gpu or not self._nvml_ready:
            return {}
        if self._pynvml is None or self._nvml_handle is None:
            return {}

        out: dict[str, float] = {}
        pynvml = self._pynvml
        handle = self._nvml_handle

        with suppress_exceptions():
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            out["resource.gpu.util_percent"] = float(util.gpu)

        with suppress_exceptions():
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            out["resource.gpu.mem_used_mb"] = float(mem.used) / (1024.0 * 1024.0)
            out["resource.gpu.mem_total_mb"] = float(mem.total) / (1024.0 * 1024.0)

        with suppress_exceptions():
            temp_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            out["resource.gpu.temp_c"] = float(temp_c)

        with suppress_exceptions():
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            out["resource.gpu.power_w"] = float(power_mw) / 1000.0

        return out

    def _collect_torch_cuda(self) -> dict[str, float]:
        if not self._include_gpu:
            return {}

        try:
            from neuroforge.core.torch_utils import require_torch

            torch = require_torch()
        except ImportError:
            return {}

        if not torch.cuda.is_available():
            return {}

        out: dict[str, float] = {}
        with suppress_exceptions():
            allocated_b = float(torch.cuda.memory_allocated())
            reserved_b = float(torch.cuda.memory_reserved())
            peak_b = float(torch.cuda.max_memory_allocated())

            allocated_mb = allocated_b / (1024.0 * 1024.0)
            reserved_mb = reserved_b / (1024.0 * 1024.0)
            peak_mb = peak_b / (1024.0 * 1024.0)

            out["resource.torch.cuda_allocated_mb"] = allocated_mb
            out["resource.torch.cuda_reserved_mb"] = reserved_mb
            out["resource.torch.cuda_max_allocated_mb"] = peak_mb

            # Keep a compact legacy alias family for dashboard fallbacks.
            out["torch_cuda_allocated_mb"] = allocated_mb
            out["torch_cuda_reserved_mb"] = reserved_mb
            out["torch_cuda_max_allocated_mb"] = peak_mb

        return out


class suppress_exceptions:
    """Tiny context manager that suppresses non-fatal metric collection errors."""

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> bool:
        return exc is not None
