"""StabilityMonitor - detect instability patterns and inject 0/1 flags."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from neuroforge.contracts.monitors import EventTopic, MonitorEvent

__all__ = ["StabilityConfig", "StabilityMonitor"]


@dataclass(slots=True)
class StabilityConfig:
    enabled: bool = True
    check_every_n_trials: int = 5
    weight_maxabs_threshold: float = 50.0
    rate_low_hz: float = 0.1
    rate_high_hz: float = 200.0
    stagnation_window: int = 200
    stagnation_min_delta: float = 0.02
    oscillation_window: int = 40
    fail_fast: bool = False

    def __post_init__(self) -> None:
        self.check_every_n_trials = max(1, int(self.check_every_n_trials))
        self.stagnation_window = max(2, int(self.stagnation_window))
        self.oscillation_window = max(4, int(self.oscillation_window))


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _contains_nan_inf(obj: object) -> bool:
    if isinstance(obj, Mapping):
        map_obj = cast("Mapping[object, object]", obj)
        return any(_contains_nan_inf(value) for value in map_obj.values())
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        seq_obj = cast("Sequence[object]", obj)
        return any(_contains_nan_inf(value) for value in seq_obj)
    f = _coerce_float(obj)
    if f is None:
        return False
    return not math.isfinite(f)


class StabilityMonitor:
    """Adds stability flags to scalar/trial events; optionally fail-fast."""

    def __init__(self, cfg: StabilityConfig | None = None) -> None:
        self._cfg = cfg or StabilityConfig()
        self.enabled = self._cfg.enabled

        self._accuracy: deque[float] = deque(maxlen=self._cfg.stagnation_window)
        self._out_counts: deque[float] = deque(maxlen=self._cfg.oscillation_window)
        self._last_accuracy_trial: int | None = None
        self._last_out_trial: int | None = None
        self._last_checked_trial: int | None = None

        self._latest_w_maxabs_ih: float | None = None
        self._latest_w_maxabs_ho: float | None = None
        self._latest_rate_out_hz: float | None = None

        self._last_flags: dict[str, int] = self._zero_flags()

    @staticmethod
    def _zero_flags() -> dict[str, int]:
        return {
            "stab_nan_inf": 0,
            "stab_weight_explode": 0,
            "stab_rate_saturation": 0,
            "stab_oscillation": 0,
            "stab_stagnation": 0,
        }

    def reset(self) -> None:
        self._accuracy.clear()
        self._out_counts.clear()
        self._last_accuracy_trial = None
        self._last_out_trial = None
        self._last_checked_trial = None
        self._latest_w_maxabs_ih = None
        self._latest_w_maxabs_ho = None
        self._latest_rate_out_hz = None
        self._last_flags = self._zero_flags()

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "last_flags": dict(self._last_flags),
            "accuracy_samples": len(self._accuracy),
            "out_count_samples": len(self._out_counts),
        }

    def on_event(self, event: MonitorEvent) -> None:
        if not self.enabled:
            return
        if event.topic == EventTopic.TRAINING_START:
            self.reset()
            return
        if event.topic == EventTopic.TRAINING_END:
            return
        if event.topic not in (EventTopic.TRAINING_TRIAL, EventTopic.SCALAR):
            return

        trial = self._trial_index(event)
        if trial is not None:
            self._update_buffers(event.data, trial)
            self._update_latest_metrics(event.data)
            if self._should_check(trial):
                self._last_flags = self._compute_flags(event.data)
                self._last_checked_trial = trial
                self._maybe_fail_fast(trial)

        event.data.update(self._last_flags)

    @staticmethod
    def _trial_index(event: MonitorEvent) -> int | None:
        raw = event.data.get("trial", event.step)
        if isinstance(raw, bool):
            return int(raw)
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float):
            return int(round(raw))
        if isinstance(raw, str):
            try:
                return int(raw)
            except ValueError:
                return None
        return None

    def _update_buffers(self, data: dict[str, Any], trial: int) -> None:
        acc = _coerce_float(data.get("accuracy"))
        if acc is not None and trial != self._last_accuracy_trial:
            self._accuracy.append(acc)
            self._last_accuracy_trial = trial

        out_count = _coerce_float(data.get("out_spike_count"))
        if out_count is None:
            out_seq_obj: object = data.get("output_spikes")
            if isinstance(out_seq_obj, (list, tuple)):
                out_seq = cast("Sequence[object]", out_seq_obj)
                total = 0.0
                for item in out_seq:
                    value = _coerce_float(item)
                    if value is not None:
                        total += value
                out_count = float(total)
        if out_count is not None and trial != self._last_out_trial:
            self._out_counts.append(out_count)
            self._last_out_trial = trial

    def _update_latest_metrics(self, data: dict[str, Any]) -> None:
        w_ih = _coerce_float(data.get("w_maxabs_ih"))
        if w_ih is not None:
            self._latest_w_maxabs_ih = w_ih
        w_ho = _coerce_float(data.get("w_maxabs_ho"))
        if w_ho is not None:
            self._latest_w_maxabs_ho = w_ho
        rate = _coerce_float(data.get("rate_out_hz"))
        if rate is not None:
            self._latest_rate_out_hz = rate

    def _should_check(self, trial: int) -> bool:
        if trial % self._cfg.check_every_n_trials != 0:
            return False
        return trial != self._last_checked_trial

    def _compute_flags(self, data: dict[str, Any]) -> dict[str, int]:
        flags = self._zero_flags()

        if _contains_nan_inf(data):
            flags["stab_nan_inf"] = 1

        maxabs_ih = self._latest_w_maxabs_ih
        maxabs_ho = self._latest_w_maxabs_ho
        if (
            (maxabs_ih is not None and maxabs_ih > self._cfg.weight_maxabs_threshold)
            or (maxabs_ho is not None and maxabs_ho > self._cfg.weight_maxabs_threshold)
        ):
            flags["stab_weight_explode"] = 1

        rate_out = self._latest_rate_out_hz
        if rate_out is not None and (
            rate_out < self._cfg.rate_low_hz or rate_out > self._cfg.rate_high_hz
        ):
            flags["stab_rate_saturation"] = 1

        if self._detect_oscillation():
            flags["stab_oscillation"] = 1

        if self._detect_stagnation():
            flags["stab_stagnation"] = 1

        return flags

    def _detect_stagnation(self) -> bool:
        if len(self._accuracy) < self._cfg.stagnation_window:
            return False
        oldest = self._accuracy[0]
        newest = self._accuracy[-1]
        return (newest - oldest) < self._cfg.stagnation_min_delta

    def _detect_oscillation(self) -> bool:
        values = list(self._out_counts)
        if len(values) < 6:
            return False
        signs: list[int] = []
        for i in range(1, len(values)):
            d = values[i] - values[i - 1]
            if d > 0:
                signs.append(1)
            elif d < 0:
                signs.append(-1)
            else:
                signs.append(0)
        nz_signs = [s for s in signs if s != 0]
        if len(nz_signs) < 4:
            return False
        alternations = 0
        for i in range(1, len(nz_signs)):
            if nz_signs[i] == -nz_signs[i - 1]:
                alternations += 1
        ratio = alternations / max(1, len(nz_signs) - 1)
        amplitude = max(values) - min(values)
        return ratio >= 0.8 and amplitude >= 1.0

    def _maybe_fail_fast(self, trial: int) -> None:
        if not self._cfg.fail_fast:
            return
        if self._last_flags["stab_nan_inf"] == 1:
            msg = f"StabilityMonitor fail-fast at trial {trial}: NaN/Inf detected"
            raise RuntimeError(msg)
        if self._last_flags["stab_weight_explode"] == 1:
            msg = (
                f"StabilityMonitor fail-fast at trial {trial}: "
                "weight maxabs threshold exceeded"
            )
            raise RuntimeError(msg)
