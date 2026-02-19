"""TrialStatsMonitor - enrich TRAINING_TRIAL events with Phase 6 scalar stats."""

from __future__ import annotations

from typing import Any, cast

from neuroforge.contracts.monitors import EventTopic, MonitorEvent

__all__ = ["TrialStatsMonitor"]


def _coerce_float(value: object, *, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return fallback
    return fallback


def _coerce_positive_float(value: object, *, fallback: float) -> float:
    out = _coerce_float(value, fallback=fallback)
    return out if out > 0.0 else fallback


def _coerce_positive_int(value: object, *, fallback: int) -> int:
    parsed = int(round(_coerce_float(value, fallback=float(fallback))))
    return parsed if parsed > 0 else fallback


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "t", "yes", "y"}:
            return True
        if norm in {"0", "false", "f", "no", "n", ""}:
            return False
    return bool(value)


def _to_numeric_list(value: object) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        seq = cast("list[object] | tuple[object, ...]", value)
        out: list[float] = []
        for item in seq:
            out.append(max(0.0, _coerce_float(item)))
        return out
    if isinstance(value, (bool, int, float, str)):
        return [max(0.0, _coerce_float(value))]
    return None


def _normalise_pattern(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        seq = cast("list[object] | tuple[object, ...]", value)
        ints: list[str] = []
        for item in seq:
            ints.append(str(int(round(_coerce_float(item)))))
        return f"({', '.join(ints)})"
    if value is None:
        return "()"
    return str(value)


class TrialStatsMonitor:
    """Computes per-trial rates/sparsity and convergence counters."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        dt_default: float = 1e-3,
        window_steps_default: int = 50,
    ) -> None:
        self.enabled = enabled
        self._dt_default = _coerce_positive_float(dt_default, fallback=1e-3)
        self._window_steps_default = max(1, int(window_steps_default))
        self._dt = self._dt_default
        self._window_steps = self._window_steps_default
        self._conv_streak = 0
        self._pattern_streaks: dict[tuple[str, str], int] = {}

    def on_event(self, event: MonitorEvent) -> None:
        """Handle TRAINING_START/TRAINING_TRIAL and inject computed scalars."""
        if not self.enabled:
            return
        if event.topic == EventTopic.TRAINING_START:
            self._on_training_start(event)
            return
        if event.topic == EventTopic.TRAINING_TRIAL:
            self._on_training_trial(event)

    def reset(self) -> None:
        """Reset monitor state to defaults."""
        self._dt = self._dt_default
        self._window_steps = self._window_steps_default
        self._conv_streak = 0
        self._pattern_streaks.clear()

    def snapshot(self) -> dict[str, Any]:
        """Return monitor state for debugging/UI."""
        return {
            "enabled": self.enabled,
            "dt": self._dt,
            "window_steps": self._window_steps,
            "conv_streak": self._conv_streak,
            "patterns_tracked": len(self._pattern_streaks),
        }

    def _on_training_start(self, event: MonitorEvent) -> None:
        data = event.data
        self._dt = _coerce_positive_float(data.get("dt"), fallback=self._dt_default)
        self._window_steps = _coerce_positive_int(
            data.get("window_steps"),
            fallback=self._window_steps_default,
        )
        self._conv_streak = 0
        self._pattern_streaks.clear()

    def _extract_counts(
        self,
        data: dict[str, Any],
        *keys: str,
    ) -> list[float] | None:
        for key in keys:
            if key not in data:
                continue
            counts = _to_numeric_list(data.get(key))
            if counts is None:
                continue
            return counts
        return None

    def _extract_out_count(self, data: dict[str, Any]) -> int:
        if "out_spike_count" in data:
            return max(0, int(round(_coerce_float(data.get("out_spike_count")))))

        output_counts = self._extract_counts(
            data,
            "output_spike_counts",
            "output_spikes",
        )
        if output_counts:
            return max(0, int(round(sum(output_counts))))
        return 0

    @staticmethod
    def _inject_pop_stats(
        data: dict[str, Any],
        *,
        counts: list[float],
        prefix: str,
        window_s: float,
    ) -> None:
        if not counts:
            return
        rates = [count / window_s for count in counts]
        data[f"rate_{prefix}_mean_hz"] = float(sum(rates) / len(rates))
        data[f"rate_{prefix}_max_hz"] = float(max(rates))
        data[f"sparsity_{prefix}"] = float(sum(1 for c in counts if c <= 0.0) / len(counts))

    def _on_training_trial(self, event: MonitorEvent) -> None:
        data = event.data
        window_s = max(self._dt * self._window_steps, 1e-12)

        out_spike_count = self._extract_out_count(data)
        data["out_spike_count"] = out_spike_count
        data["rate_out_hz"] = float(out_spike_count / window_s)

        input_counts = self._extract_counts(
            data,
            "input_spike_counts",
            "input_spikes",
        )
        if input_counts:
            self._inject_pop_stats(
                data,
                counts=input_counts,
                prefix="in",
                window_s=window_s,
            )

        hidden_counts = self._extract_counts(
            data,
            "hidden_spike_counts",
            "hidden_spikes",
        )
        if hidden_counts:
            self._inject_pop_stats(
                data,
                counts=hidden_counts,
                prefix="hid",
                window_s=window_s,
            )

        correct = _coerce_bool(data.get("correct", False))
        if correct:
            self._conv_streak += 1
        else:
            self._conv_streak = 0
        data["conv_streak"] = self._conv_streak

        gate_raw = data.get("gate", event.source)
        gate = "" if gate_raw is None else str(gate_raw)
        pattern_raw = data.get("input_pattern", data.get("input"))
        pattern_key = _normalise_pattern(pattern_raw)
        combo_key = (gate, pattern_key)
        if combo_key not in self._pattern_streaks:
            self._pattern_streaks[combo_key] = 0
        if correct:
            self._pattern_streaks[combo_key] += 1
        else:
            self._pattern_streaks[combo_key] = 0

        streak_values = list(self._pattern_streaks.values())
        if streak_values:
            data["conv_per_pattern_min"] = float(min(streak_values))
            data["conv_per_pattern_mean"] = float(sum(streak_values) / len(streak_values))
        else:
            data["conv_per_pattern_min"] = 0.0
            data["conv_per_pattern_mean"] = 0.0
