"""Stable scalar schema for metrics/scalars.csv artifacts."""

from __future__ import annotations

__all__ = [
    "BASE_FIELDS",
    "CUDA_FIELDS",
    "PHASE6_FIELDS",
    "RESOURCE_FIELDS",
    "build_scalar_fields",
]

BASE_FIELDS = [
    "trial",
    "epoch",
    "gate",
    "accuracy",
    "loss",
    "error",
    "correct",
    "wall_ms",
]

CUDA_FIELDS = [
    "cuda_mem_allocated_mb",
    "cuda_mem_reserved_mb",
    "cuda_mem_peak_mb",
]

PHASE6_FIELDS = [
    "rate_in_mean_hz",
    "rate_in_max_hz",
    "rate_hid_mean_hz",
    "rate_hid_max_hz",
    "rate_out_hz",
    "out_spike_count",
    "sparsity_in",
    "sparsity_hid",
    "w_norm_ih",
    "w_norm_ho",
    "w_maxabs_ih",
    "w_maxabs_ho",
    "g_norm_ih",
    "g_norm_ho",
    "g_maxabs_ih",
    "g_maxabs_ho",
    "conv_streak",
    "conv_per_pattern_min",
    "conv_per_pattern_mean",
    "stab_nan_inf",
    "stab_weight_explode",
    "stab_rate_saturation",
    "stab_oscillation",
    "stab_stagnation",
]

# Optional resource monitor fields (best-effort; keep additive).
RESOURCE_FIELDS = [
    "resource.cpu.system_percent",
    "resource.cpu.process_percent",
    "resource.ram.system_used_mb",
    "resource.ram.system_total_mb",
    "resource.ram.process_rss_mb",
    "resource.gpu.util_percent",
    "resource.gpu.mem_used_mb",
    "resource.gpu.mem_total_mb",
    "resource.gpu.temp_c",
    "resource.gpu.power_w",
    "resource.torch.cuda_allocated_mb",
    "resource.torch.cuda_reserved_mb",
    "resource.torch.cuda_max_allocated_mb",
    "torch_cuda_allocated_mb",
    "torch_cuda_reserved_mb",
    "torch_cuda_max_allocated_mb",
]


def build_scalar_fields(*, include_resource: bool = False) -> list[str]:
    """Return a stable ordered union of scalar CSV fields."""
    ordered_groups = [BASE_FIELDS, CUDA_FIELDS, PHASE6_FIELDS]
    if include_resource:
        ordered_groups.append(RESOURCE_FIELDS)

    out: list[str] = []
    seen: set[str] = set()
    for group in ordered_groups:
        for key in group:
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out

