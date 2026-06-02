"""Tests for trace-rule invariance (Layer A2).

The core claim: with the temporal trace, a single object cell binds an object's
*whole trajectory* of changing appearances into one invariant code — so the start
and end of a morphing sequence read out the same cell, and two different objects
read out different cells. Disabling the trace (clustering) splits the trajectory,
which proves the trace is what creates the invariance.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neuroforge.vision.encoding import TraceInvariantConfig, TraceInvariantLayer  # noqa: E402

_N = 12


def _trajectory(start: int, end: int, steps: int) -> list[torch.Tensor]:
    """A sequence morphing from a block at *start* to a block at *end*."""
    a = torch.zeros(_N)
    a[start : start + 4] = 1.0
    b = torch.zeros(_N)
    b[end : end + 4] = 1.0
    return [(1 - t) * a + t * b for t in torch.linspace(0.0, 1.0, steps)]


def _train(layer: TraceInvariantLayer, traj_a: list, traj_b: list, *, epochs: int) -> None:
    for _ in range(epochs):
        layer.reset()
        for frame in traj_a:
            layer.observe(frame)
        layer.reset()
        for frame in traj_b:
            layer.observe(frame)


@pytest.mark.unit
def test_trace_binds_a_trajectory_into_one_invariant_cell() -> None:
    traj_a = _trajectory(0, 5, steps=5)   # drifts across the left half
    traj_b = _trajectory(6, 1, steps=5)   # a different object
    layer = TraceInvariantLayer(
        TraceInvariantConfig(n_inputs=_N, n_objects=8, lr=0.1, trace_gain=3.0, seed=1),
    )
    _train(layer, traj_a, traj_b, epochs=200)

    a_start, a_end = layer.winner(traj_a[0]), layer.winner(traj_a[-1])
    b_start, b_end = layer.winner(traj_b[0]), layer.winner(traj_b[-1])
    assert a_start == a_end            # the trajectory's endpoints bind to one cell
    assert b_start == b_end
    assert a_start != b_start          # different objects -> different invariant cells


@pytest.mark.unit
def test_without_trace_the_trajectory_splits() -> None:
    traj_a = _trajectory(0, 5, steps=5)
    traj_b = _trajectory(6, 1, steps=5)
    with_trace = TraceInvariantLayer(
        TraceInvariantConfig(n_inputs=_N, n_objects=8, lr=0.1, trace_gain=3.0, seed=1),
    )
    no_trace = TraceInvariantLayer(
        TraceInvariantConfig(n_inputs=_N, n_objects=8, lr=0.1, trace_gain=0.0, seed=1),
    )
    _train(with_trace, traj_a, traj_b, epochs=200)
    _train(no_trace, traj_a, traj_b, epochs=200)

    def distinct(layer: TraceInvariantLayer) -> int:
        return len({layer.winner(f) for f in traj_a})

    # The trace collapses the trajectory; plain clustering keeps it more split.
    assert distinct(with_trace) < distinct(no_trace)


@pytest.mark.unit
def test_reset_clears_the_trace() -> None:
    layer = TraceInvariantLayer(TraceInvariantConfig(n_inputs=_N, n_objects=4, seed=0))
    layer.observe(torch.ones(_N))
    layer.reset()
    assert float(layer._trace.abs().sum()) == 0.0  # noqa: SLF001 - inspecting trace state
