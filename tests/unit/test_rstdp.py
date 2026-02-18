"""Math-predictive tests for the R-STDP learning rule.

Every test computes expected values analytically first, then verifies.
"""

from __future__ import annotations

import math

import pytest
import torch

from neuroforge.contracts.learning import LearningBatch, LearningStepResult
from neuroforge.learning.rstdp import RSTDPParams, RSTDPRule


@pytest.fixture
def params() -> RSTDPParams:
    return RSTDPParams(
        lr=0.01, tau_e=20e-3, a_plus=1.0, a_minus=0.5, w_min=-1.0, w_max=1.0
    )


@pytest.fixture
def rule(params: RSTDPParams) -> RSTDPRule:
    return RSTDPRule(params)


@pytest.fixture
def dt() -> float:
    return 1e-3


# ── Pure-math helper tests ──────────────────────────────────────────


class TestRSTDPMathHelpers:
    """Verify analytical helper methods."""

    def test_eligibility_decay(self, rule: RSTDPRule, dt: float) -> None:
        expected = math.exp(-dt / rule.params.tau_e)
        assert rule.eligibility_decay(dt) == pytest.approx(expected)

    def test_eligibility_update_pre_spike(self, rule: RSTDPRule, dt: float) -> None:
        """Pre-spike increases eligibility by a_plus."""
        e_prev = 0.5
        expected = rule.predict_eligibility_update(
            e_prev, pre_spike=True, post_spike=False, dt=dt
        )
        decay = rule.eligibility_decay(dt)
        assert expected == pytest.approx(e_prev * decay + rule.params.a_plus)

    def test_eligibility_update_post_spike(self, rule: RSTDPRule, dt: float) -> None:
        """Post-spike decreases eligibility by a_minus."""
        e_prev = 0.5
        expected = rule.predict_eligibility_update(
            e_prev, pre_spike=False, post_spike=True, dt=dt
        )
        decay = rule.eligibility_decay(dt)
        assert expected == pytest.approx(e_prev * decay - rule.params.a_minus)

    def test_eligibility_update_both_spikes(self, rule: RSTDPRule, dt: float) -> None:
        """Both spikes: net contribution = a_plus - a_minus."""
        e_prev = 0.0
        expected = rule.predict_eligibility_update(e_prev, True, True, dt)
        assert expected == pytest.approx(rule.params.a_plus - rule.params.a_minus)

    def test_eligibility_pure_decay(self, rule: RSTDPRule, dt: float) -> None:
        """No spikes: eligibility just decays."""
        e_prev = 1.0
        expected = rule.predict_eligibility_update(e_prev, False, False, dt)
        assert expected == pytest.approx(e_prev * rule.eligibility_decay(dt))

    def test_predict_dw(self, rule: RSTDPRule) -> None:
        """dw = lr * reward * eligibility."""
        assert rule.predict_dw(1.0, 0.5) == pytest.approx(0.01 * 1.0 * 0.5)
        assert rule.predict_dw(-1.0, 0.5) == pytest.approx(0.01 * -1.0 * 0.5)
        assert rule.predict_dw(0.0, 0.5) == pytest.approx(0.0)

    def test_clamping(self, rule: RSTDPRule) -> None:
        assert rule.predict_clamped_weight(0.9, 0.2) == pytest.approx(
            1.0
        )  # clamped at max
        assert rule.predict_clamped_weight(-0.9, -0.2) == pytest.approx(
            -1.0
        )  # clamped at min
        assert rule.predict_clamped_weight(0.5, 0.1) == pytest.approx(0.6)  # no clamp


# ── Tensor-level simulation tests ──────────────────────────────────


class TestRSTDPSimulation:
    """Run the actual step() method and verify against math."""

    def test_single_edge_pre_spike_positive_reward(
        self, rule: RSTDPRule, dt: float
    ) -> None:
        """Pre-spike + positive reward → positive dw."""
        state = rule.init_state(1, "cpu", "float64")

        batch = LearningBatch(
            pre_spikes=torch.tensor([True]),
            post_spikes=torch.tensor([False]),
            weights=torch.tensor([0.5], dtype=torch.float64),
            eligibility=state["eligibility"],
            reward=torch.tensor(1.0, dtype=torch.float64),
        )

        result = rule.step(state, batch, dt)
        assert isinstance(result, LearningStepResult)

        # Math prediction
        e_expected = rule.predict_eligibility_update(0.0, True, False, dt)
        dw_expected = rule.predict_dw(1.0, e_expected)

        assert result.dw.item() == pytest.approx(dw_expected)
        assert result.new_eligibility is not None
        assert result.new_eligibility.item() == pytest.approx(e_expected)

    def test_single_edge_negative_reward(self, rule: RSTDPRule, dt: float) -> None:
        """Pre-spike + negative reward → negative dw (anti-Hebbian)."""
        state = rule.init_state(1, "cpu", "float64")

        batch = LearningBatch(
            pre_spikes=torch.tensor([True]),
            post_spikes=torch.tensor([False]),
            weights=torch.tensor([0.5], dtype=torch.float64),
            eligibility=state["eligibility"],
            reward=torch.tensor(-1.0, dtype=torch.float64),
        )

        result = rule.step(state, batch, dt)

        e_expected = rule.predict_eligibility_update(0.0, True, False, dt)
        dw_expected = rule.predict_dw(-1.0, e_expected)
        assert dw_expected < 0
        assert result.dw.item() == pytest.approx(dw_expected)

    def test_zero_reward_no_weight_change(self, rule: RSTDPRule, dt: float) -> None:
        """Zero reward → dw = 0 regardless of spikes."""
        state = rule.init_state(1, "cpu", "float64")

        batch = LearningBatch(
            pre_spikes=torch.tensor([True]),
            post_spikes=torch.tensor([False]),
            weights=torch.tensor([0.5], dtype=torch.float64),
            eligibility=state["eligibility"],
            reward=torch.tensor(0.0, dtype=torch.float64),
        )

        result = rule.step(state, batch, dt)
        assert result.dw.item() == pytest.approx(0.0)

    def test_eligibility_accumulation(self, rule: RSTDPRule, dt: float) -> None:
        """Eligibility accumulates across multiple pre-spikes."""
        state = rule.init_state(1, "cpu", "float64")

        e = 0.0
        for _ in range(5):
            batch = LearningBatch(
                pre_spikes=torch.tensor([True]),
                post_spikes=torch.tensor([False]),
                weights=torch.tensor([0.5], dtype=torch.float64),
                eligibility=state["eligibility"],
                reward=torch.tensor(0.0, dtype=torch.float64),
            )
            rule.step(state, batch, dt)
            e = rule.predict_eligibility_update(e, True, False, dt)

        assert state["eligibility"].item() == pytest.approx(e, rel=1e-9)

    def test_eligibility_decay_without_spikes(self, rule: RSTDPRule, dt: float) -> None:
        """Without spikes, eligibility decays toward zero."""
        state = rule.init_state(1, "cpu", "float64")
        state["eligibility"] = torch.tensor([1.0], dtype=torch.float64)

        for _ in range(100):
            batch = LearningBatch(
                pre_spikes=torch.tensor([False]),
                post_spikes=torch.tensor([False]),
                weights=torch.tensor([0.5], dtype=torch.float64),
                eligibility=state["eligibility"],
                reward=torch.tensor(0.0, dtype=torch.float64),
            )
            rule.step(state, batch, dt)

        decay_100 = math.exp(-100 * dt / rule.params.tau_e)
        assert state["eligibility"].item() == pytest.approx(decay_100, rel=1e-9)

    def test_weight_clamping(self, rule: RSTDPRule, dt: float) -> None:
        """apply_dw clamps weights to [w_min, w_max]."""
        w = torch.tensor([0.9], dtype=torch.float64)
        dw = torch.tensor([0.5], dtype=torch.float64)

        new_w = rule.apply_dw(w, dw)
        assert new_w.item() == pytest.approx(rule.params.w_max)

        w_neg = torch.tensor([-0.9], dtype=torch.float64)
        dw_neg = torch.tensor([-0.5], dtype=torch.float64)
        new_w_neg = rule.apply_dw(w_neg, dw_neg)
        assert new_w_neg.item() == pytest.approx(rule.params.w_min)

    def test_multi_edge_batch(self, rule: RSTDPRule, dt: float) -> None:
        """Multiple edges processed in parallel."""
        n = 3
        state = rule.init_state(n, "cpu", "float64")

        batch = LearningBatch(
            pre_spikes=torch.tensor([True, False, True]),
            post_spikes=torch.tensor([False, True, True]),
            weights=torch.tensor([0.5, 0.3, -0.2], dtype=torch.float64),
            eligibility=state["eligibility"],
            reward=torch.tensor(1.0, dtype=torch.float64),
        )

        result = rule.step(state, batch, dt)

        # Edge 0: pre=T, post=F → e += a_plus
        e0 = rule.predict_eligibility_update(0.0, True, False, dt)
        # Edge 1: pre=F, post=T → e -= a_minus
        e1 = rule.predict_eligibility_update(0.0, False, True, dt)
        # Edge 2: pre=T, post=T → e += a_plus - a_minus
        e2 = rule.predict_eligibility_update(0.0, True, True, dt)

        assert result.new_eligibility is not None
        assert result.new_eligibility[0].item() == pytest.approx(e0)
        assert result.new_eligibility[1].item() == pytest.approx(e1)
        assert result.new_eligibility[2].item() == pytest.approx(e2)

        assert result.dw[0].item() == pytest.approx(rule.predict_dw(1.0, e0))
        assert result.dw[1].item() == pytest.approx(rule.predict_dw(1.0, e1))
        assert result.dw[2].item() == pytest.approx(rule.predict_dw(1.0, e2))


# ── Registry tests ──────────────────────────────────────────────────


class TestRSTDPRegistry:
    def test_registered(self) -> None:
        from neuroforge.learning.registry import LEARNING_RULES

        assert LEARNING_RULES.has("rstdp")

    def test_create(self) -> None:
        from neuroforge.learning.registry import create_learning_rule

        rule = create_learning_rule("rstdp")
        assert isinstance(rule, RSTDPRule)
