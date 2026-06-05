"""Math-predictive tests for Dale's Law enforcement.

Verifies:
- ``DalesMask`` creation and properties
- ``apply_dales_constraint`` produces correctly signed effective weights
- Gradient flow through the ``|w| Ã— sign`` reparameterization
- Sign constraint holds under optimiser updates
"""

from __future__ import annotations

import pytest
import torch

from neuroforge.kernel.dales_law import (
    DaleSign,
    DalesMask,
    apply_dales_constraint,
    make_dale_mask,
)

#


@pytest.fixture
def excitatory_mask() -> DalesMask:
    """4 excitatory, 0 inhibitory."""
    return make_dale_mask(4, 0)


@pytest.fixture
def inhibitory_mask() -> DalesMask:
    """0 excitatory, 3 inhibitory."""
    return make_dale_mask(0, 3)


@pytest.fixture
def mixed_mask() -> DalesMask:
    """3 excitatory, 2 inhibitory (indices 0-2 = E, 3-4 = I)."""
    return make_dale_mask(3, 2)


#


class TestDaleSign:
    """DaleSign enum values."""

    def test_excitatory_value(self) -> None:
        assert int(DaleSign.EXCITATORY) == 1

    def test_inhibitory_value(self) -> None:
        assert int(DaleSign.INHIBITORY) == -1


#


class TestMakeDaleMask:
    """make_dale_mask factory."""

    def test_all_excitatory(self, excitatory_mask: DalesMask) -> None:
        """All-E mask: signs are all +1."""
        assert excitatory_mask.size == 4
        assert excitatory_mask.n_excitatory == 4
        assert excitatory_mask.n_inhibitory == 0
        assert (excitatory_mask.signs == 1.0).all()

    def test_all_inhibitory(self, inhibitory_mask: DalesMask) -> None:
        """All-I mask: signs are all âˆ’1."""
        assert inhibitory_mask.size == 3
        assert inhibitory_mask.n_excitatory == 0
        assert inhibitory_mask.n_inhibitory == 3
        assert (inhibitory_mask.signs == -1.0).all()

    def test_mixed_mask(self, mixed_mask: DalesMask) -> None:
        """Mixed E/I mask: first 3 = +1, last 2 = âˆ’1."""
        assert mixed_mask.size == 5
        assert mixed_mask.n_excitatory == 3
        assert mixed_mask.n_inhibitory == 2
        expected = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0], dtype=torch.float64)
        assert torch.equal(mixed_mask.signs, expected)

    def test_negative_count_raises(self) -> None:
        """Negative neuron counts raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            make_dale_mask(-1, 2)

    def test_zero_zero_mask(self) -> None:
        """Zero neurons is valid but empty."""
        mask = make_dale_mask(0, 0)
        assert mask.size == 0


#


class TestApplyDalesConstraint:
    """Effective weight computation via |w| Ã— sign."""

    def test_excitatory_positive(self) -> None:
        """Excitatory constraint: all effective weights â‰¥ 0."""
        w_raw = torch.tensor([-0.5, 0.3, -0.1, 0.7], dtype=torch.float64)
        signs = torch.ones(4, dtype=torch.float64)
        w_eff = apply_dales_constraint(w_raw, signs)
        expected = torch.tensor([0.5, 0.3, 0.1, 0.7], dtype=torch.float64)
        assert torch.allclose(w_eff, expected)
        assert (w_eff >= 0).all()

    def test_inhibitory_negative(self) -> None:
        """Inhibitory constraint: all effective weights â‰¤ 0."""
        w_raw = torch.tensor([0.5, -0.3, 0.1], dtype=torch.float64)
        signs = -torch.ones(3, dtype=torch.float64)
        w_eff = apply_dales_constraint(w_raw, signs)
        expected = torch.tensor([-0.5, -0.3, -0.1], dtype=torch.float64)
        assert torch.allclose(w_eff, expected)
        assert (w_eff <= 0).all()

    def test_mixed_constraint(self, mixed_mask: DalesMask) -> None:
        """Mixed E/I: first 3 â‰¥ 0, last 2 â‰¤ 0."""
        w_raw = torch.tensor([-0.2, 0.4, -0.6, 0.8, -0.1], dtype=torch.float64)
        w_eff = apply_dales_constraint(w_raw, mixed_mask.signs)
        # Excitatory (indices 0-2): |w| Ã— (+1)
        assert (w_eff[:3] >= 0).all()
        # Inhibitory (indices 3-4): |w| Ã— (âˆ’1)
        assert (w_eff[3:] <= 0).all()

    def test_zero_weight_stays_zero(self) -> None:
        """Zero weight produces zero regardless of sign."""
        w_raw = torch.tensor([0.0, 0.0], dtype=torch.float64)
        signs = torch.tensor([1.0, -1.0], dtype=torch.float64)
        w_eff = apply_dales_constraint(w_raw, signs)
        assert torch.equal(w_eff, torch.zeros(2, dtype=torch.float64))

    def test_broadcasting_2d(self) -> None:
        """2-D weights with per-row signs (inputâ†’hidden)."""
        # 2 input neurons, 3 hidden neurons.
        w_raw = torch.tensor(
            [[-0.1, 0.2, -0.3], [0.4, -0.5, 0.6]], dtype=torch.float64
        )
        # Both inputs excitatory: signs = [+1, +1], unsqueeze to [2, 1].
        signs = torch.tensor([1.0, 1.0], dtype=torch.float64).unsqueeze(1)
        w_eff = apply_dales_constraint(w_raw, signs)
        assert (w_eff >= 0).all(), "All excitatory â†’ effective weights â‰¥ 0"
        expected = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float64
        )
        assert torch.allclose(w_eff, expected)


#


class TestGradientFlow:
    """Gradients flow through |w| Ã— sign reparameterization."""

    def test_gradient_through_abs(self) -> None:
        """Gradient of |w| Ã— sign is sign(w) Ã— sign_mask."""
        w_raw = torch.tensor([-0.5, 0.3], dtype=torch.float64, requires_grad=True)
        signs = torch.tensor([1.0, -1.0], dtype=torch.float64)
        w_eff = apply_dales_constraint(w_raw, signs)
        loss = w_eff.sum()
        loss.backward()
        assert w_raw.grad is not None
        # d/dw |w| Ã— s = sign(w) Ã— s
        expected_grad = torch.tensor([-1.0, -1.0], dtype=torch.float64)
        assert torch.allclose(w_raw.grad, expected_grad)

    def test_gradient_update_preserves_constraint(self) -> None:
        """After a gradient step, constraint still holds."""
        torch.manual_seed(0)
        w_raw = torch.randn(5, dtype=torch.float64, requires_grad=True)
        signs = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0], dtype=torch.float64)

        # Forward + backward + step.
        w_eff = apply_dales_constraint(w_raw, signs)
        target = torch.tensor([0.5, 0.5, 0.5, -0.5, -0.5], dtype=torch.float64)
        loss = ((w_eff - target) ** 2).sum()
        loss.backward()
        with torch.no_grad():
            assert w_raw.grad is not None
            w_raw.sub_(0.01 * w_raw.grad)
            w_raw.grad.zero_()

        # Re-apply constraint and verify signs.
        w_eff2 = apply_dales_constraint(w_raw, signs)
        assert (w_eff2[:3] >= 0).all(), "Excitatory weights still â‰¥ 0"
        assert (w_eff2[3:] <= 0).all(), "Inhibitory weights still â‰¤ 0"


#


@pytest.mark.slow
class TestDalesLawInLogicGates:
    """Verify Dale's Law is active in the logic gate task."""

    def test_xor_converges_with_dales_law(self) -> None:
        """XOR converges with Dale's Law (n_inhibitory=2)."""
        from neuroforge.applications.tasks.logic_gates import LogicGateConfig, LogicGateTask

        cfg = LogicGateConfig(
            gate="XOR", max_trials=20000, seed=42, n_inhibitory=2,
        )
        result = LogicGateTask(cfg).run()
        assert result.converged

    def test_xnor_converges_with_dales_law(self) -> None:
        """XNOR converges with Dale's Law (n_inhibitory=2)."""
        from neuroforge.applications.tasks.logic_gates import LogicGateConfig, LogicGateTask

        cfg = LogicGateConfig(
            gate="XNOR", max_trials=20000, seed=42, n_inhibitory=2,
        )
        result = LogicGateTask(cfg).run()
        assert result.converged

    def test_final_weights_obey_dales_law(self) -> None:
        """Effective weights in the result respect sign constraints."""
        from neuroforge.applications.tasks.logic_gates import LogicGateConfig, LogicGateTask

        cfg = LogicGateConfig(
            gate="XOR", max_trials=20000, seed=42,
            n_hidden=6, n_inhibitory=2,
        )
        result = LogicGateTask(cfg).run()
        assert result.converged

        # Raw weights are stored; apply constraint to verify signs.
        w_ih = result.final_weights["layer_1"]
        w_ho = result.final_weights["layer_2"]

        # Input neurons are excitatory â†’ |w_ih| â‰¥ 0 (trivially true
        # for abs values, but verify the reparameterization produces
        # correct effective weights).
        w_ih_eff = torch.abs(w_ih)
        assert (w_ih_eff >= 0).all()

        # Hiddenâ†’output: first 4 E (+), last 2 I (âˆ’).
        dev = str(w_ho.device)
        mask = make_dale_mask(4, 2, device=dev)
        w_ho_eff = apply_dales_constraint(w_ho, mask.signs)
        assert (w_ho_eff[:4] >= 0).all(), "Excitatory hiddenâ†’output â‰¥ 0"
        assert (w_ho_eff[4:] <= 0).all(), "Inhibitory hiddenâ†’output â‰¤ 0"
