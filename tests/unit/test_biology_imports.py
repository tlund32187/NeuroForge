"""Tests for biology package boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_biology_tree_contains_expected_modules() -> None:
    root = Path(__file__).parents[2] / "src" / "neuroforge" / "biology"
    expected = {
        "__init__.py",
        "compartments/types.py",
        "compartments/base.py",
        "compartments/state.py",
        "compartments/factory.py",
        "compartments/soma/params.py",
        "compartments/soma/model.py",
        "compartments/soma/factory.py",
        "compartments/dendrite/params.py",
        "compartments/dendrite/model.py",
        "compartments/dendrite/factory.py",
        "compartments/axon/params.py",
        "compartments/axon/model.py",
        "compartments/axon/factory.py",
        "neurons/base.py",
        "neurons/state.py",
        "neurons/params.py",
        "neurons/factory.py",
        "neurons/builders/single_compartment.py",
        "neurons/builders/two_compartment.py",
        "neurons/builders/multi_compartment.py",
        "neurons/models/lif/params.py",
        "neurons/models/lif/model.py",
        "neurons/models/lif/surrogate.py",
        "synapses/base.py",
        "synapses/state.py",
        "synapses/topology.py",
        "synapses/factory.py",
        "synapses/models/static.py",
        "synapses/models/delayed_static.py",
        "synapses/models/dales_static.py",
        "synapses/models/chemical.py",
        "synapses/models/electrical_gap_junction.py",
        "receptors/base.py",
        "receptors/factory.py",
        "receptors/ampa.py",
        "receptors/nmda.py",
        "receptors/gabaa.py",
        "receptors/gabab.py",
        "ion_channels/base.py",
        "ion_channels/factory.py",
        "ion_channels/sodium.py",
        "ion_channels/potassium.py",
        "ion_channels/calcium.py",
        "ion_channels/leak.py",
        "plasticity/base.py",
        "plasticity/traces.py",
        "plasticity/eligibility.py",
        "plasticity/factory.py",
        "plasticity/rules/stdp.py",
        "plasticity/rules/triplet_stdp.py",
        "plasticity/rules/bcm.py",
        "plasticity/rules/rstdp.py",
        "plasticity/rules/metaplasticity.py",
        "neuromodulators/fields.py",
        "neuromodulators/diffusion.py",
        "neuromodulators/dopamine.py",
        "neuromodulators/acetylcholine.py",
        "neuromodulators/serotonin.py",
        "astrocytes/base.py",
        "astrocytes/state.py",
        "astrocytes/factory.py",
        "astrocytes/tripartite.py",
    }

    missing = [path for path in sorted(expected) if not (root / path).is_file()]
    assert missing == []


@pytest.mark.unit
def test_compartment_canonical_import() -> None:
    from neuroforge.biology.compartments.types import Compartment

    assert Compartment.SOMA.value == "soma"


@pytest.mark.unit
def test_neuron_model_canonical_imports() -> None:
    from neuroforge.biology.neurons.models.lif.model import LIFModel
    from neuroforge.biology.neurons.models.lif.params import LIFParams
    from neuroforge.biology.neurons.models.lif.surrogate import SurrogateLIFModel

    assert LIFModel.__name__ == "LIFModel"
    assert LIFParams.__name__ == "LIFParams"
    assert SurrogateLIFModel.__name__ == "SurrogateLIFModel"


@pytest.mark.unit
def test_synapse_model_canonical_imports() -> None:
    from neuroforge.biology.synapses.models.dales_static import DalesStaticSynapseModel
    from neuroforge.biology.synapses.models.delayed_static import DelayedStaticSynapseModel
    from neuroforge.biology.synapses.models.static import StaticSynapseModel

    assert StaticSynapseModel.__name__ == "StaticSynapseModel"
    assert DelayedStaticSynapseModel.__name__ == "DelayedStaticSynapseModel"
    assert DalesStaticSynapseModel.__name__ == "DalesStaticSynapseModel"


@pytest.mark.unit
def test_plasticity_rule_canonical_imports() -> None:
    from neuroforge.biology.plasticity.rules.rstdp import RSTDPParams, RSTDPRule
    from neuroforge.learning.training_loop import OnlineRSTDPTrainer

    assert RSTDPParams.__name__ == "RSTDPParams"
    assert RSTDPRule.__name__ == "RSTDPRule"
    assert OnlineRSTDPTrainer.__name__ == "OnlineRSTDPTrainer"
