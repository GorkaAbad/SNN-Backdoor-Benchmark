"""Contract tests over every registered plugin. A new plugin gets these for free."""

import numpy as np
import pytest
import torch

from sbb.core.cards import ScenarioKind
from sbb.core.plugins import (
    STAGE_BASES,
    Attack,
    EvalContext,
    ModelAttack,
    ModelDefense,
    PoisoningAttack,
)
from sbb.core.proxies import AttackerView, DefenderView, ScenarioInfo, TrainSetView
from sbb.core.registry import ATTACKS, DEFENSES, METRICS
from sbb.core.trainer import Trainer, TrainSpec
from sbb.data.module import DataModule
from sbb.data.poison import Budget
from sbb.data.sample import DATASETS, EVENT_DTYPE, EventSample
from sbb.models.snn import ARCHITECTURES, SNNModel, build_model
from sbb.models.spec import ArchSpec
from sbb.utils.determinism import rng_for

INFO = ScenarioInfo(
    dataset="synthetic", sensor_size=(16, 16, 2), num_classes=4, T=4, input_shape=(2, 16, 16)
)
ARCH = ArchSpec(name="tiny_lif_cnn", T=4)


def _check_sample(sample: EventSample, original: EventSample, dm: DataModule) -> None:
    assert sample.events.dtype == EVENT_DTYPE
    assert np.all(np.diff(sample.events["t"]) >= 0), "events must stay sorted by t"
    assert (sample.label, sample.index) == (original.label, original.index)
    assert dm.encode(sample).shape == dm.encode(original).shape


def check_attack(cls: type[Attack], datamodule: DataModule) -> None:
    """The contract of every attack plugin. Also run on test-only attacks by the Game tests."""
    hparams = cls.card.hparams_for("synthetic", "tiny_lif_cnn")
    attack = cls(hparams, target=0)
    sample = datamodule.dataset[1]
    triggered = attack.trigger_test(sample, rng_for(0, 1))
    _check_sample(triggered, sample, datamodule)
    again = attack.trigger_test(sample, rng_for(0, 1))
    assert np.array_equal(triggered.events, again.events), "triggers must be deterministic"

    # The scenario kind decides the base class: the Game plays the two kinds differently.
    assert isinstance(attack, PoisoningAttack) == (cls.card.kind == ScenarioKind.DATA_POISONING)
    assert isinstance(attack, PoisoningAttack | ModelAttack)

    trainer = Trainer(TrainSpec(epochs=1, batch_size=32), torch.device("cpu"))
    clean = build_model(ARCH, (2, 16, 16), 4)
    view = AttackerView(cls.card, INFO, 0, datamodule, ARCH, trainer, clean)
    if isinstance(attack, PoisoningAttack):
        budget, pool = Budget(), datamodule.defender_pool
        poison = attack.poison(view, budget, pool)
        labels = cls.card.capabilities.labels
        assert labels is not None
        poison.validate(budget, 480, datamodule.labels, labels, pool)
        for s in poison.samples:
            _check_sample(s, datamodule.dataset[s.index], datamodule)
    else:
        assert isinstance(attack, ModelAttack)
        model = attack.make_model(view)
        assert (model.spec, model.input_shape, model.num_classes) == (ARCH, (2, 16, 16), 4)
        # A cached victim skips make_model, so the trigger must not depend on it.
        after = attack.trigger_test(sample, rng_for(0, 1))
        assert np.array_equal(after.events, triggered.events), "make_model changed the trigger"


@pytest.mark.parametrize("name", ATTACKS.names())
def test_attack(name: str, datamodule: DataModule) -> None:
    cls = ATTACKS.get(name)
    assert cls.card.name == name
    check_attack(cls, datamodule)


@pytest.mark.parametrize("name", DEFENSES.names())
def test_defense(name: str, datamodule: DataModule) -> None:
    cls = DEFENSES.get(name)
    assert cls.card.name == name
    assert issubclass(cls, STAGE_BASES[cls.card.stage])
    defense = cls(cls.card.hparams_for("synthetic", "tiny_lif_cnn"))
    model = build_model(ARCH, (2, 16, 16), 4)
    trainer = Trainer(TrainSpec(epochs=1, batch_size=8), torch.device("cpu"))
    view = DefenderView(
        cls.card, INFO, 0, trainer, model, TrainSetView(datamodule, None), datamodule
    )
    if isinstance(defense, ModelDefense):
        repaired = defense.repair(view)
        assert isinstance(repaired, SNNModel) and repaired is not model
        assert repaired.spec == ARCH
        assert repaired(torch.rand(2, 4, 2, 16, 16)).shape == (2, 4)


@pytest.mark.parametrize("name", METRICS.names())
def test_metric(name: str) -> None:
    cls = METRICS.get(name)
    assert cls.name == name
    rng = np.random.default_rng(0)
    ctx = EvalContext(*rng.integers(0, 4, (4, 50)), target=0)
    assert 0.0 <= cls().compute(ctx) <= 1.0


@pytest.mark.parametrize("name", DATASETS.names())
def test_dataset_declares_its_metadata(name: str) -> None:
    cls = DATASETS.get(name)
    assert cls.name == name
    W, H, P = cls.sensor_size
    assert min(W, H) > 0 and P == 2 and cls.num_classes > 1


@pytest.mark.parametrize("name", ARCHITECTURES.names())
def test_architecture(name: str) -> None:
    # 32x32 is the smallest input every architecture accepts (dvsgesture_net pools five times).
    model = build_model(ArchSpec(name=name, T=2), (2, 32, 32), 4)
    assert model(torch.rand(2, 2, 2, 32, 32)).shape == (2, 4)
    assert len(list(model.named_neurons())) > 0
