"""Contract tests over every registered plugin. A new plugin gets these for free."""

import numpy as np
import pytest
import torch

from sbb.core.cards import ScenarioKind, Stage
from sbb.core.plugins import (
    STAGE_BASES,
    Attack,
    DataDefense,
    Defense,
    DetectionDefense,
    EvalContext,
    FrameTrigger,
    InputDefense,
    ModelAttack,
    ModelDefense,
    PoisoningAttack,
    TrainingDefense,
    Trigger,
)
from sbb.core.proxies import AttackerView, DefenderView, ScenarioInfo, TrainSetView
from sbb.core.registry import ATTACKS, DEFENSES, METRICS
from sbb.core.trainer import Trainer, TrainerHooks, TrainSpec
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


def _triggered(trigger: Trigger | FrameTrigger, dm: DataModule) -> np.ndarray:
    """The triggered events, or frames, of one sample."""
    sample = dm.dataset[1]
    if isinstance(trigger, FrameTrigger):
        return trigger.apply(dm.encode(sample), rng_for(0, 1)).numpy()
    return trigger.apply(sample, rng_for(0, 1)).events


def check_trigger(trigger: Trigger | FrameTrigger, datamodule: DataModule) -> None:
    """The contract of every trigger, raw-event or frame. Learned triggers are fitted first."""
    sample = datamodule.dataset[1]
    if isinstance(trigger, FrameTrigger):
        frames = datamodule.encode(sample)
        original = frames.clone()
        out = trigger.apply(frames, rng_for(0, 1))
        assert out.shape == frames.shape and out.dtype == frames.dtype
        assert not out.requires_grad, "triggered frames must not carry autograd history"
        assert torch.equal(frames, original), "a frame trigger must not change its input"
    else:
        _check_sample(trigger.apply(sample, rng_for(0, 1)), sample, datamodule)
    first, again = _triggered(trigger, datamodule), _triggered(trigger, datamodule)
    assert np.array_equal(first, again), "triggers must be deterministic for a given RNG"


def check_attack(cls: type[Attack], datamodule: DataModule) -> None:
    """The contract of every attack plugin. Also run on test-only attacks by the Game tests."""
    hparams = cls.card.hparams_for("synthetic", "tiny_lif_cnn")
    attack = cls(hparams, target=0)
    # The card's trigger domain decides where the Game applies the trigger.
    frames = cls.card.capabilities.trigger_domain == "frames"
    assert isinstance(attack.trigger, FrameTrigger if frames else Trigger)

    # The scenario kind decides the base class: the Game plays the two kinds differently.
    assert isinstance(attack, PoisoningAttack) == (cls.card.kind == ScenarioKind.DATA_POISONING)
    assert isinstance(attack, PoisoningAttack | ModelAttack)

    trainer = Trainer(TrainSpec(epochs=1, batch_size=32), torch.device("cpu"))
    clean = build_model(ARCH, (2, 16, 16), 4)
    view = AttackerView(cls.card, INFO, 0, datamodule, ARCH, trainer, clean)
    if isinstance(attack, PoisoningAttack):
        budget, pool = Budget(), datamodule.defender_pool
        poison = attack.poison(view, budget, pool)  # also fits a learned trigger
        labels = cls.card.capabilities.labels
        assert labels is not None
        poison.validate(budget, 480, datamodule.labels, labels, pool)
        assert (poison.frame_trigger is not None) == frames
        for s in poison.samples:
            _check_sample(s, datamodule.dataset[s.index], datamodule)
    else:
        assert isinstance(attack, ModelAttack)
        before = _triggered(attack.trigger, datamodule)
        model = attack.make_model(view)
        assert (model.spec, model.input_shape, model.num_classes) == (ARCH, (2, 16, 16), 4)
        after = _triggered(attack.trigger, datamodule)
        if (learned := attack.trigger.learned()) is not None:
            # A learned trigger, which the Game stores with the victim: its state must restore it.
            restored = cls(hparams, target=0).trigger
            restored_part = restored.learned()
            assert restored_part is not None
            restored_part.load_state_dict(learned.state_dict())
            assert np.array_equal(_triggered(restored, datamodule), after), "state does not restore"
        else:
            # A cached victim skips make_model, so the trigger must not depend on it.
            assert np.array_equal(after, before), "make_model changed the trigger"
    assert isinstance(attack.trigger, FrameTrigger if frames else Trigger), "the trigger changed"
    check_trigger(attack.trigger, datamodule)


@pytest.mark.parametrize("name", ATTACKS.names())
def test_attack(name: str, datamodule: DataModule) -> None:
    cls = ATTACKS.get(name)
    assert cls.card.name == name
    check_attack(cls, datamodule)


def check_defense(cls: type[Defense], datamodule: DataModule) -> None:
    """The contract of every defense plugin. Also run on test-only defenses by the Game tests."""
    assert issubclass(cls, STAGE_BASES[cls.card.stage])
    defense = cls(cls.card.hparams_for("synthetic", "tiny_lif_cnn"))
    model = build_model(ARCH, (2, 16, 16), 4)
    trainer = Trainer(TrainSpec(epochs=1, batch_size=8), torch.device("cpu"))
    train_set = TrainSetView(datamodule, None)
    view = DefenderView(cls.card, INFO, 0, trainer, model, train_set, datamodule)
    x = torch.rand(3, 4, 2, 16, 16)
    if isinstance(defense, ModelDefense):
        repaired = defense.repair(view)
        assert isinstance(repaired, SNNModel) and repaired is not model
        assert repaired.spec == ARCH
        assert repaired(x).shape == (3, 4)
    elif isinstance(defense, DataDefense):
        keep = defense.sanitize(view)
        assert keep.ndim == 1 and np.issubdtype(keep.dtype, np.integer)
        assert len(np.unique(keep)) == len(keep) and ((keep >= 0) & (keep < len(train_set))).all()
    elif isinstance(defense, TrainingDefense):
        assert isinstance(defense.hooks(view), TrainerHooks)
    elif isinstance(defense, InputDefense):
        defense.fit(view)
        assert isinstance(defense.threshold, float)
        scores = defense.score(x)
        assert scores.shape == (3,) and torch.isfinite(scores).all()
    else:
        assert isinstance(defense, DetectionDefense)
        assert np.isfinite(defense.verdict(view))


@pytest.mark.parametrize("name", DEFENSES.names())
def test_defense(name: str, datamodule: DataModule) -> None:
    cls = DEFENSES.get(name)
    assert cls.card.name == name
    check_defense(cls, datamodule)


@pytest.mark.parametrize("name", METRICS.names())
def test_metric(name: str) -> None:
    cls = METRICS.get(name)
    assert cls.name == name and cls.stages <= frozenset(Stage)
    rng = np.random.default_rng(0)
    ctx = EvalContext(
        *rng.integers(0, 4, (4, 50)),
        target=0,
        clean_accepted=rng.random(50) < 0.9,
        triggered_accepted=rng.random(50) < 0.5,
        clean_scores=rng.random(50),
        triggered_scores=rng.random(50) + 0.5,
    )
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
