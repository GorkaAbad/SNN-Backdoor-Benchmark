import numpy as np
import pytest
import torch

from sbb.core.proxies import AttackerView, DefenderView, ScenarioInfo, TrainSetView
from sbb.core.trainer import Trainer, TrainerHooks, TrainSpec, TrainState
from sbb.data.module import DataModule
from sbb.data.poison import PoisonSet
from sbb.models.snn import build_model
from sbb.models.spec import ArchSpec
from sbb.utils.errors import SbbError, ThreatModelViolation
from tests.cards import attack_card, defense_card

INFO = ScenarioInfo(dataset="synthetic", sensor_size=(16, 16, 2), num_classes=4, T=4)
ARCH = ArchSpec(name="tiny_lif_cnn", T=4)
CPU = torch.device("cpu")


def _defender_view(datamodule: DataModule, resources: list[str]) -> DefenderView:
    card = defense_card(view={"resources": resources, "clean_fraction": 0.05})
    model = build_model(ARCH, (2, 16, 16), 4)
    trainer = Trainer(TrainSpec(epochs=1, batch_size=8), CPU)
    return DefenderView(card, INFO, 0, trainer, model, TrainSetView(datamodule, None), datamodule)


def test_defense_reading_an_undeclared_resource_raises(datamodule: DataModule) -> None:
    view = _defender_view(datamodule, ["model"])
    assert view.model is not None
    with pytest.raises(ThreatModelViolation, match="clean_subset"):
        view.clean_loader(8)
    with pytest.raises(ThreatModelViolation, match="train_set"):
        view.train_set  # noqa: B018


def test_defender_gets_a_copy_of_the_model(datamodule: DataModule) -> None:
    view = _defender_view(datamodule, ["model", "clean_subset"])
    assert view.model is view.model and len(view.clean_loader(8).dataset) == 24  # type: ignore[arg-type]


def test_attacker_without_architecture_knowledge_cannot_read_it(datamodule: DataModule) -> None:
    train = TrainSetView(datamodule, None)
    view = AttackerView(attack_card(), INFO, 0, train, ARCH)
    assert len(view.train_set) == 480
    with pytest.raises(ThreatModelViolation, match="arch"):
        view.arch  # noqa: B018
    known = attack_card(knowledge={"architecture": "known"})
    assert AttackerView(known, INFO, 0, train, ARCH).arch == ARCH


def test_train_set_view_shows_the_poisoned_split(datamodule: DataModule) -> None:
    sample = datamodule.dataset[1]
    overlay = PoisonSet(np.array([1]), (sample,), np.array([0]))
    view = TrainSetView(datamodule, overlay)
    assert view.labels()[1] == 0 and view[1].label == 0 and view[2].label == 2


def test_trainer_learns_synthetic(datamodule: DataModule) -> None:
    torch.manual_seed(0)
    model = build_model(ARCH, (2, 16, 16), 4)
    trainer = Trainer(TrainSpec(epochs=2, batch_size=32, lr=1e-2), CPU)
    trainer.fit(model, datamodule.train_loader(None, 32))
    assert trainer.evaluate(model, datamodule.test_loader(64))["accuracy"] > 0.9


class _ZeroLoss(TrainerHooks):
    def __init__(self) -> None:
        self.seen: list[int] = []

    def loss(self, state: TrainState, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert state.per_sample_loss is not None
        self.seen.append(len(state.batch_indices))
        return logits.sum() * 0


def test_hooks_override_the_loss_once(datamodule: DataModule) -> None:
    model = build_model(ARCH, (2, 16, 16), 4)
    trainer = Trainer(TrainSpec(epochs=1, batch_size=240, schedule="none"), CPU)
    hook = _ZeroLoss()
    before = [p.clone() for p in model.parameters()]
    trainer.fit(model, datamodule.train_loader(None, 240), hooks=[hook])
    assert hook.seen == [240, 240]
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    with pytest.raises(SbbError, match="more than one"):
        trainer.fit(model, datamodule.train_loader(None, 240), hooks=[hook, _ZeroLoss()])
