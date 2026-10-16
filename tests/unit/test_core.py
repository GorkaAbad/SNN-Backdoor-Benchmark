from datetime import UTC, datetime

import numpy as np
import pytest
import torch

from sbb.core import result
from sbb.core.proxies import AttackerView, DefenderView, ScenarioInfo, TrainSetView
from sbb.core.trainer import Loss, Trainer, TrainerHooks, TrainSpec, TrainState, per_sample_loss
from sbb.data.module import DataModule
from sbb.data.poison import PoisonSet
from sbb.models.snn import SNNModel, build_model
from sbb.models.spec import ArchSpec
from sbb.utils.errors import SbbError, ThreatModelViolation
from tests.cards import attack_card, defense_card

INFO = ScenarioInfo(
    dataset="synthetic", sensor_size=(16, 16, 2), num_classes=4, T=4, input_shape=(2, 16, 16)
)
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


def _attacker_view(datamodule: DataModule, clean: SNNModel, **card: object) -> AttackerView:
    trainer = Trainer(TrainSpec(epochs=1, batch_size=8), CPU)
    return AttackerView(attack_card(**card), INFO, 0, datamodule, ARCH, trainer, clean)


def test_attacker_without_architecture_knowledge_cannot_read_it(datamodule: DataModule) -> None:
    clean = build_model(ARCH, (2, 16, 16), 4)
    view = _attacker_view(datamodule, clean)
    assert len(view.train_set) == 480
    with pytest.raises(ThreatModelViolation, match="arch"):
        view.arch  # noqa: B018
    assert _attacker_view(datamodule, clean, knowledge={"architecture": "known"}).arch == ARCH


@pytest.mark.parametrize(
    ("kind", "readable", "hidden"),
    [
        ("data_poisoning", [], ["trainer", "clean_model"]),
        ("training_control", ["arch", "trainer"], ["clean_model"]),
        ("supply_chain", ["arch", "trainer", "clean_model"], []),
    ],
)
def test_attacker_view_follows_the_scenario_kind(
    datamodule: DataModule, kind: str, readable: list[str], hidden: list[str]
) -> None:
    clean = build_model(ARCH, (2, 16, 16), 4)
    labels = {"labels": "dirty"} if kind == "data_poisoning" else {}
    view = _attacker_view(datamodule, clean, kind=kind, capabilities=labels)
    assert len(view.train_set) == 480  # every attacker knows the clean train split
    for name in readable:
        assert getattr(view, name) is not None
    for name in hidden:
        with pytest.raises(ThreatModelViolation, match=name):
            getattr(view, name)
    if kind == "supply_chain":  # a deep copy: changing it does not change the clean model
        before = [p.clone() for p in clean.parameters()]
        copy = view.clean_model
        assert copy is not clean
        assert all(torch.equal(a, b) for a, b in zip(copy.parameters(), before, strict=True))
        with torch.no_grad():
            next(copy.parameters()).add_(1.0)
        assert all(torch.equal(a, b) for a, b in zip(clean.parameters(), before, strict=True))


def test_attacker_trains_on_its_own_poison_set(datamodule: DataModule) -> None:
    view = _attacker_view(
        datamodule, build_model(ARCH, (2, 16, 16), 4), kind="training_control", capabilities={}
    )
    overlay = PoisonSet(np.array([1]), (datamodule.dataset[1],), np.array([3]))
    _, y, index = next(iter(view.poisoned_loader(overlay, batch_size=480, shuffle=False)))
    assert y[index.tolist().index(1)] == 3


def test_train_set_view_shows_the_poisoned_split(datamodule: DataModule) -> None:
    sample = datamodule.dataset[1]
    overlay = PoisonSet(np.array([1]), (sample,), np.array([0]))
    view = TrainSetView(datamodule, overlay)
    assert view.labels()[1] == 0 and view[1].label == 0 and view[2].label == 2


class _LossCheck(TrainerHooks):
    """Checks that the Trainer computes the per-sample loss its TrainSpec names."""

    def __init__(self, kind: Loss) -> None:
        self.kind = kind
        self.batches = 0

    def loss(self, state: TrainState, logits: torch.Tensor, targets: torch.Tensor) -> None:
        assert state.per_sample_loss is not None
        assert torch.equal(state.per_sample_loss, per_sample_loss(self.kind, logits, targets))
        self.batches += 1


@pytest.mark.parametrize("loss", ["cross_entropy", "mse"])
def test_trainer_learns_synthetic(datamodule: DataModule, loss: Loss) -> None:
    torch.manual_seed(0)
    model = build_model(ARCH, (2, 16, 16), 4)
    trainer = Trainer(TrainSpec(epochs=2, batch_size=32, lr=1e-2, loss=loss), CPU)
    check = _LossCheck(loss)
    trainer.fit(model, datamodule.train_loader(None, 32), hooks=[check])
    assert check.batches == 30  # 2 epochs of 15 batches
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


def test_provenance_keeps_the_versions_read_at_import(monkeypatch: pytest.MonkeyPatch) -> None:
    # A commit, a lockfile or a package upgrade on disk after the code was loaded must not reach
    # later results.
    monkeypatch.setattr(result, "_commit", lambda: "changed")
    monkeypatch.setattr(result, "_lockfile_sha256", lambda: "changed")
    monkeypatch.setattr(result, "version", lambda name: "changed")
    provenance = result.collect_provenance(CPU, "test", "test", datetime.now(UTC), 0.0)
    assert provenance.git_commit != "changed"
    assert provenance.lockfile_sha256 != "changed"
    assert provenance.spikingjelly != "changed"


def test_per_sample_losses() -> None:
    output, targets = torch.rand(5, 4), torch.tensor([0, 3, 1, 1, 2])
    mse = per_sample_loss("mse", output, targets)
    # The papers' training loss is F.mse_loss(firing_rate, one_hot), the mean of these.
    one_hot = torch.nn.functional.one_hot(targets, 4).float()
    assert torch.allclose(mse.mean(), torch.nn.functional.mse_loss(output, one_hot))
    assert mse.shape == (5,)
    cross_entropy = per_sample_loss("cross_entropy", output, targets)
    assert torch.allclose(
        cross_entropy, torch.nn.functional.cross_entropy(output, targets, reduction="none")
    )
