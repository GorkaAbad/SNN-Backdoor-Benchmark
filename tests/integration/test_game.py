from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.utils.prune
from torch.utils.data import DataLoader

from sbb.attacks.labels import DirtyLabel
from sbb.attacks.no_attack import NoAttack
from sbb.attacks.selectors import RandomSelector
from sbb.attacks.static_patch import StaticPatch
from sbb.attacks.triggers import IdentityTrigger
from sbb.core.cards import AttackCard, DefenseCard
from sbb.core.game import Game
from sbb.core.plugins import DataDefense, ModelAttack, ModelDefense, PoisoningAttack
from sbb.core.proxies import AttackerView, DefenderView
from sbb.core.registry import ATTACKS, DEFENSES
from sbb.data.module import DataModule
from sbb.data.poison import Budget
from sbb.data.sample import EventSample
from sbb.models.checkpoint import load_checkpoint
from sbb.models.snn import SNNModel, build_model
from tests.cards import attack_card, defense_card
from tests.conftest import SCENARIO, make_cell, run_in_fresh_caches
from tests.contracts.test_plugins import check_attack


def test_static_patch_backdoors_and_fine_tuning_runs(game: Game) -> None:
    clean = game.run(make_cell("no_attack", "no_defense"))
    attacked = game.run(make_cell("static_patch", "no_defense"))
    tuned = game.run(make_cell("static_patch", "fine_tuning"))
    assert [r.outcome for r in (clean, attacked, tuned)] == ["ok"] * 3, attacked.reason
    assert clean.metrics["ca"] > 0.9 and clean.metrics["asr"] < 0.1
    assert attacked.metrics["ca"] > 0.9 and attacked.metrics["asr"] > 0.9
    assert set(tuned.metrics) == {"ca", "asr", "ra"}
    assert attacked.artifacts["clean_model"] == clean.artifacts["clean_model"]
    assert attacked.artifacts["model"] == tuned.artifacts["model"]


def test_same_seed_gives_identical_results(tmp_path: Path) -> None:
    (first, first_weights), (second, second_weights) = run_in_fresh_caches(
        make_cell("static_patch", "fine_tuning"), tmp_path, torch.device("cpu")
    )
    assert first.metrics == second.metrics and first_weights == second_weights


class _Peeking(ModelDefense):
    card = defense_card(name="peeking")

    def repair(self, view: DefenderView) -> SNNModel:
        view.train_set  # noqa: B018  not declared
        return view.model


class _Sanitizer(DataDefense):
    card = defense_card(name="sanitizer", stage="pre_training", view={"resources": ["train_set"]})

    def sanitize(self, view: DefenderView) -> np.ndarray:
        return np.arange(len(view.train_set))


class _SupplyChain(NoAttack):
    card = attack_card(name="supply", kind="supply_chain", capabilities={})


@pytest.fixture
def test_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    for registry, name, cls in [
        (DEFENSES, "peeking", _Peeking),
        (DEFENSES, "sanitizer", _Sanitizer),
        (ATTACKS, "supply", _SupplyChain),
    ]:
        monkeypatch.setitem(registry._items, name, cls)


@pytest.mark.usefixtures("test_plugins")
def test_undeclared_resource_fails_the_cell(game: Game) -> None:
    result = game.run(make_cell("no_attack", "peeking"))
    assert result.outcome == "failed" and "ThreatModelViolation" in (result.reason or "")
    assert result.metrics == {}


@pytest.mark.usefixtures("test_plugins")
def test_data_defense_against_supply_chain_is_reported_not_applicable(game: Game) -> None:
    result = game.run(make_cell("supply", "sanitizer"))
    assert result.outcome == "not_applicable" and "train_set" in (result.reason or "")
    assert isinstance(_Sanitizer.card, DefenseCard) and isinstance(_SupplyChain.card, AttackCard)


class _LabelFlip(PoisoningAttack):
    card = attack_card(name="label_flip")

    def __init__(self, hparams: dict[str, object], target: int) -> None:
        super().__init__(hparams, target)
        self.trigger, self.selector, self.labels = IdentityTrigger(), RandomSelector(), DirtyLabel()


def test_attack_that_destroys_clean_accuracy_is_invalid(
    game: Game, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(ATTACKS._items, "label_flip", _LabelFlip)
    budget = SCENARIO.budget.model_copy(update={"poison_rate": 0.5})
    result = game.run(
        make_cell(
            "label_flip", "no_defense", scenario=SCENARIO.model_copy(update={"budget": budget})
        )
    )
    assert result.outcome == "invalid_attack", result.reason
    assert result.metrics["ca_backdoored"] < result.metrics["ca_clean"] - 0.03


class _Injection(ModelAttack):
    """Poisons the clean split with the static patch, then trains on it (test-only attack)."""

    card = attack_card(name="supply_injection", kind="supply_chain", capabilities={})
    received: dict[str, torch.Tensor] = {}  # the clean weights the last make_model was given

    def __init__(self, hparams: dict[str, object], target: int) -> None:
        super().__init__(hparams, target)
        hparams = StaticPatch.card.hparams_for("synthetic", "tiny_lif_cnn")
        self.poisoner = StaticPatch(hparams, target)

    def _poisoned(self, view: AttackerView) -> DataLoader[Any]:
        budget = Budget(poison_rate=0.1, target=self.target)
        poison = self.poisoner.poison(view, budget, np.empty(0, dtype=np.int64))
        return view.poisoned_loader(poison, batch_size=32)

    def make_model(self, view: AttackerView) -> SNNModel:
        model = view.clean_model
        _Injection.received = {k: v.clone() for k, v in model.state_dict().items()}
        return view.trainer.fit(model, self._poisoned(view), epochs=3)

    def trigger_test(self, sample: EventSample, rng: np.random.Generator) -> EventSample:
        return self.poisoner.trigger_test(sample, rng)


class _TrainingControl(_Injection):
    card = attack_card(name="trains_own_model", kind="training_control", capabilities={})

    def make_model(self, view: AttackerView) -> SNNModel:
        model = build_model(view.arch, view.scenario.input_shape, view.scenario.num_classes)
        return view.trainer.fit(model, self._poisoned(view))


class _OtherSpec(_Injection):
    card = attack_card(name="other_spec", kind="supply_chain", capabilities={})

    def make_model(self, view: AttackerView) -> SNNModel:
        wider = view.arch.model_copy(update={"kwargs": {"unused": 1}})
        return build_model(wider, view.scenario.input_shape, view.scenario.num_classes)


class _OtherClasses(_Injection):
    card = attack_card(name="other_classes", kind="supply_chain", capabilities={})

    def make_model(self, view: AttackerView) -> SNNModel:
        return build_model(view.arch, view.scenario.input_shape, view.scenario.num_classes + 1)


class _LowThreshold(_Injection):
    """Changes a neuron parameter in place, which a checkpoint cannot keep."""

    card = attack_card(name="low_threshold", kind="supply_chain", capabilities={})

    def make_model(self, view: AttackerView) -> SNNModel:
        model = view.clean_model
        for _, node in model.named_neurons():
            node.v_threshold = 0.05
        return model


class _Pruned(_Injection):
    """Changes the state-dict layout (weight_orig and weight_mask), which would not load."""

    card = attack_card(name="pruned", kind="supply_chain", capabilities={})

    def make_model(self, view: AttackerView) -> SNNModel:
        model = view.clean_model
        torch.nn.utils.prune.l1_unstructured(model.net[0], "weight", amount=0.5)
        return model


class _Hybrid(PoisoningAttack, ModelAttack):
    """A supply-chain attack that reuses PoisoningAttack.poison, as a fine-tuning port could."""

    card = attack_card(name="hybrid", kind="supply_chain", capabilities={})
    made = 0

    def __init__(self, hparams: dict[str, object], target: int) -> None:
        super().__init__(hparams, target)
        self.trigger, self.selector, self.labels = IdentityTrigger(), RandomSelector(), DirtyLabel()

    def make_model(self, view: AttackerView) -> SNNModel:
        _Hybrid.made += 1
        return view.clean_model


MODEL_ATTACKS = (
    _Injection,
    _TrainingControl,
    _OtherSpec,
    _OtherClasses,
    _LowThreshold,
    _Pruned,
    _Hybrid,
)


@pytest.fixture
def model_attacks(monkeypatch: pytest.MonkeyPatch) -> None:
    for cls in MODEL_ATTACKS:
        monkeypatch.setitem(ATTACKS._items, cls.card.name, cls)


@pytest.mark.parametrize("cls", [_Injection, _TrainingControl])
def test_test_model_attacks_meet_the_attack_contract(
    cls: type[ModelAttack], datamodule: DataModule
) -> None:
    check_attack(cls, datamodule)


@pytest.mark.usefixtures("model_attacks")
@pytest.mark.parametrize("attack", ["supply_injection", "trains_own_model"])
def test_model_attacks_backdoor_the_victim(game: Game, attack: str) -> None:
    result = game.run(make_cell(attack, "no_defense"))
    assert result.outcome == "ok", result.reason
    assert result.metrics["ca"] > 0.9 and result.metrics["asr"] > 0.9
    assert set(result.artifacts) == {"clean_model", "model", "defended_model"}
    assert result.artifacts["model"] != result.artifacts["clean_model"]


@pytest.mark.usefixtures("model_attacks")
def test_supply_chain_attacker_receives_the_trained_clean_model(game: Game) -> None:
    result = game.run(make_cell("supply_injection", "no_defense"))
    clean = load_checkpoint(game.cache.local.root / result.artifacts["clean_model"])
    received = _Injection.received
    assert received.keys() == clean.state_dict().keys()
    assert all(torch.equal(received[k], v) for k, v in clean.state_dict().items())


@pytest.mark.usefixtures("model_attacks")
def test_model_attack_key_follows_the_target(game: Game) -> None:
    first = game.run(make_cell("supply_injection", "no_defense"))
    other_defense = game.run(make_cell("supply_injection", "fine_tuning"))
    budget = SCENARIO.budget.model_copy(update={"target": 1})
    other_target = game.run(
        make_cell(
            "supply_injection",
            "no_defense",
            scenario=SCENARIO.model_copy(update={"budget": budget}),
        )
    )
    assert first.artifacts["model"] == other_defense.artifacts["model"]  # one victim, two defenses
    assert other_target.artifacts["model"] != first.artifacts["model"]


@pytest.mark.usefixtures("model_attacks")
@pytest.mark.parametrize("attack", ["other_spec", "other_classes", "low_threshold", "pruned"])
def test_model_attack_must_return_the_scenario_architecture(game: Game, attack: str) -> None:
    result = game.run(make_cell(attack, "no_defense"))
    assert result.outcome == "failed" and "ThreatModelViolation" in (result.reason or "")
    assert not list(game.cache.local.root.glob(".staging-*"))  # nothing was cached


@pytest.mark.usefixtures("model_attacks")
def test_game_plays_the_card_kind_not_the_class(game: Game) -> None:
    result = game.run(make_cell("hybrid", "no_defense"))
    assert result.outcome == "ok", result.reason
    assert _Hybrid.made == 1 and "poison_set" not in result.artifacts
