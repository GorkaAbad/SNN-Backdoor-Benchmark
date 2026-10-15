from pathlib import Path

import numpy as np
import pytest
import torch

from sbb.attacks.labels import DirtyLabel
from sbb.attacks.no_attack import NoAttack
from sbb.attacks.selectors import RandomSelector
from sbb.attacks.triggers import IdentityTrigger
from sbb.core.cards import AttackCard, DefenseCard
from sbb.core.game import Game
from sbb.core.plugins import DataDefense, ModelDefense, PoisoningAttack
from sbb.core.proxies import DefenderView
from sbb.core.registry import ATTACKS, DEFENSES
from sbb.models.snn import SNNModel
from tests.cards import attack_card, defense_card
from tests.conftest import SCENARIO, make_cell, run_in_fresh_caches


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
