import pydantic
import pytest

from sbb.core.cards import View, compatible
from sbb.data.poison import Budget
from sbb.utils.hashing import canonical_json
from tests.cards import attack_card, defense_card


def test_data_defense_against_supply_chain_is_not_applicable() -> None:
    data_defense = defense_card(stage="pre_training", view={"resources": ["train_set"]})
    ok, reasons = compatible(attack_card(kind="supply_chain"), data_defense, Budget())
    assert not ok and len(reasons) == 1 and "train_set" in reasons[0]


def test_clean_fraction_above_budget_is_not_applicable() -> None:
    greedy = defense_card(view={"resources": ["model", "clean_subset"], "clean_fraction": 0.2})
    ok, reasons = compatible(attack_card(), greedy, Budget(defender_clean_fraction=0.05))
    assert not ok and "0.2" in reasons[0]
    assert compatible(attack_card(), defense_card(), Budget()) == (True, [])


def test_view_serializes_resources_sorted() -> None:
    view = View(resources=frozenset({"model", "clean_subset", "train_set"}))
    assert canonical_json(view) == (
        '{"clean_fraction":0.0,"resources":["clean_subset","model","train_set"]}'
    )


def test_hparams_override_is_deep_merged() -> None:
    card = attack_card(
        hparams={
            "default": {"trigger": {"size": 3, "polarity": 1}, "other": 1},
            "nmnist/tiny_lif_cnn": {"trigger": {"size": 5}},
        }
    )
    assert card.hparams_for("nmnist", "tiny_lif_cnn") == {
        "trigger": {"size": 5, "polarity": 1},
        "other": 1,
    }
    assert card.hparams_for("synthetic", "tiny_lif_cnn")["trigger"]["size"] == 3


@pytest.mark.parametrize(
    "bad",
    [
        {"version": "1.0"},
        {"name": "Bad-Name"},
        {"capabilities": {}},
        {"unknown_field": 1},
        {"knowledge": {"defense": "aware:"}},
        {"knowledge": {"defense": "aware:strip, strip_per_step"}},
        {"knowledge": {"defense": "strip"}},
    ],
)
def test_invalid_attack_cards_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(pydantic.ValidationError):
        attack_card(**bad)


@pytest.mark.parametrize(
    ("stage", "resources"),
    [
        ("pre_training", ["model"]),  # sanitize needs the train set
        ("in_training", ["train_set"]),  # hooks need the training loop too
        ("in_training", ["training_loop"]),
    ],
)
def test_defense_cards_declare_what_their_stage_needs(stage: str, resources: list[str]) -> None:
    with pytest.raises(pydantic.ValidationError, match="must declare"):
        defense_card(stage=stage, view={"resources": resources})


def test_aware_attacks_name_their_target_defenses() -> None:
    assert attack_card().knowledge.aware_of == ()
    aware = attack_card(knowledge={"defense": "aware:strip,strip_per_step"})
    assert aware.knowledge.aware_of == ("strip", "strip_per_step")
