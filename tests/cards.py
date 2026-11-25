"""Test cards shared by unit tests."""

from sbb.core.cards import AttackCard, DefenseCard


def attack_card(**overrides: object) -> AttackCard:
    fields: dict[str, object] = {
        "name": "test_attack",
        "version": "1.0.0",
        "display_name": "Test Attack",
        "title": "Test attack",
        "kind": "data_poisoning",
        "capabilities": {"labels": "dirty"},
        "knowledge": {},
    }
    return AttackCard.model_validate({**fields, **overrides})


def defense_card(**overrides: object) -> DefenseCard:
    fields: dict[str, object] = {
        "name": "test_defense",
        "version": "1.0.0",
        "display_name": "Test Defense",
        "title": "Test defense",
        "stage": "post_training",
        "view": {"resources": ["model"]},
    }
    return DefenseCard.model_validate({**fields, **overrides})
