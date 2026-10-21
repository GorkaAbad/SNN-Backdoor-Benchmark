"""Flashy Backdoor, digital flash trigger (port of the official code)."""

from pathlib import Path

from sbb.attacks.dirty_label import DirtyLabelPoisoning
from sbb.attacks.frame_triggers import FlashTrigger
from sbb.core.cards import AttackCard
from sbb.core.registry import ATTACKS


@ATTACKS.register("flashy")
class Flashy(DirtyLabelPoisoning):
    """Flashy Backdoor, digital flash trigger (port of the official code)."""

    card = AttackCard.load(Path(__file__).parent / "cards" / "flashy.yaml")
    trigger_cls = FlashTrigger
