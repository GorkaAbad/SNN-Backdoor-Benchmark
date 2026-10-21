"""Sneaky Spikes, smart trigger (port of the official code)."""

from pathlib import Path

from sbb.attacks.dirty_label import DirtyLabelPoisoning
from sbb.attacks.frame_triggers import SmartPatchTrigger
from sbb.core.cards import AttackCard
from sbb.core.registry import ATTACKS


@ATTACKS.register("sneaky_smart")
class SneakySmart(DirtyLabelPoisoning):
    """Sneaky Spikes, smart trigger (port of the official code)."""

    card = AttackCard.load(Path(__file__).parent / "cards" / "sneaky_smart.yaml")
    trigger_cls = SmartPatchTrigger
