"""Sneaky Spikes, moving trigger (port of the official code)."""

from pathlib import Path

from sbb.attacks.dirty_label import DirtyLabelPoisoning
from sbb.attacks.frame_triggers import MovingPatchTrigger
from sbb.core.cards import AttackCard
from sbb.core.registry import ATTACKS


@ATTACKS.register("sneaky_moving")
class SneakyMoving(DirtyLabelPoisoning):
    """Sneaky Spikes, moving trigger (port of the official code)."""

    card = AttackCard.load(Path(__file__).parent / "cards" / "sneaky_moving.yaml")
    trigger_cls = MovingPatchTrigger
