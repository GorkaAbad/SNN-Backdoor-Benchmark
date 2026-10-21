"""Sneaky Spikes, static trigger (port of the official code)."""

from pathlib import Path

from sbb.attacks.dirty_label import DirtyLabelPoisoning
from sbb.attacks.frame_triggers import FramePatchTrigger
from sbb.core.cards import AttackCard
from sbb.core.registry import ATTACKS


@ATTACKS.register("sneaky_static")
class SneakyStatic(DirtyLabelPoisoning):
    """Sneaky Spikes, static trigger (port of the official code)."""

    card = AttackCard.load(Path(__file__).parent / "cards" / "sneaky_static.yaml")
    trigger_cls = FramePatchTrigger
