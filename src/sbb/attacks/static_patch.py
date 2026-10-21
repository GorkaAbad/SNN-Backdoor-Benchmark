"""Static patch trigger on raw events (after Sneaky Spikes)."""

from pathlib import Path

from sbb.attacks.dirty_label import DirtyLabelPoisoning
from sbb.attacks.triggers import PatchTrigger
from sbb.core.cards import AttackCard
from sbb.core.registry import ATTACKS


@ATTACKS.register("static_patch")
class StaticPatch(DirtyLabelPoisoning):
    """Dirty-label poisoning with a static square of events."""

    card = AttackCard.load(Path(__file__).parent / "cards" / "static_patch.yaml")
    trigger_cls = PatchTrigger
