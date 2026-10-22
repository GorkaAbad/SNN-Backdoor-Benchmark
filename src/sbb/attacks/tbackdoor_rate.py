"""T-Backdoor, rate trigger (Miah, Vu, Bi)."""

from pathlib import Path

from sbb.attacks.dirty_label import DirtyLabelPoisoning
from sbb.attacks.temporal_triggers import RateTrigger
from sbb.core.cards import AttackCard
from sbb.core.registry import ATTACKS


@ATTACKS.register("tbackdoor_rate")
class TBackdoorRate(DirtyLabelPoisoning):
    """T-Backdoor, rate trigger: a purely temporal trigger that only reorders the frames."""

    card = AttackCard.load(Path(__file__).parent / "cards" / "tbackdoor_rate.yaml")
    trigger_cls = RateTrigger
