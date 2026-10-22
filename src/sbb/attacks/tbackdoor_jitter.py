"""T-Backdoor, jitter trigger (Miah, Vu, Bi)."""

from pathlib import Path

from sbb.attacks.dirty_label import DirtyLabelPoisoning
from sbb.attacks.temporal_triggers import JitterTrigger
from sbb.core.cards import AttackCard
from sbb.core.registry import ATTACKS


@ATTACKS.register("tbackdoor_jitter")
class TBackdoorJitter(DirtyLabelPoisoning):
    """T-Backdoor, jitter trigger: a purely temporal trigger that only reorders the frames."""

    card = AttackCard.load(Path(__file__).parent / "cards" / "tbackdoor_jitter.yaml")
    trigger_cls = JitterTrigger
