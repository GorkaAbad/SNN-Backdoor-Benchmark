"""T-Backdoor, latency trigger (Miah, Vu, Bi)."""

from pathlib import Path

from sbb.attacks.dirty_label import DirtyLabelPoisoning
from sbb.attacks.temporal_triggers import LatencyTrigger
from sbb.core.cards import AttackCard
from sbb.core.registry import ATTACKS


@ATTACKS.register("tbackdoor_latency")
class TBackdoorLatency(DirtyLabelPoisoning):
    """T-Backdoor, latency trigger: a purely temporal trigger that only reorders the frames."""

    card = AttackCard.load(Path(__file__).parent / "cards" / "tbackdoor_latency.yaml")
    trigger_cls = LatencyTrigger
