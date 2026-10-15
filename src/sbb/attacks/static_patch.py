"""Static patch trigger (Sneaky Spikes)."""

from pathlib import Path
from typing import Any

from sbb.attacks.labels import DirtyLabel
from sbb.attacks.selectors import RandomSelector
from sbb.attacks.triggers import PatchTrigger
from sbb.core.cards import AttackCard
from sbb.core.plugins import PoisoningAttack
from sbb.core.registry import ATTACKS


@ATTACKS.register("static_patch")
class StaticPatch(PoisoningAttack):
    """Dirty-label poisoning with a static square of events."""

    card = AttackCard.load(Path(__file__).parent / "cards" / "static_patch.yaml")

    def __init__(self, hparams: dict[str, Any], target: int) -> None:
        super().__init__(hparams, target)
        self.trigger = PatchTrigger(**hparams["trigger"])
        self.selector = RandomSelector()
        self.labels = DirtyLabel()
