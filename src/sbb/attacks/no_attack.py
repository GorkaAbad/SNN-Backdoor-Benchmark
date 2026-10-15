"""no_attack gives the clean reference and the natural ASR."""

from pathlib import Path
from typing import Any

from sbb.attacks.labels import CleanLabel
from sbb.attacks.selectors import NoSelection
from sbb.attacks.triggers import IdentityTrigger
from sbb.core.cards import AttackCard
from sbb.core.plugins import PoisoningAttack
from sbb.core.registry import ATTACKS


@ATTACKS.register("no_attack")
class NoAttack(PoisoningAttack):
    """Poisons nothing. Its trigger is the identity."""

    card = AttackCard.load(Path(__file__).parent / "cards" / "no_attack.yaml")

    def __init__(self, hparams: dict[str, Any], target: int) -> None:
        super().__init__(hparams, target)
        self.trigger = IdentityTrigger()
        self.selector = NoSelection()
        self.labels = CleanLabel()
