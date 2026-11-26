"""SIG of Barni et al. (ICIP 2019), clean-label, on frames."""

from pathlib import Path
from typing import Any

from sbb.attacks.frame_triggers import SigTrigger
from sbb.attacks.labels import CleanLabel
from sbb.attacks.selectors import TargetClassSelector
from sbb.core.cards import AttackCard
from sbb.core.plugins import PoisoningAttack
from sbb.core.registry import ATTACKS


@ATTACKS.register("sig")
class Sig(PoisoningAttack):
    """Clean-label poisoning of target-class samples with a sinusoid over the frames."""

    card = AttackCard.load(Path(__file__).parent / "cards" / "sig.yaml")

    def __init__(self, hparams: dict[str, Any], target: int) -> None:
        super().__init__(hparams, target)
        self.trigger = SigTrigger(**hparams["trigger"])
        self.selector = TargetClassSelector()
        self.labels = CleanLabel()
