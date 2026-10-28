"""Adap-Blend of Qi et al. (ICLR 2023), on frames."""

from pathlib import Path
from typing import Any

from sbb.attacks.conservative import ConservativePoisoning
from sbb.attacks.frame_triggers import BlendTrigger
from sbb.attacks.selectors import RandomSelector
from sbb.core.cards import AttackCard
from sbb.core.registry import ATTACKS


@ATTACKS.register("adap_blend")
class AdapBlend(ConservativePoisoning):
    """Blend with cover samples, a weaker training opacity and half of the pattern's pieces."""

    card = AttackCard.load(Path(__file__).parent / "cards" / "adap_blend.yaml")

    def __init__(self, hparams: dict[str, Any], target: int) -> None:
        super().__init__(hparams, target)
        pattern = hparams["pattern"]
        self.trigger = BlendTrigger(alpha=hparams["test_alpha"], **pattern)
        self.train_trigger = BlendTrigger(alpha=hparams["alpha"], keep=hparams["keep"], **pattern)
        self.selector = self.cover_selector = RandomSelector()
        self.eta = hparams["eta"]
