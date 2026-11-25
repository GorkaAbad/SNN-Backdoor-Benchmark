"""Adap-Patch of Qi et al. (ICLR 2023), on frames."""

from pathlib import Path
from typing import Any

from sbb.attacks.conservative import ConservativePoisoning
from sbb.attacks.frame_triggers import PatchesTrigger
from sbb.attacks.selectors import RandomSelector
from sbb.core.cards import AttackCard
from sbb.core.registry import ATTACKS


@ATTACKS.register("adap_patch")
class AdapPatch(ConservativePoisoning):
    """One of several faint patches per poisoned sample, some of them fully opaque at test time."""

    card = AttackCard.load(Path(__file__).parent / "cards" / "adap_patch.yaml")

    def __init__(self, hparams: dict[str, Any], target: int) -> None:
        super().__init__(hparams, target)
        patches = tuple(hparams["patches"])
        self.train_trigger = PatchesTrigger(patches, one=True)
        self.trigger = PatchesTrigger(tuple({**patches[i], "alpha": 1.0} for i in hparams["test"]))
        self.selector = self.cover_selector = RandomSelector()
        self.eta = hparams["eta"]
