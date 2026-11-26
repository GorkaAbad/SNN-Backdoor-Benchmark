"""Narcissus of Zeng et al. (CCS 2023), clean-label, on frames."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from sbb.attacks.frame_triggers import NarcissusTrigger
from sbb.attacks.labels import CleanLabel
from sbb.attacks.selectors import TargetClassSelector
from sbb.core.cards import AttackCard
from sbb.core.plugins import PoisoningAttack
from sbb.core.proxies import AttackerView
from sbb.core.registry import ATTACKS
from sbb.data.poison import Budget, PoisonSet


@ATTACKS.register("narcissus")
class Narcissus(PoisoningAttack):
    """Clean-label poisoning of target-class samples with noise optimized on a surrogate.

    trigger, the test trigger, magnifies the noise. Poisoned samples get it unmagnified.
    """

    card = AttackCard.load(Path(__file__).parent / "cards" / "narcissus.yaml")
    trigger: NarcissusTrigger

    def __init__(self, hparams: dict[str, Any], target: int) -> None:
        super().__init__(hparams, target)
        self.trigger = NarcissusTrigger(target=target, **hparams["trigger"])
        self.selector = TargetClassSelector()
        self.labels = CleanLabel()

    def poison(self, view: AttackerView, budget: Budget, pool: np.ndarray) -> PoisonSet:
        """Poison as every PoisoningAttack does, then give the samples the unmagnified noise."""
        poison = super().poison(view, budget, pool)
        return replace(poison, frame_trigger=replace(self.trigger, magnify=1.0).apply)
