"""Fine-tuning injection, the supply-chain baseline: fine-tunes the clean model on poisoned data."""

from pathlib import Path
from typing import Any

import numpy as np

from sbb.attacks.sneaky_static import SneakyStatic
from sbb.core.cards import AttackCard
from sbb.core.plugins import ModelAttack
from sbb.core.proxies import AttackerView
from sbb.core.registry import ATTACKS
from sbb.data.poison import Budget
from sbb.models.snn import SNNModel


@ATTACKS.register("finetune_injection")
class FinetuneInjection(ModelAttack):
    """Poisons the train split with the static frame patch and fine-tunes the clean model on it."""

    card = AttackCard.load(Path(__file__).parent / "cards" / "finetune_injection.yaml")

    def __init__(self, hparams: dict[str, Any], target: int) -> None:
        super().__init__(hparams, target)
        self.poisoner = SneakyStatic(hparams, target)
        self.trigger = self.poisoner.trigger

    def make_model(self, view: AttackerView) -> SNNModel:
        """Fine-tune a copy of the clean model with the scenario recipe on the poisoned split."""
        budget = Budget(poison_rate=self.hparams["poison_rate"], target=self.target)
        # The attacker trains its own copy, so the defender pool does not constrain it.
        poison = self.poisoner.poison(view, budget, np.empty(0, dtype=np.int64))
        loader = view.poisoned_loader(poison, self.hparams["batch_size"])
        return view.trainer.fit(view.clean_model, loader, epochs=self.hparams["epochs"])
