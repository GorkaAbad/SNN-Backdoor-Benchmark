"""Fine-tuning on the clean subset."""

from pathlib import Path

from sbb.core.cards import DefenseCard
from sbb.core.plugins import ModelDefense
from sbb.core.proxies import DefenderView
from sbb.core.registry import DEFENSES
from sbb.models.snn import SNNModel


@DEFENSES.register("fine_tuning")
class FineTuning(ModelDefense):
    """Retrains the model on clean data with the scenario recipe."""

    card = DefenseCard.load(Path(__file__).parent / "cards" / "fine_tuning.yaml")

    def repair(self, view: DefenderView) -> SNNModel:
        """Fine-tune the model on the clean subset."""
        data = view.clean_loader(batch_size=self.hparams["batch_size"])
        return view.trainer.fit(view.model, data, epochs=self.hparams["epochs"])
