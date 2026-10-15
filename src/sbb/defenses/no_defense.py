"""no_defense returns the model unchanged."""

from pathlib import Path

from sbb.core.cards import DefenseCard
from sbb.core.plugins import ModelDefense
from sbb.core.proxies import DefenderView
from sbb.core.registry import DEFENSES
from sbb.models.snn import SNNModel


@DEFENSES.register("no_defense")
class NoDefense(ModelDefense):
    """The undefended baseline."""

    card = DefenseCard.load(Path(__file__).parent / "cards" / "no_defense.yaml")

    def repair(self, view: DefenderView) -> SNNModel:
        """Return the model as it is."""
        return view.model
