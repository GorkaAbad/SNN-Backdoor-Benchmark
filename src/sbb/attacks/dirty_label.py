"""Dirty-label poisoning of random samples with one trigger, the shape of most ported attacks."""

from typing import Any, ClassVar

from sbb.attacks.labels import DirtyLabel
from sbb.attacks.selectors import RandomSelector
from sbb.core.plugins import FrameTrigger, PoisoningAttack, Trigger


class DirtyLabelPoisoning(PoisoningAttack):
    """Poisons random samples with trigger_cls(**hparams["trigger"]), relabelled as the target.

    hparams["selector"], when given, configures the RandomSelector. Replications of the papers set
    include_target, because the papers poison every class.
    """

    trigger_cls: ClassVar[type[Trigger] | type[FrameTrigger]]

    def __init__(self, hparams: dict[str, Any], target: int) -> None:
        super().__init__(hparams, target)
        self.trigger = self.trigger_cls(**hparams["trigger"])
        self.selector = RandomSelector(**hparams.get("selector", {}))
        self.labels = DirtyLabel()
