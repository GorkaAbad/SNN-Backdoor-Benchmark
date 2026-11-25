"""TaCT, the targeted contamination attack of Tang et al. (USENIX Security 2021), on frames."""

from pathlib import Path
from typing import Any

from sbb.attacks.conservative import ConservativePoisoning
from sbb.attacks.frame_triggers import FramePatchTrigger
from sbb.attacks.selectors import ClassSelector
from sbb.core.cards import AttackCard
from sbb.core.registry import ATTACKS
from sbb.utils.errors import SbbError


@ATTACKS.register("tact")
class TaCT(ConservativePoisoning):
    """A source-specific backdoor: the patch flips the source classes only.

    Payload samples come from the source classes, cover samples from the cover classes, and both
    carry the same patch. The cover samples teach the model to ignore the patch elsewhere, which
    keeps the prediction entropy of overlaid inputs high, so it targets STRIP.
    """

    card = AttackCard.load(Path(__file__).parent / "cards" / "tact.yaml")

    def __init__(self, hparams: dict[str, Any], target: int) -> None:
        super().__init__(hparams, target)
        source, cover = tuple(hparams["source"]), tuple(hparams["cover"])
        if target in source + cover or set(source) & set(cover):
            raise SbbError(
                f"TaCT needs disjoint source {source}, cover {cover} and target {target}"
            )
        self.trigger = self.train_trigger = FramePatchTrigger(**hparams["trigger"])
        self.selector, self.cover_selector = ClassSelector(source), ClassSelector(cover)
        self.eta = hparams["eta"]
        self.source_classes = source
