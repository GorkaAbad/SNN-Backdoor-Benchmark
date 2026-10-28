"""Threshold recalibration, an sbb SNN-native baseline: every firing threshold scaled up."""

from pathlib import Path
from typing import Any

import torch
from torch import nn

from sbb.core.cards import DefenseCard
from sbb.core.plugins import ModelDefense
from sbb.core.proxies import DefenderView
from sbb.core.registry import DEFENSES
from sbb.models.snn import SNNModel


@DEFENSES.register("threshold_recalibration")
class ThresholdRecalibration(ModelDefense):
    """Scales every neuron's threshold by the largest factor that holds the clean accuracy.

    Higher thresholds let fewer and stronger inputs through, which can silence the extra spikes a
    trigger adds. The factors of scales are tried in increasing order on the clean subset, and the
    last one whose clean accuracy stays within max_drop of the original is kept. The weights stay
    as they are.
    """

    card = DefenseCard.load(Path(__file__).parent / "cards" / "threshold_recalibration.yaml")

    def __init__(self, hparams: dict[str, Any]) -> None:
        super().__init__(hparams)
        self._scale: Scale | None = None

    def repair(self, view: DefenderView) -> SNNModel:
        """Choose the factor on the clean subset, then restore the thresholds of the model."""
        model, trainer = view.model, view.trainer
        clean = view.clean_loader(self.hparams["batch_size"])
        scale = self.extra(model)
        neurons = [neuron for _, neuron in model.named_neurons()]
        original = [neuron.v_threshold for neuron in neurons]
        baseline = trainer.evaluate(model, clean)["accuracy"]
        try:
            for factor in sorted(self.hparams["scales"]):
                for neuron, threshold in zip(neurons, original, strict=True):
                    neuron.v_threshold = threshold * factor
                if baseline - trainer.evaluate(model, clean)["accuracy"] > self.hparams["max_drop"]:
                    break
                scale.scale.fill_(factor)
        finally:
            for neuron, threshold in zip(neurons, original, strict=True):
                neuron.v_threshold = threshold
        return model

    def extra(self, model: SNNModel) -> "Scale":
        """The threshold factor, chosen by repair or loaded by the Game."""
        if self._scale is None:
            self._scale = Scale()
        return self._scale

    def attach(self, model: SNNModel, extra: nn.Module) -> None:
        """Scale the thresholds of the model loaded back."""
        assert isinstance(extra, Scale)
        for _, neuron in model.named_neurons():
            neuron.v_threshold = neuron.v_threshold * float(extra.scale)


class Scale(nn.Module):
    """The threshold factor, as a buffer the Game stores. 1 leaves the thresholds as they are."""

    scale: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(1.0))
