"""Fewer time steps, an sbb SNN-native baseline: the model sees only the first k time steps."""

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from sbb.core.cards import DefenseCard
from sbb.core.plugins import ModelDefense
from sbb.core.proxies import DefenderView
from sbb.core.registry import DEFENSES
from sbb.models.snn import SNNModel


@DEFENSES.register("fewer_steps")
class FewerSteps(ModelDefense):
    """Keeps the fewest first time steps whose clean accuracy stays within max_drop of all T.

    A trigger that needs many time steps to build up, or that sits late in the sample, loses its
    effect. k is chosen on the clean subset only, from 1 up, the first k that holds the clean
    accuracy. The weights stay as they are.
    """

    card = DefenseCard.load(Path(__file__).parent / "cards" / "fewer_steps.yaml")

    def __init__(self, hparams: dict[str, Any]) -> None:
        super().__init__(hparams)
        self._steps: Steps | None = None

    def repair(self, view: DefenderView) -> SNNModel:
        """Choose k on the clean subset. All T steps when no smaller k holds the accuracy."""
        model, trainer = view.model, view.trainer
        clean = view.clean_loader(self.hparams["batch_size"])
        steps = self.extra(model)
        floor = trainer.evaluate(model, clean)["accuracy"] - self.hparams["max_drop"]
        handle = steps.attach(model)
        try:
            for k in range(1, model.spec.T + 1):
                steps.k.fill_(k)
                if trainer.evaluate(model, clean)["accuracy"] >= floor:
                    break
            else:
                steps.k.fill_(model.spec.T)
        finally:
            handle.remove()
        return model

    def extra(self, model: SNNModel) -> "Steps":
        """The number of kept steps, chosen by repair or loaded by the Game."""
        if self._steps is None:
            self._steps = Steps(model.spec.T)
        return self._steps

    def attach(self, model: SNNModel, extra: nn.Module) -> None:
        """Make the model loaded back see only the kept steps."""
        assert isinstance(extra, Steps)
        extra.attach(model)


class Steps(nn.Module):
    """The number k of first time steps the network sees, as a buffer the Game stores."""

    k: torch.Tensor

    def __init__(self, T: int) -> None:
        super().__init__()
        self.register_buffer("k", torch.tensor(T))

    def attach(self, model: SNNModel) -> RemovableHandle:
        """Cut the input of model's network, [T, B, ...], to its first k steps."""

        def keep(_net: nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            return (inputs[0][: int(self.k)], *inputs[1:])

        return model.net.register_forward_pre_hook(keep)
