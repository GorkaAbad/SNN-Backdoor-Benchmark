"""Fine-pruning (Liu et al., RAID 2018), adapted to SNNs."""

from pathlib import Path

import torch
from torch import nn

from sbb.core.cards import DefenseCard
from sbb.core.plugins import ModelDefense
from sbb.core.proxies import DefenderView
from sbb.core.registry import DEFENSES
from sbb.core.trainer import TrainerHooks, TrainState
from sbb.defenses.utils import channel_means, conv_blocks, zero_channels
from sbb.models.snn import SNNModel
from sbb.utils.errors import SbbError


class _KeepPruned(TrainerHooks):
    """Zeroes the pruned channels after every step, so fine-tuning cannot revive them."""

    def __init__(self, bn: nn.BatchNorm2d, channels: torch.Tensor) -> None:
        self.bn = bn
        self.channels = channels

    def on_batch_end(self, state: TrainState) -> None:
        zero_channels(self.bn, self.channels)


@DEFENSES.register("fine_pruning")
class FinePruning(ModelDefense):
    """Prunes the most dormant neurons of the last conv block, then fine-tunes on clean data.

    As in the paper, neurons are pruned one by one in increasing order of their activity on clean
    data, while the clean accuracy stays within max_drop of the unpruned model's. Activity is the
    spike rate, and pruning a channel zeroes its batch norm, so its neurons never fire.
    """

    card = DefenseCard.load(Path(__file__).parent / "cards" / "fine_pruning.yaml")

    def repair(self, view: DefenderView) -> SNNModel:
        """Prune, then fine-tune on the clean subset with the pruned channels held at zero."""
        model, trainer = view.model, view.trainer
        clean = view.clean_loader(self.hparams["batch_size"])
        block = conv_blocks(model)[-1]
        if block.neuron is None:
            raise SbbError("fine-pruning needs neurons after the last convolution")
        rates = channel_means(model, block.neuron, clean, trainer.device)
        order = torch.argsort(rates, stable=True)  # the most dormant first
        baseline = trainer.evaluate(model, clean)["accuracy"]
        weight, bias = block.bn.weight.detach().clone(), block.bn.bias.detach().clone()
        pruned = 0
        for channel in order.tolist():
            zero_channels(block.bn, [channel])
            if baseline - trainer.evaluate(model, clean)["accuracy"] > self.hparams["max_drop"]:
                with torch.no_grad():  # one channel too many: give it back
                    block.bn.weight[channel] = weight[channel]
                    block.bn.bias[channel] = bias[channel]
                break
            pruned += 1
        hooks = [_KeepPruned(block.bn, order[:pruned])]
        return trainer.fit(model, clean, hooks=hooks, epochs=self.hparams["epochs"])
