"""Shared parts of the SNN adaptations of defenses (docs/PLAN.md section 12, rule 6).

A ported defense names its adaptation in the card's adaptation_note and builds it from these
helpers, so each choice is made and reviewed once. Layers of the networks run in multi-step mode,
so their outputs are [T, B, C, ...] and C is dimension 2.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from spikingjelly.activation_based.neuron import BaseNode
from torch import nn

from sbb.models.snn import SNNModel
from sbb.utils.errors import SbbError


@torch.no_grad()
def channel_means(
    model: SNNModel, module: nn.Module, loader: Iterable[Any], device: torch.device
) -> torch.Tensor:
    """Mean output per channel of module over the batches (x, y, index) of loader, shape [C].

    The model runs in eval mode, and the mean runs over time, samples and space. For a spiking
    neuron layer it is the firing rate of each channel, which fine-pruning ranks by. loader can be
    any iterable of batches, for example a part of a DataLoader.
    """
    sums: torch.Tensor | None = None
    count = 0

    def record(_module: nn.Module, _inputs: object, output: torch.Tensor) -> None:
        nonlocal sums, count
        flat = output.movedim(2, 0).flatten(1)
        part = flat.sum(1, dtype=torch.float64)
        sums = part if sums is None else sums + part
        count += flat.shape[1]

    training = model.training
    handle = module.register_forward_hook(record)
    try:
        model.eval()
        for x, _, _ in loader:
            model(x.to(device))
    finally:
        handle.remove()
        model.train(training)
    if sums is None:
        raise SbbError("channel_means needs a loader with at least one batch")
    return (sums / count).cpu()


def zero_channels(
    module: nn.Conv2d | nn.BatchNorm2d, channels: Sequence[int] | torch.Tensor
) -> None:
    """Prune output channels of a convolution or a batch norm by zeroing their weight and bias.

    The pruning lives in the weights, so it survives the checkpoint the Game stores. Zeroing a
    batch norm's scale and shift silences the channel: its neurons get no input current. Zeroing
    a convolution's channels keeps the shift of the batch norm after it.
    """
    if module.weight is None:
        raise SbbError(f"{type(module).__name__} has no weights to prune")
    # A list or tuple names channels. Used directly, a tuple would be one multi-dimensional index.
    # A tensor, of indices or a boolean mask, is used as given.
    if not isinstance(channels, torch.Tensor):
        channels = torch.tensor(list(channels), dtype=torch.long)
    with torch.no_grad():
        for parameter in (module.weight, module.bias):
            if parameter is not None:
                parameter[channels] = 0


@dataclass(frozen=True)
class ConvBlock:
    """A convolution, the batch norm after it, and the first neuron layer after that.

    Attributes:
        name: Name of the batch norm in the model.
        conv: The convolution.
        bn: The batch norm.
        neuron: The neuron layer that integrates the batch norm's output, if there is one before
            the next convolution.
    """

    name: str
    conv: nn.Conv2d
    bn: nn.BatchNorm2d
    neuron: BaseNode | None


def conv_blocks(model: nn.Module) -> list[ConvBlock]:
    """Every batch norm with the last convolution before it, as CLP pairs them, and its neurons."""
    modules = list(model.named_modules())
    blocks, conv = [], None
    for i, (name, module) in enumerate(modules):
        if isinstance(module, nn.Conv2d):
            conv = module
        elif isinstance(module, nn.BatchNorm2d) and conv is not None:
            node = None
            for _, after in modules[i + 1 :]:
                if isinstance(after, nn.Conv2d):
                    break
                if isinstance(after, BaseNode):
                    node = after
                    break
            blocks.append(ConvBlock(name, conv, module, node))
    return blocks


def superpose(x: torch.Tensor, overlay: torch.Tensor) -> torch.Tensor:
    """Superpose two inputs [..., T, P, H, W] by adding their frames, as Flashy's STRIP does.

    Inputs reach an InputDefense as frames, so superposition adds event counts. Merging the two
    event streams before the representation would give other frames, because it moves the frame
    boundaries.
    """
    return x + overlay
