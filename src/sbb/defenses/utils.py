"""Shared parts of the SNN adaptations of defenses (docs/PLAN.md section 12, rule 6).

A ported defense names its adaptation in the card's adaptation_note and builds it from these
helpers, so each choice is made and reviewed once. Layers of the networks run in multi-step mode,
so their outputs are [T, B, C, ...] and C is dimension 2.
"""

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import nn

from sbb.models.snn import SNNModel
from sbb.utils.errors import SbbError


@torch.no_grad()
def channel_means(
    model: SNNModel, module: nn.Module, loader: Iterable[Any], device: torch.device
) -> torch.Tensor:
    """Mean output per channel of module over the batches (x, y, index) of loader, shape [C].

    The model runs in eval mode, and the mean runs over time, samples and space. For a spiking
    neuron layer it is the firing rate of each channel. Flashy's fine-pruning ranks the channels
    of the last convolution by the absolute value of this mean, without the last two batches of
    its loader, which islice(loader, len(loader) - 2) reproduces.
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
    convolution's channels keeps the shift of the batch norm after it, as Flashy's fine-pruning
    does. Zeroing a batch norm's scale and shift silences the channel, as CLP does.
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


def conv_bn_pairs(model: nn.Module) -> list[tuple[str, nn.Conv2d, nn.BatchNorm2d]]:
    """(batch norm name, convolution, batch norm) for each batch norm after a convolution.

    Each batch norm pairs with the last convolution before it, as CLP pairs them.
    """
    pairs = []
    conv: nn.Conv2d | None = None
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) and conv is not None:
            pairs.append((name, conv, module))
        elif isinstance(module, nn.Conv2d):
            conv = module
    return pairs


def superpose(x: torch.Tensor, overlay: torch.Tensor) -> torch.Tensor:
    """Superpose two inputs [..., T, P, H, W] by adding their frames, as Flashy's STRIP does.

    Inputs reach an InputDefense as frames, so superposition adds event counts. Merging the two
    event streams before the representation would give other frames, because it moves the frame
    boundaries.
    """
    return x + overlay
