"""ANP, adversarial neuron pruning (Wu and Wang, NeurIPS 2021)."""

import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from sbb.core.cards import DefenseCard
from sbb.core.plugins import ModelDefense
from sbb.core.proxies import DefenderView
from sbb.core.registry import DEFENSES
from sbb.core.trainer import Trainer, per_sample_loss
from sbb.defenses.utils import conv_blocks, zero_channels
from sbb.models.snn import SNNModel


@DEFENSES.register("anp")
class ANP(ModelDefense):
    """Prunes the neurons whose mask falls when neurons are perturbed adversarially."""

    card = DefenseCard.load(Path(__file__).parent / "cards" / "anp.yaml")

    def repair(self, view: DefenderView) -> SNNModel:
        """Optimize the masks on a copy, then prune the scale of the low-mask channels."""
        h = self.hparams
        loader = view.clean_loader(h["batch_size"])
        args = (h["eps"], h["alpha"], h["lr"], h["iterations"])
        masks = neuron_masks(copy.deepcopy(view.model), loader, view.trainer, *args)
        model = view.model
        for block, mask in zip(conv_blocks(model), masks, strict=True):
            zero_channels(block.bn, (mask < h["threshold"]).cpu(), bias=False)
        return model


def perturb_bn(
    out: torch.Tensor, bn: nn.BatchNorm2d, weight_coeff: torch.Tensor, bias_coeff: torch.Tensor
) -> torch.Tensor:
    """The output of bn with its weight scaled by weight_coeff and its bias by bias_coeff.

    out is bn's output [T, B, C, H, W]: out - bias is the normalized input times the weight.
    """
    shape = [1, 1, -1] + [1] * (out.dim() - 3)
    bias = bn.bias.view(shape)
    scaled: torch.Tensor = (out - bias) * weight_coeff.view(shape) + bias * bias_coeff.view(shape)
    return scaled


def neuron_masks(
    model: SNNModel,
    loader: DataLoader[Any],
    trainer: Trainer,
    eps: float,
    alpha: float,
    lr: float,
    iterations: int,
) -> list[torch.Tensor]:
    """A mask in [0, 1] per batch-norm channel of each conv block, optimized as in ANP.

    Each iteration draws a random perturbation of the channels' scale and shift in [-eps, eps],
    takes one signed gradient step that raises the loss, then moves the masks to lower
    alpha L(masked) + (1 - alpha) L(masked and perturbed). model is changed (train mode, as in
    the official code), so pass a copy.
    """
    device, kind = trainer.device, trainer.spec.loss
    blocks = conv_blocks(model)
    model.to(device).train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def channels() -> list[torch.Tensor]:
        return [torch.zeros(b.bn.num_features, device=device, requires_grad=True) for b in blocks]

    masks = [mask.detach().add(1).requires_grad_() for mask in channels()]
    noise, noise_bias = channels(), channels()
    perturbed = [False]

    def hook(i: int) -> Any:
        def scale(bn: nn.BatchNorm2d, _: object, out: torch.Tensor) -> torch.Tensor:
            if perturbed[0]:
                return perturb_bn(out, bn, masks[i] + noise[i], 1 + noise_bias[i])
            return perturb_bn(out, bn, masks[i], torch.ones_like(masks[i]))

        return scale

    mask_optimizer = torch.optim.SGD(masks, lr=lr, momentum=0.9)
    noise_optimizer = torch.optim.SGD([*noise, *noise_bias], lr=eps)
    handles = [b.bn.register_forward_hook(hook(i)) for i, b in enumerate(blocks)]
    try:
        for _, (x, y, _) in zip(range(iterations), _forever(loader), strict=False):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                for tensor in [*noise, *noise_bias]:
                    tensor.uniform_(-eps, eps)
            perturbed[0] = True
            noise_optimizer.zero_grad()
            ascent = -per_sample_loss(kind, model(x), y).mean()
            ascent.backward()  # type: ignore[no-untyped-call]
            for tensor in [*noise, *noise_bias]:
                assert tensor.grad is not None
                tensor.grad.sign_()
            noise_optimizer.step()
            mask_optimizer.zero_grad()
            robust = per_sample_loss(kind, model(x), y).mean()
            perturbed[0] = False
            natural = per_sample_loss(kind, model(x), y).mean()
            (alpha * natural + (1 - alpha) * robust).backward()  # type: ignore[no-untyped-call]
            mask_optimizer.step()
            with torch.no_grad():
                for mask in masks:
                    mask.clamp_(0, 1)
    finally:
        for handle in handles:
            handle.remove()
    return [mask.detach() for mask in masks]


def _forever(loader: DataLoader[Any]) -> Iterator[Any]:
    """The loader's batches, epoch after epoch."""
    while True:
        yield from loader
