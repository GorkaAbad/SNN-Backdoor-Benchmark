"""The learned trigger of the Sneaky Spikes dynamic attack (MIT, dynamic.py and models.py)."""

import numpy as np
import torch
from spikingjelly.activation_based import functional, layer, neuron, surrogate
from torch import nn

from sbb.core.plugins import FrameTrigger
from sbb.utils.errors import SbbError


def spiking_autoencoder(channels: int, polarities: int = 2) -> nn.Sequential:
    """The official Autoencoder: four stride-2 conv blocks down, four transposed conv blocks up.

    Every block is [conv, batch norm, IF neurons], the last one has no batch norm, and the neurons
    use the ATan surrogate and a detached reset. A frame side must be a multiple of 16.
    """

    def block(conv: nn.Module, out: int, norm: bool = True) -> list[nn.Module]:
        node = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        return [conv, layer.BatchNorm2d(out), node] if norm else [conv, node]

    widths = [polarities, channels, 2 * channels, 4 * channels, 8 * channels]
    layers: list[nn.Module] = []
    for c_in, c_out in zip(widths, widths[1:], strict=False):
        layers += block(layer.Conv2d(c_in, c_out, 4, stride=2, padding=1, bias=False), c_out)
    up = widths[::-1]
    for c_in, c_out in zip(up, up[1:], strict=False):
        conv = layer.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1, bias=False)
        layers += block(conv, c_out, norm=c_out != polarities)
    net = nn.Sequential(*layers)
    functional.set_step_mode(net, "m")
    return net


def perturb(generator: nn.Module, x: torch.Tensor, gamma: float) -> torch.Tensor:
    """Return x + gamma * tanh(G(x)) for a batch x [B, T, P, H, W], the official clip_image."""
    if x.shape[-1] % 16 or x.shape[-2] % 16:
        side = tuple(x.shape[-2:])
        raise SbbError(f"the dynamic trigger needs frame sides a multiple of 16, got {side}")
    functional.reset_net(generator)
    noise: torch.Tensor = torch.tanh(generator(x.transpose(0, 1))).transpose(0, 1)
    return x + gamma * noise


class DynamicTrigger(FrameTrigger):
    """Adds gamma * tanh(G(x)) to the frames x, with G a spiking autoencoder.

    The attack learns G jointly with the model, so every sample gets its own trigger. G ends in
    spiking neurons, so each entry moves by 0 or gamma * tanh(1).

    Args:
        gamma: Scale of the perturbation, the official beta.
        channels: Width of the autoencoder's first block (128 in the official code).
        polarities: Polarity channels of the frames.
    """

    def __init__(self, gamma: float, channels: int = 128, polarities: int = 2) -> None:
        self.gamma = gamma
        self.generator = spiking_autoencoder(channels, polarities).eval()

    def learned(self) -> nn.Module:
        """The generator."""
        return self.generator

    @torch.no_grad()
    def apply(self, frames: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Trigger one sample [T, P, H, W] with the generator in eval mode."""
        self.generator.eval()
        return perturb(self.generator, frames.unsqueeze(0), self.gamma).squeeze(0)
