"""CLP, channel Lipschitzness-based pruning (Zheng et al., ECCV 2022). Data-free."""

from pathlib import Path

import torch
from torch import nn

from sbb.core.cards import DefenseCard
from sbb.core.plugins import ModelDefense
from sbb.core.proxies import DefenderView
from sbb.core.registry import DEFENSES
from sbb.defenses.utils import conv_blocks, zero_channels
from sbb.models.snn import SNNModel
from sbb.utils.errors import SbbError


def channel_lipschitz(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> torch.Tensor:
    """Upper bound of each output channel's Lipschitz constant (UCLC), shape [C].

    The largest singular value of the channel's kernel, as a [C_in, k*k] matrix, times
    |gamma / sigma| of its batch norm, with sigma the square root of the running variance, as in
    the official code.
    """
    if bn.running_var is None:
        raise SbbError("CLP needs batch norms with running statistics")
    scale = (bn.weight / bn.running_var.sqrt()).abs()
    bound: torch.Tensor = torch.linalg.matrix_norm(conv.weight.flatten(2) * scale[:, None, None], 2)
    return bound


@DEFENSES.register("clp")
class CLP(ModelDefense):
    """Prunes, in every conv block, the channels whose UCLC exceeds mean + u * std of the layer."""

    card = DefenseCard.load(Path(__file__).parent / "cards" / "clp.yaml")

    def repair(self, view: DefenderView) -> SNNModel:
        """Zero the batch norm of the outlier channels."""
        model = view.model
        with torch.no_grad():
            for block in conv_blocks(model):
                lipschitz = channel_lipschitz(block.conv, block.bn)
                limit = lipschitz.mean() + self.hparams["u"] * lipschitz.std()
                zero_channels(block.bn, torch.nonzero(lipschitz > limit).flatten())
        return model
