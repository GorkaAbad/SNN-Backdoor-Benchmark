"""NDSBM, neural dendrites suppression backdoor mitigation (Li et al., arXiv 2510.06629)."""

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.hooks import RemovableHandle

from sbb.core.cards import DefenseCard
from sbb.core.plugins import ModelDefense
from sbb.core.proxies import DefenderView
from sbb.core.registry import DEFENSES
from sbb.defenses.utils import conv_blocks
from sbb.models.snn import SNNModel


@DEFENSES.register("ndsbm")
class NDSBM(ModelDefense):
    """Clamps the input current of each conv block's neurons between a learned floor and ceiling."""

    card = DefenseCard.load(Path(__file__).parent / "cards" / "ndsbm.yaml")

    def __init__(self, hparams: dict[str, Any]) -> None:
        super().__init__(hparams)
        self._clamps: Clamps | None = None

    def repair(self, view: DefenderView) -> SNNModel:
        """Learn the clamps on clean data, keeping the last layer's membrane potentials unchanged.

        The loss is the MSE between the mean over T of the last spiking layer's membrane potential
        with and without the clamps, plus c times the norms of the floors and ceilings, which
        narrows them. As in the official code the model runs in train mode, because SpikingJelly
        neurons pass surrogate gradients only there, so its batch-norm statistics follow the
        clean data. Its weights stay as they are.
        """
        h, device = self.hparams, view.trainer.device
        model = view.model.to(device).train()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        clamps = self.extra(model).to(device)
        last = list(model.named_neurons())[-1][0]
        optimizer = torch.optim.Adam(clamps.parameters(), lr=h["lr"])
        handles = clamps.attach(model)
        try:
            for _ in range(h["epochs"]):
                for x, _, _ in view.clean_loader(h["batch_size"]):  # the labels are not used
                    x = x.to(device)
                    clamps.enabled = False
                    with torch.no_grad(), model.probe(last, "membrane") as free:
                        model(x)
                    clamps.enabled = True
                    with model.probe(last, "membrane", detach=False) as clamped:
                        model(x)
                    loss = F.mse_loss(clamped.membrane().mean(1), free.membrane().mean(1))
                    norms = sum(p.norm() for p in clamps.parameters())
                    optimizer.zero_grad()
                    (loss + h["c"] * norms).backward()
                    optimizer.step()
        finally:
            for handle in handles:
                handle.remove()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return model

    def extra(self, model: SNNModel) -> "Clamps":
        """The clamps, sized for model, learned by repair or loaded by the Game."""
        if self._clamps is None:
            self._clamps = Clamps(model, self.hparams["init"])
        return self._clamps

    def attach(self, model: SNNModel, extra: nn.Module) -> None:
        """Clamp the model loaded back from the checkpoint."""
        assert isinstance(extra, Clamps)
        extra.attach(model)


class Clamps(nn.Module):
    """A floor and a ceiling per time step and channel on each conv block's batch-norm output.

    The batch norm's output is the input current of the block's neurons. The bounds start at
    1 - init and 1 + init, as in the official code, wide enough not to bind.
    """

    def __init__(self, model: SNNModel, init: float) -> None:
        super().__init__()
        channels = [block.bn.num_features for block in conv_blocks(model)]
        shape = [(model.spec.T, c) for c in channels]
        self.upper = nn.ParameterList([nn.Parameter(torch.full(s, 1.0 + init)) for s in shape])
        self.lower = nn.ParameterList([nn.Parameter(torch.full(s, 1.0 - init)) for s in shape])
        self.enabled = True  # repair turns them off for the reference pass

    def attach(self, model: SNNModel) -> list[RemovableHandle]:
        """Clamp the batch norms of model's conv blocks. Remove the handles to undo it."""
        blocks = conv_blocks(model)
        return [block.bn.register_forward_hook(self._clamp(i)) for i, block in enumerate(blocks)]

    def _clamp(self, i: int) -> Any:
        def clamp(_bn: nn.Module, _inputs: object, out: torch.Tensor) -> torch.Tensor:
            if not self.enabled:
                return out
            shape = (out.shape[0], 1, -1) + (1,) * (out.dim() - 3)  # out is [T, B, C, H, W]
            lower, upper = self.lower[i].view(shape), self.upper[i].view(shape)
            return torch.clamp(out, min=lower, max=upper)

        return clamp
