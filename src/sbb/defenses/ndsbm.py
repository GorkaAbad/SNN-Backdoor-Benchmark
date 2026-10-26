"""NDSBM, neural dendrites suppression backdoor mitigation (Li et al., arXiv 2510.06629)."""

from collections import Counter
from collections.abc import Callable, Iterator
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
        narrows them. The bounds are learned for the forward pass the Game evaluates: batch norm
        with its running statistics and no dropout, so the model runs in eval mode, except its
        spiking neurons, which pass surrogate gradients only in train mode. The model's weights
        and statistics stay as they are.
        """
        h, device = self.hparams, view.trainer.device
        model = view.model.to(device).eval()
        for _, neuron in model.named_neurons():
            neuron.train()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        clamps = self.extra(model).to(device)
        last = list(model.named_neurons())[-1][0]
        optimizer = torch.optim.Adam(clamps.parameters(), lr=h["lr"])
        batches = _clean_batches(view, h["batch_size"], h.get("samples_per_class"))
        handles = clamps.attach(model)
        try:
            for _ in range(h["epochs"]):
                for x in batches():
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
        return model.eval()

    def extra(self, model: SNNModel) -> "Clamps":
        """The clamps, sized for model, learned by repair or loaded by the Game."""
        if self._clamps is None:
            self._clamps = Clamps(model, self.hparams["init"])
        return self._clamps

    def attach(self, model: SNNModel, extra: nn.Module) -> None:
        """Clamp the model loaded back from the checkpoint."""
        assert isinstance(extra, Clamps)
        extra.attach(model)


def _clean_batches(
    view: DefenderView, batch_size: int, per_class: int | None
) -> Callable[[], Iterator[torch.Tensor]]:
    """One epoch of clean input batches. The labels only pick the samples, the loss ignores them.

    With per_class, the first per_class samples of each class in the clean subset, as the official
    nsample, kept in memory and reshuffled every epoch. Adam moves the bounds by about lr per step
    whatever c is, so the number of steps, set by the number of samples, decides how far they
    shrink. None uses the whole clean subset.
    """
    if per_class is None:
        return lambda: (x for x, _, _ in view.clean_loader(batch_size))
    kept: list[torch.Tensor] = []
    counts: Counter[int] = Counter()
    for x, y, _ in view.clean_loader(batch_size):
        for sample, label in zip(x, y.tolist(), strict=True):
            if counts[label] < per_class:
                counts[label] += 1
                kept.append(sample)
    inputs = torch.stack(kept)
    generator = torch.Generator().manual_seed(int(view.rng.integers(2**31)))

    def epoch() -> Iterator[torch.Tensor]:
        order = torch.randperm(len(inputs), generator=generator)
        return (inputs[order[i : i + batch_size]] for i in range(0, len(order), batch_size))

    return epoch


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
