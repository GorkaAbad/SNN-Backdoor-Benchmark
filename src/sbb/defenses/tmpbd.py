"""TMPBD (Li et al., 2025): data-free detection from the membrane potentials of the outputs."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import gamma
from spikingjelly.activation_based import layer

from sbb.core.cards import DefenseCard
from sbb.core.plugins import DetectionDefense
from sbb.core.proxies import DefenderView
from sbb.core.registry import DEFENSES
from sbb.models.snn import SNNModel


def class_potentials(model: SNNModel, x: torch.Tensor) -> torch.Tensor:
    """Per class, the output neurons' membrane potential averaged over T and the vote, [B, K].

    A network without voting output neurons ends in a non-spiking readout, which integrates its
    input like a membrane, so its output stands in.
    """
    voting = model.net[-1] if isinstance(model.net, torch.nn.Sequential) else None
    if not isinstance(voting, layer.VotingLayer):
        output: torch.Tensor = model(x)
        return output
    name = list(model.named_neurons())[-1][0]
    with model.probe(name, "membrane", detach=False) as probe:
        model(x)
    return F.avg_pool1d(probe.membrane().mean(1), voting.voting_size, voting.voting_size)


def p_value(margins: np.ndarray) -> float:
    """Probability that the largest margin is no outlier, from a gamma fit to the others.

    A fit that fails, for example on identical margins, gives no evidence, so the p-value is 1.
    """
    others = np.delete(margins, margins.argmax())
    try:
        with np.errstate(all="ignore"):  # a degenerate fit gives nan, handled below
            shape, loc, scale = gamma.fit(others)
        p = 1 - gamma.cdf(margins.max(), shape, loc=loc, scale=scale) ** len(margins)
    except (ValueError, RuntimeError):
        return 1.0
    return float(p) if np.isfinite(p) else 1.0


@DEFENSES.register("tmpbd")
class TMPBD(DetectionDefense):
    """For each class, optimizes random inputs to raise its potential above every other class's.

    A backdoor target is reached by a far larger margin. The verdict is 1 minus the p-value of
    the largest margin under a gamma fit to the other classes' margins.
    """

    card = DefenseCard.load(Path(__file__).parent / "cards" / "tmpbd.yaml")

    def verdict(self, view: DefenderView) -> float:
        """1 - p, higher for a likely backdoor."""
        model = view.model.to(view.trainer.device).eval()
        for _, node in model.named_neurons():
            node.train()  # eval-mode SpikingJelly neurons have no surrogate gradient
        model.requires_grad_(False)
        classes = view.scenario.num_classes
        shape = (self.hparams["inputs"], view.scenario.T, *view.scenario.input_shape)
        device = view.trainer.device
        margins = [self._margin(model, k, classes, shape, device) for k in range(classes)]
        return 1.0 - p_value(np.array(margins))

    def _margin(
        self, model: SNNModel, k: int, classes: int, shape: tuple[int, ...], device: torch.device
    ) -> float:
        """The best margin of class k over the other classes after the optimization."""
        x = torch.rand(shape, device=device, requires_grad=True)
        other = torch.ones(classes, dtype=torch.bool, device=device)
        other[k] = False
        for _ in range(self.hparams["steps"]):
            potentials = class_potentials(model, x.clamp(0, 10))
            margin = potentials[:, k] - potentials[:, other].max(1).values
            x.grad = None
            (-margin.sum()).backward()  # type: ignore[no-untyped-call]
            with torch.no_grad():
                x -= self.hparams["lr"] * x.grad
        return float(margin.detach().max())
