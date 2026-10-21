"""STRIP (Gao et al., ACSAC 2019), adapted to SNNs."""

from pathlib import Path

import torch

from sbb.core.cards import DefenseCard
from sbb.core.plugins import InputDefense
from sbb.core.proxies import DefenderView
from sbb.core.registry import DEFENSES
from sbb.defenses.utils import superpose
from sbb.models.snn import SNNModel


@DEFENSES.register("strip")
class STRIP(InputDefense):
    """Rejects inputs whose predictions stay certain when clean inputs are superposed on them.

    The score is minus the mean entropy of the softmax of the output over the overlays, so low
    entropy is suspicious. The threshold keeps a fraction frr of the clean subset rejected.
    """

    card = DefenseCard.load(Path(__file__).parent / "cards" / "strip.yaml")
    model: SNNModel
    overlays: torch.Tensor

    def fit(self, view: DefenderView) -> None:
        """Draw the overlays from the clean subset and set the threshold on its own scores."""
        batch_size = self.hparams["batch_size"]
        clean = torch.cat([x for x, _, _ in view.clean_loader(batch_size)])
        n = min(self.hparams["overlays"], len(clean))
        # ponytail: one set of overlays for every input, the paper draws them per input.
        self.overlays = clean[torch.from_numpy(view.rng.choice(len(clean), n, replace=False))]
        self.model = view.model.to(view.trainer.device).eval()
        with torch.no_grad():
            batches = torch.split(clean, batch_size)
            scores = [self.score(x.to(view.trainer.device)) for x in batches]
        self.threshold = float(torch.quantile(torch.cat(scores), 1 - self.hparams["frr"]))

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Minus the mean prediction entropy over the overlays, one score per input."""
        entropy = torch.zeros(len(x), device=x.device)
        for overlay in self.overlays.to(x.device):
            p = torch.softmax(self.model(superpose(x, overlay)), dim=1)
            entropy -= (p * torch.log(p.clamp_min(1e-12))).sum(1)
        return -entropy / len(self.overlays)
