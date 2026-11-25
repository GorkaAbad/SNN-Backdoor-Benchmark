"""STRIP with the entropy taken at every time step, an sbb variant of STRIP."""

from pathlib import Path

import torch

from sbb.core.cards import DefenseCard
from sbb.core.registry import DEFENSES
from sbb.defenses.strip import STRIP


@DEFENSES.register("strip_per_step")
class STRIPPerStep(STRIP):
    """STRIP whose entropy is the mean over time steps of the entropy of each step's output."""

    card = DefenseCard.load(Path(__file__).parent / "cards" / "strip_per_step.yaml")

    def entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Mean over time steps of the entropy of the softmax of each step, one value per input."""
        p = torch.softmax(self.model.steps(x), dim=2)
        return -(p * torch.log(p.clamp_min(1e-12))).sum(2).mean(0)
