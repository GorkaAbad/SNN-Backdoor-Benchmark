"""Attack budget and the PoisonSet overlay."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel

from sbb.data.sample import EventSample
from sbb.utils.errors import BudgetViolation

# A trigger that acts on the frames [T, P, H, W] of one encoded sample, with the sample's RNG.
FrameTriggerFn = Callable[[torch.Tensor, np.random.Generator], torch.Tensor]


class Budget(BaseModel, frozen=True):
    """Limits of a scenario.

    Attributes:
        poison_rate: Maximum fraction of the train split the attacker may modify.
        target: Target class of the backdoor.
        defender_clean_fraction: Fraction of the train split reserved as the defender pool.
        delta_pp: Allowed clean-accuracy drop, in percentage points, for validity checks.
    """

    poison_rate: float = 0.1
    target: int = 0
    defender_clean_fraction: float = 0.05
    delta_pp: float = 3.0

    def max_poison(self, n_train: int) -> int:
        """Return the maximum number of poisoned samples."""
        return math.floor(self.poison_rate * n_train + 1e-9)


@dataclass(frozen=True)
class PoisonSet:
    """Modified training samples, applied as an overlay on the train split.

    Attributes:
        indices: Full-dataset indices, unique, disjoint from the defender pool.
        samples: The modified samples, in the order of indices.
        labels: Labels used for training.
        frame_trigger: Set by attacks whose trigger acts on frames. The DataModule applies it to the
            cached clean frames of every sample of the overlay, with the RNG rng_for(seed, index),
            so it never reads the samples' events.
    """

    indices: np.ndarray
    samples: tuple[EventSample, ...]
    labels: np.ndarray
    frame_trigger: FrameTriggerFn | None = None

    def by_index(self) -> dict[int, tuple[EventSample, int]]:
        """Map each poisoned index to its (sample, training label)."""
        return {
            int(i): (s, int(y))
            for i, s, y in zip(self.indices, self.samples, self.labels, strict=True)
        }

    def validate(
        self,
        budget: Budget,
        n_train: int,
        original_labels: np.ndarray,
        label_policy: Literal["clean", "dirty"],
        pool: np.ndarray,
    ) -> None:
        """Raise BudgetViolation if the overlay breaks the budget or the label policy.

        Args:
            budget: Scenario budget.
            n_train: Size of the train split.
            original_labels: Labels of the full dataset.
            label_policy: The policy declared in the attack card.
            pool: Full-dataset indices of the defender pool.
        """
        n = len(self.indices)
        if not len(self.samples) == n == len(self.labels):
            raise BudgetViolation("indices, samples and labels differ in length")
        if len(np.unique(self.indices)) != n:
            raise BudgetViolation("duplicate poisoned indices")
        if any(s.index != i for s, i in zip(self.samples, self.indices, strict=True)):
            raise BudgetViolation("a poisoned sample does not carry its own index")
        if n > budget.max_poison(n_train):
            raise BudgetViolation(
                f"{n} poisoned samples exceed the budget of {budget.max_poison(n_train)}"
            )
        if np.isin(self.indices, pool).any():
            raise BudgetViolation("the poison set modifies the defender pool")
        if label_policy == "clean" and not np.array_equal(
            self.labels, original_labels[self.indices]
        ):
            raise BudgetViolation("a clean-label attack changed labels")
