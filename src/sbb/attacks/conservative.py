"""Poisoning with payload and cover samples, the shape of the adaptive attacks."""

import numpy as np

from sbb.core.plugins import FrameTrigger, PoisoningAttack, Selector
from sbb.core.proxies import AttackerView
from sbb.data.poison import Budget, PoisonSet
from sbb.data.sample import EVENT_DTYPE, EventSample


class ConservativePoisoning(PoisoningAttack):
    """Poisons payload samples, relabelled as the target, and cover samples, which keep their label.

    Of the n samples the budget allows, round(eta * n) are cover samples (the conservatism ratio of
    Qi et al., ICLR 2023, the cover images of TaCT) and the rest payload samples. selector draws the
    payload and cover_selector the cover, outside the payload. Every one gets train_trigger, which
    may be weaker than self.trigger, the test trigger (the asymmetric triggers of Qi et al.). Both
    are frame triggers, so the DataModule applies train_trigger to the cached frames.
    """

    train_trigger: FrameTrigger
    cover_selector: Selector
    eta: float

    def poison(self, view: AttackerView, budget: Budget, pool: np.ndarray) -> PoisonSet:
        """Select the payload and cover samples outside the defender pool and label them."""
        train = view.train_set
        labels = train.labels()
        excluded = np.flatnonzero(np.isin(train.indices, pool))
        n = budget.max_poison(len(train))
        n_cover = round(self.eta * n)
        payload = self.selector.select(labels, n - n_cover, self.target, excluded, view.rng)
        cover = self.cover_selector.select(
            labels, n_cover, self.target, np.union1d(excluded, payload), view.rng
        )
        positions = np.sort(np.concatenate([payload, cover]))
        indices = train.indices[positions]
        new_labels = np.where(np.isin(positions, payload), self.target, labels[positions])
        samples = tuple(
            EventSample(np.empty(0, EVENT_DTYPE), int(labels[p]), int(i))
            for p, i in zip(positions, indices, strict=True)
        )
        return PoisonSet(
            indices, samples, new_labels.astype(np.int64), frame_trigger=self.train_trigger.apply
        )
