"""Selectors choose which training samples to poison."""

import numpy as np

from sbb.core.plugins import Selector


class NoSelection(Selector):
    """Selects nothing."""

    def select(
        self,
        labels: np.ndarray,
        n: int,
        target: int,
        excluded: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return no positions."""
        return np.empty(0, dtype=np.int64)


class RandomSelector(Selector):
    """Uniformly random samples whose label is not the target."""

    def select(
        self,
        labels: np.ndarray,
        n: int,
        target: int,
        excluded: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return n sorted random positions outside excluded with label != target."""
        candidates = np.setdiff1d(np.flatnonzero(labels != target), excluded)
        return np.sort(rng.choice(candidates, min(n, len(candidates)), replace=False))
