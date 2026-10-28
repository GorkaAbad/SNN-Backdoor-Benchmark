"""Selectors choose which training samples to poison."""

from dataclasses import dataclass

import numpy as np

from sbb.core.plugins import Selector


def _draw(
    candidates: np.ndarray, excluded: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Up to n sorted random positions among candidates, none of them in excluded."""
    candidates = np.setdiff1d(candidates, excluded)
    return np.sort(rng.choice(candidates, min(n, len(candidates)), replace=False))


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


@dataclass(frozen=True)
class RandomSelector(Selector):
    """Uniformly random samples.

    The target class is skipped by default, which is the benchmark rule. With include_target,
    samples come from every class, as in the ported papers, whose replications use it.
    """

    include_target: bool = False

    def select(
        self,
        labels: np.ndarray,
        n: int,
        target: int,
        excluded: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return n sorted random positions outside excluded."""
        candidates = (
            np.arange(len(labels)) if self.include_target else np.flatnonzero(labels != target)
        )
        return _draw(candidates, excluded, n, rng)


class TargetClassSelector(Selector):
    """Uniformly random samples of the target class, for clean-label attacks."""

    def select(
        self,
        labels: np.ndarray,
        n: int,
        target: int,
        excluded: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return n sorted random positions outside excluded with label == target."""
        return _draw(np.flatnonzero(labels == target), excluded, n, rng)


@dataclass(frozen=True)
class ClassSelector(Selector):
    """Uniformly random samples of the given classes, for source-specific attacks (TaCT)."""

    classes: tuple[int, ...]

    def select(
        self,
        labels: np.ndarray,
        n: int,
        target: int,
        excluded: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return n sorted random positions outside excluded whose label is one of classes."""
        return _draw(np.flatnonzero(np.isin(labels, self.classes)), excluded, n, rng)
