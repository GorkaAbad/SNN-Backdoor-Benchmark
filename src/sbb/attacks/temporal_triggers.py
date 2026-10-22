"""Triggers that only reorder the frames of a sample (T-Backdoor: rate, latency and jitter).

The paper defines each trigger as a remapping sigma of the frame indices, x_hat(t) = x(sigma(t)).
They act on frames because the paper and its code remap SpikingJelly frames. The official code has
no license, so this module re-implements the paper's equations. The parity with the code's
behavior was checked separately.
"""

from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from sbb.core.plugins import FrameTrigger
from sbb.utils.errors import SbbError


class _Remap(FrameTrigger):
    """A trigger that shows frame order(T)[t] at time t."""

    @abstractmethod
    def order(self, T: int) -> np.ndarray:
        """Sigma as an array of T frame indices."""

    def apply(self, frames: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Reorder the frames."""
        return frames[torch.from_numpy(self.order(len(frames)))]


@dataclass(frozen=True)
class RateTrigger(_Remap):
    """Plays the sample at another speed: sigma(t) = min(floor(scale * t), T - 1).

    Attributes:
        scale: Speed factor r. Below 1 the sample slows down and its end is cut off.
    """

    scale: float

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise SbbError(f"a rate trigger needs a positive scale, not {self.scale}")

    def order(self, T: int) -> np.ndarray:
        """Indices of the rescaled sample."""
        order: np.ndarray = np.clip(np.arange(T) * self.scale, 0, T - 1).astype(np.int64)
        return order


@dataclass(frozen=True)
class LatencyTrigger(_Remap):
    """Delays the sample and repeats its first frame: sigma(t) = max(t - delay, 0).

    Attributes:
        delay: Delay d in frames.
    """

    delay: int

    def __post_init__(self) -> None:
        if self.delay < 0:
            raise SbbError(f"a latency trigger needs a delay of at least 0, not {self.delay}")

    def order(self, T: int) -> np.ndarray:
        """Indices of the delayed sample."""
        return np.maximum(np.arange(T) - self.delay, 0)


@dataclass(frozen=True)
class JitterTrigger(_Remap):
    """Swaps pairs of frames (2ks, 2ks + s) for k < pairs, with s = max(1, T // (2 pairs)).

    A pair that would reach past the last frame is skipped.

    Attributes:
        pairs: Number of swapped pairs n_s.
    """

    pairs: int

    def __post_init__(self) -> None:
        if self.pairs < 1:
            raise SbbError(f"a jitter trigger swaps at least one pair, not {self.pairs}")

    def order(self, T: int) -> np.ndarray:
        """Indices with the pairs swapped."""
        order = np.arange(T)
        step = max(1, T // (2 * self.pairs))
        for k in range(self.pairs):
            i, j = 2 * k * step, 2 * k * step + step
            if j < T:
                order[[i, j]] = order[[j, i]]
        return order
