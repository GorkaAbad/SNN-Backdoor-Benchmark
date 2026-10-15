"""Synthetic events for tests and smoke runs. No download."""

from pathlib import Path

import numpy as np

from sbb.data.sample import DATASETS, EventDataset, EventSample, make_events
from sbb.utils.determinism import rng_for

_SEED = 1234
_CENTERS = ((5, 5), (10, 5), (5, 10), (10, 10))


@DATASETS.register("synthetic")
class SyntheticEvents(EventDataset):
    """One Gaussian blob of events per class on a 16x16 sensor, away from the corners.

    Args:
        root: Unused, kept for the common constructor.
        size: Number of samples. The split file synthetic_v1 expects 600.
        events_per_sample: Number of events in each sample.
    """

    name = "synthetic"
    sensor_size = (16, 16, 2)
    num_classes = 4

    def __init__(self, root: Path, size: int = 600, events_per_sample: int = 300) -> None:
        super().__init__(root)
        self.size = size
        self.events_per_sample = events_per_sample

    def prepare(self) -> None:
        """Nothing to download."""

    def __len__(self) -> int:
        return self.size

    def labels(self) -> np.ndarray:
        """Labels cycle through the classes."""
        return np.arange(self.size) % self.num_classes

    def __getitem__(self, index: int) -> EventSample:
        if not 0 <= index < self.size:
            raise IndexError(index)
        label = index % self.num_classes
        rng = rng_for(_SEED, index)
        n = self.events_per_sample
        cx, cy = _CENTERS[label]
        x = np.clip(np.rint(rng.normal(cx, 1.0, n)), 0, self.sensor_size[0] - 1)
        y = np.clip(np.rint(rng.normal(cy, 1.0, n)), 0, self.sensor_size[1] - 1)
        t = rng.integers(0, 100_000, n)
        p = rng.integers(0, 2, n)
        return EventSample(make_events(x, y, t, p), label, index)
