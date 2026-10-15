"""Raw event samples and the dataset interface (docs/PLAN.md section 6.2)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike

from sbb.utils.registry import Registry

EVENT_DTYPE = np.dtype([("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", np.int8)])


def make_events(x: ArrayLike, y: ArrayLike, t: ArrayLike, p: ArrayLike) -> np.ndarray:
    """Build a structured event array with EVENT_DTYPE, sorted by t."""
    t = np.asarray(t)
    events = np.empty(len(t), dtype=EVENT_DTYPE)
    events["x"], events["y"], events["t"], events["p"] = x, y, t, p
    return events[np.argsort(events["t"], kind="stable")]


@dataclass(frozen=True)
class EventSample:
    """One recording.

    Attributes:
        events: Structured array with EVENT_DTYPE, sorted by t, t in microseconds.
        label: Class label.
        index: Index in the full dataset.
    """

    events: np.ndarray
    label: int
    index: int


class EventDataset(ABC):
    """A neuromorphic dataset that yields raw EventSamples by full-dataset index."""

    name: ClassVar[str]
    sensor_size: ClassVar[tuple[int, int, int]]  # (W, H, P)
    num_classes: ClassVar[int]

    def __init__(self, root: Path) -> None:
        self.root = root

    @abstractmethod
    def prepare(self) -> None:
        """Download and verify the data, or stop with manual-download instructions."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> EventSample: ...

    @abstractmethod
    def labels(self) -> np.ndarray:
        """Return all labels without loading events."""


DATASETS: Registry[type[EventDataset]] = Registry("dataset")
