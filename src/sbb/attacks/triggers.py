"""Triggers on raw events."""

from dataclasses import dataclass

import numpy as np

from sbb.core.plugins import Trigger
from sbb.data.sample import EventSample, make_events


class IdentityTrigger(Trigger):
    """Leaves the sample unchanged."""

    def apply(self, sample: EventSample, rng: np.random.Generator) -> EventSample:
        """Return sample."""
        return sample


@dataclass(frozen=True)
class PatchTrigger(Trigger):
    """A static square of events, repeated evenly over the duration of the sample.

    Attributes:
        size: Side of the square in pixels.
        location: (x, y) of the top-left corner of the square.
        polarity: Polarity of the trigger events, 0 or 1.
        events_per_pixel: Number of events each pixel of the square emits.
    """

    size: int
    location: tuple[int, int]
    polarity: int
    events_per_pixel: int

    def apply(self, sample: EventSample, rng: np.random.Generator) -> EventSample:
        """Add the patch events between the first and the last event of the sample."""
        ev = sample.events
        t_first, t_last = (int(ev["t"][0]), int(ev["t"][-1])) if len(ev) else (0, 0)
        x0, y0 = self.location
        xs, ys = np.meshgrid(np.arange(x0, x0 + self.size), np.arange(y0, y0 + self.size))
        times = np.linspace(t_first, t_last, self.events_per_pixel).astype(np.int64)
        k = self.events_per_pixel
        patch = make_events(
            np.repeat(xs.ravel(), k),
            np.repeat(ys.ravel(), k),
            np.tile(times, xs.size),
            self.polarity,
        )
        merged = np.concatenate([ev, patch])
        return EventSample(
            merged[np.argsort(merged["t"], kind="stable")], sample.label, sample.index
        )
