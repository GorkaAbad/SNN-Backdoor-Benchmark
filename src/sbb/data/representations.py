"""Representations turn raw events into model input."""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel

from sbb.data.sample import EventSample
from sbb.utils.errors import SbbError


class ReprSpec(BaseModel, frozen=True):
    """Configuration of a representation."""

    kind: Literal["frames"] = "frames"
    T: int
    split_by: Literal["number", "time"] = "number"
    downsample: int = 1


class Representation(ABC):
    """Encodes one sample to a [T, P, H, W] float32 tensor."""

    def __init__(self, spec: ReprSpec) -> None:
        self.spec = spec

    @abstractmethod
    def encode(self, sample: EventSample, sensor_size: tuple[int, int, int]) -> torch.Tensor:
        """Encode sample. sensor_size is (W, H, P)."""


class FrameRepr(Representation):
    """Event counts per pixel and polarity in T frames.

    Frame boundaries follow SpikingJelly's integrate_events_by_fixed_frames_number:
    "number" gives every frame N // T events and the last frame the remainder,
    "time" gives every frame (t_last - t_first) // T microseconds and the last frame the remainder.
    """

    def encode(self, sample: EventSample, sensor_size: tuple[int, int, int]) -> torch.Tensor:
        """Count events per (frame, polarity, y, x) after integer downsampling."""
        W, H, P = sensor_size
        T, d = self.spec.T, self.spec.downsample
        w, h = -(-W // d), -(-H // d)
        ev = sample.events
        n = len(ev)
        if n == 0:
            return torch.zeros(T, P, h, w)
        if min(ev["x"].min(), ev["y"].min()) < 0 or ev["x"].max() >= W or ev["y"].max() >= H:
            raise SbbError(f"sample {sample.index} has events outside the {W}x{H} sensor")
        if self.spec.split_by == "number":
            per_frame = n // T
            frame = np.minimum(np.arange(n) // per_frame, T - 1) if per_frame else np.full(n, T - 1)
        else:
            t = ev["t"] - ev["t"][0]
            frame = np.minimum(t // max(int(t[-1]) // T, 1), T - 1)
        flat = ((frame * P + ev["p"]) * h + ev["y"] // d) * w + ev["x"] // d
        counts = np.bincount(flat, minlength=T * P * h * w).reshape(T, P, h, w)
        return torch.from_numpy(counts.astype(np.float32))
