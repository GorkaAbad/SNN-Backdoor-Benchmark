"""Triggers written into the frames of an encoded sample (trigger_domain frames).

Ports of the Sneaky Spikes triggers (static, moving and smart) and the Flashy Backdoor trigger
(flash), both MIT licensed. The official code writes into SpikingJelly frames [T, P, H, W] of event
counts. A polarity mode sets both channels of every pixel the trigger covers:

- 0 sets both channels to 0.
- 1 sets channel 0 (OFF events) to 0 and channel 1 (ON events) to 1.
- 2 sets channel 0 to 1 and channel 1 to 0.
- 3 sets both channels to 1.

A trigger is a square whose side is a fraction of the frame width, at least one pixel. The
official code swaps the width and the height in places. That changes nothing for square frames,
and every ported setting uses square frames.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from sbb.core.plugins import FrameTrigger
from sbb.core.proxies import AttackerView
from sbb.utils.errors import SbbError

Position = Literal["top-left", "top-right", "bottom-left", "bottom-right", "middle"]

_STEP = 2  # pixels the smart trigger moves per frame on each axis, as in the official code
_BATCH = 32  # batch size of the pass over the train split in SmartPatchTrigger.fit


def _paint(
    frames: torch.Tensor, t: int | slice | list[int], ys: slice, xs: slice, mode: int
) -> None:
    """Set both channels of frames[t, :, ys, xs] by the polarity mode, in place."""
    if mode not in range(4):
        raise SbbError(f"the polarity mode must be 0, 1, 2 or 3, got {mode}")
    if frames.shape[1] != 2:
        raise SbbError(f"frame triggers need two polarity channels, got {frames.shape[1]}")
    frames[t, 0, ys, xs] = mode >> 1
    frames[t, 1, ys, xs] = mode & 1


def _side(size: float, width: int) -> int:
    return max(int(size * width), 1)


def _corner(position: str, height: int, width: int, side: int) -> tuple[int, int]:
    """Top-left (y, x) of a square at position, where the official static trigger places it."""
    corners = {
        "top-left": (0, 0),
        "top-right": (0, width - side),
        "bottom-left": (height - side, 0),
        "bottom-right": (height - side, width - side),
        "middle": ((height - side) // 2, (width - side) // 2),
    }
    if position not in corners:
        raise SbbError(f"unknown trigger position {position!r}, use one of {sorted(corners)}")
    return corners[position]


def _square(y: int, x: int, side: int, frames: torch.Tensor) -> tuple[slice, slice]:
    """Slices of the square with top-left (y, x). A square that leaves the frame is refused."""
    height, width = frames.shape[-2:]
    if y < 0 or x < 0 or y + side > height or x + side > width:
        raise SbbError(
            f"a {side}-pixel trigger at ({y}, {x}) does not fit a {height}x{width} frame"
        )
    return slice(y, y + side), slice(x, x + side)


@dataclass(frozen=True)
class FramePatchTrigger(FrameTrigger):
    """A static square in every frame (Sneaky Spikes, static).

    Attributes:
        size: Side of the square as a fraction of the frame width.
        position: A corner of the frame, or its middle.
        polarity: Polarity mode, see the module docstring.
    """

    size: float
    position: Position
    polarity: int

    def apply(self, frames: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Paint the square into every frame."""
        height, width = frames.shape[-2:]
        side = _side(self.size, width)
        y, x = _corner(self.position, height, width, side)
        out = frames.clone()
        _paint(out, slice(None), *_square(y, x, side, frames), self.polarity)
        return out


@dataclass(frozen=True)
class MovingPatchTrigger(FrameTrigger):
    """A square that jumps between three places in a row (Sneaky Spikes, moving).

    From the corner of position, the places are x0, x0 + side + 2 and x0 + 2 side + 2, towards the
    inside of the frame. Frame t shows the square at place t mod 3. The middle position starts at
    (H // 2 - 2, W // 2 - 2) and moves right. As in the official code, the last frame stays clean.

    Attributes:
        size: Side of the square as a fraction of the frame width.
        position: A corner of the frame, or its middle.
        polarity: Polarity mode, see the module docstring.
    """

    size: float
    position: Position
    polarity: int

    def apply(self, frames: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Paint the square at its place in every frame but the last."""
        height, width = frames.shape[-2:]
        side = _side(self.size, width)
        if self.position == "middle":
            y, x = height // 2 - 2, width // 2 - 2
        else:
            y, x = _corner(self.position, height, width, side)
        direction = -1 if self.position.endswith("right") else 1
        offsets = (0, side + 2, 2 * side + 2)
        places = [_square(y, x + direction * offset, side, frames) for offset in offsets]
        out = frames.clone()
        for t in range(len(frames) - 1):
            _paint(out, t, *places[t % 3], self.polarity)
        return out


@dataclass(eq=False)
class SmartPatchTrigger(FrameTrigger):
    """A square that wanders inside the most active cell of a grid (Sneaky Spikes, smart).

    fit splits the frame into grid x grid cells, as the official get_masks does, and counts the four
    polarity cases of _mode_counts in each cell over the clean train frames. It picks the most
    active cell and the polarity mode whose case is rarest there, so the trigger contrasts with its
    background. Then it draws a random walk: a start inside the cell, at least one side away from
    its top and left borders, and per frame a step of 2 pixels on each axis, in a random direction,
    repeated until the square is back inside. Every poisoned and every test sample gets this walk.
    It comes from a child of view.rng, so it depends only on the seed and not on what the selector
    drew before, as in the official code, where it does not change with the poison rate.

    The official code draws a second walk, and picks the cell again from the test frames, for the
    triggered test set. Here the trigger is fitted once, on the train split an attacker sees, and
    the test set gets the same trigger as the poisoned samples.

    Attributes:
        size: Side of the square as a fraction of the frame width.
        grid: Cells per side. The official n_masks is grid - 1.
        least: Pick the least active cell instead.
        most_polarity: Pick the mode of the most frequent case instead.
        mask: The chosen cell (y0, y1, x0, x1), set by fit.
        polarity: The chosen polarity mode, set by fit.
        walk: Top-left (y, x) of the square in each frame, shape (T, 2), set by fit.
    """

    size: float
    grid: int = 3
    least: bool = False
    most_polarity: bool = False
    mask: tuple[int, int, int, int] = field(default=(0, 0, 0, 0), init=False)
    polarity: int = field(default=-1, init=False)
    walk: np.ndarray | None = field(default=None, init=False)

    def fit(self, view: AttackerView) -> None:
        """Choose the cell and the polarity mode from the clean train frames, then draw the walk."""
        _, height, width = view.scenario.input_shape
        rows, cols = _edges(height, self.grid), _edges(width, self.grid)
        counts = np.zeros((self.grid**2, 4), dtype=np.int64)
        for x, _, _ in view.train_set.loader(_BATCH):
            counts += _mode_counts(x, rows, cols)
        activity = counts[:, 1:].sum(axis=1)
        cell = int(np.argmin(activity) if self.least else np.argmax(activity))
        self.polarity = int(
            np.argmax(counts[cell]) if self.most_polarity else np.argmin(counts[cell])
        )
        r, c = divmod(cell, self.grid)
        self.mask = (rows[r], rows[r + 1], cols[c], cols[c + 1])
        walk_rng = view.rng.spawn(1)[0]
        self.walk = _walk(self.mask, _side(self.size, width), view.scenario.T, walk_rng)

    def apply(self, frames: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Paint the square at its place on the walk in every frame."""
        if self.walk is None:
            raise SbbError("fit the smart trigger before applying it")
        if len(self.walk) != len(frames):
            raise SbbError(f"the walk has {len(self.walk)} frames, the sample {len(frames)}")
        side = _side(self.size, frames.shape[-1])
        out = frames.clone()
        for t, (y, x) in enumerate(self.walk.tolist()):
            _paint(out, t, *_square(y, x, side, frames), self.polarity)
        return out


def _edges(n: int, cells: int) -> list[int]:
    """Borders of the cells along one side, as get_masks computes them: summed, then truncated."""
    step, edge, edges = n / cells, 0.0, [0]
    for _ in range(cells):
        edge += step
        edges.append(int(edge))
    return edges


def _mode_counts(x: torch.Tensor, rows: list[int], cols: list[int]) -> np.ndarray:
    """Count the four polarity cases in each grid cell of a batch [B, T, 2, H, W].

    The cases are those of the official smart trigger, one per polarity mode: entries equal to 0,
    pixels whose (channel 0, channel 1) is (0, 1), pixels whose pair is (1, 0), and entries equal
    to 1. Cells are ordered row by row.
    """
    counts = []
    for r in range(len(rows) - 1):
        for c in range(len(cols) - 1):
            cell = x[..., rows[r] : rows[r + 1], cols[c] : cols[c + 1]]
            off, on = cell[:, :, 0], cell[:, :, 1]
            counts.append(
                [
                    int((cell == 0).sum()),
                    int(((off == 0) & (on == 1)).sum()),
                    int(((off == 1) & (on == 0)).sum()),
                    int((cell == 1).sum()),
                ]
            )
    return np.array(counts, dtype=np.int64)


def _walk(
    mask: tuple[int, int, int, int], side: int, T: int, rng: np.random.Generator
) -> np.ndarray:
    """The random walk of the smart trigger inside mask, one top-left (y, x) per frame."""
    y0, y1, x0, x1 = mask
    if y0 + side >= y1 - side or x0 + side >= x1 - side:
        raise SbbError(f"a {side}-pixel smart trigger does not fit a {y1 - y0}x{x1 - x0} cell")
    y = int(rng.integers(y0 + side, y1 - side))
    x = int(rng.integers(x0 + side, x1 - side))
    walk = []
    for _ in range(T):  # x moves first, as in the official code
        x = _step(x, x0 + side, x1 - side, rng)
        y = _step(y, y0 + side, y1 - side, rng)
        walk.append((y, x))
    return np.array(walk, dtype=np.int64)


def _step(v: int, low: int, high: int, rng: np.random.Generator) -> int:
    """Move v by 2 in a random direction, again and again, until it lies in [low, high]."""
    while True:
        v += _STEP if rng.integers(2) == 0 else -_STEP
        if low <= v <= high:
            return v


@dataclass(frozen=True)
class FlashTrigger(FrameTrigger):
    """Whole frames set to one polarity mode, possibly as a strobe (Flashy Backdoor).

    Flashes start at frame start and then every gap + duration frames, while they start within
    length frames of start. Each lasts duration frames, and frames at or after end stay clean. As
    in the official get_attack_frames, the last flash may run past length.

    Attributes:
        polarity: Polarity mode, see the module docstring.
        start: First flashed frame.
        end: Frames from end on stay clean. 0 means T.
        length: Frames in which flashes start, the official trigger_length. 0 means T.
        gap: Clean frames between two flashes, the official strobe_gap.
        duration: Frames per flash, the official strobe_on_duration. As in the official code, 0
            paints one frame per flash, and flashes then start every gap frames.
    """

    polarity: int
    start: int = 0
    end: int = 0
    length: int = 0
    gap: int = 0
    duration: int = 1

    def frames(self, T: int) -> list[int]:
        """The flashed frames of a sample with T frames."""
        end = self.end or T
        length = min(self.length or T, end - self.start)
        starts = range(self.start, self.start + length, self.gap + self.duration)
        return [t for s in starts for t in range(s, s + max(self.duration, 1)) if t < end]

    def apply(self, frames: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Paint the flashed frames."""
        out = frames.clone()
        _paint(out, self.frames(len(frames)), slice(None), slice(None), self.polarity)
        return out
