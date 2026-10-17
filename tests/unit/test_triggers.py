"""Shared triggers and selectors. Expected values follow the official code (docs/PLAN.md 12)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sbb.attacks.frame_triggers import (
    FlashTrigger,
    FramePatchTrigger,
    MovingPatchTrigger,
    SmartPatchTrigger,
    _edges,
    _mode_counts,
)
from sbb.attacks.labels import DirtyLabel
from sbb.attacks.selectors import RandomSelector, TargetClassSelector
from sbb.attacks.temporal_triggers import JitterTrigger, LatencyTrigger, RateTrigger
from sbb.attacks.triggers import IdentityTrigger
from sbb.core.plugins import FrameTrigger, PoisoningAttack, Trigger
from sbb.core.proxies import AttackerView, ScenarioInfo
from sbb.core.trainer import Trainer, TrainSpec
from sbb.data.module import DataModule
from sbb.data.poison import Budget
from sbb.data.sample import EventSample
from sbb.models.snn import build_model
from sbb.models.spec import ArchSpec
from sbb.utils.determinism import rng_for
from sbb.utils.errors import SbbError
from tests.cards import attack_card
from tests.contracts.test_plugins import check_trigger

RNG = rng_for(0, 0)
INFO = ScenarioInfo(
    dataset="synthetic", sensor_size=(16, 16, 2), num_classes=4, T=4, input_shape=(2, 16, 16)
)
ARCH = ArchSpec(name="tiny_lif_cnn", T=4)


def _view(datamodule: DataModule, seed: int = 0) -> AttackerView:
    trainer = Trainer(TrainSpec(epochs=1, batch_size=8), torch.device("cpu"))
    clean = build_model(ARCH, (2, 16, 16), 4)
    return AttackerView(attack_card(), INFO, seed, datamodule, ARCH, trainer, clean)


def _squares(frames: torch.Tensor) -> list[tuple[int, int] | None]:
    """Top-left (y, x) of the painted pixels of channel 1 in each frame of zero frames."""
    corners: list[tuple[int, int] | None] = []
    for frame in frames[:, 1]:
        ys, xs = torch.nonzero(frame, as_tuple=True)
        corners.append((int(ys.min()), int(xs.min())) if len(ys) else None)
    return corners


@pytest.mark.parametrize(("polarity", "off", "on"), [(0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1)])
def test_patch_writes_the_official_polarity_modes(polarity: int, off: int, on: int) -> None:
    frames = torch.full((2, 2, 8, 8), 5.0)
    out = FramePatchTrigger(0.25, "top-left", polarity).apply(frames, RNG)
    assert (out[:, 0, :2, :2] == off).all() and (out[:, 1, :2, :2] == on).all()
    assert (out[:, :, 2:] == 5).all() and (out[:, :, :, 2:] == 5).all()
    assert (frames == 5).all(), "the input must stay unchanged"


@pytest.mark.parametrize(
    ("position", "corner"),
    [
        ("top-left", (0, 0)),
        ("top-right", (0, 6)),
        ("bottom-left", (6, 0)),
        ("bottom-right", (6, 6)),
        ("middle", (3, 3)),  # int((8 - 2) / 2)
    ],
)
def test_static_patch_covers_every_frame(position: str, corner: tuple[int, int]) -> None:
    out = FramePatchTrigger(0.25, position, 1).apply(torch.zeros(3, 2, 8, 8), RNG)
    assert _squares(out) == [corner] * 3
    assert out[:, 1].sum() == 3 * 2 * 2


@pytest.mark.parametrize(("size", "width", "side"), [(0.1, 128, 12), (0.1, 34, 3), (0.05, 16, 1)])
def test_patch_side_is_a_fraction_of_the_width_and_at_least_one(
    size: float, width: int, side: int
) -> None:
    out = FramePatchTrigger(size, "top-left", 1).apply(torch.zeros(1, 2, width, width), RNG)
    assert out[0, 1].sum() == side * side


@pytest.mark.parametrize(
    ("position", "width", "corners"),
    [
        # Squares of side 4 at x0, x0 + 6 and x0 + 10, in a cycle. The last frame stays clean.
        ("top-left", 16, [(0, 0), (0, 6), (0, 10), (0, 0), None]),
        ("top-right", 16, [(0, 12), (0, 6), (0, 2), (0, 12), None]),
        ("bottom-left", 16, [(12, 0), (12, 6), (12, 10), (12, 0), None]),
        ("bottom-right", 16, [(12, 12), (12, 6), (12, 2), (12, 12), None]),
        ("middle", 32, [(14, 14), (14, 20), (14, 24), (14, 14), None]),  # int(32 / 2) - 2
    ],
)
def test_moving_patch_follows_the_official_positions(
    position: str, width: int, corners: list[tuple[int, int] | None]
) -> None:
    size = 4 / width
    trigger = MovingPatchTrigger(size, position, 1)
    assert _squares(trigger.apply(torch.zeros(5, 2, width, width), RNG)) == corners


def test_moving_patch_that_leaves_the_frame_is_refused() -> None:
    with pytest.raises(SbbError, match="does not fit"):
        MovingPatchTrigger(0.25, "middle", 1).apply(torch.zeros(2, 2, 16, 16), RNG)


def test_smart_mask_grid_matches_the_official_get_masks() -> None:
    assert _edges(128, 3) == [0, 42, 85, 128]
    assert _edges(34, 3) == [0, 11, 22, 34]
    assert _edges(16, 3) == [0, 5, 10, 16]


def test_smart_counts_the_official_polarity_cases() -> None:
    # One 6x6 frame in a 3x3 grid of 2x2 masks. The center mask has three pixels with only ON
    # events (mode 1) and one with both, and every other mask is empty.
    x = torch.zeros(1, 1, 2, 6, 6)
    x[0, 0, 1, 2:4, 2:4] = 1
    x[0, 0, 0, 3, 3] = 1
    counts = _mode_counts(x, _edges(6, 3), _edges(6, 3))
    # Entries equal to 0, ON-only pixels, OFF-only pixels, entries equal to 1.
    assert counts[4].tolist() == [3, 3, 0, 5]
    assert all(counts[k].tolist() == [8, 0, 0, 0] for k in range(9) if k != 4)


def test_smart_trigger_walks_inside_the_most_active_mask(datamodule: DataModule) -> None:
    trigger = SmartPatchTrigger(0.1)
    with pytest.raises(SbbError, match="fit"):
        trigger.apply(torch.zeros(4, 2, 16, 16), RNG)
    trigger.fit(_view(datamodule))
    assert trigger.walk is not None and trigger.walk.shape == (4, 2)
    assert trigger.polarity in range(4)
    # Side 1 in a mask of the 16x16 grid: corners stay one pixel inside the mask, and every frame
    # moves the square by 2 pixels on each axis, or not at all after bouncing off the border.
    y0, y1, x0, x1 = trigger.mask
    assert ((trigger.walk[:, 0] >= y0 + 1) & (trigger.walk[:, 0] <= y1 - 1)).all()
    assert ((trigger.walk[:, 1] >= x0 + 1) & (trigger.walk[:, 1] <= x1 - 1)).all()
    assert set(np.abs(np.diff(trigger.walk, axis=0)).ravel().tolist()) <= {0, 2}
    painted = (trigger.apply(torch.full((4, 2, 16, 16), 5.0), RNG) != 5).any(dim=1)
    assert painted.sum() == 4  # one pixel per frame
    assert all(painted[t, h, w] for t, (h, w) in enumerate(trigger.walk.tolist()))

    again = SmartPatchTrigger(0.1)
    again.fit(_view(datamodule))
    assert np.array_equal(again.walk, trigger.walk), "the walk depends only on the seed"


def _cells() -> torch.Tensor:
    """One 12x12 frame in a 3x3 grid of 4x4 cells, with known polarity cases.

    Every cell holds one ON-only pixel, except the empty cell in row 2, column 1, the least
    active. The cell in row 1, column 2 is the most active. Its cases are 7 entries equal to 0,
    1 ON-only pixel, 4 OFF-only pixels and 25 entries equal to 1.
    """
    x = torch.zeros(1, 1, 2, 12, 12)
    for r in range(3):
        for c in range(3):
            if (r, c) != (2, 1):
                x[0, 0, 1, 4 * r, 4 * c] = 1
    cell = torch.zeros(2, 16)  # (channel, pixel) of the cell in row 1, column 2
    cell[1, 0] = 1  # one ON-only pixel, then one empty pixel
    cell[0, 2:6] = 1  # four OFF-only pixels
    cell[:, 6:] = 1  # ten pixels with both
    x[0, 0, :, 4:8, 8:12] = cell.reshape(2, 4, 4)
    return x


@pytest.mark.parametrize(
    ("least", "most_polarity", "mask", "polarity"),
    [
        (False, False, (4, 8, 8, 12), 1),  # the rarest case is ON-only
        (False, True, (4, 8, 8, 12), 3),  # the most frequent case is entries equal to 1
        (True, False, (8, 12, 4, 8), 1),  # an empty cell: ties go to the first case, as argmin
        (True, True, (8, 12, 4, 8), 0),
    ],
)
def test_smart_trigger_picks_the_cell_and_mode_of_the_official_code(
    least: bool, most_polarity: bool, mask: tuple[int, int, int, int], polarity: int
) -> None:
    frames = _cells()
    view = SimpleNamespace(  # only what fit reads
        scenario=SimpleNamespace(input_shape=(2, 12, 12), T=1),
        train_set=SimpleNamespace(loader=lambda batch_size: [(frames, None, None)]),
        rng=np.random.default_rng(0),
    )
    trigger = SmartPatchTrigger(0.1, least=least, most_polarity=most_polarity)
    trigger.fit(view)  # type: ignore[arg-type]
    assert (trigger.mask, trigger.polarity) == (mask, polarity)


def test_smart_walk_does_not_depend_on_what_the_selector_drew(datamodule: DataModule) -> None:
    fresh, used = _view(datamodule), _view(datamodule)
    used.rng.random(1000)  # a selector drew from the view's generator first
    first, second = SmartPatchTrigger(0.1), SmartPatchTrigger(0.1)
    first.fit(fresh)
    second.fit(used)
    assert np.array_equal(first.walk, second.walk)


@pytest.mark.parametrize(
    ("kwargs", "frames"),
    [
        ({"length": 3}, [0, 1, 2]),  # Flashy claim 1: three flashes at the start
        ({}, list(range(16))),  # length 0 means every frame
        ({"start": 4, "gap": 2, "duration": 2, "length": 8}, [4, 5, 8, 9]),
        ({"start": 10, "end": 14, "duration": 3, "length": 8}, [10, 11, 12, 13]),
        ({"gap": 1, "duration": 3, "length": 5}, [0, 1, 2, 4, 5, 6]),  # a strobe overruns length
    ],
)
def test_flash_frames_follow_get_attack_frames(kwargs: dict[str, int], frames: list[int]) -> None:
    assert FlashTrigger(polarity=1, **kwargs).frames(16) == frames


def test_flash_paints_whole_frames() -> None:
    frames = torch.full((4, 2, 8, 8), 5.0)
    out = FlashTrigger(polarity=1, start=1, length=2).apply(frames, RNG)
    assert (out[1:3, 0] == 0).all() and (out[1:3, 1] == 1).all()
    assert torch.equal(out[[0, 3]], frames[[0, 3]])


@pytest.mark.parametrize(
    ("trigger", "T", "order"),
    [
        (LatencyTrigger(2), 5, [0, 0, 0, 1, 2]),
        (LatencyTrigger(7), 5, [0, 0, 0, 0, 0]),
        (RateTrigger(0.5), 10, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
        (RateTrigger(1.8), 10, [0, 1, 3, 5, 7, 9, 9, 9, 9, 9]),
        (JitterTrigger(3), 10, [1, 0, 3, 2, 5, 4, 6, 7, 8, 9]),  # step max(1, 10 // 6) = 1
        (JitterTrigger(2), 16, [4, 1, 2, 3, 0, 5, 6, 7, 12, 9, 10, 11, 8, 13, 14, 15]),
    ],
)
def test_temporal_triggers_remap_frames_as_the_paper_defines(
    trigger: LatencyTrigger | RateTrigger | JitterTrigger, T: int, order: list[int]
) -> None:
    assert trigger.order(T).tolist() == order
    frames = torch.arange(T, dtype=torch.float32).reshape(T, 1, 1, 1).expand(T, 2, 3, 3)
    assert trigger.apply(frames, RNG)[:, 0, 0, 0].tolist() == order


SHARED: list[FrameTrigger] = [
    FramePatchTrigger(0.1, "top-left", 1),
    MovingPatchTrigger(0.1, "bottom-right", 2),
    FlashTrigger(polarity=3, length=1),
    LatencyTrigger(2),
    RateTrigger(0.5),
    JitterTrigger(1),
]


@pytest.mark.parametrize("trigger", SHARED, ids=lambda t: type(t).__name__)
def test_shared_triggers_meet_the_trigger_contract(
    trigger: FrameTrigger, datamodule: DataModule
) -> None:
    check_trigger(trigger, datamodule)


def test_fitted_smart_trigger_meets_the_trigger_contract(datamodule: DataModule) -> None:
    trigger = SmartPatchTrigger(0.1)
    trigger.fit(_view(datamodule))
    check_trigger(trigger, datamodule)


class _DrawsWhenFitted(Trigger):
    def fit(self, view: AttackerView) -> None:
        view.rng.random(1000)

    def apply(self, sample: EventSample, rng: np.random.Generator) -> EventSample:
        return sample


class _Poisoner(PoisoningAttack):
    card = attack_card()

    def __init__(self, trigger: Trigger) -> None:
        super().__init__({}, target=0)
        self.trigger, self.selector, self.labels = trigger, RandomSelector(), DirtyLabel()


def test_poison_selects_before_it_fits_the_trigger(datamodule: DataModule) -> None:
    budget, pool = Budget(), datamodule.defender_pool
    plain = _Poisoner(IdentityTrigger()).poison(_view(datamodule), budget, pool)
    drawing = _Poisoner(_DrawsWhenFitted()).poison(_view(datamodule), budget, pool)
    assert len(plain.indices) == 48 and np.array_equal(plain.indices, drawing.indices)


LABELS = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])


def test_random_selector_skips_the_target_class_unless_asked() -> None:
    picked = RandomSelector().select(LABELS, 10, 0, np.array([1]), rng_for(0, 0))
    assert sorted(picked.tolist()) == [2, 4, 5, 7, 8]
    picked = RandomSelector(include_target=True).select(LABELS, 3, 0, np.array([1]), rng_for(0, 0))
    assert len(picked) == 3 and 1 not in picked and len(set(picked.tolist())) == 3
    every = RandomSelector(include_target=True).select(LABELS, 10, 0, np.array([1]), rng_for(0, 0))
    assert sorted(every.tolist()) == [0, 2, 3, 4, 5, 6, 7, 8, 9]


def test_target_class_selector_picks_only_the_target_class() -> None:
    picked = TargetClassSelector().select(LABELS, 10, 0, np.array([3]), rng_for(0, 0))
    assert sorted(picked.tolist()) == [0, 6, 9]
    assert len(TargetClassSelector().select(LABELS, 2, 0, np.array([3]), rng_for(0, 0))) == 2
