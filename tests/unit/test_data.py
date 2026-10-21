from collections.abc import Sequence

import numpy as np
import pytest
import torch

from sbb.data.datasets.synthetic import SyntheticEvents
from sbb.data.module import DataModule
from sbb.data.poison import Budget, PoisonSet
from sbb.data.representations import FrameRepr, ReprSpec
from sbb.data.sample import EVENT_DTYPE, EventSample, make_events
from sbb.data.splits import Split, load_split, reserve_defender_pool
from sbb.utils.determinism import rng_for
from sbb.utils.errors import BudgetViolation, ReproducibilityError
from sbb.utils.storage import ArtifactCache


def test_synthetic_samples_are_sorted_and_deterministic(synthetic: SyntheticEvents) -> None:
    a, b = synthetic[5], synthetic[5]
    assert a.events.dtype == EVENT_DTYPE and np.all(np.diff(a.events["t"]) >= 0)
    assert np.array_equal(a.events, b.events) and a.label == synthetic.labels()[5]


@pytest.mark.parametrize("split_by", ["number", "time"])
def test_frames_match_spikingjelly(synthetic: SyntheticEvents, split_by: str) -> None:
    from spikingjelly.datasets import integrate_events_by_fixed_frames_number

    ev = synthetic[3].events
    ours = FrameRepr(ReprSpec(T=5, split_by=split_by)).encode(synthetic[3], (16, 16, 2))
    theirs = integrate_events_by_fixed_frames_number(
        {k: ev[k].astype(np.int64) for k in "txyp"}, split_by, 5, 16, 16
    )
    assert torch.equal(ours, torch.from_numpy(theirs).float())


def test_frames_downsample_and_empty() -> None:
    repr_ = FrameRepr(ReprSpec(T=2, downsample=2))
    sample = EventSample(make_events([0, 15], [0, 15], [0, 1], [0, 1]), 0, 0)
    x = repr_.encode(sample, (16, 16, 2))
    assert x.shape == (2, 2, 8, 8) and x[0, 0, 0, 0] == 1 and x[1, 1, 7, 7] == 1
    empty = EventSample(make_events([], [], [], []), 0, 0)
    assert repr_.encode(empty, (16, 16, 2)).sum() == 0


def test_split_hash_is_checked() -> None:
    split = load_split("synthetic_v1")
    with pytest.raises(ReproducibilityError):
        Split(**{**split.model_dump(), "test": split.test[:-1]})


def test_defender_pool_is_stratified_and_seeded(synthetic: SyntheticEvents) -> None:
    labels, train = synthetic.labels(), list(range(480))
    pool = reserve_defender_pool(labels, train, 0.05, seed=0)
    assert np.array_equal(pool, reserve_defender_pool(labels, train, 0.05, seed=0))
    assert np.bincount(labels[pool]).tolist() == [6, 6, 6, 6]


def _poison(synthetic: SyntheticEvents, indices: list[int], labels: list[int]) -> PoisonSet:
    return PoisonSet(np.array(indices), tuple(synthetic[i] for i in indices), np.array(labels))


def test_poison_set_validation(synthetic: SyntheticEvents, pool: np.ndarray) -> None:
    budget, labels = Budget(poison_rate=0.01), synthetic.labels()
    free = [i for i in range(480) if i not in pool][:5]
    ok = _poison(synthetic, free[:4], [0] * 4)
    ok.validate(budget, 480, labels, "dirty", pool)
    cases = {
        "exceed the budget": _poison(synthetic, free, [0] * 5),
        "defender pool": _poison(synthetic, [int(pool[0])], [0]),
        "duplicate": _poison(synthetic, [free[0], free[0]], [0, 0]),
    }
    for message, poison in cases.items():
        with pytest.raises(BudgetViolation, match=message):
            poison.validate(budget, 480, labels, "dirty", pool)
    with pytest.raises(BudgetViolation, match="clean-label"):
        ok.validate(budget, 480, labels, "clean", pool)


def test_loaders(datamodule: DataModule, synthetic: SyntheticEvents) -> None:
    x, y, index = next(iter(datamodule.test_loader(batch_size=8)))
    assert x.shape == (8, 4, 2, 16, 16) and x.dtype == torch.float32
    assert y.tolist() == synthetic.labels()[index.numpy()].tolist()

    blank = EventSample(make_events([], [], [], []), 3, 7)
    overlay = PoisonSet(np.array([7]), (blank,), np.array([0]))
    batches = list(datamodule.train_loader(overlay, batch_size=480, shuffle=False))
    x, y, index = batches[0]
    pos = index.tolist().index(7)
    assert y[pos] == 0 and x[pos].sum() == 0 and len(index) == 480

    def flip(sample: EventSample, rng: np.random.Generator) -> EventSample:
        return EventSample(sample.events[:0], sample.label, sample.index)

    x, y, _ = next(iter(datamodule.triggered_loader(200, events=flip)))
    assert len(y) == len(datamodule.split.test) and x.sum() == 0  # every class, target included

    _, _, index = next(iter(datamodule.clean_subset_loader(0.05, batch_size=100)))
    assert set(index.tolist()) == set(datamodule.defender_pool.tolist())


def _plus_seed(frames: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """A frame trigger that adds a number drawn from the sample's RNG."""
    return frames + float(rng.integers(1, 100))


def test_frame_triggers_act_after_the_representation(datamodule: DataModule) -> None:
    sample = datamodule.dataset[7]
    overlay = PoisonSet(np.array([7]), (sample,), np.array([0]), frame_trigger=_plus_seed)
    x, y, index = next(iter(datamodule.train_loader(overlay, batch_size=480, shuffle=False)))
    shift = float(rng_for(0, 7).integers(1, 100))
    for position, i in enumerate(index.tolist()):
        expected = datamodule.cached_frames(i) + (shift if i == 7 else 0)
        assert torch.equal(x[position], expected), "only the poisoned sample changes"
    assert y[index.tolist().index(7)] == 0

    x, y, index = next(iter(datamodule.triggered_loader(200, frames=_plus_seed)))
    assert len(y) == len(datamodule.split.test)
    for frames, i in zip(x, index.tolist(), strict=True):
        shift = float(rng_for(0, i).integers(1, 100))
        assert torch.equal(frames, datamodule.cached_frames(i) + shift)


def test_frame_cache_is_reused(
    datamodule: DataModule, synthetic: SyntheticEvents, cache: ArtifactCache
) -> None:
    again = DataModule(
        synthetic,
        datamodule.split,
        datamodule.representation,
        cache,
        datamodule.defender_pool,
        seed=0,
    )
    assert torch.equal(again.cached_frames(0), datamodule.encode(synthetic[0]))
    assert len(list(cache.local.root.iterdir())) == 1


def test_clean_subset_fits_a_pool_rounded_down_per_class(
    synthetic: SyntheticEvents, cache: ArtifactCache
) -> None:
    split = load_split("synthetic_v1")
    pool = reserve_defender_pool(synthetic.labels(), split.train, 0.07, seed=0)
    assert len(pool) == 32 < round(0.07 * 480)  # 8.4 per class rounds down
    dm = DataModule(synthetic, split, FrameRepr(ReprSpec(T=4)), cache, pool, seed=0)
    _, _, index = next(iter(dm.clean_subset_loader(0.07, batch_size=100)))
    assert sorted(index.tolist()) == pool.tolist()


def _ordered_9to1(part: range) -> list[int]:
    return [1000 * c + j for c in range(10) for j in part]


@pytest.mark.parametrize(
    ("name", "train", "test"),
    [
        ("synthetic_v1", range(480), range(480, 600)),
        ("nmnist_official_v1", range(60_000), range(60_000, 70_000)),
        ("dvsgesture_official_v1", range(1176), range(1176, 1464)),
        # SpikingJelly's split_to_train_test_set(0.9): the first 900 files of each class train.
        ("cifar10dvs_9to1_v1", _ordered_9to1(range(900)), _ordered_9to1(range(900, 1000))),
    ],
)
def test_split_files_follow_their_recipes(
    name: str, train: Sequence[int], test: Sequence[int]
) -> None:
    split = load_split(name)
    assert split.train == tuple(train) and split.test == tuple(test)
