"""DataModule, the only class that builds DataLoaders."""

import logging
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sbb import SBB_MINOR
from sbb.data.poison import PoisonSet
from sbb.data.representations import Representation
from sbb.data.sample import EventDataset, EventSample
from sbb.data.splits import Split
from sbb.utils.determinism import rng_for
from sbb.utils.errors import BudgetViolation, SbbError
from sbb.utils.hashing import sha256_hex
from sbb.utils.storage import ArtifactCache

log = logging.getLogger(__name__)

Trigger = Callable[[EventSample, np.random.Generator], EventSample]
# One loader item: full-dataset index, label, and its raw events. None means "use the frame cache",
# a callable makes the sample lazily (triggered test samples).
Entry = tuple[int, int, EventSample | Callable[[], EventSample] | None]
Item = tuple[torch.Tensor, int, int]


class _Items(Dataset[Item]):
    def __init__(self, module: "DataModule", entries: Sequence[Entry]) -> None:
        self.module = module
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> Item:
        index, label, source = self.entries[i]
        if source is None:
            return self.module.cached_frames(index), label, index
        sample = source if isinstance(source, EventSample) else source()
        return self.module.encode(sample), label, index


class DataModule:
    """Builds every loader of a scenario. Batches are (x [B, T, P, H, W], y [B], index [B]).

    Frames of clean samples are cached by (dataset, split, representation).
    Overlay and triggered samples are encoded on the fly.

    Args:
        dataset: The prepared dataset.
        split: Train and test indices.
        representation: Encoder from events to frames.
        cache: Artifact cache for the frame cache.
        defender_pool: Full-dataset indices reserved for the defender.
        seed: Seed of the loader shuffling and of per-sample trigger randomness.
    """

    def __init__(
        self,
        dataset: EventDataset,
        split: Split,
        representation: Representation,
        cache: ArtifactCache,
        defender_pool: np.ndarray,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.representation = representation
        self.defender_pool = defender_pool
        self.seed = seed
        self.labels = dataset.labels()
        key = sha256_hex(
            {
                "dataset": dataset.name,
                "split": split.sha256,
                "repr": representation.spec,
                "sbb_minor": SBB_MINOR,
            }
        )
        path = cache.get_or_build(key, self._write_frames)
        self._frames = np.load(path / "frames.npy", mmap_mode="r", allow_pickle=False)
        self._row = {index: row for row, index in enumerate(split.train + split.test)}

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Shape (P, H, W) of one frame."""
        P, H, W = self._frames.shape[2:]
        return P, H, W

    def encode(self, sample: EventSample) -> torch.Tensor:
        """Encode raw events with the scenario representation."""
        return self.representation.encode(sample, self.dataset.sensor_size)

    def cached_frames(self, index: int) -> torch.Tensor:
        """Frames of a clean sample from the frame cache."""
        return torch.from_numpy(self._frames[self._row[index]].astype(np.float32))

    def train_loader(
        self,
        overlay: PoisonSet | None,
        batch_size: int,
        keep: np.ndarray | None = None,
        shuffle: bool = True,
    ) -> DataLoader[Item]:
        """Loader over the train split with the overlay applied.

        Args:
            overlay: Poisoned samples that replace their clean versions.
            batch_size: Batch size.
            keep: Positions in the train split to keep. None keeps all.
            shuffle: Shuffle with a generator seeded by the module seed.
        """
        train = np.asarray(self.split.train)
        if keep is not None:
            train = train[keep]
        poisoned = {} if overlay is None else overlay.by_index()
        entries: list[Entry] = []
        for index in train.tolist():
            if index in poisoned:
                sample, label = poisoned[index]
                entries.append((index, label, sample))
            else:
                entries.append((index, int(self.labels[index]), None))
        return self._loader(entries, batch_size, shuffle)

    def test_loader(self, batch_size: int) -> DataLoader[Item]:
        """Loader over the clean test split, in split order."""
        entries: list[Entry] = [(i, int(self.labels[i]), None) for i in self.split.test]
        return self._loader(entries, batch_size, shuffle=False)

    def triggered_loader(self, trigger: Trigger, target: int, batch_size: int) -> DataLoader[Item]:
        """Loader over triggered test samples whose true label is not target.

        Labels stay the true labels. Each sample gets the RNG rng_for(seed, index).
        """
        entries: list[Entry] = [
            (i, int(self.labels[i]), partial(self._triggered, trigger, i))
            for i in self.split.test
            if self.labels[i] != target
        ]
        return self._loader(entries, batch_size, shuffle=False)

    def clean_subset_loader(self, fraction: float, batch_size: int) -> DataLoader[Item]:
        """Shuffled loader over a random part of the defender pool.

        Args:
            fraction: Size of the subset as a fraction of the train split.
            batch_size: Batch size.
        """
        n = round(fraction * len(self.split.train))
        if n > len(self.defender_pool):
            raise BudgetViolation(
                f"clean subset of {n} samples exceeds the defender pool "
                f"of {len(self.defender_pool)}"
            )
        subset = np.random.default_rng(self.seed).permutation(self.defender_pool)[:n]
        entries: list[Entry] = [(int(i), int(self.labels[i]), None) for i in subset]
        return self._loader(entries, batch_size, shuffle=True)

    def _triggered(self, trigger: Trigger, index: int) -> EventSample:
        return trigger(self.dataset[index], rng_for(self.seed, index))

    def _loader(self, entries: Sequence[Entry], batch_size: int, shuffle: bool) -> DataLoader[Item]:
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            _Items(self, entries), batch_size=batch_size, shuffle=shuffle, generator=generator
        )

    def _write_frames(self, directory: Path) -> None:
        indices = self.split.train + self.split.test
        shape = (len(indices), *self.encode(self.dataset[indices[0]]).shape)
        log.info("encoding %d samples of %s into the frame cache", len(indices), self.dataset.name)
        frames = np.lib.format.open_memmap(
            directory / "frames.npy", mode="w+", dtype=np.uint16, shape=shape
        )
        # ponytail: uint16 counts keep the cache small. Non-count representations need floats.
        limit = np.iinfo(np.uint16).max
        for row, index in enumerate(indices):
            x = self.encode(self.dataset[index]).numpy()
            if x.max(initial=0) > limit or not np.array_equal(x, np.rint(x)):
                raise SbbError("the frame cache stores integer counts up to 65535 only")
            frames[row] = x
            if (row + 1) % 10_000 == 0:
                log.info("encoded %d of %d", row + 1, len(indices))
        frames.flush()
