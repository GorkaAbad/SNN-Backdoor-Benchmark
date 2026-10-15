"""Train and test splits stored as JSON files, and the defender pool."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel, model_validator

from sbb.utils.errors import ReproducibilityError
from sbb.utils.hashing import sha256_hex

SPLITS_DIR = Path(__file__).parent / "splits"


def split_sha256(dataset: str, train: Sequence[int], test: Sequence[int]) -> str:
    """Hash of the canonical JSON of (dataset, train, test)."""
    return sha256_hex({"dataset": dataset, "train": list(train), "test": list(test)})


class Split(BaseModel, frozen=True):
    """Full-dataset indices of the train and test samples."""

    name: str
    dataset: str
    train: tuple[int, ...]
    test: tuple[int, ...]
    sha256: str

    @model_validator(mode="after")
    def _check_hash(self) -> "Split":
        if self.sha256 != split_sha256(self.dataset, self.train, self.test):
            raise ReproducibilityError(f"split {self.name} does not match its sha256")
        return self


class DatasetSpec(BaseModel, frozen=True):
    """Dataset of a scenario: registry name, split file name and optional Hub revision."""

    name: str
    split: str
    revision: str | None = None


def load_split(name: str) -> Split:
    """Load splits/<name>.json and verify its hash."""
    return Split.model_validate_json((SPLITS_DIR / f"{name}.json").read_text())


def reserve_defender_pool(
    labels: np.ndarray, train: Sequence[int], fraction: float, seed: int
) -> np.ndarray:
    """Draw a class-stratified fraction of the train split that the attacker cannot modify.

    Returns:
        Sorted full-dataset indices of the pool.
    """
    train_idx = np.asarray(train)
    train_labels = labels[train_idx]
    rng = np.random.default_rng(seed)
    pool = []
    for c in np.unique(train_labels):
        members = train_idx[train_labels == c]
        pool.append(rng.choice(members, round(fraction * len(members)), replace=False))
    return np.sort(np.concatenate(pool))
