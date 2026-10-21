"""Scenario, Cell and artifact keys (docs/PLAN.md sections 6.8 and 7.6)."""

from typing import Any, Literal, Self

from pydantic import BaseModel, model_validator

from sbb import SBB_MINOR
from sbb.core.trainer import TrainSpec
from sbb.data.poison import Budget
from sbb.data.representations import ReprSpec
from sbb.data.splits import DatasetSpec
from sbb.models.spec import ArchSpec
from sbb.utils.hashing import sha256_hex


class Scenario(BaseModel, frozen=True):
    """Fixed dataset, split, representation, architecture, recipe and budget."""

    name: str
    data: DatasetSpec
    repr: ReprSpec
    arch: ArchSpec
    train: TrainSpec
    budget: Budget

    @model_validator(mode="after")
    def _same_T(self) -> Self:
        if self.arch.T != self.repr.T:
            raise ValueError(f"arch.T={self.arch.T} differs from repr.T={self.repr.T}")
        return self


class Cell(BaseModel, frozen=True):
    """One evaluation of (scenario, attack, defense, seed).

    attack_hparams are merged over the attack card's hyperparameters. Benchmark cells leave them
    empty, and replications give the paper's.
    """

    benchmark_version: str
    track: Literal["lite", "full", "replication"]
    scenario: Scenario
    attack: str
    attack_version: str
    defense: str
    defense_version: str
    seed: int
    attack_hparams: dict[str, Any] = {}

    def key(self) -> str:
        """Hash of the full cell, benchmark version included."""
        return sha256_hex(self)


def artifact_key(**parts: Any) -> str:
    """Hash of everything that determines an artifact, plus the package major.minor."""
    return sha256_hex({**parts, "sbb_minor": SBB_MINOR})
