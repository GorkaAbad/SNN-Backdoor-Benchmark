import json
from pathlib import Path

import numpy as np
import pytest
import torch

import sbb.attacks  # noqa: F401  registers the plugins
import sbb.defenses  # noqa: F401
import sbb.metrics  # noqa: F401
from sbb.core.game import Game
from sbb.core.registry import ATTACKS, DEFENSES
from sbb.core.result import CellResult
from sbb.core.scenario import Cell, Scenario
from sbb.data.datasets.synthetic import SyntheticEvents
from sbb.data.module import DataModule
from sbb.data.representations import FrameRepr, ReprSpec
from sbb.data.splits import load_split, reserve_defender_pool
from sbb.utils.storage import ArtifactCache, LocalStore

SCENARIO = Scenario.model_validate(
    {
        "name": "synthetic_tiny",
        "data": {"name": "synthetic", "split": "synthetic_v1"},
        "repr": {"T": 4},
        "arch": {"name": "tiny_lif_cnn", "T": 4},
        "train": {"epochs": 3, "batch_size": 32, "lr": 0.01},
        "budget": {"poison_rate": 0.1, "target": 0},
    }
)


def make_cell(attack: str, defense: str, seed: int = 0, scenario: Scenario = SCENARIO) -> Cell:
    return Cell(
        benchmark_version="test",
        track="lite",
        scenario=scenario,
        seed=seed,
        attack=attack,
        attack_version=ATTACKS.get(attack).card.version,
        defense=defense,
        defense_version=DEFENSES.get(defense).card.version,
    )


@pytest.fixture
def synthetic() -> SyntheticEvents:
    return SyntheticEvents(Path("unused"))


@pytest.fixture
def cache(tmp_path: Path) -> ArtifactCache:
    return ArtifactCache(LocalStore(tmp_path / "cache"))


@pytest.fixture
def datamodule(synthetic: SyntheticEvents, cache: ArtifactCache) -> DataModule:
    split = load_split("synthetic_v1")
    pool = reserve_defender_pool(synthetic.labels(), split.train, 0.05, seed=0)
    return DataModule(synthetic, split, FrameRepr(ReprSpec(T=4)), cache, pool, seed=0)


@pytest.fixture
def pool(datamodule: DataModule) -> np.ndarray:
    return datamodule.defender_pool


@pytest.fixture
def game(cache: ArtifactCache, tmp_path: Path) -> Game:
    return Game(cache, torch.device("cpu"), tmp_path / "data")


def run_in_fresh_caches(
    cell: Cell, tmp_path: Path, device: torch.device, runs: int = 2
) -> list[tuple[CellResult, str]]:
    """Run cell once per fresh cache. Return each result with the sha256 of its defended model."""
    out = []
    for i in range(runs):
        root = tmp_path / f"run{i}"
        result = Game(ArtifactCache(LocalStore(root)), device, tmp_path / "data").run(cell)
        assert result.outcome == "ok", result.reason
        meta = root / result.artifacts["defended_model"] / "checkpoint.json"
        out.append((result, json.loads(meta.read_text())["sha256"]))
    return out
