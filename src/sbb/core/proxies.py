"""Views that give plugins only the resources their card declares."""

import copy
from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel
from torch.utils.data import DataLoader

from sbb.core.cards import AttackCard, DefenseCard, Resource, ScenarioKind
from sbb.core.trainer import Trainer
from sbb.data.module import DataModule
from sbb.data.poison import PoisonSet
from sbb.data.sample import EventSample
from sbb.models.snn import SNNModel
from sbb.models.spec import ArchSpec
from sbb.utils.errors import ThreatModelViolation

T = TypeVar("T")


class ScenarioInfo(BaseModel, frozen=True):
    """Public description of a scenario, always available to every plugin.

    input_shape is the (P, H, W) of one frame after the representation, the input of the model.
    """

    dataset: str
    sensor_size: tuple[int, int, int]
    num_classes: int
    T: int
    input_shape: tuple[int, int, int]


class TrainSetView:
    """The train split as the victim receives it. Poisoned samples are included, unmarked.

    A split poisoned by a frame trigger exists only as frames. Its raw events would be the clean
    ones, and comparing them with the frames from loader() would reveal every poisoned sample.

    Args:
        data: The scenario's DataModule.
        overlay: The poison set applied to the split, or None for the clean split.
    """

    def __init__(self, data: DataModule, overlay: PoisonSet | None) -> None:
        self._data = data
        self._overlay = overlay
        self._poisoned = {} if overlay is None else overlay.by_index()
        self.indices = np.asarray(data.split.train)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> EventSample:
        if self._overlay is not None and self._overlay.frame_trigger is not None:
            raise ThreatModelViolation(
                "this train split was poisoned after the representation and exists only as "
                "frames, read it through loader()"
            )
        index = int(self.indices[position])
        if index in self._poisoned:
            sample, label = self._poisoned[index]
            return EventSample(sample.events, label, index)
        return self._data.dataset[index]

    def labels(self) -> np.ndarray:
        """Training labels by position."""
        labels: np.ndarray = self._data.labels[self.indices].copy()
        for position, index in enumerate(self.indices.tolist()):
            if index in self._poisoned:
                labels[position] = self._poisoned[index][1]
        return labels

    def loader(
        self, batch_size: int, indices: np.ndarray | None = None, shuffle: bool = False
    ) -> DataLoader[Any]:
        """Loader over the given positions, or all of them."""
        return self._data.train_loader(self._overlay, batch_size, keep=indices, shuffle=shuffle)


def _require(value: T | None, resource: str, owner: str) -> T:
    if value is None:
        raise ThreatModelViolation(f"{owner} reads {resource}, which its card does not declare")
    return value


class AttackerView:
    """What an attack may use, derived from its card (docs/PLAN.md section 6.5).

    Every attacker sees the clean train split and can build loaders over its own poison set. Model
    attacks (training_control, supply_chain) also get the scenario architecture and a Trainer,
    and supply_chain attacks a copy of the clean model, which stands for the zoo model.

    Attributes:
        scenario: Public scenario description.
        seed: Cell seed, for rng_for.
        rng: Seeded generator.
    """

    def __init__(
        self,
        card: AttackCard,
        scenario: ScenarioInfo,
        seed: int,
        data: DataModule,
        arch: ArchSpec,
        trainer: Trainer,
        clean_model: SNNModel,
    ) -> None:
        if card.knowledge.data == "partial":
            raise NotImplementedError("partial data knowledge is not supported yet")
        self.scenario = scenario
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._owner = card.name
        self._data = data
        model_attack = card.kind in (ScenarioKind.TRAINING_CONTROL, ScenarioKind.SUPPLY_CHAIN)
        knows_arch = card.knowledge.architecture == "known"
        self._arch = arch if knows_arch or model_attack else None
        self._trainer = trainer if model_attack else None
        supply_chain = card.kind == ScenarioKind.SUPPLY_CHAIN
        self._clean_model = copy.deepcopy(clean_model) if supply_chain else None

    @property
    def train_set(self) -> TrainSetView:
        """The clean train split."""
        return TrainSetView(self._data, None)

    @property
    def arch(self) -> ArchSpec:
        """The scenario architecture."""
        return _require(self._arch, "arch", self._owner)

    @property
    def trainer(self) -> Trainer:
        """A Trainer with the scenario recipe."""
        return _require(self._trainer, "trainer", self._owner)

    @property
    def clean_model(self) -> SNNModel:
        """A copy of the clean model trained on the scenario."""
        return _require(self._clean_model, "clean_model", self._owner)

    def poisoned_loader(
        self, poison: PoisonSet, batch_size: int, shuffle: bool = True
    ) -> DataLoader[Any]:
        """Loader over the train split with the attacker's own poison set applied."""
        return self._data.train_loader(poison, batch_size, shuffle=shuffle)


class DefenderView:
    """What a defense may use, exactly the resources its card declares.

    Attributes:
        scenario: Public scenario description.
        arch: The scenario architecture, public, so a defense can train its own reference model.
        trainer: A fresh Trainer with the scenario recipe.
        rng: Seeded generator.
    """

    def __init__(
        self,
        card: DefenseCard,
        scenario: ScenarioInfo,
        seed: int,
        trainer: Trainer,
        model: SNNModel,
        train_set: TrainSetView,
        data: DataModule,
        arch: ArchSpec,
    ) -> None:
        declared = card.view.resources
        self.scenario = scenario
        self.arch = arch
        self.trainer = trainer
        self.rng = np.random.default_rng(seed)
        self._owner = card.name
        self._model = copy.deepcopy(model) if Resource.MODEL in declared else None
        self._train_set = train_set if Resource.TRAIN_SET in declared else None
        self._data = data if Resource.CLEAN_SUBSET in declared else None
        self._clean_fraction = card.view.clean_fraction

    @property
    def model(self) -> SNNModel:
        """A copy of the trained victim."""
        return _require(self._model, "model", self._owner)

    @property
    def train_set(self) -> TrainSetView:
        """The train split as the victim received it."""
        return _require(self._train_set, "train_set", self._owner)

    def clean_loader(self, batch_size: int) -> DataLoader[Any]:
        """Shuffled loader over the declared clean fraction of the defender pool."""
        data = _require(self._data, "clean_subset", self._owner)
        return data.clean_subset_loader(self._clean_fraction, batch_size)
