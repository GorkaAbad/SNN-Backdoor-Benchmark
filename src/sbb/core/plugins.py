"""Base classes of attacks, defenses and metrics (docs/PLAN.md sections 6.5, 6.6, 6.8)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
import torch

from sbb.core.cards import AttackCard, DefenseCard, Stage
from sbb.core.proxies import AttackerView, DefenderView
from sbb.core.trainer import TrainerHooks
from sbb.data.poison import Budget, PoisonSet
from sbb.data.sample import EventSample
from sbb.models.snn import SNNModel
from sbb.utils.determinism import rng_for


class Attack(ABC):
    """Base of every attack. Hyperparameters come from the card."""

    card: ClassVar[AttackCard]

    def __init__(self, hparams: dict[str, Any], target: int) -> None:
        self.hparams = hparams
        self.target = target

    @abstractmethod
    def trigger_test(self, sample: EventSample, rng: np.random.Generator) -> EventSample:
        """Apply the trigger to a test sample."""


class Trigger(ABC):
    """A transformation of raw events that activates the backdoor."""

    def fit(self, view: AttackerView) -> None:  # noqa: B027  optional hook
        """Learn the trigger. Only learned triggers override this."""

    @abstractmethod
    def apply(self, sample: EventSample, rng: np.random.Generator) -> EventSample:
        """Return the triggered sample with the same label and index."""


class Selector(ABC):
    """Chooses which training samples to poison."""

    @abstractmethod
    def select(
        self,
        labels: np.ndarray,
        n: int,
        target: int,
        excluded: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return at most n positions in the train set, none of them in excluded."""


class LabelPolicy(ABC):
    """Training label of a poisoned sample."""

    name: ClassVar[Literal["clean", "dirty"]]

    @abstractmethod
    def relabel(self, label: int, target: int) -> int:
        """Return the training label."""


class PoisoningAttack(Attack):
    """A data-poisoning attack composed of a trigger, a selector and a label policy."""

    trigger: Trigger
    selector: Selector
    labels: LabelPolicy

    def poison(self, view: AttackerView, budget: Budget, pool: np.ndarray) -> PoisonSet:
        """Select samples outside the defender pool, trigger them and relabel them."""
        train = view.train_set
        self.trigger.fit(view)
        labels = train.labels()
        excluded = np.flatnonzero(np.isin(train.indices, pool))
        n = budget.max_poison(len(train))
        positions = self.selector.select(labels, n, self.target, excluded, view.rng)
        indices = train.indices[positions]
        samples = tuple(
            self.trigger.apply(train[int(p)], rng_for(view.seed, int(i)))
            for p, i in zip(positions, indices, strict=True)
        )
        new_labels = [self.labels.relabel(int(labels[p]), self.target) for p in positions]
        return PoisonSet(indices, samples, np.array(new_labels, dtype=np.int64))

    def trigger_test(self, sample: EventSample, rng: np.random.Generator) -> EventSample:
        """Apply the trigger."""
        return self.trigger.apply(sample, rng)


class ModelAttack(Attack):
    """An attack that returns a backdoored model (training_control, supply_chain)."""

    @abstractmethod
    def make_model(self, view: AttackerView) -> SNNModel:
        """Return the backdoored model."""


class Defense(ABC):
    """Base of every defense. Hyperparameters come from the card."""

    card: ClassVar[DefenseCard]

    def __init__(self, hparams: dict[str, Any]) -> None:
        self.hparams = hparams


class DataDefense(Defense):
    """Pre-training defense."""

    @abstractmethod
    def sanitize(self, view: DefenderView) -> np.ndarray:
        """Return the positions in train_set to keep."""


class TrainingDefense(Defense):
    """In-training defense."""

    @abstractmethod
    def hooks(self, view: DefenderView) -> TrainerHooks:
        """Return the hooks for the victim's training run."""


class ModelDefense(Defense):
    """Post-training defense."""

    @abstractmethod
    def repair(self, view: DefenderView) -> SNNModel:
        """Return the repaired model."""


class InputDefense(Defense):
    """Inference-time defense that rejects suspicious inputs."""

    threshold: float

    @abstractmethod
    def fit(self, view: DefenderView) -> None:
        """Prepare the detector. Must set self.threshold."""

    @abstractmethod
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Return one score per input [B]. Higher means more suspicious."""


class DetectionDefense(Defense):
    """Defense that decides whether a model is backdoored."""

    @abstractmethod
    def verdict(self, view: DefenderView) -> float:
        """Return a score. Higher means more likely backdoored."""


STAGE_BASES: dict[Stage, type[Defense]] = {
    Stage.PRE_TRAINING: DataDefense,
    Stage.IN_TRAINING: TrainingDefense,
    Stage.POST_TRAINING: ModelDefense,
    Stage.INFERENCE: InputDefense,
    Stage.MODEL_DETECTION: DetectionDefense,
}


@dataclass(frozen=True)
class EvalContext:
    """Predictions on the clean and triggered test sets."""

    clean_pred: np.ndarray
    clean_true: np.ndarray
    triggered_pred: np.ndarray
    triggered_true: np.ndarray
    target: int


class Metric(ABC):
    """A metric computed from an EvalContext. name equals the registry name."""

    name: ClassVar[str]

    @abstractmethod
    def compute(self, ctx: EvalContext) -> float:
        """Return a fraction in [0, 1]."""
