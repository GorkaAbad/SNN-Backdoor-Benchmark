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


class Trigger(ABC):
    """A transformation of raw events that activates the backdoor (trigger_domain raw_events)."""

    def fit(self, view: AttackerView) -> None:  # noqa: B027  optional hook
        """Learn the trigger. Only learned triggers override this."""

    @abstractmethod
    def apply(self, sample: EventSample, rng: np.random.Generator) -> EventSample:
        """Return the triggered sample with the same label and index."""


class FrameTrigger(ABC):
    """A transformation of the frames of one encoded sample (trigger_domain frames).

    It acts after the representation, so it assumes access to the input after preprocessing
    (docs/PLAN.md section 3.2).
    """

    def fit(self, view: AttackerView) -> None:  # noqa: B027  optional hook
        """Learn the trigger. Only learned or data-dependent triggers override this."""

    @abstractmethod
    def apply(self, frames: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Return triggered frames [T, P, H, W] and leave frames unchanged."""


class Attack(ABC):
    """Base of every attack. Hyperparameters come from the card.

    Attributes:
        trigger: The trigger of the test inputs, set in __init__. A Trigger or a FrameTrigger, as
            the card's trigger_domain says.
    """

    card: ClassVar[AttackCard]
    trigger: Trigger | FrameTrigger

    def __init__(self, hparams: dict[str, Any], target: int) -> None:
        self.hparams = hparams
        self.target = target


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

    selector: Selector
    labels: LabelPolicy

    def poison(self, view: AttackerView, budget: Budget, pool: np.ndarray) -> PoisonSet:
        """Select samples outside the defender pool, relabel them and trigger them.

        The selection comes before Trigger.fit, so fitting a trigger does not change which samples
        are poisoned. A raw-event trigger is applied here. A frame trigger travels with the poison
        set, and the DataModule applies it to the frames of the poisoned samples.
        """
        train = view.train_set
        labels = train.labels()
        excluded = np.flatnonzero(np.isin(train.indices, pool))
        n = budget.max_poison(len(train))
        positions = self.selector.select(labels, n, self.target, excluded, view.rng)
        self.trigger.fit(view)
        indices = train.indices[positions]
        new_labels = np.array(
            [self.labels.relabel(int(labels[p]), self.target) for p in positions], dtype=np.int64
        )
        if isinstance(self.trigger, FrameTrigger):
            samples = tuple(train[int(p)] for p in positions)
            return PoisonSet(indices, samples, new_labels, frame_trigger=self.trigger.apply)
        samples = tuple(
            self.trigger.apply(train[int(p)], rng_for(view.seed, int(i)))
            for p, i in zip(positions, indices, strict=True)
        )
        return PoisonSet(indices, samples, new_labels)


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
    """Predictions on the clean and triggered test sets.

    With an input defense, scores are its per-sample scores (higher is more suspicious) and the
    accepted masks mark the samples at or below its threshold. Without one, every sample is
    accepted and the scores are None.
    """

    clean_pred: np.ndarray
    clean_true: np.ndarray
    triggered_pred: np.ndarray
    triggered_true: np.ndarray
    target: int
    clean_accepted: np.ndarray | None = None
    triggered_accepted: np.ndarray | None = None
    clean_scores: np.ndarray | None = None
    triggered_scores: np.ndarray | None = None

    @property
    def clean_ok(self) -> np.ndarray:
        """Mask of the accepted clean samples."""
        return _all(self.clean_pred) if self.clean_accepted is None else self.clean_accepted

    @property
    def triggered_ok(self) -> np.ndarray:
        """Mask of the accepted triggered samples."""
        accepted = self.triggered_accepted
        return _all(self.triggered_pred) if accepted is None else accepted


def _all(predictions: np.ndarray) -> np.ndarray:
    return np.ones(len(predictions), dtype=bool)


class Metric(ABC):
    """A metric computed from an EvalContext. name equals the registry name.

    stages lists the defense stages whose cells report the metric.
    """

    name: ClassVar[str]
    stages: ClassVar[frozenset[Stage]] = frozenset(Stage)

    @abstractmethod
    def compute(self, ctx: EvalContext) -> float:
        """Return a fraction in [0, 1]."""
