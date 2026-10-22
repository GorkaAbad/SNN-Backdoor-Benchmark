"""Plugin cards, defender views, the Avail table and compatibility (docs/PLAN.md section 3)."""

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from sbb.data.poison import Budget


class ScenarioKind(StrEnum):
    """What the attacker controls."""

    DATA_POISONING = "data_poisoning"
    TRAINING_CONTROL = "training_control"
    SUPPLY_CHAIN = "supply_chain"
    FEDERATED = "federated"


class Stage(StrEnum):
    """When a defense acts."""

    PRE_TRAINING = "pre_training"
    IN_TRAINING = "in_training"
    POST_TRAINING = "post_training"
    INFERENCE = "inference"
    MODEL_DETECTION = "model_detection"


class Resource(StrEnum):
    """What a defense may see."""

    TRAIN_SET = "train_set"
    TRAINING_LOOP = "training_loop"
    MODEL = "model"
    CLEAN_SUBSET = "clean_subset"


class View(BaseModel, frozen=True):
    """A set of defender resources and a clean-data fraction of the train split."""

    resources: frozenset[Resource]
    clean_fraction: float = 0.0

    @field_serializer("resources")
    def _sorted(self, resources: frozenset[Resource]) -> list[str]:
        return sorted(resources)

    def missing(self, other: "View") -> list[str]:
        """Return readable reasons why this view is not contained in other."""
        reasons = [
            f"needs {r}, which this threat model does not leave to the defender"
            for r in sorted(self.resources - other.resources)
        ]
        if self.clean_fraction > other.clean_fraction:
            reasons.append(
                f"needs a clean fraction of {self.clean_fraction}, "
                f"the budget allows {other.clean_fraction}"
            )
        return reasons

    def issubset(self, other: "View") -> bool:
        """Return True if other provides everything this view needs."""
        return not self.missing(other)


_AVAIL: dict[ScenarioKind, frozenset[Resource]] = {
    ScenarioKind.DATA_POISONING: frozenset(Resource),
    ScenarioKind.TRAINING_CONTROL: frozenset({Resource.MODEL, Resource.CLEAN_SUBSET}),
    ScenarioKind.SUPPLY_CHAIN: frozenset({Resource.MODEL, Resource.CLEAN_SUBSET}),
    ScenarioKind.FEDERATED: frozenset(
        {Resource.TRAINING_LOOP, Resource.MODEL, Resource.CLEAN_SUBSET}
    ),
}


class Capabilities(BaseModel, frozen=True):
    """What the attacker can modify."""

    model_config = ConfigDict(extra="forbid")
    labels: Literal["clean", "dirty"] | None = None
    trigger_domain: Literal["raw_events", "frames"] = "raw_events"


class Knowledge(BaseModel, frozen=True):
    """What the attacker knows."""

    model_config = ConfigDict(extra="forbid")
    architecture: Literal["known", "unknown"] = "unknown"
    data: Literal["full", "partial"] = "full"
    defense: str = "oblivious"


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class CardBase(BaseModel, frozen=True):
    """Fields shared by attack and defense cards."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    title: str
    paper: str | None = None
    # Year and month of the paper's first public version. The leaderboard lists methods oldest
    # first. Baselines without a paper leave it out.
    published: str | None = Field(None, pattern=r"^\d{4}-\d{2}$")
    code_reference: str | None = None
    license_note: str | None = None
    hparams: dict[str, dict[str, Any]] = {}

    def hparams_for(
        self, dataset: str, arch: str, override: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return the default hyperparameters deep-merged with the <dataset>/<arch> ones.

        override, for example the paper's hyperparameters in a replication, is merged last.
        """
        merged = _merge(self.hparams.get("default", {}), self.hparams.get(f"{dataset}/{arch}", {}))
        return _merge(merged, override or {})

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load and validate a YAML card."""
        return cls.model_validate(yaml.safe_load(path.read_text()))


class AttackCard(CardBase, frozen=True):
    """Threat model and hyperparameters of an attack."""

    kind: ScenarioKind
    capabilities: Capabilities
    knowledge: Knowledge

    @model_validator(mode="after")
    def _poisoning_declares_labels(self) -> Self:
        if self.kind == ScenarioKind.DATA_POISONING and self.capabilities.labels is None:
            raise ValueError("a data_poisoning attack must declare capabilities.labels")
        return self

    def avail(self, budget: Budget) -> View:
        """Resources left to the defender against this attack (docs/PLAN.md section 3.4)."""
        return View(resources=_AVAIL[self.kind], clean_fraction=budget.defender_clean_fraction)


# Resources a stage cannot work without (docs/PLAN.md section 3.3, "Required resources").
_STAGE_NEEDS = {
    Stage.PRE_TRAINING: frozenset({Resource.TRAIN_SET}),
    Stage.IN_TRAINING: frozenset({Resource.TRAIN_SET, Resource.TRAINING_LOOP}),
}


class DefenseCard(CardBase, frozen=True):
    """Stage, view and hyperparameters of a defense.

    A pre-training defense must declare train_set and an in-training defense also training_loop,
    because the Game retrains the victim on the train split for them.
    """

    stage: Stage
    view: View
    input_repr: Literal["frames"] = "frames"
    adaptation_note: str | None = None

    @model_validator(mode="after")
    def _declares_what_its_stage_needs(self) -> Self:
        missing = _STAGE_NEEDS.get(self.stage, frozenset()) - self.view.resources
        if missing:
            raise ValueError(f"a {self.stage} defense must declare {sorted(missing)}")
        return self


def compatible(attack: AttackCard, defense: DefenseCard, budget: Budget) -> tuple[bool, list[str]]:
    """Return whether the cell is defined, and the reasons if it is not."""
    reasons = defense.view.missing(attack.avail(budget))
    return not reasons, reasons
