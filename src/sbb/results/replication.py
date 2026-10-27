"""Replication records: a paper's setting and numbers, and ours (docs/PLAN.md 7.9 and 12)."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

from sbb.core.registry import ATTACKS, DEFENSES
from sbb.core.result import CellResult
from sbb.core.scenario import Cell, Scenario
from sbb.utils.errors import SbbError

Status = Literal["pending", "replicated", "explained", "failed"]
# The benchmark rule for every replication, in percentage points (user decision of 2026-10-22).
REPLICATION_TOLERANCE_PP = 3.0


class Stat(BaseModel, frozen=True):
    """Mean and population standard deviation over the seeds, in percent."""

    mean: float
    std: float


class Target(BaseModel, frozen=True):
    """One setting of the paper and the numbers it reports there.

    Attributes:
        setting: The paper's scenario.
        reported: Metric name to the paper's value in percent. The papers' ASR is asr_all.
        ours: The same metrics from our runs, filled by sbb replicate.
    """

    model_config = ConfigDict(extra="forbid")
    setting: Scenario
    reported: dict[str, float]
    ours: dict[str, Stat] | None = None


class ReplicationRecord(BaseModel, frozen=True):
    """The replication of one attack port.

    Attributes:
        method: Registry name of the attack.
        source: Paper, table and rows of the targets.
        confirmed: Whether a maintainer confirmed the targets.
        hparams: The paper's hyperparameters, merged over the attack card's.
        seeds: Seeds of every target.
        tolerance_pp: Largest allowed gap between our mean and the paper, in percentage points.
            3 for every record, the benchmark rule (user decision of 2026-10-22).
        targets: The paper's settings and numbers.
        status: replicated when every metric of every target is within the tolerance. A
            maintainer sets explained, with the explanation, for a documented miss.
        explanation: Why the port misses the paper, if it does.
    """

    model_config = ConfigDict(extra="forbid")
    method: str
    source: str
    confirmed: bool = False
    hparams: dict[str, Any] = {}
    seeds: list[int] = [0, 1, 2]
    tolerance_pp: float = REPLICATION_TOLERANCE_PP
    targets: list[Target]
    status: Status = "pending"
    explanation: str | None = None

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read a record from YAML."""
        return cls.model_validate(yaml.safe_load(path.read_text()))

    def save(self, path: Path) -> None:
        """Write the record as YAML."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))

    def cells(self) -> list[Cell]:
        """One undefended cell per target and seed, in that order."""
        attack, defense = ATTACKS.get(self.method), DEFENSES.get("no_defense")
        return [
            Cell(
                benchmark_version="replication",
                track="replication",
                scenario=target.setting,
                attack=self.method,
                attack_version=attack.card.version,
                defense="no_defense",
                defense_version=defense.card.version,
                seed=seed,
                attack_hparams=self.hparams,
            )
            for target in self.targets
            for seed in self.seeds
        ]

    def with_results(self, results: Sequence[CellResult | None]) -> Self:
        """The record with ours from the results of cells(), in the same order, and its status."""
        per_target = len(self.seeds)
        targets = []
        for t, target in enumerate(self.targets):
            runs = results[t * per_target : (t + 1) * per_target]
            for seed, result in zip(self.seeds, runs, strict=True):
                if result is None or result.outcome != "ok":
                    reason = "no result" if result is None else result.reason or result.outcome
                    raise SbbError(f"{target.setting.name} seed {seed}: {reason}")
            ours = {}
            for metric in target.reported:
                values = 100 * np.array([r.metrics[metric] for r in runs if r is not None])
                ours[metric] = Stat(mean=float(values.mean()), std=float(values.std()))
            targets.append(target.model_copy(update={"ours": ours}))
        return self.model_copy(update={"targets": targets}).judged()

    def judged(self) -> Self:
        """The record with the status its numbers give under its tolerance.

        replicated when every reported metric of every target is within tolerance_pp of our mean,
        otherwise failed, or explained if a maintainer explained the miss. A record without our
        numbers stays as it is.
        """
        if any(target.ours is None for target in self.targets):
            return self
        within = all(
            abs(target.ours[metric].mean - reported) <= self.tolerance_pp
            for target in self.targets
            if target.ours is not None
            for metric, reported in target.reported.items()
        )
        status: Status = "replicated" if within else "failed"
        if not within and self.status == "explained":
            status = "explained"
        return self.model_copy(update={"status": status})
