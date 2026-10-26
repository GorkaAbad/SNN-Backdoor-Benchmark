"""VerificationPolicy: the statuses of results, computed from verification records (PLAN 10)."""

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

import numpy as np

from sbb.core.result import CellResult
from sbb.core.scenario import Cell
from sbb.results.verification import LITE_VERIFIER, Verification, matches


class Status(StrEnum):
    """What the leaderboard shows for a result or a cell."""

    SELF_REPORTED = "self-reported"
    REPRODUCED = "reproduced"
    DISPUTED = "disputed"
    NOT_APPLICABLE = "not-applicable"
    INVALID_ATTACK = "invalid-attack"
    FAILED = "failed"


# The status of a result that did not end ok, the most serious first.
OUTCOME_STATUS = {
    "failed": Status.FAILED,
    "invalid_attack": Status.INVALID_ATTACK,
    "not_applicable": Status.NOT_APPLICABLE,
}


class Tolerances(Protocol):
    """Reproduction tolerances in percentage points, such as the benchmark spec's Tolerance."""

    lite_pp: float
    full_pp: float
    full_sigma: float


class VerificationPolicy:
    """Computes statuses. Nobody sets them by hand.

    A record counts for a result only while the result still has the numbers it checked, and
    only if it is independent: on the lite track a verify-lite rerun, otherwise a verifier of
    another organization than the submitter's. The policy judges every record from its deltas
    at its own tolerance: lite_pp on the lite track, otherwise max(full_pp, full_sigma x seed
    std) per metric.

    Args:
        tolerance: The tolerances.
        k_samples: Spot checks per submission.
    """

    def __init__(self, tolerance: Tolerances, k_samples: int = 3) -> None:
        self.tolerance = tolerance
        self.k_samples = k_samples

    def sample(self, cells: Sequence[Cell], seed: int) -> list[Cell]:
        """Pick k_samples cells, the same ones for the same seed whatever the order of cells.

        The seed is the manifest's bundle_id read as a hexadecimal number, so anyone can
        recompute the choice.
        """
        ordered = sorted(cells, key=lambda c: c.key())
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(ordered), size=min(self.k_samples, len(ordered)), replace=False)
        return [ordered[i] for i in sorted(picks)]

    def is_independent(self, v: Verification, r: CellResult) -> bool:
        """On the lite track a verify-lite rerun, otherwise a verifier of another organization."""
        if r.cell.track == "lite":
            return v.verifier == LITE_VERIFIER
        return v.verifier_org != r.provenance.submitter_org

    def status(
        self, r: CellResult, checks: Sequence[Verification], seeds: Sequence[CellResult] = ()
    ) -> Status:
        """Status of one result. seeds are the cell's results over its seeds, for the seed std.

        Reproduced with at least one independent match, disputed while the independent
        mismatches are at least as many as the matches.
        """
        if r.outcome != "ok":
            return OUTCOME_STATUS[r.outcome]
        tolerance = self._tolerance(r, seeds)
        verdicts = [
            matches(r, v.delta, tolerance, v.notes)
            for v in checks
            if v.result_key == r.key and self.is_independent(v, r) and _checked(r, v)
        ]
        hits = sum(verdicts)
        if len(verdicts) - hits >= max(hits, 1):
            return Status.DISPUTED
        return Status.REPRODUCED if hits else Status.SELF_REPORTED

    def cell_status(self, results: Sequence[CellResult], checks: Sequence[Verification]) -> Status:
        """Status of a cell over its seeds, judged on the seeds that ended ok.

        Disputed if one seed is. A lite cell is reproduced when every seed is, since verify-lite
        reruns them all, a full cell when a spot check reproduced one. With no seed ok, the most
        serious outcome.
        """
        valid = [r for r in results if r.outcome == "ok"]
        if not valid:
            return min(
                (OUTCOME_STATUS[r.outcome] for r in results),
                key=list(OUTCOME_STATUS.values()).index,
            )
        statuses = [self.status(r, checks, valid) for r in valid]
        if Status.DISPUTED in statuses:
            return Status.DISPUTED
        lite = valid[0].cell.track == "lite"
        done = all if lite else any
        return (
            Status.REPRODUCED
            if done(s == Status.REPRODUCED for s in statuses)
            else (Status.SELF_REPORTED)
        )

    def _tolerance(self, r: CellResult, seeds: Sequence[CellResult]) -> dict[str, float]:
        """Per metric, in percentage points."""
        t = self.tolerance
        if r.cell.track == "lite":
            return {m: t.lite_pp for m in r.metrics}
        values = {m: [s.metrics[m] for s in seeds if m in s.metrics] for m in r.metrics}
        return {
            m: max(t.full_pp, t.full_sigma * 100 * float(np.std(v or [0.0])))
            for m, v in values.items()
        }


def _checked(result: CellResult, check: Verification) -> bool:
    """Whether check verified this result's numbers: rerun minus delta gives each metric back.

    A record stays tied to the numbers it checked, so a result changed after its verification
    (a rerun keeps its key) is not covered by the old record.
    """
    return all(
        abs(check.metrics[m] - check.delta.get(m, 0.0) - value) <= 1e-9
        for m, value in result.metrics.items()
        if m in check.metrics
    )
