"""Verification records: a cell rerun elsewhere, compared with its result (docs/PLAN.md 7.8)."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC
from pathlib import Path

from pydantic import BaseModel

import sbb.metrics  # noqa: F401  registers the metrics, whose unit decides how a delta is judged
from sbb.core.registry import METRICS
from sbb.core.result import CellResult, Outcome, Provenance

# The verifier name of the verify-lite workflow. Its records set the status of lite cells.
LITE_VERIFIER = "github-actions"


class Verification(BaseModel, frozen=True, extra="forbid"):
    """One independent rerun of a cell. It holds facts only: what it checked and what it got.

    Whether the rerun matches is judged when it is read, at the tolerance of the time
    (matches, VerificationPolicy), so no stored verdict can contradict the leaderboard.

    Attributes:
        result_key: Key of the checked result.
        verifier: GitHub handle of the verifier.
        verifier_org: Organization of the verifier.
        provenance: Provenance of the rerun.
        checked: The metrics of the result when it was checked. A result rerun later under the
            same key with other numbers is not covered by this record.
        outcome: Outcome of the rerun.
        metrics: Metrics of the rerun.
        notes: Free text for people. No code reads it.
    """

    result_key: str
    verifier: str
    verifier_org: str
    provenance: Provenance
    checked: dict[str, float]
    outcome: Outcome
    metrics: dict[str, float]
    notes: str | None = None

    @property
    def delta(self) -> dict[str, float]:
        """Rerun minus checked, for each checked metric the rerun reports."""
        return {m: self.metrics[m] - v for m, v in self.checked.items() if m in self.metrics}

    def covers(self, result: CellResult) -> bool:
        """Whether this record checked result's numbers, the ones result has now."""
        return self.result_key == result.key and self.checked == result.metrics

    def path(self, root: Path, original: CellResult) -> Path:
        """<root>/<benchmark version>/<cell key>/<verifier>-<UTC time of the rerun>.json."""
        stamp = self.provenance.started_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        version = original.cell.benchmark_version
        return root / version / self.result_key / f"{self.verifier}-{stamp}.json"


def verify(
    original: CellResult, rerun: CellResult, verifier: str, verifier_org: str
) -> Verification:
    """The record of rerun, a rerun of original's cell."""
    return Verification(
        result_key=original.key,
        verifier=verifier,
        verifier_org=verifier_org,
        provenance=rerun.provenance,
        checked=original.metrics,
        outcome=rerun.outcome,
        metrics=rerun.metrics,
    )


def matches(
    original: CellResult, check: Verification, tolerance_pp: float | Mapping[str, float]
) -> bool:
    """Same outcome, every metric of original rerun, each within its tolerance.

    tolerance_pp is one value for every metric, or one per metric.
    """
    delta = check.delta
    if not isinstance(tolerance_pp, Mapping):
        tolerance_pp = dict.fromkeys(delta, tolerance_pp)
    return (
        check.outcome == original.outcome
        and delta.keys() == original.metrics.keys()
        and all(
            abs(d) <= tolerance_pp[m] / 100 * (abs(original.metrics[m]) if is_count(m) else 1)
            for m, d in delta.items()
        )
    )


def unverified(
    results: Sequence[CellResult], checks: Sequence[Verification], verifier: str
) -> list[CellResult]:
    """The results that no record of verifier covers, in their order."""
    mine: dict[str, list[Verification]] = defaultdict(list)
    for c in checks:
        if c.verifier == verifier:
            mine[c.result_key].append(c)
    return [r for r in results if not any(c.covers(r) for c in mine[r.key])]


def is_count(metric: str) -> bool:
    """Whether metric is a raw count (Metric.unit), whose tolerance is a percentage of its value.

    A fraction such as CA may move by tolerance_pp percentage points. A count such as synops may
    move by tolerance_pp percent of the original, as one spike more changes it by far more than
    any fixed amount.
    """
    return metric in METRICS.names() and METRICS.get(metric).unit == "count"
