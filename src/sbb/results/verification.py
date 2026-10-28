"""Verification records: a cell rerun elsewhere, compared with its result (docs/PLAN.md 7.8)."""

from collections.abc import Mapping
from datetime import UTC
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

import sbb.metrics  # noqa: F401  registers the metrics, whose unit decides how a delta is judged
from sbb.core.registry import METRICS
from sbb.core.result import CellResult, Provenance

# The verifier name of the verify-lite workflow. Its records set the status of lite cells.
LITE_VERIFIER = "github-actions"


class Verification(BaseModel, frozen=True):
    """One independent rerun of a cell.

    Attributes:
        result_key: Key of the checked result.
        verifier: GitHub handle of the verifier.
        verifier_org: Organization of the verifier.
        provenance: Provenance of the rerun.
        metrics: Metrics of the rerun.
        delta: Rerun minus original, per metric of the original.
        verdict: match when the outcomes agree and every metric is within the tolerance.
        notes: Free text.
    """

    model_config = ConfigDict(extra="forbid")
    result_key: str
    verifier: str
    verifier_org: str
    provenance: Provenance
    metrics: dict[str, float]
    delta: dict[str, float]
    verdict: Literal["match", "mismatch"]
    notes: str | None = None

    def judged(self, original: CellResult, tolerance_pp: float) -> "Verification":
        """The record with the verdict its deltas give under tolerance_pp.

        A rerun that ended with another outcome, noted as "outcome <outcome>", stays a mismatch.
        """
        verdict = "match" if matches(original, self.delta, tolerance_pp, self.notes) else "mismatch"
        return self.model_copy(update={"verdict": verdict})

    def covers(self, result: CellResult) -> bool:
        """Whether this record checked result's numbers: rerun minus delta gives each metric back.

        A record stays tied to the numbers it checked, so a result changed after its verification
        (a rerun keeps its key) is not covered by the old record.
        """
        return all(
            abs(self.metrics[m] - self.delta.get(m, 0.0) - value) <= 1e-9
            for m, value in result.metrics.items()
            if m in self.metrics
        )

    def path(self, root: Path, original: CellResult) -> Path:
        """<root>/<benchmark version>/<cell key>/<verifier>-<UTC time of the rerun>.json."""
        stamp = self.provenance.started_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        version = original.cell.benchmark_version
        return root / version / self.result_key / f"{self.verifier}-{stamp}.json"


def verify(
    original: CellResult,
    rerun: CellResult,
    verifier: str,
    verifier_org: str,
    tolerance_pp: float,
) -> Verification:
    """Compare a rerun with the original result.

    A match needs the same outcome, every metric of the original in the rerun, and every
    difference at most tolerance_pp percentage points.
    """
    delta = {
        name: rerun.metrics[name] - value
        for name, value in original.metrics.items()
        if name in rerun.metrics
    }
    notes = None if rerun.outcome == original.outcome else f"outcome {rerun.outcome}"
    match = matches(original, delta, tolerance_pp, notes)
    return Verification(
        result_key=original.key,
        verifier=verifier,
        verifier_org=verifier_org,
        provenance=rerun.provenance,
        metrics=rerun.metrics,
        delta=delta,
        verdict="match" if match else "mismatch",
        notes=notes,
    )


def matches(
    original: CellResult,
    delta: Mapping[str, float],
    tolerance_pp: float | Mapping[str, float],
    notes: str | None = None,
) -> bool:
    """Same outcome, a delta for every metric of the original, each within its tolerance.

    tolerance_pp is one value for every metric, or one per metric. A rerun that ended with
    another outcome is noted as "outcome <outcome>".
    """
    prefix = "outcome "
    outcome = notes[len(prefix) :] if notes and notes.startswith(prefix) else original.outcome
    if not isinstance(tolerance_pp, Mapping):
        tolerance_pp = dict.fromkeys(delta, tolerance_pp)
    return (
        outcome == original.outcome
        and delta.keys() == original.metrics.keys()
        and all(
            abs(d) <= tolerance_pp[m] / 100 * (abs(original.metrics[m]) if is_count(m) else 1)
            for m, d in delta.items()
        )
    )


def is_count(metric: str) -> bool:
    """Whether metric is a raw count (Metric.unit), whose tolerance is a percentage of its value.

    A fraction such as CA may move by tolerance_pp percentage points. A count such as synops may
    move by tolerance_pp percent of the original, as one spike more changes it by far more than
    any fixed amount.
    """
    return metric in METRICS.names() and METRICS.get(metric).unit == "count"
