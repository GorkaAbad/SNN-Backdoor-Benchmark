"""Verification records: a cell rerun elsewhere, compared with its result (docs/PLAN.md 7.8)."""

from datetime import UTC
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

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
    match = (
        rerun.outcome == original.outcome
        and delta.keys() == original.metrics.keys()
        and all(abs(d) <= tolerance_pp / 100 for d in delta.values())
    )
    return Verification(
        result_key=original.key,
        verifier=verifier,
        verifier_org=verifier_org,
        provenance=rerun.provenance,
        metrics=rerun.metrics,
        delta=delta,
        verdict="match" if match else "mismatch",
        notes=None if rerun.outcome == original.outcome else f"outcome {rerun.outcome}",
    )
