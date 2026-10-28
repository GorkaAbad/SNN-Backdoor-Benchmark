from datetime import UTC, datetime

import pytest
import torch

from sbb.core.result import CellResult, collect_provenance
from sbb.results.verification import verify
from sbb.runner.planner import Planner
from tests.unit.test_runner import SMOKE

PROVENANCE = collect_provenance(torch.device("cpu"), "test", "test", datetime.now(UTC), 0.0)
CELL = Planner(SMOKE).cells(attack="static_patch", defense="no_defense")[0]


def _result(outcome: str = "ok", **metrics: float) -> CellResult:
    return CellResult(
        key=CELL.key(),
        cell=CELL,
        outcome=outcome,
        metrics=metrics,
        provenance=PROVENANCE,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("rerun", "verdict"),
    [
        (_result(ca=0.985, asr=0.995), "match"),  # within 1 pp
        (_result(ca=0.97, asr=1.0), "mismatch"),  # CA moved 2 pp
        (_result("failed"), "mismatch"),  # another outcome
        (_result(ca=0.98), "mismatch"),  # a metric is missing
    ],
)
def test_verdict_follows_the_tolerance(rerun: CellResult, verdict: str) -> None:
    check = verify(_result(ca=0.99, asr=1.0), rerun, "alice", "lab", tolerance_pp=1.0)
    assert check.verdict == verdict and check.result_key == CELL.key()
    if verdict == "match":
        assert check.delta == pytest.approx({"ca": -0.005, "asr": -0.005})


@pytest.mark.parametrize(("synops", "verdict"), [(1.02e6, "match"), (1.05e6, "mismatch")])
def test_a_count_metric_is_judged_relative_to_its_value(synops: float, verdict: str) -> None:
    # 3 pp of a fraction is 0.03 absolute, of a count 3 percent of its value.
    original = _result(ca=0.99, synops=1.0e6)
    check = verify(original, _result(ca=0.99, synops=synops), "alice", "lab", tolerance_pp=3.0)
    assert check.verdict == verdict


def test_a_record_is_rejudged_from_its_deltas() -> None:
    original = _result(ca=0.99, asr=1.0)
    moved = verify(original, _result(ca=0.97, asr=1.0), "alice", "lab", tolerance_pp=1.0)
    assert moved.verdict == "mismatch"
    assert moved.judged(original, tolerance_pp=3.0).verdict == "match"  # CA moved 2 pp
    other = verify(original, _result("failed"), "alice", "lab", tolerance_pp=1.0)
    assert other.judged(original, tolerance_pp=100.0).verdict == "mismatch"  # another outcome
