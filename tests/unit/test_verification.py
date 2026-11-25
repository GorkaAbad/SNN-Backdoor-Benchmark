from datetime import UTC, datetime

import pytest
import torch

from sbb.core.result import CellResult, collect_provenance
from sbb.results.verification import LITE_VERIFIER, matches, unverified, verify
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
    ("rerun", "match"),
    [
        (_result(ca=0.985, asr=0.995), True),  # within 1 pp
        (_result(ca=0.97, asr=1.0), False),  # CA moved 2 pp
        (_result("failed"), False),  # another outcome
        (_result(ca=0.98), False),  # a metric is missing
    ],
)
def test_a_match_follows_the_tolerance(rerun: CellResult, match: bool) -> None:
    original = _result(ca=0.99, asr=1.0)
    check = verify(original, rerun, "alice", "lab")
    assert check.result_key == CELL.key() and check.outcome == rerun.outcome
    assert matches(original, check, tolerance_pp=1.0) is match
    if match:
        assert check.delta == pytest.approx({"ca": -0.005, "asr": -0.005})


@pytest.mark.parametrize(("synops", "match"), [(1.02e6, True), (1.05e6, False)])
def test_a_count_metric_is_judged_relative_to_its_value(synops: float, match: bool) -> None:
    # 3 pp of a fraction is 0.03 absolute, of a count 3 percent of its value.
    original = _result(ca=0.99, synops=1.0e6)
    check = verify(original, _result(ca=0.99, synops=synops), "alice", "lab")
    assert matches(original, check, tolerance_pp=3.0) is match


def test_a_record_holds_facts_and_is_judged_at_any_tolerance() -> None:
    original = _result(ca=0.99, asr=1.0)
    moved = verify(original, _result(ca=0.97, asr=1.0), "alice", "lab")
    assert "verdict" not in moved.model_dump() and "delta" not in moved.model_dump()
    assert not matches(original, moved, tolerance_pp=1.0)
    assert matches(original, moved, tolerance_pp=3.0)  # CA moved 2 pp


def test_a_record_covers_only_the_numbers_it_checked() -> None:
    original = _result(ca=0.99, asr=1.0)
    failed = verify(original, _result("failed"), "alice", "lab")
    # A rerun without numbers still names what it checked, so it disputes those numbers...
    assert failed.covers(original) and not matches(original, failed, tolerance_pp=100.0)
    # ...and not a later result under the same key with other numbers.
    rerun_later = original.model_copy(update={"metrics": {"ca": 0.5, "asr": 1.0}})
    assert not failed.covers(rerun_later)
    assert not verify(original, original, "alice", "lab").covers(rerun_later)


def test_unverified_lists_results_without_a_record_of_their_numbers() -> None:
    checked = _result(ca=0.99, asr=1.0)
    changed = checked.model_copy(update={"metrics": {"ca": 0.5, "asr": 1.0}})
    other = changed.model_copy(update={"key": "another"})
    records = [verify(checked, checked, LITE_VERIFIER, LITE_VERIFIER)]
    assert unverified([checked], records, LITE_VERIFIER) == []
    assert unverified([changed, other], records, LITE_VERIFIER) == [changed, other]
    assert unverified([checked], records, "alice") == [checked]  # only the named verifier counts
