from datetime import UTC, datetime

import torch

from sbb.core.result import CellResult, Outcome, collect_provenance
from sbb.results.policy import Status, VerificationPolicy
from sbb.results.verification import LITE_VERIFIER, Verification
from sbb.runner.planner import Planner
from sbb.runner.spec import Tolerance
from tests.unit.test_runner import SMOKE

PROVENANCE = collect_provenance(torch.device("cpu"), "test", "test", datetime.now(UTC), 0.0)
POLICY = VerificationPolicy(Tolerance(lite_pp=3.0, full_pp=2.0, full_sigma=2.0))


def _result(seed: int = 0, track: str = "full", outcome: Outcome = "ok", **m: float) -> CellResult:
    cell = Planner(SMOKE).cells(attack="static_patch", defense="no_defense")[0]
    cell = cell.model_copy(update={"seed": seed, "track": track})
    provenance = PROVENANCE.model_copy(update={"submitter_org": "lab"})
    return CellResult(key=cell.key(), cell=cell, outcome=outcome, metrics=m, provenance=provenance)


def _check(result: CellResult, org: str = "elsewhere", moved_pp: float = 0.0) -> Verification:
    """A rerun whose every metric moved by moved_pp. The stored verdict is left wrong on purpose:
    the policy judges the deltas itself."""
    delta = {m: moved_pp / 100 for m in result.metrics}
    return Verification(
        result_key=result.key,
        verifier=LITE_VERIFIER if org == LITE_VERIFIER else "alice",
        verifier_org=org,
        provenance=PROVENANCE,
        metrics={m: v + delta[m] for m, v in result.metrics.items()},
        delta=delta,
        verdict="mismatch" if moved_pp == 0 else "match",
    )


def test_a_seeded_mismatch_is_disputed() -> None:
    r = _result(ca=0.9, asr=1.0)
    assert POLICY.status(r, [_check(r, moved_pp=10.0)]) == Status.DISPUTED


def test_full_results_need_an_independent_match() -> None:
    r = _result(ca=0.9, asr=0.95)
    assert POLICY.status(r, []) == Status.SELF_REPORTED
    assert POLICY.status(r, [_check(r, org="lab")]) == Status.SELF_REPORTED  # own organization
    assert POLICY.status(r, [_check(r)]) == Status.REPRODUCED
    tie = [_check(r), _check(r, moved_pp=10.0)]
    assert POLICY.status(r, tie) == Status.DISPUTED  # as many mismatches as matches
    assert POLICY.status(r, [*tie, _check(r)]) == Status.REPRODUCED


def test_the_full_tolerance_widens_with_the_seed_std() -> None:
    seeds = [_result(0, ca=0.90), _result(1, ca=0.92), _result(2, ca=0.94)]  # std 1.63 pp
    r = seeds[0]
    assert POLICY.status(r, [_check(r, moved_pp=3.0)]) == Status.DISPUTED  # 3 pp > 2 pp
    assert POLICY.status(r, [_check(r, moved_pp=3.0)], seeds) == Status.REPRODUCED  # < 3.27 pp


def test_lite_results_count_the_lite_verifier() -> None:
    r = _result(track="lite", ca=0.9)
    assert POLICY.status(r, [_check(r, org=LITE_VERIFIER, moved_pp=2.5)]) == Status.REPRODUCED
    assert POLICY.status(r, [_check(r, org=LITE_VERIFIER, moved_pp=3.5)]) == Status.DISPUTED


def test_a_record_counts_only_for_the_numbers_it_checked() -> None:
    r = _result(ca=0.9)
    stale = _check(r).model_copy(update={"metrics": {"ca": 0.8}})  # the result changed since
    assert POLICY.status(r, [stale]) == Status.SELF_REPORTED


def test_a_cell_combines_its_seeds() -> None:
    full = [_result(s, ca=0.9) for s in range(3)]
    assert POLICY.cell_status(full, [_check(full[1])]) == Status.REPRODUCED  # a spot check
    disputed = [_check(full[1]), _check(full[2], moved_pp=10.0)]
    assert POLICY.cell_status(full, disputed) == Status.DISPUTED
    lite = [_result(s, track="lite", ca=0.9) for s in range(2)]
    one = [_check(lite[0], org=LITE_VERIFIER)]
    assert POLICY.cell_status(lite, one) == Status.SELF_REPORTED  # verify-lite checks every seed
    both = [*one, _check(lite[1], org=LITE_VERIFIER)]
    assert POLICY.cell_status(lite, both) == Status.REPRODUCED
    invalid = [_result(s, outcome="invalid_attack") for s in range(2)]
    assert POLICY.cell_status(invalid, []) == Status.INVALID_ATTACK


def test_spot_checks_are_stable_and_capped() -> None:
    cells = Planner(SMOKE).cells()
    picked = VerificationPolicy(Tolerance(), k_samples=3).sample(cells, seed=0xABC)
    assert len(picked) == 3 and len({c.key() for c in picked}) == 3
    assert picked == VerificationPolicy(Tolerance(), k_samples=3).sample(cells[::-1], seed=0xABC)
    assert len(VerificationPolicy(Tolerance(), k_samples=99).sample(cells, 1)) == len(cells)
