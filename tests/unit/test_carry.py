"""Results of an earlier benchmark version carried over to identical cells of a new one."""

from sbb.results.carry import carry, carry_check
from tests.unit.test_site import RESULTS, _check


def test_a_result_is_carried_to_the_identical_cell_of_a_new_version() -> None:
    old = RESULTS[0]
    cell = old.cell.model_copy(update={"benchmark_version": "9.9.9"})
    new = carry(old, cell)
    assert new is not None and new.key == cell.key() and new.cell == cell
    assert new.metrics == old.metrics and new.provenance.carried_from == old.key
    changed = cell.model_copy(update={"attack_version": "2.0.0"})
    assert carry(old, changed) is None  # a changed method reruns
    assert carry(old.model_copy(update={"outcome": "failed"}), cell) is None  # so does a failure


def test_a_record_follows_its_carried_result() -> None:
    old = RESULTS[0]
    new = carry(old, old.cell.model_copy(update={"benchmark_version": "9.9.9"}))
    assert new is not None
    moved = carry_check(_check(old, "match"), new)
    assert moved.result_key == new.key and moved.covers(new) and moved.verdict == "match"
