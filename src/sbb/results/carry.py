"""Carry results of an earlier benchmark version over to identical cells of a new one."""

from sbb.core.result import CellResult
from sbb.core.scenario import Cell
from sbb.results.verification import Verification


def unversioned(cell: Cell) -> str:
    """The key of cell without its benchmark version: equal for cells that compute the same."""
    return cell.model_copy(update={"benchmark_version": ""}).key()


def carry(result: CellResult, cell: Cell) -> CellResult | None:
    """The result rewritten for cell, or None when cell computes something else or it failed.

    Cells that differ only in the benchmark version give the same numbers, since the scenario,
    the seed and the attack and defense versions are the same. A failed result is not carried,
    so it runs again. The provenance stays that of the run, with carried_from naming the
    original result.
    """
    if result.outcome == "failed" or unversioned(result.cell) != unversioned(cell):
        return None
    provenance = result.provenance.model_copy(update={"carried_from": result.key})
    return result.model_copy(update={"cell": cell, "key": cell.key(), "provenance": provenance})


def carry_check(check: Verification, carried: CellResult) -> Verification:
    """A record of the original result, moved to the result carried from it."""
    return check.model_copy(update={"result_key": carried.key})
