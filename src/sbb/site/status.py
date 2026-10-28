"""The run status of a benchmark: one row per planned cell, done or missing, with its metrics."""

import csv
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sbb.core.cards import compatible
from sbb.core.registry import ATTACKS, DEFENSES
from sbb.core.result import CellResult
from sbb.core.scenario import Cell
from sbb.results.policy import VerificationPolicy
from sbb.results.verification import Verification
from sbb.runner.spec import Tolerance

# Labels and direction of the metrics the site shows. A metric not listed keeps its name.
METRICS = {
    "ca": ("Clean accuracy", "higher"),
    "asr": ("Attack success rate", "lower"),
    "asr_all": ("ASR, target class included", "lower"),
    "ra": ("Robust accuracy", "higher"),
    "frr": ("False rejection rate", "lower"),
    "tpr": ("Rejection of triggered inputs", "higher"),
    "auroc": ("Input AUROC", "higher"),
    "verdict": ("Detection verdict", "none"),
    "spike_rate": ("Spike rate, clean inputs", "none"),
    "spike_rate_triggered": ("Spike rate, triggered inputs", "none"),
    "synops": ("Synaptic operations per sample", "none"),
}
METRIC_COLUMNS = tuple(METRICS)
RAW_METRICS = frozenset({"verdict", "synops"})  # shown as they are, not in percent
COLUMNS = (
    "version",
    "track",
    "scenario",
    "dataset",
    "arch",
    "attack",
    "defense",
    "seed",
    "state",
    "verification",
    *METRIC_COLUMNS,
    "site",
    "device",
    "minutes",
    "reason",
    "key",
)


def percent(name: str, value: float) -> float:
    """A metric as the CSVs show it: in percent, except the raw verdict and synops counts."""
    return round(value if name in RAW_METRICS else 100 * value, 4)


def status_rows(
    cells: Sequence[Cell],
    results: Sequence[CellResult],
    checks: Sequence[Verification],
    policy: VerificationPolicy | None = None,
) -> list[dict[str, Any]]:
    """One row per cell, in plan order.

    state is the result's outcome. A cell without a result is missing, or not_applicable with
    the reason when the threat models do not fit, since such a cell never trains anything.
    verification is the VerificationPolicy's status of a result that ended ok. Metrics are in
    percent, except the detection verdict, a raw score.
    """
    policy = policy or VerificationPolicy(Tolerance())
    by_key = {r.key: r for r in results}
    seeds: dict[str, list[CellResult]] = defaultdict(list)  # a cell's results over its seeds
    for r in results:
        if r.outcome == "ok":
            seeds[_unseeded(r.cell)].append(r)
    checks_of: dict[str, list[Verification]] = defaultdict(list)
    for check in checks:
        checks_of[check.result_key].append(check)
    rows = []
    for cell in cells:
        key = cell.key()
        result = by_key.get(key)
        row: dict[str, Any] = {
            "version": cell.benchmark_version,
            "track": cell.track,
            "scenario": cell.scenario.name,
            "dataset": cell.scenario.data.name,
            "arch": cell.scenario.arch.name,
            "attack": cell.attack,
            "defense": cell.defense,
            "seed": cell.seed,
            "state": result.outcome if result else "missing",
            "verification": "",
            "key": key,
        }
        if result:
            if result.outcome == "ok":
                own = checks_of[key]
                row["verification"] = str(policy.status(result, own, seeds[_unseeded(cell)]))
            for name, value in result.metrics.items():
                if name in METRIC_COLUMNS:
                    row[name] = percent(name, value)
            row["site"] = result.provenance.site
            row["device"] = result.provenance.device
            row["minutes"] = round(result.provenance.duration_s / 60, 1)
            row["reason"] = result.reason or ""
        else:
            attack, defense = ATTACKS.get(cell.attack).card, DEFENSES.get(cell.defense).card
            ok, reasons = compatible(attack, defense, cell.scenario.budget)
            if not ok:
                row["state"], row["reason"] = "not_applicable", "; ".join(reasons)
        rows.append(row)
    return rows


def _unseeded(cell: Cell) -> str:
    """The key of a cell without its seed, shared by the cell's results over its seeds."""
    return cell.model_copy(update={"seed": 0}).key()


def summary(rows: Sequence[dict[str, Any]]) -> dict[str, Counter[str]]:
    """Per track, how many cells are in each state."""
    counts: dict[str, Counter[str]] = {}
    for row in rows:
        counts.setdefault(row["track"], Counter())[row["state"]] += 1
    return counts


def write_csv(rows: Sequence[dict[str, Any]], path: Path, columns: Sequence[str] = COLUMNS) -> None:
    """Write the rows with every column, empty where a row has no value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
