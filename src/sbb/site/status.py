"""The run status of a benchmark: one row per planned cell, done or missing, with its metrics."""

import csv
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sbb.core.cards import compatible
from sbb.core.registry import ATTACKS, DEFENSES
from sbb.core.result import CellResult
from sbb.core.scenario import Cell
from sbb.results.verification import LITE_VERIFIER, Verification

METRIC_COLUMNS = ("ca", "asr", "asr_all", "ra", "frr", "tpr", "auroc", "verdict")
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
    """A metric as the CSVs show it: in percent, except the detection verdict, a raw score."""
    return round(value if name == "verdict" else 100 * value, 4)


def status_rows(
    cells: Sequence[Cell], results: Sequence[CellResult], checks: Sequence[Verification]
) -> list[dict[str, Any]]:
    """One row per cell, in plan order.

    state is the result's outcome. A cell without a result is missing, or not_applicable with
    the reason when the threat models do not fit, since such a cell never trains anything.
    verification is the verdict of the verify-lite record, if any. Metrics are in percent,
    except the detection verdict, a raw score.
    """
    by_key = {r.key: r for r in results}
    verdicts = {v.result_key: v.verdict for v in checks if v.verifier == LITE_VERIFIER}
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
            "verification": verdicts.get(key, ""),
            "key": key,
        }
        if result:
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
