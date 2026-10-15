"""Markdown tables of results, one per (track, scenario)."""

from collections import defaultdict
from collections.abc import Sequence
from statistics import mean

from sbb.core.result import CellResult

_METRICS = ("ca", "asr", "ra")


def _entry(results: Sequence[CellResult]) -> str:
    ok = [r for r in results if r.outcome == "ok"]
    if not ok:
        outcome = results[0].outcome
        return "N/A" if outcome == "not_applicable" else outcome
    values = " · ".join(f"{m.upper()} {100 * mean(r.metrics[m] for r in ok):.1f}" for m in _METRICS)
    return f"{values} (n={len(ok)})"


def markdown_table(results: Sequence[CellResult]) -> str:
    """Render results as attack-by-defense tables in percent, averaged over seeds."""
    groups: dict[tuple[str, str], dict[tuple[str, str], list[CellResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in results:
        groups[r.cell.track, r.cell.scenario.name][r.cell.attack, r.cell.defense].append(r)
    sections = []
    for (track, scenario), cells in sorted(groups.items()):
        attacks = sorted({a for a, _ in cells})
        defenses = sorted({d for _, d in cells})
        lines = [
            f"## {track} / {scenario}",
            "",
            "| attack \\ defense | " + " | ".join(defenses) + " |",
            "|---" * (len(defenses) + 1) + "|",
        ]
        for a in attacks:
            row = [_entry(cells[a, d]) if (a, d) in cells else "" for d in defenses]
            lines.append(f"| {a} | " + " | ".join(row) + " |")
        notes = [
            f"- {a} × {d}: {r.reason}"
            for (a, d), rs in sorted(cells.items())
            for r in rs[:1]
            if r.outcome != "ok" and r.reason
        ]
        sections.append("\n".join(lines + ([""] + notes if notes else [])))
    return "\n\n".join(sections) + "\n"
