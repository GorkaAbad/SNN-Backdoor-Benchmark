from datetime import UTC, datetime
from pathlib import Path

import torch

from sbb.core.result import CellResult, Outcome, collect_provenance
from sbb.results.store import LocalResultStore
from sbb.results.table import markdown_table
from sbb.runner.planner import Planner
from sbb.runner.spec import BenchmarkSpec

SMOKE = BenchmarkSpec.load(Path(__file__).parents[2] / "benchmarks" / "smoke.yaml")
PROVENANCE = collect_provenance(torch.device("cpu"), "test", "test", datetime.now(UTC), 0.0)


def test_planner_filters() -> None:
    planner = Planner(SMOKE)
    assert len(planner.cells()) == 4
    assert len(planner.cells(track="full")) == 0
    only = planner.cells(attack="static_patch", defense="fine_tuning")
    assert [(c.attack, c.defense) for c in only] == [("static_patch", "fine_tuning")]


def _result(attack: str, defense: str, outcome: Outcome, **metrics: float) -> CellResult:
    cell = Planner(SMOKE).cells(attack=attack, defense=defense)[0]
    reason = None if outcome == "ok" else f"{outcome} because"
    return CellResult(
        key=cell.key(),
        cell=cell,
        outcome=outcome,
        reason=reason,
        metrics=metrics,
        provenance=PROVENANCE,
    )


def test_table_shows_metrics_and_reasons() -> None:
    table = markdown_table(
        [
            _result("no_attack", "no_defense", "ok", ca=0.98, asr=0.01, ra=0.97),
            _result("static_patch", "no_defense", "not_applicable"),
            _result("static_patch", "fine_tuning", "failed"),
        ]
    )
    assert "CA 98.0 · ASR 1.0 · RA 97.0 (n=1)" in table
    assert "| static_patch | failed | N/A |" in table
    assert "- static_patch × no_defense: not_applicable because" in table


def test_store_round_trip(tmp_path: Path) -> None:
    store = LocalResultStore(tmp_path)
    result = _result("no_attack", "no_defense", "ok", ca=1.0, asr=0.0, ra=1.0)
    store.save(result)
    assert store.get(result.key) == result and store.get("missing") is None
    assert store.query(attack="no_attack") == [result] and store.query(attack="x") == []
    assert [p.name for p in tmp_path.rglob("*")] == ["smoke-0.1.0", f"{result.key}.json"]
