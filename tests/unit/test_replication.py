from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from sbb.core.result import CellResult, collect_provenance
from sbb.core.scenario import Cell
from sbb.results.replication import ReplicationRecord
from sbb.utils.errors import SbbError
from tests.conftest import SCENARIO


def _record(**overrides: object) -> ReplicationRecord:
    fields: dict[str, object] = {
        "method": "static_patch",
        "source": "test",
        "seeds": [0, 1],
        "targets": [{"setting": SCENARIO.model_dump(), "reported": {"ca": 90.0, "asr_all": 100.0}}],
    }
    return ReplicationRecord.model_validate({**fields, **overrides})


def _result(cell: Cell, outcome: str = "ok", **metrics: float) -> CellResult:
    provenance = collect_provenance(torch.device("cpu"), "test", "test", datetime.now(UTC), 0.0)
    return CellResult(
        key=cell.key(),
        cell=cell,
        outcome=outcome,  # type: ignore[arg-type]
        metrics=metrics,
        provenance=provenance,
    )


def test_cells_cover_every_target_and_seed() -> None:
    record = _record(hparams={"selector": {"include_target": True}})
    cells = record.cells()
    assert [c.seed for c in cells] == [0, 1]
    assert {(c.track, c.defense, c.scenario.name) for c in cells} == {
        ("replication", "no_defense", SCENARIO.name)
    }
    assert all(c.attack_hparams == {"selector": {"include_target": True}} for c in cells)


@pytest.mark.parametrize(
    ("ca", "before", "after"),
    [
        ((0.89, 0.91), "pending", "replicated"),  # 90.0, within 2 pp of the paper
        ((0.80, 0.82), "pending", "failed"),
        ((0.80, 0.82), "explained", "explained"),  # a documented miss stays explained
        ((0.89, 0.91), "failed", "replicated"),
    ],
)
def test_status_follows_the_tolerance(ca: tuple[float, float], before: str, after: str) -> None:
    record = _record(status=before)
    results = [_result(c, ca=x, asr_all=1.0) for c, x in zip(record.cells(), ca, strict=True)]
    filled = record.with_results(results)
    assert filled.status == after
    ours = filled.targets[0].ours
    assert ours is not None
    assert ours["ca"].mean == pytest.approx(100 * np.mean(ca))
    assert ours["ca"].std == pytest.approx(100 * np.std(ca))
    assert ours["asr_all"].mean == pytest.approx(100.0)


@pytest.mark.parametrize("problem", ["missing", "failed"])
def test_every_cell_needs_an_ok_result(problem: str) -> None:
    record = _record()
    cells = record.cells()
    results: list[Any] = [_result(cells[0], ca=0.9, asr_all=1.0)]
    results.append(None if problem == "missing" else _result(cells[1], outcome="failed"))
    with pytest.raises(SbbError, match="seed 1"):
        record.with_results(results)


def test_record_round_trip(tmp_path: Path) -> None:
    record = _record(confirmed=True, explanation="a reason")
    record.save(tmp_path / "r.yaml")
    assert ReplicationRecord.load(tmp_path / "r.yaml") == record
