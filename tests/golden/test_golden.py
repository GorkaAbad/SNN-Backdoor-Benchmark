"""Frozen metrics of the smoke benchmark. A change here is a bug or a new benchmark version.

Never edit smoke.json to make this test pass. Ask a maintainer first.
"""

import json
from pathlib import Path

import pytest

from sbb.core.game import Game
from sbb.core.scenario import Cell
from sbb.runner.planner import Planner
from sbb.runner.spec import BenchmarkSpec

ROOT = Path(__file__).parents[2]
EXPECTED = json.loads((Path(__file__).parent / "smoke.json").read_text())
TOLERANCE = 0.005


@pytest.mark.parametrize(
    "cell",
    Planner(BenchmarkSpec.load(ROOT / "benchmarks/smoke.yaml")).cells(),
    ids=lambda c: f"{c.attack}-{c.defense}",
)
def test_smoke_metrics_are_frozen(cell: Cell, game: Game) -> None:
    result = game.run(cell)
    expected = EXPECTED[f"{cell.attack}/{cell.defense}"]
    assert result.outcome == "ok", result.reason
    assert result.metrics.keys() == expected.keys()
    for metric, value in expected.items():
        assert abs(result.metrics[metric] - value) <= TOLERANCE, metric
