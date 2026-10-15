import json
from pathlib import Path

from typer.testing import CliRunner

from sbb.cli import app
from sbb.runner.planner import Planner
from sbb.runner.spec import BenchmarkSpec

SMOKE = Path(__file__).parents[2] / "benchmarks" / "smoke.yaml"


def _site(tmp_path: Path) -> Path:
    site = tmp_path / "site.yaml"
    site.write_text(f"name: test\ndevice: cpu\ncache_dir: {tmp_path / 'cache'}\n")
    return site


def test_smoke_run_writes_results_and_table(tmp_path: Path) -> None:
    out, site = tmp_path / "results", _site(tmp_path)
    result = CliRunner().invoke(app, ["run", str(SMOKE), "--site", str(site), "--out", str(out)])
    assert result.exit_code == 0, result.output
    version_dir = out / "smoke-0.1.0"
    assert len(list(version_dir.glob("*.json"))) == 4
    table = (version_dir / "table.md").read_text()
    assert "static_patch" in table and "fine_tuning" in table

    again = CliRunner().invoke(app, ["run", str(SMOKE), "--site", str(site), "--out", str(out)])
    assert "4 cells, 0 pending" in again.output

    planned = CliRunner().invoke(app, ["plan", str(SMOKE), "--out", str(out)])
    assert planned.exit_code == 0 and planned.output.count(" ok ") == 4


def test_run_cell_runs_one_cell_from_json(tmp_path: Path) -> None:
    cell = Planner(BenchmarkSpec.load(SMOKE)).cells(attack="no_attack", defense="no_defense")[0]
    cell_json = tmp_path / "cell.json"
    cell_json.write_text(cell.model_dump_json())
    out = tmp_path / "results"
    args = ["run-cell", str(cell_json), "--site", str(_site(tmp_path)), "--out", str(out)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    saved = json.loads((out / "smoke-0.1.0" / f"{cell.key()}.json").read_text())
    assert saved["outcome"] == "ok" and saved["provenance"]["executor"] == "run-cell"
