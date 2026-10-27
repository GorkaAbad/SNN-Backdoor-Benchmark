import json
from pathlib import Path

import yaml
from click import unstyle
from typer.testing import CliRunner

from sbb.cli import app
from sbb.results.replication import ReplicationRecord
from sbb.runner.planner import Planner
from sbb.runner.spec import BenchmarkSpec
from tests.conftest import SCENARIO

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
    assert "4 cells, 0 pending" in unstyle(again.output)

    planned = CliRunner().invoke(app, ["plan", str(SMOKE), "--out", str(out)])
    assert planned.exit_code == 0 and unstyle(planned.output).count(" ok ") == 4


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


def test_replicate_runs_the_record_and_fills_it(tmp_path: Path) -> None:
    records, out = tmp_path / "replications", tmp_path / "results"
    records.mkdir()
    budget = SCENARIO.budget.model_copy(update={"delta_pp": 100.0})
    setting = SCENARIO.model_copy(update={"budget": budget}).model_dump(mode="json")
    record = {
        "method": "static_patch",
        "source": "test",
        "seeds": [0],
        "targets": [{"setting": setting, "reported": {"ca": 100.0, "asr_all": 100.0}}],
    }
    (records / "static_patch.yaml").write_text(yaml.safe_dump(record))
    args = ["replicate", "static_patch", "--records", str(records), "--site", str(_site(tmp_path))]
    result = CliRunner().invoke(app, [*args, "--out", str(out)])
    assert result.exit_code == 0, result.output
    filled = ReplicationRecord.load(out / "replications" / "static_patch.yaml")
    assert filled.status == "replicated" and filled.targets[0].ours is not None
    assert ReplicationRecord.load(records / "static_patch.yaml").status == "pending"

    again = CliRunner().invoke(app, [*args, "--out", str(out), "--update"])
    assert "1 cells, 0 pending" in unstyle(again.output)
    assert ReplicationRecord.load(records / "static_patch.yaml").status == "replicated"


def test_run_with_a_process_pool(tmp_path: Path) -> None:
    # max_parallel 2 runs each cell in its own sbb run-cell process.
    site = tmp_path / "site.yaml"
    site.write_text(f"name: pool\ndevice: cpu\nmax_parallel: 2\ncache_dir: {tmp_path / 'cache'}\n")
    out = tmp_path / "results"
    args = ["run", str(SMOKE), "--attack", "no_attack", "--site", str(site), "--out", str(out)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert len(list((out / "smoke-0.1.0").glob("*.json"))) == 2
    assert "no_attack" in (out / "smoke-0.1.0" / "table.md").read_text()


def test_bundle_validate_and_reproduce_a_plugin(tmp_path: Path) -> None:
    out, site = tmp_path / "results", _site(tmp_path)
    assert (
        CliRunner()
        .invoke(app, ["run", str(SMOKE), "--site", str(site), "--out", str(out)])
        .exit_code
        == 0
    )
    bundled = CliRunner().invoke(app, ["bundle", "--plugin", "static_patch", "--out", str(out)])
    assert bundled.exit_code == 0, bundled.output
    (manifest,) = (out / "manifests").iterdir()
    content = json.loads(manifest.read_text())
    assert len(content["results"]) == 2 and content["plugins"]["static_patch"] == "1.0.0"
    assert manifest.stem == content["bundle_id"]

    valid = CliRunner().invoke(
        app, ["validate", str(out), str(SMOKE.parent.parent / "replications")]
    )
    assert valid.exit_code == 0, valid.output

    original = next((out / "smoke-0.1.0").glob("*.json"))
    verifications = tmp_path / "verifications"
    args = ["reproduce", str(original), "--site", str(site), "--verifications", str(verifications)]
    args += ["--benchmarks", str(SMOKE.parent)]
    reproduced = CliRunner().invoke(app, [*args, "--verifier", "bob", "--verifier-org", "lab"])
    assert reproduced.exit_code == 0, reproduced.output
    (record,) = verifications.rglob("*.json")
    assert json.loads(record.read_text())["verdict"] == "match"  # same seed, same CPU: exact
    assert CliRunner().invoke(app, ["validate", str(record)]).exit_code == 0

    # An explicit tolerance of 0 is exact: a result half a point off no longer matches.
    shifted = json.loads(original.read_text())
    shifted["metrics"]["ca"] -= 0.005
    copy = tmp_path / "shifted" / original.name
    copy.parent.mkdir()
    copy.write_text(json.dumps(shifted))
    strict = [*args[:1], str(copy), *args[2:], "--tolerance-pp", "0"]
    assert CliRunner().invoke(app, strict).exit_code == 0
    verdicts = {json.loads(r.read_text())["verdict"] for r in verifications.rglob("*.json")}
    assert verdicts == {"match", "mismatch"}

    # By default the spec's tolerance applies: smoke allows 1 pp on lite, the default 3 pp.
    shifted["metrics"]["ca"] -= 0.015
    copy.write_text(json.dumps(shifted))
    by_spec = tmp_path / "by_spec"
    loose = ["reproduce", str(copy), "--site", str(site), "--verifications", str(by_spec)]
    assert CliRunner().invoke(app, [*loose, "--benchmarks", str(SMOKE.parent)]).exit_code == 0
    (record,) = by_spec.rglob("*.json")
    assert json.loads(record.read_text())["verdict"] == "mismatch"

    # A card with broken YAML is reported, not a crash.
    broken = tmp_path / "cards" / "broken.yaml"
    broken.parent.mkdir()
    broken.write_text("name: [unclosed\n")
    checked = CliRunner().invoke(app, ["validate", str(broken)])
    assert checked.exit_code == 1 and "broken.yaml" in unstyle(checked.output)

    # A result changed after bundling no longer matches the manifest.
    key = content["results"][0]["path"]
    tampered = json.loads((out / key).read_text())
    tampered["metrics"]["asr"] = 0.5
    (out / key).write_text(json.dumps(tampered))
    invalid = CliRunner().invoke(app, ["validate", str(manifest)])
    assert invalid.exit_code == 1 and "sha256" in unstyle(invalid.output)
