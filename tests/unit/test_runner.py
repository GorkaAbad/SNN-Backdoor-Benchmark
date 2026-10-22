import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from sbb.core.registry import ATTACKS, DEFENSES
from sbb.core.result import CellResult, Outcome, collect_provenance
from sbb.core.scenario import Cell
from sbb.data.sample import DATASETS
from sbb.models.snn import ARCHITECTURES
from sbb.results.store import LocalResultStore
from sbb.results.table import markdown_table
from sbb.runner.executors import executor_for
from sbb.runner.planner import Planner
from sbb.runner.site import SiteProfile
from sbb.runner.spec import BenchmarkSpec
from sbb.utils.errors import SbbError

BENCHMARKS = Path(__file__).parents[2] / "benchmarks"
SMOKE = BenchmarkSpec.load(BENCHMARKS / "smoke.yaml")
PROVENANCE = collect_provenance(torch.device("cpu"), "test", "test", datetime.now(UTC), 0.0)


def test_planner_filters() -> None:
    planner = Planner(SMOKE)
    assert len(planner.cells()) == 4
    assert len(planner.cells(track="full")) == 0
    only = planner.cells(attack="static_patch", defense="fine_tuning")
    assert [(c.attack, c.defense) for c in only] == [("static_patch", "fine_tuning")]


@pytest.mark.parametrize("path", sorted(BENCHMARKS.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_benchmark_names_registered_parts(path: Path) -> None:
    spec = BenchmarkSpec.load(path)
    battery = spec.battery
    assert set(battery.attacks + battery.dev) <= set(ATTACKS.names())
    assert set(battery.defenses) <= set(DEFENSES.names())
    for scenario in spec.scenarios.values():
        assert (
            scenario.data.name in DATASETS.names() and scenario.arch.name in ARCHITECTURES.names()
        )
    assert Planner(spec).cells()


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


def test_pending_skips_cells_with_a_result_that_did_not_fail(tmp_path: Path) -> None:
    store = LocalResultStore(tmp_path)
    store.save(_result("no_attack", "no_defense", "ok", ca=1.0))
    store.save(_result("no_attack", "fine_tuning", "failed"))
    store.save(_result("static_patch", "no_defense", "not_applicable"))
    pending = Planner(SMOKE).pending(Planner(SMOKE).cells(), store)
    assert [(c.attack, c.defense) for c in pending] == [
        ("no_attack", "fine_tuning"),  # failed cells run again
        ("static_patch", "fine_tuning"),
    ]


def _fake_sbatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An sbatch on PATH that keeps the script it receives and prints a job id."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(f'#!/bin/sh\ncp "$2" "{tmp_path}/submitted.sbatch"\necho 4242\n')
    sbatch.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return tmp_path / "submitted.sbatch"


def test_slurm_executor_submits_one_array_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted = _fake_sbatch(tmp_path, monkeypatch)
    site = SiteProfile(
        name="cluster",
        runtime="apptainer",
        scheduler="slurm",
        image="docker://ghcr.io/org/sbb@sha256:abc",
        cache_dir=tmp_path / "cache",
        max_parallel=2,
        slurm={"partition": "gpu", "time": "06:00:00", "extra": "#SBATCH --qos=normal"},
    )
    cells = Planner(SMOKE).cells()
    assert executor_for(site).submit(cells, tmp_path / "out") == ["4242"]
    script = submitted.read_text()
    for line in [
        "#SBATCH --array=0-3%2",
        "#SBATCH --partition=gpu",
        "#SBATCH --time=06:00:00",
        "#SBATCH --qos=normal",
        "apptainer exec --nv",
        "docker://ghcr.io/org/sbb@sha256:abc",
    ]:
        assert line in script, line
    assert "sbb run-cell" in script and str((tmp_path / "out").resolve()) in script
    (job,) = (tmp_path / "out" / "jobs").iterdir()
    written = [Cell.model_validate_json(p.read_text()) for p in sorted((job / "cells").iterdir())]
    assert sorted(c.key() for c in written) == sorted(c.key() for c in cells)
    assert SiteProfile.load(job / "site.yaml") == site


def test_slurm_needs_apptainer_or_no_container(tmp_path: Path) -> None:
    site = SiteProfile(name="x", runtime="docker", scheduler="slurm", cache_dir=tmp_path)
    with pytest.raises(SbbError, match="docker"):
        executor_for(site).submit(Planner(SMOKE).cells(), tmp_path / "out")
