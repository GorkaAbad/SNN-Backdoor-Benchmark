import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from sbb.core.registry import ATTACKS, DEFENSES
from sbb.core.result import CellResult, Outcome, collect_provenance
from sbb.core.scenario import Cell
from sbb.data.sample import DATASETS
from sbb.models.snn import ARCHITECTURES
from sbb.results.stats import seed_stats
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


def test_a_track_can_leave_out_methods_of_the_battery() -> None:
    # An attack too slow for a CPU runner can stay out of the lite track and run in the full one.
    spec = SMOKE.model_copy(deep=True)
    spec.tracks["lite"].exclude = ["fine_tuning"]
    cells = Planner(spec).cells()
    assert {c.defense for c in cells} == {"no_defense"} and len(cells) == 2
    data = SMOKE.model_dump()
    data["tracks"]["lite"]["exclude"] = ["not_in_the_battery"]
    with pytest.raises(ValueError, match="not_in_the_battery"):
        BenchmarkSpec.model_validate(data)


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


def test_seed_stats_give_a_t_interval() -> None:
    # Sample std 1 over 3 seeds and t(0.975, 2) = 4.3027: the half-width is 4.3027 / sqrt(3).
    assert seed_stats([1.0, 2.0, 3.0]) == pytest.approx(
        {"mean": 2.0, "std": (2 / 3) ** 0.5, "n": 3, "ci95": 2.4841}, abs=1e-4
    )
    assert seed_stats([0.5])["ci95"] is None  # one seed has no interval


def test_table_shows_the_interval_over_seeds() -> None:
    results = [_result("no_attack", "no_defense", "ok", ca=ca, asr=0.0, ra=ca) for ca in (0.9, 1.0)]
    # Sample std 0.0707 and t(0.975, 1) = 12.706: 63.5 points either side, in percent.
    assert "CA 95.0 ± 63.5 · ASR 0.0 ± 0.0 · RA 95.0 ± 63.5 (n=2)" in markdown_table(results)


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
    out = tmp_path / "my results"  # a path the shell must quote
    assert executor_for(site).submit(cells, out) == ["4242"]
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
    # The task runs in its job folder, so the array index is never inside a quoted path.
    (job,) = (tmp_path / "cache" / "jobs").iterdir()  # outside the results, which git tracks
    assert f"cd {job}" in script
    assert 'sbb run-cell cells/"${SLURM_ARRAY_TASK_ID}".json --site site.yaml' in script
    assert f"--out '{out.resolve()}'" in script
    assert f"--bind '{tmp_path / 'cache'},{out.resolve()}'" in script  # results leave the image
    assert not (out / "jobs").exists()
    written = [Cell.model_validate_json(p.read_text()) for p in sorted((job / "cells").iterdir())]
    assert sorted(c.key() for c in written) == sorted(c.key() for c in cells)
    assert SiteProfile.load(job / "site.yaml") == site


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_local_executor_gives_each_worker_its_own_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    # GPUs in exclusive-process mode (as on many clusters) take one process each.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    lock, busy, used = threading.Lock(), set(), []

    def fake_run(command: list[str], env: dict[str, str], **_: object) -> None:
        gpu = env["CUDA_VISIBLE_DEVICES"]
        with lock:
            assert gpu not in busy or device == "cpu", f"two workers on GPU {gpu}"
            busy.add(gpu)
            used.append(gpu)
        time.sleep(0.05)
        with lock:
            busy.discard(gpu)

    monkeypatch.setattr("sbb.runner.executors.local.subprocess.run", fake_run)
    site = SiteProfile(name="node", device=device, cache_dir=tmp_path, max_parallel=2)
    executor_for(site).submit(Planner(SMOKE).cells(), tmp_path / "out")
    assert sorted(used) == (["3", "3", "5", "5"] if device == "cuda" else ["3,5"] * 4)


def test_slurm_needs_apptainer_or_no_container(tmp_path: Path) -> None:
    site = SiteProfile(name="x", runtime="docker", scheduler="slurm", cache_dir=tmp_path)
    with pytest.raises(SbbError, match="docker"):
        executor_for(site).submit(Planner(SMOKE).cells(), tmp_path / "out")
