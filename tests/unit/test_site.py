import csv
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import torch

from sbb.core.result import CellResult, Outcome, collect_provenance
from sbb.results.replication import ReplicationRecord
from sbb.results.verification import LITE_VERIFIER, Verification
from sbb.runner.planner import Planner
from sbb.site.builder import LeaderboardBuilder, attack_group
from sbb.site.status import status_rows, write_csv
from tests.cards import attack_card
from tests.unit.test_runner import SMOKE

PROVENANCE = collect_provenance(torch.device("cpu"), "laptop", "test", datetime.now(UTC), 0.0)
# CA 0.8 and 0.9 over two seeds: sample std 0.0707 and t(0.975, 1) = 12.706 give the interval.
STAT_085 = {"mean": 0.85, "std": 0.05, "n": 2, "ci95": 0.6353}
REPLICATIONS = Path(__file__).parents[2] / "replications"


def _result(
    attack: str, defense: str, seed: int, outcome: Outcome = "ok", track: str = "lite", **m: float
) -> CellResult:
    cell = Planner(SMOKE).cells(attack=attack, defense=defense)[0]
    cell = cell.model_copy(update={"seed": seed, "track": track})
    reason = None if outcome == "ok" else "a reason"
    return CellResult(
        key=cell.key(), cell=cell, outcome=outcome, reason=reason, metrics=m, provenance=PROVENANCE
    )


RESULTS = [
    _result("static_patch", "no_defense", 0, ca=0.9, asr=1.0, verdict=0.7),
    _result("static_patch", "no_defense", 1, ca=0.8, asr=0.9, verdict=0.8),
    _result("no_attack", "fine_tuning", 0, "not_applicable"),
    _result("no_attack", "no_defense", 0, verdict=0.2),
    _result("static_patch", "fine_tuning", 0, verdict=0.9),
]


def test_cells_aggregate_the_seeds() -> None:
    data = LeaderboardBuilder(RESULTS, []).data()
    cell = next(
        c for c in data["cells"] if (c["attack"], c["defense"]) == ("static_patch", "no_defense")
    )
    assert cell["seeds"] == [0, 1] and cell["status"] == "self-reported"
    assert cell["metrics"]["ca"] == pytest.approx(STAT_085, abs=1e-4)
    assert (cell["dataset"], cell["arch"], cell["sites"]) == (
        "synthetic",
        "tiny_lif_cnn",
        ["laptop"],
    )
    na = next(
        c for c in data["cells"] if c["defense"] == "fine_tuning" and c["attack"] == "no_attack"
    )
    assert na["status"] == "not-applicable" and na["reason"] == "a reason" and na["metrics"] == {}
    assert data["metrics"]["asr"]["label"] and "static_patch" in data["methods"]
    assert data["methods"]["static_patch"]["role"] == "attack"


def _check(result: CellResult, verdict: str, verifier: str = LITE_VERIFIER) -> Verification:
    return Verification(
        result_key=result.key,
        verifier=verifier,
        verifier_org=verifier,
        provenance=PROVENANCE,
        metrics={m: v + (0.0 if verdict == "match" else 0.5) for m, v in result.metrics.items()},
        delta={m: 0.0 if verdict == "match" else 0.5 for m in result.metrics},
        verdict=verdict,  # type: ignore[arg-type]
    )


def test_the_lite_verifier_sets_the_status_of_lite_cells() -> None:
    seed0, seed1 = RESULTS[0], RESULTS[1]  # static_patch x no_defense
    full = _result("no_attack", "no_defense", 0, track="full", ca=0.9)

    def cell(*checks: Verification, attack: str = "static_patch") -> dict:
        data = LeaderboardBuilder([*RESULTS, full], [], checks).data()
        return next(
            c
            for c in data["cells"]
            if (c["attack"], c["defense"]) == (attack, "no_defense")
            and c["track"] == ("full" if attack == "no_attack" else "lite")
        )

    assert cell(_check(seed0, "match"))["status"] == "self-reported"  # seed 1 is not checked yet
    both = cell(_check(seed0, "match"), _check(seed1, "match"))
    assert both["status"] == "reproduced"
    assert [(v["seed"], v["verdict"]) for v in both["verifications"]] == [
        (0, "match"),
        (1, "match"),
    ]
    assert cell(_check(seed0, "match"), _check(seed1, "mismatch"))["status"] == "disputed"
    others = cell(_check(seed0, "match", "someone"), _check(seed1, "match", "someone"))
    assert others["status"] == "self-reported"  # only the verify-lite workflow counts for lite
    # A full cell counts a match from another organization than the submitter's (v2).
    assert cell(attack="no_attack")["status"] == "self-reported"
    assert cell(_check(full, "match"), attack="no_attack")["status"] == "reproduced"

    # A record checked the numbers of the result it reran. Once the result changes, it is stale.
    stale = _check(seed1, "match").model_copy(update={"delta": {"ca": 0.05}})
    assert cell(_check(seed0, "match"), stale)["status"] == "self-reported"


def test_statuses_use_the_tolerance_of_each_benchmark_version() -> None:
    def moved(r: CellResult) -> Verification:  # verify-lite got every metric 2 pp higher
        return _check(r, "match").model_copy(
            update={
                "delta": dict.fromkeys(r.metrics, 0.02),
                "metrics": {m: v + 0.02 for m, v in r.metrics.items()},
            }
        )

    def status(**tolerances: Any) -> str:
        data = LeaderboardBuilder(RESULTS, [], [moved(RESULTS[0]), moved(RESULTS[1])], **tolerances)
        pair = ("static_patch", "no_defense")
        return str(
            next(c for c in data.data()["cells"] if (c["attack"], c["defense"]) == pair)["status"]
        )

    assert status() == "reproduced"  # the default allows 3 pp
    assert status(tolerances={SMOKE.version: SMOKE.tolerance}) == "disputed"  # smoke allows 1


def test_model_auroc_ranks_backdoored_above_clean_models() -> None:
    # Clean models are the no_attack cells, backdoored ones the valid attack cells.
    data = LeaderboardBuilder(RESULTS, []).data()
    by_defense = {(a["scenario"], a["defense"]): a for a in data["model_auroc"]}
    assert by_defense[("synthetic_tiny", "no_defense")]["auroc"] == 1.0
    assert by_defense[("synthetic_tiny", "fine_tuning")]["auroc"] is None  # no clean verdict


def test_build_writes_a_site_with_the_data(tmp_path: Path) -> None:
    records = [ReplicationRecord.load(p) for p in sorted(REPLICATIONS.glob("*.yaml"))]
    LeaderboardBuilder(RESULTS, records).build(tmp_path)
    data = json.loads((tmp_path / "data.json").read_text())
    assert {r["method"] for r in data["replications"]} >= {"sneaky_static", "flashy"}
    target = data["replications"][0]["targets"][0]
    assert {"dataset", "arch", "reported", "ours"} <= target.keys()
    html = (tmp_path / "index.html").read_text()
    assert "app.js" in html and "style.css" in html
    with (tmp_path / "results.csv").open() as f:
        rows = {(r["attack"], r["defense"]): r for r in csv.DictReader(f)}
    assert len(rows) == len(data["cells"])
    cell = rows[("static_patch", "no_defense")]
    assert (cell["status"], cell["seeds"]) == ("self-reported", "0 1")
    assert (float(cell["ca_mean"]), float(cell["ca_std"])) == (85.0, 5.0)  # percent
    assert float(cell["ca_ci95"]) == pytest.approx(63.531, abs=1e-3)  # t(0.975, 1) x 5 points
    assert float(cell["verdict_mean"]) == 0.75  # a raw score
    na = rows[("no_attack", "fine_tuning")]
    assert (na["status"], na["reason"], na["ca_mean"]) == ("not-applicable", "a reason", "")
    if shutil.which("node"):  # the page's script parses
        subprocess.run(["node", "--check", str(tmp_path / "app.js")], check=True)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_page_lists_baselines_then_the_oldest_papers_first() -> None:
    app = Path(__file__).parents[2] / "src" / "sbb" / "site" / "templates" / "app.js"
    script = """
const { matrix } = require(process.argv[1]);
const methods = {
  no_attack: { role: "attack", group: "data_poisoning", published: null },
  flashy: { role: "attack", group: "data_poisoning", published: "2024-11" },
  sneaky_static: { role: "attack", group: "data_poisoning", published: "2023-02" },
  sneaky_dynamic: { role: "attack", group: "training_control", published: "2023-02" },
  finetune_injection: { role: "attack", group: "supply_chain", published: null },
  no_defense: { role: "defense", group: "post_training", published: null },
  strip: { role: "defense", group: "inference", published: "2019-02" },
  fine_pruning: { role: "defense", group: "post_training", published: "2018-05" },
};
const cell = (attack, defense, metrics, status) =>
  ({ attack, defense, metrics, status, seeds: [0, 1], sites: ["laptop"], reason: null });
const cells = [];
for (const a of ["flashy", "finetune_injection", "sneaky_dynamic", "no_attack", "sneaky_static"])
  for (const d of ["strip", "no_defense", "fine_pruning"])
    cells.push(cell(a, d, { asr: { mean: 0.5, std: 0.1 } }, "self-reported"));
cells.push(cell("flashy", "strip", {}, "not-applicable"));
console.log(JSON.stringify(matrix(cells.slice(-1).concat(cells.slice(1)), "asr", methods)));
"""
    out = subprocess.run(
        ["node", "-e", script, str(app)], capture_output=True, text=True, check=True
    )
    m = json.loads(out.stdout)
    # Baselines first, then by first publication (Sneaky Spikes before Flashy), undated last.
    assert m["attacks"] == [
        "no_attack",
        "sneaky_dynamic",
        "sneaky_static",
        "flashy",
        "finetune_injection",
    ]
    assert m["defenses"] == ["no_defense", "fine_pruning", "strip"]
    assert m["text"][3][2] == "not-applicable" and m["z"][3][2] is None  # flashy x strip


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_page_opens_on_the_newest_version() -> None:
    app = Path(__file__).parents[2] / "src" / "sbb" / "site" / "templates" / "app.js"
    script = (
        "const { newestFirst } = require(process.argv[1]);"
        'console.log(JSON.stringify(newestFirst(["0.9.0", "0.10.0", "0.9.1"])));'
    )
    out = subprocess.run(
        ["node", "-e", script, str(app)], capture_output=True, text=True, check=True
    )
    assert json.loads(out.stdout) == ["0.10.0", "0.9.1", "0.9.0"]


def test_status_lists_every_planned_cell_with_its_state(tmp_path: Path) -> None:
    cells = Planner(SMOKE).cells()
    done = _result("static_patch", "no_defense", 0, ca=0.9, asr=1.0)
    na = _result("no_attack", "fine_tuning", 0, "not_applicable")
    rows = status_rows(cells, [done, na], [_check(done, "match")])
    by_pair = {(r["attack"], r["defense"]): r for r in rows}
    assert len(rows) == len(cells)
    assert by_pair[("static_patch", "no_defense")]["state"] == "ok"
    assert by_pair[("static_patch", "no_defense")]["verification"] == "reproduced"
    rerun = done.model_copy(update={"metrics": {"ca": 0.5, "asr": 1.0}})  # same key, new numbers
    stale = {
        (r["attack"], r["defense"]): r for r in status_rows(cells, [rerun], [_check(done, "match")])
    }
    assert stale[("static_patch", "no_defense")]["verification"] == "self-reported"
    assert by_pair[("static_patch", "no_defense")]["ca"] == 90.0  # percent
    assert by_pair[("no_attack", "fine_tuning")]["state"] == "not_applicable"
    assert by_pair[("no_attack", "fine_tuning")]["reason"] == "a reason"
    assert by_pair[("no_attack", "no_defense")]["state"] == "missing"
    path = tmp_path / "status.csv"
    write_csv(rows, path)
    lines = path.read_text().splitlines()
    assert lines[0].startswith("version,track,scenario") and len(lines) == len(cells) + 1


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_page_exports_the_matrix_as_csv() -> None:
    app = Path(__file__).parents[2] / "src" / "sbb" / "site" / "templates" / "app.js"
    script = """
const { matrixCsv, toCsv, seedCount } = require(process.argv[1]);
const m = { attacks: ["a", "b"], defenses: ["d1", "d,2"], z: [[12.345678, null], [0, 99.5]] };
const partial = { seeds: [0, 2], excluded: [{ seed: 1 }] };
console.log(JSON.stringify([matrixCsv(m), toCsv([['say "hi"', 1]]), seedCount(partial)]));
"""
    out = subprocess.run(
        ["node", "-e", script, str(app)], capture_output=True, text=True, check=True
    )
    matrix, quoted, seeds = json.loads(out.stdout)
    assert seeds == "2 of 3"
    assert matrix == 'attack,d1,"d,2"\na,12.3457,\nb,0,99.5\n'  # empty where no value
    assert quoted == '"say ""hi""",1\n'


def test_a_cell_averages_its_valid_seeds_and_lists_the_others(tmp_path: Path) -> None:
    results = [
        _result("static_patch", "no_defense", 0, ca=0.9, asr=1.0),
        _result("static_patch", "no_defense", 1, "invalid_attack"),
        _result("static_patch", "no_defense", 2, ca=0.8, asr=0.8),
        _result("no_attack", "no_defense", 0, "invalid_attack"),
    ]
    builder = LeaderboardBuilder(results, [])
    cells = {(c["attack"], c["defense"]): c for c in builder.data()["cells"]}
    partial = cells[("static_patch", "no_defense")]
    assert partial["status"] == "self-reported" and partial["seeds"] == [0, 2]
    assert partial["metrics"]["ca"] == pytest.approx(STAT_085, abs=1e-4)
    assert partial["excluded"] == [{"seed": 1, "status": "invalid-attack", "reason": "a reason"}]
    assert partial["reason"] is None
    invalid = cells[("no_attack", "no_defense")]  # no valid seed: the whole cell is invalid
    assert (invalid["status"], invalid["reason"], invalid["metrics"]) == (
        "invalid-attack",
        "a reason",
        {},
    )
    assert invalid["seeds"] == [0] and invalid["excluded"] == []
    builder.build(tmp_path)
    with (tmp_path / "results.csv").open() as f:
        row = next(r for r in csv.DictReader(f) if r["attack"] == "static_patch")
    assert (row["seeds"], row["excluded"]) == ("0 2", "1: invalid-attack (a reason)")


def test_aware_attacks_are_grouped_as_adaptive() -> None:
    assert attack_group(attack_card()) == "data_poisoning"
    assert attack_group(attack_card(knowledge={"defense": "aware:strip"})) == "adaptive"


def _as(attack: str, defense: str, seed: int = 0, **m: float) -> CellResult:
    """A smoke result renamed to attack x defense, which need not be in the smoke battery."""
    base = _result("static_patch", "no_defense", seed, **m)
    cell = base.cell.model_copy(update={"attack": attack, "defense": defense})
    return base.model_copy(update={"cell": cell, "key": cell.key()})


def test_aware_attacks_with_half_the_asr_break_a_defense() -> None:
    results = [
        _as("tact", "fine_tuning", 0, ca=0.9, asr=0.8),
        _as("tact", "fine_tuning", 1, ca=0.9, asr=0.6),
        _as("tact", "no_defense", 0, ca=0.9, asr=1.0),  # no defense to break
        _as("tact", "tmpbd", 0, ca=0.9, asr=1.0),  # detection keeps the victim, its ASR too
        _as("static_patch", "fine_tuning", 0, ca=0.9, asr=0.9),  # not an aware attack
        _as("adap_blend", "fine_tuning", 0, ca=0.9, asr=0.3),  # below the threshold
    ]
    broken = LeaderboardBuilder(results, []).data()["broken_by"]
    assert [(b["attack"], b["defense"]) for b in broken] == [("tact", "fine_tuning")]
    assert broken[0]["asr"] == pytest.approx(0.7)


def test_pareto_averages_each_defense_over_the_attacks_of_a_group() -> None:
    results = [
        _as("static_patch", "no_defense", ca=0.9, asr=1.0),
        _as("sneaky_static", "no_defense", ca=0.9, asr=0.8),
        _as("static_patch", "fine_tuning", ca=0.8, asr=0.2),
        _as("sneaky_static", "fine_tuning", ca=0.8, asr=0.4),
        _as("static_patch", "fine_pruning", ca=0.7, asr=0.5),
        _as("sneaky_static", "fine_pruning", ca=0.7, asr=0.5),
        _as("no_attack", "no_defense", ca=0.95, asr=0.05),  # no backdoor to remove
        _as("static_patch", "tmpbd", ca=0.9, asr=1.0),  # detection leaves the model as it is
        _as("static_patch", "anp", ca=0.95, asr=0.0),  # one attack only, so off the front
        _as("tact", "fine_tuning", ca=0.85, asr=0.6),  # the adaptive group
    ]
    points = LeaderboardBuilder(results, []).data()["pareto"]
    poisoning = {
        p["defense"]: (round(p["ca"], 3), round(p["asr"], 3), p["attacks"], p["front"])
        for p in points
        if p["group"] == "data_poisoning"
    }
    assert poisoning == {
        "no_defense": (0.9, 0.9, 2, True),
        "fine_tuning": (0.8, 0.3, 2, True),
        "fine_pruning": (0.7, 0.5, 2, False),  # fine_tuning keeps more CA and removes more ASR
        "anp": (0.95, 0.0, 1, False),
    }
    assert [p["defense"] for p in points if p["group"] == "adaptive"] == ["fine_tuning"]
