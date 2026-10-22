import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from sbb.core.result import CellResult, Outcome, collect_provenance
from sbb.results.replication import ReplicationRecord
from sbb.results.verification import LITE_VERIFIER, Verification
from sbb.runner.planner import Planner
from sbb.site.builder import LeaderboardBuilder
from tests.unit.test_runner import SMOKE

PROVENANCE = collect_provenance(torch.device("cpu"), "laptop", "test", datetime.now(UTC), 0.0)
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
    assert cell["metrics"]["ca"] == pytest.approx({"mean": 0.85, "std": 0.05})
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
        metrics=result.metrics,
        delta={},
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
    assert cell(_check(full, "match"), attack="no_attack")["status"] == "self-reported"

    # A record checked the numbers of the result it reran. Once the result changes, it is stale.
    stale = _check(seed1, "match").model_copy(update={"delta": {"ca": 0.05}})
    assert cell(_check(seed0, "match"), stale)["status"] == "self-reported"


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
