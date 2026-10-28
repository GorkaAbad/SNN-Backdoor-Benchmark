"""LeaderboardBuilder: results, cards and replication records into a static, interactive site."""

import json
import shutil
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from sbb import __version__
from sbb.core.cards import AttackCard
from sbb.core.registry import ATTACKS, DEFENSES
from sbb.core.result import CellResult
from sbb.metrics.detection import auroc
from sbb.results.policy import OUTCOME_STATUS, VerificationPolicy
from sbb.results.replication import ReplicationRecord
from sbb.results.stats import seed_stats
from sbb.results.verification import Verification
from sbb.runner.spec import Tolerance
from sbb.site.status import METRICS, percent, write_csv

_TEMPLATES = ("index.html", "app.js", "style.css", "favicon.png")
# An aware attack breaks a defense when its mean ASR against it reaches this (PLAN 13.4, task 7).
BROKEN_ASR = 0.5


class LeaderboardBuilder:
    """Builds the site from results, replication records and verification records.

    Statuses come from the VerificationPolicy (docs/PLAN.md section 10).

    Args:
        results: Cell results of any benchmark versions. Replication cells are left out.
        replications: The replication records.
        verifications: The verification records.
        policy: The policy that computes the statuses, by default with the spec's tolerances.
    """

    def __init__(
        self,
        results: Sequence[CellResult],
        replications: Sequence[ReplicationRecord],
        verifications: Sequence[Verification] = (),
        policy: VerificationPolicy | None = None,
    ) -> None:
        self.results = [r for r in results if r.cell.track != "replication"]
        self.policy = policy or VerificationPolicy(Tolerance())
        self.replications = replications
        self.checks: dict[str, list[Verification]] = defaultdict(list)
        for check in verifications:
            self.checks[check.result_key].append(check)

    def build(self, out: Path) -> None:
        """Write data.json, results.csv and the page into out."""
        out.mkdir(parents=True, exist_ok=True)
        data = self.data()
        (out / "data.json").write_text(json.dumps(data, indent=1))
        rows, columns = _csv(data["cells"])
        write_csv(rows, out / "results.csv", columns)
        templates = files("sbb.site") / "templates"
        for name in _TEMPLATES:
            with (templates / name).open("rb") as source, (out / name).open("wb") as target:
                shutil.copyfileobj(source, target)

    def data(self) -> dict[str, Any]:
        """Everything the page shows."""
        methods = _methods()
        cells = [_aggregate(group, self.checks, self.policy) for group in self._groups().values()]
        return {
            "generated": datetime.now(UTC).isoformat(timespec="seconds"),
            "sbb_version": __version__,
            "metrics": {k: {"label": label, "better": d} for k, (label, d) in METRICS.items()},
            "methods": methods,
            "cells": cells,
            "model_auroc": self._model_auroc(),
            "broken_by": _broken_by(cells, methods),
            "pareto": _pareto(cells, methods),
            "replications": [_replication(r) for r in self.replications],
        }

    def _groups(self) -> dict[tuple[str, ...], list[CellResult]]:
        """The results of each (version, track, scenario, attack, defense), over the seeds."""
        groups: dict[tuple[str, ...], list[CellResult]] = defaultdict(list)
        for r in self.results:
            c = r.cell
            groups[(c.benchmark_version, c.track, c.scenario.name, c.attack, c.defense)].append(r)
        return groups

    def _model_auroc(self) -> list[dict[str, Any]]:
        """Per detection defense and scenario, how well verdicts separate backdoored models.

        Clean models are the no_attack cells, backdoored models the attack cells that ended ok.
        A defense counts as a detector when its cells report a verdict.
        """
        verdicts: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(
            lambda: {"clean": [], "backdoored": []}
        )
        for r in self.results:
            if r.outcome != "ok" or "verdict" not in r.metrics:
                continue
            c = r.cell
            side = "clean" if c.attack == "no_attack" else "backdoored"
            key = (c.benchmark_version, c.track, c.scenario.name, c.defense)
            verdicts[key][side].append(r.metrics["verdict"])
        rows = []
        for (version, track, scenario, defense), sides in sorted(verdicts.items()):
            clean, backdoored = sides["clean"], sides["backdoored"]
            value = auroc(np.array(clean), np.array(backdoored)) if clean and backdoored else None
            rows.append(
                {
                    "version": version,
                    "track": track,
                    "scenario": scenario,
                    "defense": defense,
                    "auroc": value,
                    "clean_models": len(clean),
                    "backdoored_models": len(backdoored),
                }
            )
        return rows


def _aggregate(
    results: list[CellResult], checks: dict[str, list[Verification]], policy: VerificationPolicy
) -> dict[str, Any]:
    """One row of the site: the cell over its valid seeds, each metric as seed_stats gives it.

    Attack validity (PLAN rule 6) is judged per seed. Seeds that did not end ok are left out of
    the metrics and listed in excluded, with their reason (user decision of 2026-10-23). A cell
    takes the most serious outcome of its seeds only when no seed ended ok.
    """
    first = results[0].cell
    valid = [r for r in results if r.outcome == "ok"]
    severity = list(OUTCOME_STATUS)
    bad = sorted((r for r in results if r.outcome != "ok"), key=lambda r: severity.index(r.outcome))
    status = str(policy.cell_status(results, [v for r in results for v in checks.get(r.key, [])]))
    metrics = {}
    for name in sorted({name for r in valid for name in r.metrics}):
        metrics[name] = seed_stats([r.metrics[name] for r in valid if name in r.metrics])
    return {
        "version": first.benchmark_version,
        "track": first.track,
        "scenario": first.scenario.name,
        "dataset": first.scenario.data.name,
        "arch": first.scenario.arch.name,
        "T": first.scenario.repr.T,
        "attack": first.attack,
        "defense": first.defense,
        "seeds": sorted(r.cell.seed for r in valid or results),
        "excluded": [
            {"seed": r.cell.seed, "status": OUTCOME_STATUS[r.outcome], "reason": r.reason}
            for r in sorted(bad, key=lambda r: r.cell.seed)
            if valid
        ],
        "status": status,
        "reason": None if valid else bad[0].reason,
        "metrics": metrics,
        "sites": sorted({r.provenance.site for r in results}),
        "devices": sorted({r.provenance.device for r in results}),
        "verifications": [
            {
                "seed": r.cell.seed,
                "verifier": v.verifier,
                "verifier_org": v.verifier_org,
                "verdict": v.verdict,
                "site": v.provenance.site,
                "device": v.provenance.device,
                "date": v.provenance.started_at.date().isoformat(),
            }
            for r in sorted(results, key=lambda r: r.cell.seed)
            for v in checks.get(r.key, [])
        ],
    }


_CELL = ("version", "track", "scenario", "attack", "defense")


def _broken_by(cells: list[dict[str, Any]], methods: dict[str, Any]) -> list[dict[str, Any]]:
    """The cells where an aware attack keeps a mean ASR of BROKEN_ASR or more against a defense."""
    return [
        {**{k: c[k] for k in _CELL}, "asr": c["metrics"]["asr"]["mean"]}
        for c in cells
        if methods.get(c["attack"], {}).get("aware_of")
        and c["defense"] != "no_defense"
        and "asr" in c["metrics"]
        and c["metrics"]["asr"]["mean"] >= BROKEN_ASR
    ]


def _pareto(cells: list[dict[str, Any]], methods: dict[str, Any]) -> list[dict[str, Any]]:
    """Per version, track, scenario and attack group: each defense's mean CA and ASR, front marked.

    The means run over the attacks of the group with both metrics, no_attack left out. Detection
    defenses are left out too: they leave the model as it is, so their CA and ASR are those of
    no_defense, and the model AUROC rates them. A defense is on the front when no other one keeps
    at least its CA and at most its ASR, and one of the two strictly.
    """
    groups: dict[tuple[str, ...], dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for c in cells:
        m = c["metrics"]
        detection = methods.get(c["defense"], {}).get("group") == "model_detection"
        if c["attack"] == "no_attack" or detection or "ca" not in m or "asr" not in m:
            continue
        group = methods.get(c["attack"], {}).get("group", "unknown")
        key = (c["version"], c["track"], c["scenario"], group)
        groups[key][c["defense"]].append((m["ca"]["mean"], m["asr"]["mean"]))
    points = []
    for (version, track, scenario, group), defenses in sorted(groups.items()):
        means = {
            d: (float(np.mean([ca for ca, _ in v])), float(np.mean([a for _, a in v])))
            for d, v in defenses.items()
        }
        for defense, (ca, asr) in means.items():
            dominated = any(
                o_ca >= ca and o_asr <= asr and (o_ca, o_asr) != (ca, asr)
                for o_ca, o_asr in means.values()
            )
            points.append(
                {
                    "version": version,
                    "track": track,
                    "scenario": scenario,
                    "group": group,
                    "defense": defense,
                    "ca": ca,
                    "asr": asr,
                    "attacks": len(defenses[defense]),
                    "front": not dominated,
                }
            )
    return points


def _csv(cells: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """The site's cells as CSV rows and columns, for analysis outside the page.

    One row per cell. For each metric over its valid seeds: the mean, the population std and the
    half-width of the 95% t-interval, empty for one seed.
    """
    known = [m for m in METRICS if any(m in c["metrics"] for c in cells)]
    names = known + sorted({m for c in cells for m in c["metrics"]} - set(METRICS))
    keys = ["version", "track", "scenario", "dataset", "arch", "T", "attack", "defense", "status"]
    rows = []
    for cell in sorted(cells, key=lambda c: [str(c[k]) for k in keys]):
        row = {k: cell[k] for k in keys}
        row["seeds"] = " ".join(map(str, cell["seeds"]))
        for name, stat in cell["metrics"].items():
            row[f"{name}_mean"] = percent(name, stat["mean"])
            row[f"{name}_std"] = percent(name, stat["std"])
            row[f"{name}_ci95"] = "" if stat["ci95"] is None else percent(name, stat["ci95"])
        row["excluded"] = "; ".join(
            f"{e['seed']}: {e['status']} ({e['reason']})" for e in cell["excluded"]
        )
        row["sites"], row["devices"] = "; ".join(cell["sites"]), "; ".join(cell["devices"])
        row["reason"] = cell["reason"] or ""
        rows.append(row)
    stats = [f"{name}_{s}" for name in names for s in ("mean", "std", "ci95")]
    return rows, [*keys, "seeds", "excluded", *stats, "sites", "devices", "reason"]


def _methods() -> dict[str, dict[str, Any]]:
    """Every registered attack and defense, from its card."""
    methods = {}
    for name in ATTACKS.names():
        attack = ATTACKS.get(name).card
        methods[name] = {
            "role": "attack",
            "group": attack_group(attack),
            "aware_of": list(attack.knowledge.aware_of),
            "trigger_domain": attack.capabilities.trigger_domain,
            "adaptation_note": attack.adaptation_note,
            **_common(attack.model_dump(mode="json")),
        }
    for name in DEFENSES.names():
        defense = DEFENSES.get(name).card
        methods[name] = {
            "role": "defense",
            "group": str(defense.stage),
            "resources": sorted(str(r) for r in defense.view.resources),
            "adaptation_note": defense.adaptation_note,
            **_common(defense.model_dump(mode="json")),
        }
    return methods


def attack_group(card: AttackCard) -> str:
    """The site's group of an attack: adaptive for an aware attack, otherwise its kind."""
    return "adaptive" if card.knowledge.aware_of else str(card.kind)


def _common(card: dict[str, Any]) -> dict[str, Any]:
    keys = ("version", "title", "paper", "published", "code_reference", "license_note")
    return {k: card.get(k) for k in keys}


def _replication(record: ReplicationRecord) -> dict[str, Any]:
    """A replication record with the setting reduced to what the page shows."""
    targets = []
    for target in record.targets:
        setting = target.setting
        ours = {k: v.model_dump() for k, v in (target.ours or {}).items()}
        targets.append(
            {
                "setting": setting.name,
                "dataset": setting.data.name,
                "arch": setting.arch.name,
                "epochs": setting.train.epochs,
                "reported": target.reported,
                "ours": ours or None,
            }
        )
    return {
        "method": record.method,
        "source": record.source,
        "confirmed": record.confirmed,
        "status": record.status,
        "tolerance_pp": record.tolerance_pp,
        "seeds": record.seeds,
        "explanation": record.explanation,
        "targets": targets,
    }
