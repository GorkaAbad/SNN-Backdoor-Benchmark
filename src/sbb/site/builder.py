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
from sbb.core.registry import ATTACKS, DEFENSES
from sbb.core.result import CellResult
from sbb.metrics.detection import auroc
from sbb.results.replication import ReplicationRecord

# Labels and direction of the metrics the site shows. A metric not listed keeps its name.
METRICS = {
    "ca": ("Clean accuracy", "higher"),
    "asr": ("Attack success rate", "lower"),
    "asr_all": ("ASR, target class included", "lower"),
    "ra": ("Robust accuracy", "higher"),
    "frr": ("False rejection rate", "lower"),
    "tpr": ("Rejection of triggered inputs", "higher"),
    "auroc": ("Input AUROC", "higher"),
    "verdict": ("Detection verdict", "none"),
}
# What a cell shows when its seeds did not all end ok, the most serious first.
_STATUS = {
    "failed": "failed",
    "invalid_attack": "invalid-attack",
    "not_applicable": "not-applicable",
}
_TEMPLATES = ("index.html", "app.js", "style.css")


class LeaderboardBuilder:
    """Builds the site from results and replication records.

    Every finished cell is self-reported in v1. The statuses computed from verifications land
    with the VerificationPolicy in v2.

    Args:
        results: Cell results of any benchmark versions. Replication cells are left out.
        replications: The replication records.
    """

    def __init__(
        self, results: Sequence[CellResult], replications: Sequence[ReplicationRecord]
    ) -> None:
        self.results = [r for r in results if r.cell.track != "replication"]
        self.replications = replications

    def build(self, out: Path) -> None:
        """Write data.json and the page into out."""
        out.mkdir(parents=True, exist_ok=True)
        (out / "data.json").write_text(json.dumps(self.data(), indent=1))
        templates = files("sbb.site") / "templates"
        for name in _TEMPLATES:
            with (templates / name).open("rb") as source, (out / name).open("wb") as target:
                shutil.copyfileobj(source, target)

    def data(self) -> dict[str, Any]:
        """Everything the page shows."""
        return {
            "generated": datetime.now(UTC).isoformat(timespec="seconds"),
            "sbb_version": __version__,
            "metrics": {k: {"label": label, "better": d} for k, (label, d) in METRICS.items()},
            "methods": _methods(),
            "cells": [_aggregate(group) for group in self._groups().values()],
            "model_auroc": self._model_auroc(),
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


def _aggregate(results: list[CellResult]) -> dict[str, Any]:
    """One row of the site: the cell over its seeds, metrics as mean and population std."""
    first = results[0].cell
    severity = list(_STATUS)
    bad = sorted((r for r in results if r.outcome != "ok"), key=lambda r: severity.index(r.outcome))
    status = _STATUS[bad[0].outcome] if bad else "self-reported"
    metrics = {}
    if not bad:  # metrics only when every seed finished
        for name in sorted({name for r in results for name in r.metrics}):
            values = np.array([r.metrics[name] for r in results if name in r.metrics])
            metrics[name] = {"mean": float(values.mean()), "std": float(values.std())}
    return {
        "version": first.benchmark_version,
        "track": first.track,
        "scenario": first.scenario.name,
        "dataset": first.scenario.data.name,
        "arch": first.scenario.arch.name,
        "T": first.scenario.repr.T,
        "attack": first.attack,
        "defense": first.defense,
        "seeds": sorted(r.cell.seed for r in results),
        "status": status,
        "reason": bad[0].reason if bad else None,
        "metrics": metrics,
        "sites": sorted({r.provenance.site for r in results}),
        "devices": sorted({r.provenance.device for r in results}),
    }


def _methods() -> dict[str, dict[str, Any]]:
    """Every registered attack and defense, from its card."""
    methods = {}
    for name in ATTACKS.names():
        attack = ATTACKS.get(name).card
        methods[name] = {
            "role": "attack",
            "group": str(attack.kind),
            "trigger_domain": attack.capabilities.trigger_domain,
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


def _common(card: dict[str, Any]) -> dict[str, Any]:
    keys = ("version", "title", "paper", "code_reference", "license_note")
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
