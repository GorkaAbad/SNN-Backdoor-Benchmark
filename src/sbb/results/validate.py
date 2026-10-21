"""sbb validate: checks the cards, results, manifests and records contributors add."""

from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import ValidationError

from sbb.core.cards import AttackCard, DefenseCard
from sbb.core.registry import ATTACKS
from sbb.core.result import CellResult
from sbb.results.bundle import check_manifest
from sbb.results.replication import ReplicationRecord
from sbb.results.verification import Verification


def _check(path: Path) -> list[str] | None:
    """Problems of one file, or None if it is not a kind sbb validates."""
    parts = set(path.parts)
    if path.suffix == ".yaml" and "cards" in parts:
        card = AttackCard if "attacks" in parts else DefenseCard
        card.model_validate(yaml.safe_load(path.read_text()))
        return []
    if path.suffix == ".yaml" and "replications" in parts:
        record = ReplicationRecord.load(path)
        if record.method not in ATTACKS.names():
            return [f"{record.method} is not a registered attack"]
        record.cells()
        return []
    if path.suffix != ".json":
        return None
    if "manifests" in parts:
        return check_manifest(path)
    if "verifications" in parts:
        Verification.model_validate_json(path.read_text())
        return []
    result = CellResult.model_validate_json(path.read_text())
    problems = []
    if result.key != result.cell.key():
        problems.append("the key is not the hash of the cell")
    if path.stem != result.key:
        problems.append("the file name is not the key")
    return problems


def validate(paths: Iterable[Path]) -> tuple[int, dict[Path, list[str]]]:
    """Check files and the files under directories. Return the number checked and the problems."""
    files = [
        f for p in paths for f in ([p] if p.is_file() else sorted(p.rglob("*"))) if f.is_file()
    ]
    checked, problems = 0, {}
    for file in files:
        try:
            found = _check(file)
        except (ValidationError, ValueError, KeyError) as error:
            found = [f"{type(error).__name__}: {error}"]
        if found is None:
            continue
        checked += 1
        if found:
            problems[file] = found
    return checked, problems
