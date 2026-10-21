"""Manifests: the results a submission contributes, with their hashes (docs/PLAN.md 6.10)."""

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from sbb.core.result import CellResult
from sbb.utils.errors import SbbError
from sbb.utils.hashing import file_sha256, sha256_hex

_MODEL_ARTIFACTS = ("clean_model", "model", "defended_model")


class Manifest(BaseModel, frozen=True):
    """A submission: results of one benchmark version, by path and sha256.

    Attributes:
        bundle_id: The first 16 hex digits of the hash of the version and the result hashes.
        benchmark_version: The version of every result.
        submitter: Who ran the cells, from their provenance.
        submitter_org: The submitter's organization.
        plugins: Name to version of every attack and defense in the results.
        results: Path relative to the results root, and sha256, of every result file.
        models: Optional Hub references (model_key, repo, revision) of the models.
    """

    model_config = ConfigDict(extra="forbid")
    bundle_id: str
    benchmark_version: str
    submitter: str
    submitter_org: str
    plugins: dict[str, str]
    results: list[dict[str, str]]
    models: list[dict[str, str]] = []


def bundle_id(version: str, hashes: Sequence[str]) -> str:
    """The content address of a manifest."""
    return sha256_hex({"version": version, "results": sorted(hashes)})[:16]


def make_manifest(
    root: Path, results: Sequence[CellResult], models_repo: str | None, models_revision: str
) -> Manifest:
    """The manifest of results stored under root as <version>/<key>.json."""
    versions = {r.cell.benchmark_version for r in results}
    submitters = {(r.provenance.submitter, r.provenance.submitter_org) for r in results}
    if len(versions) != 1 or len(submitters) != 1:
        raise SbbError(
            f"a bundle needs one version and one submitter, got {versions}, {submitters}"
        )
    (version,), ((submitter, org),) = versions, submitters
    files = [
        {"path": f"{version}/{r.key}.json", "sha256": file_sha256(root / version / f"{r.key}.json")}
        for r in sorted(results, key=lambda r: r.key)
    ]
    plugins = {}
    for r in results:
        plugins[r.cell.attack] = r.cell.attack_version
        plugins[r.cell.defense] = r.cell.defense_version
    keys = sorted({r.artifacts[a] for r in results for a in _MODEL_ARTIFACTS if a in r.artifacts})
    models = (
        [{"model_key": k, "repo": models_repo, "revision": models_revision} for k in keys]
        if models_repo
        else []
    )
    return Manifest(
        bundle_id=bundle_id(version, [f["sha256"] for f in files]),
        benchmark_version=version,
        submitter=submitter,
        submitter_org=org,
        plugins=dict(sorted(plugins.items())),
        results=files,
        models=models,
    )


def check_manifest(path: Path) -> list[str]:
    """Problems of a manifest at <root>/manifests/<bundle_id>.json, an empty list if none."""
    manifest = Manifest.model_validate_json(path.read_text())
    root, problems = path.parent.parent, []
    if path.stem != manifest.bundle_id:
        problems.append(f"{path.name} does not carry the bundle id {manifest.bundle_id}")
    for entry in manifest.results:
        file = root / entry["path"]
        if not file.exists():
            problems.append(f"{entry['path']} is missing")
        elif file_sha256(file) != entry["sha256"]:
            problems.append(f"{entry['path']} does not match its sha256")
        else:
            cell = CellResult.model_validate_json(file.read_text()).cell
            if cell.benchmark_version != manifest.benchmark_version:
                problems.append(f"{entry['path']} belongs to {cell.benchmark_version}")
            for name, version in (
                (cell.attack, cell.attack_version),
                (cell.defense, cell.defense_version),
            ):
                if manifest.plugins.get(name) != version:
                    problems.append(f"{entry['path']} uses {name} {version}, not in the plugins")
    hashes = [entry["sha256"] for entry in manifest.results]
    if bundle_id(manifest.benchmark_version, hashes) != manifest.bundle_id:
        problems.append("the bundle id does not match the results")
    return problems
