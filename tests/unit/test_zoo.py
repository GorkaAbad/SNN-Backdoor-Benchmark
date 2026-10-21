"""HubStore and sbb zoo with a fake Hub API that keeps the repository in a local folder."""

import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from sbb.cli import app
from sbb.utils import storage
from sbb.utils.storage import ArtifactCache, HubStore, LocalStore


class FakeHub:
    """The parts of HfApi that HubStore uses, on <root>/<repo_id>/."""

    root: Path
    commits = 0

    def __init__(self, token: str | None = None) -> None:
        pass

    def _repo(self, repo_id: str) -> Path:
        return FakeHub.root / repo_id

    def list_repo_files(self, repo_id: str, revision: str, repo_type: str) -> list[str]:
        repo = self._repo(repo_id)
        return [str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file()]

    def snapshot_download(self, repo_id: str, revision: str, repo_type: str, **kw: Any) -> str:
        target = FakeHub.root / "downloads" / repo_id
        shutil.copytree(self._repo(repo_id), target, dirs_exist_ok=True)
        return str(target)

    def upload_folder(self, folder_path: Path, repo_id: str, repo_type: str, **kw: Any) -> str:
        shutil.copytree(folder_path, self._repo(repo_id), dirs_exist_ok=True)
        FakeHub.commits += 1
        return "commit"


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> type[FakeHub]:
    FakeHub.root, FakeHub.commits = tmp_path / "hub", 0
    (FakeHub.root / "org" / "zoo").mkdir(parents=True)
    monkeypatch.setattr(storage, "HfApi", FakeHub)
    return FakeHub


def _artifact(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(name.encode())
    return path


def test_hub_store_puts_many_in_one_commit_and_gets_them(
    hub: type[FakeHub], tmp_path: Path
) -> None:
    store = HubStore("org/zoo", revision="main", prefix="v0.9")
    assert store.get("abc") is None
    store.put_many(
        {"abc": _artifact(tmp_path / "a", "abc"), "def": _artifact(tmp_path / "a", "def")}
    )
    assert hub.commits == 1
    fresh = HubStore("org/zoo", revision="main", prefix="v0.9")  # the file list is read once
    path = fresh.get("abc")
    assert path is not None and (path / "model.safetensors").read_bytes() == b"abc"
    assert (hub.root / "org" / "zoo" / "v0.9" / "def" / "model.safetensors").exists()


def test_cache_takes_an_artifact_from_the_hub_before_building(
    hub: type[FakeHub], tmp_path: Path
) -> None:
    HubStore("org/zoo", "main", "v0.9").put("abc", _artifact(tmp_path / "a", "abc"))
    cache = ArtifactCache(LocalStore(tmp_path / "cache"), HubStore("org/zoo", "main", "v0.9"))

    def build(directory: Path) -> None:
        raise AssertionError("the artifact is on the Hub")

    path = cache.get_or_build("abc", build)
    assert (path / "model.safetensors").read_bytes() == b"abc"


def test_zoo_push_and_pull_the_artifacts_of_a_version(hub: type[FakeHub], tmp_path: Path) -> None:
    out, site = tmp_path / "results", tmp_path / "site.yaml"
    site.write_text(f"name: test\ndevice: cpu\ncache_dir: {tmp_path / 'cache'}\n")
    smoke = Path(__file__).parents[2] / "benchmarks" / "smoke.yaml"
    args = ["--site", str(site), "--out", str(out)]
    assert CliRunner().invoke(app, ["run", str(smoke), *args]).exit_code == 0
    pushed = CliRunner().invoke(app, ["zoo", "push", "smoke-0.1.0", "--repo", "org/zoo", *args])
    assert pushed.exit_code == 0, pushed.output
    assert hub.commits == 1
    on_hub = {p.name for p in (hub.root / "org" / "zoo" / "smoke-0.1.0").iterdir()}
    assert on_hub, "the models of the results are on the Hub"

    elsewhere = tmp_path / "other.yaml"
    elsewhere.write_text(f"name: other\ndevice: cpu\ncache_dir: {tmp_path / 'other'}\n")
    pull = ["zoo", "pull", "smoke-0.1.0", "--repo", "org/zoo", "--site", str(elsewhere)]
    pulled = CliRunner().invoke(app, pull)
    assert pulled.exit_code == 0, pulled.output
    assert {p.name for p in (tmp_path / "other" / "artifacts").iterdir()} >= on_hub
