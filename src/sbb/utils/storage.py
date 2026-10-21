"""Content-addressed artifact storage (docs/PLAN.md section 6.4)."""

import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path

from huggingface_hub import HfApi

log = logging.getLogger(__name__)


class ArtifactStore(ABC):
    """A place that holds artifact directories by key."""

    @abstractmethod
    def get(self, key: str) -> Path | None:
        """Return the local path of the artifact directory, or None if it is missing."""

    @abstractmethod
    def put(self, key: str, path: Path) -> None:
        """Store the artifact directory at path under key."""


class LocalStore(ArtifactStore):
    """Artifacts as directories under <root>/<key>/."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser()

    def get(self, key: str) -> Path | None:
        """Return <root>/<key> if it exists."""
        path = self.root / key
        return path if path.is_dir() else None

    def put(self, key: str, path: Path) -> None:
        """Move the directory at path into the store. The first writer of a key wins."""
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(path, self.root / key)
        except OSError:
            if self.get(key) is None:
                raise
            shutil.rmtree(path)


class HubStore(ArtifactStore):
    """Artifacts in a Hugging Face Hub repository, as <prefix>/<key>/ folders.

    Reads use revision, which should be a commit for results that must not change. Writes need a
    token (HF_TOKEN, or `hf auth login`) and commit to revision, which must then be a branch.

    Args:
        repo_id: The repository, for example "<hf-org>/sbb-zoo".
        revision: The commit or branch to read, and the branch to write.
        prefix: The folder of the artifacts in the repository, the benchmark version in the zoo.
        repo_type: "model" for the zoo.
    """

    def __init__(
        self, repo_id: str, revision: str = "main", prefix: str = "", repo_type: str = "model"
    ) -> None:
        self.api = HfApi()
        self.repo_id = repo_id
        self.revision = revision
        self.prefix = prefix
        self.repo_type = repo_type
        self._files: list[str] | None = None

    def _folder(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def _listing(self) -> list[str]:
        """The repository's files, read once."""
        if self._files is None:
            self._files = list(
                self.api.list_repo_files(
                    self.repo_id, revision=self.revision, repo_type=self.repo_type
                )
            )
        return self._files

    def keys(self) -> list[str]:
        """The keys of the artifacts under prefix."""
        start = f"{self.prefix}/" if self.prefix else ""
        inside = [f.removeprefix(start) for f in self._listing() if f.startswith(start)]
        return sorted({path.split("/")[0] for path in inside if "/" in path})  # folders only

    def get(self, key: str) -> Path | None:
        """Download the artifact's folder into the Hub cache, or return None if it is missing."""
        folder = self._folder(key)
        if not any(f.startswith(f"{folder}/") for f in self._listing()):
            return None
        local = self.api.snapshot_download(
            repo_id=self.repo_id,
            revision=self.revision,
            repo_type=self.repo_type,
            allow_patterns=[f"{folder}/*"],
        )
        return Path(local) / folder

    def put(self, key: str, path: Path) -> None:
        """Upload one artifact directory."""
        self.put_many({key: path})

    def put_many(self, artifacts: Mapping[str, Path]) -> None:
        """Upload artifact directories, by key, in one commit."""
        with tempfile.TemporaryDirectory() as staging:
            for key, path in artifacts.items():
                shutil.copytree(path, Path(staging) / self._folder(key))
            self.api.upload_folder(
                folder_path=staging,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                revision=self.revision,
                commit_message=f"Add {len(artifacts)} artifacts under {self.prefix or '/'}",
            )
        self._files = None


class ArtifactCache:
    """Looks up artifacts locally, then remotely, and builds them as a last resort."""

    def __init__(self, local: LocalStore, remote: ArtifactStore | None = None) -> None:
        self.local = local
        self.remote = remote

    def get_or_build(self, key: str, build: Callable[[Path], None]) -> Path:
        """Return the artifact directory of key. build(directory) writes it if it is missing."""
        if path := self.local.get(key):
            return path
        if self.remote and (remote_path := self.remote.get(key)):
            staging = self._staging()
            shutil.copytree(remote_path, staging, dirs_exist_ok=True)
            self.local.put(key, staging)
            return self.local.root / key
        log.info("building artifact %s", key)
        staging = self._staging()
        try:
            build(staging)
            self.local.put(key, staging)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return self.local.root / key

    def _staging(self) -> Path:
        self.local.root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=".staging-", dir=self.local.root))
