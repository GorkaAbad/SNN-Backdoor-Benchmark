"""Content-addressed artifact storage (docs/PLAN.md section 6.4)."""

import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

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
