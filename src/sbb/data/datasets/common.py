"""Pieces shared by the dataset adapters: downloads, unpacking and file-per-sample datasets."""

import hashlib
import logging
import shutil
import tarfile
import tempfile
import urllib.request
import uuid
import zipfile
from abc import abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import numpy as np

from sbb.data.sample import EventDataset, EventSample
from sbb.utils.errors import SbbError
from sbb.utils.storage import LocalStore

log = logging.getLogger(__name__)


def md5_of(path: Path) -> str:
    """Return the md5 hex digest of a file."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "md5").hexdigest()


def fetch(archive: Path, md5: str, urls: Sequence[str], manual: str) -> None:
    """Make sure archive exists and has the expected md5.

    A missing archive is downloaded from the first URL that delivers the right file. An existing
    archive is checked and never replaced, so a manual download is not lost. Runs that share a root
    may download at the same time: each one writes its own partial file.

    Args:
        archive: Path of the archive.
        md5: Its expected md5 hex digest.
        urls: Download URLs, tried in order.
        manual: Manual-download instructions, shown when every URL fails.
    """
    if archive.exists():
        if md5_of(archive) != md5:
            raise SbbError(f"{archive} has the wrong md5, expected {md5}. Delete it and retry.")
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        partial = archive.with_name(f"{archive.name}.{uuid.uuid4().hex}.part")
        try:
            _download(url, partial, md5)
            partial.replace(archive)
            return
        except (OSError, SbbError) as error:
            log.warning("could not download %s from %s: %s", archive.name, url, error)
        finally:
            partial.unlink(missing_ok=True)
    raise SbbError(f"could not download {archive.name}. {manual}")


def _download(url: str, partial: Path, md5: str) -> None:
    log.info("downloading %s", url)
    # Mendeley Data answers 403 to the default Python-urllib User-Agent.
    request = urllib.request.Request(url, headers={"User-Agent": "sbb"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:  # figshare's bot check answers 202 with an empty body
            raise SbbError(f"HTTP {response.status}")
        with partial.open("wb") as f:
            shutil.copyfileobj(response, f)
    if md5_of(partial) != md5:
        raise SbbError(f"the download has the wrong md5, expected {md5}")


@contextmanager
def unpacked(archive: Path, workdir: Path) -> Iterator[Path]:
    """Extract a zip or tar archive into a temporary directory under workdir, deleted on exit.

    Move what you keep out of the directory before the context ends. Members cannot land outside
    it: zipfile drops absolute and ".." path parts, and tar extraction uses the "data" filter.
    """
    log.info("extracting %s", archive.name)
    with tempfile.TemporaryDirectory(prefix=".unpack-", dir=workdir) as tmp:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as z:
                z.extractall(tmp)
        else:
            with tarfile.open(archive) as t:
                t.extractall(tmp, filter="data")
        yield Path(tmp)


class RecordingFolder(EventDataset):
    """A dataset of one raw recording file per sample, unpacked from one zip per folder.

    Subclasses list their folders in _archives, in index order. The index order is the folders in
    that order, and inside each folder the files matching _pattern sorted by path.
    """

    _archives: ClassVar[dict[str, tuple[Sequence[str], str]]]  # folder: (URLs, md5) of <folder>.zip
    _pattern: ClassVar[str]
    _manual: ClassVar[str]  # manual-download instructions, formatted with root

    def prepare(self) -> None:
        """Download, check and unpack the zip of every missing folder. Concurrent runs are safe."""
        for folder, (urls, md5) in self._archives.items():
            if (self.root / folder).is_dir():
                continue
            archive = self.root / f"{folder}.zip"
            fetch(archive, md5, urls, self._manual.format(root=self.root))
            with unpacked(archive, self.root) as tmp:
                LocalStore(self.root).put(folder, tmp / folder)  # the first run to finish wins

    @cached_property
    def _files(self) -> list[Path]:
        files = [f for d in self._archives for f in sorted((self.root / d).glob(self._pattern))]
        if not files:
            raise SbbError(f"no {self.name} files in {self.root}, call prepare() first")
        return files

    def __len__(self) -> int:
        return len(self._files)

    def labels(self) -> np.ndarray:
        """Labels from the file paths, without loading events."""
        return np.array([self._label(f) for f in self._files])

    def __getitem__(self, index: int) -> EventSample:
        if not 0 <= index < len(self):
            raise IndexError(index)
        path = self._files[index]
        return EventSample(self._read(path), self._label(path), index)

    @abstractmethod
    def _label(self, path: Path) -> int:
        """Label of the recording at path."""

    @abstractmethod
    def _read(self, path: Path) -> np.ndarray:
        """Events of the recording at path, with EVENT_DTYPE."""
