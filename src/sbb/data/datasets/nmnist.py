"""N-MNIST (Orchard et al., 2015), read from the original ATIS binary files."""

import hashlib
import logging
import shutil
import tempfile
import urllib.request
import zipfile
from functools import cached_property
from pathlib import Path

import numpy as np

from sbb.data.sample import DATASETS, EventDataset, EventSample, make_events
from sbb.utils.errors import SbbError

log = logging.getLogger(__name__)

_MENDELEY = "https://data.mendeley.com/public-files/datasets/468j46mzdv/files"
_ARCHIVES = {  # part: (url, md5)
    "Train": (
        f"{_MENDELEY}/39c25547-014b-4137-a934-9d29fa53c7a0/file_downloaded",
        "20959b8e626244a1b502305a9e6e2031",
    ),
    "Test": (
        f"{_MENDELEY}/05a4d654-7e03-4c15-bdfa-9bb2bcbea494/file_downloaded",
        "69ca8762b2fe404d9b9bad1103e97832",
    ),
}


def read_atis_bin(path: Path) -> np.ndarray:
    """Decode 40-bit ATIS events like SpikingJelly's load_ATIS_bin.

    Bits 39-32 are x, 31-24 are y, 23 is the polarity and 22-0 the timestamp in microseconds.
    """
    raw = np.fromfile(path, dtype=np.uint8).astype(np.int64)
    t = ((raw[2::5] & 127) << 16) | (raw[3::5] << 8) | raw[4::5]
    return make_events(raw[0::5], raw[1::5], t, raw[2::5] >> 7)


@DATASETS.register("nmnist")
class NMNIST(EventDataset):
    """N-MNIST. Full-dataset index order: sorted Train files, then sorted Test files.

    prepare() downloads Train.zip and Test.zip from Mendeley Data into root, checks their md5,
    and extracts them.
    """

    name = "nmnist"
    sensor_size = (34, 34, 2)
    num_classes = 10

    def prepare(self) -> None:
        """Download, verify and extract the archives that are missing."""
        for part, (url, md5) in _ARCHIVES.items():
            if (self.root / part).is_dir():
                continue
            archive = self.root / f"{part}.zip"
            if not archive.exists():
                self._download(url, archive)
            with archive.open("rb") as f:
                if hashlib.file_digest(f, "md5").hexdigest() != md5:
                    raise SbbError(f"{archive} has the wrong md5. Delete it and retry.")
            log.info("extracting %s", archive)
            with tempfile.TemporaryDirectory(dir=self.root) as tmp:
                with zipfile.ZipFile(archive) as z:
                    z.extractall(tmp)
                Path(tmp, part).rename(self.root / part)

    def _download(self, url: str, archive: Path) -> None:
        log.info("downloading %s", archive.name)
        self.root.mkdir(parents=True, exist_ok=True)
        partial = archive.with_suffix(".part")
        try:
            # Mendeley Data answers 403 to the default Python-urllib User-Agent.
            request = urllib.request.Request(url, headers={"User-Agent": "sbb"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as f:
                shutil.copyfileobj(response, f)
        except OSError as error:
            raise SbbError(
                f"could not download {archive.name} ({error}). Download Train.zip and Test.zip "
                f"from https://www.garrickorchard.com/datasets/n-mnist into {self.root}."
            ) from error
        partial.rename(archive)

    @cached_property
    def _files(self) -> list[Path]:
        files = [f for part in _ARCHIVES for f in sorted((self.root / part).glob("*/*.bin"))]
        if not files:
            raise SbbError(f"no N-MNIST files in {self.root}, call prepare() first")
        return files

    def __len__(self) -> int:
        return len(self._files)

    def labels(self) -> np.ndarray:
        """Labels from the class directory names."""
        return np.array([int(f.parent.name) for f in self._files])

    def __getitem__(self, index: int) -> EventSample:
        path = self._files[index]
        return EventSample(read_atis_bin(path), int(path.parent.name), index)
