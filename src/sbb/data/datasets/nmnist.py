"""N-MNIST (Orchard et al., 2015), read from the original ATIS binary files."""

from pathlib import Path

import numpy as np

from sbb.data.datasets.common import RecordingFolder
from sbb.data.sample import DATASETS, make_events

_MENDELEY = "https://data.mendeley.com/public-files/datasets/468j46mzdv/files"


def read_atis_bin(path: Path) -> np.ndarray:
    """Decode 40-bit ATIS events like SpikingJelly's load_ATIS_bin.

    Bits 39-32 are x, 31-24 are y, 23 is the polarity and 22-0 the timestamp in microseconds.
    """
    raw = np.fromfile(path, dtype=np.uint8).astype(np.int64)
    t = ((raw[2::5] & 127) << 16) | (raw[3::5] << 8) | raw[4::5]
    return make_events(raw[0::5], raw[1::5], t, raw[2::5] >> 7)


@DATASETS.register("nmnist")
class NMNIST(RecordingFolder):
    """N-MNIST. Full-dataset index order: sorted Train files, then sorted Test files.

    prepare() downloads Train.zip and Test.zip from Mendeley Data into root, checks their md5,
    and extracts them.
    """

    name = "nmnist"
    sensor_size = (34, 34, 2)
    num_classes = 10
    _archives = {
        "Train": (
            (f"{_MENDELEY}/39c25547-014b-4137-a934-9d29fa53c7a0/file_downloaded",),
            "20959b8e626244a1b502305a9e6e2031",
        ),
        "Test": (
            (f"{_MENDELEY}/05a4d654-7e03-4c15-bdfa-9bb2bcbea494/file_downloaded",),
            "69ca8762b2fe404d9b9bad1103e97832",
        ),
    }
    _pattern = "*/*.bin"
    _manual = (
        "Download Train.zip and Test.zip from https://www.garrickorchard.com/datasets/n-mnist "
        "into {root}."
    )

    def _label(self, path: Path) -> int:
        return int(path.parent.name)

    def _read(self, path: Path) -> np.ndarray:
        return read_atis_bin(path)
