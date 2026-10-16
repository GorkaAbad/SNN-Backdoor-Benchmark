"""CIFAR10-DVS (Li et al., 2017), read from the original AEDAT 2.0 files. License: CC BY 4.0."""

from pathlib import Path

import numpy as np

from sbb.data.datasets.common import RecordingFolder
from sbb.data.sample import DATASETS, make_events
from sbb.utils.errors import SbbError

_FIGSHARE = "https://ndownloader.figshare.com/files"
_ZIPS = {  # class, in CIFAR-10 label order: (figshare file id, md5)
    "airplane": (7712788, "0afd5c4bf9ae06af762a77b180354fdd"),
    "automobile": (7712791, "8438dfeba3bc970c94962d995b1b9bdd"),
    "bird": (7712794, "a9c207c91c55b9dc2002dc21c684d785"),
    "cat": (7712812, "52c63c677c2b15fa5146a8daf4d56687"),
    "deer": (7712815, "b6bf21f6c04d21ba4e23fc3e36c8a4a3"),
    "dog": (7712818, "f379ebdf6703d16e0a690782e62639c3"),
    "frog": (7712842, "cad6ed91214b1c7388a5f6ee56d08803"),
    "horse": (7712851, "e7cbbf77bec584ffbf913f00e682782a"),
    "ship": (7712836, "41c7bd7d6b251be82557c6cce9a7d5c9"),
    "truck": (7712839, "89f3922fd147d9aeff89e76a2b0b70a7"),
}
CLASSES = tuple(_ZIPS)


def read_aedat2(path: Path) -> np.ndarray:
    """Decode the AEDAT 2.0 events of one recording like SpikingJelly's CIFAR10DVS.

    Each event is a big-endian address and a timestamp in microseconds. The address holds the
    polarity bit in bit 0, x in bits 1-7 and y in bits 8-14, as jAER writes them. SpikingJelly
    transposes and flips the axes, x = 127 - y_raw and y = 127 - x_raw (see
    github.com/jackd/events-tfds/pull/1), and inverts the polarity, so p = 1 is an ON event.
    Every ported paper places its triggers in those coordinates, so sbb keeps them.
    """
    raw = path.read_bytes()
    start = 0
    # Skip the header lines. Address bits 15-31 are 0, so the data never starts with "#".
    while raw.startswith(b"#", start):
        start = raw.index(b"\n", start) + 1
    if (len(raw) - start) % 8:
        raise SbbError(f"{path} is truncated")
    words = np.frombuffer(raw, ">u4", offset=start).astype(np.int64)
    address = words[0::2]
    x_raw, y_raw = (address >> 1) & 0x7F, (address >> 8) & 0x7F
    return make_events(127 - y_raw, 127 - x_raw, words[1::2], 1 - (address & 1))


@DATASETS.register("cifar10dvs")
class CIFAR10DVS(RecordingFolder):
    """CIFAR10-DVS, 1000 recordings of each CIFAR-10 class on a 128x128 DVS.

    The index order and the labels are SpikingJelly's: classes in alphabetical order, which is also
    the CIFAR-10 label order, and inside a class the files cifar10_<class>_<n>.aedat sorted as
    strings (n = 0, 1, 10, 100, ...). The split file cifar10dvs_9to1_v1 is therefore the split of
    SpikingJelly's split_to_train_test_set(0.9), which every ported paper uses.

    prepare() downloads the ten class zips from figshare into root, checks their md5 and extracts
    them. Events are decoded on access, which costs about 1 ms per recording.
    """

    name = "cifar10dvs"
    sensor_size = (128, 128, 2)
    num_classes = 10
    _archives = {cls: ((f"{_FIGSHARE}/{file_id}",), md5) for cls, (file_id, md5) in _ZIPS.items()}
    _pattern = "*.aedat"
    _manual = (
        "Download the ten class zips (airplane.zip to truck.zip) of "
        "https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671 into {root}."
    )

    def _label(self, path: Path) -> int:
        return CLASSES.index(path.parent.name)

    def _read(self, path: Path) -> np.ndarray:
        return read_aedat2(path)
