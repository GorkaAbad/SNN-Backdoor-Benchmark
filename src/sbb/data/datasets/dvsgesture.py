"""DVS128 Gesture (Amir et al., 2017), read from the original AEDAT 3.1 recordings. CC BY 4.0."""

import logging
import struct
from collections import Counter, defaultdict
from functools import cached_property
from pathlib import Path

import numpy as np

from sbb.data.datasets.common import fetch, unpacked
from sbb.data.sample import DATASETS, EventDataset, EventSample, make_events
from sbb.utils.errors import SbbError
from sbb.utils.storage import LocalStore

log = logging.getLogger(__name__)

_URLS = (  # the IBM Box file behind the folder link below, then a Dropbox mirror of the same file
    "https://ibm.ent.box.com/index.php?rm=box_download_shared_file"
    "&shared_name=3hiq58ww1pbbjrinh367ykfdf60xsfm8&file_id=f_211521748942",
    "https://www.dropbox.com/s/cct5kyilhtsliup/DvsGesture.tar.gz?dl=1",
)
_MD5 = "8a5c71fb11e24e5ca5b11866ca6c00a1"
_MANUAL = (
    "Download DvsGesture.tar.gz (2 938 955 106 bytes) from the IBM Box folder "
    "https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794 into {root}."
)
_COUNTS = (1176, 288)  # train and test samples
_SAMPLES = "samples_v1"  # folder of the converted samples. Rename it when the conversion changes.

# SpikingJelly saves a sample of CSV class c (1 to 11) in the folder str(c - 1), and torchvision
# numbers those folders in string order: "0", "1", "10", "2", ..., "9".
_FOLDERS = sorted(str(c) for c in range(11))


def label_of(csv_class: int) -> int:
    """Return the label SpikingJelly gives to a sample of CSV class csv_class (1 to 11)."""
    return _FOLDERS.index(str(csv_class - 1))


def read_aedat31(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode the polarity events of an AEDAT 3.1 file like SpikingJelly's load_aedat_v3.

    Returns x, y, t and p in file order. Like SpikingJelly, it decodes every slot of a polarity
    packet without checking the validity bit, skips other packet types, and adds the packet's
    timestamp overflow as bit 31 of t, in microseconds.
    """
    with path.open("rb") as f:
        while (line := f.readline()).startswith(b"#") and line != b"#!END-HEADER\r\n":
            pass
        data = f.read()
    parts, pos = [], 0
    while pos < len(data):
        event_type, _, size, _, overflow, capacity, _, _ = struct.unpack_from(
            "<HHIIIIII", data, pos
        )
        pos += 28
        if event_type == 1:  # polarity events: uint32 data, uint32 timestamp
            words = np.frombuffer(data, "<u4", 2 * capacity, pos).reshape(-1, 2).astype(np.int64)
            d = words[:, 0]
            parts.append(
                np.stack(
                    [
                        (d >> 17) & 0x7FFF,
                        (d >> 2) & 0x7FFF,
                        words[:, 1] | (overflow << 31),
                        (d >> 1) & 1,
                    ]
                )
            )
        pos += capacity * size
    x, y, t, p = np.concatenate(parts, axis=1)
    return x, y, t, p


@DATASETS.register("dvsgesture")
class DVSGesture(EventDataset):
    """DVS128 Gesture, 1464 gesture samples cut from 122 recordings of 29 subjects.

    The index order and the labels are SpikingJelly's. A sample is one row of a recording's label
    CSV, with the events t_start <= t < t_end. Train samples (users 01 to 23) come first, then test
    samples (users 24 to 29). Inside a split the order is by label folder and then by SpikingJelly's
    file name <recording>_<k>, both as strings. Labels follow SpikingJelly's numbering, so the
    targets of the ported papers carry over unchanged:

        label:     0  1   2  3  4  5  6  7  8  9  10
        CSV class: 1  2  11  3  4  5  6  7  8  9  10

    CSV classes 1 to 11 are hand clap, right hand wave, left hand wave, right arm clockwise, right
    arm counterclockwise, left arm clockwise, left arm counterclockwise, arm roll, air drums, air
    guitar and other gestures. Arm roll is labelled twice per recording (forward and backward), so
    it has twice as many samples. In user22_led the class 9 window contains the class 10 window,
    and both samples keep their events, as in SpikingJelly.

    prepare() downloads DvsGesture.tar.gz, checks its md5, and cuts the recordings once into one
    .npy file per sample. The extracted recordings are deleted afterwards.
    """

    name = "dvsgesture"
    sensor_size = (128, 128, 2)
    num_classes = 11

    def prepare(self) -> None:
        """Download and check the archive, then convert the recordings unless already done."""
        if (self.root / _SAMPLES / "labels.npy").exists():
            return
        archive = self.root / "DvsGesture.tar.gz"
        fetch(archive, _MD5, _URLS, _MANUAL.format(root=self.root))
        with unpacked(archive, self.root) as tmp:
            _convert(tmp / "DvsGesture", tmp / _SAMPLES)
            LocalStore(self.root).put(_SAMPLES, tmp / _SAMPLES)  # the first run to finish wins

    @cached_property
    def _labels(self) -> np.ndarray:
        path = self.root / _SAMPLES / "labels.npy"
        if not path.exists():
            raise SbbError(f"no dvsgesture samples in {self.root}, call prepare() first")
        labels: np.ndarray = np.load(path, allow_pickle=False)
        return labels

    def __len__(self) -> int:
        return len(self._labels)

    def labels(self) -> np.ndarray:
        """Labels of all samples, without loading events."""
        return self._labels.copy()

    def __getitem__(self, index: int) -> EventSample:
        if not 0 <= index < len(self):
            raise IndexError(index)
        events = np.load(self.root / _SAMPLES / f"{index:05d}.npy", allow_pickle=False)
        return EventSample(events, int(self._labels[index]), index)


def _convert(source: Path, out: Path) -> None:
    """Cut every recording into samples, in SpikingJelly's order, and save them in out."""
    samples = []  # (sort key, recording, t_start, t_end, label)
    for split, listing in enumerate(("trials_to_train.txt", "trials_to_test.txt")):
        for line in (source / listing).read_text().splitlines():
            if not (trial := line.strip()):
                continue
            recording = Path(trial).stem
            rows = np.loadtxt(
                source / f"{recording}_labels.csv",
                dtype=np.int64,
                delimiter=",",
                skiprows=1,
                ndmin=2,
            )
            repeats: Counter[int] = Counter()
            for csv_class, start, end in rows.tolist():
                file_name = f"{recording}_{repeats[csv_class]}.npz"  # SpikingJelly's name
                repeats[csv_class] += 1
                key = (split, str(csv_class - 1), file_name)
                samples.append((key, recording, start, end, label_of(csv_class)))
    samples.sort(key=lambda sample: sample[0])
    counts = tuple(sum(sample[0][0] == split for sample in samples) for split in (0, 1))
    if counts != _COUNTS:
        raise SbbError(f"expected {_COUNTS} train and test samples, found {counts}")

    windows: defaultdict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for index, (_, recording, start, end, _) in enumerate(samples):
        windows[recording].append((index, start, end))
    out.mkdir()
    for recording, cuts in windows.items():
        log.info("cutting %s", recording)
        x, y, t, p = read_aedat31(source / f"{recording}.aedat")
        for index, start, end in cuts:
            inside = (t >= start) & (t < end)
            events = make_events(x[inside], y[inside], t[inside], p[inside])
            np.save(out / f"{index:05d}.npy", events)
    np.save(out / "labels.npy", np.array([sample[4] for sample in samples]))
