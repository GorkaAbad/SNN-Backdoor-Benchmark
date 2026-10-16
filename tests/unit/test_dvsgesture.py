import hashlib
import struct
import tarfile
from pathlib import Path

import numpy as np
import pytest

from sbb.data.datasets import dvsgesture
from sbb.data.datasets.dvsgesture import DVSGesture, label_of, read_aedat31
from sbb.utils.errors import SbbError

HEADER = (
    b"#!AER-DAT3.1\r\n#Format: RAW\r\n#Source 1: DVS128\r\n"
    b"#Start-Time: 2016-06-15 17:57:46 (TZ-0700)\r\n#!END-HEADER\r\n"
)


def _packet(
    events: list[tuple[int, int, int, int]], etype: int = 1, overflow: int = 0, valid: int = 1
) -> bytes:
    """One AEDAT 3.1 packet. events: (x, y, p, t)."""
    body = b"".join(
        struct.pack("<II", (x << 17) | (y << 2) | (p << 1) | valid, t) for x, y, p, t in events
    )
    n = len(events)
    return struct.pack("<HHIIIIII", etype, 1, 8, 4, overflow, n, n, n) + body


def _recording(path: Path, duration: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    times = np.arange(0, duration, 3)
    events = [
        (int(x), int(y), int(p), int(t))
        for x, y, p, t in zip(
            rng.integers(0, 128, len(times)),
            rng.integers(0, 128, len(times)),
            rng.integers(0, 2, len(times)),
            times,
            strict=True,
        )
    ]
    half = len(events) // 2
    path.write_bytes(
        HEADER + _packet(events[:half]) + _packet([(1, 1, 1, 0)], etype=0) + _packet(events[half:])
    )


def test_decoder_matches_spikingjelly(tmp_path: Path) -> None:
    from spikingjelly.datasets import load_aedat_v3

    path = tmp_path / "a.aedat"
    path.write_bytes(
        HEADER
        + _packet([(1, 2, 1, 10), (127, 0, 0, 20)])
        + _packet([(3, 3, 1, 30)], etype=0)
        + _packet([(5, 6, 0, 40)], valid=0)
        + _packet([(7, 8, 1, 5)], overflow=1)
    )
    ours = read_aedat31(path)
    theirs = load_aedat_v3(str(path))
    for got, key in zip(ours, "xytp", strict=True):
        assert np.array_equal(got, theirs[key]), key
    assert ours[2][-1] == 2**31 + 5


def test_labels_follow_spikingjelly() -> None:
    # SpikingJelly's folders "0" to "10", sorted as strings: CSV class 11 (other gestures) is 2.
    assert [label_of(c) for c in range(1, 12)] == [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 2]


TRIALS = {
    "train": {
        "user01_lab": [(3, 100, 200), (1, 0, 100), (8, 200, 260), (8, 260, 300), (11, 300, 400)],
        "user02_led": [(2, 0, 50), (9, 40, 120), (10, 60, 110)],  # class 10 inside class 9
    },
    "test": {"user24_natural": [(1, 0, 30), (11, 30, 60)]},
}


def _archive(root: Path, source: Path) -> str:
    folder = source / "DvsGesture"
    folder.mkdir(parents=True)
    for i, (split, trials) in enumerate(TRIALS.items()):
        listing = "".join(f"{stem}.aedat\n" for stem in trials) + "\n"
        (folder / f"trials_to_{split}.txt").write_text(listing)
        for j, (stem, rows) in enumerate(trials.items()):
            _recording(folder / f"{stem}.aedat", max(end for _, _, end in rows) + 20, 10 * i + j)
            csv = "class,startTime_usec,endTime_usec\n" + "".join(
                f"{c},{a},{b}\n" for c, a, b in rows
            )
            (folder / f"{stem}_labels.csv").write_text(csv)
    root.mkdir()
    with tarfile.open(root / "DvsGesture.tar.gz", "w:gz") as t:
        t.add(folder, arcname="DvsGesture")
    return hashlib.md5((root / "DvsGesture.tar.gz").read_bytes()).hexdigest()


def _spikingjelly_samples(source: Path, out: Path) -> list[tuple[Path, int]]:
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
    from torchvision.datasets.folder import find_classes, make_dataset

    samples = []
    for split, trials in TRIALS.items():
        for label in range(11):
            (out / split / str(label)).mkdir(parents=True)
        for stem in trials:
            folder = source / "DvsGesture"
            DVS128Gesture.split_aedat_files_to_np(
                stem,
                str(folder / f"{stem}.aedat"),
                str(folder / f"{stem}_labels.csv"),
                str(out / split),
            )
        _, class_to_idx = find_classes(str(out / split))  # all 11 folders, some of them empty
        found = make_dataset(str(out / split), class_to_idx, extensions=(".npz",), allow_empty=True)
        samples += [(Path(path), label) for path, label in found]
    return samples


def test_prepare_reproduces_spikingjelly_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    md5 = _archive(root, tmp_path / "source")
    monkeypatch.setattr(dvsgesture, "_URLS", ())
    monkeypatch.setattr(dvsgesture, "_COUNTS", (8, 2))
    monkeypatch.setattr(dvsgesture, "_MD5", "0" * 32)
    with pytest.raises(SbbError, match="wrong md5"):
        DVSGesture(root).prepare()
    monkeypatch.setattr(dvsgesture, "_MD5", md5)
    dataset = DVSGesture(root)
    dataset.prepare()

    reference = _spikingjelly_samples(tmp_path / "source", tmp_path / "sj")
    assert len(dataset) == len(reference) == 10
    for i, (path, label) in enumerate(reference):
        sample, theirs = dataset[i], np.load(path)
        assert sample.label == label and sample.index == i
        for key in "xytp":
            assert np.array_equal(sample.events[key], theirs[key]), (i, key)
    assert dataset.labels().tolist() == [label for _, label in reference]
    assert sorted(p.name for p in root.iterdir()) == ["DvsGesture.tar.gz", "samples_v1"]

    DVSGesture(root).prepare()  # converted samples are reused, nothing is downloaded or extracted
    with pytest.raises(IndexError):
        dataset[10]


def test_prepare_checks_the_sample_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    monkeypatch.setattr(dvsgesture, "_MD5", _archive(root, tmp_path / "source"))
    monkeypatch.setattr(dvsgesture, "_URLS", ())
    with pytest.raises(SbbError, match="1176"):
        DVSGesture(root).prepare()
    assert sorted(p.name for p in root.iterdir()) == ["DvsGesture.tar.gz"]
