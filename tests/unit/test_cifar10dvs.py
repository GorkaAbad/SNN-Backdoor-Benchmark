import hashlib
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from sbb.data.datasets.cifar10dvs import CIFAR10DVS, CLASSES, read_aedat2
from sbb.utils.errors import SbbError

HEADER = (
    b"#!AER-DAT2.0\r\n# This is a raw AE data file - do not edit\r\n"
    b"# Data format is int32 address, int32 timestamp (8 bytes total), repeated for each event\r\n"
    b"# Timestamps tick is 1 us\r\n# created Tue Apr 29 11:36:59 CEST 2008\r\n"
)


def _write_aedat2(path: Path, events: list[tuple[int, int, int, int]]) -> None:
    """events: (x_raw, y_raw, polarity bit, t) as jAER stores them."""
    body = b"".join(struct.pack(">II", (y << 8) | (x << 1) | bit, t) for x, y, bit, t in events)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(HEADER + body)


def test_decoder_matches_spikingjelly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS as SpikingJellyCIFAR10DVS

    rng = np.random.default_rng(0)
    xs, ys, bits = rng.integers(0, 128, 500), rng.integers(0, 128, 500), rng.integers(0, 2, 500)
    events = [
        (int(x), int(y), int(b), t) for t, (x, y, b) in enumerate(zip(xs, ys, bits, strict=True))
    ]
    events += [(0, 0, 0, 500), (127, 127, 1, 501), (0, 127, 0, 502), (127, 0, 1, 503)]
    path = tmp_path / "a.aedat"
    _write_aedat2(path, events)
    # SpikingJelly 0.0.0.0.14 calls np.fromstring in binary mode, which NumPy 2.3 removed.
    monkeypatch.setattr(np, "fromstring", np.frombuffer)
    theirs = SpikingJellyCIFAR10DVS.load_origin_data(str(path))
    ours = read_aedat2(path)
    for key in "xytp":
        assert np.array_equal(ours[key], np.asarray(theirs[key]).astype(np.int64)), key


def test_decoder_uses_spikingjelly_convention(tmp_path: Path) -> None:
    path = tmp_path / "a.aedat"
    _write_aedat2(path, [(5, 3, 1, 7)])
    (event,) = read_aedat2(path).tolist()
    assert event == (124, 122, 7, 0)  # x = 127 - y_raw, y = 127 - x_raw, p = 1 - bit
    path.write_bytes(HEADER + b"\x00\x00\x00")
    with pytest.raises(SbbError, match="truncated"):
        read_aedat2(path)


def test_index_order_and_labels_match_spikingjelly(tmp_path: Path) -> None:
    from torchvision.datasets.folder import find_classes, make_dataset

    for c, cls in enumerate(CLASSES):
        for n in (0, 1, 2, 10):
            # The single event's t encodes the class and the file number.
            _write_aedat2(tmp_path / cls / f"cifar10_{cls}_{n}.aedat", [(1, 1, 1, 1000 * c + n)])
    dataset = CIFAR10DVS(tmp_path)
    # SpikingJelly orders and labels samples with torchvision's DatasetFolder machinery.
    reference = make_dataset(str(tmp_path), find_classes(str(tmp_path))[1], extensions=(".aedat",))
    expected = [
        (
            1000 * CLASSES.index(Path(path).parent.name) + int(Path(path).stem.rsplit("_", 1)[1]),
            label,
        )
        for path, label in reference
    ]
    ours = [(int(dataset[i].events["t"][0]), dataset[i].label) for i in range(len(dataset))]
    assert ours == expected
    assert [t for t, _ in ours[:4]] == [0, 1, 10, 2]  # file names sort as strings
    assert dataset.labels().tolist() == [label for _, label in expected]


def test_prepare_checks_md5_and_extracts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    md5s = {}
    for cls in ("airplane", "truck"):
        _write_aedat2(source / cls / f"cifar10_{cls}_0.aedat", [(2, 3, 0, 9)])
        archive = shutil.make_archive(str(tmp_path / "root" / cls), "zip", source, cls)
        md5s[cls] = hashlib.md5(Path(archive).read_bytes()).hexdigest()
    root = tmp_path / "root"
    monkeypatch.setattr(
        CIFAR10DVS, "_archives", {"airplane": ((), "0" * 32), "truck": ((), md5s["truck"])}
    )
    with pytest.raises(SbbError, match="wrong md5"):
        CIFAR10DVS(root).prepare()
    monkeypatch.setattr(CIFAR10DVS, "_archives", {c: ((), m) for c, m in md5s.items()})
    dataset = CIFAR10DVS(root)
    dataset.prepare()
    assert dataset.labels().tolist() == [0, 9] and len(dataset) == 2
    assert sorted(p.name for p in root.iterdir()) == [
        "airplane",
        "airplane.zip",
        "truck",
        "truck.zip",
    ]
