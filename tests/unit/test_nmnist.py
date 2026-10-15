import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

from sbb.data.datasets import nmnist
from sbb.data.datasets.nmnist import NMNIST, read_atis_bin
from sbb.utils.errors import SbbError


def _write_bin(path: Path, events: list[tuple[int, int, int, int]]) -> None:
    raw = bytearray()
    for x, y, p, t in events:
        raw += bytes([x, y, (p << 7) | (t >> 16), (t >> 8) & 255, t & 255])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(raw))


def test_atis_decoding_matches_spikingjelly(tmp_path: Path) -> None:
    from spikingjelly.datasets import load_ATIS_bin

    path = tmp_path / "a.bin"
    _write_bin(path, [(3, 4, 1, 10), (33, 0, 0, 70_000), (0, 33, 1, 8_000_000)])
    ours, theirs = read_atis_bin(path), load_ATIS_bin(str(path))
    for k in "xytp":
        assert np.array_equal(ours[k], theirs[k])


def test_index_order_is_train_then_test(tmp_path: Path) -> None:
    for part, label, name in [("Test", 1, "00001"), ("Train", 7, "00002"), ("Train", 2, "00009")]:
        _write_bin(tmp_path / part / str(label) / f"{name}.bin", [(1, 1, 1, 5)])
    dataset = NMNIST(tmp_path)
    dataset.prepare()  # extracted parts exist, so nothing is downloaded
    assert dataset.labels().tolist() == [2, 7, 1]
    assert dataset[2].label == 1 and dataset[2].index == 2 and len(dataset) == 3


def _zip(root: Path, part: str) -> str:
    _write_bin(root / "src" / part / "4" / "00001.bin", [(2, 3, 0, 9)])
    archive = shutil.make_archive(str(root / part), "zip", root / "src", part)
    return hashlib.md5(Path(archive).read_bytes()).hexdigest()


def test_prepare_checks_md5_and_extracts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    md5s = {part: _zip(tmp_path, part) for part in ("Train", "Test")}
    monkeypatch.setattr(
        nmnist, "_ARCHIVES", {"Train": ("unused", "0" * 32), "Test": ("unused", md5s["Test"])}
    )
    with pytest.raises(SbbError, match="wrong md5"):
        NMNIST(tmp_path).prepare()
    monkeypatch.setattr(nmnist, "_ARCHIVES", {p: ("unused", m) for p, m in md5s.items()})
    dataset = NMNIST(tmp_path)
    dataset.prepare()
    assert len(dataset) == 2 and dataset[0].label == 4 and not list(tmp_path.glob("tmp*"))
