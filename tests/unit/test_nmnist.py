from pathlib import Path

import numpy as np

from sbb.data.datasets.nmnist import NMNIST, read_atis_bin


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
