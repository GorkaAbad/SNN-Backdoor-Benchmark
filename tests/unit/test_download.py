import hashlib
import io
import tarfile
import threading
import urllib.request
import zipfile
from pathlib import Path

import pytest

from sbb.data.datasets.common import fetch, unpacked
from sbb.utils.errors import SbbError

BODY = b"archive bytes"
MD5 = hashlib.md5(BODY).hexdigest()


class _Response(io.BytesIO):
    def __init__(self, body: bytes, status: int = 200) -> None:
        super().__init__(body)
        self.status = status


def _serve(monkeypatch: pytest.MonkeyPatch, answers: dict[str, _Response | Exception]) -> list[str]:
    asked: list[str] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        asked.append(request.full_url)
        assert request.get_header("User-agent") == "sbb"
        answer = answers[request.full_url]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return asked


def test_fetch_falls_back_until_a_url_delivers_the_right_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = _serve(
        monkeypatch,
        {
            "https://down": OSError("HTTP 403"),
            "https://challenge": _Response(b"", status=202),
            "https://corrupt": _Response(b"something else"),
            "https://good": _Response(BODY),
        },
    )
    archive = tmp_path / "sub" / "a.zip"
    fetch(
        archive, MD5, ["https://down", "https://challenge", "https://corrupt", "https://good"], ""
    )
    assert archive.read_bytes() == BODY and len(asked) == 4
    assert [p.name for p in archive.parent.iterdir()] == ["a.zip"]


def test_fetch_gives_manual_instructions_when_every_url_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, {"https://corrupt": _Response(b"something else")})
    with pytest.raises(SbbError, match="put it in /here"):
        fetch(tmp_path / "a.zip", MD5, ["https://corrupt"], "put it in /here")
    assert list(tmp_path.iterdir()) == []


def test_fetch_checks_but_never_replaces_an_existing_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = _serve(monkeypatch, {})
    archive = tmp_path / "a.zip"
    archive.write_bytes(BODY)
    fetch(archive, MD5, ["https://unused"], "")
    archive.write_bytes(b"tampered")
    with pytest.raises(SbbError, match="wrong md5"):
        fetch(archive, MD5, ["https://unused"], "")
    assert asked == [] and archive.read_bytes() == b"tampered"


def test_unpacked_extracts_zip_and_tar_and_cleans_up(tmp_path: Path) -> None:
    with zipfile.ZipFile(tmp_path / "a.zip", "w") as z:
        z.writestr("top/file.txt", "zip")
    with tarfile.open(tmp_path / "b.tar.gz", "w:gz") as t:
        info = tarfile.TarInfo("top/file.txt")
        info.size = 3
        t.addfile(info, io.BytesIO(b"tar"))
    for name, content in [("a.zip", "zip"), ("b.tar.gz", "tar")]:
        with unpacked(tmp_path / name, tmp_path) as tmp:
            assert (tmp / "top" / "file.txt").read_text() == content
        assert not tmp.exists()


def test_unpacked_refuses_tar_members_outside_the_directory(tmp_path: Path) -> None:
    with tarfile.open(tmp_path / "evil.tar.gz", "w:gz") as t:
        info = tarfile.TarInfo("../escaped.txt")
        info.size = 4
        t.addfile(info, io.BytesIO(b"evil"))
    with pytest.raises(tarfile.FilterError), unpacked(tmp_path / "evil.tar.gz", tmp_path):
        pass
    assert not (tmp_path / "escaped.txt").exists()


class _SlowResponse(_Response):
    """Holds its first read until every concurrent download has started."""

    def __init__(self, body: bytes, barrier: threading.Barrier) -> None:
        super().__init__(body)
        self.barrier = barrier
        self.waited = False

    def read(self, size: int | None = -1) -> bytes:
        if not self.waited:
            self.waited = True
            self.barrier.wait(timeout=10)
        return super().read(size)


def test_concurrent_fetches_of_one_archive_both_succeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    barrier = threading.Barrier(2)
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout: _SlowResponse(BODY, barrier)
    )
    archive = tmp_path / "a.zip"
    outcome: dict[str, str] = {}

    def run(name: str) -> None:
        try:
            fetch(archive, MD5, ["https://good"], "")
            outcome[name] = "ok"
        except Exception as error:  # noqa: BLE001  reported by the assertion below
            outcome[name] = repr(error)

    threads = [threading.Thread(target=run, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcome == {"a": "ok", "b": "ok"}
    assert archive.read_bytes() == BODY and [p.name for p in tmp_path.iterdir()] == ["a.zip"]
