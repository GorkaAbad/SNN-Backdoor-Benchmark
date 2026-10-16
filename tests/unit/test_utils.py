from pathlib import Path

import pytest

from sbb.utils.determinism import rng_for
from sbb.utils.errors import SbbError
from sbb.utils.hashing import canonical_json, sha256_hex
from sbb.utils.registry import Registry
from sbb.utils.storage import ArtifactCache, LocalStore


def test_canonical_json_is_order_independent() -> None:
    assert canonical_json({"b": 1, "a": [0.1, "é"]}) == '{"a":[0.1,"é"],"b":1}'
    assert sha256_hex({"a": 1, "b": 2}) == sha256_hex({"b": 2, "a": 1})


def test_rng_for_depends_only_on_seed_and_index() -> None:
    assert rng_for(0, 7).random() == rng_for(0, 7).random()
    assert rng_for(0, 7).random() != rng_for(0, 8).random()


def test_registry_rejects_duplicates_and_unknown_names() -> None:
    reg: Registry[object] = Registry("thing")
    reg.register("a")(int)
    assert reg.get("a") is int and reg.names() == ["a"]
    with pytest.raises(SbbError):
        reg.register("a")(str)
    with pytest.raises(SbbError, match="unknown thing"):
        reg.get("b")


def test_cache_builds_once(tmp_path: Path) -> None:
    calls = []

    def build(d: Path) -> None:
        calls.append(d)
        (d / "x.txt").write_text("hi")

    cache = ArtifactCache(LocalStore(tmp_path / "local"))
    first = cache.get_or_build("k", build)
    second = cache.get_or_build("k", build)
    assert first == second and (first / "x.txt").read_text() == "hi" and len(calls) == 1


def test_cache_copies_from_remote(tmp_path: Path) -> None:
    remote = LocalStore(tmp_path / "remote")
    (tmp_path / "remote" / "k").mkdir(parents=True)
    (tmp_path / "remote" / "k" / "x.txt").write_text("remote")
    cache = ArtifactCache(LocalStore(tmp_path / "local"), remote)
    path = cache.get_or_build("k", lambda d: pytest.fail("should not build"))
    assert (path / "x.txt").read_text() == "remote"


def test_failed_build_leaves_nothing(tmp_path: Path) -> None:
    def build(d: Path) -> None:
        (d / "partial").write_text("")
        raise RuntimeError("boom")

    cache = ArtifactCache(LocalStore(tmp_path))
    with pytest.raises(RuntimeError):
        cache.get_or_build("k", build)
    assert list(tmp_path.iterdir()) == []


def test_put_keeps_the_first_writer(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "store")
    for content in ("first", "second"):
        staged = tmp_path / f"staged-{content}"
        staged.mkdir()
        (staged / "file").write_text(content)
        store.put("key", staged)  # a second writer of the same key must not fail
        assert not staged.exists()
    path = store.get("key")
    assert path is not None and (path / "file").read_text() == "first"
