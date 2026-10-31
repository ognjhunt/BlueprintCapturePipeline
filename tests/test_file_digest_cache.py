"""A retained file's digest is reused only while its stat fingerprint is unchanged (git's index)."""
import hashlib
import json
import os
import time
from types import SimpleNamespace

import pytest

from blueprint_pipeline import file_digest_cache as cache
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


@pytest.fixture
def hashed(monkeypatch):
    """Count the files actually hashed; files written by the test count as settled."""
    seen = []

    def sha256():
        seen.append(1)
        return hashlib.sha256()
    monkeypatch.setattr(cache, "hashlib", SimpleNamespace(sha256=sha256))
    monkeypatch.setattr(cache, "RACY_NANOSECONDS", -60 * 10**9)
    return seen


def _expected(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _digest(sidecar, path):
    with cache.file_digest_cache(sidecar):
        return cache.file_sha256(path)


def test_an_unchanged_file_is_hashed_once_across_ticks(tmp_path, hashed):
    path, sidecar = tmp_path / "splat.ply", tmp_path / cache.SIDECAR
    path.write_bytes(b"gaussians" * 1000)
    with cache.file_digest_cache(sidecar):
        assert cache.file_sha256(path) == cache.file_sha256(str(path)) == _expected(path.read_bytes())
    assert len(hashed) == 1
    assert _digest(sidecar, path) == _expected(path.read_bytes())  # the next tick reads the sidecar
    assert len(hashed) == 1
    entry = json.loads(sidecar.read_text())["entries"][str(path)]
    info = path.stat()
    assert entry == [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
                     _expected(path.read_bytes())]


@pytest.mark.parametrize("change", ["content", "size", "mtime"])
def test_a_changed_file_is_hashed_again(tmp_path, hashed, change):
    path, sidecar = tmp_path / "mesh.glb", tmp_path / cache.SIDECAR
    path.write_bytes(b"a" * 64)
    original = path.stat()
    assert _digest(sidecar, path) == _expected(b"a" * 64)
    if change == "content":
        time.sleep(.05)  # past the kernel's timestamp tick, as the racy window guarantees in use
        path.write_bytes(b"b" * 64)  # same size
        # Even with its mtime put back, the kernel's ctime gives the change away.
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    elif change == "size":
        path.write_bytes(b"a" * 65)
    else:
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns + 10**9))
    assert _digest(sidecar, path) == _expected(path.read_bytes())
    assert len(hashed) == 2


@pytest.mark.parametrize("field", range(5), ids=["dev", "ino", "size", "mtime_ns", "ctime_ns"])
def test_every_fingerprint_field_must_match_or_the_file_is_hashed_again(tmp_path, hashed, field):
    path, sidecar = tmp_path / "frame.png", tmp_path / cache.SIDECAR
    path.write_bytes(b"pixels")
    _digest(sidecar, path)
    value = json.loads(sidecar.read_text())
    row = value["entries"][str(path)]
    row[field] += 1
    row[5] = "sha256:" + "0" * 64  # a stale digest a mismatched fingerprint must never return
    sidecar.write_text(json.dumps(value))
    assert _digest(sidecar, path) == _expected(b"pixels")
    assert len(hashed) == 2


def test_a_replaced_file_with_the_same_bytes_and_mtime_is_hashed_again(tmp_path, hashed):
    path, sidecar = tmp_path / "result.json", tmp_path / cache.SIDECAR
    path.write_bytes(b"{}")
    original = path.stat()
    _digest(sidecar, path)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"{}")
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    os.replace(replacement, path)
    assert path.stat().st_ino != original.st_ino
    assert _digest(sidecar, path) == _expected(b"{}")
    assert len(hashed) == 2


def test_a_file_changed_within_the_racy_window_is_never_cached(tmp_path, hashed, monkeypatch):
    monkeypatch.setattr(cache, "RACY_NANOSECONDS", 2 * 10**9)
    path, sidecar = tmp_path / "fresh.ply", tmp_path / cache.SIDECAR
    path.write_bytes(b"just written")
    _digest(sidecar, path)
    _digest(sidecar, path)
    assert len(hashed) == 2 and not sidecar.exists()


def test_nothing_is_cached_outside_a_job_step(tmp_path, hashed):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"frames")
    assert cache.file_sha256(path) == cache.file_sha256(path) == _expected(b"frames")
    assert len(hashed) == 2 and not list(tmp_path.glob("*.json"))


def test_the_shared_helper_returns_the_plain_digest_in_and_out_of_the_cache(tmp_path, hashed):
    path = tmp_path / "bundle.zip"
    path.write_bytes(b"zip bytes")
    with cache.file_digest_cache(tmp_path / cache.SIDECAR):
        inside = [_sha256_file(path), _sha256_file(path)]
    assert inside == [_sha256_file(path), _expected(b"zip bytes")]


def test_an_unreadable_sidecar_is_ignored_and_gone_files_are_dropped(tmp_path, hashed):
    kept, gone, sidecar = tmp_path / "kept", tmp_path / "gone", tmp_path / cache.SIDECAR
    sidecar.write_text("not json")
    kept.write_bytes(b"1")
    gone.write_bytes(b"2")
    with cache.file_digest_cache(sidecar):
        cache.file_sha256(kept)
        cache.file_sha256(gone)
    gone.unlink()
    with cache.file_digest_cache(sidecar):
        pass  # a step that hashed nothing keeps what is still valid
    assert set(json.loads(sidecar.read_text())["entries"]) == {str(kept)}
    with pytest.raises(FileNotFoundError):
        _digest(sidecar, gone)


def test_a_passed_check_runs_once_per_job_for_the_same_binding(tmp_path):
    sidecar, runs = tmp_path / cache.SIDECAR, []
    binding = {"check": "square_padding.v1", "source": "sha256:" + "a" * 64, "padded": "sha256:" + "b" * 64}
    for _tick in range(2):
        with cache.file_digest_cache(sidecar):
            cache.verify_once(binding, lambda: runs.append("first"))
            cache.verify_once(binding, lambda: runs.append("first"))
    assert runs == ["first"]
    with cache.file_digest_cache(sidecar):
        cache.verify_once({**binding, "padded": "sha256:" + "c" * 64}, lambda: runs.append("changed input"))
    cache.verify_once(binding, lambda: runs.append("outside a job step"))
    assert runs == ["first", "changed input", "outside a job step"]


def test_a_failed_check_is_never_remembered(tmp_path):
    sidecar, runs = tmp_path / cache.SIDECAR, []

    def refuse():
        runs.append(1)
        raise ValueError("atlas_context_padding_changed")
    for _tick in range(2):
        with cache.file_digest_cache(sidecar), pytest.raises(ValueError, match="padding_changed"):
            cache.verify_once({"check": "square_padding.v1"}, refuse)
    assert runs == [1, 1] and not sidecar.exists()
