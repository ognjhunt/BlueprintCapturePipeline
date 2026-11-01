"""A full-file digest is reused only while the file's inode identity is exact."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline import sealed_file_digest as sealed
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count full reads, and treat every file as past the stability window."""

    calls: list[int] = []
    original = sealed._hash_stream

    def counted(stream):
        calls.append(1)
        return original(stream)

    monkeypatch.setattr(sealed, "_hash_stream", counted)
    monkeypatch.setattr(sealed, "time_ns", lambda: os.stat(".").st_ctime_ns + 10 * sealed.STABLE_AGE_NS + 10**18)
    return calls


def _file(tmp_path: Path, payload: bytes = b"bundle-bytes" * 1000) -> Path:
    path = tmp_path / "bundle.zip"
    path.write_bytes(payload)
    path.chmod(0o644)
    return path


def test_a_seal_is_reused_only_for_the_exact_same_inode(tmp_path: Path, reads: list[int]) -> None:
    path = _file(tmp_path)
    expected = _digest(path.read_bytes())

    assert sealed.sealed_sha256(path) == expected
    assert sealed.seal_path(path).is_file()
    assert sealed.sealed_sha256(path) == expected
    assert sealed.sealed_sha256(path) == expected
    assert len(reads) == 1


def test_same_size_rewrite_with_restored_mtime_is_rehashed(tmp_path: Path, reads: list[int]) -> None:
    path = _file(tmp_path)
    sealed.sealed_sha256(path)
    before = path.stat()
    replacement = b"BUNDLE-BYTES" * 1000
    with path.open("r+b") as stream:
        stream.write(replacement)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))  # ctime still moves
    assert path.stat().st_mtime_ns == before.st_mtime_ns

    assert sealed.sealed_sha256(path) == _digest(replacement)
    assert len(reads) == 2


@pytest.mark.parametrize("change", ["chmod", "replace", "touch"])
def test_any_identity_change_invalidates_the_seal(
    tmp_path: Path, reads: list[int], change: str
) -> None:
    path = _file(tmp_path)
    sealed.sealed_sha256(path)
    if change == "chmod":
        path.chmod(0o600)
    elif change == "replace":
        other = tmp_path / "other.zip"
        other.write_bytes(path.read_bytes())  # same bytes, new inode
        os.replace(other, path)
    else:
        os.utime(path)

    assert sealed.sealed_sha256(path) == _digest(path.read_bytes())
    assert len(reads) == 2


def test_a_recently_changed_file_is_never_sealed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _file(tmp_path)
    monkeypatch.setattr(sealed, "time_ns", lambda: path.stat().st_ctime_ns + 1)

    assert sealed.sealed_sha256(path) == _digest(path.read_bytes())
    assert not sealed.seal_path(path).exists()


def test_a_seal_for_another_path_or_a_malformed_seal_is_ignored(tmp_path: Path, reads: list[int]) -> None:
    path = _file(tmp_path)
    sealed.sealed_sha256(path)
    seal = sealed.seal_path(path)
    value = json.loads(seal.read_text())
    value["sha256"] = "sha256:" + "0" * 64  # breaks the self-digest
    seal.write_text(json.dumps(value))
    assert sealed.sealed_sha256(path) == _digest(path.read_bytes())

    # A self-consistent seal written for another path never vouches for this one.
    value = json.loads(seal.read_text())
    value["path"] = str(tmp_path / "elsewhere.zip")
    value["seal_digest"] = canonical_digest(value, digest_field="seal_digest")
    seal.write_text(json.dumps(value))
    assert sealed.sealed_sha256(path) == _digest(path.read_bytes())
    assert len(reads) == 3
