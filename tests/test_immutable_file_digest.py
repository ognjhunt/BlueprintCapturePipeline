"""Publication reuse must preserve actual bytes and detect changed files."""

import hashlib
import os

import pytest

from blueprint_pipeline import immutable_file_digest as digests


def _expected(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


@pytest.fixture
def reads(monkeypatch):
    observed = []
    original = digests._read_digest

    def counted(stream):
        observed.append(stream.name)
        return original(stream)

    monkeypatch.setattr(digests, "_read_digest", counted)
    return observed


def test_reuses_read_only_inode_across_aliases_only_within_one_publication(tmp_path, reads):
    first, alias = tmp_path / "first", tmp_path / "alias"
    first.write_bytes(b"sealed runtime bytes")
    first.chmod(0o444)
    os.link(first, alias)
    with digests.immutable_file_digest_cache():
        assert digests.sha256_file(first) == _expected(first.read_bytes())
        with digests.immutable_file_digest_cache():
            assert digests.sha256_file(alias) == _expected(first.read_bytes())
        assert len(reads) == 1
    assert digests.sha256_file(alias) == _expected(first.read_bytes())
    with digests.immutable_file_digest_cache():
        assert digests.sha256_file(first) == _expected(first.read_bytes())
    assert len(reads) == 3


def test_mutable_files_and_symlink_leaves_are_always_read_in_full(tmp_path, reads):
    mutable, sealed, link = [tmp_path / name for name in ("mutable", "sealed", "link")]
    mutable.write_bytes(b"mutable")
    sealed.write_bytes(b"sealed")
    sealed.chmod(0o444)
    link.symlink_to(sealed)
    with digests.immutable_file_digest_cache():
        for path in (mutable, mutable, link, link):
            assert digests.sha256_file(path) == _expected(path.read_bytes())
    assert len(reads) == 4


@pytest.mark.parametrize("change", ("rewrite", "replace", "chmod", "link"))
def test_content_replacement_permission_and_link_changes_invalidate(tmp_path, reads, change):
    path = tmp_path / "package"
    path.write_bytes(b"original")
    path.chmod(0o444)
    original = path.stat()
    with digests.immutable_file_digest_cache():
        assert digests.sha256_file(path) == _expected(b"original")
        if change == "rewrite":
            path.chmod(0o644)
            path.write_bytes(b"modified")
            path.chmod(0o444)
            os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        elif change == "replace":
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"modified")
            replacement.chmod(0o444)
            os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
            os.replace(replacement, path)
        elif change == "chmod":
            path.chmod(0o644)
        else:
            os.link(path, tmp_path / "new-alias")
        assert digests.sha256_file(path) == _expected(path.read_bytes())
        assert len(reads) == 2


def test_change_during_full_hash_is_rejected_and_never_cached(tmp_path, monkeypatch):
    path = tmp_path / "package"
    path.write_bytes(b"original")
    path.chmod(0o444)
    original = digests._read_digest

    def changed(stream):
        value = original(stream)
        path.chmod(0o644)
        path.write_bytes(b"modified")
        path.chmod(0o444)
        return value

    with digests.immutable_file_digest_cache():
        monkeypatch.setattr(digests, "_read_digest", changed)
        with pytest.raises(OSError, match="changed_during_digest"):
            digests.sha256_file(path)
        monkeypatch.setattr(digests, "_read_digest", original)
        assert digests.sha256_file(path) == _expected(b"modified")


def test_bounded_cache_evicts_without_rejecting_larger_packages(tmp_path, reads, monkeypatch):
    monkeypatch.setattr(digests, "_MAX_ENTRIES", 1)
    first, second = tmp_path / "first", tmp_path / "second"
    for path in (first, second):
        path.write_bytes(path.name.encode())
        path.chmod(0o444)
    with digests.immutable_file_digest_cache():
        for path in (first, second, first):
            assert digests.sha256_file(path) == _expected(path.read_bytes())
    assert len(reads) == 3
