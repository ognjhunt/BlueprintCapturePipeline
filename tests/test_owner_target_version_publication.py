"""ADP-009D/day28: immutable metadata uses retained proofs before mutations."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_owner_target_publication.py
#   src/blueprint_pipeline/control_plane_lane_owner_target_io.py

import os
import stat
from types import SimpleNamespace

import pytest

from tests.test_owner_target_version_descriptors import files


@pytest.fixture
def root_metadata(monkeypatch):
    # Only hermetic Mac source tests. The real foreign-UID Linux gate is separate.
    original_stat, original_fstat = os.stat, os.fstat
    def root(info):
        fields = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
        fields.update(st_uid=0, st_gid=0)
        return SimpleNamespace(**fields)
    monkeypatch.setattr(os, "stat", lambda *a, **kw: root(original_stat(*a, **kw)))
    monkeypatch.setattr(os, "fstat", lambda *a, **kw: root(original_fstat(*a, **kw)))


def published(tmp_path, payload=b'{"kept":true}\n', **changes):
    from blueprint_pipeline.control_plane_lane_owner_target_publication import _publish_owned_metadata
    tmp_path.chmod(0o700)
    owner = files()
    parent = owner.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        return _publish_owned_metadata(owner, parent, "a"*32 + ".json", payload,
                                       mode=0o600, artifact_kind="attestation", **changes)
    finally:
        owner.finish()


def test_ordinary_immutable_guarded_publication_and_readback(tmp_path, root_metadata, monkeypatch):
    from blueprint_pipeline import control_plane_lane_owner_consents as old
    for name in ("_publish", "_clean_temp", "_publish_reports"):
        monkeypatch.setattr(old, name, lambda *a, **kw: pytest.fail("unsafe old publisher"))
    monkeypatch.setattr(os, "replace", lambda *a, **kw: pytest.fail("replace fallback"))
    monkeypatch.setattr(os, "fdopen", lambda *a, **kw: pytest.fail("fdopen fallback"))
    result = published(tmp_path)
    final = tmp_path / ("a"*32 + ".json")
    assert final.read_bytes() == b'{"kept":true}\n'
    assert stat.S_IMODE(final.stat().st_mode) == 0o600 and final.stat().st_nlink == 1
    assert result["publication_checked"] is True
    assert not list(tmp_path.glob("*.tmp"))


def test_existing_destination_never_overwritten(tmp_path, root_metadata):
    from blueprint_pipeline.control_plane_lane_owner_target_versions import OwnerTargetVersionError
    final = tmp_path / ("a"*32 + ".json")
    final.write_bytes(b"preserve")
    with pytest.raises(OwnerTargetVersionError, match="owner_target_publication_destination_exists"):
        published(tmp_path)
    assert final.read_bytes() == b"preserve"


@pytest.mark.parametrize("operation", ["fchmod", "write", "fsync"])
def test_observed_temp_reuse_before_mutation_preserves_foreign(tmp_path, root_metadata, monkeypatch, operation):
    from blueprint_pipeline import control_plane_lane_owner_target_publication as p
    foreign = tmp_path / "foreign"
    foreign.write_bytes(b"FOREIGN")
    fd = os.open(foreign, os.O_RDWR)
    original = p._publication_guard
    swapped = False
    reused = None
    def guard(owner, state, *, stage, cleanup=False):
        nonlocal swapped, reused
        if stage == operation and not swapped:
            reused = state.fd
            os.dup2(fd, reused)
            swapped = True
        return original(owner, state, stage=stage, cleanup=cleanup)
    monkeypatch.setattr(p, "_publication_guard", guard)
    try:
        with pytest.raises(ValueError):
            published(tmp_path)
        assert swapped and foreign.read_bytes() == b"FOREIGN"
        assert os.fstat(reused).st_ino == foreign.stat().st_ino
        assert not (tmp_path / ("a"*32 + ".json")).exists()
    finally:
        if reused is not None:
            os.close(reused)
        os.close(fd)


def test_parent_reuse_before_link_preserves_foreign_directory(tmp_path, root_metadata, monkeypatch):
    from blueprint_pipeline import control_plane_lane_owner_target_publication as p
    foreign = tmp_path / "foreign"
    foreign.mkdir(mode=0o700)
    fd = os.open(foreign, os.O_RDONLY | os.O_DIRECTORY)
    original = p._publication_guard
    replaced = None
    def guard(owner, state, *, stage, cleanup=False):
        nonlocal replaced
        if stage == "link" and replaced is None:
            replaced = state.parent
            os.dup2(fd, replaced)
        return original(owner, state, stage=stage, cleanup=cleanup)
    monkeypatch.setattr(p, "_publication_guard", guard)
    try:
        with pytest.raises(ValueError):
            published(tmp_path)
        assert not list(foreign.iterdir())
        assert os.fstat(replaced).st_ino == foreign.stat().st_ino
    finally:
        if replaced is not None:
            os.close(replaced)
        os.close(fd)


def test_partial_published_record_is_not_rolled_back_or_replaced(tmp_path, root_metadata, monkeypatch):
    real_fsync = os.fsync
    def failed(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("private")
        return real_fsync(fd)
    monkeypatch.setattr(os, "fsync", failed)
    with pytest.raises(ValueError, match="owner_target_publication_failed"):
        published(tmp_path)
    assert (tmp_path / ("a"*32 + ".json")).read_bytes() == b'{"kept":true}\n'


def test_oversized_metadata_refuses_before_new_object(tmp_path, root_metadata):
    with pytest.raises(ValueError, match="owner_target_resource_exhausted"):
        published(tmp_path, b"x"*32769)
    assert list(tmp_path.iterdir()) == []
