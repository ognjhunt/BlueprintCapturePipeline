import zipfile

import pytest
import blueprint_pipeline.verified_archive_cache_reclamation as subject

from blueprint_pipeline.verified_archive_cache_reclamation import (
    apply_archive_cache_reclamation,
    plan_archive_cache_reclamation,
)


def fixture(tmp_path):
    root = tmp_path / "extracted"
    root.mkdir()
    archive = tmp_path / "output.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name, data in [
            ("a.ply", b"original-a"),
            ("b.ply", b"original-b"),
            ("receipt.json", b"{}"),
            ("training.log", b"complete"),
        ]:
            z.writestr(name, data)
            (root / name).write_bytes(data)
    (root / "unknown.ply").write_bytes(b"not archived")
    return root, archive


def test_only_verified_unprotected_cache_is_removed(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(subject, "_process_root", lambda: proc)
    root, archive = fixture(tmp_path)
    before = archive.read_bytes()
    plan = plan_archive_cache_reclamation(
        archive_path=archive,
        extraction_root=root,
        protected_paths=[root / "b.ply"],
        minimum_size_bytes=1,
    )
    assert [r["relative_path"] for r in plan["candidates"]] == ["a.ply"]
    assert (root / "a.ply").exists()
    result = apply_archive_cache_reclamation(plan, ack="reclaim-byte-verified-extraction-cache")
    assert result["removed_count"] == 1 and not (root / "a.ply").exists()
    assert archive.read_bytes() == before
    assert all(
        (root / n).exists() for n in ["b.ply", "receipt.json", "training.log", "unknown.ply"]
    )


def test_changed_candidate_refuses_whole_apply_before_any_deletion(tmp_path):
    root, archive = fixture(tmp_path)
    plan = plan_archive_cache_reclamation(
        archive_path=archive, extraction_root=root, minimum_size_bytes=1
    )
    (root / "b.ply").write_bytes(b"new-state!")
    with pytest.raises(ValueError, match="candidate_changed"):
        apply_archive_cache_reclamation(plan, ack="reclaim-byte-verified-extraction-cache")
    assert (root / "a.ply").exists() and archive.exists()


def test_changed_archive_cannot_authorize_cache_removal(tmp_path):
    root, archive = fixture(tmp_path)
    plan = plan_archive_cache_reclamation(
        archive_path=archive, extraction_root=root, minimum_size_bytes=1
    )
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive_changed"):
        apply_archive_cache_reclamation(plan, ack="reclaim-byte-verified-extraction-cache")
    assert (root / "a.ply").exists()


def test_live_reader_refuses_whole_apply(tmp_path, monkeypatch):
    root, archive = fixture(tmp_path)
    proc = tmp_path / "proc"
    descriptors = proc / "123" / "fd"
    descriptors.mkdir(parents=True)
    (descriptors / "4").symlink_to(root / "b.ply")
    monkeypatch.setattr(subject, "_process_root", lambda: proc)
    plan = plan_archive_cache_reclamation(archive_path=archive, extraction_root=root, minimum_size_bytes=1)
    with pytest.raises(ValueError, match="active_reader"):
        apply_archive_cache_reclamation(plan, ack="reclaim-byte-verified-extraction-cache")
    assert (root / "a.ply").exists() and (root / "b.ply").exists()
