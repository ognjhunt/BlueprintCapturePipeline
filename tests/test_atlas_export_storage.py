"""Preserve sealed export bytes while sharing duplicate binary storage."""

from pathlib import Path

import pytest

from blueprint_pipeline import atlas_result_bundle as bundle


def test_replacing_a_member_preserves_another_hard_linked_snapshot(tmp_path: Path) -> None:
    source, current, historical = [tmp_path / name for name in ("source", "current", "historical")]
    current.write_bytes(b"observed original")
    bundle.os.link(current, historical)
    source.write_bytes(b"updated export")

    bundle._replace_bundle_member(source, current)

    assert current.read_bytes() == b"updated export"
    assert historical.read_bytes() == b"observed original"
    assert current.stat().st_ino != historical.stat().st_ino


def test_failed_copy_keeps_the_existing_member_and_removes_temporary_files(tmp_path, monkeypatch) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"new export")
    destination.write_bytes(b"existing export")

    def fail(_source, temporary):
        Path(temporary).write_bytes(b"partial")
        raise OSError("simulated interrupted copy")

    monkeypatch.setattr(bundle.shutil, "copyfile", fail)
    with pytest.raises(OSError, match="interrupted"):
        bundle._replace_bundle_member(source, destination)

    assert destination.read_bytes() == b"existing export"
    assert not list(tmp_path.glob(".atlas-member-*"))


def test_unsupported_hard_links_fall_back_to_a_complete_copy(tmp_path, monkeypatch) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"identical retained bytes")

    def unsupported(*_args):
        raise OSError("hard links unavailable")

    monkeypatch.setattr(bundle.os, "link", unsupported)
    bundle._replace_bundle_member(source, destination, duplicate=source)

    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_ino != source.stat().st_ino


def test_link_reuse_preserves_existing_permissions_without_changing_the_donor(tmp_path) -> None:
    source, donor, destination = [tmp_path / name for name in ("source", "donor", "destination")]
    source.write_bytes(b"same bytes")
    donor.write_bytes(source.read_bytes())
    donor.chmod(0o644)
    destination.write_bytes(b"previous bytes")
    destination.chmod(0o600)

    bundle._replace_bundle_member(source, destination, duplicate=donor)

    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mode & 0o777 == 0o600
    assert donor.stat().st_mode & 0o777 == 0o644
    assert destination.stat().st_ino != donor.stat().st_ino
