from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from blueprint_pipeline.immutable_directory_publication import (
    ImmutableDirectoryPublicationError,
    publish_staged_immutable_directory,
)


def _staging(root: Path, name: str) -> Path:
    staging = root / f"staging-{name}"
    staging.mkdir()
    (staging / "payload.bin").write_bytes(name.encode("utf-8"))
    (staging / "manifest.json").write_text(name, encoding="utf-8")
    return staging


def test_concurrent_publishers_cannot_replace_immutable_directory(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"

    def publish(name: str) -> str:
        try:
            publish_staged_immutable_directory(
                staging=_staging(tmp_path, name),
                destination=destination,
                manifest_name="manifest.json",
                output_exists_code="immutable_output_exists",
            )
        except ImmutableDirectoryPublicationError as exc:
            return str(exc)
        return "published"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(publish, ("first", "second")))

    assert sorted(results) == ["immutable_output_exists", "published"]
    manifest = (destination / "manifest.json").read_text(encoding="utf-8")
    assert (destination / "payload.bin").read_text(encoding="utf-8") == manifest
    assert not destination.stat().st_mode & 0o222


def test_existing_destination_is_untouched(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    sentinel = destination / "sentinel"
    sentinel.write_text("owned", encoding="utf-8")

    with pytest.raises(
        ImmutableDirectoryPublicationError,
        match="immutable_output_exists",
    ):
        publish_staged_immutable_directory(
            staging=_staging(tmp_path, "candidate"),
            destination=destination,
            manifest_name="manifest.json",
            output_exists_code="immutable_output_exists",
        )

    assert sentinel.read_text(encoding="utf-8") == "owned"
    assert set(path.name for path in destination.iterdir()) == {"sentinel"}


def test_publication_preserves_immutable_file_inode_without_staging_tree(
    tmp_path: Path,
) -> None:
    staging = _staging(tmp_path, "content-addressed")
    payload_inode = (staging / "payload.bin").stat().st_ino
    manifest_inode = (staging / "manifest.json").stat().st_ino
    destination = tmp_path / "published"

    publish_staged_immutable_directory(
        staging=staging,
        destination=destination,
        manifest_name="manifest.json",
        output_exists_code="immutable_output_exists",
    )

    assert not staging.exists()
    assert (destination / "payload.bin").stat().st_ino == payload_inode
    assert (destination / "manifest.json").stat().st_ino == manifest_inode


def test_dangling_destination_symlink_is_not_followed(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    escaped = tmp_path / "escaped"
    destination.symlink_to(escaped, target_is_directory=True)

    with pytest.raises(
        ImmutableDirectoryPublicationError,
        match="immutable_output_exists",
    ):
        publish_staged_immutable_directory(
            staging=_staging(tmp_path, "candidate"),
            destination=destination,
            manifest_name="manifest.json",
            output_exists_code="immutable_output_exists",
        )

    assert destination.is_symlink()
    assert not escaped.exists()


def test_a_stopped_publication_leaves_no_partial_destination(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import immutable_directory_publication as publication

    def stopped(_source, _destination) -> None:
        raise SystemExit(143)  # systemctl stop, through the provision step's SIGTERM handler

    monkeypatch.setattr(publication.os, "rename", stopped)
    destination = tmp_path / "published"
    with pytest.raises(SystemExit):
        publish_staged_immutable_directory(
            staging=_staging(tmp_path, "stopped"),
            destination=destination,
            manifest_name="manifest.json",
            output_exists_code="immutable_output_exists",
        )
    assert not destination.exists()


def test_publishing_moves_no_files_link_count_or_change_time(tmp_path: Path) -> None:
    """Both are in the identity the digest cache keys on; a hard link and unlink moved them."""
    staging = _staging(tmp_path, "held")
    (staging / "nested").mkdir()
    (staging / "nested" / "payload.bin").write_bytes(b"held")
    (staging / "nested").chmod(0o555)
    before = {name: (staging / name).stat() for name in ("payload.bin", "nested/payload.bin")}

    publish_staged_immutable_directory(staging=staging, destination=tmp_path / "published",
                                       manifest_name="manifest.json", output_exists_code="immutable_output_exists")

    for name, stat_before in before.items():
        after = (tmp_path / "published" / name).stat()
        assert (after.st_ino, after.st_nlink, after.st_ctime_ns) == (
            stat_before.st_ino, stat_before.st_nlink, stat_before.st_ctime_ns)
    assert not staging.exists()
    assert not (tmp_path / "published").stat().st_mode & 0o222


def test_a_destination_something_filled_after_the_reservation_is_not_overwritten(tmp_path: Path, monkeypatch) -> None:
    """The rename replaces an empty directory only."""
    from blueprint_pipeline import immutable_directory_publication as publication

    destination = tmp_path / "published"
    rename = publication.os.rename

    def filled_first(source, target) -> None:
        (Path(target) / "theirs").write_text("theirs", encoding="utf-8")
        rename(source, target)

    monkeypatch.setattr(publication.os, "rename", filled_first)
    staging = _staging(tmp_path, "late")

    with pytest.raises(OSError):
        publish_staged_immutable_directory(staging=staging, destination=destination, manifest_name="manifest.json",
                                           output_exists_code="immutable_output_exists")

    assert (staging / "payload.bin").read_text(encoding="utf-8") == "late"
