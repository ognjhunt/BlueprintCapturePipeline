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
