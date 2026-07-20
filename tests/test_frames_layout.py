"""Frames sub-layout contract tests (SCALE2-03, frames_index.v2).

Fixtures build one capture in the legacy per-object layout and the same
capture packed into tar archives, then assert both resolve identical
downstream frame content — the rollout invariant for the coordinated
extract-frames change in BlueprintCapture.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from blueprint_pipeline.frames_layout import (
    FRAMES_INDEX_SCHEMA_V1,
    FRAMES_INDEX_SCHEMA_V2,
    iter_frame_payloads,
    load_frames_layout,
    read_frame_bytes,
)

FRAME_PAYLOADS = {
    "000001.jpg": b"\xff\xd8\xff\xe0frame-one",
    "000002.jpg": b"\xff\xd8\xff\xe0frame-two",
    "000003.jpg": b"\xff\xd8\xff\xe0frame-three",
}


def _index_entry(frame_id: str, packed: bool, archive: str | None = None) -> dict:
    entry = {"frame_id": frame_id, "t_video_sec": int(frame_id) / 5.0}
    if packed:
        entry["packaging"] = "tar"
        entry["archive"] = archive
        entry["archive_member"] = f"{frame_id}.jpg"
    return entry


def build_legacy_capture(frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True)
    for name, payload in FRAME_PAYLOADS.items():
        (frames_dir / name).write_bytes(payload)
    lines = [
        json.dumps(_index_entry(name.split(".")[0], packed=False))
        for name in sorted(FRAME_PAYLOADS)
    ]
    (frames_dir / "index.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_packed_capture(frames_dir: Path, frames_per_archive: int = 2) -> None:
    frames_dir.mkdir(parents=True)
    names = sorted(FRAME_PAYLOADS)
    archives: list[tuple[str, list[str]]] = []
    for start in range(0, len(names), frames_per_archive):
        archive_name = f"frames_{len(archives):03d}.tar"
        archives.append((archive_name, names[start : start + frames_per_archive]))

    member_to_archive: dict[str, str] = {}
    for archive_name, members in archives:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for member in members:
                payload = FRAME_PAYLOADS[member]
                info = tarfile.TarInfo(name=member)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
                member_to_archive[member] = archive_name
        (frames_dir / archive_name).write_bytes(buffer.getvalue())

    lines = [
        json.dumps(
            _index_entry(name.split(".")[0], packed=True, archive=member_to_archive[name])
        )
        for name in names
    ]
    (frames_dir / "index.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (frames_dir / "packing_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "frames_index.v2",
                "packaging": "tar",
                "frames_per_archive": frames_per_archive,
                "frame_count": len(names),
                "archives": [
                    {"archive": archive_name, "member_count": len(members)}
                    for archive_name, members in archives
                ],
            }
        ),
        encoding="utf-8",
    )


def test_legacy_per_object_layout_detected_and_readable(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    build_legacy_capture(frames_dir)

    layout = load_frames_layout(frames_dir)
    assert layout.schema_version == FRAMES_INDEX_SCHEMA_V1
    assert layout.packaging == "per_object"
    assert [record.frame_id for record in layout.records] == ["000001", "000002", "000003"]
    assert read_frame_bytes(layout, layout.records[1]) == FRAME_PAYLOADS["000002.jpg"]


def test_packed_layout_detected_and_readable(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    build_packed_capture(frames_dir)

    layout = load_frames_layout(frames_dir)
    assert layout.schema_version == FRAMES_INDEX_SCHEMA_V2
    assert layout.packaging == "tar"
    assert layout.records[0].archive == "frames_000.tar"
    assert layout.records[2].archive == "frames_001.tar"
    assert read_frame_bytes(layout, layout.records[2]) == FRAME_PAYLOADS["000003.jpg"]


def test_both_layouts_yield_identical_downstream_frame_content(tmp_path: Path) -> None:
    legacy_dir = tmp_path / "legacy" / "frames"
    packed_dir = tmp_path / "packed" / "frames"
    build_legacy_capture(legacy_dir)
    build_packed_capture(packed_dir)

    legacy = {record.frame_id: payload for record, payload in iter_frame_payloads(legacy_dir)}
    packed = {record.frame_id: payload for record, payload in iter_frame_payloads(packed_dir)}
    assert legacy == packed
    assert legacy == {
        name.split(".")[0]: payload for name, payload in FRAME_PAYLOADS.items()
    }


def test_unknown_packing_schema_fails_closed(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    build_packed_capture(frames_dir)
    manifest_path = frames_dir / "packing_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "frames_index.v99"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported frames packing schema_version"):
        load_frames_layout(frames_dir)


def test_packed_capture_with_missing_archive_linkage_fails_closed(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    build_packed_capture(frames_dir)
    index_path = frames_dir / "index.jsonl"
    lines = index_path.read_text(encoding="utf-8").splitlines()
    broken = json.loads(lines[0])
    del broken["archive"]
    del broken["packaging"]
    lines[0] = json.dumps(broken)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="without archive linkage"):
        load_frames_layout(frames_dir)


def test_manifestless_mixed_index_fails_closed(tmp_path: Path) -> None:
    # A packed capture whose manifest upload was lost AND whose index mixes
    # packed and unpacked rows is a partial upload/corruption: it must fail
    # closed, not silently drop the unarchived frames.
    frames_dir = tmp_path / "frames"
    build_packed_capture(frames_dir)
    (frames_dir / "packing_manifest.json").unlink()
    index_path = frames_dir / "index.jsonl"
    lines = index_path.read_text(encoding="utf-8").splitlines()
    broken = json.loads(lines[1])
    del broken["archive"]
    del broken["packaging"]
    lines[1] = json.dumps(broken)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="without archive linkage"):
        load_frames_layout(frames_dir)


def test_missing_member_in_archive_fails_closed(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    build_packed_capture(frames_dir)
    layout = load_frames_layout(frames_dir)
    ghost = layout.records[0]
    object.__setattr__(ghost, "member_name", "does-not-exist.jpg")
    with pytest.raises(FileNotFoundError):
        read_frame_bytes(layout, ghost)
