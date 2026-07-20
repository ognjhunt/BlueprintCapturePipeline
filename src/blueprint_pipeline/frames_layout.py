"""Frames sub-layout reader supporting both per-object (v1) and packed (v2) captures.

Contract revision SCALE2-03 (see docs/CAPTURE_BRIDGE_CONTRACT.md, "Frames
Sub-Layout"): BlueprintCapture's extract-frames function historically wrote
one GCS object per extracted JPEG (~900 objects for a 3-minute capture).
Behind ``BLUEPRINT_EXTRACT_FRAMES_PACKING_ENABLED`` it now packs frames into
a small number of USTAR archives (``frames_NNN.tar``, default 200
frames/archive) and declares the layout in ``frames/packing_manifest.json``
(``schema_version: "frames_index.v2"``). ``frames/index.jsonl`` stays
one-entry-per-frame in both layouts; packed entries additionally carry
``packaging``/``archive``/``archive_member``.

This module is the single place pipeline code should go through to resolve
frame payloads, so every consumer transparently supports both layouts during
rollout. The ``capture_bridge_handoff.v1`` message shape and the
completion-marker protocol are untouched by this revision.
"""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

FRAMES_INDEX_SCHEMA_V1 = "frames_index.v1"
FRAMES_INDEX_SCHEMA_V2 = "frames_index.v2"
PACKING_MANIFEST_NAME = "packing_manifest.json"
FRAMES_INDEX_NAME = "index.jsonl"


@dataclass(frozen=True)
class FrameRecord:
    """One frame from index.jsonl, normalized across layouts."""

    frame_id: str
    packaging: str  # "per_object" | "tar"
    member_name: str  # object/file name of the frame image
    archive: Optional[str]  # tar archive name when packaging == "tar"
    entry: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FramesLayout:
    schema_version: str  # frames_index.v1 | frames_index.v2
    packaging: str  # "per_object" | "tar"
    frames_dir: Path
    records: List[FrameRecord]


def _read_index_entries(index_path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def _record_from_entry(entry: Mapping[str, Any]) -> FrameRecord:
    frame_id = str(entry.get("frame_id") or "").strip()
    archive = entry.get("archive")
    packaging = str(entry.get("packaging") or "").strip().lower()
    member = str(entry.get("archive_member") or "").strip() or (
        f"{frame_id}.jpg" if frame_id else ""
    )
    if packaging == "tar" or archive:
        return FrameRecord(
            frame_id=frame_id,
            packaging="tar",
            member_name=member,
            archive=str(archive) if archive else None,
            entry=dict(entry),
        )
    return FrameRecord(
        frame_id=frame_id,
        packaging="per_object",
        member_name=member,
        archive=None,
        entry=dict(entry),
    )


def load_frames_layout(frames_dir: Path) -> FramesLayout:
    """Resolve the frames layout for a capture's local/mounted frames dir.

    Detection is explicit-first: a ``packing_manifest.json`` declaring
    ``frames_index.v2`` marks a packed capture; otherwise per-entry
    ``archive`` fields are honored (defensive), and a plain index is legacy
    per-object v1. Fail closed on a manifest with an unknown schema version
    rather than guessing.
    """

    index_path = frames_dir / FRAMES_INDEX_NAME
    if not index_path.is_file():
        raise FileNotFoundError(f"frames index missing: {index_path}")

    manifest_path = frames_dir / PACKING_MANIFEST_NAME
    manifest: Optional[Dict[str, Any]] = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared = str(manifest.get("schema_version") or "")
        if declared != FRAMES_INDEX_SCHEMA_V2:
            raise ValueError(
                f"unsupported frames packing schema_version: {declared!r} "
                f"(supported: {FRAMES_INDEX_SCHEMA_V2})"
            )
        declared_packaging = str(manifest.get("packaging") or "")
        if declared_packaging != "tar":
            raise ValueError(
                f"unsupported frames packaging: {declared_packaging!r} (supported: tar)"
            )

    records = [_record_from_entry(entry) for entry in _read_index_entries(index_path)]
    packed_records = [record for record in records if record.packaging == "tar"]

    if manifest is not None:
        missing_archive = [r.frame_id for r in records if r.packaging != "tar" or not r.archive]
        if missing_archive:
            raise ValueError(
                "packed capture has index entries without archive linkage: "
                + ", ".join(missing_archive[:5])
            )
        return FramesLayout(
            schema_version=FRAMES_INDEX_SCHEMA_V2,
            packaging="tar",
            frames_dir=frames_dir,
            records=records,
        )

    if packed_records:
        # No manifest but archive fields present: still readable as v2, with
        # the SAME fail-closed linkage check as the manifest path — a mixed
        # index (some rows packed, some not) means a partial upload or
        # corruption, and silently dropping the unarchived rows would corrupt
        # downstream frame counts.
        missing_archive = [r.frame_id for r in records if r.packaging != "tar" or not r.archive]
        if missing_archive:
            raise ValueError(
                "packed capture has index entries without archive linkage: "
                + ", ".join(missing_archive[:5])
            )
        return FramesLayout(
            schema_version=FRAMES_INDEX_SCHEMA_V2,
            packaging="tar",
            frames_dir=frames_dir,
            records=records,
        )

    return FramesLayout(
        schema_version=FRAMES_INDEX_SCHEMA_V1,
        packaging="per_object",
        frames_dir=frames_dir,
        records=records,
    )


def read_frame_bytes(layout: FramesLayout, record: FrameRecord) -> bytes:
    """Return a frame's JPEG bytes regardless of layout."""

    if record.packaging == "per_object":
        frame_path = layout.frames_dir / record.member_name
        return frame_path.read_bytes()

    if not record.archive:
        raise ValueError(f"packed frame {record.frame_id} has no archive reference")
    archive_path = layout.frames_dir / record.archive
    with tarfile.open(archive_path, mode="r:") as archive:
        try:
            member = archive.getmember(record.member_name)
        except KeyError as exc:
            raise FileNotFoundError(
                f"frame member {record.member_name} missing from {archive_path.name}"
            ) from exc
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(
                f"frame member {record.member_name} unreadable in {archive_path.name}"
            )
        return extracted.read()


def iter_frame_payloads(frames_dir: Path) -> Iterator[tuple[FrameRecord, bytes]]:
    """Convenience iterator over (record, jpeg_bytes) for either layout.

    For packed captures archives are opened once each, not per frame.
    """

    layout = load_frames_layout(frames_dir)
    if layout.packaging == "per_object":
        for record in layout.records:
            yield record, read_frame_bytes(layout, record)
        return

    by_archive: Dict[str, List[FrameRecord]] = {}
    for record in layout.records:
        if record.archive:
            by_archive.setdefault(record.archive, []).append(record)
    for archive_name, records in by_archive.items():
        archive_path = frames_dir / archive_name
        with tarfile.open(archive_path, mode="r:") as archive:
            for record in records:
                extracted = archive.extractfile(record.member_name)
                if extracted is None:
                    raise FileNotFoundError(
                        f"frame member {record.member_name} unreadable in {archive_name}"
                    )
                yield record, extracted.read()
