"""Resolve a same-job observation set without registering independent captures."""
from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
import os
import hashlib
from pathlib import Path
import tempfile
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .website_recording_contracts import SCHEMA, recording_sources

def _verified_digest(path: Path, source: Mapping[str, Any]) -> str | None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != source["size_bytes"]:
        return None
    import google_crc32c
    checksum = google_crc32c.Checksum()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    value = "sha256:" + digest.hexdigest()
    if (before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns
            or base64.b64encode(checksum.digest()).decode("ascii") != source["crc32c"]
            or "sha256" in source and value != source["sha256"]):
        return None
    return value


def _directory(root: Path, path: Path) -> None:
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("website_recording_output_symlink")
        current.mkdir(exist_ok=True, mode=0o700)


def _download(source: Mapping[str, Any], target: Path) -> None:
    from google.cloud import storage
    row = source["raw_video"]
    blob = storage.Client().bucket(source["bucket"]).blob(row["object_name"], generation=int(row["generation"]))
    blob.reload(if_generation_match=int(row["generation"]), timeout=30)
    if (str(blob.generation) != row["generation"] or blob.size != row["size_bytes"] or blob.crc32c != row["crc32c"]):
        raise ValueError("website_recording_source_changed")
    blob.download_to_filename(str(target), if_generation_match=int(row["generation"]), checksum="crc32c", timeout=180)


def resolve_recording_video(*, task_context: Mapping[str, Any], capture_root: Path,
                            source_video: Path | None, bucket: str | None = None,
                            download: Callable[[Mapping[str, Any], Path], None] | None = None) -> Path | None:
    """Use the current exact recordings in the existing single-video processors.

    Rights and lineage are current WebApp evidence, never inferred from local
    filenames. Downloads are generation-pinned and verified before decoding.
    Original media, including a mismatching local copy, is never overwritten.
    """
    binding = task_context.get("capture_binding") or {}
    if binding.get("schema_version") != SCHEMA:
        if binding and (binding.get("schema_version") != "website_capture_continuation.v1" or any(
                entry.get("recording_scope") == "additional_views"
                or (entry.get("supplement") or {}).get("recording_scope") == "additional_views"
                for entry in binding.get("lineage", []))):
            raise ValueError("website_recording_binding_invalid")
        return source_video
    sources = recording_sources(binding, request_id=task_context["request_id"], scene_id=task_context["scene_id"],
                                capture_id=task_context["capture_id"], bucket=bucket)
    bucket = sources[0]["bucket"]
    rights = task_context.get("capture_rights") or {}
    if rights.get("derived_scene_generation_allowed") is not True or rights.get("consent_revoked") is True:
        raise ValueError("source_revoked")
    root = capture_root.resolve()
    if (capture_root.is_symlink() or root.name != task_context["capture_id"]
            or root.parent.parent.name != task_context["scene_id"]):
        raise ValueError("website_recording_capture_root_invalid")
    output = root / "pipeline/recording_inputs"
    if (root / "pipeline").is_symlink() or output.is_symlink():
        raise ValueError("website_recording_output_symlink")
    from .consent_takedown import read_consent_state

    def current():
        state = read_consent_state(root)
        if state.get("state") == "revoked" or "website_withdrawal/tombstone.json" in (state.get("source_path") or ""):
            raise ValueError("website_capture_withdrawn")

    current()
    recordings = []
    for source in sources:
        current()
        video = source["raw_video"]
        relative = Path(video["object_name"]).name
        local = source_video if source["capture_id"] == root.name and source_video is not None else (
            root.parent / source["capture_id"] / "raw" / relative)
        # A staged original is reusable only after proving its actual bytes.
        content_digest = (_verified_digest(local, video)
                          if local.resolve().is_relative_to(root.parent) else None)
        if content_digest is None:
            directory = output / "sources" / canonical_digest({"bucket": bucket, "raw_video": video})[7:]
            _directory(root, directory)
            local = directory / relative
            if local.is_symlink():
                raise ValueError("website_recording_output_symlink")
            content_digest = _verified_digest(local, video)
            if content_digest is None:
                fd, name = tempfile.mkstemp(prefix=".download-", dir=directory)
                os.close(fd)
                temporary = Path(name)
                try:
                    (download or _download)(source, temporary)
                    content_digest = _verified_digest(temporary, video)
                    if content_digest is None:
                        raise ValueError("website_recording_source_changed")
                    if local.exists():
                        # Preserve an unexpected retained derivative rather than
                        # silently replacing it with newly downloaded bytes.
                        raise ValueError("website_recording_cached_source_changed")
                    try:
                        os.link(temporary, local)
                    except FileExistsError:
                        content_digest = _verified_digest(local, video)
                        if content_digest is None:
                            raise ValueError("website_recording_cached_source_changed") from None
                finally:
                    temporary.unlink(missing_ok=True)
        recordings.append({"capture_id": source["capture_id"], "source_video": local,
                           "source_video_digest": content_digest, "source_identity": source["source_identity"]})
    current()
    if len(recordings) == 1:
        return Path(recordings[0]["source_video"])
    from .website_recording_composition import compose_recordings
    _directory(root, output / "compositions")
    result = compose_recordings(recordings=recordings, output_root=output / "compositions")
    current()
    return Path(result["source_video"])
