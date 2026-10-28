"""Resolve a same-job observation set without registering independent captures."""
from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
import os
import hashlib
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.parse import urlsplit

from .decision_evidence_contracts import canonical_digest

SCHEMA = "website_capture_continuation.v2"


def recording_sources(binding: Mapping[str, Any], *, request_id: str, scene_id: str,
                      capture_id: str, bucket: str | None = None) -> list[dict[str, Any]]:
    """Validate the exact active set; older replacements remain historical."""
    def require(condition):
        if not condition:
            raise ValueError("website_recording_binding_invalid")

    original = f"walkthrough-{request_id}"
    require(set(binding) == {"schema_version", "original_capture_id", "capture_id", "lineage",
                             "coordinate_frames_independent", "recording_capture_ids"})
    require(scene_id == f"site-{request_id}" and binding["schema_version"] == SCHEMA
            and binding["original_capture_id"] == original and binding["capture_id"] == capture_id
            and binding["coordinate_frames_independent"] is True
            and isinstance(binding["lineage"], list) and 1 <= len(binding["lineage"]) <= 8)
    expected, seen, sources, active, taking, previous = capture_id, set(), {}, [capture_id], True, None
    for entry in binding["lineage"]:
        require(isinstance(entry, Mapping) and set(entry) == {"child", "parent", "supplement", "recording_scope"})
        child, parent, link = entry["child"], entry["parent"], entry["supplement"]
        require(all(isinstance(value, Mapping) for value in (child, parent, link)))
        require(child.get("capture_id") == expected and expected not in seen and expected != original)
        require(previous is None or dict(child) == dict(previous))
        seen.add(expected)
        scope = entry["recording_scope"]
        require(isinstance(scope, str) and scope in {"complete_walkthrough", "additional_views"}
                and scope == link.get("recording_scope", "complete_walkthrough"))
        require(set(link) == {"parent_capture_id", "parent_bundle_digest", "parent_manifest_uri", "coverage_digest"}
                | ({"recording_scope"} if scope == "additional_views" else set()))
        require(link.get("parent_capture_id") == parent.get("capture_id")
                and link.get("parent_bundle_digest") == parent.get("raw_bundle_digest")
                and link.get("parent_manifest_uri") == parent.get("raw_manifest_uri")
                and re.fullmatch(r"[a-f0-9]{64}", str(link.get("coverage_digest", ""))))
        for source in (child, parent):
            identifier = source.get("capture_id")
            require(isinstance(identifier, str) and (identifier == original or re.fullmatch(r"supplement-[a-f0-9]{40}", identifier)))
            parsed = urlsplit(str(source.get("raw_manifest_uri", "")))
            source_bucket = parsed.netloc
            require(parsed.scheme == "gs" and source_bucket and not parsed.query and not parsed.fragment
                    and (bucket is None or source_bucket == bucket)
                    and parsed.path == f"/scenes/{scene_id}/captures/{identifier}/raw/manifest.json"
                    and re.fullmatch(r"sha256:[a-f0-9]{64}", str(source.get("raw_bundle_digest", ""))))
            bucket = source_bucket
            require(identifier not in sources or dict(source) == sources[identifier])
            sources[identifier] = dict(source)
        if taking:
            if scope == "additional_views":
                active.append(parent["capture_id"])
            else:
                taking = False
        expected, previous = parent["capture_id"], parent
    require(expected == original and binding["recording_capture_ids"] == list(reversed(active)))
    result = []
    for identifier in reversed(active):
        source = sources[identifier]
        video = source.get("raw_video")
        require(isinstance(video, Mapping)
                and set(video) == {"object_name", "generation", "size_bytes", "crc32c"} | ({"sha256"} if "sha256" in video else set()))
        prefix = f"scenes/{scene_id}/captures/{identifier}/raw/"
        require(isinstance(video["object_name"], str) and video["object_name"] in {prefix + "walkthrough.mov", prefix + "walkthrough.mp4"}
                and re.fullmatch(r"[1-9][0-9]{0,19}", str(video["generation"]))
                and type(video["size_bytes"]) is int and video["size_bytes"] > 0
                and re.fullmatch(r"[A-Za-z0-9+/]{6}==", str(video["crc32c"]))
                and ("sha256" not in video or re.fullmatch(r"sha256:[a-f0-9]{64}", str(video["sha256"]))))
        # Verification/readback annotations may be enriched after publication.
        # They do not change the actual recordings or invalidate method reuse.
        immutable = {key: source.get(key) for key in ("capture_id", "raw_bundle_digest", "raw_manifest_uri", "upload_completion_digest")}
        immutable["raw_video"] = dict(video)
        result.append({"capture_id": identifier, "bucket": bucket, "raw_video": dict(video), "source_identity": immutable})
    return result


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
