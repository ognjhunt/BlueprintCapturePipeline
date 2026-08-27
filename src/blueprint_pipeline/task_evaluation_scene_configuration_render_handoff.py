"""Carry provider-rendered reference frames between configuration stages.

The construction envelope is immutable.  When stage 1 completes a render that
the control plane deliberately deferred to the provider, later stages cannot
learn about those new files by rereading that envelope.  This module seals the
completed reference frames as an ordinary digest-bound stage artifact instead.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_scene_configuration_disclosure import MATERIALIZED_STATUS


SCHEMA_VERSION = "task_evaluation_scene_configuration_render_handoff.v1"
ARTIFACT_ROLE = "provider_render_reference_manifest"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


class TaskEvaluationSceneConfigurationRenderHandoffError(RuntimeError):
    """A completed render could not cross the immutable stage boundary."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def materialize_provider_render_handoff(
    *, render_inputs: Mapping[str, Any], output_root: str | Path
) -> dict[str, Any]:
    """Copy completed reference frames into one portable sealed artifact."""

    frames = render_inputs.get("derived_frames")
    if (
        render_inputs.get("status") != MATERIALIZED_STATUS
        or render_inputs.get("result_digest")
        != canonical_digest(render_inputs, digest_field="result_digest")
        or not isinstance(frames, list)
        or not frames
        or render_inputs.get("derived_frame_count") != len(frames)
    ):
        raise TaskEvaluationSceneConfigurationRenderHandoffError(
            "scene_configuration_render_handoff_input_invalid"
        )
    root = Path(output_root).expanduser().resolve()
    references = root / "provider_render_references"
    references.mkdir(mode=0o750)
    sealed_frames: list[dict[str, Any]] = []
    camera_ids: set[str] = set()
    for index, row in enumerate(frames):
        if not isinstance(row, Mapping):
            raise TaskEvaluationSceneConfigurationRenderHandoffError(
                "scene_configuration_render_handoff_frame_invalid"
            )
        camera_id = str(row.get("camera_id") or "")
        unresolved_source = Path(str(row.get("path") or "")).expanduser()
        source = unresolved_source.resolve()
        if (
            not camera_id
            or camera_id in camera_ids
            or unresolved_source.is_symlink()
            or not source.is_file()
            or source.stat().st_size != row.get("size_bytes")
            or _sha256(source) != row.get("digest")
        ):
            raise TaskEvaluationSceneConfigurationRenderHandoffError(
                "scene_configuration_render_handoff_frame_invalid"
            )
        camera_ids.add(camera_id)
        suffix = source.suffix.lower() or ".bin"
        destination = references / f"{index:04d}{suffix}"
        shutil.copyfile(source, destination)
        if destination.stat().st_size != source.stat().st_size or _sha256(
            destination
        ) != _sha256(source):
            raise TaskEvaluationSceneConfigurationRenderHandoffError(
                "scene_configuration_render_handoff_copy_mismatch"
            )
        sealed_frames.append(
            {
                "camera_id": camera_id,
                "relative_path": destination.relative_to(root).as_posix(),
                "digest": _sha256(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
    control_plane_digest = str(
        render_inputs.get("control_plane_result_digest")
        or render_inputs.get("result_digest")
        or ""
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_render_references_sealed",
        "source_render_result_digest": render_inputs["result_digest"],
        "control_plane_render_result_digest": control_plane_digest,
        "render_completed_on_provider": (
            render_inputs.get("render_completed_on_provider") is True
        ),
        "frame_count": len(sealed_frames),
        "frames": sealed_frames,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    path = root / f"{SCHEMA_VERSION}.json"
    path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return {
        "role": ARTIFACT_ROLE,
        "path": str(path),
        "digest": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def validate_provider_render_handoff(
    path: str | Path,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Reopen the manifest and every referenced frame byte for byte."""

    unresolved_manifest_path = Path(path).expanduser()
    manifest_path = unresolved_manifest_path.resolve()
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationRenderHandoffError(
            "scene_configuration_render_handoff_invalid"
        ) from exc
    expected_keys = {
        "schema_version",
        "status",
        "source_render_result_digest",
        "control_plane_render_result_digest",
        "render_completed_on_provider",
        "frame_count",
        "frames",
        "manifest_digest",
    }
    frames = value.get("frames") if isinstance(value, Mapping) else None
    if (
        unresolved_manifest_path.is_symlink()
        or not isinstance(value, Mapping)
        or set(value) != expected_keys
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "completed_render_references_sealed"
        or value.get("manifest_digest")
        != canonical_digest(value, digest_field="manifest_digest")
        or _DIGEST.fullmatch(str(value.get("source_render_result_digest") or ""))
        is None
        or _DIGEST.fullmatch(
            str(value.get("control_plane_render_result_digest") or "")
        )
        is None
        or not isinstance(value.get("render_completed_on_provider"), bool)
        or not isinstance(frames, list)
        or not frames
        or value.get("frame_count") != len(frames)
    ):
        raise TaskEvaluationSceneConfigurationRenderHandoffError(
            "scene_configuration_render_handoff_invalid"
        )
    resolved: list[Path] = []
    camera_ids: set[str] = set()
    for row in frames:
        relative = str(row.get("relative_path") or "") if isinstance(row, Mapping) else ""
        camera_id = str(row.get("camera_id") or "") if isinstance(row, Mapping) else ""
        unresolved_candidate = manifest_path.parent / relative
        candidate = unresolved_candidate.resolve()
        try:
            candidate.relative_to(manifest_path.parent)
        except ValueError as exc:
            raise TaskEvaluationSceneConfigurationRenderHandoffError(
                "scene_configuration_render_handoff_frame_invalid"
            ) from exc
        if (
            not isinstance(row, Mapping)
            or set(row) != {"camera_id", "relative_path", "digest", "size_bytes"}
            or not camera_id
            or camera_id in camera_ids
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or unresolved_candidate.is_symlink()
            or not candidate.is_file()
            or candidate.stat().st_size != row.get("size_bytes")
            or _sha256(candidate) != row.get("digest")
        ):
            raise TaskEvaluationSceneConfigurationRenderHandoffError(
                "scene_configuration_render_handoff_frame_invalid"
            )
        camera_ids.add(camera_id)
        resolved.append(candidate)
    return dict(value), tuple(resolved)


__all__ = [
    "ARTIFACT_ROLE",
    "SCHEMA_VERSION",
    "TaskEvaluationSceneConfigurationRenderHandoffError",
    "materialize_provider_render_handoff",
    "validate_provider_render_handoff",
]
