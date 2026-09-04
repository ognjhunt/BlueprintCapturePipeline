"""Seal executor-owned native measurements for one rigid destination placement.

This module is deliberately policy-free.  The Isaac worker supplies measured
poses, contact forces, penetration depths, semantic pixels, camera calibration,
and retained artifact paths.  This boundary re-hashes those artifacts and
binds the measurements to the exact release and input identities; it never
turns them into a qualification decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "task_evaluation_rigid_destination_native_observation.v1"
REQUEST_SCHEMA_VERSION = "task_evaluation_rigid_destination_native_probe_request.v1"
PRODUCER = "native_task_arena_destination_qualification"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}\Z")


class RigidDestinationNativeObservationError(ValueError):
    """The executor attempted to seal incomplete or rebound measurements."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy(value: Any, *, blocker: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise RigidDestinationNativeObservationError(blocker) from exc


def _pose(value: Any, *, blocker: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 7
    ):
        raise RigidDestinationNativeObservationError(blocker)
    try:
        pose = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise RigidDestinationNativeObservationError(blocker) from exc
    if not all(math.isfinite(item) for item in pose) or not math.isclose(
        sum(item * item for item in pose[3:]),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise RigidDestinationNativeObservationError(blocker)
    return pose


def _positive(mapping: Mapping[str, Any], field: str) -> float:
    try:
        value = float(mapping[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise RigidDestinationNativeObservationError(
            f"rigid_destination_native_probe_limit_invalid:{field}"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise RigidDestinationNativeObservationError(
            f"rigid_destination_native_probe_limit_invalid:{field}"
        )
    return value


def validate_rigid_destination_native_probe_request(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the control-plane-authored, executor-consumed probe request."""

    request = _copy(
        dict(value), blocker="rigid_destination_native_probe_request_invalid"
    )
    identity = request.get("destination_identity")
    runtime = request.get("runtime_identity")
    container = request.get("container_identity")
    pose = request.get("pose_world")
    limits = request.get("qualification_limits")
    if (
        request.get("schema_version") != REQUEST_SCHEMA_VERSION
        or _COMMIT.fullmatch(str(request.get("execution_commit") or "")) is None
        or not isinstance(identity, Mapping)
        or set(identity) != {"id", "version"}
        or any(_IDENTIFIER.fullmatch(str(identity.get(field) or "")) is None for field in identity)
        or not isinstance(runtime, Mapping)
        or set(runtime) != {"id", "version"}
        or any(_IDENTIFIER.fullmatch(str(runtime.get(field) or "")) is None for field in runtime)
        or not isinstance(container, Mapping)
        or set(container) != {"image", "digest"}
        or "@sha256:" not in str(container.get("image") or "")
        or _DIGEST.fullmatch(str(container.get("digest") or "")) is None
        or any(
            _DIGEST.fullmatch(str(request.get(field) or "")) is None
            for field in (
                "configured_scene_revision_digest",
                "configured_scene_collision_digest",
                "configured_scene_support_plane_digest",
                "destination_asset_digest",
                "destination_static_qualification_digest",
                "destination_native_import_qualification_digest",
                "destination_geometry_digest",
            )
        )
        or not isinstance(pose, Mapping)
        or set(pose) != {"position_world_m", "orientation_xyzw"}
        or not isinstance(limits, Mapping)
        or set(limits)
        != {
            "maximum_penetration_m",
            "minimum_support_contact_force_n",
            "maximum_forbidden_contact_force_n",
            "settle_translation_tolerance_m",
            "settle_rotation_tolerance_rad",
            "reset_translation_tolerance_m",
            "reset_rotation_tolerance_rad",
            "minimum_camera_pixels",
        }
        or not isinstance(request.get("settle_sample_count"), int)
        or isinstance(request.get("settle_sample_count"), bool)
        or request["settle_sample_count"] < 3
        or not isinstance(request.get("settle_steps_per_sample"), int)
        or isinstance(request.get("settle_steps_per_sample"), bool)
        or request["settle_steps_per_sample"] < 1
        or request.get("candidate_policy_queried") is not False
        or request.get("policy_loaded") is not False
        or request.get("request_digest")
        != canonical_digest(request, digest_field="request_digest")
    ):
        raise RigidDestinationNativeObservationError(
            "rigid_destination_native_probe_request_invalid"
        )
    _pose(
        [*(pose.get("position_world_m") or []), *(pose.get("orientation_xyzw") or [])],
        blocker="rigid_destination_native_probe_pose_invalid",
    )
    for field in (
        "maximum_penetration_m",
        "minimum_support_contact_force_n",
        "maximum_forbidden_contact_force_n",
        "settle_translation_tolerance_m",
        "settle_rotation_tolerance_rad",
        "reset_translation_tolerance_m",
        "reset_rotation_tolerance_rad",
    ):
        _positive(limits, field)
    camera_limits = limits.get("minimum_camera_pixels")
    if (
        not isinstance(camera_limits, Mapping)
        or set(camera_limits) != {"external", "wrist", "overview"}
        or any(
            not isinstance(camera_limits[role], int)
            or isinstance(camera_limits[role], bool)
            or camera_limits[role] < 1
            for role in camera_limits
        )
    ):
        raise RigidDestinationNativeObservationError(
            "rigid_destination_native_probe_camera_limits_invalid"
        )
    return request


def _artifact_rows(
    artifacts: Sequence[Mapping[str, Any]], *, artifact_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(artifacts):
        role = str(raw.get("role") or "") if isinstance(raw, Mapping) else ""
        relative_text = (
            str(raw.get("relative_path") or "") if isinstance(raw, Mapping) else ""
        )
        relative = PurePosixPath(relative_text)
        if (
            _IDENTIFIER.fullmatch(role) is None
            or not relative_text
            or relative.is_absolute()
            or ".." in relative.parts
            or relative_text in seen
        ):
            raise RigidDestinationNativeObservationError(
                f"rigid_destination_native_probe_artifact_invalid:{index}"
            )
        path = artifact_root.joinpath(*relative.parts).resolve()
        if (
            path.is_symlink()
            or not path.is_file()
            or (path != artifact_root and artifact_root not in path.parents)
        ):
            raise RigidDestinationNativeObservationError(
                f"rigid_destination_native_probe_artifact_invalid:{index}"
            )
        seen.add(relative_text)
        rows.append(
            {
                "role": role,
                "relative_path": relative.as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if not rows:
        raise RigidDestinationNativeObservationError(
            "rigid_destination_native_probe_artifacts_missing"
        )
    return rows


def materialize_rigid_destination_native_observation(
    *,
    request: Mapping[str, Any],
    execution_manifest: Mapping[str, Any],
    settle_samples: Sequence[Mapping[str, Any]],
    reset_samples: Sequence[Mapping[str, Any]],
    camera_observations: Sequence[Mapping[str, Any]],
    raw_measurement_artifacts: Sequence[Mapping[str, Any]],
    artifact_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal native measurements without deciding whether the placement passes."""

    probe = validate_rigid_destination_native_probe_request(request)
    manifest = _copy(
        dict(execution_manifest),
        blocker="rigid_destination_native_probe_execution_manifest_invalid",
    )
    if (
        manifest.get("implementation_commit") != probe["execution_commit"]
        or manifest.get("container_image") != probe["container_identity"]["image"]
        or manifest.get("execution_mode") != "destination_qualification"
        or manifest.get("policy_candidate_id") not in (None, "")
    ):
        raise RigidDestinationNativeObservationError(
            "rigid_destination_native_probe_execution_manifest_invalid"
        )
    settle = _copy(
        list(settle_samples), blocker="rigid_destination_native_probe_settle_invalid"
    )
    resets = _copy(
        list(reset_samples), blocker="rigid_destination_native_probe_reset_invalid"
    )
    cameras = _copy(
        list(camera_observations),
        blocker="rigid_destination_native_probe_cameras_invalid",
    )
    if len(settle) != probe["settle_sample_count"] or len(resets) < 3:
        raise RigidDestinationNativeObservationError(
            "rigid_destination_native_probe_sample_count_invalid"
        )
    for index, row in enumerate(settle):
        if not isinstance(row, Mapping) or set(row) != {
            "sample_index",
            "destination_pose_world",
            "maximum_penetration_m",
            "support_contact_peak_force_n",
            "forbidden_contact_peak_force_n",
        }:
            raise RigidDestinationNativeObservationError(
                f"rigid_destination_native_probe_settle_invalid:{index}"
            )
        _pose(
            row["destination_pose_world"],
            blocker=f"rigid_destination_native_probe_settle_pose_invalid:{index}",
        )
        numeric = [
            float(row["maximum_penetration_m"]),
            float(row["support_contact_peak_force_n"]),
            float(row["forbidden_contact_peak_force_n"]),
        ]
        if row["sample_index"] != index or not all(
            math.isfinite(item) and item >= 0.0 for item in numeric
        ):
            raise RigidDestinationNativeObservationError(
                f"rigid_destination_native_probe_settle_invalid:{index}"
            )
    for index, row in enumerate(resets):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"sample_index", "destination_pose_world"}
            or row["sample_index"] != index
        ):
            raise RigidDestinationNativeObservationError(
                f"rigid_destination_native_probe_reset_invalid:{index}"
            )
        _pose(
            row["destination_pose_world"],
            blocker=f"rigid_destination_native_probe_reset_pose_invalid:{index}",
        )
    by_role: dict[str, Mapping[str, Any]] = {}
    for row in cameras:
        role = str(row.get("role") or "") if isinstance(row, Mapping) else ""
        if role in by_role:
            raise RigidDestinationNativeObservationError(
                "rigid_destination_native_probe_cameras_invalid"
            )
        by_role[role] = row
    if set(by_role) != {"external", "wrist", "overview"}:
        raise RigidDestinationNativeObservationError(
            "rigid_destination_native_probe_cameras_invalid"
        )
    for role, row in by_role.items():
        if (
            not isinstance(row.get("task_support_pixel_count"), int)
            or isinstance(row.get("task_support_pixel_count"), bool)
            or row["task_support_pixel_count"] < 0
            or not isinstance(row.get("camera_calibration"), Mapping)
            or _DIGEST.fullmatch(str(row.get("render_receipt_digest") or "")) is None
        ):
            raise RigidDestinationNativeObservationError(
                f"rigid_destination_native_probe_camera_invalid:{role}"
            )
    root = Path(artifact_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise RigidDestinationNativeObservationError(
            "rigid_destination_native_probe_artifact_root_invalid"
        )
    artifact_rows = _artifact_rows(raw_measurement_artifacts, artifact_root=root)
    observation: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "producer": PRODUCER,
        "native_isaac_executed": True,
        "execution_commit": probe["execution_commit"],
        "runtime_identity": probe["runtime_identity"],
        "container_identity": probe["container_identity"],
        "destination_identity": probe["destination_identity"],
        "configured_scene_revision_digest": probe[
            "configured_scene_revision_digest"
        ],
        "configured_scene_collision_digest": probe[
            "configured_scene_collision_digest"
        ],
        "configured_scene_support_plane_digest": probe[
            "configured_scene_support_plane_digest"
        ],
        "destination_asset_digest": probe["destination_asset_digest"],
        "destination_static_qualification_digest": probe[
            "destination_static_qualification_digest"
        ],
        "destination_native_import_qualification_digest": probe[
            "destination_native_import_qualification_digest"
        ],
        "destination_geometry_digest": probe["destination_geometry_digest"],
        "pose_world": probe["pose_world"],
        "qualification_limits": probe["qualification_limits"],
        "settle_samples": settle,
        "reset_samples": resets,
        "camera_observations": [by_role[role] for role in ("external", "wrist", "overview")],
        "raw_measurement_artifacts": artifact_rows,
        "no_policy_execution": {
            "policy_loaded": False,
            "candidate_policy_queried": False,
            "candidate_outcomes_accessed": False,
            "policy_actions_executed": 0,
        },
        "observation_digest": "",
    }
    observation["observation_digest"] = canonical_digest(
        observation, digest_field="observation_digest"
    )
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise RigidDestinationNativeObservationError(
            "rigid_destination_native_probe_output_exists"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, observation)
    destination.chmod(0o440)
    return observation


__all__ = [
    "PRODUCER",
    "REQUEST_SCHEMA_VERSION",
    "RigidDestinationNativeObservationError",
    "SCHEMA_VERSION",
    "materialize_rigid_destination_native_observation",
    "validate_rigid_destination_native_probe_request",
]
