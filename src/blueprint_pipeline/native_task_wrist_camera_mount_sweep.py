"""Deterministic wrist-camera candidate generation and selection.

An optional agent may propose robot-preset camera mounts, but it cannot select
or qualify one.  This module validates a digest-bound registry, resolves every
candidate from the live controlled-body position toward the frozen task, and
selects only from real native-render measurements.  It has no Isaac imports so
the decision contract is hermetically testable.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


REGISTRY_SCHEMA_VERSION = "policy_canary_wrist_camera_mount_registry.v1"
CANDIDATE_SCHEMA_VERSION = "policy_canary_wrist_camera_mount_candidate.v1"
SELECTION_SCHEMA_VERSION = "policy_canary_wrist_camera_mount_selection.v1"
MINIMUM_TASK_PIXELS = 120
MINIMUM_TASK_PIXEL_FRACTION = 0.002
MAXIMUM_ROBOT_PIXEL_FRACTION = 0.30
POLICY_RENDER_RESOLUTION = (640, 360)
OVERVIEW_RENDER_RESOLUTION = (1280, 720)


class WristCameraMountSweepError(ValueError):
    """The registry, geometry, or rendered candidate evidence is invalid."""


def _vector(value: Any, *, blocker: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise WristCameraMountSweepError(blocker)
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise WristCameraMountSweepError(blocker) from exc
    if not all(math.isfinite(item) for item in result):
        raise WristCameraMountSweepError(blocker)
    return result


def _normalize(value: Sequence[float], *, blocker: str) -> list[float]:
    result = _vector(value, blocker=blocker)
    norm = math.sqrt(sum(item * item for item in result))
    if norm <= 1.0e-9:
        raise WristCameraMountSweepError(blocker)
    return [item / norm for item in result]


def validate_wrist_camera_mount_registry(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WristCameraMountSweepError("wrist_camera_mount_registry_invalid")
    registry = json.loads(json.dumps(dict(value), allow_nan=False))
    rows = registry.get("candidates")
    if (
        registry.get("schema_version") != REGISTRY_SCHEMA_VERSION
        or not str(registry.get("robot_preset_id") or "").strip()
        or not isinstance(rows, list)
        or not 2 <= len(rows) <= 12
        or registry.get("selection_authority") != "native_render_measurements"
        or registry.get("registry_digest")
        != canonical_digest(registry, digest_field="registry_digest")
    ):
        raise WristCameraMountSweepError("wrist_camera_mount_registry_invalid")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise WristCameraMountSweepError("wrist_camera_mount_candidate_invalid")
        candidate = dict(row)
        candidate_id = str(candidate.get("candidate_id") or "")
        try:
            forward = float(candidate["forward_offset_m"])
            lateral = float(candidate["lateral_offset_m"])
            vertical = float(candidate["vertical_offset_m"])
            maximum = float(candidate["maximum_offset_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WristCameraMountSweepError(
                "wrist_camera_mount_candidate_invalid"
            ) from exc
        magnitude = math.sqrt(forward * forward + lateral * lateral + vertical * vertical)
        if (
            candidate.get("schema_version") != CANDIDATE_SCHEMA_VERSION
            or not candidate_id
            or candidate_id in seen
            or candidate.get("source") != "robot_preset_registry"
            or candidate.get("target_binding") != "task_start_position"
            or not 0.05 <= forward <= 0.24
            or not -0.14 <= lateral <= 0.14
            or not -0.08 <= vertical <= 0.08
            or not magnitude <= maximum <= 0.30
            or candidate.get("candidate_digest")
            != canonical_digest(candidate, digest_field="candidate_digest")
        ):
            raise WristCameraMountSweepError("wrist_camera_mount_candidate_invalid")
        seen.add(candidate_id)
    return registry


def _scaled_intrinsics(
    value: Any, *, width: int, height: int
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WristCameraMountSweepError("wrist_camera_mount_intrinsics_invalid")
    try:
        old_width = int(value["width"])
        old_height = int(value["height"])
        fx = float(value["fx"])
        fy = float(value["fy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WristCameraMountSweepError(
            "wrist_camera_mount_intrinsics_invalid"
        ) from exc
    if old_width < 1 or old_height < 1 or width % old_width or height % old_height:
        raise WristCameraMountSweepError("wrist_camera_mount_resolution_invalid")
    result = json.loads(json.dumps(dict(value), allow_nan=False))
    result.update(
        fx=fx * width / old_width,
        fy=fy * height / old_height,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
        width=width,
        height=height,
    )
    return result


def materialize_wrist_camera_mount_sweep_request(
    *,
    base_request: Mapping[str, Any],
    registry: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind one robot registry and review-quality render sizes to a request."""

    from .native_task_arena_packet import validate_native_task_arena_packet_request

    request = validate_native_task_arena_packet_request(base_request)
    validated_registry = validate_wrist_camera_mount_registry(registry)
    cameras = [dict(row) for row in request.get("cameras") or []]
    by_role = {str(row.get("role") or ""): row for row in cameras}
    if set(by_role) != {"external", "wrist", "overview"}:
        raise WristCameraMountSweepError("wrist_camera_mount_camera_roles_invalid")
    for role, camera in by_role.items():
        width, height = (
            OVERVIEW_RENDER_RESOLUTION
            if role == "overview"
            else POLICY_RENDER_RESOLUTION
        )
        camera["intrinsics"] = _scaled_intrinsics(
            camera.get("intrinsics"), width=width, height=height
        )
    source_request_digest = request["request_digest"]
    request["cameras"] = [by_role[role] for role in ("external", "wrist", "overview")]
    request["wrist_camera_mount_registry"] = validated_registry
    request["camera_resolution_contract"] = {
        "policy_master_resolution_wh": list(POLICY_RENDER_RESOLUTION),
        "overview_review_resolution_wh": list(OVERVIEW_RENDER_RESOLUTION),
        "policy_preprocessing": (
            "candidate_specific_aspect_preserving_resize_with_centred_black_pad"
        ),
        "source_request_digest": source_request_digest,
        "fresh_native_mount_sweep_required": True,
    }
    request["request_digest"] = ""
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    destination = Path(output_path).expanduser()
    if destination.exists() or destination.is_symlink():
        raise WristCameraMountSweepError("wrist_camera_mount_request_output_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return json.loads(json.dumps(request))


def resolve_wrist_camera_mount_eyes(
    *,
    registry: Mapping[str, Any],
    controlled_body_position_world_m: Sequence[float],
    task_target_position_world_m: Sequence[float],
) -> list[dict[str, Any]]:
    """Resolve robot-relative scalar candidates in a task-facing world basis."""

    validated = validate_wrist_camera_mount_registry(registry)
    body = _vector(
        controlled_body_position_world_m,
        blocker="wrist_camera_mount_body_position_invalid",
    )
    target = _vector(
        task_target_position_world_m,
        blocker="wrist_camera_mount_task_target_invalid",
    )
    forward = _normalize(
        [target[index] - body[index] for index in range(3)],
        blocker="wrist_camera_mount_task_direction_degenerate",
    )
    right = _normalize(
        [forward[1], -forward[0], 0.0],
        blocker="wrist_camera_mount_task_direction_degenerate",
    )
    up = [0.0, 0.0, 1.0]
    resolved = []
    for row in validated["candidates"]:
        eye = [
            body[index]
            + float(row["forward_offset_m"]) * forward[index]
            + float(row["lateral_offset_m"]) * right[index]
            + float(row["vertical_offset_m"]) * up[index]
            for index in range(3)
        ]
        resolved.append(
            {
                "candidate_id": row["candidate_id"],
                "candidate_digest": row["candidate_digest"],
                "eye_position_world_m": eye,
                "target_position_world_m": target,
            }
        )
    return resolved


def select_wrist_camera_mount_candidate(
    *, registry: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Select one candidate only from native semantic and RGB measurements."""

    validated = validate_wrist_camera_mount_registry(registry)
    candidate_by_id = {
        str(row["candidate_id"]): row for row in validated["candidates"]
    }
    if len(observations) != len(candidate_by_id):
        raise WristCameraMountSweepError("wrist_camera_mount_observations_invalid")
    rows: list[dict[str, Any]] = []
    for raw in observations:
        row = json.loads(json.dumps(dict(raw), allow_nan=False))
        candidate = candidate_by_id.get(str(row.get("candidate_id") or ""))
        task = row.get("task_object")
        robot = row.get("robot")
        if (
            candidate is None
            or row.get("candidate_digest") != candidate["candidate_digest"]
            or not isinstance(task, Mapping)
            or not isinstance(robot, Mapping)
            or not isinstance(task.get("pixel_count"), int)
            or not isinstance(task.get("pixel_fraction"), (int, float))
            or not isinstance(robot.get("pixel_fraction"), (int, float))
            or row.get("frame_structure_passed") is not True
            or not isinstance(row.get("eye_position_world_m"), list)
            or len(row["eye_position_world_m"]) != 3
            or not isinstance(row.get("target_position_world_m"), list)
            or len(row["target_position_world_m"]) != 3
        ):
            raise WristCameraMountSweepError(
                "wrist_camera_mount_observation_invalid"
            )
        row["admitted"] = bool(
            task["pixel_count"] >= MINIMUM_TASK_PIXELS
            and float(task["pixel_fraction"]) >= MINIMUM_TASK_PIXEL_FRACTION
            and float(robot["pixel_fraction"]) <= MAXIMUM_ROBOT_PIXEL_FRACTION
        )
        row["blockers"] = []
        if task["pixel_count"] < MINIMUM_TASK_PIXELS:
            row["blockers"].append("wrist_camera_mount_task_pixels_below_floor")
        if float(task["pixel_fraction"]) < MINIMUM_TASK_PIXEL_FRACTION:
            row["blockers"].append("wrist_camera_mount_task_fraction_below_floor")
        if float(robot["pixel_fraction"]) > MAXIMUM_ROBOT_PIXEL_FRACTION:
            row["blockers"].append("wrist_camera_mount_robot_occlusion_above_ceiling")
        rows.append(row)
    ids = [str(row["candidate_id"]) for row in rows]
    if set(ids) != set(candidate_by_id) or len(ids) != len(set(ids)):
        raise WristCameraMountSweepError("wrist_camera_mount_observations_invalid")
    admitted = [row for row in rows if row["admitted"]]
    selected = (
        sorted(
            admitted,
            key=lambda row: (
                -int(row["task_object"]["pixel_count"]),
                float(row["robot"]["pixel_fraction"]),
                str(row["candidate_id"]),
            ),
        )[0]
        if admitted
        else None
    )
    result = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": "selected" if selected else "blocked",
        "registry_digest": validated["registry_digest"],
        "thresholds": {
            "minimum_task_pixels": MINIMUM_TASK_PIXELS,
            "minimum_task_pixel_fraction": MINIMUM_TASK_PIXEL_FRACTION,
            "maximum_robot_pixel_fraction": MAXIMUM_ROBOT_PIXEL_FRACTION,
        },
        "observations": rows,
        "selected_candidate": selected,
        "candidate_policy_loaded": False,
        "candidate_policy_queried": False,
        "selection_authority": "native_semantic_aov_and_lossless_rgb",
        "blockers": [] if selected else ["wrist_camera_mount_no_admissible_candidate"],
        "selection_digest": "",
    }
    result["selection_digest"] = canonical_digest(
        result, digest_field="selection_digest"
    )
    return result


__all__ = [
    "CANDIDATE_SCHEMA_VERSION",
    "MAXIMUM_ROBOT_PIXEL_FRACTION",
    "MINIMUM_TASK_PIXEL_FRACTION",
    "MINIMUM_TASK_PIXELS",
    "OVERVIEW_RENDER_RESOLUTION",
    "POLICY_RENDER_RESOLUTION",
    "REGISTRY_SCHEMA_VERSION",
    "SELECTION_SCHEMA_VERSION",
    "WristCameraMountSweepError",
    "materialize_wrist_camera_mount_sweep_request",
    "resolve_wrist_camera_mount_eyes",
    "select_wrist_camera_mount_candidate",
    "validate_wrist_camera_mount_registry",
]
