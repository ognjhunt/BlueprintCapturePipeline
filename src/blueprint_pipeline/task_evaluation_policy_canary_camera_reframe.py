"""Derive a high-resolution, task-framed camera request for a policy canary.

The configured camera candidate is intentionally checked in native Isaac
before it is trusted.  Scene 839873 proved why: the external camera framed the
task, but the copied DROID wrist mount pointed away from the object and the
review camera rendered only 320x180.  This adapter consumes that exact native
readback, keeps every camera pose fixed except the wrist rotation, and emits a
new immutable packet request:

* policy cameras render at 640x360 and retain those lossless frames;
* the review-only overview renders at 1280x720;
* the wrist translation stays byte-for-value identical while its +Z OpenCV
  optical axis is aimed at the frozen task start position;
* policy-specific resizing remains owned by ``adp009d_droid_observation``.

No provider is contacted and no camera is called qualified until the derived
request passes a fresh native render.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .native_task_arena_packet import validate_native_task_arena_packet_request


SCHEMA_VERSION = "task_evaluation_policy_canary_camera_reframe.v1"
POLICY_RENDER_RESOLUTION = (640, 360)
OVERVIEW_RENDER_RESOLUTION = (1280, 720)


class PolicyCanaryCameraReframeError(ValueError):
    """The exact request/readback pair cannot author a safe camera variant."""


def _vector(value: Any, length: int, *, blocker: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise PolicyCanaryCameraReframeError(blocker)
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise PolicyCanaryCameraReframeError(blocker) from exc
    if not all(math.isfinite(item) for item in result):
        raise PolicyCanaryCameraReframeError(blocker)
    return result


def _normalize(value: Sequence[float], *, blocker: str) -> list[float]:
    result = _vector(value, 3, blocker=blocker)
    norm = math.sqrt(sum(item * item for item in result))
    if norm <= 1.0e-9:
        raise PolicyCanaryCameraReframeError(blocker)
    return [item / norm for item in result]


def _cross(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _matrix(value: Any, *, blocker: str) -> list[list[float]]:
    flat = _vector(value, 16, blocker=blocker)
    rows = [flat[index : index + 4] for index in range(0, 16, 4)]
    if any(abs(rows[3][index] - expected) > 1.0e-9 for index, expected in enumerate((0, 0, 0, 1))):
        raise PolicyCanaryCameraReframeError(blocker)
    return rows


def _flatten(value: Sequence[Sequence[float]]) -> list[float]:
    return [float(value[row][column]) for row in range(4) for column in range(4)]


def _multiply(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> list[list[float]]:
    return [
        [
            sum(float(left[row][k]) * float(right[k][column]) for k in range(4))
            for column in range(4)
        ]
        for row in range(4)
    ]


def _inverse_rigid(value: Sequence[Sequence[float]]) -> list[list[float]]:
    rotation = [[float(value[row][column]) for column in range(3)] for row in range(3)]
    translation = [float(value[row][3]) for row in range(3)]
    inverse_rotation = [[rotation[column][row] for column in range(3)] for row in range(3)]
    inverse_translation = [
        -sum(inverse_rotation[row][column] * translation[column] for column in range(3))
        for row in range(3)
    ]
    return [
        [*inverse_rotation[0], inverse_translation[0]],
        [*inverse_rotation[1], inverse_translation[1]],
        [*inverse_rotation[2], inverse_translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _quaternion_matrix_xyzw(value: Any) -> list[list[float]]:
    x, y, z, w = _vector(
        value, 4, blocker="policy_canary_camera_reframe_quaternion_invalid"
    )
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-9:
        raise PolicyCanaryCameraReframeError(
            "policy_canary_camera_reframe_quaternion_invalid"
        )
    x, y, z, w = (item / norm for item in (x, y, z, w))
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _pose_matrix(position: Any, quaternion: Any) -> list[list[float]]:
    xyz = _vector(
        position, 3, blocker="policy_canary_camera_reframe_world_pose_invalid"
    )
    rotation = _quaternion_matrix_xyzw(quaternion)
    return [
        [*rotation[0], xyz[0]],
        [*rotation[1], xyz[1]],
        [*rotation[2], xyz[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _look_at(position: Sequence[float], target: Sequence[float]) -> list[list[float]]:
    forward = _normalize(
        [float(target[index]) - float(position[index]) for index in range(3)],
        blocker="policy_canary_camera_reframe_aim_degenerate",
    )
    right = _normalize(
        [forward[1], -forward[0], 0.0],
        blocker="policy_canary_camera_reframe_aim_degenerate",
    )
    down = _normalize(
        _cross(forward, right),
        blocker="policy_canary_camera_reframe_aim_degenerate",
    )
    return [
        [right[0], down[0], forward[0], float(position[0])],
        [right[1], down[1], forward[1], float(position[1])],
        [right[2], down[2], forward[2], float(position[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _scaled_intrinsics(
    value: Any, *, width: int, height: int
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyCanaryCameraReframeError(
            "policy_canary_camera_reframe_intrinsics_invalid"
        )
    try:
        old_width = int(value["width"])
        old_height = int(value["height"])
        fx = float(value["fx"])
        fy = float(value["fy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyCanaryCameraReframeError(
            "policy_canary_camera_reframe_intrinsics_invalid"
        ) from exc
    if old_width < 1 or old_height < 1 or width % old_width or height % old_height:
        raise PolicyCanaryCameraReframeError(
            "policy_canary_camera_reframe_resolution_invalid"
        )
    scaled = json.loads(json.dumps(dict(value), allow_nan=False))
    scaled.update(
        fx=fx * width / old_width,
        fy=fy * height / old_height,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
        width=width,
        height=height,
    )
    return scaled


def materialize_policy_canary_camera_reframe(
    *,
    base_request: Mapping[str, Any],
    runtime_preflight: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Write one digest-bound camera variant requiring fresh native review."""

    request = validate_native_task_arena_packet_request(base_request)
    if (
        runtime_preflight.get("schema_version")
        != "native_task_arena_runtime_preflight.v1"
        or runtime_preflight.get("candidate_policy_queried") is not False
        or runtime_preflight.get("phase_reached") != "environment_built"
    ):
        raise PolicyCanaryCameraReframeError(
            "policy_canary_camera_reframe_preflight_invalid"
        )
    cameras = [dict(row) for row in request.get("cameras") or []]
    by_role = {str(row.get("role") or ""): row for row in cameras}
    snapshots = (runtime_preflight.get("camera_snapshot") or {}).get("cameras")
    snapshot_by_role = {
        str(row.get("role") or ""): dict(row)
        for row in snapshots or []
        if isinstance(row, Mapping)
    }
    if set(by_role) != {"external", "wrist", "overview"} or set(snapshot_by_role) != set(by_role):
        raise PolicyCanaryCameraReframeError(
            "policy_canary_camera_reframe_roles_invalid"
        )
    task_spec = request.get("task_spec") or {}
    start_pose = _vector(
        task_spec.get("start_pose_world"),
        7,
        blocker="policy_canary_camera_reframe_task_pose_invalid",
    )
    target = start_pose[:3]

    for role, camera in by_role.items():
        width, height = (
            OVERVIEW_RENDER_RESOLUTION
            if role == "overview"
            else POLICY_RENDER_RESOLUTION
        )
        camera["intrinsics"] = _scaled_intrinsics(
            camera.get("intrinsics"), width=width, height=height
        )

    wrist = by_role["wrist"]
    if wrist.get("pose_frame") != "robot_body" or wrist.get("policy_input") is not True:
        raise PolicyCanaryCameraReframeError(
            "policy_canary_camera_reframe_wrist_contract_invalid"
        )
    old_relative = _matrix(
        wrist.get("frame_from_camera_matrix"),
        blocker="policy_canary_camera_reframe_wrist_matrix_invalid",
    )
    wrist_snapshot = snapshot_by_role["wrist"]
    old_world = _pose_matrix(
        wrist_snapshot.get("position_world_m"),
        wrist_snapshot.get("quaternion_world_opengl_xyzw"),
    )
    parent_world = _multiply(old_world, _inverse_rigid(old_relative))
    desired_world = _look_at(
        [old_world[row][3] for row in range(3)], target
    )
    new_relative = _multiply(_inverse_rigid(parent_world), desired_world)
    if any(abs(new_relative[row][3] - old_relative[row][3]) > 1.0e-5 for row in range(3)):
        raise PolicyCanaryCameraReframeError(
            "policy_canary_camera_reframe_wrist_translation_changed"
        )
    wrist["frame_from_camera_matrix"] = _flatten(new_relative)

    source_request_digest = request["request_digest"]
    request["cameras"] = [by_role[role] for role in ("external", "wrist", "overview")]
    request["camera_reframe"] = {
        "schema_version": SCHEMA_VERSION,
        "source_request_digest": source_request_digest,
        "source_runtime_preflight_document_digest": canonical_digest(
            runtime_preflight
        ),
        "task_start_position_world_m": target,
        "policy_render_resolution_wh": list(POLICY_RENDER_RESOLUTION),
        "overview_render_resolution_wh": list(OVERVIEW_RENDER_RESOLUTION),
        "policy_preprocessing_contract": (
            "candidate_specific_aspect_preserving_resize_with_centred_black_pad"
        ),
        "wrist_translation_preserved": True,
        "wrist_rotation_aimed_at_task_start": True,
        "camera_configuration_qualified": False,
        "fresh_native_render_required": True,
    }
    request["request_digest"] = ""
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    destination = Path(output_path).expanduser()
    if destination.exists() or destination.is_symlink():
        raise PolicyCanaryCameraReframeError(
            "policy_canary_camera_reframe_output_exists"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, request)
    return json.loads(json.dumps(request))


__all__ = [
    "OVERVIEW_RENDER_RESOLUTION",
    "POLICY_RENDER_RESOLUTION",
    "PolicyCanaryCameraReframeError",
    "SCHEMA_VERSION",
    "materialize_policy_canary_camera_reframe",
]
