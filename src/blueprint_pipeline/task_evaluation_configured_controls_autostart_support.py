"""Helpers the configured-controls autostart shares with its sibling modules.

The autostart spine sits at its module line budget.  Its error type, small
validators, and the placement-aware camera derivation live here so the deferred
input resolver and the camera candidates can import them without a cycle.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_robot_placement_trajectory import (
    validate_robot_placement_trajectory,
)


_PLACEMENT_CAMERA_SCHEMA_VERSION = (
    "task_evaluation_placement_aware_camera_candidates.v1"
)


class TaskEvaluationConfiguredControlsAutostartError(RuntimeError):
    """Automatic continuation input or CPU evidence was unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, blocker: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationConfiguredControlsAutostartError(blocker) from exc
    if path.is_symlink() or not path.is_file() or not isinstance(value, Mapping):
        raise TaskEvaluationConfiguredControlsAutostartError(blocker)
    return dict(value)


def _finite_vector(value: Any, length: int, *, blocker: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(blocker)
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationConfiguredControlsAutostartError(blocker) from exc
    if not all(math.isfinite(item) for item in result):
        raise TaskEvaluationConfiguredControlsAutostartError(blocker)
    return result


def _normalize_vector(value: Sequence[float], *, blocker: str) -> list[float]:
    norm = math.sqrt(sum(float(item) ** 2 for item in value))
    if not math.isfinite(norm) or norm <= 1.0e-9:
        raise TaskEvaluationConfiguredControlsAutostartError(blocker)
    return [float(item) / norm for item in value]


def _look_at_matrix(
    position: Sequence[float], target: Sequence[float]
) -> list[float]:
    forward = _normalize_vector(
        [float(target[index]) - float(position[index]) for index in range(3)],
        blocker="configured_controls_autostart_camera_aim_degenerate",
    )
    right = _normalize_vector(
        [forward[1], -forward[0], 0.0],
        blocker="configured_controls_autostart_camera_aim_degenerate",
    )
    down = _normalize_vector(
        [
            forward[1] * right[2] - forward[2] * right[1],
            forward[2] * right[0] - forward[0] * right[2],
            forward[0] * right[1] - forward[1] * right[0],
        ],
        blocker="configured_controls_autostart_camera_aim_degenerate",
    )
    return [
        right[0], down[0], forward[0], float(position[0]),
        right[1], down[1], forward[1], float(position[1]),
        right[2], down[2], forward[2], float(position[2]),
        0.0, 0.0, 0.0, 1.0,
    ]


def _world_camera_candidate(
    role: str,
    *,
    position: Sequence[float],
    target: Sequence[float],
    intrinsics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "role": role,
        "policy_input": role == "external",
        "scoring_input": False,
        "pose_frame": "world",
        "parent_prim_path": "{ENV_REGEX_NS}",
        "optical_convention": "opencv",
        "frame_from_camera_matrix": _look_at_matrix(position, target),
        "intrinsics": json.loads(json.dumps(dict(intrinsics), allow_nan=False)),
    }


def _placement_aware_camera_candidates(
    *,
    camera_template: Mapping[str, Any],
    accepted_pose: Mapping[str, Any],
    selected_candidate_id: str,
    trajectory: Mapping[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    """Derive world cameras after CPU placement; retain only the wrist mount.

    A prelaunch profile can bind immutable DROID intrinsics and its robot-body
    wrist mount.  Its world cameras cannot be authoritative because the exact
    Franka base does not exist until deterministic trajectory placement has
    selected one inventory member.  Recompute those two cameras from the exact
    selected pose and full trajectory, and leave native observability as the
    final authority.
    """

    validated_trajectory = validate_robot_placement_trajectory(trajectory)
    rows = camera_template.get("cameras")
    by_role = {
        str(row.get("role") or ""): json.loads(json.dumps(dict(row), allow_nan=False))
        for row in rows or []
        if isinstance(row, Mapping)
    }
    wrist = by_role.get("wrist") or {}
    intrinsics = (by_role.get("external") or {}).get("intrinsics")
    if (
        not isinstance(rows, list)
        or len(rows) != 3
        or set(by_role) != {"external", "wrist", "overview"}
        or wrist.get("pose_frame") != "robot_body"
        or wrist.get("parent_prim_path")
        != "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
        or wrist.get("policy_input") is not True
        or wrist.get("scoring_input") is not False
        or wrist.get("optical_convention") != "opencv"
        or not isinstance(intrinsics, Mapping)
        or intrinsics != wrist.get("intrinsics")
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_camera_template_invalid"
        )
    base = _finite_vector(
        accepted_pose.get("position_world_m"),
        3,
        blocker="configured_controls_autostart_camera_base_pose_invalid",
    )
    orientation = _finite_vector(
        accepted_pose.get("orientation_xyzw"),
        4,
        blocker="configured_controls_autostart_camera_base_pose_invalid",
    )
    if not math.isclose(
        sum(item * item for item in orientation),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_camera_base_pose_invalid"
        )
    phases = validated_trajectory["phases"]
    points = [
        _finite_vector(
            row.get("position_world_m"),
            3,
            blocker="configured_controls_autostart_camera_trajectory_invalid",
        )
        for row in phases
    ]
    focus = [
        sum(point[index] for point in points) / len(points) for index in range(3)
    ]
    longest = max(
        (
            (
                math.hypot(right[0] - left[0], right[1] - left[1]),
                left,
                right,
            )
            for left in points
            for right in points
        ),
        key=lambda row: row[0],
    )
    if longest[0] <= 1.0e-9:
        base_to_focus = [focus[0] - base[0], focus[1] - base[1], 0.0]
        if math.hypot(base_to_focus[0], base_to_focus[1]) <= 1.0e-9:
            x, y, z, w = orientation
            base_to_focus = [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y + w * z),
                0.0,
            ]
        direction = _normalize_vector(
            base_to_focus,
            blocker="configured_controls_autostart_camera_trajectory_invalid",
        )
    else:
        direction = _normalize_vector(
            [
                longest[2][0] - longest[1][0],
                longest[2][1] - longest[1][1],
                0.0,
            ],
            blocker="configured_controls_autostart_camera_trajectory_invalid",
        )
    lateral = [-direction[1], direction[0], 0.0]
    external_position = [base[0], base[1], base[2] + 1.35]
    overview_position = [
        focus[0] + 0.9 * lateral[0],
        focus[1] + 0.9 * lateral[1],
        max(base[2], max(point[2] for point in points)) + 1.45,
    ]
    cameras = [
        _world_camera_candidate(
            "external",
            position=external_position,
            target=focus,
            intrinsics=intrinsics,
        ),
        wrist,
        _world_camera_candidate(
            "overview",
            position=overview_position,
            target=focus,
            # The review-only overview may run at a higher resolution than the
            # policy cameras; keep the template's own intrinsics for it.
            intrinsics=(by_role.get("overview") or {}).get("intrinsics") or intrinsics,
        ),
    ]
    cameras[2]["policy_input"] = False
    result: dict[str, Any] = {
        "schema_version": _PLACEMENT_CAMERA_SCHEMA_VERSION,
        "status": "candidate_pending_native_observability_readback",
        "source_commit": source_commit,
        "selected_candidate_id": selected_candidate_id,
        "accepted_pose": {
            "position_world_m": base,
            "orientation_xyzw": orientation,
        },
        "trajectory_digest": validated_trajectory["trajectory_digest"],
        "camera_template_digest": canonical_digest(camera_template),
        "derivation_method": "selected_base_and_full_trajectory_look_at",
        "world_camera_positions_depend_on_selected_base": True,
        "wrist_mount_copied_from_immutable_profile": True,
        "camera_configuration_qualified": False,
        "native_observability_readback_required": True,
        "cameras": cameras,
        "document_digest": "",
    }
    result["document_digest"] = canonical_digest(
        result, digest_field="document_digest"
    )
    return result


def _materialize_placement_aware_cameras(
    *,
    root: Path,
    camera_template_path: Path,
    accepted_pose: Mapping[str, Any],
    selected_candidate_id: str,
    trajectory: Mapping[str, Any],
    source_commit: str,
) -> Path:
    value = _placement_aware_camera_candidates(
        camera_template=_read(
            camera_template_path,
            blocker="configured_controls_autostart_camera_template_invalid",
        ),
        accepted_pose=accepted_pose,
        selected_candidate_id=selected_candidate_id,
        trajectory=trajectory,
        source_commit=source_commit,
    )
    # The document embeds source_commit, so the filename must carry it too;
    # otherwise each redeploy collides with its predecessor bytes.
    destination = (
        root / f"placement-aware-camera-candidates-{source_commit[:12]}.v1.json"
    )
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
        destination.chmod(0o440)
    except FileExistsError:
        if destination.is_symlink() or destination.read_bytes() != payload:
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_camera_candidate_conflict"
            ) from None
    return destination


__all__ = [
    "TaskEvaluationConfiguredControlsAutostartError",
    "_finite_vector",
    "_look_at_matrix",
    "_materialize_placement_aware_cameras",
    "_normalize_vector",
    "_placement_aware_camera_candidates",
    "_read",
    "_sha256",
    "_world_camera_candidate",
]
