"""Author a task-direction-aware Franka base and DROID camera candidate.

The scene configuration output is robot neutral.  Its workspace packet may
retain a reach-oriented base proposal, but that proposal did not consider the
configured planar-push direction and explicitly abstained from native
qualification.  This adapter reflects the retained standoff behind the task
start along the frozen push direction, copies only the reusable DROID wrist
mount/intrinsics, and leaves every native reachability/collision/camera claim
false until construction readback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_configured_scene_revision import (
    validate_configured_scene_revision,
)


SCHEMA_VERSION = "task_evaluation_planar_push_readiness_candidate.v1"


class TaskEvaluationPlanarPushReadinessCandidateError(ValueError):
    """The exact configured evidence cannot author a bounded candidate."""


def _identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _read_bound(
    path: str | Path, reference: Mapping[str, Any], *, blocker: str
) -> dict[str, Any]:
    unresolved = Path(path).expanduser()
    resolved = unresolved.resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationPlanarPushReadinessCandidateError(blocker) from exc
    if (
        unresolved.is_symlink()
        or not resolved.is_file()
        or _identity(resolved)
        != (reference.get("digest"), reference.get("size_bytes"))
        or not isinstance(value, Mapping)
    ):
        raise TaskEvaluationPlanarPushReadinessCandidateError(blocker)
    return dict(value)


def _vector(value: Any, length: int, *, blocker: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise TaskEvaluationPlanarPushReadinessCandidateError(blocker)
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationPlanarPushReadinessCandidateError(blocker) from exc
    if not all(math.isfinite(item) for item in result):
        raise TaskEvaluationPlanarPushReadinessCandidateError(blocker)
    return result


def _normalize(value: Sequence[float], *, blocker: str) -> list[float]:
    norm = math.sqrt(sum(float(item) ** 2 for item in value))
    if not math.isfinite(norm) or norm <= 1.0e-9:
        raise TaskEvaluationPlanarPushReadinessCandidateError(blocker)
    return [float(item) / norm for item in value]


def _cross(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _look_at(position: Sequence[float], target: Sequence[float]) -> list[float]:
    forward = _normalize(
        [target[index] - position[index] for index in range(3)],
        blocker="planar_push_readiness_camera_aim_degenerate",
    )
    right = _normalize(
        _cross(forward, [0.0, 0.0, 1.0]),
        blocker="planar_push_readiness_camera_aim_degenerate",
    )
    down = _normalize(
        _cross(forward, right),
        blocker="planar_push_readiness_camera_aim_degenerate",
    )
    return [
        right[0], down[0], forward[0], float(position[0]),
        right[1], down[1], forward[1], float(position[1]),
        right[2], down[2], forward[2], float(position[2]),
        0.0, 0.0, 0.0, 1.0,
    ]


def _world_camera(
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
        "frame_from_camera_matrix": _look_at(position, target),
        "intrinsics": json.loads(json.dumps(intrinsics)),
    }


def materialize_planar_push_readiness_candidate(
    *,
    configured_revision: Mapping[str, Any],
    task_definition_path: str | Path,
    workspace_clearance_path: str | Path,
    droid_profile_path: str | Path,
    droid_profile_reference: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Write one digest-bound diagnostic base/camera candidate."""

    revision = validate_configured_scene_revision(configured_revision)
    task = _read_bound(
        task_definition_path,
        revision["task_template"]["definition"],
        blocker="planar_push_readiness_task_definition_binding_invalid",
    )
    workspace = _read_bound(
        workspace_clearance_path,
        revision["registration"]["workspace_clearance"],
        blocker="planar_push_readiness_workspace_binding_invalid",
    )
    profile = _read_bound(
        droid_profile_path,
        droid_profile_reference,
        blocker="planar_push_readiness_droid_profile_binding_invalid",
    )
    placement = workspace.get("placement")
    if (
        task.get("schema_version")
        != "task_evaluation_rigid_relocation_template.v1"
        or task.get("task_identity") != revision["task_template"]["identity"]
        or task.get("object_identity") != revision["replacement"]["identity"]
        or task.get("strategy") != "planar_push"
        or workspace.get("schema_version")
        != "registered_sage_franka_placement_packet.v1"
        or workspace.get("packet_digest")
        != canonical_digest(workspace, digest_field="packet_digest")
        or not isinstance(placement, Mapping)
        or placement.get("candidate_may_self_authorize") is not False
        or placement.get("physical_execution_authorized") is not False
        or profile.get("schema_version") != "native_task_arena_packet_request.v1"
        or profile.get("request_digest")
        != canonical_digest(profile, digest_field="request_digest")
    ):
        raise TaskEvaluationPlanarPushReadinessCandidateError(
            "planar_push_readiness_source_contract_invalid"
        )
    start = _vector(
        task.get("start_center_xyz_m"),
        3,
        blocker="planar_push_readiness_task_direction_invalid",
    )
    target = _vector(
        task.get("target_center_xyz_m"),
        3,
        blocker="planar_push_readiness_task_direction_invalid",
    )
    direction = _normalize(
        [target[0] - start[0], target[1] - start[1], 0.0],
        blocker="planar_push_readiness_task_direction_invalid",
    )
    source_pose = _vector(
        placement.get("robot_pose_xyzyaw_collision_stage"),
        4,
        blocker="planar_push_readiness_source_base_invalid",
    )
    standoff = math.hypot(source_pose[0] - start[0], source_pose[1] - start[1])
    if standoff <= 0.05 or standoff > 1.155:
        raise TaskEvaluationPlanarPushReadinessCandidateError(
            "planar_push_readiness_source_base_invalid"
        )
    position = [
        start[0] - direction[0] * standoff,
        start[1] - direction[1] * standoff,
        source_pose[2],
    ]
    yaw = math.atan2(direction[1], direction[0])
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_pending_native_construction_readback",
        "scene_identity": revision["scene_identity"],
        "configured_scene_revision_digest": revision["revision_digest"],
        "robot_mount_interface_digest": revision["registration"][
            "robot_mount_interface"
        ]["digest"],
        "task_definition_digest": revision["task_template"]["definition"][
            "digest"
        ],
        "workspace_clearance_digest": revision["registration"][
            "workspace_clearance"
        ]["digest"],
        "derivation_method": (
            "reflect_reach_candidate_behind_start_along_frozen_planar_push"
        ),
        "source_reach_candidate_xyzyaw": source_pose,
        "horizontal_standoff_m": standoff,
        "pose_world": {
            "position_world_m": position,
            "orientation_xyzw": [
                0.0,
                0.0,
                math.sin(yaw / 2.0),
                math.cos(yaw / 2.0),
            ],
        },
        "task_direction_considered": True,
        "robot_base_qualified": False,
        "reachability_qualified": False,
        "collision_clearance_qualified": False,
        "learned_policy_outcomes_consulted": False,
        "native_construction_readback_completed": False,
        "base_pose_candidate_digest": "",
    }
    base["base_pose_candidate_digest"] = canonical_digest(
        base, digest_field="base_pose_candidate_digest"
    )
    rows = profile.get("cameras")
    by_role = {
        str(row.get("role") or ""): dict(row)
        for row in rows or []
        if isinstance(row, Mapping)
    }
    if set(by_role) != {"external", "wrist", "overview"}:
        raise TaskEvaluationPlanarPushReadinessCandidateError(
            "planar_push_readiness_droid_profile_invalid"
        )
    wrist = by_role["wrist"]
    intrinsics = by_role["external"].get("intrinsics")
    if (
        wrist.get("pose_frame") != "robot_body"
        or wrist.get("parent_prim_path")
        != "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
        or wrist.get("policy_input") is not True
        or wrist.get("scoring_input") is not False
        or not isinstance(intrinsics, Mapping)
        or intrinsics != wrist.get("intrinsics")
    ):
        raise TaskEvaluationPlanarPushReadinessCandidateError(
            "planar_push_readiness_droid_profile_invalid"
        )
    external_position = [position[0], position[1], position[2] + 1.35]
    lateral = [-direction[1], direction[0], 0.0]
    midpoint = [(start[index] + target[index]) / 2.0 for index in range(3)]
    overview_position = [
        midpoint[0] + 0.9 * lateral[0],
        midpoint[1] + 0.9 * lateral[1],
        max(position[2], start[2]) + 1.45,
    ]
    cameras = [
        _world_camera(
            "external",
            position=external_position,
            target=start,
            intrinsics=intrinsics,
        ),
        json.loads(json.dumps(wrist)),
        _world_camera(
            "overview",
            position=overview_position,
            target=midpoint,
            intrinsics=intrinsics,
        ),
    ]
    cameras[2]["policy_input"] = False
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_pending_native_construction_readback",
        "configured_scene_revision_digest": revision["revision_digest"],
        "base_pose_candidate": base,
        "cameras": cameras,
        "droid_profile_reference": dict(droid_profile_reference),
        "camera_configuration_qualified": False,
        "candidate_policy_queried": False,
        "native_construction_readback_required": True,
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(
        result, digest_field="receipt_digest"
    )
    destination = Path(output_path).expanduser()
    if destination.exists() or destination.is_symlink():
        raise TaskEvaluationPlanarPushReadinessCandidateError(
            "planar_push_readiness_destination_exists"
        )
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    destination.chmod(0o440)
    return json.loads(json.dumps(result))


def _mapping_file(path: str | Path, *, blocker: str) -> dict[str, Any]:
    source = Path(path).expanduser()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationPlanarPushReadinessCandidateError(blocker) from exc
    if source.is_symlink() or not source.is_file() or not isinstance(value, Mapping):
        raise TaskEvaluationPlanarPushReadinessCandidateError(blocker)
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    """Materialize the candidate through a production-callable entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--configured-revision", required=True)
    parser.add_argument("--task-definition", required=True)
    parser.add_argument("--workspace-clearance", required=True)
    parser.add_argument("--droid-profile", required=True)
    parser.add_argument("--droid-profile-reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    materialize_planar_push_readiness_candidate(
        configured_revision=_mapping_file(
            args.configured_revision,
            blocker="planar_push_readiness_configured_revision_invalid",
        ),
        task_definition_path=args.task_definition,
        workspace_clearance_path=args.workspace_clearance,
        droid_profile_path=args.droid_profile,
        droid_profile_reference=_mapping_file(
            args.droid_profile_reference,
            blocker="planar_push_readiness_droid_profile_reference_invalid",
        ),
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "TaskEvaluationPlanarPushReadinessCandidateError",
    "main",
    "materialize_planar_push_readiness_candidate",
]
