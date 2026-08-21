"""Derive a front-facing Franka stance from the task affordance geometry."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_packet import REQUEST_SCHEMA_VERSION


FRANKA_ROBOTIQ_READY_RESET = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "finger_joint": 0.0,
    "right_outer_knuckle_joint": 0.0,
    "right_inner_finger_joint": 0.0,
    "right_inner_finger_knuckle_joint": 0.0,
    "left_inner_finger_knuckle_joint": 0.0,
    "left_inner_finger_joint": 0.0,
}
RESET_SOURCE = (
    "isaac-sim/IsaacLab:isaaclab_assets/robots/franka.py:"
    "FRANKA_ROBOTIQ_GRIPPER_CFG@ffff603eafc6b74264a5261cc0183d6a65390d78"
)
RETREAT_STRATEGY_ID = "world_up_when_reverse_approach_enters_base_dead_zone_v1"


class NativeTaskArenaStanceVariantError(ValueError):
    """Stable failures before a corrected stance reaches packet materialization."""


def _normalize(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if len(values) != 3 or norm <= 1.0e-9 or not math.isfinite(norm):
        raise NativeTaskArenaStanceVariantError(
            "native_task_arena_stance_vector_invalid"
        )
    return [value / norm for value in values]


def _rotate_xyzw(quaternion: Sequence[float], vector: Sequence[float]) -> list[float]:
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if abs(norm - 1.0) > 1.0e-5:
        raise NativeTaskArenaStanceVariantError(
            "native_task_arena_stance_task_orientation_invalid"
        )
    vx, vy, vz = (float(value) for value in vector)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def _look_at(position: Sequence[float], target: Sequence[float]) -> list[float]:
    forward = _normalize(
        [float(target[index]) - float(position[index]) for index in range(3)]
    )
    right = _normalize([forward[1], -forward[0], 0.0])
    down = _normalize(
        [
            forward[1] * right[2] - forward[2] * right[1],
            forward[2] * right[0] - forward[0] * right[2],
            forward[0] * right[1] - forward[1] * right[0],
        ]
    )
    return [
        right[0],
        down[0],
        forward[0],
        float(position[0]),
        right[1],
        down[1],
        forward[1],
        float(position[1]),
        right[2],
        down[2],
        forward[2],
        float(position[2]),
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def _world_point(
    root_position: Sequence[float],
    root_orientation: Sequence[float],
    local_position: Sequence[float],
) -> list[float]:
    rotated = _rotate_xyzw(root_orientation, local_position)
    return [float(root_position[index]) + rotated[index] for index in range(3)]


def materialize_native_task_arena_stance_variant_request(
    *,
    base_request_path: str | Path,
    output_path: str | Path,
    door_standoff_m: float = 0.55,
) -> dict[str, Any]:
    """Place the fixed-base Franka in front of the exact door contact sweep."""

    try:
        request = json.loads(
            Path(base_request_path).expanduser().read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeTaskArenaStanceVariantError(
            "native_task_arena_stance_request_invalid"
        ) from exc
    if (
        not isinstance(request, dict)
        or request.get("schema_version") != REQUEST_SCHEMA_VERSION
        or request.get("request_digest")
        != canonical_digest(request, digest_field="request_digest")
    ):
        raise NativeTaskArenaStanceVariantError(
            "native_task_arena_stance_request_invalid"
        )
    standoff = float(door_standoff_m)
    if not math.isfinite(standoff) or not 0.45 <= standoff <= 0.75:
        raise NativeTaskArenaStanceVariantError(
            "native_task_arena_stance_standoff_invalid"
        )

    task_spec = request.get("task_spec") or {}
    affordance = task_spec.get("interaction_affordance") or {}
    asset_id = str(
        task_spec.get("subject_asset_id") or task_spec.get("asset_id") or ""
    )
    assets = [row for row in request.get("assets") or [] if isinstance(row, Mapping)]
    task_assets = [row for row in assets if str(row.get("asset_id") or "") == asset_id]
    path = affordance.get("joint_contact_path") or []
    approach_local = affordance.get("approach_unit_asset_root")
    if len(task_assets) != 1 or not isinstance(path, list) or len(path) < 2:
        raise NativeTaskArenaStanceVariantError(
            "native_task_arena_stance_affordance_invalid"
        )
    task_asset = task_assets[0]
    task_pose = task_asset.get("pose_world") or {}
    root_position = task_pose.get("position_world_m")
    root_orientation = task_pose.get("orientation_xyzw")
    try:
        phase_targets = [
            _world_point(
                root_position,
                root_orientation,
                row["contact_pose_asset_root"]["position_m"],
            )
            for row in path
        ]
        approach_world = _normalize(
            _rotate_xyzw(root_orientation, approach_local)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeTaskArenaStanceVariantError(
            "native_task_arena_stance_affordance_invalid"
        ) from exc
    if abs(approach_world[2]) > 0.05:
        raise NativeTaskArenaStanceVariantError(
            "native_task_arena_stance_front_normal_not_horizontal"
        )
    closed = phase_targets[0]
    old_base = request.get("robot_base_pose_world") or {}
    base_z = float((old_base.get("position_world_m") or [0.0, 0.0, 0.0])[2])
    base = [
        closed[0] + standoff * approach_world[0],
        closed[1] + standoff * approach_world[1],
        base_z,
    ]

    # A retreat is task geometry, not automatically the negated approach.
    # r38 proved the washer's authored -Y retreat moved the TCP *toward* the
    # Franka base: joint 4 and joint 7 reached their lower limits, while every
    # earlier construction phase and recovery passed. MoveIt's Franka examples
    # likewise choose post-release retreat in an explicit world direction,
    # including world-up. Preserve the authored direction when it increases
    # horizontal base clearance; otherwise lift in world Z so the retreat adds
    # clearance instead of folding the arm into its base dead zone.
    try:
        authored_retreat_asset = _normalize(
            affordance["retreat_unit_asset_root"]
        )
        retreat_clearance = float(affordance["retreat_clearance_m"])
        if not math.isfinite(retreat_clearance) or retreat_clearance <= 0.0:
            raise ValueError("retreat clearance")
        authored_retreat_world = _normalize(
            _rotate_xyzw(root_orientation, authored_retreat_asset)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeTaskArenaStanceVariantError(
            "native_task_arena_stance_retreat_invalid"
        ) from exc
    final_contact = phase_targets[-1]
    authored_retreat_target = [
        final_contact[index]
        + authored_retreat_world[index] * retreat_clearance
        for index in range(3)
    ]
    final_horizontal_base_clearance = math.dist(final_contact[:2], base[:2])
    authored_horizontal_base_clearance = math.dist(
        authored_retreat_target[:2], base[:2]
    )
    retreat_enters_base_dead_zone = (
        authored_horizontal_base_clearance + 1.0e-6
        < final_horizontal_base_clearance
    )
    if retreat_enters_base_dead_zone:
        qx, qy, qz, qw = (float(value) for value in root_orientation)
        resolved_retreat_asset = _normalize(
            _rotate_xyzw((-qx, -qy, -qz, qw), (0.0, 0.0, 1.0))
        )
        resolved_retreat_world = [0.0, 0.0, 1.0]
    else:
        resolved_retreat_asset = authored_retreat_asset
        resolved_retreat_world = authored_retreat_world
    affordance["retreat_unit_asset_root"] = resolved_retreat_asset
    affordance["affordance_digest"] = canonical_digest(
        affordance, digest_field="affordance_digest"
    )
    request["task_spec"] = task_spec
    into_door = [-approach_world[0], -approach_world[1], 0.0]
    yaw = math.atan2(into_door[1], into_door[0])
    phase_bearings = [
        math.atan2(target[1] - base[1], target[0] - base[0])
        for target in phase_targets
    ]
    deviations = [
        abs(math.atan2(math.sin(bearing - yaw), math.cos(bearing - yaw)))
        for bearing in phase_bearings
    ]
    maximum_deviation = max(deviations)
    if maximum_deviation > math.pi / 4.0:
        raise NativeTaskArenaStanceVariantError(
            "native_task_arena_stance_door_sweep_outside_front_arc"
        )

    request["robot_base_pose_world"] = {
        "position_world_m": base,
        "orientation_xyzw": [
            0.0,
            0.0,
            math.sin(yaw / 2.0),
            math.cos(yaw / 2.0),
        ],
    }
    request["robot_joint_reset_positions_rad"] = dict(
        FRANKA_ROBOTIQ_READY_RESET
    )

    cameras = [dict(row) for row in request.get("cameras") or []]
    by_role = {str(row.get("role") or ""): row for row in cameras}
    if set(by_role) != {"external", "wrist", "overview"}:
        raise NativeTaskArenaStanceVariantError(
            "native_task_arena_stance_camera_roles_invalid"
        )
    external_eye = [closed[0] + 0.75, closed[1] - 1.0, closed[2] + 0.85]
    external_target = [closed[0], closed[1] - 0.2, closed[2] + 0.095]
    overview_eye = [closed[0] - 1.1, closed[1] - 1.3, closed[2] + 1.245]
    overview_target = [closed[0], closed[1] - 0.275, closed[2] - 0.005]
    by_role["external"]["frame_from_camera_matrix"] = _look_at(
        external_eye, external_target
    )
    by_role["overview"]["frame_from_camera_matrix"] = _look_at(
        overview_eye, overview_target
    )
    request["cameras"] = [by_role[role] for role in ("external", "wrist", "overview")]
    request["stance_variant"] = {
        "base_request_digest": request["request_digest"],
        "derivation": "door_contact_point_plus_outward_normal_standoff",
        "door_standoff_m": standoff,
        "closed_contact_world_m": closed,
        "approach_outward_world": approach_world,
        "resolved_base_position_world_m": base,
        "resolved_base_yaw_world_rad": yaw,
        "maximum_door_sweep_bearing_deviation_rad": maximum_deviation,
        "full_sweep_within_front_quarter_sphere": True,
        "retreat_strategy_id": RETREAT_STRATEGY_ID,
        "authored_retreat_unit_asset_root": authored_retreat_asset,
        "resolved_retreat_unit_asset_root": resolved_retreat_asset,
        "resolved_retreat_unit_world": resolved_retreat_world,
        "authored_retreat_target_world_m": authored_retreat_target,
        "final_contact_horizontal_base_clearance_m": (
            final_horizontal_base_clearance
        ),
        "authored_retreat_horizontal_base_clearance_m": (
            authored_horizontal_base_clearance
        ),
        "authored_retreat_enters_base_dead_zone": (
            retreat_enters_base_dead_zone
        ),
        "reset_source": RESET_SOURCE,
        "external_camera_source": "front_side_zero_roll_vision_geometry_candidate",
        "overview_camera_source": "front_side_zero_roll_vision_geometry_candidate",
        "native_ik_qualified": False,
        "native_collision_clearance_qualified": False,
        "native_reset_task_space_readback_qualified": False,
    }
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise NativeTaskArenaStanceVariantError(
            "native_task_arena_stance_destination_exists"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return json.loads(json.dumps(request))


__all__ = [
    "FRANKA_ROBOTIQ_READY_RESET",
    "NativeTaskArenaStanceVariantError",
    "RETREAT_STRATEGY_ID",
    "materialize_native_task_arena_stance_variant_request",
]
