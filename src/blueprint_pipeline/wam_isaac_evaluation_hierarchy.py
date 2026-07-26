"""WAM-primary evaluation hierarchy and Isaac contradiction evidence.

The WAM/policy lane owns rollout state and its primary score. Isaac may run in
parallel, but its measurements are diagnostic: they can cap downstream claims
when contradictory or unavailable, never replace the WAM observation stream.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import utc_now_iso, write_json
from .gear_sonic_joint_order_contract import (
    JOINT_ORDER_SCHEMA_VERSION,
    PROTOCOL_V4_BODY_JOINT_NAMES,
    PROTOCOL_V4_FULL_JOINT_ORDER,
    PROTOCOL_V4_LEFT_HAND_JOINT_NAMES,
    PROTOCOL_V4_MAPPING_DIGEST,
    PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES,
)

WAM_PRIMARY_AUTHORITY = "wam_primary_isaac_diagnostic"
LEGACY_ISAAC_AUTHORITY = "legacy_isaac_authoritative"
SUPPORTED_EVALUATION_AUTHORITIES = frozenset(
    {WAM_PRIMARY_AUTHORITY, LEGACY_ISAAC_AUTHORITY}
)
POST_ACTION_POLICY_STATE_SOURCE = "post_action_live_isaac_articulation"
LIVE_ISAAC_STATE_SNAPSHOT_ENV = "BLUEPRINT_GEAR_SONIC_ISAAC_STATE_SNAPSHOT"
LIVE_ISAAC_STATE_SNAPSHOT_DEFAULT_PATH = (
    "/workspace/closed_loop_out/gear_sonic_isaac_state_snapshot.json"
)
FK_STATE_SEED_ACTION_KEY = "controller_fk_state_seed"
UNITREE_G1_SONIC_STATE_DIMS = {
    "left_leg": 6,
    "right_leg": 6,
    "waist": 3,
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
    "projected_gravity": 3,
}
UNSAFE_STANCE_MAX_HORIZONTAL_PROJECTED_GRAVITY = 0.5
UNSAFE_STANCE_MIN_UPRIGHT_PROJECTED_GRAVITY_Z = -0.7
MANIPULATION_EFFECTOR_PROGRESS_MINIMUM_M = 0.015
MANIPULATION_EFFECTOR_PROJECTED_MOTION_MINIMUM_PX = 8.0
MANIPULATION_EFFECTOR_PROGRESS_REFERENCE_FRAME_COUNT = 16
MANIPULATION_EFFECTOR_PROGRESS_FLOOR_M = 0.005
MANIPULATION_EFFECTOR_PROJECTED_MOTION_FLOOR_PX = 2.0


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value) if value is not None else ""


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_numeric_sequence(value: Any, *, minimum_length: int = 1) -> bool:
    return bool(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) >= minimum_length
        and all(_finite_float(item) is not None for item in value)
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = _string(value).strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def emit_closed_loop_live_progress(
    output_dir: Path, *, stage: str, status: str, step_index: int | None = None
) -> None:
    payload: dict[str, Any] = {
        "schema_version": "oscar_isaac_closed_loop_live_progress.v1",
        "observed_at": utc_now_iso(),
        "stage": str(stage),
        "status": str(status),
        "raw_secret_values_recorded": False,
    }
    if step_index is not None:
        payload["step_index"] = int(step_index)
    write_json(output_dir / "closed_loop_live_progress.json", payload)
    print(
        "blueprint_closed_loop_progress="
        + json.dumps(payload, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def validated_post_action_policy_state(
    value: Any,
    *,
    simulator_session_id: str,
    stage_id: str,
    source_action_sha256: str,
    source_step_index: int,
) -> dict[str, Any]:
    state = _mapping(value)
    if not state:
        raise RuntimeError("post_action_policy_state_missing_or_not_object")
    normalized = dict(state)
    for field, dimension in UNITREE_G1_SONIC_STATE_DIMS.items():
        values = state.get(field)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise RuntimeError(f"post_action_policy_state_{field}_not_sequence")
        if len(values) != dimension:
            raise RuntimeError(
                f"post_action_policy_state_{field}_dimension_{len(values)}_expected_{dimension}"
            )
        finite_values = [_finite_float(item) for item in values]
        if any(item is None for item in finite_values):
            raise RuntimeError(f"post_action_policy_state_{field}_nonfinite")
        normalized[field] = [float(item) for item in finite_values if item is not None]
    measurement = _mapping(state.get("measurement"))
    if not measurement:
        raise RuntimeError("post_action_policy_state_measurement_missing_or_not_object")
    if measurement.get("surrogate") is not False:
        raise RuntimeError("post_action_policy_state_surrogate_not_false")
    if _string(measurement.get("source")).strip() != POST_ACTION_POLICY_STATE_SOURCE:
        raise RuntimeError("post_action_policy_state_source_not_live_post_action_isaac")
    expected_session_id = _string(simulator_session_id).strip()
    expected_stage_id = _string(stage_id).strip()
    expected_action_sha256 = _string(source_action_sha256).strip().lower()
    if not expected_session_id:
        raise RuntimeError("post_action_policy_state_expected_simulator_session_id_missing")
    if not expected_stage_id:
        raise RuntimeError("post_action_policy_state_expected_stage_id_missing")
    if not _is_sha256(expected_action_sha256):
        raise RuntimeError("post_action_policy_state_expected_action_sha256_invalid")
    if _string(measurement.get("simulator_session_id")).strip() != expected_session_id:
        raise RuntimeError("post_action_policy_state_simulator_session_id_mismatch")
    if _string(measurement.get("stage_id")).strip() != expected_stage_id:
        raise RuntimeError("post_action_policy_state_stage_id_mismatch")
    if _string(measurement.get("source_action_sha256")).strip().lower() != expected_action_sha256:
        raise RuntimeError("post_action_policy_state_source_action_sha256_mismatch")
    observed_step = measurement.get("source_step_index")
    if isinstance(observed_step, bool) or not isinstance(observed_step, int) or observed_step != int(source_step_index):
        raise RuntimeError("post_action_policy_state_source_step_index_mismatch")
    captured_at_ns = measurement.get("captured_at_ns")
    if isinstance(captured_at_ns, bool):
        raise RuntimeError("post_action_policy_state_captured_at_ns_invalid")
    try:
        captured_at_ns_int = int(captured_at_ns)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("post_action_policy_state_captured_at_ns_invalid") from exc
    if captured_at_ns_int <= 0:
        raise RuntimeError("post_action_policy_state_captured_at_ns_invalid")
    normalized["measurement"] = dict(measurement)
    return normalized


def validated_live_isaac_fk_seed(
    value: Any,
    *,
    source_step_index: int | None = None,
    source_action_sha256: str = "",
    require_fresh: bool = False,
) -> dict[str, Any]:
    seed = _mapping(value)
    if seed.get("schema_version") != "gear_sonic_isaac_state_snapshot.v1":
        raise RuntimeError("live_fk_seed_schema_invalid")
    if (
        seed.get("status") != "live"
        or seed.get("surrogate") is not False
        or seed.get("ready_for_native_dds_bridge") is not True
        or _string(seed.get("source")).strip() != "live_isaac_articulation"
    ):
        raise RuntimeError("live_fk_seed_not_live_isaac")
    if seed.get("joint_order_schema_version") != JOINT_ORDER_SCHEMA_VERSION:
        raise RuntimeError("live_fk_seed_joint_order_schema_invalid")
    if _string(seed.get("mapping_digest")).strip() != PROTOCOL_V4_MAPPING_DIGEST:
        raise RuntimeError("live_fk_seed_mapping_digest_mismatch")
    normalized = dict(seed)
    for field, expected in (
        ("body_q", PROTOCOL_V4_BODY_JOINT_NAMES),
        ("left_hand_q", PROTOCOL_V4_LEFT_HAND_JOINT_NAMES),
        ("right_hand_q", PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES),
        ("base_quaternion_wxyz", ("w", "x", "y", "z")),
    ):
        values = seed.get(field)
        if not _finite_numeric_sequence(values, minimum_length=len(expected)) or len(values) != len(expected):
            raise RuntimeError(f"live_fk_seed_{field}_invalid")
        normalized[field] = [float(item) for item in values]
    for field, expected in (
        ("body_joint_names", PROTOCOL_V4_BODY_JOINT_NAMES),
        ("left_hand_joint_names", PROTOCOL_V4_LEFT_HAND_JOINT_NAMES),
        ("right_hand_joint_names", PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES),
    ):
        if list(seed.get(field) or []) != list(expected):
            raise RuntimeError(f"live_fk_seed_{field}_invalid")
    full_names = _mapping(seed.get("name_order_metadata")).get("protocol_v4_full_joint_names")
    if list(full_names or []) != list(PROTOCOL_V4_FULL_JOINT_ORDER):
        raise RuntimeError("live_fk_seed_full_joint_order_invalid")
    payload_sha256 = _string(seed.get("payload_sha256")).strip().lower()
    payload = {key: item for key, item in seed.items() if key != "payload_sha256"}
    if not _is_sha256(payload_sha256) or payload_sha256 != _canonical_sha256(payload):
        raise RuntimeError("live_fk_seed_payload_sha256_mismatch")
    if not _string(seed.get("simulator_session_id")).strip() or not _string(seed.get("stage_id")).strip():
        raise RuntimeError("live_fk_seed_session_identity_missing")
    if source_step_index is not None:
        if seed.get("source_step_index") != int(source_step_index):
            raise RuntimeError("live_fk_seed_source_step_index_mismatch")
        expected_action = _string(source_action_sha256).strip().lower()
        if not _is_sha256(expected_action) or _string(seed.get("source_action_sha256")).strip().lower() != expected_action:
            raise RuntimeError("live_fk_seed_source_action_sha256_mismatch")
    if require_fresh:
        fresh_until = seed.get("fresh_until_ns")
        if isinstance(fresh_until, bool) or not isinstance(fresh_until, int):
            raise RuntimeError("live_fk_seed_freshness_invalid")
        if time.time_ns() > fresh_until:
            raise RuntimeError("live_fk_seed_initial_snapshot_stale")
    normalized["payload_sha256"] = payload_sha256
    return normalized


def load_initial_live_isaac_fk_seed() -> dict[str, Any]:
    path = Path(
        _string(os.environ.get(LIVE_ISAAC_STATE_SNAPSHOT_ENV)).strip()
        or LIVE_ISAAC_STATE_SNAPSHOT_DEFAULT_PATH
    ).expanduser()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("live_fk_seed_initial_snapshot_missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("live_fk_seed_initial_snapshot_invalid") from exc
    return validated_live_isaac_fk_seed(payload, require_fresh=True)


def controller_fk_prediction_seed(
    generated_state: Mapping[str, Any], *, source_step_index: int, source_action_sha256: str
) -> dict[str, Any]:
    state = _mapping(generated_state)
    positions = state.get("joint_positions")
    if (
        state.get("proxy_or_surrogate") is not False
        or not _finite_numeric_sequence(positions, minimum_length=len(PROTOCOL_V4_FULL_JOINT_ORDER))
        or len(positions) != len(PROTOCOL_V4_FULL_JOINT_ORDER)
        or list(state.get("joint_names") or []) != list(PROTOCOL_V4_FULL_JOINT_ORDER)
        or state.get("joint_order_schema_version") != JOINT_ORDER_SCHEMA_VERSION
        or _string(state.get("mapping_digest")).strip() != PROTOCOL_V4_MAPPING_DIGEST
    ):
        raise RuntimeError("controller_fk_prediction_seed_state_invalid")
    expected_action = _string(source_action_sha256).strip().lower()
    if not _is_sha256(expected_action) or _string(state.get("source_action_sha256")).strip().lower() != expected_action:
        raise RuntimeError("controller_fk_prediction_seed_action_identity_mismatch")
    quaternion = _mapping(state.get("proprioceptive_state")).get("base_quat_measured")
    if not _finite_numeric_sequence(quaternion, minimum_length=4) or len(quaternion) != 4:
        raise RuntimeError("controller_fk_prediction_seed_quaternion_invalid")
    q = [float(item) for item in positions]
    payload = {
        "schema_version": "controller_fk_state_seed.v1",
        "status": "predicted",
        "surrogate": False,
        "seed_authority": "wam_controller_fk_prediction",
        "source": "controller_fk_prediction",
        "joint_order_schema_version": JOINT_ORDER_SCHEMA_VERSION,
        "mapping_digest": PROTOCOL_V4_MAPPING_DIGEST,
        "body_joint_names": list(PROTOCOL_V4_BODY_JOINT_NAMES),
        "body_q": q[: len(PROTOCOL_V4_BODY_JOINT_NAMES)],
        "left_hand_joint_names": list(PROTOCOL_V4_LEFT_HAND_JOINT_NAMES),
        "left_hand_q": q[29:36],
        "right_hand_joint_names": list(PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES),
        "right_hand_q": q[36:43],
        "base_quaternion_wxyz": [float(item) for item in quaternion],
        "name_order_metadata": {"protocol_v4_full_joint_names": list(PROTOCOL_V4_FULL_JOINT_ORDER)},
        "source_step_index": int(source_step_index),
        "source_action_sha256": expected_action,
        "source_generated_robot_state_sha256": _canonical_sha256(state),
    }
    payload["payload_sha256"] = _canonical_sha256(payload)
    return payload


def validated_controller_fk_state_seed(value: Any) -> dict[str, Any]:
    seed = _mapping(value)
    if seed.get("schema_version") == "gear_sonic_isaac_state_snapshot.v1":
        return validated_live_isaac_fk_seed(seed)
    if (
        seed.get("schema_version") != "controller_fk_state_seed.v1"
        or seed.get("status") != "predicted"
        or seed.get("surrogate") is not False
        or seed.get("seed_authority") != "wam_controller_fk_prediction"
        or seed.get("source") != "controller_fk_prediction"
    ):
        raise RuntimeError("controller_fk_state_seed_authority_invalid")
    if seed.get("joint_order_schema_version") != JOINT_ORDER_SCHEMA_VERSION:
        raise RuntimeError("controller_fk_state_seed_joint_order_schema_invalid")
    if _string(seed.get("mapping_digest")).strip() != PROTOCOL_V4_MAPPING_DIGEST:
        raise RuntimeError("controller_fk_state_seed_mapping_digest_mismatch")
    for field, expected in (
        ("body_q", PROTOCOL_V4_BODY_JOINT_NAMES),
        ("left_hand_q", PROTOCOL_V4_LEFT_HAND_JOINT_NAMES),
        ("right_hand_q", PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES),
        ("base_quaternion_wxyz", ("w", "x", "y", "z")),
    ):
        values = seed.get(field)
        if not _finite_numeric_sequence(values, minimum_length=len(expected)) or len(values) != len(expected):
            raise RuntimeError(f"controller_fk_state_seed_{field}_invalid")
    for field, expected in (
        ("body_joint_names", PROTOCOL_V4_BODY_JOINT_NAMES),
        ("left_hand_joint_names", PROTOCOL_V4_LEFT_HAND_JOINT_NAMES),
        ("right_hand_joint_names", PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES),
    ):
        if list(seed.get(field) or []) != list(expected):
            raise RuntimeError(f"controller_fk_state_seed_{field}_invalid")
    if list(_mapping(seed.get("name_order_metadata")).get("protocol_v4_full_joint_names") or []) != list(PROTOCOL_V4_FULL_JOINT_ORDER):
        raise RuntimeError("controller_fk_state_seed_full_joint_order_invalid")
    payload_sha256 = _string(seed.get("payload_sha256")).strip().lower()
    payload = {key: item for key, item in seed.items() if key != "payload_sha256"}
    if not _is_sha256(payload_sha256) or payload_sha256 != _canonical_sha256(payload):
        raise RuntimeError("controller_fk_state_seed_payload_sha256_mismatch")
    return dict(seed)


def wam_isaac_prefix_prediction_error(
    *, wam_output: Mapping[str, Any], live_seed: Mapping[str, Any]
) -> dict[str, Any]:
    generated = _mapping(wam_output.get("generated_robot_state"))
    sequence = generated.get("controller_fk_sequence")
    if not isinstance(sequence, Sequence) or isinstance(sequence, (str, bytes, Mapping)):
        return {"status": "not_measured", "blockers": ["wam_fk_sequence_missing"]}
    frames = [dict(row) for row in sequence if isinstance(row, Mapping)]
    if not frames:
        return {"status": "not_measured", "blockers": ["wam_fk_sequence_missing"]}
    predicted = frames[-1].get("joint_positions")
    actual = [
        *list(live_seed.get("body_q") or []),
        *list(live_seed.get("left_hand_q") or []),
        *list(live_seed.get("right_hand_q") or []),
    ]
    expected = len(PROTOCOL_V4_FULL_JOINT_ORDER)
    if not _finite_numeric_sequence(predicted, minimum_length=expected) or len(predicted) != expected or not _finite_numeric_sequence(actual, minimum_length=expected) or len(actual) != expected:
        return {"status": "not_measured", "blockers": ["wam_isaac_joint_vector_invalid"]}
    errors = [float(left) - float(right) for left, right in zip(predicted, actual, strict=True)]
    return {
        "schema_version": "wam_isaac_prefix_prediction_error.v1",
        "status": "measured",
        "joint_count": len(errors),
        "mean_absolute_error_rad": sum(abs(value) for value in errors) / len(errors),
        "root_mean_square_error_rad": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "maximum_absolute_error_rad": max(abs(value) for value in errors),
        "predicted_final_joint_positions_sha256": _canonical_sha256(list(predicted)),
        "actual_final_joint_positions_sha256": _canonical_sha256(actual),
        "actual_live_state_payload_sha256": live_seed.get("payload_sha256"),
        "threshold_status": "not_calibrated_numeric_error_is_advisory",
        "claim_boundary": "Joint error is diagnostic and does not prove task success.",
    }


def post_action_stance_report(state: Mapping[str, Any]) -> dict[str, Any]:
    gravity = state.get("projected_gravity")
    if not _finite_numeric_sequence(gravity, minimum_length=3) or len(gravity) != 3:
        return {
            "schema_version": "post_action_stance_report.v1",
            "status": "invalid",
            "unsafe_stance_detected": True,
            "blockers": ["post_action_projected_gravity_missing_or_invalid"],
        }
    gx, gy, gz = (float(value) for value in gravity)
    unsafe = bool(
        abs(gx) > UNSAFE_STANCE_MAX_HORIZONTAL_PROJECTED_GRAVITY
        or abs(gy) > UNSAFE_STANCE_MAX_HORIZONTAL_PROJECTED_GRAVITY
        or gz > UNSAFE_STANCE_MIN_UPRIGHT_PROJECTED_GRAVITY_Z
    )
    return {
        "schema_version": "post_action_stance_report.v1",
        "status": "unsafe" if unsafe else "upright",
        "unsafe_stance_detected": unsafe,
        "projected_gravity": [gx, gy, gz],
        "thresholds": {
            "maximum_absolute_horizontal_component": UNSAFE_STANCE_MAX_HORIZONTAL_PROJECTED_GRAVITY,
            "maximum_z_for_upright": UNSAFE_STANCE_MIN_UPRIGHT_PROJECTED_GRAVITY_Z,
        },
        "blockers": ["unsafe_post_action_robot_stance"] if unsafe else [],
    }


def task_progress_report(
    completion_result: Mapping[str, Any], *, minimum_progress_fraction: float
) -> dict[str, Any]:
    comparison = _string(completion_result.get("comparison")).strip()
    initial = _finite_float(completion_result.get("episode_initial_value"))
    current = _finite_float(completion_result.get("after_value"))
    tolerance = _finite_float(completion_result.get("tolerance"))
    target = _finite_float(completion_result.get("target_value"))
    fraction = float(minimum_progress_fraction)
    if comparison not in {"increase_at_least", "decrease_at_least", "absolute_change_at_least", "within_tolerance", "at_or_above", "at_or_below"} or initial is None or current is None or tolerance is None or not math.isfinite(fraction) or fraction <= 0.0:
        return {"schema_version": "online_task_progress_report.v1", "status": "unavailable", "resource_control_only": True, "blockers": ["online_task_progress_measurement_unavailable"]}
    if comparison in {"increase_at_least", "at_or_above"}:
        progress = current - initial
    elif comparison in {"decrease_at_least", "at_or_below"}:
        progress = initial - current
    elif comparison == "absolute_change_at_least":
        progress = abs(current - initial)
    elif target is not None:
        progress = abs(initial - target) - abs(current - target)
    else:
        return {"schema_version": "online_task_progress_report.v1", "status": "unavailable", "resource_control_only": True, "blockers": ["online_task_progress_target_value_unavailable"]}
    return {
        "schema_version": "online_task_progress_report.v1",
        "status": "measured",
        "resource_control_only": True,
        "criterion_id": completion_result.get("criterion_id"),
        "comparison": comparison,
        "episode_initial_value": initial,
        "current_value": current,
        "target_value": target,
        "success_tolerance": tolerance,
        "progress_toward_criterion": float(progress),
        "minimum_meaningful_progress_delta": max(1e-6, abs(tolerance) * fraction),
        "registered_transition_passed": completion_result.get("registered_transition_passed") is True,
        "blockers": [],
        "claim_boundary": "This resource-control signal cannot prove task success.",
    }


def manipulation_effector_progress_report(
    projection: Mapping[str, Any],
    *,
    minimum_progress_m: float = MANIPULATION_EFFECTOR_PROGRESS_MINIMUM_M,
    minimum_projected_motion_px: float = MANIPULATION_EFFECTOR_PROJECTED_MOTION_MINIMUM_PX,
) -> dict[str, Any]:
    target = projection.get("task_target_world_xyz_m")
    if not _finite_numeric_sequence(target, minimum_length=3) or len(target) != 3:
        return {"schema_version": "manipulation_effector_progress_report.v1", "status": "blocked", "capability_gate_passed": False, "blockers": ["manipulation_task_target_world_xyz_missing_or_invalid"]}
    target_xyz = tuple(float(value) for value in target)
    sequence = projection.get("controller_fk_sequence")
    if not isinstance(sequence, Sequence) or isinstance(sequence, (str, bytes, bytearray)):
        sequence = _mapping(projection.get("generated_robot_state")).get("controller_fk_sequence")
    if not isinstance(sequence, Sequence) or isinstance(sequence, (str, bytes, bytearray)):
        sequence = []
    distances: dict[str, list[float]] = {}
    positions: dict[str, list[tuple[float, float, float]]] = {}
    projected: dict[str, list[tuple[float, float]]] = {}
    in_frame: dict[str, int] = {}
    for frame in sequence:
        landmarks = _mapping(frame).get("landmarks")
        if not isinstance(landmarks, Sequence) or isinstance(landmarks, (str, bytes, bytearray)):
            continue
        observed: set[str] = set()
        for landmark in landmarks:
            row = _mapping(landmark)
            name = _string(row.get("name") or row.get("landmark_id")).strip().lower()
            world_xyz = row.get("world_xyz") or row.get("world_xyz_m")
            if not name or ("wrist" not in name and "hand" not in name) or name in observed or not _finite_numeric_sequence(world_xyz, minimum_length=3) or len(world_xyz) != 3:
                continue
            observed.add(name)
            xyz = tuple(float(value) for value in world_xyz)
            distances.setdefault(name, []).append(math.dist(xyz, target_xyz))
            positions.setdefault(name, []).append(xyz)
            image_projection = _mapping(row.get("image_projection"))
            u_px, v_px = _finite_float(image_projection.get("u_px")), _finite_float(image_projection.get("v_px"))
            if u_px is not None and v_px is not None:
                projected.setdefault(name, []).append((u_px, v_px))
                if image_projection.get("available") is True:
                    in_frame[name] = in_frame.get(name, 0) + 1
    rows: list[dict[str, Any]] = []
    for name, values in sorted(distances.items()):
        if len(values) < 2:
            continue
        xyzs, pixels = positions[name], projected.get(name, [])
        rows.append({
            "effector": name,
            "frame_count": len(values),
            "first_distance_m": round(values[0], 9),
            "minimum_distance_m": round(min(values), 9),
            "final_distance_m": round(values[-1], 9),
            "maximum_progress_toward_target_m": round(max(0.0, values[0] - min(values)), 9),
            "maximum_displacement_from_first_frame_m": round(max(math.dist(xyzs[0], xyz) for xyz in xyzs), 9),
            "projected_frame_count": len(pixels),
            "in_frame_projection_count": in_frame.get(name, 0),
            "maximum_projected_displacement_from_first_frame_px": round(max((math.dist(pixels[0], pixel) for pixel in pixels), default=0.0), 6),
        })
    observed_frame_count = max(
        (int(row.get("frame_count") or 0) for row in rows),
        default=0,
    )
    duration_scale = min(
        1.0,
        max(
            0.0,
            float(observed_frame_count)
            / float(MANIPULATION_EFFECTOR_PROGRESS_REFERENCE_FRAME_COUNT),
        ),
    )
    required_progress_m = max(
        MANIPULATION_EFFECTOR_PROGRESS_FLOOR_M,
        float(minimum_progress_m) * duration_scale,
    )
    required_projected_motion_px = max(
        MANIPULATION_EFFECTOR_PROJECTED_MOTION_FLOOR_PX,
        float(minimum_projected_motion_px) * duration_scale,
    )
    best_progress = max((float(row["maximum_progress_toward_target_m"]) for row in rows), default=0.0)
    best_displacement = max((float(row["maximum_displacement_from_first_frame_m"]) for row in rows), default=0.0)
    best_projected = max((float(row["maximum_projected_displacement_from_first_frame_px"]) for row in rows if int(row["in_frame_projection_count"]) >= 2), default=0.0)
    directional = bool(rows and best_progress >= required_progress_m)
    motion = bool(rows and best_displacement >= required_progress_m)
    visible = bool(rows and best_projected >= required_projected_motion_px)
    blockers = []
    if not directional:
        blockers.append("manipulation_controller_fk_no_directional_effector_progress")
    if not motion:
        blockers.append("manipulation_controller_fk_no_meaningful_effector_motion")
    if not visible:
        blockers.append("manipulation_controller_fk_no_visible_projected_effector_motion")
    return {
        "schema_version": "manipulation_effector_progress_report.v1",
        "status": "passed" if directional and motion and visible else "blocked",
        "capability_gate_passed": directional and motion and visible,
        "task_target_world_xyz_m": list(target_xyz),
        "reference_frame_count": MANIPULATION_EFFECTOR_PROGRESS_REFERENCE_FRAME_COUNT,
        "observed_frame_count": observed_frame_count,
        "duration_scale": round(duration_scale, 6),
        "unscaled_minimum_progress_m": float(minimum_progress_m),
        "unscaled_minimum_projected_motion_px": float(minimum_projected_motion_px),
        "minimum_required_progress_m": required_progress_m,
        "minimum_required_projected_motion_px": required_projected_motion_px,
        "best_progress_toward_target_m": round(best_progress, 9),
        "best_effector_displacement_m": round(best_displacement, 9),
        "best_visible_projected_effector_displacement_px": round(best_projected, 6),
        "directional_progress_passed": directional,
        "motion_capability_passed": motion,
        "projected_motion_capability_passed": visible,
        "effectors": rows,
        "blockers": blockers,
        "warnings": [],
        "claim_boundary": "Controller-FK motion is an action-conditioning capability check, not task-success proof.",
    }


def build_wam_isaac_disagreement_contract(
    *,
    trace_rows: Sequence[Mapping[str, Any]],
    primary_wam_score: Mapping[str, Any],
    independent_consistency_proven: bool,
    accepted_calibration_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Preserve WAM scoring while fail-closing claims on unresolved contradiction."""
    contradictions: list[dict[str, Any]] = []
    missing_steps: list[int] = []
    measured_steps: list[int] = []
    for row in trace_rows:
        step = int(row.get("step_index") or 0)
        wam_stance = _mapping(row.get("wam_predicted_stance_report"))
        isaac_stance = _mapping(row.get("post_action_stance_report"))
        if isaac_stance.get("status") not in {"upright", "unsafe"}:
            missing_steps.append(step)
            continue
        measured_steps.append(step)
        if wam_stance.get("status") in {"upright", "unsafe"} and wam_stance.get("status") != isaac_stance.get("status"):
            contradictions.append({
                "step_index": step,
                "kind": "stance_outcome_mismatch",
                "wam_status": wam_stance.get("status"),
                "isaac_status": isaac_stance.get("status"),
            })
    primary_success = primary_wam_score.get("generated_video_success_label_passed") is True
    if primary_success:
        for row in trace_rows:
            stance = _mapping(row.get("post_action_stance_report"))
            if stance.get("unsafe_stance_detected") is True:
                contradictions.append({
                    "step_index": int(row.get("step_index") or 0),
                    "kind": "wam_success_vs_isaac_unsafe_stance",
                    "wam_status": "success",
                    "isaac_status": "unsafe",
                })
    calibration = _mapping(accepted_calibration_authority)
    calibration_accepted = bool(
        calibration.get("status") == "accepted"
        and calibration.get("authority_kind") in {"physical_robot", "frozen_benchmark"}
    )
    disagreement_unresolved = bool(contradictions or missing_steps)
    usable = bool(
        independent_consistency_proven
        and calibration_accepted
        and not disagreement_unresolved
    )
    blockers: list[str] = []
    if contradictions:
        blockers.append("wam_isaac_categorical_disagreement_unresolved")
    if missing_steps:
        blockers.append("isaac_contradiction_check_incomplete")
    if not independent_consistency_proven:
        blockers.append("independent_wam_consistency_not_proven")
    if not calibration_accepted:
        blockers.append("accepted_physical_or_frozen_benchmark_calibration_missing")
    numeric_error_rows = [
        {
            "step_index": int(row.get("step_index") or 0),
            **_mapping(row.get("wam_isaac_prefix_prediction_error")),
        }
        for row in trace_rows
        if _mapping(row.get("wam_isaac_prefix_prediction_error")).get("status")
        == "measured"
    ]
    mean_absolute_errors = [
        float(row["mean_absolute_error_rad"])
        for row in numeric_error_rows
        if _finite_float(row.get("mean_absolute_error_rad")) is not None
    ]
    maximum_absolute_errors = [
        float(row["maximum_absolute_error_rad"])
        for row in numeric_error_rows
        if _finite_float(row.get("maximum_absolute_error_rad")) is not None
    ]
    numeric_error_aggregate = {
        "schema_version": "wam_isaac_prefix_prediction_error_aggregate.v1",
        "status": "measured" if numeric_error_rows else "not_measured",
        "measured_step_count": len(numeric_error_rows),
        "measured_step_indices": [row["step_index"] for row in numeric_error_rows],
        "mean_of_step_mean_absolute_error_rad": (
            sum(mean_absolute_errors) / len(mean_absolute_errors)
            if mean_absolute_errors
            else None
        ),
        "maximum_step_mean_absolute_error_rad": (
            max(mean_absolute_errors) if mean_absolute_errors else None
        ),
        "maximum_joint_absolute_error_rad": (
            max(maximum_absolute_errors) if maximum_absolute_errors else None
        ),
        "cumulative_step_mean_absolute_error_rad": sum(mean_absolute_errors),
        "threshold_status": "not_calibrated_numeric_error_is_advisory",
        "claim_boundary": (
            "The aggregate exposes accumulated WAM-versus-Isaac prefix drift. "
            "No numeric pass threshold is claimed before accepted calibration."
        ),
    }
    return {
        "schema_version": "wam_isaac_disagreement.v1",
        "status": "usable_for_calibrated_claim" if usable else "uncalibrated_debug_evidence",
        "primary_rollout_authority": "wam_policy_evaluator",
        "isaac_role": "parallel_contradiction_detector",
        "primary_wam_score": dict(primary_wam_score),
        "primary_wam_score_preserved": True,
        "isaac_can_overwrite_or_terminate_wam_rollout": False,
        "measured_step_indices": measured_steps,
        "missing_diagnostic_step_indices": missing_steps,
        "categorical_contradictions": contradictions,
        "numeric_prefix_prediction_error_aggregate": numeric_error_aggregate,
        "disagreement_unresolved": disagreement_unresolved,
        "independent_consistency_proven": bool(independent_consistency_proven),
        "accepted_calibration_authority": calibration or None,
        "claim_ceiling": "calibrated_sim_evaluation" if usable else "uncalibrated_debug_evidence",
        "task_success_claim_allowed": usable and primary_success,
        "rank_fidelity_claim_allowed": usable,
        "blockers": blockers,
        "claim_boundary": (
            "Isaac disagreement never changes the WAM score or rollout. Unresolved contradiction, "
            "missing independent consistency, or missing accepted calibration caps claims."
        ),
    }
