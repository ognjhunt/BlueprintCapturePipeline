"""GR00T N1.7 + SONIC ZMQ policy endpoint for the Isaac closed loop (T4).

Builds a ``PolicyEndpoint`` callable (adapted_observation, action_history,
step_index) -> action dict for ``run_oscar_isaac_closed_loop``, backed by a
live GR00T policy server via the existing in-process
``run_policy_server_command`` client. Each requery performs REAL model
inference on the WAM-generated observation frame the harness adapted.

Honesty: SONIC returns a horizon of 78-dim upper-body/motion-token control
frames, not a root waypoint.  Existing callers still execute one explicit
frame.  A caller may opt into a bounded prefix of the exact model horizon; that
prefix is carried as a hash-bound controller action sequence while the declared
root visualization remains review-only and is not a controller command.
T4's proof gates measure that the policy was genuinely requeried per step and
that its actions vary with the generated observation — they make no claim
that this projection is semantically meaningful locomotion.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .unitree_groot_n17_sonic_policy_server_command import (
    SONIC_ACTION_SEQUENCE_SCHEMA_VERSION,
    SONIC_CONTROL_FRAME_DIM,
    run_policy_server_command,
)

ENDPOINT_LABEL = "groot_n17_sonic_zmq_policy_endpoint"
PROJECTION_STEP_M = 0.06
PROJECTION_YAW_RAD = 0.15
CONTROLLER_ACTION_SEQUENCE_SCHEMA_VERSION = "gear_sonic_controller_action_sequence.v1"


def _floats(value: Any) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        out = []
        for item in value:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                out.append(0.0)
        return out
    return []


def _chunk_from_response(response: Mapping[str, Any]) -> list[float]:
    action = response.get("action")
    if isinstance(action, Mapping):
        chunk = _floats(action.get("action_chunk"))
        if chunk:
            return chunk
    chunk = _floats(response.get("action_chunk"))
    return chunk


def _strict_execution_frames(value: Any) -> list[list[float]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RuntimeError("groot_sonic_requery_blocked:sonic_action_sequence_missing")
    frames: list[list[float]] = []
    for raw_frame in value:
        if isinstance(raw_frame, (str, bytes)) or not isinstance(raw_frame, Sequence):
            raise RuntimeError(
                "groot_sonic_requery_blocked:sonic_action_sequence_frame_invalid"
            )
        try:
            frame = [float(item) for item in raw_frame]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "groot_sonic_requery_blocked:sonic_action_sequence_frame_invalid"
            ) from exc
        if len(frame) != SONIC_CONTROL_FRAME_DIM or not all(
            math.isfinite(item) for item in frame
        ):
            raise RuntimeError(
                "groot_sonic_requery_blocked:sonic_action_sequence_frame_invalid"
            )
        frames.append(frame)
    if not frames:
        raise RuntimeError("groot_sonic_requery_blocked:sonic_action_sequence_empty")
    return frames


def _frames_sha256(frames: Sequence[Sequence[float]]) -> str:
    return hashlib.sha256(
        json.dumps(frames, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _controller_action_sequence(
    *,
    response_action: Mapping[str, Any],
    selected_chunk: Sequence[float],
    execution_frame_count: int,
) -> dict[str, Any]:
    raw_sequence = response_action.get("sonic_action_sequence")
    if not isinstance(raw_sequence, Mapping):
        if execution_frame_count != 1:
            raise RuntimeError(
                "groot_sonic_requery_blocked:full_sonic_action_sequence_required"
            )
        frames = [[float(item) for item in selected_chunk]]
        source_frames_sha256 = _frames_sha256(frames)
        source_fieldwise_sha256 = None
        control_hz = float(
            dict(response_action.get("action_timing") or {}).get("control_hz")
            or 50.0
        )
    else:
        sequence = dict(raw_sequence)
        if sequence.get("schema_version") != SONIC_ACTION_SEQUENCE_SCHEMA_VERSION:
            raise RuntimeError(
                "groot_sonic_requery_blocked:sonic_action_sequence_schema_mismatch"
            )
        frames = _strict_execution_frames(sequence.get("frames"))
        if (
            int(sequence.get("frame_count") or 0) != len(frames)
            or int(sequence.get("frame_dimension") or 0) != SONIC_CONTROL_FRAME_DIM
        ):
            raise RuntimeError(
                "groot_sonic_requery_blocked:sonic_action_sequence_shape_mismatch"
            )
        source_frames_sha256 = _frames_sha256(frames)
        if str(sequence.get("frames_sha256") or "") != source_frames_sha256:
            raise RuntimeError(
                "groot_sonic_requery_blocked:sonic_action_sequence_sha256_mismatch"
            )
        source_fieldwise_sha256 = str(
            sequence.get("source_fieldwise_horizon_sha256") or ""
        ) or None
        control_hz = float(sequence.get("control_hz") or 0.0)
    if not math.isfinite(control_hz) or control_hz <= 0.0:
        raise RuntimeError(
            "groot_sonic_requery_blocked:sonic_action_sequence_control_hz_invalid"
        )
    if list(selected_chunk) != frames[0]:
        raise RuntimeError(
            "groot_sonic_requery_blocked:selected_chunk_sequence_frame_zero_mismatch"
        )
    if execution_frame_count > len(frames):
        raise RuntimeError(
            "groot_sonic_requery_blocked:execution_frame_count_exceeds_model_horizon"
        )
    execution_frames = frames[:execution_frame_count]
    execution_sha256 = _frames_sha256(execution_frames)
    return {
        "schema_version": CONTROLLER_ACTION_SEQUENCE_SCHEMA_VERSION,
        "execution_mode": (
            "single_frame_receding_horizon"
            if execution_frame_count == 1
            else "bounded_model_horizon_prefix"
        ),
        "execution_frame_count": execution_frame_count,
        "source_horizon_frame_count": len(frames),
        "frame_dimension": SONIC_CONTROL_FRAME_DIM,
        "control_hz": control_hz,
        "sample_period_seconds": 1.0 / control_hz,
        "execution_duration_seconds": execution_frame_count / control_hz,
        "frames": execution_frames,
        "frames_sha256": execution_sha256,
        "source_frames_sha256": source_frames_sha256,
        "source_fieldwise_horizon_sha256": source_fieldwise_sha256,
    }


def project_chunk_to_root_delta(chunk: Sequence[float]) -> tuple[float, float, float]:
    """Declared deterministic projection: SONIC chunk -> (dx, dy, dyaw).

    Uses leading chunk components directly (not segment means — SONIC chunks
    are near zero-mean, so means collapse every step to an identical zero
    delta and erase the model's step-to-step variation).
    """
    values = [float(v) for v in chunk] or [0.0]

    def _component(index: int) -> float:
        return values[index] if index < len(values) else 0.0

    dx = PROJECTION_STEP_M * math.tanh(_component(0))
    dy = PROJECTION_STEP_M * math.tanh(_component(1))
    dyaw = PROJECTION_YAW_RAD * math.tanh(_component(3))
    return round(dx, 6), round(dy, 6), round(dyaw, 6)


def nominal_unitree_g1_sonic_state() -> dict[str, list[float]]:
    """Declared nominal-stance proprio surrogate for WAM-frame observations.

    A WAM generates video, not joint sensing, so the requery observation has
    no measured proprio. This constant, labeled stance keeps the server's
    state contract satisfied; the generated camera frame is the channel that
    varies per step.
    """
    return {
        "left_leg": [0.0] * 6,
        "right_leg": [0.0] * 6,
        "waist": [0.0] * 3,
        "left_arm": [0.0] * 7,
        "right_arm": [0.0] * 7,
        "left_hand": [0.0] * 7,
        "right_hand": [0.0] * 7,
        "projected_gravity": [0.0, 0.0, -1.0],
    }


def make_groot_sonic_zmq_policy_endpoint(
    *,
    policy_server_url: str,
    groot_root: str | None = None,
    timeout_ms: int = 30000,
    sonic_state: Mapping[str, Any] | None = None,
    execution_frame_count: int = 1,
    run_command=run_policy_server_command,
):
    """Return a PolicyEndpoint backed by the live GR00T ZMQ server."""
    if isinstance(execution_frame_count, bool) or int(execution_frame_count) < 1:
        raise ValueError("groot_sonic_execution_frame_count_invalid")
    configured_execution_frame_count = int(execution_frame_count)
    configured_initial_state = dict(sonic_state) if sonic_state else None

    def endpoint(
        adapted_observation: Mapping[str, Any],
        action_history: Sequence[Mapping[str, Any]],
        step_index: int,
    ) -> dict[str, Any]:
        observation = dict(adapted_observation)
        if "unitree_g1_sonic_state" not in observation:
            carried = adapted_observation.get("generated_robot_state")
            carried_state = (
                carried.get("unitree_g1_sonic_state")
                if isinstance(carried, Mapping)
                else None
            )
            state = carried_state or configured_initial_state
            if not isinstance(state, Mapping):
                raise RuntimeError(
                    "groot_sonic_requery_blocked:measured_or_controller_carried_proprio_missing"
                )
            observation["unitree_g1_sonic_state"] = dict(state)
            observation["unitree_g1_sonic_state_source"] = (
                "controller_fk_carried_state"
                if carried_state is not None
                else "attempt_bound_initial_simulator_proprioception"
            )
            observation["unitree_g1_sonic_state_metadata"] = {
                "complete": True,
                "surrogate": False,
                "measured_proprio_available": True,
            }
        response, exit_code = run_command(
            payload={"observation": observation},
            policy_server_url=policy_server_url,
            groot_root=groot_root,
            timeout_ms=timeout_ms,
        )
        response = dict(response or {})
        status = str(response.get("status") or "").strip()
        if exit_code != 0 or status != "completed":
            # Raise so the loop records a FAILED requery instead of counting a
            # blocked payload as a completed policy decision.
            raise RuntimeError(
                "groot_sonic_requery_blocked:"
                + ",".join(str(b) for b in (response.get("blockers") or [f"exit_code:{exit_code}"]))
            )
        chunk = _chunk_from_response(response)
        if not chunk:
            raise RuntimeError("groot_sonic_requery_blocked:blocked_empty_sonic_action_chunk")
        response_action = (
            dict(response.get("action"))
            if isinstance(response.get("action"), Mapping)
            else {}
        )
        controller_action = _controller_action_sequence(
            response_action=response_action,
            selected_chunk=chunk,
            execution_frame_count=configured_execution_frame_count,
        )
        # Use the same exact float32-derived representation that is bound into
        # the controller sequence. Rounding only the compatibility field made
        # ordinary model values such as -0.13916015625 diverge from frame zero,
        # which the official executor correctly rejects before transport.
        canonical_selected_chunk = [
            float(value) for value in controller_action["frames"][0]
        ]
        dx, dy, dyaw = project_chunk_to_root_delta(chunk)
        action_timing = dict(
            response_action.get("action_timing")
            or {"control_hz": 50.0, "sample_period_seconds": 0.02}
        )
        previous = None
        for row in reversed(list(action_history or [])):
            if isinstance(row, Mapping) and _floats(row.get("root_position")):
                previous = _floats(row.get("root_position"))
                break
        base = previous or [0.0, 0.0, 0.79]
        prev_yaw = 0.0
        for row in reversed(list(action_history or [])):
            if isinstance(row, Mapping) and row.get("root_yaw_radians") is not None:
                try:
                    prev_yaw = float(row.get("root_yaw_radians"))
                except (TypeError, ValueError):
                    prev_yaw = 0.0
                break
        return {
            "status": "completed",
            "policy_action": "UNITREE_G1_SONIC",
            "endpoint": ENDPOINT_LABEL,
            "root_position": [
                round(base[0] + dx, 6),
                round(base[1] + dy, 6),
                round(base[2] if len(base) > 2 else 0.79, 6),
            ],
            "root_yaw_radians": round(prev_yaw + dyaw, 6),
            "sonic_action_chunk_dim": len(chunk),
            "sonic_action_chunk": canonical_selected_chunk,
            "action_units": list(
                response_action.get("action_units") or ["latent"] * len(chunk)
            ),
            "action_timing": action_timing,
            "action_horizon": dict(response_action.get("action_horizon") or {}),
            "controller_action": controller_action,
            "sonic_action_execution_frame_count": controller_action[
                "execution_frame_count"
            ],
            "sonic_action_execution_frames_sha256": controller_action[
                "frames_sha256"
            ],
            "learned_policy_runtime_result_id": response.get("runtime_result_id"),
            "sonic_action_chunk_head": [round(v, 6) for v in chunk[:8]],
            "requery_step_index": int(step_index),
            "out_of_distribution_action_projection": False,
            "not_a_learned_robot_policy_action": False,
            "projection": {
                "kind": "review_only_root_visualization_not_controller_command",
                "step_m": PROJECTION_STEP_M,
                "yaw_rad": PROJECTION_YAW_RAD,
            },
            "claim_boundary": {
                "real_model_inference": True,
                "projection_is_not_semantic_locomotion": True,
                "gear_sonic_controller_fk_required_for_execution_proof": True,
                "task_success_proven": False,
            },
        }

    endpoint.__name__ = ENDPOINT_LABEL
    return endpoint
