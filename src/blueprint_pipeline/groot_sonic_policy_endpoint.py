"""GR00T N1.7 + SONIC ZMQ policy endpoint for the Isaac closed loop (T4).

Builds a ``PolicyEndpoint`` callable (adapted_observation, action_history,
step_index) -> action dict for ``run_oscar_isaac_closed_loop``, backed by a
live GR00T policy server via the existing in-process
``run_policy_server_command`` client. Each requery performs REAL model
inference on the WAM-generated observation frame the harness adapted.

Honesty: SONIC returns a 78-dim upper-body/motion-token action chunk, not a
root waypoint. The chunk -> (root_position, root_yaw) mapping here is a
DECLARED deterministic projection so the closed loop has a drivable action;
every returned action carries ``out_of_distribution_action_projection: true``.
T4's proof gates measure that the policy was genuinely requeried per step and
that its actions vary with the generated observation — they make no claim
that this projection is semantically meaningful locomotion.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .unitree_groot_n17_sonic_policy_server_command import run_policy_server_command

ENDPOINT_LABEL = "groot_n17_sonic_zmq_policy_endpoint"
PROJECTION_STEP_M = 0.06
PROJECTION_YAW_RAD = 0.15


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
    run_command=run_policy_server_command,
):
    """Return a PolicyEndpoint backed by the live GR00T ZMQ server."""
    state = dict(sonic_state) if sonic_state else nominal_unitree_g1_sonic_state()

    def endpoint(
        adapted_observation: Mapping[str, Any],
        action_history: Sequence[Mapping[str, Any]],
        step_index: int,
    ) -> dict[str, Any]:
        observation = dict(adapted_observation)
        if "unitree_g1_sonic_state" not in observation:
            observation["unitree_g1_sonic_state"] = state
            observation["unitree_g1_sonic_state_source"] = (
                "nominal_stance_proprio_surrogate_constant"
            )
            observation["unitree_g1_sonic_state_metadata"] = {
                "complete": True,
                "surrogate": True,
                "measured_proprio_available": False,
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
        dx, dy, dyaw = project_chunk_to_root_delta(chunk)
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
            "policy_action": "learned_policy_action",
            "endpoint": ENDPOINT_LABEL,
            "root_position": [
                round(base[0] + dx, 6),
                round(base[1] + dy, 6),
                round(base[2] if len(base) > 2 else 0.79, 6),
            ],
            "root_yaw_radians": round(prev_yaw + dyaw, 6),
            "sonic_action_chunk_dim": len(chunk),
            "sonic_action_chunk_head": [round(v, 6) for v in chunk[:8]],
            "requery_step_index": int(step_index),
            "out_of_distribution_action_projection": True,
            "projection": {
                "kind": "declared_deterministic_leading_chunk_components_tanh",
                "step_m": PROJECTION_STEP_M,
                "yaw_rad": PROJECTION_YAW_RAD,
            },
            "claim_boundary": {
                "real_model_inference": True,
                "projection_is_not_semantic_locomotion": True,
                "task_success_proven": False,
            },
        }

    endpoint.__name__ = ENDPOINT_LABEL
    return endpoint
