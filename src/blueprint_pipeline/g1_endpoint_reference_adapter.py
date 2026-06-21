"""Reference command adapter for the local Unitree G1 policy endpoint.

This adapter is intentionally heuristic. Its purpose is to prove the real HTTP
endpoint path and Blueprint action contract without using the in-process fixture
fallback in the MuJoCo evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Mapping, Sequence


POLICY_ID = "blueprint_g1_endpoint_reference_adapter"
POLICY_KIND = "command_policy_reference_heuristic"
SUPPORTED_ACTION_TYPES = (
    "waypoint",
    "base_velocity",
    "stop",
    "inspect_look",
    "manipulation_contact",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _target_pose(observation: Mapping[str, Any]) -> list[float]:
    route = _mapping(observation.get("route_task_state"))
    target = route.get("target_pose") or [0.0, 0.0, 0.79]
    if isinstance(target, Sequence) and not isinstance(target, (str, bytes)) and len(target) >= 2:
        return [
            _number(target[0], 0.0),
            _number(target[1], 0.0),
            _number(target[2], 0.79) if len(target) > 2 else 0.79,
        ]
    return [0.0, 0.0, 0.79]


def _target_error(observation: Mapping[str, Any]) -> float:
    route = _mapping(observation.get("route_task_state"))
    return _number(route.get("target_error_m"), 99.0)


def _object_waypoint(observation: Mapping[str, Any]) -> list[float]:
    object_state = _mapping(observation.get("object_state"))
    position = object_state.get("position") or [0.36, -0.65, 0.27]
    if isinstance(position, Sequence) and not isinstance(position, (str, bytes)) and len(position) >= 2:
        return [_number(position[0], 0.36) + 0.18, _number(position[1], -0.65), 0.79]
    return [0.54, -0.65, 0.79]


def choose_action(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Choose a supported Blueprint action for a single observation packet."""

    task_id = str(observation.get("task_id") or "")
    step = int(_number(observation.get("step_index"), 0.0))
    target = _target_pose(observation)
    target_error = _target_error(observation)

    if task_id == "inspect_target":
        if step < 80:
            return {"action_type": "inspect_look", "yaw_rate_rad_s": 0.35}
        return {"action_type": "stop", "report": "inspection_sweep_complete"}

    if task_id == "approach_target":
        if target_error <= 0.35:
            return {"action_type": "stop", "report": "within_goal_tolerance"}
        if step < 80:
            return {
                "action_type": "base_velocity",
                "linear_velocity_mps": 0.28,
                "lateral_velocity_mps": 0.0,
                "yaw_rate_rad_s": 0.0,
            }
        return {"action_type": "waypoint", "waypoint": target}

    if task_id == "route_around_obstruction":
        route = _mapping(observation.get("route_task_state"))
        waypoints = route.get("route_waypoints") or [target[:2]]
        waypoint = target
        if isinstance(waypoints, Sequence) and not isinstance(waypoints, (str, bytes)) and waypoints:
            index = min(len(waypoints) - 1, max(0, step // 120))
            candidate = waypoints[index]
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)) and len(candidate) >= 2:
                waypoint = [
                    _number(candidate[0], target[0]),
                    _number(candidate[1], target[1]),
                    _number(candidate[2], 0.79) if len(candidate) > 2 else 0.79,
                ]
        return {"action_type": "waypoint", "waypoint": waypoint}

    if task_id == "contact_or_push_light_object":
        return {
            "action_type": "manipulation_contact",
            "target_object_id": "blueprint_light_object",
            "waypoint": _object_waypoint(observation),
        }

    if task_id == "stop_at_goal_and_report":
        if target_error <= 0.38 or step >= 160:
            return {"action_type": "stop", "report": "stopped_for_review"}
        return {"action_type": "waypoint", "waypoint": target}

    return {"action_type": "waypoint", "waypoint": target}


def adapter_manifest() -> dict[str, Any]:
    return {
        "schema_version": "policy_command_adapter_manifest.v1",
        "policy_id": POLICY_ID,
        "policy_kind": POLICY_KIND,
        "adapter_family": "command_policy",
        "supported_action_types": list(SUPPORTED_ACTION_TYPES),
        "reads_json_from_stdin": True,
        "writes_json_to_stdout": True,
        "http_endpoint_required_for_mujoco_lane": True,
        "fixture_policy_used": False,
        "real_wam_vla_model": False,
        "claim_boundary": {
            "heuristic_endpoint_reference_only": True,
            "not_real_wam_vla": True,
            "not_official_unitree_controller": True,
            "physical_robot_readiness_proven": False,
        },
    }


def build_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    observation = _mapping(payload.get("observation"))
    return {
        "policy_id": POLICY_ID,
        "policy_kind": POLICY_KIND,
        "action": choose_action(observation),
        "adapter_metadata": {
            "adapter_family": "command_policy",
            "fixture_policy_used": False,
            "real_wam_vla_model": False,
            "supported_action_types": list(SUPPORTED_ACTION_TYPES),
            "raw_token_values_returned": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-manifest", action="store_true")
    args = parser.parse_args(argv)
    if args.print_manifest:
        print(json.dumps(adapter_manifest(), sort_keys=True))
        return 0
    raw = sys.stdin.read()
    payload = json.loads(raw or "{}")
    if not isinstance(payload, Mapping):
        raise SystemExit("stdin_json_must_be_object")
    print(json.dumps(build_response(payload), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
