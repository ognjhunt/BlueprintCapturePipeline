"""Bridge Blueprint policy observations to a Unitree UnifoLM-VLA server."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


SCHEMA_VERSION = "unitree_unifolm_vla_server_bridge.v1"
DEFAULT_SERVER_URL = "http://127.0.0.1:8777/act"
DEFAULT_TASK_NAME = "g1_stack_block"
DEFAULT_PROPRIO_DIM = 23


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _read_payload() -> dict[str, Any]:
    input_path = os.getenv("BLUEPRINT_POLICY_ACTION_INPUT", "").strip()
    if input_path:
        value = json.loads(Path(input_path).expanduser().read_text(encoding="utf-8"))
    else:
        raw = sys.stdin.read().strip()
        value = json.loads(raw) if raw else {}
    if not isinstance(value, Mapping):
        raise ValueError("policy bridge input must be a JSON object")
    return dict(value)


def _write_payload(payload: Mapping[str, Any], output_path: str | None = None) -> None:
    encoded = json.dumps(dict(payload), sort_keys=True)
    target = (
        output_path
        or os.getenv("BLUEPRINT_UNITREE_UNIFOLM_POLICY_OUTPUT", "").strip()
        or os.getenv("BLUEPRINT_POLICY_ACTION_OUTPUT", "").strip()
    )
    if target:
        path = Path(target).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def _observation(payload: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("observation"), Mapping):
        return dict(payload["observation"])  # type: ignore[index]
    return dict(payload)


def _camera_frame_path(observation: Mapping[str, Any]) -> Path | None:
    visual = _mapping(observation.get("visual_observation"))
    candidates = [
        os.getenv("BLUEPRINT_UNITREE_UNIFOLM_POLICY_FRAME", "").strip(),
        visual.get("camera_frame_path"),
        _mapping(observation.get("sensor_surrogates")).get("camera_frame_path"),
        observation.get("camera_frame_path"),
    ]
    for candidate in candidates:
        if candidate:
            path = Path(str(candidate)).expanduser()
            if path.is_file():
                return path
    return None


def _task_prompt(observation: Mapping[str, Any]) -> str:
    return (
        _string(observation.get("task_prompt"))
        or _string(observation.get("instruction"))
        or "move the Unitree G1 hand toward the target object"
    )


def _task_name(observation: Mapping[str, Any], task_name: str | None = None) -> str:
    return (
        _string(task_name)
        or os.getenv("BLUEPRINT_UNITREE_UNIFOLM_TASK_NAME", "").strip()
        or _string(observation.get("unitree_unifolm_task_name"))
        or DEFAULT_TASK_NAME
    )


def _proprio_dim(value: int | None = None) -> int:
    if value and value > 0:
        return int(value)
    env_value = os.getenv("BLUEPRINT_UNITREE_UNIFOLM_PROPRIO_DIM", "").strip()
    if env_value:
        try:
            parsed = int(env_value)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return DEFAULT_PROPRIO_DIM


def _sequence_numbers(value: Any) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_number(item, 0.0) for item in value]
    return []


def _proprio_state(observation: Mapping[str, Any], dim: int) -> list[float]:
    candidates = [
        observation.get("unitree_unifolm_state"),
        observation.get("unitree_proprio"),
        observation.get("proprio"),
        observation.get("state"),
        _mapping(observation.get("robot_state")).get("unitree_unifolm_state"),
        _mapping(observation.get("robot_state")).get("proprio"),
    ]
    for candidate in candidates:
        values = _sequence_numbers(candidate)
        if values:
            padded = (values + [0.0] * dim)[:dim]
            return padded
    return [0.0] * dim


def _object_waypoint(observation: Mapping[str, Any]) -> list[float]:
    object_state = _mapping(observation.get("object_state"))
    position = object_state.get("position") or [0.36, -0.65, 0.27]
    values = _sequence_numbers(position)
    x = values[0] if len(values) > 0 else 0.36
    y = values[1] if len(values) > 1 else -0.65
    return [round(x + 0.18, 6), round(y, 6), 0.79]


def _image_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _json_numpy_dumps(payload: Mapping[str, Any]) -> str:
    try:
        import json_numpy  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("blocked_missing_json_numpy_for_unitree_unifolm_bridge") from exc
    json_numpy.patch()
    return json_numpy.dumps(dict(payload))


def _json_numpy_loads(payload: str) -> Any:
    try:
        import json_numpy  # type: ignore[import-not-found]
    except ImportError:
        return json.loads(payload)
    json_numpy.patch()
    return json_numpy.loads(payload)


def build_unitree_unifolm_server_payload(
    *,
    observation: Mapping[str, Any],
    frame_path: Path,
    task_name: str | None = None,
    proprio_dim: int | None = None,
) -> dict[str, Any]:
    dim = _proprio_dim(proprio_dim)
    return {
        "observations": [
            {
                "full_image": _image_array(frame_path),
                "instruction": _task_prompt(observation),
                "state": np.asarray(_proprio_state(observation, dim), dtype=np.float32),
                "task_name": _task_name(observation, task_name),
            }
        ]
    }


def _post_unitree_server(
    *,
    server_url: str,
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> Any:
    body = json.dumps({"encoded": _json_numpy_dumps(payload)}).encode("utf-8")
    request = urllib.request.Request(
        server_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "BlueprintUnitreeUnifoLMBridge/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        response_text = response.read().decode("utf-8")
    parsed = json.loads(response_text)
    if isinstance(parsed, str):
        return _json_numpy_loads(parsed)
    return parsed


def _action_chunk(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item.tolist() if isinstance(item, np.ndarray) else item for item in value]
    return []


def run_bridge_policy(
    *,
    payload: Mapping[str, Any],
    server_url: str = DEFAULT_SERVER_URL,
    timeout_seconds: float = 30.0,
    task_name: str | None = None,
    proprio_dim: int | None = None,
) -> tuple[dict[str, Any], int]:
    observation = _observation(payload)
    frame = _camera_frame_path(observation)
    if frame is None:
        return (
            {
                "schema_version": SCHEMA_VERSION,
                "status": "blocked",
                "policy_id": "unitree_unifolm_vla_policy",
                "model_ran": False,
                "unitree_unifolm_policy_action_command_ran": False,
                "blockers": ["blocked_missing_policy_visual_observation_frame"],
                "raw_credentials_written_to_artifacts": False,
                "secret_hashes_written_to_artifacts": False,
            },
            2,
        )
    try:
        request_payload = build_unitree_unifolm_server_payload(
            observation=observation,
            frame_path=frame,
            task_name=task_name,
            proprio_dim=proprio_dim,
        )
        server_action = _post_unitree_server(
            server_url=server_url,
            payload=request_payload,
            timeout_seconds=timeout_seconds,
        )
    except (OSError, urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
        blocker = (
            str(exc)
            if str(exc).startswith("blocked_")
            else f"blocked_unitree_unifolm_vla_server_call_failed:{type(exc).__name__}"
        )
        return (
            {
                "schema_version": SCHEMA_VERSION,
                "status": "blocked",
                "policy_id": "unitree_unifolm_vla_policy",
                "model_ran": False,
                "unitree_unifolm_policy_action_command_ran": False,
                "server_url": server_url,
                "blockers": [blocker],
                "raw_credentials_written_to_artifacts": False,
                "secret_hashes_written_to_artifacts": False,
            },
            2,
        )

    chunk = _action_chunk(server_action)
    if not chunk:
        return (
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "policy_id": "unitree_unifolm_vla_policy",
                "model_ran": True,
                "unitree_unifolm_policy_action_command_ran": False,
                "server_url": server_url,
                "blockers": ["unitree_unifolm_vla_server_returned_empty_action"],
                "raw_credentials_written_to_artifacts": False,
                "secret_hashes_written_to_artifacts": False,
            },
            1,
        )

    response = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "policy_id": "unitree_unifolm_vla_policy",
        "policy_kind": "unitree_unifolm_vla_policy_server_bridge",
        "model_ran": True,
        "unitree_unifolm_policy_action_command_ran": True,
        "server_url": server_url,
        "task_name": _task_name(observation, task_name),
        "action": {
            "action_type": "manipulation_contact",
            "target_object_id": _string(_mapping(observation.get("object_state")).get("object_id"))
            or "blueprint_light_object",
            "waypoint": _object_waypoint(observation),
            "approach_speed_mps": 0.04,
            "unitree_unifolm_action_chunk_present": True,
            "unitree_unifolm_action_chunk": chunk,
        },
        "claim_boundary": {
            "unitree_unifolm_policy_command_ran": True,
            "unitree_hand_manipulation_policy_used": True,
            "real_vla_or_unitree_hand_policy_endpoint_used": True,
            "single_action_is_not_episode_success": True,
            "unitree_g1_dexterous_manipulation_proven": False,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    return response, 0


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default=os.getenv("BLUEPRINT_UNITREE_UNIFOLM_VLA_SERVER_URL", DEFAULT_SERVER_URL))
    parser.add_argument("--timeout-seconds", type=float, default=float(os.getenv("BLUEPRINT_UNITREE_UNIFOLM_TIMEOUT_SECONDS", "30")))
    parser.add_argument("--task-name", default=os.getenv("BLUEPRINT_UNITREE_UNIFOLM_TASK_NAME"))
    parser.add_argument("--proprio-dim", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    response, exit_code = run_bridge_policy(
        payload=_read_payload(),
        server_url=args.server_url,
        timeout_seconds=args.timeout_seconds,
        task_name=args.task_name,
        proprio_dim=args.proprio_dim,
    )
    _write_payload(response)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
