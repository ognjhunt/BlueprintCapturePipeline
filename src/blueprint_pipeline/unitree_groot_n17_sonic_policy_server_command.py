"""Blueprint command for a GR00T N1.7 + UNITREE_G1_SONIC PolicyServer.

This is a simulator/open-loop action-command bridge only. It sends a
Blueprint-provided UNITREE_G1_SONIC observation to an already-running
Isaac-GR00T PolicyServer and normalizes the returned SONIC latent action
fields. It never launches physical robot commands.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np

from .unitree_groot_n17_sonic_policy_runtime import (
    GROOT_ROOT_ENV,
    POLICY_ID,
    POLICY_SERVER_URL_ENV,
)


SCHEMA_VERSION = "unitree_groot_n17_sonic_policy_server_command.v1"
TOKEN_FILE_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_TOKEN_FILE"
DEFAULT_TIMEOUT_MS = 15000
REQUIRED_STATE_DIMS = {
    "left_leg": 6,
    "right_leg": 6,
    "waist": 3,
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
    "projected_gravity": 3,
}
REQUIRED_STATE_KEYS = tuple(REQUIRED_STATE_DIMS)


class _ZmqPolicyClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout_ms: int,
        api_token: str | None = None,
        strict: bool = False,
    ) -> None:
        import msgpack
        import msgpack_numpy as msgpack_numpy
        import zmq

        self._msgpack = msgpack
        self._msgpack_numpy = msgpack_numpy
        self._zmq = zmq
        self.api_token = api_token
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self.socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self.socket.connect(f"tcp://{host}:{int(port)}")

    def _to_bytes(self, payload: Mapping[str, Any]) -> bytes:
        return self._msgpack_numpy.packb(dict(payload), use_bin_type=True)

    def _from_bytes(self, payload: bytes) -> Any:
        return self._msgpack_numpy.unpackb(payload, raw=False)

    def call_endpoint(
        self,
        endpoint: str,
        data: Mapping[str, Any] | None = None,
        *,
        requires_input: bool = True,
    ) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = dict(data or {})
        if self.api_token:
            request["api_token"] = self.api_token
        try:
            self.socket.send(self._to_bytes(request))
            response = self.socket.recv()
        except self._zmq.error.Again as exc:
            raise TimeoutError(f"gr00t_policy_server_timeout:{endpoint}") from exc
        value = self._from_bytes(response)
        if isinstance(value, Mapping) and value.get("error"):
            raise RuntimeError(f"gr00t_policy_server_error:{value['error']}")
        return value

    def ping(self) -> bool:
        self.call_endpoint("ping", requires_input=False)
        return True

    def get_action(self, observation: Mapping[str, Any]) -> tuple[Any, Any]:
        response = self.call_endpoint(
            "get_action",
            {"observation": dict(observation), "options": None},
        )
        if isinstance(response, Sequence) and not isinstance(response, (str, bytes, bytearray)):
            if len(response) >= 2:
                return response[0], response[1]
        raise RuntimeError("gr00t_policy_server_get_action_response_not_action_info_pair")

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
        finally:
            self.context.term()


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if any(marker in key_text.lower() for marker in ("token", "secret", "password", "key")):
                result[key_text] = "<redacted>"
            else:
                result[key_text] = _redact(child)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _read_payload() -> dict[str, Any]:
    input_path = os.getenv("BLUEPRINT_POLICY_ACTION_INPUT", "").strip()
    if input_path:
        value = json.loads(Path(input_path).expanduser().read_text(encoding="utf-8"))
    else:
        raw = sys.stdin.read().strip()
        value = json.loads(raw) if raw else {}
    if not isinstance(value, Mapping):
        raise ValueError("policy input must be a JSON object")
    return dict(value)


def _write_payload(payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(_jsonable(dict(payload)), sort_keys=True)
    output_path = os.getenv("BLUEPRINT_POLICY_ACTION_OUTPUT", "").strip()
    if output_path:
        path = Path(output_path).expanduser()
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


def _parse_policy_server_url(value: str) -> tuple[str, int] | None:
    text = value.strip()
    if not text:
        return None
    parsed = urlparse(text if "://" in text else f"tcp://{text}")
    if parsed.hostname and parsed.port:
        return parsed.hostname, int(parsed.port)
    return None


def _token_from_file(path_value: str | None) -> str | None:
    path_text = _string(path_value)
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{TOKEN_FILE_ENV}_missing")
    return path.read_text(encoding="utf-8").strip()


def _load_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("blocked_pillow_not_available_for_policy_frame_load") from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _unitree_g1_sonic_state(blueprint_observation: Mapping[str, Any]) -> dict[str, Any] | None:
    value = blueprint_observation.get("unitree_g1_sonic_state")
    if isinstance(value, Mapping):
        return dict(value)
    state = _mapping(blueprint_observation.get("state"))
    if all(key in state for key in REQUIRED_STATE_KEYS):
        return state
    return None


def _state_value(raw_state: Mapping[str, Any], key: str) -> np.ndarray:
    value = raw_state.get(key)
    if value is None:
        value = raw_state.get(f"state.{key}")
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, 1, -1)
    elif arr.ndim == 2:
        arr = arr.reshape(1, *arr.shape)
    return arr


def build_unitree_g1_sonic_policy_observation(
    *,
    blueprint_observation: Mapping[str, Any],
    frame_path: Path,
) -> dict[str, Any]:
    raw_state = _unitree_g1_sonic_state(blueprint_observation)
    if raw_state is None:
        raise ValueError("blocked_missing_unitree_g1_sonic_state_fields")
    image = _load_image(frame_path)
    prompt = (
        _string(blueprint_observation.get("task_prompt"))
        or _string(blueprint_observation.get("prompt"))
        or _string(blueprint_observation.get("task_description"))
        or "Return one safe Unitree G1 SONIC action chunk for the simulated task."
    )
    return {
        "video": {"ego_view": image[np.newaxis, np.newaxis]},
        "state": {key: _state_value(raw_state, key) for key in REQUIRED_STATE_KEYS},
        "language": {"annotation.human.task_description": [[prompt]]},
    }


def _action_field(action: Mapping[str, Any], key: str) -> Any:
    if key in action:
        return action[key]
    return action.get(f"action.{key}")


def _normalize_policy_server_action(action: Mapping[str, Any]) -> dict[str, Any] | None:
    motion_token = _action_field(action, "motion_token")
    left_hand = _action_field(action, "left_hand_joints")
    right_hand = _action_field(action, "right_hand_joints")
    if motion_token is None and left_hand is None and right_hand is None:
        return None
    action_chunk: list[float] = []
    for value in (motion_token, left_hand, right_hand):
        if value is not None:
            action_chunk.extend(np.asarray(value, dtype=np.float32).reshape(-1).tolist())
    fields = sorted(
        key
        for key, value in (
            ("motion_token", motion_token),
            ("left_hand_joints", left_hand),
            ("right_hand_joints", right_hand),
        )
        if value is not None
    )
    normalized: dict[str, Any] = {
        "action_type": "unitree_g1_sonic_latent_action_chunk",
        "action_chunk": action_chunk,
        "unitree_groot_n17_sonic_action_payload_present": True,
        "unitree_groot_n17_sonic_action_chunk_present": True,
        "unitree_g1_sonic_control_fields": fields,
    }
    if motion_token is not None:
        normalized["sonic_latent_action"] = _jsonable(motion_token)
    hand_targets: dict[str, Any] = {}
    if left_hand is not None:
        hand_targets["left_hand_joints"] = _jsonable(left_hand)
    if right_hand is not None:
        hand_targets["right_hand_joints"] = _jsonable(right_hand)
    if hand_targets:
        normalized["hand_targets"] = hand_targets
    return normalized


def _blocked_payload(
    *,
    blockers: Sequence[str],
    server_url: str | None,
    frame_path: Path | None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "policy_id": POLICY_ID,
        "selected_candidate_id": POLICY_ID,
        "unitree_groot_n17_sonic_policy_action_command_ran": False,
        "unitree_policy_action_command_ran": False,
        "unitree_specific_manipulation_candidate_ran": False,
        "openvla_policy_action_command_ran": False,
        "model_ran": False,
        "policy_server_url_configured": bool(server_url),
        "policy_server_url_redacted": "<configured>" if server_url else None,
        "camera_frame_path": str(frame_path) if frame_path else None,
        "blockers": sorted(set(blockers)),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "claim_boundary": {
            "simulator_only": True,
            "policy_server_command_is_open_loop_action_probe": True,
            "policy_server_command_is_not_model_proof_when_blocked": True,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
            "accepted_anchor_manipulation_success_proven": False,
        },
    }
    if error_type:
        payload["error_type"] = error_type
    if error_message:
        payload["error_message_redacted"] = _redact(error_message)
    return payload


def _resolve_server(
    *,
    policy_server_url: str | None,
    policy_server_host: str | None,
    policy_server_port: int | None,
    server_url: str | None,
) -> tuple[str | None, tuple[str, int] | None]:
    resolved_url = policy_server_url or server_url
    parsed = _parse_policy_server_url(resolved_url or "")
    if parsed is None and policy_server_host and policy_server_port:
        parsed = (policy_server_host, int(policy_server_port))
        resolved_url = f"tcp://{policy_server_host}:{int(policy_server_port)}"
    return resolved_url, parsed


def run_policy_server_command(
    *,
    payload: Mapping[str, Any],
    policy_server_url: str | None = None,
    policy_server_host: str | None = None,
    policy_server_port: int | None = None,
    server_url: str | None = None,
    groot_root: str | None = None,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    token_file: str | None = None,
    policy_client_factory: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], int]:
    observation = _observation(payload)
    frame_path = _camera_frame_path(observation)
    resolved_url, parsed_url = _resolve_server(
        policy_server_url=policy_server_url,
        policy_server_host=policy_server_host,
        policy_server_port=policy_server_port,
        server_url=server_url,
    )
    state = _unitree_g1_sonic_state(observation)
    state_source = _string(observation.get("unitree_g1_sonic_state_source"))
    state_metadata = _mapping(observation.get("unitree_g1_sonic_state_metadata"))
    missing_state_keys = [
        key for key in REQUIRED_STATE_KEYS if not isinstance(state, Mapping) or key not in state
    ]
    blockers: list[str] = []
    if parsed_url is None:
        blockers.append(f"set_{POLICY_SERVER_URL_ENV}_to_running_gr00t_policy_server")
    if frame_path is None:
        blockers.append("blocked_missing_policy_visual_observation_frame")
    if missing_state_keys:
        blockers.append("blocked_missing_unitree_g1_sonic_state_fields")
    if state_metadata.get("complete") is False:
        blockers.append("blocked_incomplete_unitree_g1_sonic_state_from_mujoco")
    if groot_root:
        root = Path(groot_root).expanduser()
        if root.is_dir() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
        elif not root.is_dir():
            blockers.append(f"blocked_missing_path_for_{GROOT_ROOT_ENV}")
    if blockers:
        return (
            _blocked_payload(
                blockers=blockers,
                server_url=resolved_url,
                frame_path=frame_path,
            )
            | {
                "missing_state_keys": missing_state_keys,
                "unitree_g1_sonic_state_source": state_source or None,
                "unitree_g1_sonic_state_source_is_contract_probe": (
                    "contract_probe" in state_source
                ),
                "unitree_g1_sonic_state_metadata": _jsonable(state_metadata),
            },
            2,
        )

    try:
        api_token = _token_from_file(token_file)
        if policy_client_factory is None:
            policy_client_factory = _ZmqPolicyClient
        host, port = parsed_url or ("localhost", 5550)
        client = policy_client_factory(
            host=host,
            port=port,
            timeout_ms=timeout_ms,
            api_token=api_token,
            strict=False,
        )
        try:
            enter = getattr(client, "__enter__", None)
            if callable(enter):
                client = enter()
            ping = getattr(client, "ping", None)
            if callable(ping) and not ping():
                return (
                    _blocked_payload(
                        blockers=["blocked_gr00t_policy_server_ping_failed"],
                        server_url=resolved_url,
                        frame_path=frame_path,
                    ),
                    2,
                )
            policy_observation = build_unitree_g1_sonic_policy_observation(
                blueprint_observation=observation,
                frame_path=frame_path or Path(),
            )
            action, info = client.get_action(policy_observation)
        finally:
            exit_method = getattr(client, "__exit__", None)
            close = getattr(client, "close", None)
            if callable(exit_method):
                exit_method(None, None, None)
            elif callable(close):
                close()
    except Exception as exc:
        return (
            _blocked_payload(
                blockers=[
                    "blocked_unitree_groot_n17_sonic_policy_server_command_failed:"
                    f"{type(exc).__name__}"
                ],
                server_url=resolved_url,
                frame_path=frame_path,
                error_type=type(exc).__name__,
                error_message=str(exc),
            ),
            2,
        )

    action_mapping = _mapping(action)
    normalized_action = _normalize_policy_server_action(action_mapping)
    if normalized_action is None:
        return (
            _blocked_payload(
                blockers=["blocked_gr00t_policy_server_response_missing_unitree_g1_sonic_action"],
                server_url=resolved_url,
                frame_path=frame_path,
            )
            | {"policy_server_action_keys": sorted(str(key) for key in action_mapping)},
            2,
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "policy_id": POLICY_ID,
        "selected_candidate_id": POLICY_ID,
        "unitree_groot_n17_sonic_policy_action_command_ran": True,
        "unitree_policy_action_command_ran": True,
        "unitree_specific_manipulation_candidate_ran": True,
        "openvla_policy_action_command_ran": False,
        "model_ran": True,
        "fresh_unitree_groot_n17_sonic_model_executed_this_invocation": True,
        "policy_server_url_configured": True,
        "policy_server_url_redacted": "<configured>",
        "policy_server_host": parsed_url[0] if parsed_url else None,
        "policy_server_port": parsed_url[1] if parsed_url else None,
        "camera_frame_path": str(frame_path),
        "observation_metadata": {
            "state_keys": list(REQUIRED_STATE_KEYS),
            "video_keys": ["ego_view"],
            "language_keys": ["annotation.human.task_description"],
            "unitree_g1_sonic_state_source": state_source or None,
            "unitree_g1_sonic_state_source_is_contract_probe": (
                "contract_probe" in state_source
            ),
            "unitree_g1_sonic_state_metadata": _jsonable(state_metadata),
        },
        "action": normalized_action,
        "sonic_latent_action": normalized_action.get("sonic_latent_action"),
        "hand_targets": normalized_action.get("hand_targets"),
        "policy_server_action_keys": sorted(str(key) for key in action_mapping),
        "policy_server_info_redacted": _redact(_jsonable(info)),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "claim_boundary": {
            "simulator_only": True,
            "policy_server_command_is_open_loop_action_probe": True,
            "simulated_or_contract_probe_state_can_be_used_for_attempt": True,
            "simulated_or_contract_probe_state_does_not_prove_real_robot_state": True,
            "gr00t_sonic_action_command_is_single_step_not_task_success": True,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
            "accepted_anchor_manipulation_success_proven": False,
        },
    }
    return result, 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-server-url", default=os.getenv(POLICY_SERVER_URL_ENV, ""))
    parser.add_argument("--groot-root", default=os.getenv(GROOT_ROOT_ENV, ""))
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=int(
            os.getenv(
                "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_TIMEOUT_MS",
                DEFAULT_TIMEOUT_MS,
            )
        ),
    )
    parser.add_argument("--token-file", default=os.getenv(TOKEN_FILE_ENV, ""))
    args = parser.parse_args(argv)
    result, exit_code = run_policy_server_command(
        payload=_read_payload(),
        policy_server_url=args.policy_server_url,
        groot_root=args.groot_root,
        timeout_ms=args.timeout_ms,
        token_file=args.token_file,
    )
    _write_payload(result)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
