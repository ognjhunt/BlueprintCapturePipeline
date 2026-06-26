"""Unitree LeRobot G1 manipulation policy command adapter.

This adapter is the Blueprint endpoint boundary for a Unitree-specific hand or
gripper policy. It reads a Blueprint observation packet and either invokes a
configured Unitree/LeRobot manipulation runner or imports a provider output.

The adapter does not claim task success by itself. A completed response proves
that a Unitree/LeRobot policy command produced a Blueprint-compatible action;
episode success still requires simulator traces and review/scoring artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


POLICY_ID = "unitree_lerobot_g1_policy"
SCHEMA_VERSION = "unitree_lerobot_policy_command_adapter.v1"
RUNNER_COMMAND_ENV = "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND"
POLICY_PATH_ENV = "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT"
SOURCE_ROOT_ENV = "BLUEPRINT_UNITREE_LEROBOT_ROOT"
PROVIDER_OUTPUT_ENV = "BLUEPRINT_UNITREE_LEROBOT_PROVIDER_OUTPUT"
DEFAULT_REPO_ID = "unitreerobotics/G1_Dex3_ToastedBread_Dataset"
DEFAULT_ARM = "G1_29"
DEFAULT_EE = "dex3"
SUPPORTED_ACTION_TYPES = (
    "manipulation_contact",
    "waypoint",
    "stop",
    "inspect_look",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


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
        payload = json.loads(Path(input_path).expanduser().read_text(encoding="utf-8"))
    else:
        raw = sys.stdin.read().strip()
        payload = json.loads(raw) if raw else {}
    if not isinstance(payload, Mapping):
        raise ValueError("policy input must be a JSON object")
    return dict(payload)


def _write_payload(payload: Mapping[str, Any]) -> None:
    output_path = os.getenv("BLUEPRINT_POLICY_ACTION_OUTPUT", "").strip()
    encoded = json.dumps(dict(payload), sort_keys=True)
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


def _object_waypoint(observation: Mapping[str, Any]) -> list[float]:
    object_state = _mapping(observation.get("object_state"))
    position = object_state.get("position") or [0.36, -0.65, 0.27]
    if isinstance(position, Sequence) and not isinstance(position, (str, bytes)):
        x = _number(position[0], 0.36) if len(position) > 0 else 0.36
        y = _number(position[1], -0.65) if len(position) > 1 else -0.65
    else:
        x, y = 0.36, -0.65
    return [round(x + 0.18, 6), round(y, 6), 0.79]


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


def _normalize_runner_action(
    *,
    runner_payload: Mapping[str, Any],
    observation: Mapping[str, Any],
    allow_task_fallback: bool = True,
) -> dict[str, Any] | None:
    action = runner_payload.get("action") or runner_payload.get("normalized_action")
    if isinstance(action, Mapping):
        return dict(action)
    action_type = _string(runner_payload.get("action_type"))
    if action_type in SUPPORTED_ACTION_TYPES:
        return dict(runner_payload)
    action_chunk = (
        runner_payload.get("action_chunk")
        or runner_payload.get("actions")
        or runner_payload.get("joint_positions")
    )
    if (
        isinstance(action_chunk, Sequence)
        and not isinstance(action_chunk, (str, bytes, bytearray))
    ) or isinstance(runner_payload.get("end_effector_target"), Sequence) or isinstance(
        runner_payload.get("joint_targets"), Mapping
    ):
        return {
            "action_type": "manipulation_contact",
            "target_object_id": "blueprint_light_object",
            "waypoint": _object_waypoint(observation),
            "approach_speed_mps": 0.04,
            "unitree_lerobot_action_chunk_present": True,
            "unitree_lerobot_raw_action_keys": sorted(str(key) for key in runner_payload.keys()),
        }
    if not allow_task_fallback:
        return None
    task_id = _string(observation.get("task_id"))
    if task_id == "contact_or_push_light_object":
        return {
            "action_type": "manipulation_contact",
            "target_object_id": "blueprint_light_object",
            "waypoint": _object_waypoint(observation),
            "approach_speed_mps": 0.04,
        }
    if task_id == "inspect_target":
        return {"action_type": "inspect_look", "yaw_rate_rad_s": 0.18}
    if task_id == "stop_at_goal_and_report":
        return {"action_type": "stop", "report": "unitree_lerobot_policy_stop"}
    return {"action_type": "waypoint", "waypoint": _target_pose(observation), "max_speed_mps": 0.05}


def _command_available(command: str | None) -> bool:
    if not command:
        return False
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    if not parts:
        return False
    executable = parts[0]
    return bool(shutil.which(executable) or Path(executable).expanduser().exists())


def _policy_path_configured(policy_path: str | None) -> bool:
    text = _string(policy_path)
    if not text:
        return False
    path = Path(text).expanduser()
    if path.exists():
        return True
    return "/" in text and not text.startswith(("/", "./", "../", "~"))


def _claim_boundary(
    *,
    policy_command_ran: bool,
    provider_output_replay_used: bool = False,
) -> dict[str, Any]:
    return {
        "unitree_lerobot_policy_command_ran": bool(policy_command_ran),
        "unitree_hand_manipulation_policy_used": bool(policy_command_ran),
        "unitree_lerobot_or_isaaclab_manipulation_policy_used": bool(policy_command_ran),
        "provider_output_replay_used": bool(provider_output_replay_used),
        "provider_output_replay_is_not_fresh_per_request_model_inference": bool(
            provider_output_replay_used
        ),
        "unitree_g1_dexterous_manipulation_proven": False,
        "single_action_is_not_episode_success": True,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
    }


def _blocked_payload(
    *,
    blockers: Sequence[str],
    observation: Mapping[str, Any],
    command: str | None,
    policy_path: str | None,
    source_root: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "policy_id": POLICY_ID,
        "unitree_lerobot_policy_action_command_ran": False,
        "model_ran": False,
        "task_id": observation.get("task_id"),
        "blockers": sorted(set(blockers)),
        "command_configured": bool(command),
        "command_available": _command_available(command),
        "command_value_redacted": "<configured>" if command else None,
        "policy_path_configured": bool(policy_path),
        "policy_path": policy_path,
        "source_root_configured": bool(source_root),
        "source_root_path": source_root,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "claim_boundary": _claim_boundary(policy_command_ran=False),
    }


def _load_json_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("json file must contain an object")
    return dict(value)


def _provider_replay_payload(
    *,
    observation: Mapping[str, Any],
    provider_output: Path,
) -> tuple[dict[str, Any], int]:
    try:
        provider = _load_json_file(provider_output)
    except Exception as exc:
        return (
            _blocked_payload(
                blockers=[f"blocked_unitree_lerobot_provider_output_load_failed:{type(exc).__name__}"],
                observation=observation,
                command=None,
                policy_path=str(provider_output),
                source_root=None,
            )
            | {"provider_output_path": str(provider_output)},
            2,
        )
    provider_schema_valid = _string(provider.get("schema_version")) == SCHEMA_VERSION
    provider_status_completed = _string(provider.get("status")) == "completed"
    model_ran = bool(
        provider_schema_valid
        and provider_status_completed
        and (
            provider.get("unitree_lerobot_policy_action_command_ran") is True
            or provider.get("unitree_lerobot_policy_command_ran") is True
        )
    )
    action = _normalize_runner_action(
        runner_payload=provider,
        observation=observation,
        allow_task_fallback=False,
    )
    blockers: list[str] = []
    if not provider_schema_valid:
        blockers.append("blocked_unitree_lerobot_provider_output_schema_not_trusted")
    if provider_schema_valid and not provider_status_completed:
        blockers.append("blocked_unitree_lerobot_provider_output_not_completed")
    if not model_ran:
        blockers.append("blocked_unitree_lerobot_provider_output_missing_model_execution_proof")
    if not isinstance(action, Mapping):
        blockers.append("blocked_unitree_lerobot_provider_output_missing_action")
    if blockers:
        return (
            _blocked_payload(
                blockers=blockers,
                observation=observation,
                command=None,
                policy_path=str(provider_output),
                source_root=None,
            )
            | {"provider_output_path": str(provider_output), "provider_output_replay_used": True},
            2,
        )
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "policy_id": "unitree_lerobot_g1_policy_provider_replay",
            "policy_kind": "unitree_lerobot_g1_policy_provider_replay",
            "unitree_lerobot_policy_action_command_ran": True,
            "model_ran": True,
            "fresh_unitree_lerobot_model_executed_this_invocation": False,
            "provider_output_replay_used": True,
            "provider_output_path": str(provider_output),
            "task_id": observation.get("task_id") or provider.get("task_id"),
            "action": dict(action),
            "adapter_metadata": {
                "adapter_family": "unitree_lerobot_policy_provider_replay",
                "supported_action_types": list(SUPPORTED_ACTION_TYPES),
                "raw_token_values_returned": False,
            },
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
            "claim_boundary": _claim_boundary(
                policy_command_ran=True,
                provider_output_replay_used=True,
            ),
        },
        0,
    )


def _run_runner_command(
    *,
    command: str,
    payload: Mapping[str, Any],
    policy_path: str,
    source_root: str | None,
    repo_id: str,
    arm: str,
    end_effector: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="blueprint_unitree_lerobot_policy_") as tmp:
        temp_dir = Path(tmp)
        input_path = temp_dir / "policy_input.json"
        output_path = temp_dir / "policy_output.json"
        input_path.write_text(json.dumps(dict(payload), sort_keys=True) + "\n", encoding="utf-8")
        env = {
            **os.environ,
            "BLUEPRINT_POLICY_ACTION_INPUT": str(input_path),
            "BLUEPRINT_POLICY_ACTION_OUTPUT": str(output_path),
            "BLUEPRINT_UNITREE_LEROBOT_POLICY_INPUT": str(input_path),
            "BLUEPRINT_UNITREE_LEROBOT_POLICY_OUTPUT": str(output_path),
            "BLUEPRINT_UNITREE_LEROBOT_POLICY_PATH": policy_path,
            "BLUEPRINT_UNITREE_LEROBOT_REPO_ID": repo_id,
            "BLUEPRINT_UNITREE_LEROBOT_ARM": arm,
            "BLUEPRINT_UNITREE_LEROBOT_EE": end_effector,
        }
        if source_root:
            env["BLUEPRINT_UNITREE_LEROBOT_ROOT"] = source_root
        result = subprocess.run(
            shlex.split(command),
            input=json.dumps(dict(payload)),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
            env=env,
        )
        meta = {
            "command_exit_code": result.returncode,
            "stderr_size_bytes": len(result.stderr or ""),
            "stderr_omitted_to_avoid_secret_leakage": bool(result.stderr),
            "stdout_size_bytes": len(result.stdout or ""),
            "subprocess_spawned": True,
            "policy_output_file_used": output_path.is_file(),
        }
        if result.returncode != 0:
            raise RuntimeError(f"unitree_lerobot_policy_command_failed:{json.dumps(meta, sort_keys=True)}")
        if output_path.is_file():
            runner_payload = _load_json_file(output_path)
        else:
            runner_payload = json.loads(result.stdout or "{}")
            if not isinstance(runner_payload, Mapping):
                raise RuntimeError("unitree_lerobot_policy_stdout_not_json_object")
        return dict(runner_payload), meta


def run_unitree_lerobot_policy(
    *,
    payload: Mapping[str, Any],
    command: str | None,
    policy_path: str | None,
    source_root: Path | None,
    repo_id: str = DEFAULT_REPO_ID,
    arm: str = DEFAULT_ARM,
    end_effector: str = DEFAULT_EE,
    timeout_seconds: float = 30.0,
    provider_output: Path | None = None,
) -> tuple[dict[str, Any], int]:
    observation = _observation(payload)
    if provider_output is not None:
        return _provider_replay_payload(observation=observation, provider_output=provider_output)
    source_root_text = str(source_root) if source_root else None
    blockers: list[str] = []
    if not command:
        blockers.append(f"set_{RUNNER_COMMAND_ENV}_to_runnable_unitree_lerobot_policy_command")
    elif not _command_available(command):
        blockers.append("make_configured_unitree_lerobot_policy_command_executable_or_on_path")
    if not _policy_path_configured(policy_path):
        blockers.append(f"set_{POLICY_PATH_ENV}_to_trained_unitree_lerobot_policy_path_or_repo_id")
    if source_root is not None and not source_root.exists():
        blockers.append("blocked_unitree_lerobot_source_root_missing")
    if _camera_frame_path(observation) is None:
        blockers.append("blocked_missing_policy_visual_observation_frame")
    if blockers:
        return (
            _blocked_payload(
                blockers=blockers,
                observation=observation,
                command=command,
                policy_path=policy_path,
                source_root=source_root_text,
            ),
            2,
        )
    try:
        runner_payload, meta = _run_runner_command(
            command=command or "",
            payload={"observation": observation},
            policy_path=policy_path or "",
            source_root=source_root_text,
            repo_id=repo_id,
            arm=arm,
            end_effector=end_effector,
            timeout_seconds=timeout_seconds,
        )
        action = _normalize_runner_action(
            runner_payload=runner_payload,
            observation=observation,
            allow_task_fallback=False,
        )
        if not isinstance(action, Mapping):
            raise RuntimeError("unitree_lerobot_policy_response_missing_action")
    except Exception as exc:
        return (
            _blocked_payload(
                blockers=[f"blocked_unitree_lerobot_policy_command_failed:{type(exc).__name__}"],
                observation=observation,
                command=command,
                policy_path=policy_path,
                source_root=source_root_text,
            )
            | {"error": str(exc)[:500]},
            2,
        )
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "policy_id": POLICY_ID,
            "policy_kind": "unitree_lerobot_g1_manipulation_policy",
            "unitree_lerobot_policy_action_command_ran": True,
            "model_ran": True,
            "fresh_unitree_lerobot_model_executed_this_invocation": True,
            "task_id": observation.get("task_id"),
            "camera_frame_path": str(_camera_frame_path(observation)),
            "policy_path": policy_path,
            "source_root_path": source_root_text,
            "repo_id": repo_id,
            "arm": arm,
            "end_effector": end_effector,
            "action": dict(action),
            "runner_metadata": meta,
            "runner_response_redacted": _redact(runner_payload),
            "adapter_metadata": {
                "adapter_family": "unitree_lerobot_policy",
                "supported_action_types": list(SUPPORTED_ACTION_TYPES),
                "raw_token_values_returned": False,
            },
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
            "claim_boundary": _claim_boundary(policy_command_ran=True),
        },
        0,
    )


def adapter_manifest() -> dict[str, Any]:
    return {
        "schema_version": "policy_command_adapter_manifest.v1",
        "policy_id": POLICY_ID,
        "adapter_family": "unitree_lerobot_policy",
        "supported_action_types": list(SUPPORTED_ACTION_TYPES),
        "reads_json_from_stdin": True,
        "also_reads_BLUEPRINT_POLICY_ACTION_INPUT": True,
        "writes_json_to_stdout": True,
        "also_writes_BLUEPRINT_POLICY_ACTION_OUTPUT": True,
        "required_env": [
            RUNNER_COMMAND_ENV,
            POLICY_PATH_ENV,
        ],
        "optional_env": [
            SOURCE_ROOT_ENV,
            PROVIDER_OUTPUT_ENV,
            "BLUEPRINT_UNITREE_LEROBOT_REPO_ID",
            "BLUEPRINT_UNITREE_LEROBOT_ARM",
            "BLUEPRINT_UNITREE_LEROBOT_EE",
        ],
        "claim_boundary": _claim_boundary(policy_command_ran=False),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command")
    parser.add_argument("--policy-path")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--repo-id", default=os.getenv("BLUEPRINT_UNITREE_LEROBOT_REPO_ID", DEFAULT_REPO_ID))
    parser.add_argument("--arm", default=os.getenv("BLUEPRINT_UNITREE_LEROBOT_ARM", DEFAULT_ARM))
    parser.add_argument("--ee", default=os.getenv("BLUEPRINT_UNITREE_LEROBOT_EE", DEFAULT_EE))
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--provider-output", type=Path)
    parser.add_argument("--print-manifest", action="store_true")
    args = parser.parse_args(argv)
    if args.print_manifest:
        _write_payload(adapter_manifest())
        return 0
    command = args.command or os.getenv(RUNNER_COMMAND_ENV)
    policy_path = args.policy_path or os.getenv(POLICY_PATH_ENV)
    source_root = (
        args.source_root
        or (Path(os.getenv(SOURCE_ROOT_ENV, "")).expanduser() if os.getenv(SOURCE_ROOT_ENV) else None)
    )
    provider_output = args.provider_output or (
        Path(os.getenv(PROVIDER_OUTPUT_ENV, "")).expanduser()
        if os.getenv(PROVIDER_OUTPUT_ENV)
        else None
    )
    response, exit_code = run_unitree_lerobot_policy(
        payload=_read_payload(),
        command=command,
        policy_path=policy_path,
        source_root=source_root,
        repo_id=args.repo_id,
        arm=args.arm,
        end_effector=args.ee,
        timeout_seconds=args.timeout_seconds,
        provider_output=provider_output,
    )
    _write_payload(response)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
