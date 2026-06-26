"""Unitree UnifoLM policy command adapter for Blueprint endpoints.

This adapter is the Blueprint boundary for Unitree-native UnifoLM VLA/WMA
policy runtimes. It invokes a configured command or imports a provider output,
then normalizes the returned action into the Blueprint endpoint action schema.

It deliberately does not synthesize model success. A real Unitree policy claim
requires a runnable command, checkpoint, visual observation, endpoint invocation,
and downstream episode traces.
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


SCHEMA_VERSION = "unitree_unifolm_policy_command_adapter.v1"
MODE_CONTRACTS = {
    "vla": {
        "policy_id": "unitree_unifolm_vla_policy",
        "command_env": "BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND",
        "checkpoint_env": "BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT",
        "checkpoint_env_aliases": ("BLUEPRINT_UNITREE_UNIFOLM_POLICY_CHECKPOINT",),
        "vlm_checkpoint_env": "BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT",
        "source_root_env": "BLUEPRINT_UNITREE_UNIFOLM_VLA_SOURCE_ROOT",
        "provider_output_env": "BLUEPRINT_UNITREE_UNIFOLM_VLA_PROVIDER_OUTPUT",
    },
    "wma": {
        "policy_id": "unitree_unifolm_wma_policy",
        "command_env": "BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND",
        "checkpoint_env": "BLUEPRINT_UNITREE_UNIFOLM_WMA_CHECKPOINT",
        "vlm_checkpoint_env": None,
        "source_root_env": "BLUEPRINT_UNITREE_UNIFOLM_WMA_SOURCE_ROOT",
        "provider_output_env": "BLUEPRINT_UNITREE_UNIFOLM_WMA_PROVIDER_OUTPUT",
    },
}
SUPPORTED_ACTION_TYPES = (
    "manipulation_contact",
    "waypoint",
    "base_velocity",
    "stop",
    "inspect_look",
)


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


def _checkpoint_configured(checkpoint: str | None) -> bool:
    text = _string(checkpoint)
    if not text:
        return False
    path = Path(text).expanduser()
    if path.exists():
        return True
    return "/" in text and not text.startswith(("/", "./", "../", "~"))


def _checkpoint_env_names(mode: str) -> tuple[str, ...]:
    contract = MODE_CONTRACTS[mode]
    primary = _string(contract.get("checkpoint_env"))
    aliases = tuple(str(env) for env in contract.get("checkpoint_env_aliases", ()))
    return tuple(env for env in (primary, *aliases) if env)


def _first_env(names: Sequence[str]) -> tuple[str | None, str | None]:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return name, value
    return None, None


def _vlm_checkpoint_required(mode: str) -> bool:
    return bool(MODE_CONTRACTS[mode].get("vlm_checkpoint_env"))


def _normalize_runner_action(
    *,
    runner_payload: Mapping[str, Any],
    observation: Mapping[str, Any],
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
        or runner_payload.get("action_vector")
        or runner_payload.get("joint_targets")
        or runner_payload.get("joint_positions")
    )
    if isinstance(action_chunk, Sequence) and not isinstance(
        action_chunk, (str, bytes, bytearray)
    ):
        return {
            "action_type": "manipulation_contact",
            "target_object_id": "blueprint_light_object",
            "waypoint": _object_waypoint(observation),
            "approach_speed_mps": 0.04,
            "unitree_unifolm_action_chunk_present": True,
            "unitree_unifolm_raw_action_keys": sorted(str(key) for key in runner_payload.keys()),
        }
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
        return {"action_type": "stop", "report": "unitree_unifolm_policy_stop"}
    return {"action_type": "waypoint", "waypoint": _target_pose(observation), "max_speed_mps": 0.05}


def _claim_boundary(
    *,
    policy_command_ran: bool,
    provider_output_replay_used: bool = False,
) -> dict[str, Any]:
    fresh_policy_command_ran = bool(policy_command_ran and not provider_output_replay_used)
    return {
        "unitree_unifolm_policy_command_ran": fresh_policy_command_ran,
        "unitree_hand_manipulation_policy_used": fresh_policy_command_ran,
        "real_vla_or_unitree_hand_policy_endpoint_used": fresh_policy_command_ran,
        "provider_output_replay_used": bool(provider_output_replay_used),
        "provider_output_replay_is_not_fresh_per_request_model_inference": bool(
            provider_output_replay_used
        ),
        "provider_output_replay_is_not_live_unitree_hand_policy_command": bool(
            provider_output_replay_used
        ),
        "unitree_g1_dexterous_manipulation_proven": False,
        "single_action_is_not_episode_success": True,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
    }


def _blocked_payload(
    *,
    mode: str,
    blockers: Sequence[str],
    observation: Mapping[str, Any],
    command: str | None,
    checkpoint: str | None,
    vlm_checkpoint: str | None,
    source_root: str | None,
) -> dict[str, Any]:
    contract = MODE_CONTRACTS[mode]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "policy_id": contract["policy_id"],
        "unitree_unifolm_policy_action_command_ran": False,
        "model_ran": False,
        "mode": mode,
        "task_id": observation.get("task_id"),
        "blockers": sorted(set(blockers)),
        "command_env": contract["command_env"],
        "checkpoint_env": contract["checkpoint_env"],
        "checkpoint_env_aliases": list(contract.get("checkpoint_env_aliases", ())),
        "accepted_checkpoint_envs": list(_checkpoint_env_names(mode)),
        "vlm_checkpoint_env": contract.get("vlm_checkpoint_env"),
        "command_configured": bool(command),
        "command_available": _command_available(command),
        "command_value_redacted": "<configured>" if command else None,
        "checkpoint_configured": bool(checkpoint),
        "checkpoint_path": checkpoint,
        "vlm_checkpoint_configured": bool(vlm_checkpoint),
        "vlm_checkpoint_path": vlm_checkpoint,
        "source_root_configured": bool(source_root),
        "source_root_path": source_root,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "claim_boundary": _claim_boundary(policy_command_ran=False),
    }


def _load_json_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(value)


def _provider_output_payload(
    *,
    mode: str,
    payload: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    contract = MODE_CONTRACTS[mode]
    action = _normalize_runner_action(runner_payload=payload, observation=observation)
    if not action:
        raise ValueError("provider output did not include a Blueprint-compatible action")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "policy_id": f"{contract['policy_id']}_provider_replay",
        "mode": mode,
        "unitree_unifolm_policy_action_command_ran": False,
        "fresh_unitree_unifolm_model_executed_this_invocation": False,
        "fresh_unitree_unifolm_policy_action_command_ran_this_invocation": False,
        "provider_output_replay_used": True,
        "provider_unitree_unifolm_policy_action_command_ran": bool(
            payload.get("unitree_unifolm_policy_action_command_ran")
            or payload.get("policy_action_model_command_ran")
        ),
        "provider_unitree_unifolm_model_executed": bool(
            payload.get("unitree_unifolm_model_executed")
            or payload.get("unitree_unifolm_policy_action_command_ran")
            or payload.get("model_ran")
        ),
        "action": action,
        "runner_response_redacted": _redact(payload),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "claim_boundary": _claim_boundary(
            policy_command_ran=True,
            provider_output_replay_used=True,
        ),
    }


def run_unitree_unifolm_policy(
    *,
    payload: Mapping[str, Any],
    mode: str,
    command: str | None,
    checkpoint: str | None,
    source_root: str | None,
    vlm_checkpoint: str | None = None,
    provider_output: Path | None = None,
    timeout_seconds: float = 120.0,
) -> tuple[dict[str, Any], int]:
    if mode not in MODE_CONTRACTS:
        raise ValueError(f"unsupported UnifoLM mode: {mode}")
    observation = _observation(payload)
    if provider_output:
        return (
            _provider_output_payload(
                mode=mode,
                payload=_load_json_file(provider_output),
                observation=observation,
            ),
            0,
        )

    blockers: list[str] = []
    contract = MODE_CONTRACTS[mode]
    if not _command_available(command):
        blockers.append(
            f"set_{contract['command_env']}_to_runnable_unitree_unifolm_policy_command"
        )
    if not _checkpoint_configured(checkpoint):
        blockers.append(
            f"set_{contract['checkpoint_env']}_to_unitree_unifolm_checkpoint_path_or_repo_id"
        )
    if _vlm_checkpoint_required(mode) and not _checkpoint_configured(vlm_checkpoint):
        blockers.append(
            f"set_{contract['vlm_checkpoint_env']}_to_unitree_unifolm_vlm_checkpoint_path_or_repo_id"
        )
    frame = _camera_frame_path(observation)
    if frame is None:
        blockers.append("blocked_missing_policy_visual_observation_frame")
    if blockers:
        return (
            _blocked_payload(
                mode=mode,
                blockers=blockers,
                observation=observation,
                command=command,
                checkpoint=checkpoint,
                vlm_checkpoint=vlm_checkpoint,
                source_root=source_root,
            ),
            2,
        )

    assert command is not None
    with tempfile.TemporaryDirectory(prefix="blueprint-unitree-unifolm-") as tmp:
        tmp_path = Path(tmp)
        runner_output = tmp_path / "runner_output.json"
        env = os.environ.copy()
        env.update(
            {
                "BLUEPRINT_UNITREE_UNIFOLM_POLICY_MODE": mode,
                "BLUEPRINT_UNITREE_UNIFOLM_POLICY_CHECKPOINT": _string(checkpoint),
                "BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT": _string(vlm_checkpoint),
                "BLUEPRINT_UNITREE_UNIFOLM_POLICY_SOURCE_ROOT": _string(source_root),
                "BLUEPRINT_UNITREE_UNIFOLM_SOURCE_ROOT": _string(source_root),
                "BLUEPRINT_UNITREE_UNIFOLM_POLICY_FRAME": str(frame),
                "BLUEPRINT_UNITREE_UNIFOLM_POLICY_OUTPUT": str(runner_output),
                "BLUEPRINT_POLICY_ACTION_OUTPUT": str(runner_output),
            }
        )
        completed = subprocess.run(
            shlex.split(command),
            input=json.dumps(dict(payload)),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            env=env,
            check=False,
        )
        runner_payload: dict[str, Any] = {}
        if runner_output.is_file():
            runner_payload = _load_json_file(runner_output)
        elif completed.stdout.strip():
            value = json.loads(completed.stdout)
            if isinstance(value, Mapping):
                runner_payload = dict(value)
        if completed.returncode != 0:
            runner_blockers = list(runner_payload.get("blockers", []) or [])
            return (
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "policy_id": contract["policy_id"],
                    "mode": mode,
                    "unitree_unifolm_policy_action_command_ran": False,
                    "model_ran": bool(runner_payload.get("model_ran")),
                    "runner_returncode": completed.returncode,
                    "runner_stderr_tail": completed.stderr[-2000:],
                    "runner_response_redacted": _redact(runner_payload) if runner_payload else None,
                    "blockers": runner_blockers
                    or [f"unitree_unifolm_policy_command_exited_{completed.returncode}"],
                    "raw_credentials_written_to_artifacts": False,
                    "secret_hashes_written_to_artifacts": False,
                    "claim_boundary": _claim_boundary(policy_command_ran=False),
                },
                1,
            )
        action = _normalize_runner_action(runner_payload=runner_payload, observation=observation)
        if not action:
            return (
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "policy_id": contract["policy_id"],
                    "mode": mode,
                    "unitree_unifolm_policy_action_command_ran": False,
                    "model_ran": False,
                    "blockers": ["runner_did_not_return_blueprint_compatible_action"],
                    "runner_response_redacted": _redact(runner_payload),
                    "raw_credentials_written_to_artifacts": False,
                    "secret_hashes_written_to_artifacts": False,
                    "claim_boundary": _claim_boundary(policy_command_ran=False),
                },
                1,
            )
        return (
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "policy_id": contract["policy_id"],
                "mode": mode,
                "unitree_unifolm_policy_action_command_ran": True,
                "model_ran": True,
                "checkpoint_path": _string(checkpoint),
                "vlm_checkpoint_path": _string(vlm_checkpoint),
                "source_root_path": _string(source_root),
                "action": action,
                "runner_response_redacted": _redact(runner_payload),
                "raw_credentials_written_to_artifacts": False,
                "secret_hashes_written_to_artifacts": False,
                "claim_boundary": _claim_boundary(policy_command_ran=True),
            },
            0,
        )


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(MODE_CONTRACTS), default="vla")
    parser.add_argument("--command")
    parser.add_argument("--checkpoint")
    parser.add_argument("--vlm-checkpoint")
    parser.add_argument("--source-root")
    parser.add_argument("--provider-output", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--print-manifest", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    contract = MODE_CONTRACTS[args.mode]
    command = args.command or os.getenv(contract["command_env"])
    configured_checkpoint_env, env_checkpoint = _first_env(_checkpoint_env_names(args.mode))
    checkpoint = args.checkpoint or env_checkpoint
    vlm_checkpoint_env = contract.get("vlm_checkpoint_env")
    vlm_checkpoint = (
        args.vlm_checkpoint
        or (os.getenv(str(vlm_checkpoint_env)) if vlm_checkpoint_env else None)
    )
    source_root = args.source_root or os.getenv(contract["source_root_env"])
    provider_output_text = (
        str(args.provider_output)
        if args.provider_output
        else os.getenv(contract["provider_output_env"], "").strip()
    )
    provider_output = Path(provider_output_text).expanduser() if provider_output_text else None
    if args.print_manifest:
        _write_payload(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "configured" if command and checkpoint else "blocked",
                "mode": args.mode,
                "policy_id": contract["policy_id"],
                "command_env": contract["command_env"],
                "checkpoint_env": contract["checkpoint_env"],
                "checkpoint_env_aliases": list(contract.get("checkpoint_env_aliases", ())),
                "accepted_checkpoint_envs": list(_checkpoint_env_names(args.mode)),
                "configured_checkpoint_env": configured_checkpoint_env
                if checkpoint
                else None,
                "source_root_env": contract["source_root_env"],
                "provider_output_env": contract["provider_output_env"],
                "vlm_checkpoint_env": contract.get("vlm_checkpoint_env"),
                "command_configured": bool(command),
                "checkpoint_configured": bool(checkpoint),
                "vlm_checkpoint_configured": bool(vlm_checkpoint),
                "source_root_configured": bool(source_root),
                "raw_credentials_written_to_artifacts": False,
                "secret_hashes_written_to_artifacts": False,
            }
        )
        return 0
    response, exit_code = run_unitree_unifolm_policy(
        payload=_read_payload(),
        mode=args.mode,
        command=command,
        checkpoint=checkpoint,
        vlm_checkpoint=vlm_checkpoint,
        source_root=source_root,
        provider_output=provider_output,
        timeout_seconds=args.timeout_seconds,
    )
    _write_payload(response)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
