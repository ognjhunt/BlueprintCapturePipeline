"""GR00T N1.7 + UNITREE_G1_SONIC policy command adapter.

The adapter reads a Blueprint observation packet, invokes a configured
GR00T/SONIC runner command, and normalizes returned action chunks into the
Blueprint policy-action contract. It deliberately does not synthesize an action
from the task when the runner is missing or returns no action payload.
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

from .unitree_groot_n17_sonic_policy_runtime import (
    GROOT_ROOT_ENV,
    N17_CHECKPOINT_ENV,
    POLICY_COMMAND_ENV,
    POLICY_ID,
    POLICY_SERVER_URL_ENV,
    SIM2SIM_COMMAND_ENV,
    SONIC_CHECKPOINT_ENV,
    WBC_ROOT_ENV,
    configured_checkpoint_reference,
)


SCHEMA_VERSION = "unitree_groot_n17_sonic_policy_command_adapter.v1"
PROVIDER_OUTPUT_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT"
SUPPORTED_ACTION_KEYS = (
    "action",
    "normalized_action",
    "action_chunk",
    "actions",
    "action_vector",
    "sonic_latent_action",
    "sonic_latents",
    "latent_action_tokens",
    "joint_targets",
    "joint_positions",
    "arm_targets",
    "hand_targets",
    "gripper_targets",
    "motion_token",
    "left_hand_joints",
    "right_hand_joints",
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


def _command_available(command: str | None) -> bool:
    text = _string(command)
    if not text:
        return False
    try:
        parts = shlex.split(text)
    except ValueError:
        return False
    if not parts:
        return False
    executable = parts[0]
    return bool(shutil.which(executable) or Path(executable).expanduser().is_file())


def _checkpoint_configured(value: str | None) -> bool:
    configured, _, _, _ = configured_checkpoint_reference(value)
    return configured


def _normalize_sequence_action(key: str, value: Any) -> dict[str, Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {
            "action_type": "unitree_g1_sonic_action_chunk",
            "action_chunk": list(value),
            "unitree_groot_n17_sonic_action_chunk_present": True,
            "unitree_g1_sonic_control_fields": [key],
            "source_action_key": key,
        }
    return None


def _normalize_runner_action(runner_payload: Mapping[str, Any]) -> dict[str, Any] | None:
    action = runner_payload.get("action") or runner_payload.get("normalized_action")
    if isinstance(action, Mapping):
        normalized = dict(action)
        normalized.setdefault("unitree_groot_n17_sonic_action_payload_present", True)
        return normalized
    for key in (
        "action_chunk",
        "actions",
        "action_vector",
        "sonic_latent_action",
        "sonic_latents",
        "latent_action_tokens",
        "joint_positions",
    ):
        normalized = _normalize_sequence_action(key, runner_payload.get(key))
        if normalized is not None:
            return normalized
    control_fields = {
        key: runner_payload[key]
        for key in (
            "joint_targets",
            "arm_targets",
            "hand_targets",
            "gripper_targets",
            "motion_token",
            "left_hand_joints",
            "right_hand_joints",
        )
        if key in runner_payload
    }
    if control_fields:
        return {
            "action_type": "unitree_g1_sonic_control_targets",
            **control_fields,
            "unitree_groot_n17_sonic_action_chunk_present": True,
            "unitree_g1_sonic_control_fields": sorted(control_fields),
        }
    return None


def _claim_boundary(
    *,
    policy_command_ran: bool,
    provider_output_replay_used: bool = False,
) -> dict[str, Any]:
    return {
        "unitree_groot_n17_sonic_policy_command_ran": bool(policy_command_ran),
        "unitree_groot_n17_sonic_policy_action_command_ran": bool(policy_command_ran),
        "unitree_hand_manipulation_policy_used": bool(policy_command_ran),
        "unitree_g1_sonic_action_command_is_single_step_not_task_success": True,
        "provider_output_replay_used": bool(provider_output_replay_used),
        "provider_output_replay_is_not_fresh_per_request_model_inference": bool(
            provider_output_replay_used
        ),
        "unitree_g1_dexterous_manipulation_proven": False,
        "physical_robot_readiness_proven": False,
        "deployment_readiness_proven": False,
        "safety_validation_proven": False,
        "real_world_manipulation_success_proven": False,
    }


def _blocked_payload(
    *,
    blockers: Sequence[str],
    observation: Mapping[str, Any],
    command: str | None,
    n17_checkpoint: str | None,
    sonic_checkpoint: str | None,
    groot_root: str | None,
    wbc_root: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "policy_id": POLICY_ID,
        "selected_candidate_id": POLICY_ID,
        "unitree_groot_n17_sonic_policy_configured": False,
        "unitree_groot_n17_sonic_policy_action_command_ran": False,
        "unitree_policy_action_command_ran": False,
        "unitree_specific_manipulation_candidate_ran": False,
        "openvla_policy_action_command_ran": False,
        "model_ran": False,
        "task_id": observation.get("task_id"),
        "blockers": sorted(set(blockers)),
        "command_env": POLICY_COMMAND_ENV,
        "command_configured": bool(command),
        "command_available": _command_available(command),
        "command_value_redacted": "<configured>" if command else None,
        "n17_checkpoint_env": N17_CHECKPOINT_ENV,
        "n17_checkpoint_configured": bool(n17_checkpoint),
        "n17_checkpoint_path": n17_checkpoint,
        "g1_sonic_checkpoint_env": SONIC_CHECKPOINT_ENV,
        "g1_sonic_checkpoint_configured": bool(sonic_checkpoint),
        "g1_sonic_checkpoint_path": sonic_checkpoint,
        "groot_root_env": GROOT_ROOT_ENV,
        "groot_root_configured": bool(groot_root),
        "groot_root_path": groot_root,
        "wbc_root_env": WBC_ROOT_ENV,
        "wbc_root_configured": bool(wbc_root),
        "wbc_root_path": wbc_root,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "claim_boundary": _claim_boundary(policy_command_ran=False),
    }


def _load_json_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
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
                blockers=[f"blocked_unitree_groot_n17_sonic_provider_output_load_failed:{type(exc).__name__}"],
                observation=observation,
                command=None,
                n17_checkpoint=str(provider_output),
                sonic_checkpoint=str(provider_output),
                groot_root=None,
                wbc_root=None,
            )
            | {"provider_output_path": str(provider_output), "provider_output_replay_used": True},
            2,
        )
    action = _normalize_runner_action(provider)
    completed = bool(
        provider.get("status") == "completed"
        and provider.get("unitree_groot_n17_sonic_policy_action_command_ran") is True
        and isinstance(action, Mapping)
    )
    if not completed:
        return (
            _blocked_payload(
                blockers=[
                    "blocked_unitree_groot_n17_sonic_provider_output_not_fresh_action_command"
                ],
                observation=observation,
                command=None,
                n17_checkpoint=str(provider_output),
                sonic_checkpoint=str(provider_output),
                groot_root=None,
                wbc_root=None,
            )
            | {"provider_output_path": str(provider_output), "provider_output_replay_used": True},
            2,
        )
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "policy_id": f"{POLICY_ID}_provider_replay",
            "selected_candidate_id": POLICY_ID,
            "unitree_groot_n17_sonic_policy_configured": True,
            "unitree_groot_n17_sonic_policy_action_command_ran": False,
            "unitree_policy_action_command_ran": False,
            "unitree_specific_manipulation_candidate_ran": False,
            "openvla_policy_action_command_ran": False,
            "model_ran": False,
            "fresh_unitree_groot_n17_sonic_model_executed_this_invocation": False,
            "provider_output_replay_used": True,
            "provider_output_path": str(provider_output),
            "task_id": observation.get("task_id") or provider.get("task_id"),
            "action": action,
            "runner_response_redacted": _redact(provider),
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
            "claim_boundary": _claim_boundary(
                policy_command_ran=False,
                provider_output_replay_used=True,
            ),
        },
        0,
    )


def _run_runner_command(
    *,
    command: str,
    payload: Mapping[str, Any],
    n17_checkpoint: str,
    sonic_checkpoint: str,
    groot_root: str | None,
    wbc_root: str | None,
    policy_server_url: str | None,
    sim2sim_command: str | None,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="blueprint-unitree-groot-sonic-") as tmp:
        temp_dir = Path(tmp)
        input_path = temp_dir / "policy_input.json"
        output_path = temp_dir / "policy_output.json"
        input_path.write_text(json.dumps(dict(payload), sort_keys=True) + "\n", encoding="utf-8")
        env = {
            **os.environ,
            "BLUEPRINT_POLICY_ACTION_INPUT": str(input_path),
            "BLUEPRINT_POLICY_ACTION_OUTPUT": str(output_path),
            N17_CHECKPOINT_ENV: n17_checkpoint,
            SONIC_CHECKPOINT_ENV: sonic_checkpoint,
            "BLUEPRINT_UNITREE_GROOT_N17_SONIC_EMBODIMENT_TAG": "UNITREE_G1_SONIC",
        }
        if groot_root:
            env[GROOT_ROOT_ENV] = groot_root
        if wbc_root:
            env[WBC_ROOT_ENV] = wbc_root
        if policy_server_url:
            env[POLICY_SERVER_URL_ENV] = policy_server_url
        if sim2sim_command:
            env[SIM2SIM_COMMAND_ENV] = sim2sim_command
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
        if output_path.is_file():
            runner_payload = _load_json_file(output_path)
        else:
            value = json.loads(result.stdout or "{}")
            if not isinstance(value, Mapping):
                raise RuntimeError("unitree_groot_n17_sonic_policy_stdout_not_json_object")
            runner_payload = dict(value)
        if result.returncode != 0 and not runner_payload:
            raise RuntimeError(
                f"unitree_groot_n17_sonic_policy_command_failed:{json.dumps(meta, sort_keys=True)}"
            )
        return runner_payload, meta


def run_unitree_groot_n17_sonic_policy(
    *,
    payload: Mapping[str, Any],
    command: str | None,
    n17_checkpoint: str | None,
    sonic_checkpoint: str | None,
    groot_root: str | None = None,
    wbc_root: str | None = None,
    policy_server_url: str | None = None,
    sim2sim_command: str | None = None,
    timeout_seconds: float = 120.0,
    provider_output: Path | None = None,
) -> tuple[dict[str, Any], int]:
    observation = _observation(payload)
    if provider_output is not None:
        return _provider_replay_payload(observation=observation, provider_output=provider_output)

    blockers: list[str] = []
    if not command:
        blockers.append(f"set_{POLICY_COMMAND_ENV}_to_runnable_unitree_groot_n17_sonic_policy_command")
    elif not _command_available(command):
        blockers.append("make_configured_unitree_groot_n17_sonic_policy_command_executable_or_on_path")
    if not _checkpoint_configured(n17_checkpoint):
        blockers.append(f"set_{N17_CHECKPOINT_ENV}_to_nvidia_groot_n17_checkpoint_or_repo_id")
    if not _checkpoint_configured(sonic_checkpoint):
        blockers.append(f"set_{SONIC_CHECKPOINT_ENV}_to_finetuned_unitree_g1_sonic_checkpoint_or_repo_id")
    frame = _camera_frame_path(observation)
    if frame is None:
        blockers.append("blocked_missing_policy_visual_observation_frame")
    if blockers:
        return (
            _blocked_payload(
                blockers=blockers,
                observation=observation,
                command=command,
                n17_checkpoint=n17_checkpoint,
                sonic_checkpoint=sonic_checkpoint,
                groot_root=groot_root,
                wbc_root=wbc_root,
            ),
            2,
        )
    try:
        runner_payload, meta = _run_runner_command(
            command=command or "",
            payload={"observation": observation},
            n17_checkpoint=n17_checkpoint or "",
            sonic_checkpoint=sonic_checkpoint or "",
            groot_root=groot_root,
            wbc_root=wbc_root,
            policy_server_url=policy_server_url,
            sim2sim_command=sim2sim_command,
            timeout_seconds=timeout_seconds,
        )
        if runner_payload.get("status") == "blocked" or runner_payload.get("blockers"):
            blockers = [
                _string(item)
                for item in runner_payload.get("blockers", [])
                if _string(item)
            ] or ["blocked_unitree_groot_n17_sonic_policy_command_returned_blocked"]
            return (
                _blocked_payload(
                    blockers=blockers,
                    observation=observation,
                    command=command,
                    n17_checkpoint=n17_checkpoint,
                    sonic_checkpoint=sonic_checkpoint,
                    groot_root=groot_root,
                    wbc_root=wbc_root,
                )
                | {
                    "runner_metadata": meta,
                    "runner_response_redacted": _redact(runner_payload),
                    "child_command_blocked": True,
                },
                2,
            )
        action = _normalize_runner_action(runner_payload)
        if not isinstance(action, Mapping):
            raise RuntimeError("unitree_groot_n17_sonic_policy_response_missing_action")
    except Exception as exc:
        return (
            _blocked_payload(
                blockers=[f"blocked_unitree_groot_n17_sonic_policy_command_failed:{type(exc).__name__}"],
                observation=observation,
                command=command,
                n17_checkpoint=n17_checkpoint,
                sonic_checkpoint=sonic_checkpoint,
                groot_root=groot_root,
                wbc_root=wbc_root,
            )
            | {"error_type": type(exc).__name__},
            2,
        )
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "policy_id": POLICY_ID,
            "selected_candidate_id": POLICY_ID,
            "policy_kind": "unitree_groot_n17_sonic_manipulation_policy",
            "unitree_groot_n17_sonic_policy_configured": True,
            "unitree_groot_n17_sonic_policy_action_command_ran": True,
            "unitree_policy_action_command_ran": True,
            "unitree_specific_manipulation_candidate_ran": True,
            "openvla_policy_action_command_ran": False,
            "model_ran": True,
            "fresh_unitree_groot_n17_sonic_model_executed_this_invocation": True,
            "task_id": observation.get("task_id"),
            "camera_frame_path": str(frame),
            "n17_checkpoint_path": n17_checkpoint,
            "g1_sonic_checkpoint_path": sonic_checkpoint,
            "groot_root_path": groot_root,
            "wbc_root_path": wbc_root,
            "policy_server_url_configured": bool(policy_server_url),
            "sim2sim_command_configured": bool(sim2sim_command),
            "action": dict(action),
            "runner_metadata": meta,
            "runner_response_redacted": _redact(runner_payload),
            "adapter_metadata": {
                "adapter_family": POLICY_ID,
                "embodiment_tag": "UNITREE_G1_SONIC",
                "supported_action_keys": list(SUPPORTED_ACTION_KEYS),
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
        "adapter_family": POLICY_ID,
        "supported_action_keys": list(SUPPORTED_ACTION_KEYS),
        "reads_json_from_stdin": True,
        "also_reads_BLUEPRINT_POLICY_ACTION_INPUT": True,
        "writes_json_to_stdout": True,
        "also_writes_BLUEPRINT_POLICY_ACTION_OUTPUT": True,
        "required_env": [
            POLICY_COMMAND_ENV,
            N17_CHECKPOINT_ENV,
            SONIC_CHECKPOINT_ENV,
        ],
        "optional_env": [
            GROOT_ROOT_ENV,
            WBC_ROOT_ENV,
            POLICY_SERVER_URL_ENV,
            SIM2SIM_COMMAND_ENV,
            PROVIDER_OUTPUT_ENV,
        ],
        "claim_boundary": _claim_boundary(policy_command_ran=False),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command")
    parser.add_argument("--n17-checkpoint")
    parser.add_argument("--sonic-checkpoint")
    parser.add_argument("--groot-root")
    parser.add_argument("--wbc-root")
    parser.add_argument("--policy-server-url")
    parser.add_argument("--sim2sim-command")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--provider-output", type=Path)
    parser.add_argument("--print-manifest", action="store_true")
    args = parser.parse_args(argv)
    if args.print_manifest:
        _write_payload(adapter_manifest())
        return 0
    provider_output = args.provider_output or (
        Path(os.getenv(PROVIDER_OUTPUT_ENV, "")).expanduser()
        if os.getenv(PROVIDER_OUTPUT_ENV)
        else None
    )
    response, exit_code = run_unitree_groot_n17_sonic_policy(
        payload=_read_payload(),
        command=args.command or os.getenv(POLICY_COMMAND_ENV),
        n17_checkpoint=args.n17_checkpoint or os.getenv(N17_CHECKPOINT_ENV),
        sonic_checkpoint=args.sonic_checkpoint or os.getenv(SONIC_CHECKPOINT_ENV),
        groot_root=args.groot_root or os.getenv(GROOT_ROOT_ENV),
        wbc_root=args.wbc_root or os.getenv(WBC_ROOT_ENV),
        policy_server_url=args.policy_server_url or os.getenv(POLICY_SERVER_URL_ENV),
        sim2sim_command=args.sim2sim_command or os.getenv(SIM2SIM_COMMAND_ENV),
        timeout_seconds=args.timeout_seconds,
        provider_output=provider_output,
    )
    _write_payload(response)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
