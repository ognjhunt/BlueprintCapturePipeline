"""OpenVLA command adapter for Blueprint policy endpoints.

The adapter reads a Blueprint observation packet, loads a configured OpenVLA
checkpoint through Hugging Face AutoClasses, predicts a raw VLA action from the
simulated egocentric frame plus task instruction, and decodes that action into
one of Blueprint's supported endpoint actions.

This is a command boundary, not a manipulation proof by itself. Generic
OpenVLA actions are not Unitree G1 hand commands unless a task/embodiment
decoder and fine-tuned checkpoint are configured and verified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


POLICY_ID = "openvla_policy"
SCHEMA_VERSION = "openvla_policy_command_adapter.v1"
DEFAULT_UNNORM_KEY = "bridge_orig"
PROVIDER_OUTPUT_ENV = "BLUEPRINT_OPENVLA_PROVIDER_OUTPUT"
SUPPORTED_ACTION_TYPES = (
    "waypoint",
    "base_velocity",
    "stop",
    "inspect_look",
    "manipulation_contact",
)
OPENVLA_EMPTY_TOKEN_ID = 29871


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


def _as_float_list(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result: list[float] = []
        for item in value:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                result.extend(_as_float_list(item))
            else:
                result.append(float(item))
        return result
    return [float(value)]


def _tensor_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(item) for item in shape]
    except Exception:
        return None


def _prepare_openvla_predict_action_inputs(inputs: Any, torch: Any) -> tuple[Any, dict[str, Any]]:
    """Keep OpenVLA's special prompt token aligned with the attention mask."""
    diagnostics: dict[str, Any] = {
        "input_ids_shape_before": _tensor_shape(inputs.get("input_ids")),
        "attention_mask_shape_before": _tensor_shape(inputs.get("attention_mask")),
        "pixel_values_shape": _tensor_shape(inputs.get("pixel_values")),
        "openvla_empty_token_id": OPENVLA_EMPTY_TOKEN_ID,
        "openvla_empty_token_appended_before_predict_action": False,
    }
    input_ids = inputs.get("input_ids")
    if input_ids is not None and getattr(input_ids, "shape", None) is not None and input_ids.shape[-1] > 0:
        try:
            last_token = int(input_ids[0, -1].item())
        except Exception:
            last_token = None
        diagnostics["input_ids_last_token_before"] = last_token
        if last_token != OPENVLA_EMPTY_TOKEN_ID:
            empty_token = torch.full(
                (input_ids.shape[0], 1),
                OPENVLA_EMPTY_TOKEN_ID,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            inputs["input_ids"] = torch.cat((input_ids, empty_token), dim=1)
            attention_mask = inputs.get("attention_mask")
            if (
                attention_mask is not None
                and getattr(attention_mask, "shape", None) is not None
                and attention_mask.shape[0] == input_ids.shape[0]
            ):
                attention_token = torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                inputs["attention_mask"] = torch.cat((attention_mask, attention_token), dim=1)
            diagnostics["openvla_empty_token_appended_before_predict_action"] = True
    diagnostics["input_ids_shape_after"] = _tensor_shape(inputs.get("input_ids"))
    diagnostics["attention_mask_shape_after"] = _tensor_shape(inputs.get("attention_mask"))
    return inputs, diagnostics


def _read_payload() -> dict[str, Any]:
    input_path = os.getenv("BLUEPRINT_POLICY_ACTION_INPUT", "").strip()
    if input_path:
        payload = json.loads(Path(input_path).expanduser().read_text(encoding="utf-8"))
    else:
        raw = sys.stdin.read().strip()
        if not raw:
            payload = {}
        else:
            payload = json.loads(raw)
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


def _task_prompt(observation: Mapping[str, Any]) -> str:
    prompt = str(observation.get("task_prompt") or "").strip()
    if not prompt:
        route = _mapping(observation.get("route_task_state"))
        prompt = str(route.get("task_prompt") or "").strip()
    return prompt or "complete the requested robot task safely"


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
    return _number(_mapping(observation.get("route_task_state")).get("target_error_m"), 99.0)


def _object_waypoint(observation: Mapping[str, Any], raw_action: Sequence[float]) -> list[float]:
    object_state = _mapping(observation.get("object_state"))
    position = object_state.get("position") or [0.36, -0.65, 0.27]
    x = _number(position[0], 0.36) if isinstance(position, Sequence) and len(position) > 0 else 0.36
    y = _number(position[1], -0.65) if isinstance(position, Sequence) and len(position) > 1 else -0.65
    dx = max(-0.08, min(0.08, float(raw_action[0]) * 0.05)) if raw_action else 0.0
    dy = max(-0.08, min(0.08, float(raw_action[1]) * 0.05)) if len(raw_action) > 1 else 0.0
    return [round(x + 0.18 + dx, 6), round(y + dy, 6), 0.79]


def decode_openvla_action(
    *,
    raw_action: Sequence[float],
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Conservatively map a generic OpenVLA action vector to Blueprint actions."""

    task_id = str(observation.get("task_id") or "")
    target = _target_pose(observation)
    target_error = _target_error(observation)
    if task_id == "inspect_target":
        return {"action_type": "inspect_look", "yaw_rate_rad_s": 0.25}
    if task_id == "stop_at_goal_and_report" and target_error <= 0.42:
        return {"action_type": "stop", "report": "openvla_decoder_goal_tolerance"}
    if task_id == "contact_or_push_light_object":
        return {
            "action_type": "manipulation_contact",
            "target_object_id": "blueprint_light_object",
            "waypoint": _object_waypoint(observation, raw_action),
            "approach_speed_mps": 0.06,
        }
    return {
        "action_type": "waypoint",
        "waypoint": target,
        "max_speed_mps": 0.08,
    }


def _blocked_payload(
    *,
    generated_blockers: Sequence[str],
    observation: Mapping[str, Any],
    checkpoint: str | None,
    source_root: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "policy_id": POLICY_ID,
        "openvla_policy_action_command_ran": False,
        "model_ran": False,
        "blockers": list(generated_blockers),
        "task_id": observation.get("task_id"),
        "checkpoint_configured": bool(checkpoint),
        "checkpoint_path": checkpoint,
        "source_root_configured": bool(source_root),
        "source_root_path": source_root,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "claim_boundary": _claim_boundary(model_ran=False),
    }


def _claim_boundary(
    *,
    model_ran: bool,
    provider_output_replay_used: bool = False,
) -> dict[str, Any]:
    return {
        "openvla_model_executed": bool(model_ran),
        "provider_output_replay_used": bool(provider_output_replay_used),
        "provider_output_replay_is_not_fresh_per_request_model_inference": bool(
            provider_output_replay_used
        ),
        "generic_openvla_action_vector_is_not_unitree_hand_control": True,
        "blueprint_action_decoder_is_conservative_task_mapping": True,
        "unitree_g1_dexterous_manipulation_proven": False,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
    }


def _load_provider_output(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("openvla provider output must be a JSON object")
    return dict(value)


def _provider_replay_payload(
    *,
    observation: Mapping[str, Any],
    provider_output: Path,
) -> tuple[dict[str, Any], int]:
    try:
        provider = _load_provider_output(provider_output)
    except Exception as exc:
        return (
            _blocked_payload(
                generated_blockers=[f"blocked_openvla_provider_output_load_failed:{type(exc).__name__}"],
                observation=observation,
                checkpoint=str(provider_output),
                source_root=None,
            )
            | {"provider_output_path": str(provider_output)},
            2,
        )
    action = provider.get("action") or provider.get("normalized_action")
    provider_schema_valid = str(provider.get("schema_version") or "").strip() == SCHEMA_VERSION
    provider_status_completed = str(provider.get("status") or "").strip() == "completed"
    provider_model_ran = bool(
        provider_schema_valid
        and provider_status_completed
        and (
            provider.get("openvla_model_executed") is True
            or provider.get("model_ran") is True
        )
    )
    provider_command_ran = bool(
        provider_model_ran
        and (
            provider.get("openvla_policy_action_command_ran") is True
            or provider.get("openvla_predict_action_invoked") is True
        )
    )
    blockers: list[str] = []
    if not provider_schema_valid:
        blockers.append("blocked_openvla_provider_output_schema_not_trusted")
    if provider_schema_valid and not provider_status_completed:
        blockers.append("blocked_openvla_provider_output_not_completed")
    if not provider_model_ran:
        blockers.append("blocked_openvla_provider_output_missing_model_execution_proof")
    if provider_model_ran and not provider_command_ran:
        blockers.append("blocked_openvla_provider_output_missing_predict_action_proof")
    if not isinstance(action, Mapping):
        blockers.append("blocked_openvla_provider_output_missing_action")
    if blockers:
        return (
            _blocked_payload(
                generated_blockers=blockers,
                observation=observation,
                checkpoint=str(provider_output),
                source_root=None,
            )
            | {
                "provider_output_path": str(provider_output),
                "provider_output_replay_used": True,
            },
            2,
        )
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "policy_id": "openvla_policy_provider_replay",
            "policy_kind": "openvla_policy_provider_replay",
            "openvla_policy_action_command_ran": True,
            "model_ran": True,
            "fresh_openvla_model_executed_this_invocation": False,
            "provider_output_replay_used": True,
            "provider_openvla_model_executed": provider_model_ran,
            "provider_openvla_policy_action_command_ran": provider_command_ran,
            "task_id": observation.get("task_id") or provider.get("task_id"),
            "provider_output_path": str(provider_output),
            "provider_policy_id": provider.get("policy_id"),
            "provider_model_repo_id": provider.get("model_repo_id"),
            "raw_openvla_action_vector": provider.get("raw_openvla_action_vector"),
            "action": dict(action),
            "adapter_metadata": {
                "adapter_family": "openvla_policy_provider_replay",
                "supported_action_types": list(SUPPORTED_ACTION_TYPES),
                "raw_token_values_returned": False,
            },
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
            "claim_boundary": _claim_boundary(
                model_ran=True,
                provider_output_replay_used=True,
            ),
        },
        0,
    )


def run_openvla_policy(
    *,
    payload: Mapping[str, Any],
    checkpoint: Path | None,
    source_root: Path | None,
    device: str | None,
    unnorm_key: str,
    allow_cpu: bool,
    provider_output: Path | None = None,
) -> tuple[dict[str, Any], int]:
    observation = _observation(payload)
    if provider_output is not None:
        return _provider_replay_payload(
            observation=observation,
            provider_output=provider_output,
        )
    blockers: list[str] = []
    if source_root is not None:
        if source_root.exists():
            sys.path.insert(0, str(source_root))
        else:
            blockers.append("blocked_openvla_source_root_missing")
    if checkpoint is None:
        blockers.append("blocked_missing_openvla_policy_checkpoint")
    elif not checkpoint.exists():
        blockers.append("blocked_openvla_policy_checkpoint_missing")
    frame_path = _camera_frame_path(observation)
    if frame_path is None:
        blockers.append("blocked_missing_policy_visual_observation_frame")
    if blockers:
        return (
            _blocked_payload(
                generated_blockers=blockers,
                observation=observation,
                checkpoint=str(checkpoint) if checkpoint else None,
                source_root=str(source_root) if source_root else None,
            ),
            2,
        )

    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except Exception as exc:
        return (
            _blocked_payload(
                generated_blockers=[f"blocked_openvla_runtime_import_failed:{type(exc).__name__}"],
                observation=observation,
                checkpoint=str(checkpoint) if checkpoint else None,
                source_root=str(source_root) if source_root else None,
            ),
            2,
        )

    selected_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if selected_device.startswith("cpu") and not allow_cpu:
        return (
            _blocked_payload(
                generated_blockers=["blocked_openvla_gpu_runtime_required"],
                observation=observation,
                checkpoint=str(checkpoint) if checkpoint else None,
                source_root=str(source_root) if source_root else None,
            ),
            2,
        )
    dtype = torch.bfloat16 if selected_device.startswith("cuda") else torch.float32
    prompt = f"In: What action should the robot take to {_task_prompt(observation)}?\nOut:"
    try:
        image = Image.open(frame_path).convert("RGB")
        processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True)
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        attn = os.getenv("BLUEPRINT_OPENVLA_ATTN_IMPLEMENTATION", "").strip()
        if attn:
            model_kwargs["attn_implementation"] = attn
        model = AutoModelForVision2Seq.from_pretrained(str(checkpoint), **model_kwargs)
        model = model.to(selected_device)
        model.eval()
        inputs = processor(prompt, image).to(selected_device, dtype=dtype)
        inputs, openvla_input_diagnostics = _prepare_openvla_predict_action_inputs(inputs, torch)
        with torch.no_grad():
            raw_action = model.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
            )
        raw_action_values = [round(value, 8) for value in _as_float_list(raw_action)]
        action = decode_openvla_action(raw_action=raw_action_values, observation=observation)
    except Exception as exc:
        return (
            _blocked_payload(
                generated_blockers=[f"blocked_openvla_policy_inference_failed:{type(exc).__name__}"],
                observation=observation,
                checkpoint=str(checkpoint) if checkpoint else None,
                source_root=str(source_root) if source_root else None,
            )
            | {
                "error": str(exc)[:500],
                "openvla_input_diagnostics": locals().get("openvla_input_diagnostics", {}),
            },
            2,
        )

    return (
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "policy_id": POLICY_ID,
            "policy_kind": "openvla_policy",
            "openvla_policy_action_command_ran": True,
            "model_ran": True,
            "task_id": observation.get("task_id"),
            "prompt": prompt,
            "camera_frame_path": str(frame_path),
            "checkpoint_path": str(checkpoint),
            "source_root_path": str(source_root) if source_root else None,
            "device": selected_device,
            "unnorm_key": unnorm_key,
            "raw_openvla_action_vector": raw_action_values,
            "openvla_input_diagnostics": openvla_input_diagnostics,
            "action": action,
            "adapter_metadata": {
                "adapter_family": "openvla_policy",
                "supported_action_types": list(SUPPORTED_ACTION_TYPES),
                "raw_token_values_returned": False,
            },
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
            "claim_boundary": _claim_boundary(model_ran=True),
        },
        0,
    )


def adapter_manifest() -> dict[str, Any]:
    return {
        "schema_version": "policy_command_adapter_manifest.v1",
        "policy_id": POLICY_ID,
        "adapter_family": "openvla_policy",
        "supported_action_types": list(SUPPORTED_ACTION_TYPES),
        "reads_json_from_stdin": True,
        "also_reads_BLUEPRINT_POLICY_ACTION_INPUT": True,
        "writes_json_to_stdout": True,
        "also_writes_BLUEPRINT_POLICY_ACTION_OUTPUT": True,
        "required_env": [
            "BLUEPRINT_OPENVLA_POLICY_CHECKPOINT",
            "BLUEPRINT_ALLOW_POLICY_ACTION_MODEL_COMMAND",
        ],
        "optional_env": [
            PROVIDER_OUTPUT_ENV,
            "BLUEPRINT_OPENVLA_POLICY_SOURCE_ROOT",
            "BLUEPRINT_OPENVLA_POLICY_DEVICE",
            "BLUEPRINT_OPENVLA_UNNORM_KEY",
            "BLUEPRINT_OPENVLA_ALLOW_CPU",
            "BLUEPRINT_OPENVLA_ATTN_IMPLEMENTATION",
        ],
        "claim_boundary": _claim_boundary(model_ran=False),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--provider-output", type=Path)
    parser.add_argument("--unnorm-key", default=os.getenv("BLUEPRINT_OPENVLA_UNNORM_KEY", DEFAULT_UNNORM_KEY))
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--print-manifest", action="store_true")
    args = parser.parse_args(argv)
    if args.print_manifest:
        _write_payload(adapter_manifest())
        return 0
    checkpoint_value = (
        args.checkpoint
        or (Path(os.getenv("BLUEPRINT_OPENVLA_POLICY_CHECKPOINT", "")).expanduser()
            if os.getenv("BLUEPRINT_OPENVLA_POLICY_CHECKPOINT")
            else None)
        or (Path(os.getenv("BLUEPRINT_POLICY_MODEL_CHECKPOINT", "")).expanduser()
            if os.getenv("BLUEPRINT_POLICY_MODEL_CHECKPOINT")
            else None)
    )
    source_root_value = (
        args.source_root
        or (Path(os.getenv("BLUEPRINT_OPENVLA_POLICY_SOURCE_ROOT", "")).expanduser()
            if os.getenv("BLUEPRINT_OPENVLA_POLICY_SOURCE_ROOT")
            else None)
    )
    payload = _read_payload()
    response, exit_code = run_openvla_policy(
        payload=payload,
        checkpoint=checkpoint_value,
        source_root=source_root_value,
        device=args.device or os.getenv("BLUEPRINT_OPENVLA_POLICY_DEVICE") or None,
        unnorm_key=args.unnorm_key,
        allow_cpu=args.allow_cpu
        or os.getenv("BLUEPRINT_OPENVLA_ALLOW_CPU", "").strip().lower()
        in {"1", "true", "yes", "y"},
        provider_output=args.provider_output
        or (Path(os.getenv(PROVIDER_OUTPUT_ENV, "")).expanduser()
            if os.getenv(PROVIDER_OUTPUT_ENV)
            else None),
    )
    _write_payload(response)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
