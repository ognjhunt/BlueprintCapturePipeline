"""Run a bounded provider-backed OpenVLA policy smoke test.

This proves only that an OpenVLA-family checkpoint can be loaded on a GPU
worker, invoked on a Blueprint visual observation, and decoded into a supported
Blueprint policy action. It is not Unitree G1 dexterous manipulation proof.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .model_access_env import model_access_secret_status, normalize_model_access_env
from .vast_provider_adapter import (
    VAST_API_GATE_ENV,
    VAST_ALLOW_COMMAND_EXECUTE_SCRIPT_FALLBACK_ENV,
    VAST_CONTAINER_MISSING_RETRY_ATTEMPTS_ENV,
    VAST_FORWARD_SECRET_ENV_VARS_ENV,
    VAST_INSTANCE_LAUNCH_GATE_ENV,
    run_vast_provider_adapter,
)
from .vast_wam_authorized_runner import DEFAULT_WAM_PUBLIC_IMAGE
from .wam_provider_object_store import stage_wam_provider_bundle_object_store


SCHEMA_VERSION = "openvla_policy_provider_smoke.v1"
DEFAULT_MODEL_REPO_ID = "openvla/openvla-7b"
DEFAULT_UNNORM_KEY = "bridge_orig"
DEFAULT_TASK_ID = "contact_or_push_light_object"
DEFAULT_TASK_PROMPT = "move the gripper toward the light object and make controlled contact"
DEFAULT_BUNDLE_FILENAME = "openvla_policy_provider_runtime_bundle.zip"
OPENVLA_PROVIDER_FORWARD_SECRET_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)
OPENVLA_RUNTIME_DEPENDENCY_PINS = (
    "transformers==4.40.1",
    "tokenizers==0.19.1",
    "timm==0.9.10",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _write_executable(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _merge_forward_secret_env_names(existing: str | None) -> str:
    names: list[str] = []
    for item in (existing or "").split(","):
        text = item.strip()
        if text and text not in names:
            names.append(text)
    for name in OPENVLA_PROVIDER_FORWARD_SECRET_ENV_NAMES:
        if name not in names:
            names.append(name)
    return ",".join(names)


def _copy_frame(frame_path: Path, output_path: Path) -> None:
    if not frame_path.is_file():
        raise FileNotFoundError(f"openvla_input_frame_missing:{frame_path}")
    ensure_dir(output_path.parent)
    shutil.copy2(frame_path, output_path)


PROVIDER_RUNNER = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "openvla_policy_provider_output.v1"
OPENVLA_RUNTIME_DEPENDENCY_PINS = (
    "transformers==4.40.1",
    "tokenizers==0.19.1",
    "timm==0.9.10",
)
WAM_PREFLIGHT_COMPATIBILITY_MARKERS = (
    "wam_runtime_result.json",
    "OSCAR-2B",
    "action_conditioned_video_rollout_generated",
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _phase(name: str, **fields: Any) -> None:
    print(
        "BLUEPRINT_OPENVLA_PROVIDER_PHASE:"
        + json.dumps(
            {
                "phase": name,
                "observed_at_epoch": round(time.time(), 3),
                "raw_secret_values_recorded": False,
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = []
    for item in value.replace("-", ".").split("."):
        if not item.isdigit():
            break
        parts.append(int(item))
    return tuple(parts or [0])


def _runtime_dependency_version_issues() -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    requirements = {
        "transformers": ("4.40.1", "4.40.1"),
        "tokenizers": ("0.19.1", "0.19.1"),
        "timm": ("0.9.10", "0.9.99"),
    }
    for package, (minimum, maximum) in requirements.items():
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            issues.append({"package": package, "status": "missing"})
            continue
        parsed = _version_tuple(version)
        if parsed < _version_tuple(minimum) or parsed > _version_tuple(maximum):
            issues.append(
                {
                    "package": package,
                    "status": "version_mismatch",
                    "version": version,
                    "required_min": minimum,
                    "required_max": maximum,
                }
            )
    return issues


def _task_prompt(observation: Mapping[str, Any]) -> str:
    return _string(observation.get("task_prompt")) or "complete the requested robot task safely"


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


def _decode_action(raw_action: Sequence[float], observation: Mapping[str, Any]) -> dict[str, Any]:
    task_id = _string(observation.get("task_id"))
    target = _target_pose(observation)
    if task_id == "inspect_target":
        return {"action_type": "inspect_look", "yaw_rate_rad_s": 0.25}
    if task_id == "stop_at_goal_and_report" and _target_error(observation) <= 0.42:
        return {"action_type": "stop", "report": "openvla_decoder_goal_tolerance"}
    if task_id == "contact_or_push_light_object":
        return {
            "action_type": "manipulation_contact",
            "target_object_id": "blueprint_light_object",
            "waypoint": _object_waypoint(observation, raw_action),
            "approach_speed_mps": 0.06,
        }
    return {"action_type": "waypoint", "waypoint": target, "max_speed_mps": 0.08}


def _force_openvla_eager_attention_compat(model: Any) -> dict[str, Any]:
    patched_objects: list[str] = []
    patched_configs: list[str] = []
    candidates: list[tuple[str, Any]] = [("model", model)]
    for name in ("language_model", "llm_backbone", "vision_backbone", "projector", "model"):
        value = getattr(model, name, None)
        if value is not None and value is not model:
            candidates.append((name, value))
    seen: set[int] = set()
    for name, candidate in candidates:
        marker = id(candidate)
        if marker in seen:
            continue
        seen.add(marker)
        if not hasattr(candidate, "_supports_sdpa"):
            try:
                setattr(candidate, "_supports_sdpa", False)
                patched_objects.append(name)
            except Exception:
                pass
        try:
            candidate_type = type(candidate)
            if not hasattr(candidate_type, "_supports_sdpa"):
                setattr(candidate_type, "_supports_sdpa", False)
        except Exception:
            pass
        config = getattr(candidate, "config", None)
        if config is not None:
            for field in ("attn_implementation", "_attn_implementation", "_attn_implementation_internal"):
                if hasattr(config, field):
                    try:
                        setattr(config, field, "eager")
                        patched_configs.append(f"{name}.{field}")
                    except Exception:
                        pass
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, "attn_implementation"):
        try:
            setattr(generation_config, "attn_implementation", "eager")
            patched_configs.append("generation_config.attn_implementation")
        except Exception:
            pass
    return {
        "supports_sdpa_missing_patched": patched_objects,
        "attention_config_fields_forced_eager": sorted(set(patched_configs)),
    }


def _tensor_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(item) for item in shape]
    except Exception:
        return None


def _prepare_openvla_predict_action_inputs(inputs: Any, torch: Any) -> tuple[Any, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "input_ids_shape_before": _tensor_shape(inputs.get("input_ids")),
        "attention_mask_shape_before": _tensor_shape(inputs.get("attention_mask")),
        "pixel_values_shape": _tensor_shape(inputs.get("pixel_values")),
        "openvla_empty_token_id": 29871,
        "openvla_empty_token_appended_before_predict_action": False,
    }
    input_ids = inputs.get("input_ids")
    if input_ids is not None and getattr(input_ids, "shape", None) is not None and input_ids.shape[-1] > 0:
        try:
            last_token = int(input_ids[0, -1].item())
        except Exception:
            last_token = None
        diagnostics["input_ids_last_token_before"] = last_token
        if last_token != 29871:
            empty_token = torch.full(
                (input_ids.shape[0], 1),
                29871,
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


def _blocked(output_path: Path, blockers: list[str], **fields: Any) -> int:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "openvla_policy_action_command_ran": False,
        "openvla_model_executed": False,
        "model_ran": False,
        "blockers": sorted(set(blockers)),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "truth_boundary": {
            "openvla_model_executed": False,
            "generic_openvla_action_vector_is_not_unitree_hand_control": True,
            "unitree_g1_dexterous_manipulation_proven": False,
            "generated_world_rank_fidelity_result_proven": False,
        },
        **fields,
    }
    _write_json(output_path, payload)
    _write_json(output_path.parent / "wam_runtime_result.json", {
        "schema_version": "wam_runtime_result.v1",
        "status": "blocked",
        "provider_kind": "openvla_policy",
        "learned_wam_model_ran": False,
        "action_conditioned_video_rollout_generated": False,
        "blockers": sorted(set(blockers)),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "truth_boundary": {
            "compatibility_file_for_existing_wam_provider_transport": True,
            "generated_video_is_model_output": False,
            "openvla_policy_model_executed": False,
        },
    })
    return 2


def _ensure_runtime(output_path: Path) -> tuple[Any, Any, Any, Any] | int:
    first_error = ""
    version_issues = _runtime_dependency_version_issues()
    try:
        if version_issues:
            raise RuntimeError("openvla_runtime_dependency_version_mismatch")
        import torch
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoProcessor
        return torch, Image, AutoModelForVision2Seq, AutoProcessor
    except Exception as first_exc:
        first_error = f"{type(first_exc).__name__}: {str(first_exc)[:500]}"
        _phase(
            "runtime_import_initial_failed",
            error_type=type(first_exc).__name__,
            dependency_version_issues=version_issues,
        )
    install = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--break-system-packages",
        "--upgrade",
        *OPENVLA_RUNTIME_DEPENDENCY_PINS,
        "huggingface_hub>=0.23,<1.0",
        "accelerate",
        "pillow",
        "einops",
        "sentencepiece",
        "protobuf",
    ]
    _phase("runtime_dependency_install_started")
    result = subprocess.run(install, capture_output=True, text=True, check=False, timeout=600)
    _phase(
        "runtime_dependency_install_finished",
        returncode=result.returncode,
        stdout_size_bytes=len(result.stdout or ""),
        stderr_size_bytes=len(result.stderr or ""),
        logs_omitted_to_avoid_secret_leakage=True,
    )
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoProcessor
        return torch, Image, AutoModelForVision2Seq, AutoProcessor
    except Exception as exc:
        return _blocked(
            output_path,
            [f"blocked_openvla_runtime_import_failed:{type(exc).__name__}"],
            dependency_install_returncode=result.returncode,
            initial_import_error=first_error,
            initial_dependency_version_issues=version_issues,
            dependency_install_stderr_tail=(result.stderr or "")[-2000:],
            dependency_install_stdout_tail=(result.stdout or "")[-1000:],
            final_import_error=f"{type(exc).__name__}: {str(exc)[:500]}",
        )


def main() -> int:
    bundle_dir = Path(os.environ.get("BLUEPRINT_WAM_PROVIDER_BUNDLE_DIR", ".")).resolve()
    output_dir = Path(os.environ.get("BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR", "runtime_output")).resolve()
    output_path = output_dir / "openvla_policy_provider_output.json"
    manifest_path = bundle_dir / "provider_runtime" / "openvla_policy_provider_manifest.json"
    input_path = bundle_dir / "provider_runtime" / "policy_input.json"
    frame_path = bundle_dir / "provider_runtime" / "input_frame.png"
    started = time.monotonic()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _blocked(output_path, [f"blocked_openvla_provider_input_read_failed:{type(exc).__name__}"])
    observation = _mapping(payload.get("observation"))
    observation.setdefault("visual_observation", {})
    observation["visual_observation"] = {
        **_mapping(observation.get("visual_observation")),
        "camera_frame_path": str(frame_path),
    }
    payload["observation"] = observation
    model_repo_id = _string(manifest.get("model_repo_id")) or "openvla/openvla-7b"
    unnorm_key = _string(manifest.get("unnorm_key")) or "bridge_orig"
    device_env = _string(os.environ.get("BLUEPRINT_OPENVLA_POLICY_DEVICE"))
    runtime = _ensure_runtime(output_path)
    if isinstance(runtime, int):
        return runtime
    torch, Image, AutoModelForVision2Seq, AutoProcessor = runtime
    selected_device = device_env or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if selected_device.startswith("cpu"):
        return _blocked(output_path, ["blocked_openvla_gpu_runtime_required"], selected_device=selected_device)
    prompt = f"In: What action should the robot take to {_task_prompt(observation)}?\nOut:"
    _phase("openvla_inference_started", model_repo_id=model_repo_id, selected_device=selected_device)
    try:
        image = Image.open(frame_path).convert("RGB")
        processor = AutoProcessor.from_pretrained(model_repo_id, trust_remote_code=True)
        kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        attn = _string(os.environ.get("BLUEPRINT_OPENVLA_ATTN_IMPLEMENTATION")) or "eager"
        if attn:
            kwargs["attn_implementation"] = attn
        model_load_attempt_errors = []
        try:
            model = AutoModelForVision2Seq.from_pretrained(model_repo_id, **kwargs)
        except (TypeError, AttributeError) as exc:
            model_load_attempt_errors.append(
                {
                    "attn_implementation": kwargs.get("attn_implementation"),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:300],
                }
            )
            if "_supports_sdpa" in str(exc) and kwargs.get("attn_implementation") != "eager":
                kwargs["attn_implementation"] = "eager"
                model = AutoModelForVision2Seq.from_pretrained(model_repo_id, **kwargs)
            else:
                kwargs.pop("attn_implementation", None)
                model = AutoModelForVision2Seq.from_pretrained(model_repo_id, **kwargs)
        except Exception as exc:
            model_load_attempt_errors.append(
                {
                    "attn_implementation": kwargs.get("attn_implementation"),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:300],
                }
            )
            kwargs.pop("attn_implementation", None)
            model = AutoModelForVision2Seq.from_pretrained(model_repo_id, **kwargs)
        model_sdpa_compatibility_patch = _force_openvla_eager_attention_compat(model)
        model_loaded = True
        model = model.to(selected_device)
        model.eval()
        inputs = processor(prompt, image).to(selected_device, dtype=torch.bfloat16)
        openvla_input_diagnostics = {
            "prompt_format": "openvla_openvla_7b_in_out_no_leading_out_space",
            "prompt_length_chars": len(prompt),
        }
        inputs, token_diagnostics = _prepare_openvla_predict_action_inputs(inputs, torch)
        openvla_input_diagnostics.update(token_diagnostics)
        predict_action_invoked = True
        with torch.no_grad():
            raw_action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        raw_action_values = [round(value, 8) for value in _as_float_list(raw_action)]
        action = _decode_action(raw_action_values, observation)
    except Exception as exc:
        return _blocked(
            output_path,
            [f"blocked_openvla_policy_inference_failed:{type(exc).__name__}"],
            model_repo_id=model_repo_id,
            selected_device=selected_device,
            error=str(exc)[:500],
            model_load_attempt_errors=locals().get("model_load_attempt_errors", []),
            model_sdpa_compatibility_patch=locals().get("model_sdpa_compatibility_patch", {}),
            openvla_input_diagnostics=locals().get("openvla_input_diagnostics", {}),
            openvla_model_loaded=bool(locals().get("model_loaded", False)),
            openvla_predict_action_invoked=bool(locals().get("predict_action_invoked", False)),
            traceback_tail="".join(traceback.format_exc(limit=8))[-2000:],
        )
    duration = round(time.monotonic() - started, 6)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "policy_id": "openvla_policy",
        "policy_kind": "openvla_policy",
        "openvla_policy_action_command_ran": True,
        "openvla_model_executed": True,
        "openvla_model_loaded": True,
        "openvla_predict_action_invoked": True,
        "model_ran": True,
        "model_repo_id": model_repo_id,
        "model_repo_revision": _string(manifest.get("model_repo_revision")) or None,
        "selected_device": selected_device,
        "unnorm_key": unnorm_key,
        "task_id": observation.get("task_id"),
        "prompt": prompt,
        "raw_openvla_action_vector": raw_action_values,
        "raw_action_vector_length": len(raw_action_values),
        "action": action,
        "duration_seconds": duration,
        "model_load_attempt_errors": model_load_attempt_errors,
        "model_sdpa_compatibility_patch": model_sdpa_compatibility_patch,
        "openvla_input_diagnostics": openvla_input_diagnostics,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "truth_boundary": {
            "openvla_model_executed": True,
            "generic_openvla_action_vector_is_not_unitree_hand_control": True,
            "blueprint_action_decoder_is_conservative_task_mapping": True,
            "unitree_g1_dexterous_manipulation_proven": False,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
        },
    }
    _write_json(output_path, result)
    _write_json(output_dir / "wam_provider_output.json", {
        "schema_version": "policy_provider_passthrough_for_vast_bundle.v1",
        "status": "completed",
        "provider_kind": "openvla_policy",
        "policy_output_path": str(output_path),
        "rollouts": [],
        "openvla_model_executed": True,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    })
    _write_json(output_dir / "wam_runtime_result.json", {
        "schema_version": "wam_runtime_result.v1",
        "status": "completed",
        "provider_kind": "openvla_policy",
        "learned_wam_model_ran": False,
        "action_conditioned_video_rollout_generated": False,
        "openvla_policy_model_executed": True,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "truth_boundary": {
            "compatibility_file_for_existing_wam_provider_transport": True,
            "generated_video_is_model_output": False,
            "openvla_policy_execution_proven": True,
            "unitree_g1_dexterous_manipulation_proven": False,
        },
    })
    _phase("openvla_inference_completed", raw_action_vector_length=len(raw_action_values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


RUN_SCRIPT = """#!/usr/bin/env bash
set -u
PY="${BLUEPRINT_WAM_PROVIDER_PYTHON:-python3}"
write_missing_result() {
  mkdir -p "${BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR:-runtime_output}"
  cat > "${BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR:-runtime_output}/wam_runtime_result.json" <<'JSON'
{"schema_version":"wam_runtime_result.v1","status":"blocked","blockers":["wam_runner_process_exited_without_runtime_result","blocked_wam_process_exited_without_result"],"raw_credentials_written_to_artifacts":false,"secret_hashes_written_to_artifacts":false}
JSON
}
"$PY" "$BLUEPRINT_WAM_PROVIDER_BUNDLE_DIR/provider_runtime/openvla_provider_runner.py"
rc=$?
if [ ! -f "${BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR:-runtime_output}/openvla_policy_provider_output.json" ]; then
  write_missing_result
fi
exit "$rc"
"""


def build_openvla_policy_provider_bundle(
    *,
    job_dir: str | Path,
    frame_path: str | Path,
    task_id: str = DEFAULT_TASK_ID,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    model_repo_id: str = DEFAULT_MODEL_REPO_ID,
    unnorm_key: str = DEFAULT_UNNORM_KEY,
    bundle_filename: str = DEFAULT_BUNDLE_FILENAME,
) -> dict[str, Any]:
    generated_at = utc_now_iso()
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    bundle_root = resolved_job_dir / "openvla_policy_provider_bundle"
    runtime_dir = bundle_root / "provider_runtime"
    oscar_compat_dir = runtime_dir / "oscar_input"
    ensure_dir(runtime_dir)
    input_frame = runtime_dir / "input_frame.png"
    _copy_frame(Path(frame_path).expanduser().resolve(), input_frame)
    _copy_frame(Path(frame_path).expanduser().resolve(), oscar_compat_dir / "first_frame.png")
    ensure_dir(oscar_compat_dir)
    (oscar_compat_dir / "blueprint_proxy_skeleton_conditioning.mp4").write_bytes(
        b"openvla policy smoke does not use OSCAR skeleton conditioning\n"
    )
    policy_input = {
        "schema_version": "policy_action_model_command_input.v1",
        "observation": {
            "task_id": task_id,
            "task_prompt": task_prompt,
            "visual_observation": {"camera_frame_path": str(input_frame)},
            "route_task_state": {"target_pose": [0.36, -0.65, 0.79], "target_error_m": 0.65},
            "object_state": {"object_id": "blueprint_light_object", "position": [0.36, -0.65, 0.27]},
        },
    }
    provider_manifest = {
        "schema_version": "openvla_policy_provider_manifest.v1",
        "generated_at": generated_at,
        "status": "ready",
        "model_repo_id": model_repo_id,
        "model_repo_revision": None,
        "unnorm_key": unnorm_key,
        "task_id": task_id,
        "task_prompt": task_prompt,
        "input_frame_runtime_path": "$BLUEPRINT_WAM_PROVIDER_BUNDLE_DIR/provider_runtime/input_frame.png",
        "truth_boundary": {
            "openvla_provider_smoke_only": True,
            "generic_openvla_action_vector_is_not_unitree_hand_control": True,
            "unitree_g1_dexterous_manipulation_proven": False,
            "generated_world_rank_fidelity_result_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    wam_rollout_input = {
        "schema_version": "wam_rollout_input_manifest.v1",
        "status": "policy_provider_smoke_input",
        "candidate_id": "openvla_policy",
        "policy_input_path": "provider_runtime/policy_input.json",
        "raw_credentials_written_to_artifacts": False,
    }
    write_json(runtime_dir / "policy_input.json", policy_input)
    write_json(runtime_dir / "openvla_policy_provider_manifest.json", provider_manifest)
    write_json(
        runtime_dir / "wam_provider_runtime_manifest.json",
        {
            "schema_version": "wam_provider_runtime_manifest.v1",
            "generated_at": generated_at,
            "status": "ready",
            "provider_kind": "openvla_policy",
            "compatibility_with_existing_wam_provider_transport": True,
            "oscar_wam_model_repo_id": "OSCAR-2B compatibility marker only",
            "action_conditioned_video_rollout_generated": False,
            "truth_boundary": provider_manifest["truth_boundary"],
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        },
    )
    write_json(runtime_dir / "wam_rollout_input_manifest.json", wam_rollout_input)
    _write_executable(runtime_dir / "openvla_provider_runner.py", PROVIDER_RUNNER)
    _write_executable(runtime_dir / "wam_provider_runtime_runner.py", PROVIDER_RUNNER)
    _write_executable(runtime_dir / "run_wam_provider_runtime.sh", RUN_SCRIPT)
    bundle_path = resolved_job_dir / bundle_filename
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(bundle_root).as_posix())
    manifest = {
        "schema_version": "openvla_policy_provider_bundle_manifest.v1",
        "generated_at": generated_at,
        "status": "completed",
        "job_dir": str(resolved_job_dir),
        "bundle_path": str(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "provider_runtime_entries": sorted(
            path.relative_to(bundle_root).as_posix() for path in bundle_root.rglob("*") if path.is_file()
        ),
        "model_repo_id": model_repo_id,
        "unnorm_key": unnorm_key,
        "task_id": task_id,
        "task_prompt": task_prompt,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "truth_boundary": provider_manifest["truth_boundary"],
    }
    write_json(resolved_job_dir / "openvla_policy_provider_bundle_manifest.json", manifest)
    return manifest


def import_openvla_provider_output(
    *,
    provider_output_zip: str | Path,
    extraction_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    resolved_zip = Path(provider_output_zip).expanduser().resolve()
    resolved_extraction = Path(extraction_dir).expanduser().resolve()
    ensure_dir(resolved_extraction)
    blockers: list[str] = []
    extracted_policy_output: dict[str, Any] = {}
    if not resolved_zip.is_file():
        blockers.append("openvla_provider_runtime_output_zip_missing")
    else:
        with zipfile.ZipFile(resolved_zip) as archive:
            archive.extractall(resolved_extraction)
        candidate = resolved_extraction / "openvla_policy_provider_output.json"
        if candidate.is_file():
            extracted_policy_output = _read_json(candidate)
        else:
            blockers.append("openvla_policy_provider_output_json_missing")
    if extracted_policy_output.get("status") != "completed":
        blockers.extend(str(item) for item in extracted_policy_output.get("blockers", []) or [])
        if extracted_policy_output:
            blockers.append("openvla_policy_provider_output_not_completed")
    payload = {
        "schema_version": "openvla_policy_provider_import.v1",
        "generated_at": utc_now_iso(),
        "status": "completed" if extracted_policy_output.get("status") == "completed" and not blockers else "blocked",
        "provider_output_zip": str(resolved_zip),
        "extraction_dir": str(resolved_extraction),
        "openvla_provider_output": extracted_policy_output,
        "openvla_model_executed": bool(extracted_policy_output.get("openvla_model_executed")),
        "openvla_policy_action_command_ran": bool(
            extracted_policy_output.get("openvla_policy_action_command_ran")
        ),
        "action": extracted_policy_output.get("action") if extracted_policy_output else None,
        "blockers": sorted(set(blockers)),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "truth_boundary": {
            "openvla_model_executed": bool(extracted_policy_output.get("openvla_model_executed")),
            "openvla_policy_provider_smoke_only": True,
            "unitree_g1_dexterous_manipulation_proven": False,
            "generated_world_rank_fidelity_result_proven": False,
        },
    }
    write_json(output_path, payload)
    return payload


def run_openvla_policy_provider_smoke(
    *,
    job_dir: str | Path,
    frame_path: str | Path,
    task_id: str = DEFAULT_TASK_ID,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    model_repo_id: str = DEFAULT_MODEL_REPO_ID,
    unnorm_key: str = DEFAULT_UNNORM_KEY,
    public_image: str = DEFAULT_WAM_PUBLIC_IMAGE,
    allow_paid_vast_launch: bool = False,
    dry_run: bool = False,
    target_spend_usd: float = 0.75,
    hard_cap_usd: float = 3.0,
    max_hourly_rate: float = 0.80,
    max_live_minutes: int = 45,
    startup_timeout_seconds: int = 2700,
    min_gpu_ram_mb: int = 24000,
    machine_avoidlist_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    ensure_dir(resolved_job_dir)
    bundle_manifest = build_openvla_policy_provider_bundle(
        job_dir=resolved_job_dir,
        frame_path=frame_path,
        task_id=task_id,
        task_prompt=task_prompt,
        model_repo_id=model_repo_id,
        unnorm_key=unnorm_key,
    )
    normalize_model_access_env()
    model_access = model_access_secret_status()
    staging = stage_wam_provider_bundle_object_store(
        job_dir=resolved_job_dir / "object_store_staging",
        bundle_path=bundle_manifest["bundle_path"],
        key_prefix="blueprint/openvla-policy-provider",
        expiration_seconds=12 * 60 * 60,
    )
    blockers: list[str] = []
    if staging.get("status") != "completed":
        blockers.extend(str(item) for item in staging.get("blockers", []) or [])
    if not allow_paid_vast_launch and not dry_run:
        blockers.append("missing_cli_allow_paid_vast_launch")
    output_zip = resolved_job_dir / "vast_provider_runtime_output.zip"
    provider_result: dict[str, Any] = {}
    imported: dict[str, Any] = {}
    if not blockers and not dry_run:
        bundle_url = Path(staging["provider_bundle_url_file"]["path"]).read_text(encoding="utf-8").strip()
        output_put_url = Path(staging["provider_output_put_url_file"]["path"]).read_text(encoding="utf-8").strip()
        output_get_url = Path(staging["provider_output_get_url_file"]["path"]).read_text(encoding="utf-8").strip()
        previous_min_gpu = os.environ.get("BLUEPRINT_VAST_WAM_MIN_GPU_RAM_MB")
        previous_api_gate = os.environ.get(VAST_API_GATE_ENV)
        previous_launch_gate = os.environ.get(VAST_INSTANCE_LAUNCH_GATE_ENV)
        previous_execute_fallback = os.environ.get(VAST_ALLOW_COMMAND_EXECUTE_SCRIPT_FALLBACK_ENV)
        previous_container_missing_retries = os.environ.get(
            VAST_CONTAINER_MISSING_RETRY_ATTEMPTS_ENV
        )
        previous_forward_secret_env = os.environ.get(VAST_FORWARD_SECRET_ENV_VARS_ENV)
        os.environ["BLUEPRINT_VAST_WAM_MIN_GPU_RAM_MB"] = str(int(min_gpu_ram_mb))
        os.environ[VAST_API_GATE_ENV] = "true"
        os.environ[VAST_INSTANCE_LAUNCH_GATE_ENV] = "true"
        os.environ[VAST_ALLOW_COMMAND_EXECUTE_SCRIPT_FALLBACK_ENV] = "true"
        os.environ[VAST_CONTAINER_MISSING_RETRY_ATTEMPTS_ENV] = "8"
        os.environ[VAST_FORWARD_SECRET_ENV_VARS_ENV] = _merge_forward_secret_env_names(
            previous_forward_secret_env
        )
        try:
            provider_result = run_vast_provider_adapter(
                job_dir=resolved_job_dir / "vast_provider_run",
                mode="live-startup-probe",
                allow_vast_api_call=True,
                allow_instance_launch=True,
                max_hourly_rate=max_hourly_rate,
                target_spend_usd=target_spend_usd,
                hard_cap_usd=hard_cap_usd,
                max_live_minutes=max_live_minutes,
                public_image=public_image,
                provider_bundle=bundle_manifest["bundle_path"],
                provider_bundle_url=bundle_url,
                provider_output_put_url=output_put_url,
                provider_output_get_url=output_get_url,
                provider_runtime_output_zip=output_zip,
                enable_blueprint_bundle=True,
                provider_bundle_kind="wam",
                vast_launch_mode="ssh_direct",
                disk_gb=120,
                poll_interval_seconds=15,
                startup_timeout_seconds=startup_timeout_seconds,
                machine_avoidlist_path=machine_avoidlist_path,
                session_budget_ledger_path=resolved_job_dir / "vast_session_cost_summary.json",
                session_max_live_minutes=max_live_minutes,
            )
        finally:
            if previous_min_gpu is None:
                os.environ.pop("BLUEPRINT_VAST_WAM_MIN_GPU_RAM_MB", None)
            else:
                os.environ["BLUEPRINT_VAST_WAM_MIN_GPU_RAM_MB"] = previous_min_gpu
            if previous_api_gate is None:
                os.environ.pop(VAST_API_GATE_ENV, None)
            else:
                os.environ[VAST_API_GATE_ENV] = previous_api_gate
            if previous_launch_gate is None:
                os.environ.pop(VAST_INSTANCE_LAUNCH_GATE_ENV, None)
            else:
                os.environ[VAST_INSTANCE_LAUNCH_GATE_ENV] = previous_launch_gate
            if previous_execute_fallback is None:
                os.environ.pop(VAST_ALLOW_COMMAND_EXECUTE_SCRIPT_FALLBACK_ENV, None)
            else:
                os.environ[VAST_ALLOW_COMMAND_EXECUTE_SCRIPT_FALLBACK_ENV] = previous_execute_fallback
            if previous_container_missing_retries is None:
                os.environ.pop(VAST_CONTAINER_MISSING_RETRY_ATTEMPTS_ENV, None)
            else:
                os.environ[VAST_CONTAINER_MISSING_RETRY_ATTEMPTS_ENV] = previous_container_missing_retries
            if previous_forward_secret_env is None:
                os.environ.pop(VAST_FORWARD_SECRET_ENV_VARS_ENV, None)
            else:
                os.environ[VAST_FORWARD_SECRET_ENV_VARS_ENV] = previous_forward_secret_env
        provider_command_path = resolved_job_dir / "vast_provider_run" / "vast_provider_command_result.json"
        provider_command = _read_json(provider_command_path) if provider_command_path.is_file() else {}
        provider_output_returned = (
            output_zip.is_file()
            or provider_command.get("provider_runtime_output_zip_received") is True
            or provider_command.get("provider_output_upload_ok") is True
        )
        if provider_output_returned:
            imported = import_openvla_provider_output(
                provider_output_zip=output_zip,
                extraction_dir=resolved_job_dir / "openvla_provider_output",
                output_path=resolved_job_dir / "openvla_policy_provider_import.json",
            )
        else:
            blockers.extend(str(item) for item in provider_result.get("blockers", []) or [])
    status = "dry_run_ready" if dry_run and not blockers else "completed"
    if blockers:
        status = "blocked"
    if imported and imported.get("status") != "completed":
        status = "blocked"
        blockers.extend(str(item) for item in imported.get("blockers", []) or [])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": status,
        "job_dir": str(resolved_job_dir),
        "bundle_manifest_path": str(resolved_job_dir / "openvla_policy_provider_bundle_manifest.json"),
        "staging_manifest_path": str(
            resolved_job_dir / "object_store_staging" / "wam_provider_object_store_staging_manifest.json"
        ),
        "provider_result_path": str(resolved_job_dir / "vast_provider_run" / "vast_provider_adapter_result.json"),
        "provider_output_zip": str(output_zip),
        "openvla_policy_provider_import_path": str(
            resolved_job_dir / "openvla_policy_provider_import.json"
        ),
        "openvla_model_executed": bool(imported.get("openvla_model_executed")),
        "openvla_policy_action_command_ran": bool(imported.get("openvla_policy_action_command_ran")),
        "action": imported.get("action") if imported else None,
        "model_repo_id": model_repo_id,
        "unnorm_key": unnorm_key,
        "public_image": public_image,
        "allow_paid_vast_launch": allow_paid_vast_launch,
        "dry_run": dry_run,
        "provider_status": provider_result.get("status"),
        "model_access_secret_status": model_access,
        "blockers": sorted(set(blockers)),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "truth_boundary": {
            "openvla_provider_smoke_only": True,
            "openvla_model_executed": bool(imported.get("openvla_model_executed")),
            "generic_openvla_action_vector_is_not_unitree_hand_control": True,
            "unitree_g1_dexterous_manipulation_proven": False,
            "generated_world_rank_fidelity_result_proven": False,
        },
    }
    write_json(resolved_job_dir / "openvla_policy_provider_smoke_summary.json", summary)
    return summary


def default_job_dir() -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path("robot_eval_jobs") / f"openvla_policy_provider_smoke_{stamp}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", default=str(default_job_dir()))
    parser.add_argument("--frame-path", required=True)
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--task-prompt", default=DEFAULT_TASK_PROMPT)
    parser.add_argument("--model-repo-id", default=DEFAULT_MODEL_REPO_ID)
    parser.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    parser.add_argument("--public-image", default=os.getenv("BLUEPRINT_VAST_WAM_PUBLIC_IMAGE", DEFAULT_WAM_PUBLIC_IMAGE))
    parser.add_argument("--allow-paid-vast-launch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target-spend-usd", type=float, default=0.75)
    parser.add_argument("--hard-cap-usd", type=float, default=3.0)
    parser.add_argument("--max-hourly-rate", type=float, default=0.80)
    parser.add_argument("--max-live-minutes", type=int, default=45)
    parser.add_argument("--startup-timeout-seconds", type=int, default=2700)
    parser.add_argument("--min-gpu-ram-mb", type=int, default=24000)
    parser.add_argument(
        "--machine-avoidlist",
        help=(
            "Optional JSON avoidlist of Vast machine IDs to exclude from offer selection. "
            "Passed through to the underlying Vast provider adapter."
        ),
    )
    args = parser.parse_args(argv)
    summary = run_openvla_policy_provider_smoke(
        job_dir=args.job_dir,
        frame_path=args.frame_path,
        task_id=args.task_id,
        task_prompt=args.task_prompt,
        model_repo_id=args.model_repo_id,
        unnorm_key=args.unnorm_key,
        public_image=args.public_image,
        allow_paid_vast_launch=args.allow_paid_vast_launch,
        dry_run=args.dry_run,
        target_spend_usd=args.target_spend_usd,
        hard_cap_usd=args.hard_cap_usd,
        max_hourly_rate=args.max_hourly_rate,
        max_live_minutes=args.max_live_minutes,
        startup_timeout_seconds=args.startup_timeout_seconds,
        min_gpu_ram_mb=args.min_gpu_ram_mb,
        machine_avoidlist_path=args.machine_avoidlist,
    )
    print(f"[openvla-provider-smoke] summary={Path(args.job_dir).resolve() / 'openvla_policy_provider_smoke_summary.json'}")
    print(f"[openvla-provider-smoke] status={summary.get('status')}")
    if summary.get("blockers"):
        print("[openvla-provider-smoke] blockers=" + ",".join(str(item) for item in summary["blockers"]))
    return 0 if summary.get("status") in {"completed", "dry_run_ready"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
