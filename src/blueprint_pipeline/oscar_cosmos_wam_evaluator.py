"""SC3/OSCAR-style WAM evaluator over MuJoCo Unitree G1 endpoint traces.

The evaluator prepares action-conditioned WAM rollout inputs from MuJoCo traces
and review videos. It only runs a learned model when an explicit local command,
checkpoint, and opt-in gate are present; otherwise it writes blocked dry-run
artifacts without faking generated video or success labels.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .model_access_env import model_access_secret_status, normalize_model_access_env


WAM_EVALUATOR_SCHEMA_VERSION = "oscar_cosmos_wam_evaluator.v1"
DEFAULT_MODEL_CANDIDATES = ("oscar_wam", "cosmos_wam")
LOCAL_MODEL_GATE_ENV = "BLUEPRINT_ALLOW_LOCAL_WAM_MODEL"

MODEL_RUNTIME_CONTRACTS = {
    "oscar_wam": {
        "command_envs": ("BLUEPRINT_OSCAR_WAM_COMMAND", "BLUEPRINT_OSCAR_WAM_PROVIDER_COMMAND"),
        "checkpoint_envs": ("BLUEPRINT_OSCAR_WAM_CHECKPOINT",),
        "runtime_role": "action_conditioned_world_model_evaluator",
    },
    "cosmos_wam": {
        "command_envs": ("BLUEPRINT_COSMOS_WAM_COMMAND", "BLUEPRINT_COSMOS_WAM_PROVIDER_COMMAND"),
        "checkpoint_envs": ("BLUEPRINT_COSMOS_WAM_CHECKPOINT",),
        "runtime_role": "world_video_rollout_or_review_substrate",
    },
    "cosmos3_wam": {
        "command_envs": ("BLUEPRINT_COSMOS3_WAM_COMMAND", "BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND"),
        "checkpoint_envs": ("BLUEPRINT_COSMOS3_WAM_CHECKPOINT",),
        "runtime_role": "world_video_rollout_or_review_substrate",
    },
    "openvla_policy": {
        "command_envs": ("BLUEPRINT_OPENVLA_POLICY_COMMAND",),
        "checkpoint_envs": ("BLUEPRINT_OPENVLA_POLICY_CHECKPOINT",),
        "runtime_role": "vla_policy_endpoint_candidate",
    },
    "unitree_g1_policy": {
        "command_envs": (
            "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
            "BLUEPRINT_REALISTIC_G1_POLICY_COMMAND",
        ),
        "checkpoint_envs": ("BLUEPRINT_UNITREE_G1_POLICY_CHECKPOINT",),
        "runtime_role": "realistic_g1_navigation_or_control_candidate",
    },
}
GPU_PROVIDER_GATES = {
    "runpod": {
        "api_key_file_env": "RUNPOD_API_KEY_FILE",
        "default_api_key_file": "~/.blueprint-secrets/runpod_api_key",
        "api_gate_env": "BLUEPRINT_ALLOW_RUNPOD_API_CALLS",
        "launch_gate_env": "BLUEPRINT_ALLOW_GPU_PROVIDER_LAUNCH",
        "adapter_cli": "blueprint-run-runpod-provider-adapter",
    },
    "vast": {
        "api_key_file_env": "VAST_API_KEY_FILE",
        "default_api_key_file": "~/.blueprint-secrets/vast_api_key",
        "api_gate_env": "BLUEPRINT_ALLOW_VAST_API_CALLS",
        "launch_gate_env": "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH",
        "adapter_cli": "blueprint-run-vast-provider-adapter",
    },
    "digitalocean_gpu": {
        "api_key_file_env": "DIGITALOCEAN_API_TOKEN_FILE",
        "default_api_key_file": "~/.blueprint-secrets/digitalocean_api_token",
        "api_gate_env": "BLUEPRINT_ALLOW_DIGITALOCEAN_API_CALLS",
        "launch_gate_env": "BLUEPRINT_ALLOW_DIGITALOCEAN_GPU_DROPLET_LAUNCH",
        "adapter_cli": "blocked_no_digitalocean_gpu_adapter_implemented",
        "object_store_note": "DigitalOcean Spaces can be used through blueprint-stage-wam-provider-object-store as S3-compatible staging.",
    },
}
MODEL_SOURCE_HINTS = {
    "oscar_wam": {
        "provider": "oscar_public_inference",
        "source_urls": [
            "https://github.com/wuzy2115/oscar-public",
            "https://huggingface.co/zywu2115/OSCAR-2B",
            "https://wuzy2115.github.io/oscar-project-page/",
            "https://arxiv.org/abs/2606.04463",
        ],
        "auth_groups": ["huggingface"],
        "cloud_gpu_required_without_local_gpu": True,
        "host_requirements": {
            "os": "Linux x86_64",
            "accelerator": "NVIDIA CUDA GPU",
            "vram_recommended_gb": 24,
            "cuda_runtime": "12.4+",
        },
    },
    "cosmos_wam": {
        "provider": "nvidia_cosmos_predict",
        "source_urls": [
            "https://github.com/nvidia-cosmos/cosmos-predict2.5",
            "https://huggingface.co/nvidia/Cosmos-Predict2.5-2B",
        ],
        "auth_groups": ["huggingface", "ngc"],
        "cloud_gpu_required_without_local_gpu": True,
        "host_requirements": {
            "accelerator": "NVIDIA CUDA GPU",
            "note": "Robot action-conditioned and policy variants require the provider runtime or a compatible local adapter.",
        },
    },
    "cosmos3_wam": {
        "provider": "nvidia_cosmos_family",
        "source_urls": ["https://github.com/nvidia/cosmos"],
        "auth_groups": ["huggingface", "ngc"],
        "cloud_gpu_required_without_local_gpu": True,
    },
    "openvla_policy": {
        "provider": "openvla",
        "source_urls": [
            "https://github.com/openvla/openvla",
            "https://huggingface.co/openvla/openvla-7b",
            "https://openvla.github.io/",
        ],
        "auth_groups": ["huggingface"],
        "cloud_gpu_required_without_local_gpu": True,
        "host_requirements": {
            "accelerator": "CUDA GPU strongly recommended",
            "note": "OpenVLA must be adapted from its native robot action output into the Blueprint action schema.",
        },
    },
    "unitree_g1_policy": {
        "provider": "unitree_g1_policy_stack",
        "source_urls": [
            "https://github.com/unitreerobotics/unitree_rl_gym",
            "https://github.com/unitreerobotics/unitree_rl_lab",
            "https://github.com/unitreerobotics/unitree_mujoco",
            "https://github.com/unitreerobotics/unitree_lerobot",
        ],
        "auth_groups": [],
        "cloud_gpu_required_without_local_gpu": False,
        "host_requirements": {
            "note": "Requires a task-conditioned bridge from Blueprint endpoint actions into the Unitree controller command stream.",
        },
    },
}
ENDPOINT_READINESS_CANDIDATES = (
    "oscar_wam",
    "cosmos_wam",
    "cosmos3_wam",
    "openvla_policy",
    "unitree_g1_policy",
)
MODEL_SOURCE_ROOT_ENVS = {
    "oscar_wam": "BLUEPRINT_OSCAR_WAM_SOURCE_ROOT",
    "cosmos_wam": "BLUEPRINT_COSMOS_WAM_SOURCE_ROOT",
    "cosmos3_wam": "BLUEPRINT_COSMOS3_WAM_SOURCE_ROOT",
    "openvla_policy": "BLUEPRINT_OPENVLA_POLICY_SOURCE_ROOT",
    "unitree_g1_policy": "BLUEPRINT_UNITREE_G1_POLICY_SOURCE_ROOT",
}
MODEL_CHECKPOINT_FILE_SUFFIXES = (
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".gguf",
    ".distcp",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item for item in (_string(item) for item in value) if item]
    return []


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "y"}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _first_configured_env(env_names: Sequence[str]) -> tuple[str | None, str | None]:
    for env_name in env_names:
        value = os.getenv(env_name)
        if value:
            return env_name, value
    return None, None


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
    return bool(Path(executable).expanduser().is_file() or shutil.which(executable))


def _checkpoint_available(path: str | None) -> bool:
    return bool(path and Path(path).expanduser().exists())


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_source_roots_for_candidate(candidate: str) -> list[tuple[str, Path]]:
    workspace = _workspace_root()
    repo = Path(__file__).resolve().parents[2]
    roots: dict[str, list[tuple[str, Path]]] = {
        "oscar_wam": [
            ("workspace_oscar_vendor", workspace / "BlueprintValidation" / "data" / "vendor" / "oscar"),
            ("workspace_oscar_vendor_caps", workspace / "BlueprintValidation" / "data" / "vendor" / "OSCAR"),
            ("workspace_oscar_repo", workspace / "oscar"),
        ],
        "cosmos_wam": [
            (
                "workspace_dreamdojo_cosmos_predict2",
                workspace / "BlueprintValidation" / "data" / "vendor" / "DreamDojo" / "cosmos_predict2",
            ),
            (
                "workspace_cosmos_transfer",
                workspace / "BlueprintValidation" / "data" / "vendor" / "cosmos-transfer",
            ),
            (
                "workspace_cosmos_policy_adapter",
                workspace / "BlueprintPipeline" / "tools" / "cosmos_policy_adapter",
            ),
        ],
        "cosmos3_wam": [
            (
                "workspace_cosmos_transfer",
                workspace / "BlueprintValidation" / "data" / "vendor" / "cosmos-transfer",
            ),
        ],
        "openvla_policy": [
            (
                "workspace_openvla_oft",
                workspace / "BlueprintValidation" / "data" / "vendor" / "openvla-oft",
            ),
        ],
        "unitree_g1_policy": [
            ("workspace_unitree_rl_gym", workspace / "unitree_rl_gym"),
            ("workspace_unitree_rl_gym_caps", workspace / "Unitree_RL_Gym"),
        ],
    }
    oscar_runtime_roots = [
        ("robot_eval_job_oscar_wam_runtime", path)
        for path in sorted((repo / "robot_eval_jobs").glob("*/runtime_sources/oscar_wam/source"))
    ]
    unitree_runtime_roots = [
        ("robot_eval_job_unitree_rl_gym_runtime", path)
        for path in sorted((repo / "robot_eval_jobs").glob("*/runtime_sources/unitree_rl_gym"))
    ]
    if candidate == "oscar_wam":
        return roots.get(candidate, []) + oscar_runtime_roots
    if candidate == "unitree_g1_policy":
        return roots.get(candidate, []) + unitree_runtime_roots
    return roots.get(candidate, [])


def _source_roots_for_candidate(candidate: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env_name = MODEL_SOURCE_ROOT_ENVS.get(candidate)
    if env_name and os.getenv(env_name):
        rows.append(
            {
                "label": env_name,
                "path": Path(_string(os.getenv(env_name))).expanduser(),
                "configured_by_env": True,
            }
        )
    rows.extend(
        {"label": label, "path": path.expanduser(), "configured_by_env": False}
        for label, path in _default_source_roots_for_candidate(candidate)
    )
    return rows


def _checkpoint_like_files(root: Path, *, max_files_scanned: int = 2000) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    files_scanned = 0
    truncated = False
    if root.is_file():
        suffix = root.suffix.lower()
        if suffix in MODEL_CHECKPOINT_FILE_SUFFIXES:
            try:
                size_bytes = root.stat().st_size
            except OSError:
                size_bytes = None
            return {
                "files_scanned": 1,
                "truncated": False,
                "checkpoint_files_found": [
                    {
                        "relative_path": root.name,
                        "size_bytes": size_bytes,
                        "large_enough_for_wam_or_vla_weights": bool(
                            size_bytes and size_bytes >= 50 * 1024 * 1024
                        ),
                    }
                ],
            }
        return {"files_scanned": 1, "truncated": False, "checkpoint_files_found": []}
    if not root.is_dir():
        return {"files_scanned": 0, "truncated": False, "checkpoint_files_found": []}
    for current, dirs, files in os.walk(root):
        dirs[:] = [
            name
            for name in dirs
            if name not in {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache"}
        ]
        for filename in files:
            files_scanned += 1
            path = Path(current) / filename
            if path.suffix.lower() in MODEL_CHECKPOINT_FILE_SUFFIXES:
                try:
                    size_bytes = path.stat().st_size
                except OSError:
                    size_bytes = None
                matches.append(
                    {
                        "relative_path": str(path.relative_to(root)),
                        "size_bytes": size_bytes,
                        "large_enough_for_wam_or_vla_weights": bool(
                            size_bytes and size_bytes >= 50 * 1024 * 1024
                        ),
                    }
                )
                if len(matches) >= 12:
                    truncated = True
                    break
            if files_scanned >= max_files_scanned:
                truncated = True
                break
        if truncated:
            break
    return {
        "files_scanned": files_scanned,
        "truncated": truncated,
        "checkpoint_files_found": matches,
    }


def _local_source_tree_probe(
    candidate: str,
    *,
    command_configured: bool = False,
    checkpoint_configured: bool = False,
    checkpoint_exists: bool = False,
) -> dict[str, Any]:
    source_rows: list[dict[str, Any]] = []
    for root_row in _source_roots_for_candidate(candidate):
        path = Path(root_row["path"]).expanduser()
        present = path.exists()
        checkpoint_scan = _checkpoint_like_files(path) if present else {
            "files_scanned": 0,
            "truncated": False,
            "checkpoint_files_found": [],
        }
        checkpoint_files = list(checkpoint_scan["checkpoint_files_found"])
        source_rows.append(
            {
                "label": root_row["label"],
                "path": str(path),
                "configured_by_env": bool(root_row["configured_by_env"]),
                "present": present,
                "is_dir": path.is_dir(),
                "checkpoint_file_count": len(checkpoint_files),
                "large_checkpoint_file_count": sum(
                    1
                    for file_row in checkpoint_files
                    if file_row.get("large_enough_for_wam_or_vla_weights")
                ),
                "checkpoint_files_found": checkpoint_files,
                "files_scanned": checkpoint_scan["files_scanned"],
                "scan_truncated": checkpoint_scan["truncated"],
            }
        )
    present_rows = [row for row in source_rows if row["present"]]
    large_checkpoint_count = sum(int(row["large_checkpoint_file_count"]) for row in present_rows)
    blockers: list[str] = []
    if not present_rows:
        blockers.append("blocked_missing_local_model_source_tree")
    if present_rows and not command_configured:
        blockers.append("blocked_source_tree_present_without_runnable_adapter_command")
    if present_rows and not checkpoint_configured:
        blockers.append("blocked_source_tree_present_without_configured_checkpoint")
    if checkpoint_configured and not checkpoint_exists:
        blockers.append("blocked_configured_checkpoint_path_missing")
    if present_rows and large_checkpoint_count == 0:
        blockers.append("blocked_no_large_model_checkpoint_files_found_in_source_tree")
    status = (
        "source_tree_and_configured_checkpoint_present"
        if present_rows and command_configured and checkpoint_exists
        else "blocked_source_tree_is_not_runtime"
    )
    return {
        "schema_version": "local_model_source_tree_probe.v1",
        "candidate_id": candidate,
        "status": status,
        "source_tree_present": bool(present_rows),
        "present_source_tree_count": len(present_rows),
        "large_checkpoint_file_count": large_checkpoint_count,
        "source_roots": source_rows,
        "blockers": blockers,
        "why_source_tree_is_not_runtime": [
            "A source checkout does not prove a runnable model endpoint.",
            "Blueprint needs a command that accepts the rollout/observation contract and writes Blueprint-compatible JSON.",
            "Blueprint also needs a configured checkpoint or mounted weights path with provenance.",
        ],
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


def discover_local_model_source_trees(
    *,
    generated_at: str,
    candidates: Sequence[str] = ENDPOINT_READINESS_CANDIDATES,
) -> dict[str, Any]:
    rows = [_local_source_tree_probe(candidate) for candidate in candidates]
    return {
        "schema_version": "local_model_source_tree_discovery.v1",
        "generated_at": generated_at,
        "status": "completed",
        "candidates": rows,
        "claim_boundary": {
            "source_tree_present_is_not_model_runtime_proof": True,
            "checkpoint_like_file_discovery_is_not_model_execution_proof": True,
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        },
    }


def _local_host_probe() -> dict[str, Any]:
    torch_cuda_available: bool | None = None
    torch_import_error: str | None = None
    nvidia_smi_available = bool(shutil.which("nvidia-smi"))
    if platform.system() == "Darwin" and not nvidia_smi_available:
        torch_import_error = "skipped_non_cuda_macos_host"
    else:
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import json, torch; print(json.dumps({'cuda': bool(torch.cuda.is_available())}))",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=8,
                env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
            )
            if result.returncode == 0:
                torch_payload = json.loads(result.stdout.strip() or "{}")
                torch_cuda_available = bool(torch_payload.get("cuda"))
            else:
                torch_import_error = "torch_probe_subprocess_failed"
        except Exception as exc:
            torch_import_error = type(exc).__name__
    return {
        "schema_version": "local_model_host_probe.v1",
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "nvidia_smi_available": nvidia_smi_available,
        "cuda_visible_devices_configured": bool(os.getenv("CUDA_VISIBLE_DEVICES")),
        "torch_cuda_available": torch_cuda_available,
        "torch_probe_error_type": torch_import_error,
        "local_host_is_likely_cuda_gpu_host": bool(
            nvidia_smi_available or torch_cuda_available
        ),
    }


def _secret_file_probe(env_name: str, default_path: str) -> dict[str, Any]:
    configured_path = _string(os.getenv(env_name))
    path = Path(configured_path or default_path).expanduser()
    return {
        "env_name": env_name,
        "path": str(path),
        "configured_by_env": bool(configured_path),
        "present": path.is_file(),
        "permission_recommended": "0600",
        "raw_secret_written_to_artifacts": False,
        "secret_hash_written_to_artifacts": False,
    }


def build_policy_model_endpoint_readiness_manifest(
    *,
    generated_at: str,
    candidates: Sequence[str] = ENDPOINT_READINESS_CANDIDATES,
    explicit_candidate_id: str | None = None,
    explicit_command: str | None = None,
    explicit_checkpoint: Path | None = None,
) -> dict[str, Any]:
    normalize_model_access_env()
    model_access = model_access_secret_status()
    host_probe = _local_host_probe()
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        contract = MODEL_RUNTIME_CONTRACTS.get(candidate, {})
        source_hint = MODEL_SOURCE_HINTS.get(candidate, {})
        command_env, command = _first_configured_env(contract.get("command_envs", ()))
        checkpoint_env, checkpoint = _first_configured_env(contract.get("checkpoint_envs", ()))
        if explicit_command and candidate == (explicit_candidate_id or candidate):
            command_env = "cli:--wam-model-command"
            command = explicit_command
        if explicit_checkpoint and candidate == (explicit_candidate_id or candidate):
            checkpoint_env = "cli:--wam-model-checkpoint"
            checkpoint = str(explicit_checkpoint)
        command_ok = _command_available(command)
        checkpoint_ok = _checkpoint_available(checkpoint)
        local_source = _local_source_tree_probe(
            candidate,
            command_configured=bool(command),
            checkpoint_configured=bool(checkpoint),
            checkpoint_exists=checkpoint_ok,
        )
        auth_groups = list(source_hint.get("auth_groups", []))
        auth_ready = {
            group_id: bool(_mapping(model_access.get(group_id)).get("auth_ready"))
            for group_id in auth_groups
        }
        missing: list[str] = []
        if not command:
            preferred_env = next(iter(contract.get("command_envs", ())), "MODEL_COMMAND")
            missing.append(f"set_{preferred_env}_to_runnable_adapter_command")
        elif not command_ok:
            missing.append("make_configured_model_command_executable_or_on_path")
        if not checkpoint:
            preferred_checkpoint_env = next(
                iter(contract.get("checkpoint_envs", ())),
                "MODEL_CHECKPOINT",
            )
            missing.append(f"set_{preferred_checkpoint_env}_to_local_checkpoint_path")
        elif not checkpoint_ok:
            missing.append("download_or_mount_configured_model_checkpoint_path")
        for group_id, ready in auth_ready.items():
            if not ready:
                missing.append(f"configure_file_based_{group_id}_auth")
        if candidate.endswith("_wam") and not _env_truthy(LOCAL_MODEL_GATE_ENV):
            missing.append(f"set_{LOCAL_MODEL_GATE_ENV}=true")

        auth_values = list(auth_ready.values())
        endpoint_wrapper_can_be_created = command_ok
        real_model_runtime_ready = bool(
            command_ok
            and checkpoint_ok
            and (all(auth_values) if auth_values else True)
            and (not candidate.endswith("_wam") or _env_truthy(LOCAL_MODEL_GATE_ENV))
        )
        rows.append(
            {
                "candidate_id": candidate,
                "runtime_role": contract.get("runtime_role", "replaceable_model_adapter"),
                "status": "ready_for_real_model_endpoint"
                if real_model_runtime_ready
                else "blocked",
                "command_envs": list(contract.get("command_envs", ())),
                "configured_command_env": command_env,
                "command_configured": bool(command),
                "command_available": command_ok,
                "command_value_redacted": "<configured>" if command else None,
                "checkpoint_envs": list(contract.get("checkpoint_envs", ())),
                "configured_checkpoint_env": checkpoint_env,
                "checkpoint_configured": bool(checkpoint),
                "checkpoint_exists": checkpoint_ok,
                "checkpoint_path": str(Path(checkpoint).expanduser()) if checkpoint else None,
                "endpoint_wrapper_can_be_created": endpoint_wrapper_can_be_created,
                "real_model_runtime_ready": real_model_runtime_ready,
                "model_access_auth_groups": auth_groups,
                "model_access_auth_ready": auth_ready,
                "local_model_gate_env": LOCAL_MODEL_GATE_ENV
                if candidate.endswith("_wam")
                else None,
                "local_model_gate_enabled": _env_truthy(LOCAL_MODEL_GATE_ENV)
                if candidate.endswith("_wam")
                else None,
                "source_urls": source_hint.get("source_urls", []),
                "local_source_discovery": local_source,
                "host_requirements": source_hint.get("host_requirements", {}),
                "local_host_probe": host_probe,
                "cloud_gpu_required_without_local_gpu": bool(
                    source_hint.get("cloud_gpu_required_without_local_gpu")
                ),
                "what_is_needed_to_make_true": missing,
                "why_we_cannot_just_create_real_endpoint": [
                    "The HTTP wrapper can only call a command that already exists.",
                    "A real model claim requires model weights/checkpoint provenance plus a command that emits Blueprint-compatible JSON.",
                    "Starting the wrapper around a missing command would only produce 503/502 responses.",
                    "Using the reference adapter proves endpoint plumbing, not OSCAR/Cosmos/OpenVLA/Unitree model behavior.",
                ],
                "raw_credentials_written_to_artifacts": False,
                "raw_credential_hashes_written_to_artifacts": False,
            }
        )
    ready = [row for row in rows if row["real_model_runtime_ready"]]
    wrapper_ready = [row for row in rows if row["endpoint_wrapper_can_be_created"]]
    blockers = sorted(
        {
            item
            for row in rows
            for item in row["what_is_needed_to_make_true"]
        }
    )
    return {
        "schema_version": "policy_model_endpoint_readiness_manifest.v1",
        "generated_at": generated_at,
        "status": "ready_for_real_model_endpoint" if ready else "blocked",
        "http_endpoint_wrapper_available": True,
        "endpoint_wrapper_ready_candidate_count": len(wrapper_ready),
        "real_model_ready_candidate_count": len(ready),
        "local_host_probe": host_probe,
        "candidates": rows,
        "blockers": blockers,
        "claim_boundary": {
            "endpoint_creation_is_not_model_execution_proof": True,
            "real_model_endpoint_requires_command_checkpoint_and_provenance": True,
            "reference_adapter_is_not_real_wam_vla": True,
            "physical_robot_readiness_proven": False,
            "raw_credentials_written_to_artifacts": False,
        },
    }


def build_policy_model_endpoint_creation_plan(
    *,
    generated_at: str,
    readiness_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    readiness = (
        dict(readiness_manifest)
        if isinstance(readiness_manifest, Mapping)
        else build_policy_model_endpoint_readiness_manifest(generated_at=generated_at)
    )
    rows: list[dict[str, Any]] = []
    for row_value in readiness.get("candidates", []) or []:
        if not isinstance(row_value, Mapping):
            continue
        row = dict(row_value)
        candidate_id = _string(row.get("candidate_id"))
        configured_command_env = _string(row.get("configured_command_env"))
        command_env = configured_command_env or _string(
            next(iter(row.get("command_envs", []) or []), "")
        )
        configured_checkpoint_env = _string(row.get("configured_checkpoint_env"))
        checkpoint_env = configured_checkpoint_env or _string(
            next(iter(row.get("checkpoint_envs", []) or []), "")
        )
        command_from_cli = command_env.startswith("cli:")
        checkpoint_from_cli = checkpoint_env.startswith("cli:")
        auth_ready = _mapping(row.get("model_access_auth_ready"))
        auth_blockers = [
            f"configure_file_based_{group_id}_auth"
            for group_id, ready in sorted(auth_ready.items())
            if not ready
        ]
        command_available = bool(row.get("command_available"))
        checkpoint_exists = bool(row.get("checkpoint_exists"))
        real_ready = bool(row.get("real_model_runtime_ready"))
        endpoint_wrapper_can_be_created = bool(row.get("endpoint_wrapper_can_be_created"))
        rows.append(
            {
                "candidate_id": candidate_id,
                "runtime_role": row.get("runtime_role"),
                "can_start_http_wrapper_now": endpoint_wrapper_can_be_created,
                "can_claim_real_model_endpoint_now": real_ready,
                "can_run_wam_generated_rollout_now": real_ready
                and candidate_id.endswith("_wam"),
                "command_env": command_env or None,
                "checkpoint_env": checkpoint_env or None,
                "command_configured_from_cli": command_from_cli,
                "checkpoint_configured_from_cli": checkpoint_from_cli,
                "required_env_exports": [
                    export
                    for export in [
                        f"export {command_env}=<runnable_adapter_command>"
                        if command_env and not command_available and not command_from_cli
                        else None,
                        f"export {checkpoint_env}=<local_checkpoint_or_weights_path>"
                        if checkpoint_env and not checkpoint_exists and not checkpoint_from_cli
                        else None,
                        f"export {LOCAL_MODEL_GATE_ENV}=true"
                        if candidate_id.endswith("_wam")
                        and not bool(row.get("local_model_gate_enabled"))
                        else None,
                    ]
                    if export
                ],
                "auth_blockers": auth_blockers,
                "launch_endpoint_command": (
                    (
                        "BLUEPRINT_WAM_VLA_POLICY_COMMAND='<configured --wam-model-command>' "
                        "BLUEPRINT_WAM_VLA_POLICY_AUTH_TOKEN_FILE=\"$TEAM_POLICY_AUTH_TOKEN_FILE\" "
                        "blueprint-serve-wam-vla-policy-endpoint --host 127.0.0.1 --port 8765"
                    )
                    if command_from_cli
                    else (
                        "BLUEPRINT_WAM_VLA_POLICY_COMMAND=\"$"
                        f"{command_env}\" "
                        "BLUEPRINT_WAM_VLA_POLICY_AUTH_TOKEN_FILE=\"$TEAM_POLICY_AUTH_TOKEN_FILE\" "
                        "blueprint-serve-wam-vla-policy-endpoint --host 127.0.0.1 --port 8765"
                    )
                    if command_env
                    else None
                ),
                "wam_evaluator_command": (
                    "BLUEPRINT_ALLOW_LOCAL_WAM_MODEL=true "
                    "python -m blueprint_pipeline.oscar_cosmos_wam_evaluator "
                    "--input-job-dir <mujoco_endpoint_eval_job_dir> "
                    "--allow-wam-model-run"
                    if candidate_id.endswith("_wam")
                    else None
                ),
                "blocked_reason": None
                if real_ready
                else "missing_runnable_adapter_command_or_local_checkpoint_or_required_auth",
                "what_is_needed_to_make_true": row.get("what_is_needed_to_make_true", []),
                "raw_credentials_written_to_artifacts": False,
                "raw_credential_hashes_written_to_artifacts": False,
            }
        )
    real_ready_count = sum(1 for row in rows if row["can_claim_real_model_endpoint_now"])
    wrapper_ready_count = sum(1 for row in rows if row["can_start_http_wrapper_now"])
    return {
        "schema_version": "policy_model_endpoint_creation_plan.v1",
        "generated_at": generated_at,
        "status": "ready_for_real_model_endpoint" if real_ready_count else "blocked",
        "http_wrapper_binary_available": True,
        "can_create_http_wrapper_for_configured_commands_now": bool(wrapper_ready_count),
        "can_create_real_model_endpoint_now": bool(real_ready_count),
        "real_model_ready_candidate_count": real_ready_count,
        "endpoint_wrapper_ready_candidate_count": wrapper_ready_count,
        "candidate_creation_plans": rows,
        "why_cannot_just_create_missing_model_endpoints": [
            "An HTTP endpoint without a runnable command would return 503 or 502 rather than model actions.",
            "A real OSCAR/Cosmos/OpenVLA/Unitree claim requires local weights/checkpoint provenance.",
            "Blueprint still needs an adapter that maps each model input/output to the Blueprint observation/action contract.",
            "Cloud GPU launch requires file-based provider credentials and explicit spend gates before starting infrastructure.",
        ],
        "minimum_user_supplied_inputs": [
            "runnable adapter command for the selected model family",
            "local checkpoint or mounted weights path for the selected model family",
            "file-based provider/auth tokens when the model source requires them",
            "explicit local/cloud run gates such as BLUEPRINT_ALLOW_LOCAL_WAM_MODEL=true",
        ],
        "claim_boundary": {
            "http_endpoint_creation_is_not_model_execution_proof": True,
            "reference_adapter_endpoint_is_not_learned_wam_or_vla": True,
            "raw_credentials_written_to_artifacts": False,
            "raw_credential_hashes_written_to_artifacts": False,
        },
    }


def discover_cloud_gpu_setup(*, generated_at: str) -> dict[str, Any]:
    providers: list[dict[str, Any]] = []
    for provider_id, contract in GPU_PROVIDER_GATES.items():
        secret = _secret_file_probe(
            _string(contract["api_key_file_env"]),
            _string(contract["default_api_key_file"]),
        )
        api_gate = _string(contract["api_gate_env"])
        launch_gate = _string(contract["launch_gate_env"])
        blockers: list[str] = []
        if not secret["present"]:
            blockers.append(f"missing_file_based_secret_{secret['env_name']}")
        if not _env_truthy(api_gate):
            blockers.append(f"missing_env_{api_gate}")
        if not _env_truthy(launch_gate):
            blockers.append(f"missing_env_{launch_gate}")
        if _string(contract.get("adapter_cli")).startswith("blocked_"):
            blockers.append(_string(contract["adapter_cli"]))
        providers.append(
            {
                "provider_id": provider_id,
                "status": "ready_for_gated_launch" if not blockers else "blocked",
                "adapter_cli": contract["adapter_cli"],
                "api_key_file": secret,
                "api_gate_env": api_gate,
                "api_gate_enabled": _env_truthy(api_gate),
                "launch_gate_env": launch_gate,
                "launch_gate_enabled": _env_truthy(launch_gate),
                "blockers": blockers,
                "object_store_note": contract.get("object_store_note"),
            }
        )
    provider_ready_for_gated_launch = any(row["status"] == "ready_for_gated_launch" for row in providers)
    provider_secret_available = any(row["api_key_file"]["present"] for row in providers)
    return {
        "schema_version": "policy_cloud_gpu_setup_manifest.v1",
        "generated_at": generated_at,
        "status": "ready_for_gated_launch"
        if provider_ready_for_gated_launch
        else "blocked_pending_cloud_provider_gates",
        "provider_secret_available": provider_secret_available,
        "providers": providers,
        "no_local_gpu_assumption": True,
        "spend_requires_explicit_gates": True,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


def discover_wam_model_runtimes(
    *,
    candidates: Sequence[str] = DEFAULT_MODEL_CANDIDATES,
    generated_at: str,
    explicit_candidate_id: str | None = None,
    explicit_command: str | None = None,
    explicit_checkpoint: Path | None = None,
) -> dict[str, Any]:
    normalize_model_access_env()
    model_access = model_access_secret_status()
    cloud_gpu_setup = discover_cloud_gpu_setup(generated_at=generated_at)
    local_source_discovery = discover_local_model_source_trees(
        generated_at=generated_at,
        candidates=tuple(dict.fromkeys(tuple(candidates) + ENDPOINT_READINESS_CANDIDATES)),
    )
    host_probe = _local_host_probe()
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        contract = MODEL_RUNTIME_CONTRACTS.get(candidate, {})
        source_hint = MODEL_SOURCE_HINTS.get(candidate, {})
        command_env, command = _first_configured_env(contract.get("command_envs", ()))
        checkpoint_env, checkpoint = _first_configured_env(contract.get("checkpoint_envs", ()))
        selected_explicit_candidate = explicit_candidate_id or (candidates[0] if candidates else None)
        if explicit_command and candidate == selected_explicit_candidate:
            command_env = "cli:--wam-model-command"
            command = explicit_command
        if explicit_checkpoint and candidate == selected_explicit_candidate:
            checkpoint_env = "cli:--wam-model-checkpoint"
            checkpoint = str(explicit_checkpoint)
        command_ok = _command_available(command)
        checkpoint_ok = _checkpoint_available(checkpoint)
        blockers: list[str] = []
        if not command_ok:
            blockers.append("blocked_missing_wam_runtime")
        if not checkpoint_ok:
            blockers.append("blocked_missing_wam_model_checkpoint")
        official_adapter_host_blockers: list[str] = []
        command_text = _string(command)
        official_oscar_adapter_configured = (
            candidate == "oscar_wam"
            and "blueprint_pipeline.oscar_wam_command_adapter" in command_text
        )
        if official_oscar_adapter_configured and not host_probe.get("local_host_is_likely_cuda_gpu_host"):
            if platform.system() == "Darwin":
                official_adapter_host_blockers.append("blocked_oscar_linux_cuda_runtime_required")
            official_adapter_host_blockers.append("blocked_oscar_requires_cuda_gpu_runtime")
        configured_ready = bool(command_ok and checkpoint_ok)
        direct_local_ready = bool(configured_ready and not official_adapter_host_blockers)
        row_status = (
            "ready_for_local_model_run"
            if direct_local_ready
            else "configured_model_runtime_host_blocked"
            if configured_ready and official_adapter_host_blockers
            else "blocked"
        )
        local_source = _local_source_tree_probe(
            candidate,
            command_configured=bool(command),
            checkpoint_configured=bool(checkpoint),
            checkpoint_exists=checkpoint_ok,
        )
        rows.append(
            {
                "candidate_id": candidate,
                "runtime_role": contract.get("runtime_role", "replaceable_wam_adapter"),
                "status": row_status,
                "command_env": command_env,
                "command_configured": bool(command),
                "command_available": command_ok,
                "command_value_redacted": "<configured>" if command else None,
                "checkpoint_env": checkpoint_env,
                "checkpoint_configured": bool(checkpoint),
                "checkpoint_exists": checkpoint_ok,
                "checkpoint_path": str(Path(checkpoint).expanduser()) if checkpoint else None,
                "model_access_auth_groups": source_hint.get("auth_groups", []),
                "model_access_auth_ready": {
                    group_id: bool(_mapping(model_access.get(group_id)).get("auth_ready"))
                    for group_id in source_hint.get("auth_groups", [])
                },
                "source_urls": source_hint.get("source_urls", []),
                "local_source_discovery": local_source,
                "cloud_gpu_required_without_local_gpu": bool(
                    source_hint.get("cloud_gpu_required_without_local_gpu")
                ),
                "local_host_probe": host_probe,
                "local_cuda_gpu_required_for_direct_local_model_run": bool(
                    source_hint.get("cloud_gpu_required_without_local_gpu")
                ),
                "direct_local_model_run_likely_blocked_without_cuda_gpu": bool(
                    source_hint.get("cloud_gpu_required_without_local_gpu")
                    and not host_probe.get("local_host_is_likely_cuda_gpu_host")
                ),
                "configured_command_checkpoint_ready": configured_ready,
                "official_adapter_host_preflight_blockers": official_adapter_host_blockers,
                "official_adapter_direct_local_model_run_ready": direct_local_ready,
                "direct_local_model_run_ready": direct_local_ready,
                "provider_or_linux_cuda_runtime_required": bool(official_adapter_host_blockers),
                "blockers": blockers,
                "raw_credentials_written_to_artifacts": False,
            }
        )
    ready = [row for row in rows if row.get("direct_local_model_run_ready")]
    configured = [row for row in rows if row.get("configured_command_checkpoint_ready")]
    selected_row = ready[0] if ready else (configured[0] if configured else (rows[0] if rows else None))
    selected_blockers = (
        list(selected_row.get("blockers", []))
        + list(selected_row.get("official_adapter_host_preflight_blockers", []))
        if selected_row
        else []
    )
    all_candidate_blockers = sorted(
        {
            blocker
            for row in rows
            for blocker in (
                list(row.get("blockers", []))
                + list(row.get("official_adapter_host_preflight_blockers", []))
            )
        }
    )
    configured_but_host_blocked = any(
        row.get("configured_command_checkpoint_ready")
        and row.get("official_adapter_host_preflight_blockers")
        for row in rows
    )
    return {
        "schema_version": "wam_model_runtime_discovery.v1",
        "generated_at": generated_at,
        "status": (
            "ready_for_local_model_run"
            if ready
            else "configured_model_runtime_host_blocked"
            if configured_but_host_blocked
            else "blocked"
        ),
        "local_model_gate_env": LOCAL_MODEL_GATE_ENV,
        "local_model_gate_enabled": _env_truthy(LOCAL_MODEL_GATE_ENV),
        "local_host_probe": host_probe,
        "candidates": rows,
        "selected_candidate": selected_row.get("candidate_id") if selected_row else None,
        "selected_candidate_blockers": selected_blockers,
        "all_candidate_blockers": all_candidate_blockers,
        "blockers": [] if ready else all_candidate_blockers,
        "model_access_secret_status": model_access,
        "local_model_source_tree_discovery": local_source_discovery,
        "cloud_gpu_setup": cloud_gpu_setup,
        "claim_boundary": {
            "missing_runtime_or_checkpoint_blocks_generated_rollout_claims": True,
            "model_provider_replaceable": True,
            "raw_credentials_written_to_artifacts": False,
            "raw_credential_hashes_written_to_artifacts": False,
        },
    }


def _candidate_matrix(generated_at: str) -> dict[str, Any]:
    return {
        "schema_version": "policy_model_candidate_matrix.v1",
        "generated_at": generated_at,
        "status": "adapter_boundary_defined",
        "candidates": [
            {
                "id": "oscar_wam",
                "runtime_role": "action_conditioned_world_model_evaluator",
                "command_env": "BLUEPRINT_OSCAR_WAM_COMMAND",
                "checkpoint_env": "BLUEPRINT_OSCAR_WAM_CHECKPOINT",
                "auth_file_envs": ["HF_TOKEN_FILE", "HUGGINGFACE_HUB_TOKEN_FILE"],
                "cloud_gpu_provider_options": ["runpod", "vast"],
            },
            {
                "id": "cosmos_wam",
                "runtime_role": "world_video_rollout_or_review_substrate",
                "command_env": "BLUEPRINT_COSMOS_WAM_COMMAND",
                "checkpoint_env": "BLUEPRINT_COSMOS_WAM_CHECKPOINT",
                "source_urls": [
                    "https://github.com/nvidia-cosmos/cosmos-predict2.5",
                    "https://huggingface.co/nvidia/Cosmos-Predict2.5-2B",
                ],
                "auth_file_envs": ["HF_TOKEN_FILE", "NGC_API_KEY_FILE"],
                "cloud_gpu_provider_options": ["runpod", "vast"],
            },
            {
                "id": "openvla_policy",
                "runtime_role": "vla_policy_endpoint_candidate",
                "command_env": "BLUEPRINT_OPENVLA_POLICY_COMMAND",
                "checkpoint_env": "BLUEPRINT_OPENVLA_POLICY_CHECKPOINT",
                "source_urls": [
                    "https://github.com/openvla/openvla",
                    "https://huggingface.co/openvla/openvla-7b",
                ],
                "auth_file_envs": ["HF_TOKEN_FILE"],
                "cloud_gpu_provider_options": ["runpod", "vast"],
            },
            {
                "id": "unitree_g1_policy",
                "runtime_role": "realistic_g1_navigation_or_control_candidate",
                "command_env": "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
                "checkpoint_env": "BLUEPRINT_UNITREE_G1_POLICY_CHECKPOINT",
                "source_urls": [
                    "https://github.com/unitreerobotics/unitree_rl_gym",
                    "https://github.com/unitreerobotics/unitree_rl_lab",
                    "https://github.com/unitreerobotics/unitree_mujoco",
                    "https://github.com/unitreerobotics/unitree_lerobot",
                    "https://huggingface.co/docs/lerobot/unitree_g1",
                ],
                "cloud_gpu_provider_options": ["runpod", "vast"],
            },
            {
                "id": "command_policy",
                "runtime_role": "local_command_policy_endpoint",
                "command": "blueprint-g1-endpoint-reference-adapter",
                "checkpoint_required": False,
            },
        ],
        "backend_swap_boundary": "model adapters sit behind Blueprint observation/action and WAM rollout contracts",
    }


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def _review_videos(input_job_dir: Path) -> list[dict[str, Any]]:
    selection = _load_json(input_job_dir / "review_video_selection_manifest.json")
    selected: list[dict[str, Any]] = []
    for item in selection.get("selected_review_videos", []) or []:
        if isinstance(item, Mapping):
            selected.append(dict(item))
    if selected:
        return selected
    status = _load_json(input_job_dir / "video_generation_status.json")
    for item in status.get("videos", []) or []:
        if isinstance(item, Mapping):
            selected.append(dict(item))
    return selected


def _action_counts(action_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in action_rows:
        action = _mapping(row.get("normalized_action"))
        action_type = _string(action.get("action_type")) or "unknown"
        counts[action_type] = counts.get(action_type, 0) + 1
    return [{"action_type": key, "count": counts[key]} for key in sorted(counts)]


def _run_local_wam_command(
    *,
    command: str,
    input_manifest_path: Path,
    output_path: Path,
    candidate_id: str,
    checkpoint_path: str | None,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    env = {
        **os.environ,
        "BLUEPRINT_WAM_ROLLOUT_INPUT": str(input_manifest_path),
        "BLUEPRINT_WAM_ROLLOUT_OUTPUT": str(output_path),
        "BLUEPRINT_WAM_MODEL_CANDIDATE": candidate_id,
    }
    if checkpoint_path:
        env["BLUEPRINT_WAM_MODEL_CHECKPOINT"] = checkpoint_path
    try:
        result = subprocess.run(
            shlex.split(command),
            cwd=str(input_manifest_path.parent),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        return {}, {
            "status": "blocked",
            "blockers": [f"wam_model_command_failed:{type(exc).__name__}"],
            "duration_seconds": round(time.monotonic() - started, 6),
        }
    detail = {
        "status": "completed" if result.returncode == 0 else "blocked",
        "command_exit_code": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "stdout_size_bytes": len(result.stdout or ""),
        "stderr_size_bytes": len(result.stderr or ""),
        "stderr_omitted_to_avoid_secret_leakage": bool(result.stderr),
        "blockers": [] if result.returncode == 0 else ["wam_model_command_nonzero_exit"],
    }
    payload: dict[str, Any] = {}
    if output_path.is_file():
        payload = _load_json(output_path)
    elif result.stdout.strip():
        try:
            value = json.loads(result.stdout)
            payload = dict(value) if isinstance(value, Mapping) else {}
        except json.JSONDecodeError:
            detail["status"] = "blocked"
            detail["blockers"] = ["wam_model_stdout_json_invalid"]
    return payload, detail


def _wam_rollout_blocked_reason(blockers: Sequence[str]) -> str:
    blocker_set = set(blockers)
    missing_runtime = "blocked_missing_wam_runtime" in blocker_set
    missing_checkpoint = "blocked_missing_wam_model_checkpoint" in blocker_set
    if "blocked_local_wam_model_run_not_enabled" in blocker_set:
        return "blocked_local_wam_model_run_not_enabled"
    if "wam_model_command_nonzero_exit" in blocker_set:
        return "blocked_wam_model_command_failed"
    if any(str(blocker).startswith("wam_model_command_failed:") for blocker in blockers):
        return "blocked_wam_model_command_failed"
    if missing_runtime and not missing_checkpoint:
        return "blocked_missing_wam_runtime"
    if missing_checkpoint and not missing_runtime:
        return "blocked_missing_wam_model_checkpoint"
    if missing_runtime and missing_checkpoint:
        return "blocked_missing_wam_runtime_and_checkpoint"
    for blocker in blockers:
        if _string(blocker).startswith("blocked_"):
            return _string(blocker)
    return "blocked_missing_wam_model_runtime_or_checkpoint"


def _rollout_video_path(row: Mapping[str, Any], *, base_dir: Path) -> Path | None:
    value = _string(row.get("generated_video_path"))
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def run_oscar_cosmos_wam_evaluator(
    *,
    input_job_dir: Path,
    job_dir: Path | None = None,
    job_root: Path | None = None,
    model_candidates: Sequence[str] = DEFAULT_MODEL_CANDIDATES,
    wam_model_command: str | None = None,
    wam_model_checkpoint: Path | None = None,
    allow_wam_model_run: bool = False,
    timeout_seconds: float = 60.0,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    input_dir = Path(input_job_dir).resolve()
    if job_dir is None:
        root = Path(job_root or (_repo_root() / "robot_eval_jobs"))
        job_dir = root / f"oscar_cosmos_wam_evaluator_{_timestamp()}"
    output_dir = Path(job_dir).resolve()
    ensure_dir(output_dir)

    scenario_matrix = _load_json(input_dir / "scenario_eval_matrix.json")
    attempt_trace = _load_json(input_dir / "normalized_attempt_trace.json")
    action_rows = _read_jsonl(input_dir / "normalized_policy_action_trace.jsonl")
    locomotion_rows = _read_jsonl(input_dir / "g1_mujoco_locomotion_trace.jsonl")
    videos = _review_videos(input_dir)
    attempts = [
        dict(item)
        for item in attempt_trace.get("attempts", []) or []
        if isinstance(item, Mapping)
    ]
    matrix_runs = [
        dict(item)
        for item in scenario_matrix.get("runs", []) or []
        if isinstance(item, Mapping)
    ]
    runtime_discovery = discover_wam_model_runtimes(
        candidates=model_candidates,
        generated_at=generated,
        explicit_candidate_id=model_candidates[0] if model_candidates else None,
        explicit_command=wam_model_command,
        explicit_checkpoint=wam_model_checkpoint,
    )
    write_json(output_dir / "wam_model_runtime_discovery.json", runtime_discovery)
    endpoint_readiness = build_policy_model_endpoint_readiness_manifest(
        generated_at=generated,
        candidates=tuple(dict.fromkeys(tuple(model_candidates) + ENDPOINT_READINESS_CANDIDATES)),
        explicit_candidate_id=model_candidates[0] if model_candidates else None,
        explicit_command=wam_model_command,
        explicit_checkpoint=wam_model_checkpoint,
    )
    write_json(
        output_dir / "policy_model_endpoint_readiness_manifest.json",
        endpoint_readiness,
    )
    endpoint_creation_plan = build_policy_model_endpoint_creation_plan(
        generated_at=generated,
        readiness_manifest=endpoint_readiness,
    )
    write_json(
        output_dir / "policy_model_endpoint_creation_plan.json",
        endpoint_creation_plan,
    )
    write_json(
        output_dir / "policy_cloud_gpu_setup_manifest.json",
        _mapping(runtime_discovery.get("cloud_gpu_setup")),
    )
    write_json(
        output_dir / "local_model_source_tree_discovery.json",
        _mapping(runtime_discovery.get("local_model_source_tree_discovery")),
    )
    write_json(output_dir / "policy_model_candidate_matrix.json", _candidate_matrix(generated))

    rollout_input_manifest = {
        "schema_version": "wam_rollout_input_manifest.v1",
        "generated_at": generated,
        "status": "ready_for_model" if attempts and action_rows else "blocked_missing_inputs",
        "source_mujoco_endpoint_eval_job_dir": str(input_dir),
        "inputs": {
            "scenario_eval_matrix": str(input_dir / "scenario_eval_matrix.json"),
            "normalized_attempt_trace": str(input_dir / "normalized_attempt_trace.json"),
            "normalized_policy_action_trace_jsonl": str(
                input_dir / "normalized_policy_action_trace.jsonl"
            ),
            "g1_mujoco_locomotion_trace_jsonl": str(
                input_dir / "g1_mujoco_locomotion_trace.jsonl"
            ),
            "review_video_selection_manifest": str(
                input_dir / "review_video_selection_manifest.json"
            ),
        },
        "counts": {
            "matrix_run_count": len(matrix_runs),
            "attempt_count": len(attempts),
            "action_row_count": len(action_rows),
            "locomotion_row_count": len(locomotion_rows),
            "selected_review_video_count": len(videos),
        },
        "selected_review_videos": videos,
        "task_prompts": [
            {
                "scenario_eval_run_id": run.get("scenario_eval_run_id"),
                "task_id": run.get("task_id"),
                "spawn_id": run.get("spawn_id"),
                "task_prompt": run.get("task_prompt"),
            }
            for run in matrix_runs
        ],
    }
    write_json(output_dir / "wam_rollout_input_manifest.json", rollout_input_manifest)

    action_conditioning = {
        "schema_version": "wam_action_conditioning_manifest.v1",
        "generated_at": generated,
        "status": "completed" if action_rows else "blocked_missing_action_trace",
        "conditioning_sources": [
            "normalized_policy_action_trace.jsonl",
            "g1_mujoco_locomotion_trace.jsonl",
            "selected_review_videos_or_posters",
            "task_prompts",
        ],
        "action_type_counts": _action_counts(action_rows),
        "robot_pose_encoding": {
            "source": "g1_mujoco_locomotion_trace.root_position/root_quaternion_wxyz/root_yaw_rad",
            "skeleton_encoding_available": bool(locomotion_rows),
            "unitree_g1_joint_state_available": bool(locomotion_rows),
        },
        "sc3_style_hooks": [
            "forward_dynamics_consistency",
            "inverse_dynamics_consistency",
            "cross_view_consistency",
            "test_time_consistency_uncertainty_termination",
            "generated_rollout_termination_reason",
            "model_rollout_confidence",
        ],
    }
    write_json(output_dir / "wam_action_conditioning_manifest.json", action_conditioning)

    command_probe_candidates = [
        row
        for row in runtime_discovery.get("candidates", [])
        if isinstance(row, Mapping)
        and row.get("configured_command_checkpoint_ready")
        and not row.get("blockers")
    ]
    model_run_allowed = bool(
        allow_wam_model_run
        and _env_truthy(LOCAL_MODEL_GATE_ENV)
        and command_probe_candidates
        and rollout_input_manifest["status"] == "ready_for_model"
    )
    model_payload: dict[str, Any] = {}
    model_execution_detail = {
        "status": "blocked",
        "blockers": runtime_discovery.get("blockers", [])
        or ["blocked_local_wam_model_run_not_enabled"],
        "local_model_gate_enabled": _env_truthy(LOCAL_MODEL_GATE_ENV),
        "allow_wam_model_run": bool(allow_wam_model_run),
    }
    if model_run_allowed:
        selected = dict(command_probe_candidates[0])
        command_value = wam_model_command
        if not command_value:
            command_env = _string(selected.get("command_env"))
            command_value = os.getenv(command_env, "")
        model_payload, model_execution_detail = _run_local_wam_command(
            command=command_value or "",
            input_manifest_path=output_dir / "wam_rollout_input_manifest.json",
            output_path=output_dir / "wam_provider_output.json",
            candidate_id=_string(selected.get("candidate_id")),
            checkpoint_path=_string(selected.get("checkpoint_path")) or None,
            timeout_seconds=timeout_seconds,
        )

    rollouts = []
    raw_rollouts = model_payload.get("rollouts") or _mapping(
        model_payload.get("wam_generated_rollout_results")
    ).get("rollouts")
    missing_video_count = 0
    if isinstance(raw_rollouts, list):
        for item in raw_rollouts:
            if not isinstance(item, Mapping):
                continue
            row = dict(item)
            video_path = _rollout_video_path(row, base_dir=output_dir)
            if video_path and video_path.is_file():
                row["generated_video_path"] = str(video_path.resolve())
                rollouts.append(row)
            else:
                missing_video_count += 1
    payload_blockers = _string_list(model_payload.get("blockers"))
    detail_blockers = _string_list(model_execution_detail.get("blockers"))
    runtime_blockers = _string_list(runtime_discovery.get("blockers"))
    if payload_blockers:
        generated_blocker_set = set(payload_blockers + detail_blockers)
    elif detail_blockers:
        generated_blocker_set = set(detail_blockers)
    else:
        generated_blocker_set = set(detail_blockers + runtime_blockers)
    if missing_video_count:
        generated_blocker_set.add("blocked_generated_rollout_video_missing")
    generated_blockers = [] if rollouts else sorted(
        generated_blocker_set or {"blocked_missing_wam_runtime", "blocked_missing_wam_model_checkpoint"}
    )
    blocked_reason = None if rollouts else _wam_rollout_blocked_reason(generated_blockers)
    generated_status = "completed" if rollouts else blocked_reason
    generated_manifest = {
        "schema_version": "wam_generated_rollout_manifest.v1",
        "generated_at": generated,
        "status": generated_status,
        "selected_model_candidate": runtime_discovery.get("selected_candidate"),
        "action_conditioned_video_rollout_generated": bool(rollouts),
        "generated_rollout_count": len(rollouts),
        "model_execution_detail": model_execution_detail,
        "blockers": generated_blockers,
    }
    write_json(output_dir / "wam_generated_rollout_manifest.json", generated_manifest)
    generated_results = {
        "schema_version": "wam_generated_rollout_results.v1",
        "generated_at": generated,
        "status": "completed" if rollouts else "blocked",
        "rollout_count": len(rollouts),
        "rollouts": rollouts,
        "blocked_reason": blocked_reason,
        "blockers": generated_blockers,
    }
    write_json(output_dir / "wam_generated_rollout_results.json", generated_results)
    _write_jsonl(output_dir / "wam_generated_rollout_results.jsonl", rollouts)

    consistency = {
        "schema_version": "wam_consistency_checks.v1",
        "generated_at": generated,
        "status": "completed" if rollouts else "blocked",
        "forward_inverse_consistency_proven": False,
        "action_conditioned_video_rollout_generated": bool(rollouts),
        "wam_success_label_from_generated_video": False,
        "checks": [
            {
                "check_id": "forward_dynamics_consistency",
                "status": "blocked" if not rollouts else "requires_model_specific_scoring",
                "proven": False,
                "blocker": None if rollouts else "blocked_missing_generated_rollout",
            },
            {
                "check_id": "inverse_dynamics_consistency",
                "status": "blocked" if not rollouts else "requires_model_specific_scoring",
                "proven": False,
                "blocker": None if rollouts else "blocked_missing_generated_rollout",
            },
            {
                "check_id": "cross_view_consistency",
                "status": "blocked" if not rollouts else "requires_multiview_model_output",
                "proven": False,
                "blocker": None if rollouts else "blocked_missing_generated_rollout",
            },
            {
                "check_id": "test_time_consistency_uncertainty_termination",
                "status": "blocked" if not rollouts else "requires_uncertainty_estimates",
                "proven": False,
                "blocker": None if rollouts else "blocked_missing_generated_rollout",
            },
        ],
        "generated_rollout_termination_reason": blocked_reason
        if not rollouts
        else "model_output_available_needs_review",
        "model_rollout_confidence": None,
    }
    write_json(output_dir / "wam_consistency_checks.json", consistency)

    success_labels = {
        "schema_version": "wam_success_labels.v1",
        "generated_at": generated,
        "status": "blocked" if not rollouts else "requires_review",
        "wam_success_label_from_generated_video": False,
        "label_count": 0,
        "labels": [],
        "blockers": generated_blockers if not rollouts else ["requires_wam_success_review"],
    }
    write_json(output_dir / "wam_success_labels.json", success_labels)

    scorecard = {
        "schema_version": "wam_policy_scorecard.v1",
        "generated_at": generated,
        "status": "blocked" if not rollouts else "requires_review",
        "policy_count": len({row.get("policy_id") for row in rollouts if row.get("policy_id")}),
        "generated_rollout_count": len(rollouts),
        "success_label_count": 0,
        "score_source": "none_blocked" if not rollouts else "wam_generated_rollouts_pending_review",
        "blockers": generated_blockers if not rollouts else ["requires_wam_success_review"],
    }
    write_json(output_dir / "wam_policy_scorecard.json", scorecard)

    trace_binding = {
        "schema_version": "wam_evaluator_trace_binding.v1",
        "generated_at": generated,
        "source_mujoco_endpoint_eval_job_dir": str(input_dir),
        "source_paths": {
            "scenario_eval_matrix": str(input_dir / "scenario_eval_matrix.json"),
            "normalized_attempt_trace": str(input_dir / "normalized_attempt_trace.json"),
            "normalized_policy_action_trace_jsonl": str(
                input_dir / "normalized_policy_action_trace.jsonl"
            ),
            "g1_mujoco_locomotion_trace_jsonl": str(
                input_dir / "g1_mujoco_locomotion_trace.jsonl"
            ),
            "review_video_selection_manifest": str(
                input_dir / "review_video_selection_manifest.json"
            ),
        },
        "output_paths": {
            name: str(output_dir / f"{name}.json")
            for name in [
                "wam_model_runtime_discovery",
                "wam_rollout_input_manifest",
                "wam_action_conditioning_manifest",
                "wam_generated_rollout_manifest",
                "wam_generated_rollout_results",
                "wam_consistency_checks",
                "wam_success_labels",
                "wam_policy_scorecard",
                "wam_evaluator_truth_boundary",
                "policy_model_truth_boundary",
                "policy_model_endpoint_readiness_manifest",
                "policy_model_endpoint_creation_plan",
                "policy_cloud_gpu_setup_manifest",
                "local_model_source_tree_discovery",
            ]
        },
    }
    write_json(output_dir / "wam_evaluator_trace_binding.json", trace_binding)

    truth_boundary = {
        "schema_version": "wam_evaluator_truth_boundary.v1",
        "generated_at": generated,
        "status": "completed",
        "mujoco_source_job": str(input_dir),
        "mujoco_evidence_is_simulator_only": True,
        "learned_wam_model_ran": bool(rollouts),
        "oscar_cosmos_openvla_unitree_model_ran": bool(rollouts),
        "action_conditioned_video_rollout_generated": bool(rollouts),
        "wam_success_label_from_generated_video": False,
        "forward_inverse_consistency_proven": False,
        "fixture_policy_used_as_wam_model": False,
        "generated_outputs_are_raw_capture_evidence": False,
        "physical_robot_readiness_proven": False,
        "deployment_readiness_proven": False,
        "official_unitree_controller_proven": False,
        "blockers": generated_blockers,
    }
    write_json(output_dir / "wam_evaluator_truth_boundary.json", truth_boundary)
    policy_model_truth_boundary = {
        **truth_boundary,
        "schema_version": "policy_model_truth_boundary.v1",
        "policy_model_candidate_matrix": str(output_dir / "policy_model_candidate_matrix.json"),
        "replaceable_model_adapter_boundary": True,
    }
    write_json(output_dir / "policy_model_truth_boundary.json", policy_model_truth_boundary)

    summary = {
        "schema_version": WAM_EVALUATOR_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed" if rollout_input_manifest["status"] == "ready_for_model" else "blocked",
        "job_dir": str(output_dir),
        "input_job_dir": str(input_dir),
        "wam_generated_rollout_status": generated_manifest["status"],
        "wam_consistency_check_status": consistency["status"],
        "learned_wam_model_ran": bool(rollouts),
        "oscar_cosmos_openvla_unitree_model_ran": bool(rollouts),
        "blockers": generated_blockers,
        "artifact_paths": {
            "wam_model_runtime_discovery": str(output_dir / "wam_model_runtime_discovery.json"),
            "wam_rollout_input_manifest": str(output_dir / "wam_rollout_input_manifest.json"),
            "wam_action_conditioning_manifest": str(output_dir / "wam_action_conditioning_manifest.json"),
            "wam_generated_rollout_manifest": str(output_dir / "wam_generated_rollout_manifest.json"),
            "wam_generated_rollout_results": str(output_dir / "wam_generated_rollout_results.json"),
            "wam_consistency_checks": str(output_dir / "wam_consistency_checks.json"),
            "wam_success_labels": str(output_dir / "wam_success_labels.json"),
            "wam_policy_scorecard": str(output_dir / "wam_policy_scorecard.json"),
            "wam_evaluator_trace_binding": str(output_dir / "wam_evaluator_trace_binding.json"),
            "wam_evaluator_truth_boundary": str(output_dir / "wam_evaluator_truth_boundary.json"),
            "policy_model_truth_boundary": str(output_dir / "policy_model_truth_boundary.json"),
            "policy_model_endpoint_readiness_manifest": str(
                output_dir / "policy_model_endpoint_readiness_manifest.json"
            ),
            "policy_model_endpoint_creation_plan": str(
                output_dir / "policy_model_endpoint_creation_plan.json"
            ),
            "policy_cloud_gpu_setup_manifest": str(output_dir / "policy_cloud_gpu_setup_manifest.json"),
            "local_model_source_tree_discovery": str(
                output_dir / "local_model_source_tree_discovery.json"
            ),
        },
    }
    write_json(output_dir / "oscar_cosmos_wam_evaluator_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-job-dir", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path)
    parser.add_argument("--job-root", type=Path)
    parser.add_argument("--model-candidate", action="append", dest="model_candidates")
    parser.add_argument("--wam-model-command")
    parser.add_argument("--wam-model-checkpoint", type=Path)
    parser.add_argument("--allow-wam-model-run", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    summary = run_oscar_cosmos_wam_evaluator(
        input_job_dir=args.input_job_dir,
        job_dir=args.job_dir,
        job_root=args.job_root,
        model_candidates=args.model_candidates or DEFAULT_MODEL_CANDIDATES,
        wam_model_command=args.wam_model_command,
        wam_model_checkpoint=args.wam_model_checkpoint,
        allow_wam_model_run=args.allow_wam_model_run,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "status": summary.get("status"),
                "job_dir": summary.get("job_dir"),
                "wam_generated_rollout_status": summary.get("wam_generated_rollout_status"),
                "learned_wam_model_ran": summary.get("learned_wam_model_ran"),
                "blockers": summary.get("blockers"),
            },
            sort_keys=True,
        )
    )
    return 0 if summary.get("status") == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
