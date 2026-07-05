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
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .failure_diagnosis_contract import (
    FAILURE_LABEL_PROOF_EFFECT,
    dedupe as _dedupe_refs,
    evidence_refs as _failure_evidence_refs,
    failure_root_cause_category as _failure_root_cause_category,
    frame_or_clip_refs as _failure_frame_or_clip_refs,
    remediation_candidate as _failure_remediation_candidate,
    review_status_for_failure_label as _failure_review_status,
)
from .model_access_env import model_access_secret_status, normalize_model_access_env
from .policy_model_runtime_proofs import (
    discover_openvla_provider_smoke_proof,
    discover_unitree_unifolm_provider_smoke_proof,
)
from .wam_backend_strategy import (
    build_wam_backend_strategy_manifest,
    get_wam_backend_strategy,
)
from .wam_generated_video_review import (
    validate_generated_mp4_for_review,
    visual_smoke_generated_rollouts_for_review,
)
from .wam_score_claim_gate import (
    apply_wam_score_claim_gate,
    evaluate_wam_calibration_anchors,
    score_wam_rollout_set_consistency,
)


WAM_EVALUATOR_SCHEMA_VERSION = "oscar_cosmos_wam_evaluator.v1"
DEFAULT_MODEL_CANDIDATES = ("cosmos3_wam", "oscar_wam", "cosmos_wam")
LOCAL_MODEL_GATE_ENV = "BLUEPRINT_ALLOW_LOCAL_WAM_MODEL"
WAM_SUCCESS_LABEL_GATE_ENV = "BLUEPRINT_ALLOW_WAM_SUCCESS_LABELING"
WAM_SUCCESS_LABEL_COMMAND_ENV = "BLUEPRINT_WAM_SUCCESS_LABEL_COMMAND"
WAM_SUCCESS_LABEL_COMMAND_OUTPUT = "wam_success_labels.command.json"
WAM_CONSISTENCY_GATE_ENV = "BLUEPRINT_ALLOW_WAM_EPISODE_CONSISTENCY_SCORING"
WAM_CONSISTENCY_COMMAND_ENV = "BLUEPRINT_WAM_EPISODE_CONSISTENCY_COMMAND"
WAM_CONSISTENCY_COMMAND_OUTPUT = "wam_episode_consistency.command.json"
EVAL_READY_TASK_GROUNDING_ENV = "BLUEPRINT_EVAL_READY_TASK_GROUNDING"
EGOCENTRIC_WAM_INPUT_CAMERAS = ("head_pov", "torso_pov", "robot_pov")
DIAGNOSTIC_REVIEW_CAMERAS = ("third_person", "robot_follow", "overhead")

MODEL_RUNTIME_CONTRACTS = {
    "oscar_wam": {
        "command_envs": ("BLUEPRINT_OSCAR_WAM_COMMAND", "BLUEPRINT_OSCAR_WAM_PROVIDER_COMMAND"),
        "checkpoint_envs": ("BLUEPRINT_OSCAR_WAM_CHECKPOINT",),
        "runtime_role": "action_conditioned_world_model_rollout_generator",
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
    "unitree_unifolm_vla_policy": {
        "command_envs": ("BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND",),
        "checkpoint_envs": (
            "BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT",
            "BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT",
        ),
        "checkpoint_env_aliases": {
            "BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT": (
                "BLUEPRINT_UNITREE_UNIFOLM_POLICY_CHECKPOINT",
            ),
        },
        "runtime_role": "unitree_native_vla_policy_endpoint_candidate",
    },
    "unitree_unifolm_wma_policy": {
        "command_envs": ("BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND",),
        "checkpoint_envs": ("BLUEPRINT_UNITREE_UNIFOLM_WMA_CHECKPOINT",),
        "runtime_role": "unitree_native_world_model_action_policy_candidate",
    },
    "unitree_lerobot_policy": {
        "command_envs": ("BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",),
        "checkpoint_envs": ("BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT",),
        "runtime_role": "unitree_g1_hand_or_gripper_manipulation_policy_candidate",
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
        "source_urls": [
            "https://github.com/NVIDIA/Cosmos",
            "https://research.nvidia.com/labs/cosmos-lab/cosmos3/",
            "https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf",
        ],
        "auth_groups": ["huggingface", "ngc"],
        "cloud_gpu_required_without_local_gpu": True,
        "host_requirements": {
            "accelerator": "NVIDIA CUDA GPU",
            "note": "Cosmos 3 is preferred only as an explicitly configured adapter; model family choice is not rank-fidelity proof.",
        },
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
    "unitree_unifolm_vla_policy": {
        "provider": "unitree_unifolm_vla",
        "source_urls": [
            "https://github.com/unitreerobotics/unifolm-vla",
            "https://huggingface.co/unitreerobotics/UnifoLM-VLA-Base",
            "https://huggingface.co/unitreerobotics/UnifoLM-VLA-Libero",
            "https://unigen-x.github.io/unifolm-vla.github.io/",
        ],
        "auth_groups": ["huggingface"],
        "cloud_gpu_required_without_local_gpu": True,
        "host_requirements": {
            "accelerator": "CUDA GPU strongly recommended",
            "note": "Unitree-native VLA checkpoint; still requires a Blueprint/Unitree action decoder and endpoint command.",
        },
    },
    "unitree_unifolm_wma_policy": {
        "provider": "unitree_unifolm_wma",
        "source_urls": [
            "https://github.com/unitreerobotics/unifolm-world-model-action",
            "https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Dual",
            "https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Base",
        ],
        "auth_groups": ["huggingface"],
        "cloud_gpu_required_without_local_gpu": True,
        "host_requirements": {
            "accelerator": "CUDA GPU strongly recommended",
            "note": "Unitree-native WMA stack can provide world-model/action behavior, but the server/client scripts must be wrapped as a Blueprint policy endpoint.",
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
    "unitree_lerobot_policy": {
        "provider": "unitree_lerobot_g1_manipulation",
        "source_urls": [
            "https://github.com/unitreerobotics/unitree_lerobot",
            "https://huggingface.co/docs/lerobot/unitree_g1",
            "https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ToastedBread_Dataset",
            "https://github.com/unitreerobotics/unitree_sim_isaaclab",
        ],
        "auth_groups": ["huggingface"],
        "cloud_gpu_required_without_local_gpu": True,
        "host_requirements": {
            "accelerator": "CUDA GPU recommended for policy inference",
            "note": "Requires a trained Unitree G1 Dex1/Dex3/gripper LeRobot policy plus a Blueprint action adapter.",
        },
    },
}
ENDPOINT_READINESS_CANDIDATES = (
    "oscar_wam",
    "cosmos_wam",
    "cosmos3_wam",
    "openvla_policy",
    "unitree_unifolm_vla_policy",
    "unitree_unifolm_wma_policy",
    "unitree_lerobot_policy",
    "unitree_g1_policy",
)
MODEL_SOURCE_ROOT_ENVS = {
    "oscar_wam": "BLUEPRINT_OSCAR_WAM_SOURCE_ROOT",
    "cosmos_wam": "BLUEPRINT_COSMOS_WAM_SOURCE_ROOT",
    "cosmos3_wam": "BLUEPRINT_COSMOS3_WAM_SOURCE_ROOT",
    "openvla_policy": "BLUEPRINT_OPENVLA_POLICY_SOURCE_ROOT",
    "unitree_unifolm_vla_policy": "BLUEPRINT_UNITREE_UNIFOLM_VLA_SOURCE_ROOT",
    "unitree_unifolm_wma_policy": "BLUEPRINT_UNITREE_UNIFOLM_WMA_SOURCE_ROOT",
    "unitree_lerobot_policy": "BLUEPRINT_UNITREE_LEROBOT_ROOT",
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


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if any(marker in key_text.lower() for marker in ("token", "secret", "password", "key")):
                redacted[key_text] = "<redacted>"
            else:
                redacted[key_text] = _redact(child)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


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


def _source_unitree_controller_proof(
    input_job_dir: Path,
    *,
    locomotion_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    controller_truth = _load_json(input_job_dir / "controller_truth_boundary.json")
    same_scene = _load_json(input_job_dir / "same_scene_unitree_controller_backend_manifest.json")
    bridge = _load_json(input_job_dir / "unitree_controller_bridge_manifest.json")
    controller_truth_trusted = bool(
        controller_truth.get("schema_version") == "controller_truth_boundary.v1"
        and not _string_list(controller_truth.get("blockers"))
    )
    same_scene_trusted = bool(
        same_scene.get("schema_version")
        == "same_scene_unitree_rl_gym_controller_backend.v1"
        and same_scene.get("status") == "completed"
        and not _string_list(same_scene.get("blockers"))
    )
    bridge_trusted = bool(
        bridge.get("schema_version") == "unitree_controller_bridge_manifest.v1"
        and bridge.get("status") == "completed"
        and not _string_list(bridge.get("blockers"))
    )
    trace_rows_with_unitree_controller = sum(
        1 for row in locomotion_rows if row.get("official_unitree_controller_used") is True
    )
    trace_rows_with_freejoint_proxy = sum(
        1 for row in locomotion_rows if row.get("freejoint_proxy_used") is True
    )
    trace_rows_with_fall = sum(1 for row in locomotion_rows if row.get("fall_detected") is True)
    official_unitree_controller_used = bool(
        (
            controller_truth_trusted
            and controller_truth.get("official_unitree_controller_used") is True
            and controller_truth.get("official_policy_execution_proven") is True
        )
        or (
            same_scene_trusted
            and bridge_trusted
            and same_scene.get("official_unitree_controller_used") is True
            and bridge.get("official_unitree_controller_used") is True
        )
    )
    balanced_walking_controller_proven = bool(
        official_unitree_controller_used
        and not trace_rows_with_fall
        and (
            (
                controller_truth_trusted
                and controller_truth.get("balanced_walking_controller_proven") is True
            )
            or (
                same_scene_trusted
                and same_scene.get("balanced_walking_controller_proven") is True
            )
            or (
                bridge_trusted
                and bridge.get("balanced_walking_controller_proven") is True
            )
        )
    )
    realistic_navigation_policy_used = bool(
        (
            controller_truth_trusted
            and (
                controller_truth.get("realistic_navigation_policy_used") is True
                or controller_truth.get("realistic_navigation_policy_used_for_endpoint_rollouts")
                is True
            )
        )
        or (
            same_scene_trusted
            and same_scene.get("realistic_navigation_policy_used_for_endpoint_rollouts")
            is True
        )
        or (
            bridge_trusted
            and bridge.get("realistic_navigation_policy_used_for_endpoint_rollouts") is True
        )
    )
    freejoint_proxy_used = bool(
        (
            controller_truth_trusted
            and controller_truth.get("freejoint_proxy_used") is True
        )
        or trace_rows_with_freejoint_proxy
    )
    unitree_locomotion_policy_used = bool(
        official_unitree_controller_used
        and (
            (
                controller_truth_trusted
                and controller_truth.get("unitree_lower_body_locomotion_policy_used") is True
            )
            or (
                same_scene_trusted
                and (
                    same_scene.get("controller_backend") == "unitree_rl_gym"
                    or same_scene.get("backend_id")
                    == "unitree_rl_gym_same_scene_lower_body_policy"
                )
            )
        )
    )
    unitree_hand_manipulation_policy_used = bool(
        controller_truth_trusted
        and (
            controller_truth.get("unitree_hand_manipulation_policy_used") is True
            or controller_truth.get("unitree_lerobot_or_isaaclab_manipulation_policy_used")
            is True
        )
    )
    return {
        "schema_version": "source_unitree_controller_proof.v1",
        "status": "completed" if official_unitree_controller_used else "not_proven",
        "source_artifacts": {
            "controller_truth_boundary": str(input_job_dir / "controller_truth_boundary.json"),
            "same_scene_unitree_controller_backend_manifest": str(
                input_job_dir / "same_scene_unitree_controller_backend_manifest.json"
            ),
            "unitree_controller_bridge_manifest": str(
                input_job_dir / "unitree_controller_bridge_manifest.json"
            ),
            "g1_mujoco_locomotion_trace_jsonl": str(
                input_job_dir / "g1_mujoco_locomotion_trace.jsonl"
            ),
        },
        "trusted_artifact_checks": {
            "controller_truth_boundary_trusted": controller_truth_trusted,
            "same_scene_unitree_controller_backend_manifest_trusted": same_scene_trusted,
            "unitree_controller_bridge_manifest_trusted": bridge_trusted,
            "trace_rows_are_supporting_evidence_only": True,
        },
        "official_unitree_controller_used": official_unitree_controller_used,
        "official_unitree_controller_proven": official_unitree_controller_used,
        "official_policy_execution_proven": bool(
            controller_truth_trusted
            and controller_truth.get("official_policy_execution_proven") is True
            and official_unitree_controller_used
        ),
        "realistic_navigation_policy_used": realistic_navigation_policy_used,
        "realistic_navigation_policy_used_for_endpoint_rollouts": realistic_navigation_policy_used,
        "balanced_walking_controller_proven": balanced_walking_controller_proven,
        "freejoint_proxy_used": freejoint_proxy_used,
        "freejoint_proxy_used_for_endpoint_rollouts": freejoint_proxy_used,
        "unitree_locomotion_policy_used": unitree_locomotion_policy_used,
        "unitree_locomotion_policy_ran": bool(
            official_unitree_controller_used and unitree_locomotion_policy_used
        ),
        "unitree_locomotion_policy_kind": controller_truth.get("unitree_locomotion_policy_kind")
        or same_scene.get("backend_id"),
        "unitree_locomotion_policy_checkpoint_path": controller_truth.get(
            "unitree_locomotion_policy_checkpoint_path"
        )
        or same_scene.get("policy_path"),
        "unitree_locomotion_policy_config_path": controller_truth.get(
            "unitree_locomotion_policy_config_path"
        )
        or same_scene.get("config_path"),
        "trace_rows_with_unitree_controller": trace_rows_with_unitree_controller,
        "trace_rows_with_freejoint_proxy": trace_rows_with_freejoint_proxy,
        "trace_rows_with_fall": trace_rows_with_fall,
        "unitree_hand_manipulation_policy_used": unitree_hand_manipulation_policy_used,
        "unitree_lerobot_or_isaaclab_manipulation_policy_used": bool(
            controller_truth.get("unitree_lerobot_or_isaaclab_manipulation_policy_used")
        ),
        "unitree_g1_dexterous_manipulation_proven": False,
        "claim_boundary": {
            "unitree_locomotion_policy_is_mujoco_simulator_only": True,
            "unitree_locomotion_policy_is_not_vla_manipulation_policy": True,
            "unitree_locomotion_policy_does_not_prove_generated_world_rank_fidelity": True,
            "unitree_hand_or_dexterous_manipulation_policy_not_proven": (
                not unitree_hand_manipulation_policy_used
            ),
        },
    }


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


def _eval_ready_grounding_candidates(input_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    env_path = _string(os.getenv(EVAL_READY_TASK_GROUNDING_ENV))
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            input_dir / "eval_ready_task_grounding.json",
            input_dir / "simulation_automation" / "eval_ready_task_grounding.json",
            input_dir / "pipeline" / "simulation_automation" / "eval_ready_task_grounding.json",
            input_dir.parent / "simulation_automation" / "eval_ready_task_grounding.json",
            input_dir.parent / "pipeline" / "simulation_automation" / "eval_ready_task_grounding.json",
            input_dir.parent.parent / "pipeline" / "simulation_automation" / "eval_ready_task_grounding.json",
        ]
    )
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


def _load_eval_ready_task_grounding(input_dir: Path) -> tuple[dict[str, Any], Path | None]:
    for candidate in _eval_ready_grounding_candidates(input_dir):
        payload = _load_json(candidate)
        if payload.get("schema_version") == "eval_ready_task_grounding.v1":
            return payload, candidate
    return {}, None


def _grounding_artifact_path(grounding: Mapping[str, Any], key: str) -> Path | None:
    artifacts = _mapping(grounding.get("generated_artifacts"))
    value = _string(artifacts.get(key))
    if not value:
        nested = _mapping(grounding.get(key))
        value = _string(nested.get("path"))
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_file() else None


def _copy_grounding_support_artifacts(
    *,
    grounding: Mapping[str, Any],
    grounding_path: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    if grounding_path is not None and grounding_path.is_file():
        target = output_dir / "eval_ready_task_grounding.json"
        if grounding_path.resolve() != target.resolve():
            shutil.copyfile(grounding_path, target)
        copied["eval_ready_task_grounding"] = str(target)
    for key, filename in (
        ("camera_calibration_quality_gate", "camera_calibration_quality_gate.json"),
        ("robot_fk_projection_manifest", "robot_fk_projection_manifest.json"),
        ("robot_fk_projected_skeleton_trace", "robot_fk_projected_skeleton_trace.jsonl"),
        ("handle_proxy_state_check", "handle_proxy_state_check.json"),
    ):
        source = _grounding_artifact_path(grounding, key)
        if source is None:
            continue
        target = output_dir / filename
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        copied[key] = str(target)
    return copied


def _grounding_enriched_task_prompts(
    *,
    matrix_runs: Sequence[Mapping[str, Any]],
    grounding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    task = _mapping(grounding.get("task"))
    target_prompts = _string_list(task.get("target_prompts_for_object_index_backends"))
    selected_target = _mapping(grounding.get("selected_task_target"))
    success_check_plan = _mapping(grounding.get("success_check_plan"))
    grounding_task_text = _string(task.get("task_text"))
    grounding_task_id = _string(task.get("task_id"))
    rows: list[dict[str, Any]] = []
    for run in matrix_runs:
        task_prompt = _string(run.get("task_prompt")) or grounding_task_text
        rows.append(
            {
                "scenario_eval_run_id": run.get("scenario_eval_run_id"),
                "task_id": run.get("task_id") or grounding_task_id,
                "spawn_id": run.get("spawn_id"),
                "task_prompt": task_prompt,
                "eval_ready_task_grounding_used": bool(grounding),
                "target_prompts": target_prompts,
                "selected_task_target": selected_target or None,
                "success_check_plan": success_check_plan or None,
            }
        )
    if not rows and grounding:
        rows.append(
            {
                "scenario_eval_run_id": None,
                "task_id": grounding_task_id,
                "spawn_id": None,
                "task_prompt": grounding_task_text,
                "eval_ready_task_grounding_used": True,
                "target_prompts": target_prompts,
                "selected_task_target": selected_target or None,
                "success_check_plan": success_check_plan or None,
            }
        )
    return rows


def _build_prediction_outcome_correlation_ledger(
    *,
    generated_at: str,
    input_dir: Path,
    output_dir: Path,
    rollouts: Sequence[Mapping[str, Any]],
    success_labels: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    consistency: Mapping[str, Any],
    visual_smoke: Mapping[str, Any],
    grounding: Mapping[str, Any],
) -> dict[str, Any]:
    outcome_sources = [
        input_dir / "deployment_outcome_ledger.json",
        input_dir / "prediction_outcome_ledger.json",
        input_dir / "actual_outcome_ledger.json",
    ]
    outcomes: list[dict[str, Any]] = []
    for path in outcome_sources:
        payload = _load_json(path)
        raw_rows = payload.get("outcomes") or payload.get("records") or payload.get("items")
        if isinstance(raw_rows, list):
            for row in raw_rows:
                if isinstance(row, Mapping):
                    outcomes.append({**dict(row), "source_path": str(path)})
    labels_by_rollout = {
        _string(row.get("rollout_id")): dict(row)
        for row in success_labels.get("labels", []) or []
        if isinstance(row, Mapping)
    }
    state_check = _mapping(grounding.get("handle_proxy_state_check"))
    records: list[dict[str, Any]] = []
    for rollout in rollouts:
        rollout_id = _string(rollout.get("rollout_id"))
        run_id = _string(rollout.get("scenario_eval_run_id"))
        task_id = _string(rollout.get("task_id"))
        matching_outcomes = [
            row
            for row in outcomes
            if _string(row.get("scenario_eval_run_id")) == run_id
            or (_string(row.get("task_id")) == task_id and task_id)
        ]
        records.append(
            {
                "rollout_id": rollout_id,
                "scenario_eval_run_id": run_id or None,
                "task_id": task_id or None,
                "wam_generated_video_path": rollout.get("generated_video_path"),
                "visual_smoke_status": visual_smoke.get("status"),
                "visual_rollout_useful_for_success_review": _mapping(
                    visual_smoke.get("claim_boundary")
                ).get("visual_rollout_useful_for_task_success_review"),
                "wam_success_label": labels_by_rollout.get(rollout_id),
                "score_source": scorecard.get("score_source"),
                "forward_inverse_consistency_proven": consistency.get(
                    "forward_inverse_consistency_proven"
                ),
                "lightweight_state_check": state_check or None,
                "matched_real_world_outcome_count": len(matching_outcomes),
                "matched_real_world_outcomes": matching_outcomes,
                "calibration_status": _mapping(
                    grounding.get("camera_calibration_quality_gate")
                ).get("status"),
                "robot_projection_ready": _mapping(grounding.get("readiness")).get(
                    "robot_projection_ready"
                ),
            }
        )
    ledger = {
        "schema_version": "wam_prediction_outcome_correlation_ledger.v1",
        "generated_at": generated_at,
        "status": "completed_with_real_world_outcomes"
        if any(row["matched_real_world_outcome_count"] for row in records)
        else "awaiting_real_world_outcomes",
        "source_mujoco_endpoint_eval_job_dir": str(input_dir),
        "prediction_record_count": len(records),
        "matched_real_world_outcome_count": sum(
            row["matched_real_world_outcome_count"] for row in records
        ),
        "records": records,
        "outcome_source_paths_checked": [str(path) for path in outcome_sources],
        "claim_boundary": {
            "correlation_ledger_does_not_upgrade_current_rollout_claims": True,
            "real_world_outcome_required_for_calibration": True,
            "generated_rollout_predictions_are_model_derived_support_artifacts": True,
        },
    }
    write_json(output_dir / "wam_prediction_outcome_correlation_ledger.json", ledger)
    return ledger


def _first_configured_env(env_names: Sequence[str]) -> tuple[str | None, str | None]:
    for env_name in env_names:
        value = os.getenv(env_name)
        if value:
            return env_name, value
    return None, None


def _required_checkpoint_env_statuses(env_names: Sequence[str]) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for env_name in env_names:
        value = os.getenv(env_name)
        statuses.append(
            {
                "env": env_name,
                "configured": bool(value),
                "exists": _checkpoint_available(value),
                "path": str(Path(value).expanduser()) if value else None,
            }
        )
    return statuses


def _checkpoint_env_aliases(contract: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    raw = contract.get("checkpoint_env_aliases", {})
    if not isinstance(raw, Mapping):
        return {}
    aliases: dict[str, tuple[str, ...]] = {}
    for env_name, alias_values in raw.items():
        if isinstance(alias_values, str):
            aliases[str(env_name)] = (alias_values,)
        elif isinstance(alias_values, Sequence):
            aliases[str(env_name)] = tuple(
                str(value) for value in alias_values if str(value).strip()
            )
    return aliases


def _checkpoint_env_names_with_aliases(contract: Mapping[str, Any]) -> tuple[str, ...]:
    aliases = _checkpoint_env_aliases(contract)
    names: list[str] = []
    for env_name in tuple(contract.get("checkpoint_envs", ())) or ():
        names.append(str(env_name))
        names.extend(aliases.get(str(env_name), ()))
    return tuple(dict.fromkeys(names))


def _required_checkpoint_env_statuses_for_contract(
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    aliases = _checkpoint_env_aliases(contract)
    for env_name in tuple(contract.get("checkpoint_envs", ())) or ():
        accepted_envs = (str(env_name), *aliases.get(str(env_name), ()))
        configured_env, value = _first_configured_env(accepted_envs)
        statuses.append(
            {
                "env": str(env_name),
                "env_aliases": list(aliases.get(str(env_name), ())),
                "accepted_envs": list(accepted_envs),
                "configured_env": configured_env,
                "configured": bool(value),
                "exists": _checkpoint_available(value),
                "path": str(Path(value).expanduser()) if value else None,
            }
        )
    return statuses


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


def _provider_command_delegates_checkpoint(candidate_id: str, command: str | None) -> bool:
    command_text = _string(command)
    if not command_text:
        return False
    provider_adapter_markers = {
        "oscar_wam": (
            "blueprint_pipeline.oscar_wam_provider_command_adapter",
            "blueprint-run-oscar-wam-provider-command-adapter",
        ),
    }
    return any(marker in command_text for marker in provider_adapter_markers.get(candidate_id, ()))


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
    local_model_gate_enabled_override: bool | None = None,
) -> dict[str, Any]:
    normalize_model_access_env()
    model_access = model_access_secret_status()
    host_probe = _local_host_probe()
    local_model_gate_enabled = (
        bool(local_model_gate_enabled_override)
        if local_model_gate_enabled_override is not None
        else _env_truthy(LOCAL_MODEL_GATE_ENV)
    )
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        contract = MODEL_RUNTIME_CONTRACTS.get(candidate, {})
        source_hint = MODEL_SOURCE_HINTS.get(candidate, {})
        backend_strategy = get_wam_backend_strategy(candidate)
        command_env, command = _first_configured_env(contract.get("command_envs", ()))
        checkpoint_env, checkpoint = _first_configured_env(
            _checkpoint_env_names_with_aliases(contract)
        )
        checkpoint_env_statuses = _required_checkpoint_env_statuses_for_contract(contract)
        selected_explicit_candidate = explicit_candidate_id or (candidates[0] if candidates else None)
        if explicit_command and candidate == (explicit_candidate_id or candidate):
            command_env = "cli:--wam-model-command"
            command = explicit_command
        if explicit_checkpoint and candidate == selected_explicit_candidate:
            checkpoint_env = "cli:--wam-model-checkpoint"
            checkpoint = str(explicit_checkpoint)
            explicit_checkpoint_status = {
                "env": checkpoint_env,
                "configured": True,
                "exists": _checkpoint_available(checkpoint),
                "path": str(Path(checkpoint).expanduser()),
            }
            if len(checkpoint_env_statuses) <= 1:
                checkpoint_env_statuses = [explicit_checkpoint_status]
            else:
                checkpoint_env_statuses = [
                    explicit_checkpoint_status,
                    *checkpoint_env_statuses[1:],
                ]
        command_ok = _command_available(command)
        local_checkpoint_ok = bool(
            checkpoint_env_statuses
            and all(row["configured"] and row["exists"] for row in checkpoint_env_statuses)
        )
        checkpoint_delegated_to_provider = bool(
            command_ok and _provider_command_delegates_checkpoint(candidate, command)
        )
        checkpoint_ok = bool(local_checkpoint_ok or checkpoint_delegated_to_provider)
        local_source = _local_source_tree_probe(
            candidate,
            command_configured=bool(command),
            checkpoint_configured=any(row["configured"] for row in checkpoint_env_statuses),
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
        if not checkpoint_delegated_to_provider:
            for status in checkpoint_env_statuses:
                checkpoint_status_env = _string(status.get("env")) or "MODEL_CHECKPOINT"
                if not status.get("configured"):
                    missing.append(f"set_{checkpoint_status_env}_to_local_checkpoint_path")
                elif not status.get("exists"):
                    missing.append("download_or_mount_configured_model_checkpoint_path")
                    missing.append(f"download_or_mount_configured_{checkpoint_status_env}_path")
        for group_id, ready in auth_ready.items():
            if not ready:
                missing.append(f"configure_file_based_{group_id}_auth")
        if candidate.endswith("_wam") and not local_model_gate_enabled:
            missing.append(f"set_{LOCAL_MODEL_GATE_ENV}=true")

        auth_values = list(auth_ready.values())
        endpoint_wrapper_can_be_created = command_ok
        real_model_runtime_ready = bool(
            command_ok
            and checkpoint_ok
            and (all(auth_values) if auth_values else True)
            and (not candidate.endswith("_wam") or local_model_gate_enabled)
        )
        rows.append(
            {
                "candidate_id": candidate,
                "runtime_role": contract.get("runtime_role", "replaceable_model_adapter"),
                "backend_strategy": backend_strategy,
                "backend_recommendation_tier": backend_strategy.get(
                    "recommendation_tier"
                ),
                "preferred_for_new_configured_learned_wam": bool(
                    backend_strategy.get("preferred_for_new_configured_learned_wam")
                ),
                "status": "ready_for_real_model_endpoint"
                if real_model_runtime_ready
                else "blocked",
                "command_envs": list(contract.get("command_envs", ())),
                "configured_command_env": command_env,
                "command_configured": bool(command),
                "command_available": command_ok,
                "command_value_redacted": "<configured>" if command else None,
                "checkpoint_envs": list(contract.get("checkpoint_envs", ())),
                "checkpoint_env_aliases": {
                    key: list(values)
                    for key, values in _checkpoint_env_aliases(contract).items()
                },
                "configured_checkpoint_env": checkpoint_env,
                "checkpoint_configured": bool(checkpoint),
                "checkpoint_exists": local_checkpoint_ok,
                "checkpoint_requirement_satisfied_by_provider_runtime": checkpoint_delegated_to_provider,
                "checkpoint_requirement_satisfied": checkpoint_ok,
                "checkpoint_path": str(Path(checkpoint).expanduser()) if checkpoint else None,
                "required_checkpoint_envs": checkpoint_env_statuses,
                "missing_required_checkpoint_envs": [
                    row["env"]
                    for row in checkpoint_env_statuses
                    if not checkpoint_delegated_to_provider
                    and (not row["configured"] or not row["exists"])
                ],
                "endpoint_wrapper_can_be_created": endpoint_wrapper_can_be_created,
                "real_model_runtime_ready": real_model_runtime_ready,
                "model_access_auth_groups": auth_groups,
                "model_access_auth_ready": auth_ready,
                "local_model_gate_env": LOCAL_MODEL_GATE_ENV
                if candidate.endswith("_wam")
                else None,
                "local_model_gate_enabled": local_model_gate_enabled
                if candidate.endswith("_wam")
                else None,
                "local_model_gate_enabled_by_cli_or_function_flag": bool(
                    local_model_gate_enabled_override and not _env_truthy(LOCAL_MODEL_GATE_ENV)
                )
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
            "generated_world_rank_fidelity_result_proven": False,
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
        checkpoint_exists = bool(
            row.get("checkpoint_requirement_satisfied", row.get("checkpoint_exists"))
        )
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
                "http_wrapper_without_real_model_claim": bool(
                    endpoint_wrapper_can_be_created and not real_ready
                ),
                "http_wrapper_blocked_without_command": not endpoint_wrapper_can_be_created,
                "real_model_claim_blocked_without_checkpoint_or_auth_or_gate": bool(
                    endpoint_wrapper_can_be_created and not real_ready
                ),
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
    missing_command_count = sum(1 for row in rows if row["http_wrapper_blocked_without_command"])
    wrapper_only_count = sum(1 for row in rows if row["http_wrapper_without_real_model_claim"])
    wam_rows = [row for row in rows if str(row["candidate_id"]).endswith("_wam")]
    policy_rows = [
        row
        for row in rows
        if row["candidate_id"]
        in {
            "openvla_policy",
            "unitree_unifolm_vla_policy",
            "unitree_unifolm_wma_policy",
            "unitree_lerobot_policy",
            "unitree_g1_policy",
        }
    ]
    wam_rollout_ready_count = sum(
        1 for row in wam_rows if row["can_run_wam_generated_rollout_now"]
    )
    vla_manipulation_ready_count = sum(
        1
        for row in rows
        if row["candidate_id"]
        in {
            "openvla_policy",
            "unitree_unifolm_vla_policy",
            "unitree_unifolm_wma_policy",
            "unitree_lerobot_policy",
        }
        and row["can_claim_real_model_endpoint_now"]
    )
    unitree_unifolm_policy_ready_count = sum(
        1
        for row in rows
        if row["candidate_id"]
        in {"unitree_unifolm_vla_policy", "unitree_unifolm_wma_policy"}
        and row["can_claim_real_model_endpoint_now"]
    )
    unitree_lerobot_policy_ready_count = sum(
        1
        for row in rows
        if row["candidate_id"] == "unitree_lerobot_policy"
        and row["can_claim_real_model_endpoint_now"]
    )
    g1_policy_ready_count = sum(
        1
        for row in rows
        if row["candidate_id"] == "unitree_g1_policy"
        and row["can_claim_real_model_endpoint_now"]
    )
    closed_loop_prerequisites_configured = bool(
        wam_rollout_ready_count and vla_manipulation_ready_count
    )
    closed_loop_wam_policy_ready = False
    closed_loop_blockers = []
    if not wam_rollout_ready_count:
        closed_loop_blockers.append("blocked_no_ready_action_conditioned_wam_rollout_provider")
    if not vla_manipulation_ready_count:
        closed_loop_blockers.append("blocked_no_ready_vla_manipulation_policy_endpoint")
    closed_loop_blockers.append("blocked_closed_loop_wam_policy_requery_not_yet_proven")
    return {
        "schema_version": "policy_model_endpoint_creation_plan.v1",
        "generated_at": generated_at,
        "status": "ready_for_real_model_endpoint" if real_ready_count else "blocked",
        "http_wrapper_binary_available": True,
        "readiness_layer_summary": {
            "reference_endpoint_wrapper_ready": True,
            "reference_endpoint_real_model_claim_allowed": False,
            "reference_endpoint_proves": ["HTTP/auth/JSON observation-action plumbing"],
            "wam_rollout_provider_ready_candidate_count": wam_rollout_ready_count,
            "wam_rollout_provider_ready": bool(wam_rollout_ready_count),
            "real_policy_action_endpoint_ready_candidate_count": sum(
                1 for row in policy_rows if row["can_claim_real_model_endpoint_now"]
            ),
            "vla_manipulation_policy_ready_candidate_count": vla_manipulation_ready_count,
            "unitree_unifolm_policy_ready_candidate_count": unitree_unifolm_policy_ready_count,
            "unitree_lerobot_policy_ready_candidate_count": unitree_lerobot_policy_ready_count,
            "unitree_g1_policy_ready_candidate_count": g1_policy_ready_count,
            "closed_loop_wam_policy_endpoint_ready": False,
            "closed_loop_wam_policy_endpoint_prerequisites_configured": (
                closed_loop_prerequisites_configured
            ),
            "closed_loop_wam_policy_endpoint_contract_defined": True,
            "closed_loop_wam_policy_endpoint_blockers": closed_loop_blockers,
            "why_wam_evaluator_exists_before_policy_ready": [
                "validates rollout input/output contracts",
                "validates model-runtime packaging and provenance capture",
                "validates generated-video review and blocker reporting",
                "does not prove manipulation until a real policy endpoint emits task actions",
            ],
            "claim_boundary": {
                "wam_rollout_provider_ready_is_not_robot_policy_ready": True,
                "reference_endpoint_ready_is_not_vla_manipulation_proof": True,
                "closed_loop_policy_wam_requery_requires_runtime_artifact": True,
            },
        },
        "can_create_http_wrapper_for_configured_commands_now": bool(wrapper_ready_count),
        "can_create_real_model_endpoint_now": bool(real_ready_count),
        "real_model_ready_candidate_count": real_ready_count,
        "wam_rollout_provider_ready_candidate_count": wam_rollout_ready_count,
        "vla_manipulation_policy_ready_candidate_count": vla_manipulation_ready_count,
        "unitree_unifolm_policy_ready_candidate_count": unitree_unifolm_policy_ready_count,
        "unitree_lerobot_policy_ready_candidate_count": unitree_lerobot_policy_ready_count,
        "unitree_g1_policy_ready_candidate_count": g1_policy_ready_count,
        "closed_loop_wam_policy_endpoint_ready": False,
        "endpoint_wrapper_ready_candidate_count": wrapper_ready_count,
        "endpoint_wrapper_missing_command_candidate_count": missing_command_count,
        "endpoint_wrapper_only_not_real_model_candidate_count": wrapper_only_count,
        "endpoint_creation_modes": [
            {
                "mode": "reference_endpoint_wrapper",
                "can_create_now": True,
                "proves": ["HTTP/auth/JSON observation-action plumbing"],
                "does_not_prove": ["learned WAM/VLA inference", "checkpoint execution"],
            },
            {
                "mode": "model_http_wrapper",
                "can_create_now": bool(wrapper_ready_count),
                "requires": ["runnable adapter command"],
                "proves": ["HTTP wrapper can invoke the configured command"],
                "does_not_prove": ["model weights loaded", "generated rollout quality"],
            },
            {
                "mode": "real_model_endpoint",
                "can_create_now": bool(real_ready_count),
                "requires": [
                    "runnable adapter command",
                    "local checkpoint or mounted weights",
                    "required file-based model-source auth",
                    "explicit local/cloud run gates",
                ],
                "proves": [
                    "configured command and checkpoint are ready for model execution"
                ],
                "does_not_prove": [
                    "task success",
                    "forward/inverse consistency",
                    "generated-world rank fidelity",
                ],
            },
            {
                "mode": "closed_loop_wam_plus_policy_endpoint",
                "can_create_now": closed_loop_wam_policy_ready,
                "requires": [
                    "ready action-conditioned WAM rollout provider",
                    "ready VLA or Unitree manipulation policy endpoint",
                    "scheduler that feeds WAM-generated next observations back to the policy",
                    "success judge that scores reviewable generated or simulated video",
                ],
                "proves": [
                    "policy and WAM can exchange observations/actions through Blueprint contracts"
                ],
                "does_not_prove": [
                    "generated-world rank fidelity",
                    "task success without reviewable video and judge output",
                ],
                "blockers": closed_loop_blockers,
            },
        ],
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
    local_model_gate_enabled_override: bool | None = None,
) -> dict[str, Any]:
    normalize_model_access_env()
    model_access = model_access_secret_status()
    cloud_gpu_setup = discover_cloud_gpu_setup(generated_at=generated_at)
    local_source_discovery = discover_local_model_source_trees(
        generated_at=generated_at,
        candidates=tuple(dict.fromkeys(tuple(candidates) + ENDPOINT_READINESS_CANDIDATES)),
    )
    host_probe = _local_host_probe()
    local_model_gate_enabled = (
        bool(local_model_gate_enabled_override)
        if local_model_gate_enabled_override is not None
        else _env_truthy(LOCAL_MODEL_GATE_ENV)
    )
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        contract = MODEL_RUNTIME_CONTRACTS.get(candidate, {})
        source_hint = MODEL_SOURCE_HINTS.get(candidate, {})
        backend_strategy = get_wam_backend_strategy(candidate)
        command_env, command = _first_configured_env(contract.get("command_envs", ()))
        checkpoint_envs = tuple(contract.get("checkpoint_envs", ()))
        checkpoint_env, checkpoint = _first_configured_env(
            _checkpoint_env_names_with_aliases(contract)
        )
        checkpoint_env_statuses = _required_checkpoint_env_statuses_for_contract(contract)
        selected_explicit_candidate = explicit_candidate_id or (candidates[0] if candidates else None)
        if explicit_command and candidate == selected_explicit_candidate:
            command_env = "cli:--wam-model-command"
            command = explicit_command
        if explicit_checkpoint and candidate == selected_explicit_candidate:
            checkpoint_env = "cli:--wam-model-checkpoint"
            checkpoint = str(explicit_checkpoint)
            checkpoint_env_statuses = [
                {
                    "env": checkpoint_env,
                    "configured": True,
                    "exists": _checkpoint_available(checkpoint),
                    "path": str(Path(checkpoint).expanduser()),
                }
            ]
        command_ok = _command_available(command)
        local_checkpoint_ok = bool(
            checkpoint_env_statuses
            and all(row["configured"] and row["exists"] for row in checkpoint_env_statuses)
        )
        checkpoint_delegated_to_provider = bool(
            command_ok and _provider_command_delegates_checkpoint(candidate, command)
        )
        checkpoint_ok = bool(local_checkpoint_ok or checkpoint_delegated_to_provider)
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
                "backend_strategy": backend_strategy,
                "backend_recommendation_tier": backend_strategy.get(
                    "recommendation_tier"
                ),
                "preferred_for_new_configured_learned_wam": bool(
                    backend_strategy.get("preferred_for_new_configured_learned_wam")
                ),
                "status": row_status,
                "command_env": command_env,
                "command_configured": bool(command),
                "command_available": command_ok,
                "command_value_redacted": "<configured>" if command else None,
                "checkpoint_env": checkpoint_env,
                "checkpoint_envs": list(checkpoint_envs),
                "checkpoint_env_aliases": {
                    key: list(values)
                    for key, values in _checkpoint_env_aliases(contract).items()
                },
                "checkpoint_configured": bool(checkpoint),
                "checkpoint_exists": local_checkpoint_ok,
                "checkpoint_requirement_satisfied_by_provider_runtime": checkpoint_delegated_to_provider,
                "checkpoint_requirement_satisfied": checkpoint_ok,
                "checkpoint_path": str(Path(checkpoint).expanduser()) if checkpoint else None,
                "required_checkpoint_envs": checkpoint_env_statuses,
                "missing_required_checkpoint_envs": [
                    row["env"]
                    for row in checkpoint_env_statuses
                    if not checkpoint_delegated_to_provider
                    and (not row["configured"] or not row["exists"])
                ],
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
        "local_model_gate_enabled": local_model_gate_enabled,
        "local_model_gate_enabled_by_cli_or_function_flag": bool(
            local_model_gate_enabled_override and not _env_truthy(LOCAL_MODEL_GATE_ENV)
        ),
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


def _skipped_openvla_provider_smoke_proof(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "openvla_provider_smoke_proof.v1",
        "status": "skipped",
        "provider_smoke_completed": False,
        "job_dir": None,
        "summary_path": None,
        "output_path": None,
        "openvla_model_executed": False,
        "openvla_model_loaded": False,
        "openvla_predict_action_invoked": False,
        "policy_action_model_command_ran": False,
        "openvla_policy_action_command_ran": False,
        "policy_action_model_provider_smoke_imported": False,
        "openvla_policy_action_command_imported": False,
        "action": None,
        "model_execution_scope": None,
        "endpoint_closed_loop_policy_proven": False,
        "unitree_g1_dexterous_manipulation_proven": False,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "blockers": [reason],
    }


def _candidate_matrix(
    generated_at: str,
    *,
    openvla_provider_smoke_proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    openvla_proof = _mapping(openvla_provider_smoke_proof)
    openvla_provider_smoke_completed = bool(openvla_proof.get("provider_smoke_completed"))
    oscar_strategy = get_wam_backend_strategy("oscar_wam")
    cosmos25_strategy = get_wam_backend_strategy("cosmos_wam")
    cosmos3_strategy = get_wam_backend_strategy("cosmos3_wam")
    cosmos3_super_strategy = get_wam_backend_strategy("cosmos3_super")
    cosmos3_edge_strategy = get_wam_backend_strategy("cosmos3_edge")
    return {
        "schema_version": "policy_model_candidate_matrix.v1",
        "generated_at": generated_at,
        "status": "adapter_boundary_defined",
        "preferred_configured_learned_wam_backend_candidate": "cosmos3_wam",
        "preferred_configured_backend_is_not_permanent_dependency": True,
        "candidates": [
            {
                "id": "oscar_wam",
                "runtime_role": "action_conditioned_world_model_rollout_generator",
                "backend_strategy": oscar_strategy,
                "command_env": "BLUEPRINT_OSCAR_WAM_COMMAND",
                "checkpoint_env": "BLUEPRINT_OSCAR_WAM_CHECKPOINT",
                "auth_file_envs": ["HF_TOKEN_FILE", "HUGGINGFACE_HUB_TOKEN_FILE"],
                "cloud_gpu_provider_options": ["runpod", "vast"],
            },
            {
                "id": "cosmos_wam",
                "runtime_role": "world_video_rollout_or_review_substrate",
                "backend_strategy": cosmos25_strategy,
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
                "id": "cosmos3_wam",
                "runtime_role": "preferred_configured_world_action_model_evaluator_candidate",
                "backend_strategy": cosmos3_strategy,
                "command_env": "BLUEPRINT_COSMOS3_WAM_COMMAND",
                "checkpoint_env": "BLUEPRINT_COSMOS3_WAM_CHECKPOINT",
                "source_urls": [
                    "https://github.com/NVIDIA/Cosmos",
                    "https://research.nvidia.com/labs/cosmos-lab/cosmos3/",
                    "https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf",
                ],
                "auth_file_envs": ["HF_TOKEN_FILE", "NGC_API_KEY_FILE"],
                "cloud_gpu_provider_options": ["runpod", "vast"],
                "preferred_for_new_configured_learned_wam": True,
                "auto_run_allowed_without_gate": False,
                "claim_boundary": {
                    "cosmos3_preference_is_not_universal_grading_proof": True,
                    "requires_adapter_calibration_and_external_consistency_scorer": True,
                    "generated_world_rank_fidelity_result_proven": False,
                },
            },
            {
                "id": "cosmos3_super",
                "runtime_role": "high_cost_adjudication_candidate",
                "backend_strategy": cosmos3_super_strategy,
                "default_local_runtime_candidate": False,
                "auto_run_allowed_without_gate": False,
                "claim_boundary": {
                    "high_cost_adjudication_candidate_not_default_local_path": True,
                    "generated_world_rank_fidelity_result_proven": False,
                },
            },
            {
                "id": "cosmos3_edge",
                "runtime_role": "announced_edge_candidate_not_default",
                "backend_strategy": cosmos3_edge_strategy,
                "release_status_from_primary_source": (
                    "technical_report_says_included_in_later_release"
                ),
                "default_local_runtime_candidate": False,
                "treat_as_released_default": False,
                "claim_boundary": {
                    "cosmos3_edge_treated_as_released_default": False,
                    "generated_world_rank_fidelity_result_proven": False,
                },
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
                "provider_smoke_completed": openvla_provider_smoke_completed,
                "provider_smoke_job_dir": openvla_proof.get("job_dir"),
                "openvla_model_executed": bool(openvla_proof.get("openvla_model_executed")),
                "policy_action_model_command_ran": bool(
                    openvla_proof.get("policy_action_model_command_ran")
                ),
                "openvla_policy_action_command_ran": bool(
                    openvla_proof.get("openvla_policy_action_command_ran")
                ),
                "policy_action_model_provider_smoke_imported": bool(
                    openvla_proof.get("policy_action_model_provider_smoke_imported")
                ),
                "openvla_policy_action_command_imported": bool(
                    openvla_proof.get("openvla_policy_action_command_imported")
                ),
                "last_provider_action": openvla_proof.get("action"),
                "endpoint_closed_loop_policy_proven": False,
                "unitree_g1_dexterous_manipulation_proven": False,
                "claim_boundary": {
                    "provider_smoke_is_model_execution_proof": openvla_provider_smoke_completed,
                    "provider_smoke_is_not_closed_loop_endpoint_control": True,
                    "provider_smoke_is_not_dexterous_manipulation_proof": True,
                },
            },
            {
                "id": "unitree_unifolm_vla_policy",
                "runtime_role": "unitree_native_vla_policy_endpoint_candidate",
                "command_env": "BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND",
                "checkpoint_env": "BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT",
                "vlm_checkpoint_env": "BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT",
                "source_root_env": "BLUEPRINT_UNITREE_UNIFOLM_VLA_SOURCE_ROOT",
                "source_urls": [
                    "https://github.com/unitreerobotics/unifolm-vla",
                    "https://huggingface.co/unitreerobotics/UnifoLM-VLA-Base",
                    "https://huggingface.co/unitreerobotics/UnifoLM-VLA-Libero",
                ],
                "known_public_checkpoint_files": [
                    "unitreerobotics/UnifoLM-VLA-Base:checkpoints/pytorch_model.pt",
                    "unitreerobotics/UnifoLM-VLA-Libero:checkpoints/pytorch_model.pt",
                    "unitreerobotics/UnifoLM-VLM-Base:<model repository root>",
                ],
                "auth_file_envs": ["HF_TOKEN_FILE"],
                "cloud_gpu_provider_options": ["runpod", "vast"],
                "preferred_adapter": "blueprint-unitree-unifolm-policy-command-adapter --mode vla",
                "endpoint_closed_loop_policy_proven": False,
                "unitree_g1_dexterous_manipulation_proven": False,
                "claim_boundary": {
                    "unitree_native_vla_candidate": True,
                    "checkpoint_presence_is_not_endpoint_execution": True,
                    "requires_action_decoder_and_unitree_controller_bridge": True,
                },
            },
            {
                "id": "unitree_unifolm_wma_policy",
                "runtime_role": "unitree_native_world_model_action_policy_candidate",
                "command_env": "BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND",
                "checkpoint_env": "BLUEPRINT_UNITREE_UNIFOLM_WMA_CHECKPOINT",
                "source_root_env": "BLUEPRINT_UNITREE_UNIFOLM_WMA_SOURCE_ROOT",
                "source_urls": [
                    "https://github.com/unitreerobotics/unifolm-world-model-action",
                    "https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Dual",
                    "https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Base",
                ],
                "known_public_checkpoint_files": [
                    "unitreerobotics/UnifoLM-WMA-0-Dual:unifolm_wma_dual.ckpt",
                    "unitreerobotics/UnifoLM-WMA-0-Base:unifolm_wma_base.ckpt",
                ],
                "auth_file_envs": ["HF_TOKEN_FILE"],
                "cloud_gpu_provider_options": ["runpod", "vast"],
                "preferred_adapter": "blueprint-unitree-unifolm-policy-command-adapter --mode wma",
                "endpoint_closed_loop_policy_proven": False,
                "unitree_g1_dexterous_manipulation_proven": False,
                "claim_boundary": {
                    "unitree_native_wma_candidate": True,
                    "world_model_action_stack_is_not_automatically_endpoint_ready": True,
                    "requires_server_client_wrapper_and_action_schema_mapping": True,
                },
            },
            {
                "id": "unitree_lerobot_policy",
                "runtime_role": "unitree_g1_hand_or_gripper_manipulation_policy_candidate",
                "command_env": "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
                "checkpoint_env": "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT",
                "source_root_env": "BLUEPRINT_UNITREE_LEROBOT_ROOT",
                "source_urls": [
                    "https://github.com/unitreerobotics/unitree_lerobot",
                    "https://huggingface.co/docs/lerobot/unitree_g1",
                    "https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ToastedBread_Dataset",
                    "https://github.com/unitreerobotics/unitree_sim_isaaclab",
                ],
                "auth_file_envs": ["HF_TOKEN_FILE"],
                "cloud_gpu_provider_options": ["runpod", "vast"],
                "preferred_adapter": "blueprint-unitree-lerobot-policy-command-adapter",
                "endpoint_closed_loop_policy_proven": False,
                "claim_boundary": {
                    "unitree_specific_hand_policy_candidate": True,
                    "requires_task_specific_trained_policy": True,
                    "single_action_is_not_episode_success": True,
                },
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


def _is_egocentric_wam_input_video(row: Mapping[str, Any]) -> bool:
    camera = _string(row.get("camera"))
    return bool(row.get("egocentric_sensor_view") or camera in EGOCENTRIC_WAM_INPUT_CAMERAS)


def _action_counts(action_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in action_rows:
        action = _mapping(row.get("normalized_action"))
        action_type = _string(action.get("action_type")) or "unknown"
        counts[action_type] = counts.get(action_type, 0) + 1
    return [{"action_type": key, "count": counts[key]} for key in sorted(counts)]


def _action_type(row: Mapping[str, Any]) -> str:
    action = _mapping(row.get("normalized_action")) or _mapping(row.get("action"))
    return _string(action.get("action_type")) or "unknown"


def _build_manipulation_loop_readiness_manifest(
    *,
    generated_at: str,
    input_dir: Path,
    attempts: Sequence[Mapping[str, Any]],
    matrix_runs: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    unitree_controller_proof: Mapping[str, Any],
    openvla_provider_smoke_proof: Mapping[str, Any],
    rollouts: Sequence[Mapping[str, Any]],
    visual_rollout_useful: bool,
    visual_smoke_status: str,
    wam_policy_requery: Mapping[str, Any],
    policy_requery_ran: bool,
) -> dict[str, Any]:
    manipulation_task_ids = {"contact_or_push_light_object", "pick_place", "grasp_lift_place"}
    manipulation_matrix_runs = [
        row for row in matrix_runs if _string(row.get("task_id")) in manipulation_task_ids
    ]
    manipulation_attempts = [
        row for row in attempts if _string(row.get("task_id")) in manipulation_task_ids
    ]
    manipulation_action_rows = [
        row for row in action_rows if _action_type(row) == "manipulation_contact"
    ]
    source_manipulation_discovery = _load_json(
        input_dir / "unitree_g1_manipulation_policy_discovery.json"
    )
    source_manipulation_truth = _load_json(input_dir / "manipulation_truth_boundary.json")
    source_manipulation_report = _load_json(input_dir / "manipulation_endpoint_task_report.json")
    source_eval_summary = _load_json(
        input_dir / "mujoco_g1_wam_vla_policy_endpoint_eval_summary.json"
    ) or _load_json(input_dir / "policy_evaluation_summary.json")
    source_unitree_endpoint_hand_policy_used = bool(
        source_manipulation_report.get("unitree_endpoint_hand_policy_used")
        or source_eval_summary.get("unitree_endpoint_hand_policy_used")
    )
    source_unitree_endpoint_fresh_policy_action_command_ran = bool(
        source_manipulation_report.get("unitree_endpoint_fresh_policy_action_command_ran")
        or source_eval_summary.get("unitree_endpoint_fresh_policy_action_command_ran")
    )
    source_unitree_endpoint_provider_output_replay_used = bool(
        source_manipulation_report.get("unitree_endpoint_provider_output_replay_used")
        or source_eval_summary.get("unitree_endpoint_provider_output_replay_used")
    )
    unitree_hand_manipulation_policy_used = bool(
        unitree_controller_proof.get("unitree_hand_manipulation_policy_used")
        or unitree_controller_proof.get("unitree_lerobot_or_isaaclab_manipulation_policy_used")
        or source_manipulation_discovery.get("unitree_hand_manipulation_policy_used")
        or source_manipulation_discovery.get("unitree_lerobot_or_isaaclab_manipulation_policy_used")
        or source_manipulation_discovery.get("can_claim_vla_or_dexterous_manipulation")
        or source_unitree_endpoint_hand_policy_used
        or wam_policy_requery.get("unitree_hand_policy_requery_used")
    )
    requery_unitree_policy_used = bool(wam_policy_requery.get("unitree_hand_policy_requery_used"))
    openvla_action_smoke_only = bool(
        openvla_provider_smoke_proof.get("openvla_policy_action_command_ran")
    )
    blockers: list[str] = []
    if not manipulation_matrix_runs and not manipulation_attempts:
        blockers.append("blocked_missing_manipulation_contact_task_attempts")
    if not manipulation_action_rows:
        blockers.append("blocked_no_manipulation_contact_actions_in_endpoint_trace")
    if not unitree_hand_manipulation_policy_used:
        blockers.extend(
            [
                "blocked_missing_unitree_g1_hand_manipulation_policy",
                "blocked_missing_real_vla_or_unitree_hand_manipulation_policy",
            ]
        )
    if not rollouts:
        blockers.append("blocked_missing_wam_generated_rollout_for_manipulation_loop")
    elif not visual_rollout_useful:
        blockers.append("blocked_wam_generated_rollout_not_reviewable_for_manipulation_loop")
    if not policy_requery_ran:
        blockers.append("blocked_policy_did_not_observe_wam_generated_next_observation")
    ready = not blockers
    return {
        "schema_version": "wam_manipulation_loop_readiness_manifest.v1",
        "generated_at": generated_at,
        "status": "ready_for_closed_loop_manipulation_evaluation" if ready else "blocked",
        "source_mujoco_endpoint_eval_job_dir": str(input_dir),
        "manipulation_matrix_run_count": len(manipulation_matrix_runs),
        "manipulation_attempt_count": len(manipulation_attempts),
        "manipulation_contact_action_count": len(manipulation_action_rows),
        "source_manipulation_endpoint_path_used": bool(
            source_manipulation_report.get("manipulation_endpoint_path_used")
        ),
        "source_manipulation_contact_dynamics_validated": bool(
            source_manipulation_truth.get("manipulation_contact_dynamics_validated")
        ),
        "source_unitree_endpoint_hand_policy_used": source_unitree_endpoint_hand_policy_used,
        "source_unitree_endpoint_fresh_policy_action_command_ran": (
            source_unitree_endpoint_fresh_policy_action_command_ran
        ),
        "source_unitree_endpoint_provider_output_replay_used": (
            source_unitree_endpoint_provider_output_replay_used
        ),
        "unitree_hand_manipulation_policy_used": unitree_hand_manipulation_policy_used,
        "unitree_hand_manipulation_policy_scope": "endpoint_action_command"
        if source_unitree_endpoint_hand_policy_used
        else "runtime_discovery"
        if unitree_hand_manipulation_policy_used
        else None,
        "unitree_g1_robot_policy_is_unitree_native": unitree_hand_manipulation_policy_used,
        "g1_robot_policy_selection_contract": "unitree_native_policy_required_for_g1_claims",
        "g1_robot_policy_selected_family": "unitree_native_hand_manipulation_policy"
        if unitree_hand_manipulation_policy_used
        else None,
        "openvla_selected_as_g1_robot_policy": False,
        "wam_rollout_selected_as_g1_robot_policy": False,
        "real_vla_or_unitree_hand_manipulation_policy_used": (
            unitree_hand_manipulation_policy_used
        ),
        "real_vla_or_unitree_hand_policy_requery_used": requery_unitree_policy_used,
        "unitree_hand_policy_requery_used": bool(
            wam_policy_requery.get("unitree_hand_policy_requery_used")
        ),
        "openvla_policy_requery_used": bool(wam_policy_requery.get("openvla_policy_requery_used")),
        "policy_requery_policy_id": wam_policy_requery.get("policy_requery_policy_id"),
        "policy_requery_provider_replay_used": bool(
            wam_policy_requery.get("policy_requery_provider_replay_used")
        ),
        "openvla_policy_action_smoke_available": openvla_action_smoke_only,
        "openvla_policy_action_smoke_is_not_closed_loop_manipulation": openvla_action_smoke_only,
        "unitree_hand_policy_required_for_g1_manipulation": True,
        "wam_generated_rollout_count": len(rollouts),
        "wam_generated_rollout_visual_smoke_status": visual_smoke_status,
        "wam_generated_rollout_visually_useful_for_manipulation_review": visual_rollout_useful,
        "policy_observes_wam_generated_next_observation": policy_requery_ran,
        "closed_loop_manipulation_policy_wam_interaction_ready": ready,
        "scaffolding_value_before_manipulation_policy": [
            "validates Blueprint observation/action packets",
            "validates endpoint invocation and action normalization",
            "validates WAM input packaging from egocentric review video and traces",
            "validates generated-rollout visual quality gates",
            "validates the requery contract for policy observing WAM-generated next observations",
        ],
        "required_to_become_true": [
            "source job includes contact/manipulation task attempts",
            "endpoint policy emits manipulation_contact or lower-level arm/hand actions",
            "real Unitree LeRobot or Unitree UnifoLM hand/manipulation policy command and checkpoint are configured",
            "WAM generates reviewable action-conditioned next-observation video",
            "policy is requeried on that generated next observation",
            "success judge or human/VLM scorer labels the generated/executed episode",
        ],
        "source_artifacts": {
            "unitree_g1_manipulation_policy_discovery": str(
                input_dir / "unitree_g1_manipulation_policy_discovery.json"
            ),
            "manipulation_truth_boundary": str(input_dir / "manipulation_truth_boundary.json"),
            "manipulation_endpoint_task_report": str(
                input_dir / "manipulation_endpoint_task_report.json"
            ),
        },
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "wam_evaluator_is_test_bench_not_robot_manipulation_policy": True,
            "openvla_provider_smoke_is_not_closed_loop_robot_control": True,
            "openvla_policy_is_not_selected_g1_robot_policy": True,
            "wam_rollout_is_not_selected_g1_robot_policy": True,
            "g1_robot_policy_selection_contract": "unitree_native_policy_required_for_g1_claims",
            "unitree_locomotion_policy_is_not_dexterous_manipulation": True,
            "simulated_contact_dynamics_do_not_prove_vla_manipulation": True,
            "generated_world_rank_fidelity_result_proven": False,
            "raw_credentials_written_to_artifacts": False,
        },
    }


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
    stale_output_removed = False
    if output_path.exists():
        try:
            output_path.unlink()
            stale_output_removed = True
        except OSError as exc:
            return {}, {
                "status": "blocked",
                "blockers": [f"wam_model_stale_output_unlink_failed:{type(exc).__name__}"],
                "duration_seconds": round(time.monotonic() - started, 6),
                "stale_output_removed_before_launch": False,
            }
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
        "stale_output_removed_before_launch": stale_output_removed,
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


def _rollouts_from_model_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_rollouts = payload.get("rollouts") or _mapping(
        payload.get("wam_generated_rollout_results")
    ).get("rollouts")
    if not isinstance(raw_rollouts, list):
        return []
    return [dict(item) for item in raw_rollouts if isinstance(item, Mapping)]


TRUSTED_WAM_MODEL_PAYLOAD_SCHEMAS = {
    "oscar_wam_command_adapter.v1",
    "oscar_cosmos_wam_command_adapter.v1",
    "oscar_wam_provider_command_adapter.v1",
    "cosmos3_wam_command_adapter.v1",
}


def _model_payload_truth_boundary_flag(
    payload: Mapping[str, Any],
    key: str,
) -> bool:
    truth_boundary = _mapping(payload.get("truth_boundary"))
    return truth_boundary.get(key) is True


def _wam_model_payload_proves_fresh_execution(
    *,
    model_payload: Mapping[str, Any],
    model_execution_detail: Mapping[str, Any],
    provider_output_replay_used: bool,
) -> bool:
    if provider_output_replay_used:
        return False
    if model_execution_detail.get("status") != "completed":
        return False
    if _string(model_payload.get("schema_version")) not in TRUSTED_WAM_MODEL_PAYLOAD_SCHEMAS:
        return False
    if model_payload.get("status") != "completed":
        return False
    if not _rollouts_from_model_payload(model_payload):
        return False
    return bool(
        model_payload.get("fresh_model_command_executed_this_invocation") is True
        or model_payload.get("fresh_model_run_claimed") is True
        or model_payload.get("learned_wam_model_ran") is True
        or _model_payload_truth_boundary_flag(
            model_payload,
            "generated_video_is_model_output",
        )
    )


def _provider_wam_payload_proves_model_output(payload: Mapping[str, Any]) -> bool:
    if _string(payload.get("schema_version")) not in TRUSTED_WAM_MODEL_PAYLOAD_SCHEMAS:
        return False
    if payload.get("status") != "completed":
        return False
    if _string(payload.get("mode")) != "replay_existing_provider_output":
        return False
    if not _rollouts_from_model_payload(payload):
        return False
    return bool(
        payload.get("provider_output_replayed") is True
        and payload.get("provider_learned_wam_model_ran") is True
        and payload.get("provider_generated_video_is_model_output") is True
        and payload.get("provider_runtime_result_present") is True
        and payload.get("provider_runtime_result_status") == "completed"
        and payload.get("provider_runtime_result_proves_model_output") is True
    )


def _build_policy_model_endpoint_probe_results(
    *,
    generated_at: str,
    readiness_manifest: Mapping[str, Any],
    selected_candidate_id: str | None,
    model_run_allowed: bool,
    model_payload: Mapping[str, Any],
    model_execution_detail: Mapping[str, Any],
) -> dict[str, Any]:
    selected = _string(selected_candidate_id)
    payload_rollouts = _rollouts_from_model_payload(model_payload)
    command_completed = bool(model_execution_detail.get("status") == "completed")
    payload_status = _string(model_payload.get("status"))
    provider_output_replay_used = (
        _string(model_payload.get("mode")) == "replay_existing_provider_output"
    )
    wam_rollout_contract_valid = bool(
        payload_rollouts
        and (
            _wam_model_payload_proves_fresh_execution(
                model_payload=model_payload,
                model_execution_detail=model_execution_detail,
                provider_output_replay_used=provider_output_replay_used,
            )
            or _provider_wam_payload_proves_model_output(model_payload)
        )
    )
    payload_contract_valid = bool(
        command_completed
        and (
            (payload_status == "completed" and wam_rollout_contract_valid)
            or model_payload.get("action")
            or model_payload.get("policy_action")
        )
    )
    rows: list[dict[str, Any]] = []
    for row_value in readiness_manifest.get("candidates", []) or []:
        if not isinstance(row_value, Mapping):
            continue
        row = dict(row_value)
        candidate_id = _string(row.get("candidate_id"))
        is_selected = bool(candidate_id and candidate_id == selected)
        command_available = bool(row.get("command_available"))
        real_model_runtime_ready = bool(row.get("real_model_runtime_ready"))
        invocation_attempted = bool(is_selected and model_run_allowed)
        blockers: list[str] = []
        if not command_available:
            blockers.append("blocked_model_command_not_available")
        if command_available and not invocation_attempted:
            blockers.append("blocked_model_command_not_invoked_in_this_probe")
        if invocation_attempted and not command_completed:
            blockers.extend(
                _string_list(model_execution_detail.get("blockers"))
                or ["blocked_model_command_probe_failed"]
            )
        if invocation_attempted and command_completed and not payload_contract_valid:
            blockers.append("blocked_model_command_output_contract_not_proven")
        rows.append(
            {
                "candidate_id": candidate_id,
                "runtime_role": row.get("runtime_role"),
                "selected_for_probe": is_selected,
                "command_available": command_available,
                "real_model_runtime_ready_by_static_readiness": real_model_runtime_ready,
                "command_invocation_attempted": invocation_attempted,
                "command_probe_completed": bool(invocation_attempted and command_completed),
                "blueprint_output_contract_valid": bool(invocation_attempted and payload_contract_valid),
                "payload_status": (payload_status or None) if invocation_attempted else None,
                "rollout_count": len(payload_rollouts) if invocation_attempted else 0,
                "action_response_present": bool(
                    invocation_attempted
                    and (model_payload.get("action") or model_payload.get("policy_action"))
                ),
                "fresh_model_command_executed_this_invocation": bool(
                    invocation_attempted
                    and model_payload.get("fresh_model_command_executed_this_invocation")
                ),
                "provider_output_replayed": bool(
                    invocation_attempted and model_payload.get("provider_output_replayed")
                ),
                "provider_generated_video_is_model_output": bool(
                    invocation_attempted and model_payload.get("provider_generated_video_is_model_output")
                ),
                "command_exit_code": model_execution_detail.get("command_exit_code")
                if invocation_attempted
                else None,
                "duration_seconds": model_execution_detail.get("duration_seconds")
                if invocation_attempted
                else None,
                "blockers": sorted(set(blockers)),
                "claim_boundary": {
                    "http_wrapper_start_is_not_model_execution": True,
                    "static_readiness_is_not_output_contract_probe": True,
                    "provider_output_replay_is_not_fresh_model_execution": bool(
                        invocation_attempted and model_payload.get("provider_output_replayed")
                    ),
                    "raw_credentials_written_to_artifacts": False,
                },
            }
        )
    probed_rows = [row for row in rows if row["command_invocation_attempted"]]
    passed_rows = [row for row in rows if row["blueprint_output_contract_valid"]]
    all_row_blockers = sorted(
        {
            blocker
            for row in rows
            for blocker in row.get("blockers", [])
            if blocker
        }
    )
    selected_row_blockers = sorted(
        {
            blocker
            for row in rows
            if row.get("selected_for_probe")
            for blocker in row.get("blockers", [])
            if blocker
        }
    )
    return {
        "schema_version": "policy_model_endpoint_probe_results.v1",
        "generated_at": generated_at,
        "status": "completed" if passed_rows else "blocked",
        "selected_candidate_id": selected or None,
        "probe_attempted_candidate_count": len(probed_rows),
        "probe_passed_candidate_count": len(passed_rows),
        "can_create_http_wrapper": bool(
            readiness_manifest.get("http_endpoint_wrapper_available")
        ),
        "can_claim_real_model_endpoint_after_probe": bool(passed_rows),
        "why_cannot_just_create_endpoints": [
            "The HTTP wrapper can be created around any configured command, but it only proves HTTP/auth plumbing.",
            "A real model endpoint claim requires the wrapped command to run and emit Blueprint-compatible JSON.",
            "If the command needs a GPU or checkpoint and that runtime is absent, the endpoint would return 502/503.",
            "Provider-output replay can prove imported model output compatibility, but not fresh per-request inference.",
        ],
        "candidates": rows,
        "blockers": [] if passed_rows else selected_row_blockers or all_row_blockers,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


def _run_wam_success_label_command(
    *,
    command: str,
    input_path: Path,
    output_path: Path,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    env = {
        **os.environ,
        "BLUEPRINT_WAM_SUCCESS_LABEL_INPUT": str(input_path),
        "BLUEPRINT_WAM_SUCCESS_LABEL_OUTPUT": str(output_path),
        "BLUEPRINT_WAM_SUCCESS_LABEL_JOB_DIR": str(input_path.parent),
    }
    try:
        result = subprocess.run(
            shlex.split(command),
            cwd=str(input_path.parent),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        return {}, {
            "status": "blocked",
            "blockers": [f"wam_success_label_command_failed:{type(exc).__name__}"],
            "duration_seconds": round(time.monotonic() - started, 6),
        }
    detail = {
        "status": "completed" if result.returncode == 0 else "blocked",
        "command_exit_code": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "stdout_size_bytes": len(result.stdout or ""),
        "stderr_size_bytes": len(result.stderr or ""),
        "stderr_omitted_to_avoid_secret_leakage": bool(result.stderr),
        "blockers": [] if result.returncode == 0 else ["wam_success_label_command_nonzero_exit"],
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
            detail["blockers"] = ["wam_success_label_stdout_json_invalid"]
    return payload, detail


def _normalize_wam_success_labels(
    *,
    command_payload: Mapping[str, Any],
    rollouts: Sequence[Mapping[str, Any]],
    generated_at: str,
    visual_smoke_status: str,
    visual_rollout_useful: bool,
) -> dict[str, Any]:
    rollout_by_id = {
        _string(row.get("rollout_id")): dict(row)
        for row in rollouts
        if _string(row.get("rollout_id"))
    }
    labels: list[dict[str, Any]] = []
    blockers = _string_list(command_payload.get("blockers"))
    payload_status = _string(command_payload.get("status"))
    if payload_status and payload_status not in {"completed", "completed_review_required"}:
        blockers.append("wam_success_label_command_payload_not_completed")
    if rollouts and not visual_rollout_useful:
        blockers.append("blocked_generated_rollout_not_visually_useful_for_success_review")
    for item in command_payload.get("labels", []) or []:
        if not isinstance(item, Mapping):
            continue
        rollout_id = _string(item.get("rollout_id"))
        source_rollout = rollout_by_id.get(rollout_id, {})
        if not source_rollout:
            blockers.append("wam_success_label_unknown_rollout_id")
            continue
        if not bool(item.get("visual_evidence_used", True)):
            blockers.append("wam_success_label_missing_visual_evidence")
            continue
        success_value = item.get("success")
        if not isinstance(success_value, bool):
            success_value = None
        # A non-boolean verdict is a review gap, not a quiet "uncertain" pass-through:
        # it must keep the label set below review grade.
        strict_boolean_verdict = success_value is not None
        confidence_value = item.get("confidence")
        confidence = (
            float(confidence_value)
            if isinstance(confidence_value, (int, float))
            else None
        )
        labels.append(
            {
                "label_id": _string(item.get("label_id")) or f"wam_success_{rollout_id or len(labels) + 1}",
                "rollout_id": rollout_id or None,
                "scenario_eval_run_id": item.get("scenario_eval_run_id")
                or source_rollout.get("scenario_eval_run_id"),
                "policy_id": item.get("policy_id") or source_rollout.get("policy_id"),
                "status": "review_required",
                "semantic_result": (
                    "success"
                    if success_value is True
                    else "failure"
                    if success_value is False
                    else "uncertain"
                ),
                "success": success_value,
                "confidence": confidence,
                "rationale": _string(item.get("rationale")) or None,
                "task_completion_evidence": item.get("task_completion_evidence")
                if isinstance(item.get("task_completion_evidence"), list)
                else [],
                "failure_modes": item.get("failure_modes")
                if isinstance(item.get("failure_modes"), list)
                else [],
                "evidence_refs": item.get("evidence_refs")
                if isinstance(item.get("evidence_refs"), list)
                else [source_rollout.get("generated_video_path")]
                if source_rollout.get("generated_video_path")
                else [],
                "label_source": _string(item.get("label_source"))
                or _string(command_payload.get("provider"))
                or "wam_success_label_command",
                "model": _string(item.get("model")) or _string(command_payload.get("model")) or None,
                "visual_evidence_used": bool(item.get("visual_evidence_used", True)),
                "visual_smoke_status": visual_smoke_status,
                "visual_rollout_useful_for_task_success_review": visual_rollout_useful,
                "review_grade_visual_evidence_available": visual_rollout_useful,
                # Media validity alone never makes a label authoritative: the reviewer
                # must also have returned a strict boolean verdict.
                "media_validity_passed": visual_rollout_useful,
                "reviewer_verdict_strict_boolean": strict_boolean_verdict,
                "authoritative_task_success_label": bool(
                    visual_rollout_useful and strict_boolean_verdict
                ),
                "review_task_success": bool(
                    visual_rollout_useful and success_value is True
                ),
                "failure_diagnosis_blocked_by_visual_quality": (
                    success_value is False and not visual_rollout_useful
                ),
                "human_review_required": bool(item.get("human_review_required", False)),
                "human_review_recommended": bool(item.get("human_review_recommended", True)),
                "proof_effect": "semantic_label_on_generated_video_only",
                "rank_fidelity_result_proven": False,
                "generated_world_policy_evaluation_scope_proven": False,
                "generated_world_rank_fidelity_result_proven": False,
                "safety_or_contact_validation_proven": False,
                "srcc_or_policy_ranking_proven": False,
                "public_claim_upgrade_allowed": False,
            }
        )
    strict_boolean_label_count = sum(
        1 for row in labels if row.get("reviewer_verdict_strict_boolean")
    )
    if labels and strict_boolean_label_count < len(labels):
        blockers.append("wam_success_label_verdict_not_strict_boolean")
    status = "completed" if labels and not blockers else "blocked"
    return {
        "schema_version": "wam_success_labels.v1",
        "generated_at": generated_at,
        "status": status,
        "wam_success_label_from_generated_video": bool(
            labels and not blockers and strict_boolean_label_count == len(labels)
        ),
        "visual_smoke_status": visual_smoke_status,
        "visual_rollout_useful_for_task_success_review": visual_rollout_useful,
        "review_grade_visual_evidence_available": visual_rollout_useful,
        "strict_boolean_label_count": strict_boolean_label_count,
        "review_grade_success_labels": bool(
            labels
            and not blockers
            and visual_rollout_useful
            and strict_boolean_label_count == len(labels)
        ),
        "label_count": len(labels),
        "labels": labels,
        "provider": _string(command_payload.get("provider")) or None,
        "model": _string(command_payload.get("model")) or None,
        "blockers": blockers,
        "visual_evidence_used": any(bool(row.get("visual_evidence_used")) for row in labels),
        "human_review_required": any(bool(row.get("human_review_required")) for row in labels),
        "human_review_recommended": bool(labels),
        "claim_boundary": {
            "success_label_is_from_generated_video_not_physical_robot": True,
            "success_label_requires_passed_visual_smoke": True,
            "visual_rollout_useful_for_task_success_review": visual_rollout_useful,
            "success_label_does_not_prove_forward_inverse_consistency": True,
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        },
    }


def _generated_rollout_failure_labels(
    *,
    rollouts: Sequence[Mapping[str, Any]],
    success_labels: Mapping[str, Any],
    visual_smoke: Mapping[str, Any],
    visual_rollout_useful: bool,
    generated_at: str,
    output_dir: Path,
    blockers: Sequence[str],
) -> dict[str, Any]:
    rollout_by_id = {
        _string(row.get("rollout_id")): dict(row)
        for row in rollouts
        if _string(row.get("rollout_id"))
    }
    visual_smoke_status = _string(visual_smoke.get("status"))
    visual_review_blockers = _dedupe_refs(
        [
            *(_string_list(visual_smoke.get("blockers")) if not visual_rollout_useful else []),
            *(_string_list(blockers) if not visual_rollout_useful else []),
        ]
    )
    rows: list[dict[str, Any]] = []
    failed_success_labels = [
        dict(row)
        for row in success_labels.get("labels", []) or []
        if isinstance(row, Mapping) and row.get("success") is False
    ]
    for index, source_label in enumerate(failed_success_labels, start=1):
        rollout_id = _string(source_label.get("rollout_id"))
        source_rollout = rollout_by_id.get(rollout_id, {})
        failure_mode_ids = _string_list(source_label.get("failure_modes")) or _string_list(
            source_label.get("failure_mode_ids")
        ) or ["wam_generated_rollout_task_failure"]
        frame_refs = _failure_frame_or_clip_refs(source_label) or _failure_frame_or_clip_refs(
            source_rollout
        )
        source_trace_refs = _dedupe_refs(
            [
                str(output_dir / "wam_generated_rollout_results.json"),
                str(output_dir / "wam_success_labels.json"),
                str(output_dir / "wam_evaluator_trace_binding.json"),
            ]
        )
        evidence_refs = _failure_evidence_refs(
            source_label,
            extra_refs=tuple(
                [
                    *source_trace_refs,
                    str(output_dir / "wam_generated_rollout_visual_smoke.json"),
                    *frame_refs,
                ]
            ),
        )
        review_status = _failure_review_status(
            supplied_review_status=source_label.get("review_status"),
            supplied_status=source_label.get("status"),
            generated_rollout=True,
            frame_or_clip_ref_count=len(frame_refs),
        )
        root_cause_category = _failure_root_cause_category(
            failure_mode_ids,
            failure_reason=_string(source_label.get("rationale")) or None,
        )
        rows.append(
            {
                "label_id": f"wam_generated_failure_label_{index:04d}",
                "attempt_id": source_rollout.get("attempt_id"),
                "rollout_id": rollout_id or source_rollout.get("rollout_id"),
                "scenario_eval_run_id": source_label.get("scenario_eval_run_id")
                or source_rollout.get("scenario_eval_run_id"),
                "scenario_variation_instance_id": source_rollout.get(
                    "scenario_variation_instance_id"
                ),
                "task_id": source_rollout.get("task_id"),
                "scenario_id": source_rollout.get("scenario_id"),
                "policy_id": source_label.get("policy_id") or source_rollout.get("policy_id"),
                "failure_mode_ids": failure_mode_ids,
                "failure_reason": _string(source_label.get("rationale")) or None,
                "source": "wam_success_labels",
                "evidence_refs": evidence_refs,
                "source_trace_refs": source_trace_refs,
                "frame_or_clip_refs": frame_refs,
                "visual_smoke_ref": str(output_dir / "wam_generated_rollout_visual_smoke.json"),
                "confidence": source_label.get("confidence"),
                "status": "review_required",
                "review_status": review_status,
                "reviewer_acceptance_required": True,
                "root_cause_category": root_cause_category,
                "remediation_candidate": _failure_remediation_candidate(
                    root_cause_category,
                    failure_mode_ids,
                ),
                "unknown_when_evidence_weak": bool(
                    not frame_refs or review_status == "non_reviewable_failure_hypothesis"
                ),
                "non_reviewable_failure_hypothesis": (
                    review_status == "non_reviewable_failure_hypothesis"
                ),
                "visual_smoke_status": visual_smoke_status,
                "visual_rollout_useful_for_task_success_review": visual_rollout_useful,
                "visual_review_blockers": visual_review_blockers,
                "review_grade_failure_diagnosis": bool(
                    visual_rollout_useful
                    and review_status != "non_reviewable_failure_hypothesis"
                ),
                "authoritative_failure_diagnosis": False,
                "generated_wam_rollout": True,
                "model_derived_support_artifact": True,
                "proof_effect": FAILURE_LABEL_PROOF_EFFECT,
            }
        )
    if rollouts and not rows and not visual_rollout_useful:
        visual_blockers = visual_review_blockers or _string_list(blockers)
        for index, source_rollout in enumerate(rollouts, start=1):
            frame_refs = _failure_frame_or_clip_refs(source_rollout)
            source_trace_refs = _dedupe_refs(
                [
                    str(output_dir / "wam_generated_rollout_results.json"),
                    str(output_dir / "wam_generated_rollout_visual_smoke.json"),
                    str(output_dir / "wam_evaluator_trace_binding.json"),
                ]
            )
            failure_mode_ids = ["generated_rollout_visual_quality_not_reviewable"]
            root_cause_category = _failure_root_cause_category(
                failure_mode_ids,
                failure_reason=";".join(visual_blockers),
            )
            evidence_refs = _failure_evidence_refs(
                source_rollout,
                extra_refs=tuple([*source_trace_refs, *frame_refs]),
            )
            rows.append(
                {
                    "label_id": f"wam_nonreviewable_failure_hypothesis_{index:04d}",
                    "attempt_id": source_rollout.get("attempt_id"),
                    "rollout_id": source_rollout.get("rollout_id"),
                    "scenario_eval_run_id": source_rollout.get("scenario_eval_run_id"),
                    "scenario_variation_instance_id": source_rollout.get(
                        "scenario_variation_instance_id"
                    ),
                    "task_id": source_rollout.get("task_id"),
                    "scenario_id": source_rollout.get("scenario_id"),
                    "policy_id": source_rollout.get("policy_id"),
                    "failure_mode_ids": failure_mode_ids,
                    "failure_reason": "generated_rollout_visual_smoke_failed",
                    "source": "wam_generated_rollout_visual_smoke",
                    "evidence_refs": evidence_refs,
                    "source_trace_refs": source_trace_refs,
                    "frame_or_clip_refs": frame_refs,
                    "visual_smoke_ref": str(output_dir / "wam_generated_rollout_visual_smoke.json"),
                    "confidence": None,
                    "status": "review_required",
                    "review_status": "non_reviewable_failure_hypothesis",
                    "reviewer_acceptance_required": True,
                    "root_cause_category": root_cause_category,
                    "remediation_candidate": _failure_remediation_candidate(
                        root_cause_category,
                        failure_mode_ids,
                    ),
                    "unknown_when_evidence_weak": True,
                    "non_reviewable_failure_hypothesis": True,
                    "visual_smoke_status": visual_smoke_status,
                    "visual_rollout_useful_for_task_success_review": visual_rollout_useful,
                    "visual_review_blockers": visual_blockers,
                    "review_grade_failure_diagnosis": False,
                    "authoritative_failure_diagnosis": False,
                    "blockers": visual_blockers,
                    "generated_wam_rollout": True,
                    "model_derived_support_artifact": True,
                    "proof_effect": FAILURE_LABEL_PROOF_EFFECT,
                }
            )
    coverage_complete = all(
        row.get("failure_mode_ids") and row.get("evidence_refs") and row.get("review_status")
        for row in rows
    )
    nonreviewable_label_ids = [
        _string(row.get("label_id"))
        for row in rows
        if row.get("review_status") == "non_reviewable_failure_hypothesis"
    ]
    failure_diagnosis_blockers: list[str] = []
    if rows and not coverage_complete:
        failure_diagnosis_blockers.append("failure_diagnosis_coverage_incomplete")
    if rows and not visual_rollout_useful:
        failure_diagnosis_blockers.extend(
            visual_review_blockers or ["generated_rollout_visual_smoke_missing_or_failed"]
        )
        failure_diagnosis_blockers.append(
            "failure_diagnosis_blocked_by_generated_rollout_visual_quality"
        )
    if nonreviewable_label_ids:
        failure_diagnosis_blockers.append("failure_labels_nonreviewable_failure_hypotheses")
    failure_diagnosis_blockers = _dedupe_refs(failure_diagnosis_blockers)
    failure_diagnosis_complete = bool(
        coverage_complete
        and not nonreviewable_label_ids
        and (visual_rollout_useful or not rows)
    )
    review_grade_failure_diagnosis = bool(
        rows and failure_diagnosis_complete and visual_rollout_useful
    )
    return {
        "schema_version": "wam_generated_rollout_failure_labels.v1",
        "generated_at": generated_at,
        "status": "review_required"
        if rows
        else "blocked"
        if blockers
        else "no_failures_labeled",
        "visual_smoke_status": visual_smoke_status,
        "visual_rollout_useful_for_task_success_review": visual_rollout_useful,
        "visual_review_blockers": visual_review_blockers,
        "review_grade_failure_diagnosis": review_grade_failure_diagnosis,
        "authoritative_failure_diagnosis": False,
        "label_count": len(rows),
        "failed_attempt_count": len(rows),
        "failed_run_label_coverage_complete": coverage_complete,
        "failure_diagnosis_coverage_complete": coverage_complete,
        "failure_diagnosis_review_complete": not nonreviewable_label_ids,
        "failure_diagnosis_complete": failure_diagnosis_complete,
        "failure_diagnosis_blockers": failure_diagnosis_blockers,
        "nonreviewable_failure_hypothesis_label_ids": nonreviewable_label_ids,
        "blockers": _dedupe_refs([*blockers, *failure_diagnosis_blockers]),
        "labels": rows,
        "claim_boundary": {
            "failure_labels_are_generated_rollout_support_artifacts": True,
            "visual_smoke_required_for_review_grade_failure_diagnosis": True,
            "visual_rollout_useful_for_task_success_review": visual_rollout_useful,
            "review_grade_failure_diagnosis": review_grade_failure_diagnosis,
            "failure_labels_do_not_prove_generated_world_rank_fidelity": True,
            "proof_effect": FAILURE_LABEL_PROOF_EFFECT,
        },
    }


def _run_wam_consistency_command(
    *,
    command: str,
    input_path: Path,
    output_path: Path,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    env = {
        **os.environ,
        "BLUEPRINT_WAM_CONSISTENCY_INPUT": str(input_path),
        "BLUEPRINT_WAM_CONSISTENCY_OUTPUT": str(output_path),
        "BLUEPRINT_WAM_CONSISTENCY_JOB_DIR": str(input_path.parent),
    }
    try:
        result = subprocess.run(
            shlex.split(command),
            cwd=str(input_path.parent),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        return {}, {
            "status": "blocked",
            "blockers": [f"wam_consistency_command_failed:{type(exc).__name__}"],
            "duration_seconds": round(time.monotonic() - started, 6),
        }
    detail = {
        "status": "completed" if result.returncode == 0 else "blocked",
        "command_exit_code": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "stdout_size_bytes": len(result.stdout or ""),
        "stderr_size_bytes": len(result.stderr or ""),
        "stderr_omitted_to_avoid_secret_leakage": bool(result.stderr),
        "blockers": [] if result.returncode == 0 else ["wam_consistency_command_nonzero_exit"],
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
            detail["blockers"] = ["wam_consistency_stdout_json_invalid"]
    return payload, detail


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _string(value).lower()
    if text in {"true", "yes", "pass", "passed", "consistent"}:
        return True
    if text in {"false", "no", "fail", "failed", "inconsistent"}:
        return False
    return None


def _confidence_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    try:
        return max(0.0, min(1.0, float(_string(value))))
    except ValueError:
        return None


def _normalize_wam_episode_consistency(
    *,
    command_payload: Mapping[str, Any],
    rollouts: Sequence[Mapping[str, Any]],
    generated_at: str,
    action_conditioned_video_rollout_generated: bool,
    action_conditioned_video_rollout_available: bool,
    provider_output_replay_used: bool,
    success_label_generated: bool,
    visual_smoke_status: str,
    visual_rollout_useful: bool,
    command_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rollout_by_id = {
        _string(row.get("rollout_id")): dict(row)
        for row in rollouts
        if _string(row.get("rollout_id"))
    }
    blockers = _string_list(command_payload.get("blockers"))
    payload_status = _string(command_payload.get("status"))
    if payload_status and payload_status not in {"completed", "completed_review_required"}:
        blockers.append("wam_consistency_command_payload_not_completed")
    if rollouts and not visual_rollout_useful:
        blockers.append("generated_rollout_not_visually_useful_for_consistency_proof")
    raw_checks = command_payload.get("rollout_checks")
    if raw_checks is None:
        raw_checks = command_payload.get("checks")
    checks: list[dict[str, Any]] = []
    for item in raw_checks or []:
        if not isinstance(item, Mapping):
            continue
        rollout_id = _string(item.get("rollout_id"))
        source_rollout = rollout_by_id.get(rollout_id, {})
        if not source_rollout:
            blockers.append("wam_consistency_unknown_rollout_id")
            continue
        forward_value = _bool_or_none(
            item.get("forward_consistent", item.get("forward_dynamics_consistent"))
        )
        inverse_value = _bool_or_none(
            item.get("inverse_consistent", item.get("inverse_dynamics_consistent"))
        )
        visual_evidence_used = bool(item.get("visual_evidence_used", True))
        action_trace_evidence_used = bool(item.get("action_trace_evidence_used", True))
        if forward_value is not True:
            blockers.append("wam_consistency_forward_not_proven")
        if inverse_value is not True:
            blockers.append("wam_consistency_inverse_not_proven")
        if not visual_evidence_used:
            blockers.append("wam_consistency_missing_visual_evidence")
        if not action_trace_evidence_used:
            blockers.append("wam_consistency_missing_action_trace_evidence")
        checks.append(
            {
                "rollout_id": rollout_id,
                "scenario_eval_run_id": item.get("scenario_eval_run_id")
                or source_rollout.get("scenario_eval_run_id"),
                "policy_id": item.get("policy_id") or source_rollout.get("policy_id"),
                "model_candidate": item.get("model_candidate")
                or source_rollout.get("model_candidate")
                or command_payload.get("model_candidate"),
                "forward_consistent": forward_value,
                "inverse_consistent": inverse_value,
                "confidence": _confidence_or_none(item.get("confidence")),
                "rationale": _string(item.get("rationale")) or None,
                "visible_action_alignment_evidence": item.get(
                    "visible_action_alignment_evidence"
                )
                if isinstance(item.get("visible_action_alignment_evidence"), list)
                else [],
                "inconsistency_evidence": item.get("inconsistency_evidence")
                if isinstance(item.get("inconsistency_evidence"), list)
                else [],
                "evidence_refs": item.get("evidence_refs")
                if isinstance(item.get("evidence_refs"), list)
                else [source_rollout.get("generated_video_path")]
                if source_rollout.get("generated_video_path")
                else [],
                "visual_evidence_used": visual_evidence_used,
                "action_trace_evidence_used": action_trace_evidence_used,
                "label_source": _string(item.get("label_source"))
                or _string(command_payload.get("provider"))
                or "wam_episode_consistency_command",
                "model": _string(item.get("model")) or _string(command_payload.get("model")) or None,
                "proof_effect": "external_episode_consistency_label_on_generated_video_and_trace_context",
                "task_success_proven": False,
                "generated_world_rank_fidelity_result_proven": False,
                "generated_world_policy_evaluation_scope_proven": False,
                "safety_or_contact_validation_proven": False,
                "srcc_or_policy_ranking_proven": False,
                "public_claim_upgrade_allowed": False,
            }
        )
    forward_proven = bool(checks) and all(row["forward_consistent"] is True for row in checks)
    inverse_proven = bool(checks) and all(row["inverse_consistent"] is True for row in checks)
    evidence_complete = bool(checks) and all(
        row["visual_evidence_used"] and row["action_trace_evidence_used"] for row in checks
    )
    proven = forward_proven and inverse_proven and evidence_complete and not blockers
    result = {
        "schema_version": "wam_consistency_checks.v1",
        "generated_at": generated_at,
        "status": "completed" if proven else "blocked" if not rollouts else "requires_review",
        "external_episode_consistency_scorer_ran": bool(command_payload),
        "external_episode_consistency_scorer_required": not bool(command_payload),
        "external_episode_consistency_scorer_id": _string(command_payload.get("provider"))
        or "wam_episode_consistency_command",
        "model": _string(command_payload.get("model")) or None,
        "forward_inverse_consistency_proven": proven,
        "forward_dynamics_consistency_proven": forward_proven and evidence_complete and not blockers,
        "inverse_dynamics_consistency_proven": inverse_proven and evidence_complete and not blockers,
        "action_conditioned_video_rollout_generated": action_conditioned_video_rollout_generated,
        "action_conditioned_video_rollout_available": action_conditioned_video_rollout_available,
        "provider_output_replay_used": provider_output_replay_used,
        "wam_success_label_from_generated_video": success_label_generated,
        "generated_rollout_visual_smoke_status": visual_smoke_status,
        "generated_rollout_visually_useful_for_success_review": visual_rollout_useful,
        "success_label_can_be_vlm_judged_separately_from_consistency": True,
        "rollout_count": len(rollouts),
        "rollout_checks": checks,
        "checks": [
            {
                "check_id": "forward_dynamics_consistency",
                "status": "passed" if proven and forward_proven else "requires_review",
                "proven": forward_proven and evidence_complete and not blockers,
                "blockers": [] if forward_proven and evidence_complete and not blockers else sorted(set(blockers)),
                "scorer": _string(command_payload.get("provider"))
                or "wam_episode_consistency_command",
                "proof_scope": "external_vlm_episode_consistency_label",
            },
            {
                "check_id": "inverse_dynamics_consistency",
                "status": "passed" if proven and inverse_proven else "requires_review",
                "proven": inverse_proven and evidence_complete and not blockers,
                "blockers": [] if inverse_proven and evidence_complete and not blockers else sorted(set(blockers)),
                "scorer": _string(command_payload.get("provider"))
                or "wam_episode_consistency_command",
                "proof_scope": "external_vlm_episode_consistency_label",
            },
        ],
        "what_is_needed_to_make_forward_inverse_consistency_true": []
        if proven
        else sorted(set(blockers or ["external_wam_episode_consistency_scorer_must_pass"])),
        "generated_rollout_termination_reason": (
            "external_episode_consistency_scorer_passed" if proven else "needs_external_episode_consistency_review"
        ),
        "model_rollout_confidence": None,
        "claim_boundary": {
            "forward_inverse_consistency_is_external_episode_label_not_wam_execution": True,
            "forward_inverse_consistency_does_not_prove_task_success": True,
            "forward_inverse_consistency_does_not_prove_visual_rollout_useful_for_success_review": True,
            "forward_inverse_consistency_does_not_prove_generated_world_rank_fidelity": True,
            "forward_inverse_consistency_does_not_prove_evaluation_readiness": True,
            "forward_inverse_consistency_does_not_prove_safety_or_srcc": True,
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        },
    }
    if command_result is not None:
        result["command_result"] = dict(command_result)
    return result


def _unscored_wam_episode_consistency(
    *,
    generated_at: str,
    rollouts: Sequence[Mapping[str, Any]],
    action_conditioned_video_rollout_generated: bool,
    action_conditioned_video_rollout_available: bool,
    provider_output_replay_used: bool,
    success_label_generated: bool,
    visual_smoke_status: str,
    visual_rollout_useful: bool,
    blockers: Sequence[str],
    blocked_reason: str | None,
) -> dict[str, Any]:
    visual_quality_blocked = bool(rollouts and not visual_rollout_useful)
    if visual_quality_blocked:
        needed = [
            "generated rollout video that passes visual-quality smoke review",
            "external VLM episode-consistency scorer command after video is reviewable",
        ]
        status = "blocked_generated_rollout_visual_quality"
        check_status = "blocked_generated_rollout_visual_quality"
        check_blocker = "blocked_generated_rollout_not_visually_useful_for_success_review"
        termination_reason = check_blocker
    elif rollouts:
        needed = [
            "external VLM episode-consistency scorer command",
            f"{WAM_CONSISTENCY_GATE_ENV}=true when automated scoring is used",
            "--allow-wam-consistency-scoring",
            "generated rollout video plus source action/trace context",
        ]
        status = "requires_review"
        check_status = "requires_external_episode_scorer"
        check_blocker = "requires_external_wam_episode_consistency_scorer"
        termination_reason = "model_output_available_needs_external_episode_consistency_scorer"
    else:
        needed = [
            "action-conditioned generated rollout video",
            "external VLM episode-consistency scorer command",
        ]
        status = "blocked"
        check_status = "blocked"
        check_blocker = blocked_reason
        termination_reason = blocked_reason
    return {
        "schema_version": "wam_consistency_checks.v1",
        "generated_at": generated_at,
        "status": status,
        "external_episode_consistency_scorer_ran": False,
        "external_episode_consistency_scorer_required": True,
        "external_episode_consistency_scorer_id": None,
        "forward_inverse_consistency_proven": False,
        "forward_dynamics_consistency_proven": False,
        "inverse_dynamics_consistency_proven": False,
        "action_conditioned_video_rollout_generated": action_conditioned_video_rollout_generated,
        "action_conditioned_video_rollout_available": action_conditioned_video_rollout_available,
        "provider_output_replay_used": provider_output_replay_used,
        "wam_success_label_from_generated_video": success_label_generated,
        "generated_rollout_visual_smoke_status": visual_smoke_status,
        "generated_rollout_visually_useful_for_success_review": visual_rollout_useful,
        "success_label_can_be_vlm_judged_separately_from_consistency": True,
        "what_is_needed_to_make_forward_inverse_consistency_true": needed,
        "checks": [
            {
                "check_id": "forward_dynamics_consistency",
                "status": check_status,
                "proven": False,
                "blocker": check_blocker,
                "what_is_needed_to_make_true": needed,
            },
            {
                "check_id": "inverse_dynamics_consistency",
                "status": check_status,
                "proven": False,
                "blocker": check_blocker,
                "what_is_needed_to_make_true": needed,
            },
        ],
        "blockers": list(blockers),
        "generated_rollout_termination_reason": termination_reason,
        "model_rollout_confidence": None,
        "claim_boundary": {
            "forward_inverse_consistency_is_external_episode_label_not_wam_execution": True,
            "forward_inverse_consistency_does_not_prove_task_success": True,
            "forward_inverse_consistency_does_not_prove_visual_rollout_useful_for_success_review": True,
            "forward_inverse_consistency_does_not_prove_generated_world_rank_fidelity": True,
            "forward_inverse_consistency_does_not_prove_evaluation_readiness": True,
            "forward_inverse_consistency_does_not_prove_safety_or_srcc": True,
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        },
    }


def _wam_consistency_blockers(consistency: Mapping[str, Any]) -> list[str]:
    blockers = _string_list(consistency.get("blockers"))
    blockers.extend(
        _string_list(consistency.get("what_is_needed_to_make_forward_inverse_consistency_true"))
    )
    raw_checks = consistency.get("checks")
    if isinstance(raw_checks, Sequence) and not isinstance(raw_checks, (str, bytes, bytearray)):
        for item in raw_checks:
            if not isinstance(item, Mapping):
                continue
            blockers.extend(_string_list(item.get("blockers")))
            blocker = _string(item.get("blocker"))
            if blocker:
                blockers.append(blocker)
    return sorted(dict.fromkeys(blocker for blocker in blockers if blocker))


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


def _generated_rollout_visual_smoke(
    *,
    rollouts: Sequence[Mapping[str, Any]],
    output_dir: Path,
    generated_at: str,
) -> dict[str, Any]:
    """Lightweight visual sanity check for generated rollout reviewability."""
    resolved_rollouts: list[dict[str, Any]] = []
    for rollout in rollouts:
        row = dict(rollout)
        video_path = _rollout_video_path(row, base_dir=output_dir)
        if video_path is not None:
            row["generated_video_path"] = str(video_path)
        resolved_rollouts.append(row)
    return visual_smoke_generated_rollouts_for_review(
        rollouts=resolved_rollouts,
        output_dir=output_dir,
        generated_at=generated_at,
    )


def _single_step_policy_requery_visual_candidate(
    visual_smoke: Mapping[str, Any],
) -> dict[str, Any]:
    """Return whether an early generated frame is usable for one policy requery."""
    rollout_rows = [
        row for row in visual_smoke.get("rollouts", []) if isinstance(row, Mapping)
    ]
    first_rollout = dict(rollout_rows[0]) if rollout_rows else {}
    flags = _mapping(first_rollout.get("visual_quality_flags"))
    sampled_frames = [
        row for row in first_rollout.get("sampled_frames", []) if isinstance(row, Mapping)
    ]
    first_sample = dict(sampled_frames[0]) if sampled_frames else {}
    first_frame_preserves_scene = bool(flags.get("first_frame_preserves_source_scene"))
    candidate_ready = bool(first_rollout and first_frame_preserves_scene and first_sample)
    blockers: list[str] = []
    if not first_rollout:
        blockers.append("blocked_missing_generated_rollout_for_policy_requery")
    if first_rollout and not first_frame_preserves_scene:
        blockers.append("blocked_generated_rollout_first_frame_not_scene_like_for_policy_requery")
    if first_rollout and not first_sample:
        blockers.append("blocked_missing_generated_rollout_sample_frame_for_policy_requery")
    return {
        "schema_version": "single_step_wam_policy_requery_visual_candidate.v1",
        "status": "ready_for_single_step_policy_requery" if candidate_ready else "blocked",
        "single_step_policy_requery_frame_useful": candidate_ready,
        "first_frame_preserves_source_scene": first_frame_preserves_scene,
        "first_sampled_frame": first_sample,
        "full_rollout_visual_smoke_status": visual_smoke.get("status"),
        "full_rollout_blockers": _string_list(visual_smoke.get("blockers")),
        "full_rollout_visually_useful_for_success_review": bool(
            _mapping(visual_smoke.get("claim_boundary")).get(
                "visual_rollout_useful_for_task_success_review"
            )
        ),
        "blockers": blockers,
        "claim_boundary": {
            "single_step_requery_candidate_is_not_success_review": True,
            "single_step_requery_candidate_is_not_forward_inverse_consistency": True,
            "raw_secret_values_recorded": False,
            "secret_hashes_recorded": False,
        },
    }


POLICY_REQUERY_ENDPOINT_ENVS = (
    {
        "runtime": "vla",
        "endpoint_env": "VLA_POLICY_ENDPOINT_URL",
        "auth_file_env": "VLA_POLICY_AUTH_TOKEN_FILE",
    },
    {
        "runtime": "team",
        "endpoint_env": "TEAM_POLICY_ENDPOINT_URL",
        "auth_file_env": "TEAM_POLICY_AUTH_TOKEN_FILE",
    },
)


def _policy_requery_endpoint_row() -> dict[str, Any] | None:
    for spec in POLICY_REQUERY_ENDPOINT_ENVS:
        endpoint = _string(os.getenv(spec["endpoint_env"]))
        token_file_raw = _string(os.getenv(spec["auth_file_env"]))
        token_file = Path(token_file_raw).expanduser() if token_file_raw else None
        if endpoint and token_file and token_file.is_file():
            return {
                "runtime": spec["runtime"],
                "endpoint_env": spec["endpoint_env"],
                "endpoint_url": endpoint,
                "auth_file_env": spec["auth_file_env"],
                "auth_token_file_path": str(token_file),
                "auth_token_file_exists": True,
                "auth_token_file_size_bytes": token_file.stat().st_size,
                "raw_token_values_persisted": False,
                "raw_token_hashes_persisted": False,
            }
    return None


def _policy_endpoint_health_url(endpoint_url: str) -> str:
    if endpoint_url.endswith("/policy/action"):
        return endpoint_url[: -len("/policy/action")] + "/health"
    return endpoint_url.rstrip("/") + "/health"


def _probe_policy_requery_endpoint_health(endpoint_url: str, timeout_seconds: float = 2.0) -> dict[str, Any]:
    health_url = _policy_endpoint_health_url(endpoint_url)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(health_url, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8") or "{}")
        if not isinstance(payload, Mapping):
            raise ValueError("endpoint_health_response_not_json_object")
        return {
            "status": "completed",
            "health_url": health_url,
            "http_status": 200,
            "duration_seconds": round(time.monotonic() - started, 6),
            "health_payload_redacted": _redact(payload),
            "raw_token_values_persisted": False,
            "raw_token_hashes_persisted": False,
        }
    except urllib.error.HTTPError as exc:
        return {
            "status": "blocked",
            "health_url": health_url,
            "http_status": exc.code,
            "duration_seconds": round(time.monotonic() - started, 6),
            "blockers": ["blocked_policy_requery_endpoint_health_http_error"],
            "raw_token_values_persisted": False,
            "raw_token_hashes_persisted": False,
        }
    except Exception as exc:
        return {
            "status": "blocked",
            "health_url": health_url,
            "duration_seconds": round(time.monotonic() - started, 6),
            "blockers": [f"blocked_policy_requery_endpoint_health_failed:{type(exc).__name__}"],
            "raw_token_values_persisted": False,
            "raw_token_hashes_persisted": False,
        }


def _build_policy_requery_endpoint_readiness_manifest(
    *,
    generated_at: str,
    input_dir: Path,
    visual_rollout_useful: bool,
    single_step_policy_requery_frame_useful: bool,
    visual_smoke_status: str,
) -> dict[str, Any]:
    source_summary = _load_json(
        input_dir / "mujoco_g1_wam_vla_policy_endpoint_eval_summary.json"
    ) or _load_json(input_dir / "policy_evaluation_summary.json")
    source_manipulation_report = _load_json(input_dir / "manipulation_endpoint_task_report.json")
    source_endpoint_policy_used = bool(source_summary.get("endpoint_policy_used"))
    source_fixture_policy_used = bool(source_summary.get("fixture_policy_used"))
    source_unitree_endpoint_hand_policy_used = bool(
        source_summary.get("unitree_endpoint_hand_policy_used")
        or source_manipulation_report.get("unitree_endpoint_hand_policy_used")
    )
    source_unitree_endpoint_fresh_policy_action_command_ran = bool(
        source_summary.get("unitree_endpoint_fresh_policy_action_command_ran")
        or source_manipulation_report.get("unitree_endpoint_fresh_policy_action_command_ran")
    )
    endpoint_rows: list[dict[str, Any]] = []
    live_ready = False
    configured_ready = False
    for spec in POLICY_REQUERY_ENDPOINT_ENVS:
        endpoint = _string(os.getenv(spec["endpoint_env"]))
        token_file_raw = _string(os.getenv(spec["auth_file_env"]))
        token_file = Path(token_file_raw).expanduser() if token_file_raw else None
        token_file_exists = bool(token_file and token_file.is_file())
        row_configured_ready = bool(endpoint and token_file_exists)
        configured_ready = configured_ready or row_configured_ready
        health_probe = (
            _probe_policy_requery_endpoint_health(endpoint) if row_configured_ready else None
        )
        row_ready = bool(
            row_configured_ready
            and isinstance(health_probe, Mapping)
            and health_probe.get("status") == "completed"
        )
        live_ready = live_ready or row_ready
        endpoint_rows.append(
            {
                "runtime": spec["runtime"],
                "endpoint_env": spec["endpoint_env"],
                "endpoint_url_configured": bool(endpoint),
                "auth_file_env": spec["auth_file_env"],
                "auth_token_file_configured": bool(token_file_raw),
                "auth_token_file_exists": token_file_exists,
                "configured_for_policy_requery": row_configured_ready,
                "health_probe": health_probe,
                "ready_for_policy_requery": row_ready,
                "raw_token_values_persisted": False,
                "raw_token_hashes_persisted": False,
            }
        )
    blockers: list[str] = []
    if not single_step_policy_requery_frame_useful:
        blockers.append("blocked_generated_rollout_not_visually_useful_for_policy_requery")
    if not configured_ready:
        blockers.append("blocked_missing_live_policy_requery_endpoint_env_or_auth")
    elif not live_ready:
        blockers.append("blocked_policy_requery_endpoint_health_probe_failed")
    if source_endpoint_policy_used and not live_ready:
        blockers.append("source_endpoint_proof_exists_but_endpoint_not_currently_live_for_requery")
    if not source_unitree_endpoint_hand_policy_used:
        blockers.append("blocked_source_job_did_not_use_unitree_hand_policy_endpoint")
    status = "ready_for_policy_requery" if not blockers else "blocked"
    return {
        "schema_version": "policy_requery_endpoint_readiness_manifest.v1",
        "generated_at": generated_at,
        "status": status,
        "source_mujoco_endpoint_eval_job_dir": str(input_dir),
        "source_endpoint_policy_used": source_endpoint_policy_used,
        "source_fixture_policy_used": source_fixture_policy_used,
        "source_unitree_endpoint_hand_policy_used": source_unitree_endpoint_hand_policy_used,
        "source_unitree_endpoint_fresh_policy_action_command_ran": (
            source_unitree_endpoint_fresh_policy_action_command_ran
        ),
        "source_endpoint_proof_is_not_current_live_endpoint": bool(
            source_endpoint_policy_used and not live_ready
        ),
        "generated_rollout_visual_smoke_status": visual_smoke_status,
        "generated_rollout_visually_useful_for_policy_requery": bool(
            single_step_policy_requery_frame_useful
        ),
        "full_rollout_visually_useful_for_success_review": bool(visual_rollout_useful),
        "policy_requery_endpoint_env_auth_configured": configured_ready,
        "live_policy_requery_endpoint_ready": live_ready,
        "endpoint_candidates": endpoint_rows,
        "required_endpoint_envs": [
            "VLA_POLICY_ENDPOINT_URL + VLA_POLICY_AUTH_TOKEN_FILE",
            "TEAM_POLICY_ENDPOINT_URL + TEAM_POLICY_AUTH_TOKEN_FILE",
        ],
        "reference_endpoint_creation_command": (
            "BLUEPRINT_WAM_VLA_POLICY_COMMAND=<runnable_unitree_policy_bridge_command> "
            "BLUEPRINT_WAM_VLA_POLICY_AUTH_TOKEN_FILE=$TEAM_POLICY_AUTH_TOKEN_FILE "
            "blueprint-serve-wam-vla-policy-endpoint --host 127.0.0.1 --port 8765"
        ),
        "unitree_unifolm_requery_bridge_command_shape": (
            "blueprint-unitree-unifolm-vla-server-bridge "
            "--server-url https://<pod_id>-8777.proxy.runpod.net/act"
        ),
        "what_is_needed_to_make_policy_requery_true": [
            "reviewable WAM-generated next-observation frame",
            "currently running Unitree-specific policy endpoint",
            "file-based endpoint auth token configured through endpoint URL/auth envs",
            "endpoint response from the generated observation returns a Unitree hand/manipulation action",
        ],
        "why_cannot_just_create_endpoint": [
            (
                "The HTTP wrapper can be created, but without a live runnable Unitree policy "
                "bridge command/server it only proves HTTP/auth plumbing."
            ),
            (
                "A previous source endpoint proof does not mean the same pod/server is still "
                "running for WAM requery."
            ),
            (
                "The current generated WAM rollout is not visually useful enough to feed back "
                "to the policy endpoint."
            ),
        ],
        "blockers": blockers,
        "claim_boundary": {
            "endpoint_creation_is_not_model_execution_proof": True,
            "source_endpoint_proof_is_not_current_live_requery_proof": True,
            "policy_requery_requires_reviewable_wam_generated_observation": True,
            "raw_credentials_written_to_artifacts": False,
            "raw_credential_hashes_written_to_artifacts": False,
        },
    }


def _extract_wam_requery_frame(
    *,
    rollout: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Path | None, dict[str, Any]]:
    video_path = _rollout_video_path(rollout, base_dir=output_dir)
    if video_path is None or not video_path.is_file():
        return None, {
            "status": "blocked",
            "blockers": ["blocked_generated_rollout_video_missing"],
        }
    if not shutil.which("ffmpeg"):
        return None, {
            "status": "blocked",
            "blockers": ["blocked_ffmpeg_unavailable_for_wam_requery_frame"],
        }
    frame_dir = output_dir / "wam_policy_requery_frames"
    ensure_dir(frame_dir)
    rollout_id = _string(rollout.get("rollout_id")) or "rollout"
    safe_rollout_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in rollout_id)
    frame_path = frame_dir / f"{safe_rollout_id}_frame_0000.jpg"
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(frame_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    if result.returncode != 0 or not frame_path.is_file():
        return None, {
            "status": "blocked",
            "returncode": result.returncode,
            "stderr_size_bytes": len(result.stderr or ""),
            "stderr_omitted_to_avoid_secret_leakage": bool(result.stderr),
            "blockers": ["blocked_wam_requery_frame_extract_failed"],
        }
    return frame_path, {
        "status": "completed",
        "source_generated_video_path": str(video_path),
        "extracted_frame_path": str(frame_path),
        "returncode": result.returncode,
    }


def _call_policy_requery_endpoint(
    *,
    endpoint_row: Mapping[str, Any],
    observation: Mapping[str, Any],
    timeout_seconds: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    token_file = Path(str(endpoint_row.get("auth_token_file_path"))).expanduser()
    endpoint_url = _string(endpoint_row.get("endpoint_url"))
    if not endpoint_url or not token_file.is_file():
        return None, {
            "status": "blocked",
            "endpoint_invoked": False,
            "blockers": ["blocked_missing_policy_requery_endpoint_or_auth"],
        }
    token = token_file.read_text(encoding="utf-8").strip()
    request = urllib.request.Request(
        endpoint_url,
        data=json.dumps({"observation": dict(observation)}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8") or "{}")
        if not isinstance(payload, Mapping):
            raise ValueError("policy_requery_response_not_json_object")
        return dict(payload), {
            "status": "completed",
            "endpoint_invoked": True,
            "duration_seconds": round(time.monotonic() - started, 6),
            "http_status": 200,
            "raw_token_values_persisted": False,
            "raw_token_hashes_persisted": False,
        }
    except urllib.error.HTTPError as exc:
        return None, {
            "status": "blocked",
            "endpoint_invoked": True,
            "duration_seconds": round(time.monotonic() - started, 6),
            "http_status": exc.code,
            "blockers": ["blocked_policy_requery_endpoint_http_error"],
            "raw_token_values_persisted": False,
            "raw_token_hashes_persisted": False,
        }
    except Exception as exc:
        return None, {
            "status": "blocked",
            "endpoint_invoked": False,
            "duration_seconds": round(time.monotonic() - started, 6),
            "blockers": [f"blocked_policy_requery_endpoint_call_failed:{type(exc).__name__}"],
            "raw_token_values_persisted": False,
            "raw_token_hashes_persisted": False,
        }


def _run_wam_policy_requery(
    *,
    output_dir: Path,
    generated_at: str,
    input_dir: Path,
    rollouts: Sequence[Mapping[str, Any]],
    visual_rollout_useful: bool,
    single_step_policy_requery_frame_useful: bool,
    visual_smoke_status: str,
    task_prompts: Sequence[Mapping[str, Any]],
    timeout_seconds: float,
) -> dict[str, Any]:
    base = {
        "schema_version": "wam_policy_requery_manifest.v1",
        "generated_at": generated_at,
        "source_mujoco_endpoint_eval_job_dir": str(input_dir),
        "scheduler_implemented": True,
        "policy_observes_wam_generated_next_observation": False,
        "closed_loop_policy_wam_interaction": False,
        "full_closed_loop_episode_proven": False,
        "every_frame_policy_wam_exchange": False,
        "generated_rollout_visual_smoke_status": visual_smoke_status,
        "generated_rollout_visually_useful_for_policy_requery": bool(
            single_step_policy_requery_frame_useful
        ),
        "full_rollout_visually_useful_for_success_review": bool(visual_rollout_useful),
        "raw_tokens_written_to_artifacts": False,
        "raw_token_hashes_written_to_artifacts": False,
    }
    if not rollouts:
        return {
            **base,
            "status": "blocked_missing_generated_rollout",
            "blockers": ["blocked_missing_generated_rollout"],
        }
    if not single_step_policy_requery_frame_useful:
        return {
            **base,
            "status": "blocked_generated_rollout_visual_quality",
            "blockers": ["blocked_generated_rollout_not_visually_useful_for_policy_requery"],
        }
    endpoint_row = _policy_requery_endpoint_row()
    if endpoint_row is None:
        return {
            **base,
            "status": "blocked_missing_policy_requery_endpoint",
            "blockers": ["blocked_missing_policy_requery_endpoint"],
            "required_endpoint_envs": [
                "VLA_POLICY_ENDPOINT_URL + VLA_POLICY_AUTH_TOKEN_FILE",
                "TEAM_POLICY_ENDPOINT_URL + TEAM_POLICY_AUTH_TOKEN_FILE",
            ],
        }
    rollout = dict(rollouts[0])
    frame_path, frame_meta = _extract_wam_requery_frame(rollout=rollout, output_dir=output_dir)
    if frame_path is None:
        return {
            **base,
            "status": "blocked_wam_requery_frame_unavailable",
            "selected_endpoint_runtime": endpoint_row.get("runtime"),
            "frame_extraction": frame_meta,
            "blockers": _string_list(frame_meta.get("blockers")),
        }
    prompt_by_run = {
        _string(row.get("scenario_eval_run_id")): row for row in task_prompts if isinstance(row, Mapping)
    }
    prompt_row = prompt_by_run.get(_string(rollout.get("scenario_eval_run_id")), {})
    observation = {
        "schema_version": "wam_generated_next_observation.v1",
        "observation_source": "wam_generated_rollout_video_frame",
        "scenario_eval_run_id": rollout.get("scenario_eval_run_id"),
        "task_id": rollout.get("task_id") or prompt_row.get("task_id"),
        "spawn_id": rollout.get("spawn_id") or prompt_row.get("spawn_id"),
        "task_prompt": rollout.get("task_prompt") or prompt_row.get("task_prompt"),
        "visual_observation": {
            "available": True,
            "camera_frame_path": str(frame_path),
            "generated_video_path": rollout.get("generated_video_path"),
            "camera_id": "wam_generated_next_observation",
            "wam_generated_observation": True,
            "simulated_camera_view": True,
            "physical_robot_sensor_proof": False,
        },
        "wam_context": {
            "rollout_id": rollout.get("rollout_id"),
            "policy_id": rollout.get("policy_id"),
            "model_rollout_confidence": rollout.get("model_rollout_confidence"),
            "generated_rollout_visual_smoke_status": visual_smoke_status,
        },
    }
    payload, endpoint_meta = _call_policy_requery_endpoint(
        endpoint_row=endpoint_row,
        observation=observation,
        timeout_seconds=timeout_seconds,
    )
    action = _mapping(payload).get("action") if payload else None
    completed = bool(payload and isinstance(action, Mapping))
    payload_mapping = _mapping(payload)
    raw_response = _mapping(_mapping(payload_mapping.get("endpoint_metadata")).get("raw_response_redacted"))
    response_claim_boundary = _mapping(raw_response.get("claim_boundary"))
    policy_id = _string(payload_mapping.get("policy_id") or raw_response.get("policy_id"))
    policy_kind = _string(raw_response.get("policy_kind"))
    unitree_hand_policy_used = bool(
        response_claim_boundary.get("unitree_hand_manipulation_policy_used")
        or response_claim_boundary.get("unitree_lerobot_or_isaaclab_manipulation_policy_used")
        or (
            policy_id
            in {
                "unitree_lerobot_g1_policy",
                "unitree_lerobot_g1_policy_provider_replay",
                "unitree_unifolm_vla_policy",
                "unitree_unifolm_vla_policy_provider_replay",
                "unitree_unifolm_wma_policy",
                "unitree_unifolm_wma_policy_provider_replay",
            }
        )
    )
    openvla_policy_used = bool(
        response_claim_boundary.get("openvla_model_executed")
        or policy_id in {"openvla_policy", "openvla_policy_provider_replay"}
    )
    provider_replay_used = bool(response_claim_boundary.get("provider_output_replay_used"))
    unitree_g1_hand_policy_output_observed = bool(
        unitree_hand_policy_used
        or response_claim_boundary.get("unitree_g1_embodiment_decoder_configured")
    )
    unitree_g1_hand_policy_endpoint_used = bool(
        unitree_g1_hand_policy_output_observed and not provider_replay_used
    )
    real_vla_or_unitree_hand_policy_endpoint_used = unitree_g1_hand_policy_endpoint_used
    single_step_policy_requery_proven = bool(
        completed and unitree_g1_hand_policy_endpoint_used
    )
    if single_step_policy_requery_proven:
        blockers: list[str] = []
    elif completed and provider_replay_used and unitree_g1_hand_policy_output_observed:
        blockers = [
            "blocked_policy_requery_provider_replay_not_fresh_unitree_hand_policy",
            "blocked_policy_requery_endpoint_not_real_vla_or_unitree_hand_policy",
        ]
    elif completed:
        blockers = [
            "blocked_policy_requery_endpoint_not_unitree_g1_hand_policy",
            "blocked_policy_requery_endpoint_not_real_vla_or_unitree_hand_policy",
        ]
    else:
        blockers = _string_list(endpoint_meta.get("blockers")) or [
            "blocked_policy_requery_missing_action"
        ]
    status = "completed" if single_step_policy_requery_proven else (
        "blocked_policy_requery_provider_replay_not_fresh_unitree_hand_policy"
        if completed and provider_replay_used and unitree_g1_hand_policy_output_observed
        else "blocked_policy_requery_endpoint_not_unitree_g1_hand_policy"
        if completed
        else "blocked_policy_requery_failed"
    )
    return {
        **base,
        "status": status,
        "selected_endpoint_runtime": endpoint_row.get("runtime"),
        "selected_endpoint_env": endpoint_row.get("endpoint_env"),
        "endpoint_url": endpoint_row.get("endpoint_url"),
        "endpoint_invoked": bool(endpoint_meta.get("endpoint_invoked")),
        "endpoint_action_returned_for_wam_generated_next_observation": completed,
        "policy_observes_wam_generated_next_observation": single_step_policy_requery_proven,
        "closed_loop_policy_wam_interaction": single_step_policy_requery_proven,
        "single_step_wam_policy_requery_proven": single_step_policy_requery_proven,
        "policy_requery_policy_id": policy_id or None,
        "policy_requery_policy_kind": policy_kind or None,
        "real_vla_or_unitree_hand_policy_endpoint_used": (
            real_vla_or_unitree_hand_policy_endpoint_used
        ),
        "unitree_g1_hand_policy_output_observed": unitree_g1_hand_policy_output_observed,
        "unitree_family_policy_output_observed_for_wam_requery": (
            unitree_g1_hand_policy_output_observed
        ),
        "unitree_g1_hand_policy_endpoint_used": unitree_g1_hand_policy_endpoint_used,
        "g1_robot_policy_is_unitree_native": unitree_g1_hand_policy_endpoint_used,
        "g1_robot_policy_selection_contract": "unitree_native_policy_required_for_g1_claims",
        "g1_robot_policy_selected_family": "unitree_native_hand_policy_endpoint"
        if unitree_g1_hand_policy_endpoint_used
        else None,
        "openvla_selected_as_g1_robot_policy": False,
        "wam_rollout_selected_as_g1_robot_policy": False,
        "unitree_hand_policy_requery_used": unitree_g1_hand_policy_endpoint_used,
        "unitree_hand_policy_requery_output_observed": unitree_hand_policy_used,
        "openvla_policy_requery_used": openvla_policy_used,
        "policy_requery_provider_replay_used": provider_replay_used,
        "fresh_unitree_hand_policy_requery_inference_proven": (
            unitree_g1_hand_policy_endpoint_used
        ),
        "policy_requery_provider_replay_is_not_fresh_policy_observation": bool(
            provider_replay_used
        ),
        "full_closed_loop_episode_proven": False,
        "frame_extraction": frame_meta,
        "policy_requery_observation": observation,
        "policy_requery_response_redacted": _redact(payload) if payload else None,
        "policy_requery_action": dict(action) if isinstance(action, Mapping) else None,
        "endpoint_meta": endpoint_meta,
        "blockers": blockers,
        "claim_boundary": {
            "single_step_wam_policy_requery_is_not_task_success": True,
            "full_closed_loop_episode_proven": False,
            "generated_observation_is_model_output_not_raw_capture": True,
            "generic_team_endpoint_action_is_not_real_vla_or_unitree_hand_policy_proof": True,
            "openvla_policy_is_not_selected_g1_robot_policy": True,
            "wam_rollout_is_not_selected_g1_robot_policy": True,
            "g1_robot_policy_selection_contract": "unitree_native_policy_required_for_g1_claims",
            "real_vla_or_unitree_hand_policy_endpoint_used": (
                real_vla_or_unitree_hand_policy_endpoint_used
            ),
            "unitree_g1_hand_policy_output_observed": unitree_g1_hand_policy_output_observed,
            "unitree_g1_hand_policy_endpoint_used": unitree_g1_hand_policy_endpoint_used,
            "unitree_hand_manipulation_policy_used": unitree_hand_policy_used,
            "openvla_policy_used": openvla_policy_used,
            "provider_output_replay_used": provider_replay_used,
            "provider_replay_is_not_fresh_policy_observation": bool(provider_replay_used),
            "generated_world_rank_fidelity_result_proven": False,
        },
    }


def run_oscar_cosmos_wam_evaluator(
    *,
    input_job_dir: Path,
    job_dir: Path | None = None,
    job_root: Path | None = None,
    model_candidates: Sequence[str] = DEFAULT_MODEL_CANDIDATES,
    wam_model_command: str | None = None,
    wam_model_checkpoint: Path | None = None,
    allow_wam_model_run: bool = False,
    wam_success_label_command: str | None = None,
    allow_wam_success_labeling: bool = False,
    wam_consistency_command: str | None = None,
    allow_wam_consistency_scoring: bool = False,
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
    eval_ready_grounding, eval_ready_grounding_source_path = _load_eval_ready_task_grounding(
        input_dir
    )
    eval_ready_grounding_artifacts = (
        _copy_grounding_support_artifacts(
            grounding=eval_ready_grounding,
            grounding_path=eval_ready_grounding_source_path,
            output_dir=output_dir,
        )
        if eval_ready_grounding
        else {}
    )

    scenario_matrix = _load_json(input_dir / "scenario_eval_matrix.json")
    attempt_trace = _load_json(input_dir / "normalized_attempt_trace.json")
    action_rows = _read_jsonl(input_dir / "normalized_policy_action_trace.jsonl")
    locomotion_rows = _read_jsonl(input_dir / "g1_mujoco_locomotion_trace.jsonl")
    g1_projected_skeleton_trace_path = input_dir / "g1_projected_skeleton_trace.jsonl"
    g1_projected_skeleton_rows = _read_jsonl(g1_projected_skeleton_trace_path)
    g1_projected_skeleton_manifest = _load_json(
        input_dir / "g1_projected_skeleton_manifest.json"
    )
    generic_fk_projection_manifest_path = Path(
        eval_ready_grounding_artifacts.get("robot_fk_projection_manifest", "")
    )
    generic_fk_projection_manifest = _load_json(generic_fk_projection_manifest_path)
    generic_fk_projection_trace_path = Path(
        eval_ready_grounding_artifacts.get("robot_fk_projected_skeleton_trace", "")
    )
    generic_fk_projection_rows = _read_jsonl(generic_fk_projection_trace_path)
    if eval_ready_grounding:
        handle_check_path = Path(eval_ready_grounding_artifacts.get("handle_proxy_state_check", ""))
        if handle_check_path.is_file():
            eval_ready_grounding["handle_proxy_state_check"] = _load_json(handle_check_path)
        calibration_gate_path = Path(
            eval_ready_grounding_artifacts.get("camera_calibration_quality_gate", "")
        )
        if calibration_gate_path.is_file():
            eval_ready_grounding["camera_calibration_quality_gate"] = _load_json(
                calibration_gate_path
            )
        if generic_fk_projection_manifest:
            eval_ready_grounding["robot_fk_projection"] = generic_fk_projection_manifest
    generic_fk_projection_available = bool(
        generic_fk_projection_manifest.get("status") == "completed"
        and generic_fk_projection_rows
        and any(
            int(row.get("projected_landmark_count") or 0) > 0
            for row in generic_fk_projection_rows
        )
    )
    g1_projected_skeleton_available = bool(
        g1_projected_skeleton_rows
        and any(
            int(row.get("projected_landmark_count") or 0) > 0
            for row in g1_projected_skeleton_rows
        )
    )
    unitree_controller_proof = _source_unitree_controller_proof(
        input_dir,
        locomotion_rows=locomotion_rows,
    )
    write_json(output_dir / "source_unitree_controller_proof.json", unitree_controller_proof)
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
    local_model_gate_enabled = bool(
        allow_wam_model_run or _env_truthy(LOCAL_MODEL_GATE_ENV)
    )
    runtime_discovery = discover_wam_model_runtimes(
        candidates=model_candidates,
        generated_at=generated,
        explicit_candidate_id=model_candidates[0] if model_candidates else None,
        explicit_command=wam_model_command,
        explicit_checkpoint=wam_model_checkpoint,
        local_model_gate_enabled_override=local_model_gate_enabled,
    )
    write_json(output_dir / "wam_model_runtime_discovery.json", runtime_discovery)
    endpoint_readiness = build_policy_model_endpoint_readiness_manifest(
        generated_at=generated,
        candidates=tuple(dict.fromkeys(tuple(model_candidates) + ENDPOINT_READINESS_CANDIDATES)),
        explicit_candidate_id=model_candidates[0] if model_candidates else None,
        explicit_command=wam_model_command,
        explicit_checkpoint=wam_model_checkpoint,
        local_model_gate_enabled_override=local_model_gate_enabled,
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
    configured_backend_ids = [
        _string(row.get("candidate_id"))
        for row in runtime_discovery.get("candidates", []) or []
        if isinstance(row, Mapping)
        and (row.get("command_configured") or row.get("checkpoint_configured"))
    ]
    backend_strategy_manifest = build_wam_backend_strategy_manifest(
        generated_at=generated,
        selected_backend_ids=model_candidates,
        configured_backend_ids=configured_backend_ids,
    )
    write_json(
        output_dir / "wam_backend_strategy_manifest.json",
        backend_strategy_manifest,
    )
    openvla_provider_smoke_requested = bool(
        "openvla_policy" in model_candidates
        or os.getenv("BLUEPRINT_OPENVLA_PROVIDER_SMOKE_JOB_DIR")
    )
    openvla_provider_smoke_proof = (
        discover_openvla_provider_smoke_proof(repo_root=_repo_root())
        if openvla_provider_smoke_requested
        else _skipped_openvla_provider_smoke_proof(
            "skipped_openvla_provider_smoke_not_selected_for_this_run"
        )
    )
    write_json(
        output_dir / "openvla_provider_smoke_proof.json",
        openvla_provider_smoke_proof,
    )
    unitree_unifolm_provider_smoke_proof = discover_unitree_unifolm_provider_smoke_proof()
    write_json(
        output_dir / "unitree_unifolm_provider_smoke_proof.json",
        unitree_unifolm_provider_smoke_proof,
    )
    write_json(
        output_dir / "policy_model_candidate_matrix.json",
        _candidate_matrix(
            generated,
            openvla_provider_smoke_proof=openvla_provider_smoke_proof,
        ),
    )

    wam_input_videos = [
        row for row in videos if _is_egocentric_wam_input_video(_mapping(row))
    ]
    diagnostic_review_videos = [
        row for row in videos if not _is_egocentric_wam_input_video(_mapping(row))
    ]
    rollout_input_blockers: list[str] = []
    if not attempts or not action_rows:
        rollout_input_blockers.append("blocked_missing_inputs")
    if not wam_input_videos:
        rollout_input_blockers.append("blocked_missing_egocentric_wam_input_video")
    if eval_ready_grounding:
        grounding_readiness = _mapping(eval_ready_grounding.get("readiness"))
        if not grounding_readiness.get("learned_rollout_request_ready"):
            rollout_input_blockers.append("blocked_eval_ready_task_grounding_not_ready")
            for blocker in _string_list(grounding_readiness.get("blockers")):
                rollout_input_blockers.append(f"eval_ready:{blocker}")
        if not grounding_readiness.get("robot_projection_ready"):
            rollout_input_blockers.append("blocked_eval_ready_robot_projection_not_ready")
        if _mapping(eval_ready_grounding.get("camera_calibration_quality_gate")).get(
            "status"
        ) == "blocked":
            rollout_input_blockers.append("blocked_eval_ready_camera_calibration_quality")
    rollout_input_status = (
        "ready_for_model"
        if not rollout_input_blockers
        else "blocked_missing_egocentric_wam_input_video"
        if rollout_input_blockers == ["blocked_missing_egocentric_wam_input_video"]
        else "blocked_missing_inputs"
    )
    task_prompt_rows = _grounding_enriched_task_prompts(
        matrix_runs=matrix_runs,
        grounding=eval_ready_grounding,
    )
    rollout_input_manifest = {
        "schema_version": "wam_rollout_input_manifest.v1",
        "generated_at": generated,
        "status": rollout_input_status,
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
            "g1_projected_skeleton_trace_jsonl": str(g1_projected_skeleton_trace_path)
            if g1_projected_skeleton_trace_path.is_file()
            else None,
            "g1_projected_skeleton_manifest": str(
                input_dir / "g1_projected_skeleton_manifest.json"
            )
            if (input_dir / "g1_projected_skeleton_manifest.json").is_file()
            else None,
            "eval_ready_task_grounding": eval_ready_grounding_artifacts.get(
                "eval_ready_task_grounding"
            ),
            "robot_fk_projection_manifest": eval_ready_grounding_artifacts.get(
                "robot_fk_projection_manifest"
            ),
            "robot_fk_projected_skeleton_trace_jsonl": eval_ready_grounding_artifacts.get(
                "robot_fk_projected_skeleton_trace"
            ),
            "camera_calibration_quality_gate": eval_ready_grounding_artifacts.get(
                "camera_calibration_quality_gate"
            ),
            "handle_proxy_state_check": eval_ready_grounding_artifacts.get(
                "handle_proxy_state_check"
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
            "g1_projected_skeleton_row_count": len(g1_projected_skeleton_rows),
            "g1_projected_skeleton_projectable_row_count": sum(
                1
                for row in g1_projected_skeleton_rows
                if int(row.get("projected_landmark_count") or 0) > 0
            ),
            "robot_fk_projection_trace_row_count": len(generic_fk_projection_rows),
            "robot_fk_projection_projectable_row_count": sum(
                1
                for row in generic_fk_projection_rows
                if int(row.get("projected_landmark_count") or 0) > 0
            ),
            "selected_review_video_count": len(videos),
            "wam_input_video_count": len(wam_input_videos),
            "diagnostic_review_video_count": len(diagnostic_review_videos),
        },
        "eval_ready_task_grounding": {
            "available": bool(eval_ready_grounding),
            "source_path": str(eval_ready_grounding_source_path)
            if eval_ready_grounding_source_path
            else None,
            "status": eval_ready_grounding.get("status"),
            "learned_rollout_request_ready": _mapping(
                eval_ready_grounding.get("readiness")
            ).get("learned_rollout_request_ready"),
            "robot_projection_ready": _mapping(eval_ready_grounding.get("readiness")).get(
                "robot_projection_ready"
            ),
            "selected_task_target": eval_ready_grounding.get("selected_task_target"),
            "success_check_plan": eval_ready_grounding.get("success_check_plan"),
        },
        "wam_input_video_contract": {
            "required_camera_role": "egocentric_robot_policy_observation_candidate",
            "egocentric_wam_input_camera_ids": list(EGOCENTRIC_WAM_INPUT_CAMERAS),
            "diagnostic_review_camera_ids": list(DIAGNOSTIC_REVIEW_CAMERAS),
            "third_person_overview_is_diagnostic_not_policy_observation": True,
        },
        "wam_input_videos": wam_input_videos,
        "diagnostic_review_videos": diagnostic_review_videos,
        "selected_review_videos": videos,
        "blockers": rollout_input_blockers,
        "task_prompts": task_prompt_rows,
    }
    write_json(output_dir / "wam_rollout_input_manifest.json", rollout_input_manifest)

    conditioning_sources = [
        "normalized_policy_action_trace.jsonl",
        "g1_mujoco_locomotion_trace.jsonl",
        "selected_review_videos_or_posters",
        "task_prompts",
    ]
    if g1_projected_skeleton_available:
        conditioning_sources.insert(2, "g1_projected_skeleton_trace.jsonl")
    if generic_fk_projection_available:
        conditioning_sources.insert(2, "robot_fk_projected_skeleton_trace.jsonl")

    action_conditioning = {
        "schema_version": "wam_action_conditioning_manifest.v1",
        "generated_at": generated,
        "status": "completed" if action_rows else "blocked_missing_action_trace",
        "conditioning_sources": conditioning_sources,
        "action_type_counts": _action_counts(action_rows),
        "robot_pose_encoding": {
            "source": (
                "g1_projected_skeleton_trace.simulated_g1_upper_body_landmarks"
                if g1_projected_skeleton_available
                else "g1_mujoco_locomotion_trace.root_position/root_quaternion_wxyz/root_yaw_rad"
            ),
            "skeleton_encoding_available": bool(locomotion_rows),
            "unitree_g1_joint_state_available": bool(locomotion_rows),
            "projected_g1_upper_body_skeleton_available": g1_projected_skeleton_available,
            "projected_g1_upper_body_skeleton_trace": str(g1_projected_skeleton_trace_path)
            if g1_projected_skeleton_trace_path.is_file()
            else None,
            "projected_g1_upper_body_skeleton_manifest": str(
                input_dir / "g1_projected_skeleton_manifest.json"
            )
            if (input_dir / "g1_projected_skeleton_manifest.json").is_file()
            else None,
            "projected_g1_upper_body_skeleton_row_count": len(g1_projected_skeleton_rows),
            "projected_g1_upper_body_skeleton_projectable_row_count": sum(
                1
                for row in g1_projected_skeleton_rows
                if int(row.get("projected_landmark_count") or 0) > 0
            ),
            "projected_g1_upper_body_skeleton_manifest_status": g1_projected_skeleton_manifest.get(
                "status"
            ),
            "generic_robot_fk_projection_available": generic_fk_projection_available,
            "generic_robot_fk_projection_manifest": str(generic_fk_projection_manifest_path)
            if generic_fk_projection_manifest_path.is_file()
            else None,
            "generic_robot_fk_projected_skeleton_trace": str(generic_fk_projection_trace_path)
            if generic_fk_projection_trace_path.is_file()
            else None,
            "generic_robot_fk_projection_row_count": len(generic_fk_projection_rows),
            "generic_robot_fk_projection_projectable_row_count": sum(
                1
                for row in generic_fk_projection_rows
                if int(row.get("projected_landmark_count") or 0) > 0
            ),
            "generic_robot_fk_projection_confidence": generic_fk_projection_manifest.get(
                "projection_confidence"
            ),
            "projected_skeleton_is_simulated_mujoco_state": True,
            "projected_skeleton_is_not_physical_robot_proprioception": True,
            "projected_skeleton_does_not_prove_wam_visual_usefulness": True,
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
        and local_model_gate_enabled
        and command_probe_candidates
        and rollout_input_manifest["status"] == "ready_for_model"
    )
    model_payload: dict[str, Any] = {}
    model_execution_detail = {
        "status": "blocked",
        "blockers": runtime_discovery.get("blockers", [])
        or ["blocked_local_wam_model_run_not_enabled"],
        "local_model_gate_enabled": local_model_gate_enabled,
        "local_model_gate_enabled_by_cli_or_function_flag": bool(
            allow_wam_model_run and not _env_truthy(LOCAL_MODEL_GATE_ENV)
        ),
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
    endpoint_probe_results = _build_policy_model_endpoint_probe_results(
        generated_at=generated,
        readiness_manifest=endpoint_readiness,
        selected_candidate_id=_string(runtime_discovery.get("selected_candidate")) or None,
        model_run_allowed=model_run_allowed,
        model_payload=model_payload,
        model_execution_detail=model_execution_detail,
    )
    write_json(output_dir / "policy_model_endpoint_probe_results.json", endpoint_probe_results)

    rollouts = []
    generated_video_review_validations: list[dict[str, Any]] = []
    raw_rollouts = model_payload.get("rollouts") or _mapping(
        model_payload.get("wam_generated_rollout_results")
    ).get("rollouts")
    missing_video_count = 0
    invalid_video_count = 0
    if isinstance(raw_rollouts, list):
        for item in raw_rollouts:
            if not isinstance(item, Mapping):
                continue
            row = dict(item)
            video_path = _rollout_video_path(row, base_dir=output_dir)
            if video_path and video_path.is_file():
                resolved_video = video_path.resolve()
                video_validation = validate_generated_mp4_for_review(resolved_video)
                generated_video_review_validations.append(
                    {
                        "rollout_id": row.get("rollout_id"),
                        **video_validation,
                    }
                )
                if video_validation.get("status") == "completed":
                    row["generated_video_path"] = str(resolved_video)
                    row["generated_video_review_validation"] = video_validation
                    rollouts.append(row)
                else:
                    invalid_video_count += 1
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
    generated_blocker_set.update(rollout_input_blockers)
    if missing_video_count:
        generated_blocker_set.add("blocked_generated_rollout_video_missing")
    if invalid_video_count:
        generated_blocker_set.add("blocked_generated_rollout_video_not_reviewable")
        for validation in generated_video_review_validations:
            generated_blocker_set.update(_string_list(validation.get("blockers")))
    generated_blockers = [] if rollouts else sorted(
        generated_blocker_set or {"blocked_missing_wam_runtime", "blocked_missing_wam_model_checkpoint"}
    )
    model_payload_mode = _string(model_payload.get("mode"))
    provider_output_replay_used = model_payload_mode == "replay_existing_provider_output"
    model_command_executed_this_invocation = bool(
        model_run_allowed and model_execution_detail.get("status") == "completed"
    )
    fresh_model_command_executed_this_invocation = bool(
        model_command_executed_this_invocation and not provider_output_replay_used
    )
    learned_wam_model_output_available = bool(rollouts)
    provider_learned_wam_model_ran = bool(model_payload.get("provider_learned_wam_model_ran"))
    provider_generated_video_is_model_output = bool(
        model_payload.get("provider_generated_video_is_model_output")
    )
    fresh_wam_model_execution_proven = _wam_model_payload_proves_fresh_execution(
        model_payload=model_payload,
        model_execution_detail=model_execution_detail,
        provider_output_replay_used=provider_output_replay_used,
    )
    provider_wam_model_output_proven = _provider_wam_payload_proves_model_output(
        model_payload
    )
    learned_wam_model_ran_or_imported = bool(
        learned_wam_model_output_available
        and (
            fresh_wam_model_execution_proven
            or provider_wam_model_output_proven
        )
    )
    openvla_provider_smoke_imported = bool(
        openvla_provider_smoke_proof.get("openvla_policy_action_command_imported")
    )
    openvla_policy_action_command_ran = bool(
        openvla_provider_smoke_proof.get("openvla_policy_action_command_ran")
    )
    unitree_policy_action_command_ran = bool(
        unitree_unifolm_provider_smoke_proof.get(
            "unitree_unifolm_policy_action_command_ran"
        )
    )
    unitree_locomotion_policy_ran = bool(
        unitree_controller_proof.get("unitree_locomotion_policy_ran")
    )
    policy_action_model_command_ran = bool(
        openvla_policy_action_command_ran or unitree_policy_action_command_ran
    )
    policy_action_model_provider_smoke_imported = bool(
        openvla_provider_smoke_imported or unitree_policy_action_command_ran
    )
    oscar_cosmos_openvla_unitree_model_ran = bool(
        learned_wam_model_ran_or_imported
        or policy_action_model_command_ran
        or unitree_locomotion_policy_ran
    )
    learned_wam_model_ran_this_invocation = bool(fresh_wam_model_execution_proven)
    blocked_reason = None if rollouts else _wam_rollout_blocked_reason(generated_blockers)
    generated_status = "completed" if rollouts else blocked_reason
    generated_manifest = {
        "schema_version": "wam_generated_rollout_manifest.v1",
        "generated_at": generated,
        "status": generated_status,
        "selected_model_candidate": runtime_discovery.get("selected_candidate"),
        "model_command_executed_this_invocation": model_command_executed_this_invocation,
        "fresh_model_command_executed_this_invocation": fresh_model_command_executed_this_invocation,
        "action_conditioned_video_rollout_generated": learned_wam_model_output_available,
        "action_conditioned_video_rollout_available": learned_wam_model_output_available,
        "valid_reviewable_generated_video_available": learned_wam_model_output_available,
        "provider_output_replay_used": provider_output_replay_used,
        "provider_learned_wam_model_ran": provider_learned_wam_model_ran,
        "provider_generated_video_is_model_output": provider_generated_video_is_model_output,
        "fresh_wam_model_execution_proven": fresh_wam_model_execution_proven,
        "provider_wam_model_output_proven": provider_wam_model_output_proven,
        "learned_wam_model_ran_or_imported": learned_wam_model_ran_or_imported,
        "generated_rollout_count": len(rollouts),
        "generated_video_review_validations": generated_video_review_validations,
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
        "generated_video_review_validations": generated_video_review_validations,
        "blocked_reason": blocked_reason,
        "blockers": generated_blockers,
        "provider_output_replay_used": provider_output_replay_used,
        "model_command_executed_this_invocation": model_command_executed_this_invocation,
        "fresh_model_command_executed_this_invocation": fresh_model_command_executed_this_invocation,
        "learned_wam_model_ran_this_invocation": learned_wam_model_ran_this_invocation,
        "provider_learned_wam_model_ran": provider_learned_wam_model_ran,
        "provider_generated_video_is_model_output": provider_generated_video_is_model_output,
        "learned_wam_model_ran_or_imported": learned_wam_model_ran_or_imported,
    }
    write_json(output_dir / "wam_generated_rollout_results.json", generated_results)
    _write_jsonl(output_dir / "wam_generated_rollout_results.jsonl", rollouts)
    visual_smoke = _generated_rollout_visual_smoke(
        rollouts=rollouts,
        output_dir=output_dir,
        generated_at=generated,
    )
    write_json(output_dir / "wam_generated_rollout_visual_smoke.json", visual_smoke)
    visual_smoke_status = _string(visual_smoke.get("status"))
    visual_rollout_useful = bool(
        _mapping(visual_smoke.get("claim_boundary")).get(
            "visual_rollout_useful_for_task_success_review"
        )
    )
    single_step_policy_requery_visual_candidate = (
        _single_step_policy_requery_visual_candidate(visual_smoke)
    )
    write_json(
        output_dir / "single_step_wam_policy_requery_visual_candidate.json",
        single_step_policy_requery_visual_candidate,
    )
    single_step_policy_requery_frame_useful = bool(
        single_step_policy_requery_visual_candidate.get(
            "single_step_policy_requery_frame_useful"
        )
    )
    visual_quality_blockers = (
        sorted(
            set(
                ["blocked_generated_rollout_not_visually_useful_for_success_review"]
                + _string_list(visual_smoke.get("blockers"))
            )
        )
        if rollouts and not visual_rollout_useful
        else []
    )
    generated_manifest["generated_rollout_visual_smoke_status"] = visual_smoke_status
    generated_manifest["generated_rollout_visually_useful_for_success_review"] = (
        visual_rollout_useful
    )
    generated_results["generated_rollout_visual_smoke_status"] = visual_smoke_status
    generated_results["generated_rollout_visually_useful_for_success_review"] = (
        visual_rollout_useful
    )
    if visual_quality_blockers:
        generated_manifest["status"] = "completed_visual_quality_failed"
        generated_manifest["valid_reviewable_generated_video_available"] = False
        generated_manifest["generated_rollout_visually_useful_for_success_review"] = False
        generated_manifest["blockers"] = visual_quality_blockers
        generated_results["status"] = "completed_visual_quality_failed"
        generated_results["valid_reviewable_generated_video_available"] = False
        generated_results["generated_rollout_visually_useful_for_success_review"] = False
        generated_results["blockers"] = visual_quality_blockers
    write_json(output_dir / "wam_generated_rollout_manifest.json", generated_manifest)
    write_json(output_dir / "wam_generated_rollout_results.json", generated_results)

    success_label_request = {
        "schema_version": "wam_success_label_request.v1",
        "generated_at": generated,
        "status": "ready_for_vlm_judge"
        if rollouts and visual_rollout_useful
        else "blocked_generated_rollout_visual_quality"
        if rollouts
        else "blocked_missing_generated_rollout",
        "source_mujoco_endpoint_eval_job_dir": str(input_dir),
        "generated_rollout_results": str(output_dir / "wam_generated_rollout_results.json"),
        "generated_rollout_visual_smoke": str(output_dir / "wam_generated_rollout_visual_smoke.json"),
        "generated_rollout_visual_smoke_status": visual_smoke_status,
        "generated_rollout_visually_useful_for_success_review": visual_rollout_useful,
        "rollouts": rollouts,
        "blockers": visual_quality_blockers if rollouts and not visual_rollout_useful else generated_blockers,
        "task_prompts": rollout_input_manifest["task_prompts"],
        "eval_ready_task_grounding": {
            "available": bool(eval_ready_grounding),
            "path": eval_ready_grounding_artifacts.get("eval_ready_task_grounding"),
            "selected_task_target": eval_ready_grounding.get("selected_task_target"),
            "success_check_plan": eval_ready_grounding.get("success_check_plan"),
            "handle_proxy_state_check": eval_ready_grounding.get("handle_proxy_state_check"),
            "camera_calibration_quality_gate": eval_ready_grounding.get(
                "camera_calibration_quality_gate"
            ),
        },
        "success_label_contract": {
            "expected_output_path": str(output_dir / WAM_SUCCESS_LABEL_COMMAND_OUTPUT),
            "required_top_level_keys": ["labels"],
            "label_required_keys": ["rollout_id", "success", "confidence", "rationale"],
        },
        "claim_boundary": {
            "judge_input_is_generated_video_not_raw_robot_evidence": True,
            "judge_success_label_does_not_prove_generated_world_rank_fidelity": True,
            "judge_success_label_does_not_prove_forward_inverse_consistency": True,
            "raw_credentials_written_to_artifacts": False,
        },
    }
    write_json(output_dir / "wam_success_label_request.json", success_label_request)

    configured_success_label_command = _string(
        wam_success_label_command or os.getenv(WAM_SUCCESS_LABEL_COMMAND_ENV)
    )
    success_label_blockers: list[str] = []
    success_label_command_result: dict[str, Any] | None = None
    success_label_command_payload: dict[str, Any] = {}
    if not rollouts:
        success_label_blockers = list(generated_blockers)
    elif not visual_rollout_useful:
        success_label_blockers = list(visual_quality_blockers)
    elif allow_wam_success_labeling or configured_success_label_command:
        if not _env_truthy(WAM_SUCCESS_LABEL_GATE_ENV):
            success_label_blockers.append(f"missing_env_{WAM_SUCCESS_LABEL_GATE_ENV}")
        if not allow_wam_success_labeling:
            success_label_blockers.append("missing_cli_allow_wam_success_labeling")
        if not configured_success_label_command:
            success_label_blockers.append("missing_wam_success_label_command")
        if not success_label_blockers:
            success_label_command_payload, success_label_command_result = _run_wam_success_label_command(
                command=configured_success_label_command,
                input_path=output_dir / "wam_success_label_request.json",
                output_path=output_dir / WAM_SUCCESS_LABEL_COMMAND_OUTPUT,
                timeout_seconds=timeout_seconds,
            )
            if success_label_command_result.get("status") != "completed":
                success_label_blockers.extend(
                    _string_list(success_label_command_result.get("blockers"))
                    or ["wam_success_label_command_blocked"]
                )
    else:
        success_label_blockers = ["requires_wam_success_review"]

    if success_label_command_payload and not success_label_blockers:
        success_labels = _normalize_wam_success_labels(
            command_payload=success_label_command_payload,
            rollouts=rollouts,
            generated_at=generated,
            visual_smoke_status=visual_smoke_status,
            visual_rollout_useful=visual_rollout_useful,
        )
        success_label_blockers = _string_list(success_labels.get("blockers"))
    else:
        success_labels = {
            "schema_version": "wam_success_labels.v1",
            "generated_at": generated,
            "status": "blocked" if not rollouts or not visual_rollout_useful else "requires_review",
            "wam_success_label_from_generated_video": False,
            "visual_smoke_status": visual_smoke_status,
            "visual_rollout_useful_for_task_success_review": visual_rollout_useful,
            "review_grade_visual_evidence_available": visual_rollout_useful,
            "review_grade_success_labels": False,
            "label_count": 0,
            "labels": [],
            "blockers": success_label_blockers,
            "command_result": success_label_command_result,
            "human_review_required": bool(rollouts and visual_rollout_useful),
            "claim_boundary": {
                "success_label_is_from_generated_video_not_physical_robot": True,
                "success_label_requires_passed_visual_smoke": True,
                "visual_rollout_useful_for_task_success_review": visual_rollout_useful,
                "success_label_does_not_prove_forward_inverse_consistency": True,
                "raw_credentials_written_to_artifacts": False,
                "secret_hashes_written_to_artifacts": False,
            },
        }
    if success_label_command_result is not None:
        success_labels["command_result"] = success_label_command_result
    success_label_generated = bool(success_labels.get("wam_success_label_from_generated_video"))

    consistency_request = {
        "schema_version": "wam_episode_consistency_request.v1",
        "generated_at": generated,
        "status": "ready_for_external_episode_scorer"
        if rollouts and visual_rollout_useful
        else "blocked_generated_rollout_visual_quality"
        if rollouts
        else "blocked_missing_generated_rollout",
        "source_mujoco_endpoint_eval_job_dir": str(input_dir),
        "generated_rollout_results": str(output_dir / "wam_generated_rollout_results.json"),
        "generated_rollout_visual_smoke": str(output_dir / "wam_generated_rollout_visual_smoke.json"),
        "generated_rollout_visual_smoke_status": visual_smoke_status,
        "generated_rollout_visually_useful_for_success_review": visual_rollout_useful,
        "rollouts": rollouts,
        "task_prompts": rollout_input_manifest["task_prompts"],
        "source_trace_paths": {
            "normalized_policy_action_trace_jsonl": str(
                input_dir / "normalized_policy_action_trace.jsonl"
            ),
            "g1_mujoco_locomotion_trace_jsonl": str(
                input_dir / "g1_mujoco_locomotion_trace.jsonl"
            ),
            "normalized_attempt_trace": str(input_dir / "normalized_attempt_trace.json"),
            "eval_ready_task_grounding": eval_ready_grounding_artifacts.get(
                "eval_ready_task_grounding"
            ),
            "robot_fk_projected_skeleton_trace_jsonl": eval_ready_grounding_artifacts.get(
                "robot_fk_projected_skeleton_trace"
            ),
            "handle_proxy_state_check": eval_ready_grounding_artifacts.get(
                "handle_proxy_state_check"
            ),
        },
        "trace_summary": {
            "action_row_count": len(action_rows),
            "locomotion_row_count": len(locomotion_rows),
            "action_type_counts": _action_counts(action_rows),
        },
        "expected_output_path": str(output_dir / WAM_CONSISTENCY_COMMAND_OUTPUT),
        "consistency_label_contract": {
            "required_top_level_keys": ["rollout_checks"],
            "rollout_check_required_keys": [
                "rollout_id",
                "forward_consistent",
                "inverse_consistent",
                "confidence",
                "rationale",
            ],
        },
        "claim_boundary": {
            "scorer_is_separate_from_wam_execution_and_evaluator": True,
            "scorer_input_is_generated_video_and_trace_context_not_physical_robot": True,
            "consistency_label_does_not_prove_task_success": True,
            "consistency_label_does_not_prove_generated_world_rank_fidelity": True,
            "raw_credentials_written_to_artifacts": False,
        },
    }
    write_json(output_dir / "wam_episode_consistency_request.json", consistency_request)

    configured_consistency_command = _string(
        wam_consistency_command or os.getenv(WAM_CONSISTENCY_COMMAND_ENV)
    )
    consistency_blockers: list[str] = []
    consistency_command_result: dict[str, Any] | None = None
    consistency_command_payload: dict[str, Any] = {}
    if not rollouts:
        consistency_blockers = list(generated_blockers)
    elif not visual_rollout_useful:
        consistency_blockers = list(visual_quality_blockers)
    elif allow_wam_consistency_scoring or configured_consistency_command:
        if not _env_truthy(WAM_CONSISTENCY_GATE_ENV):
            consistency_blockers.append(f"missing_env_{WAM_CONSISTENCY_GATE_ENV}")
        if not allow_wam_consistency_scoring:
            consistency_blockers.append("missing_cli_allow_wam_consistency_scoring")
        if not configured_consistency_command:
            consistency_blockers.append("missing_wam_episode_consistency_command")
        if not consistency_blockers:
            consistency_command_payload, consistency_command_result = _run_wam_consistency_command(
                command=configured_consistency_command,
                input_path=output_dir / "wam_episode_consistency_request.json",
                output_path=output_dir / WAM_CONSISTENCY_COMMAND_OUTPUT,
                timeout_seconds=timeout_seconds,
            )
            if consistency_command_result.get("status") != "completed":
                consistency_blockers.extend(
                    _string_list(consistency_command_result.get("blockers"))
                    or ["wam_episode_consistency_command_blocked"]
                )
    else:
        consistency_blockers = ["requires_external_wam_episode_consistency_scorer"]

    if consistency_command_payload and not consistency_blockers:
        consistency = _normalize_wam_episode_consistency(
            command_payload=consistency_command_payload,
            rollouts=rollouts,
            generated_at=generated,
            action_conditioned_video_rollout_generated=learned_wam_model_output_available,
            action_conditioned_video_rollout_available=learned_wam_model_output_available,
            provider_output_replay_used=provider_output_replay_used,
            success_label_generated=success_label_generated,
            visual_smoke_status=visual_smoke_status,
            visual_rollout_useful=visual_rollout_useful,
            command_result=consistency_command_result,
        )
    else:
        consistency = _unscored_wam_episode_consistency(
            generated_at=generated,
            rollouts=rollouts,
            action_conditioned_video_rollout_generated=learned_wam_model_output_available,
            action_conditioned_video_rollout_available=learned_wam_model_output_available,
            provider_output_replay_used=provider_output_replay_used,
            success_label_generated=success_label_generated,
            visual_smoke_status=visual_smoke_status,
            visual_rollout_useful=visual_rollout_useful,
            blockers=consistency_blockers,
            blocked_reason=blocked_reason,
        )
        if consistency_command_result is not None:
            consistency["command_result"] = consistency_command_result
    write_json(output_dir / "wam_consistency_checks.json", consistency)
    forward_inverse_consistency_proven = bool(
        consistency.get("forward_inverse_consistency_proven")
    )
    forward_inverse_scorer_ran = bool(consistency.get("external_episode_consistency_scorer_ran"))

    write_json(output_dir / "wam_success_labels.json", success_labels)
    generated_failure_labels = _generated_rollout_failure_labels(
        rollouts=rollouts,
        success_labels=success_labels,
        visual_smoke=visual_smoke,
        visual_rollout_useful=visual_rollout_useful,
        generated_at=generated,
        output_dir=output_dir,
        blockers=visual_quality_blockers
        if rollouts and not visual_rollout_useful
        else success_label_blockers,
    )
    write_json(output_dir / "failure_labels.json", generated_failure_labels)

    scored_labels = [
        row for row in success_labels.get("labels", []) or [] if isinstance(row, Mapping)
    ]
    success_count = sum(1 for row in scored_labels if row.get("success") is True)
    failure_count = sum(1 for row in scored_labels if row.get("success") is False)
    uncertain_count = max(0, len(scored_labels) - success_count - failure_count)

    scorecard = {
        "schema_version": "wam_policy_scorecard.v1",
        "generated_at": generated,
        "status": "completed"
        if success_label_generated
        else "blocked"
        if not rollouts or not visual_rollout_useful
        else "requires_review",
        "policy_count": len({row.get("policy_id") for row in rollouts if row.get("policy_id")}),
        "generated_rollout_count": len(rollouts),
        "visual_smoke_status": visual_smoke_status,
        "visual_rollout_useful_for_task_success_review": visual_rollout_useful,
        "visual_review_blockers": visual_quality_blockers,
        "review_grade_success_labels": bool(success_label_generated and visual_rollout_useful),
        "review_grade_policy_ranking": bool(success_label_generated and visual_rollout_useful),
        "review_grade_policy_ranking_status": "completed"
        if success_label_generated and visual_rollout_useful
        else "blocked_visual_review_required"
        if rollouts and not visual_rollout_useful
        else "requires_review",
        "success_label_count": len(scored_labels),
        "success_count": success_count,
        "failure_count": failure_count,
        "uncertain_count": uncertain_count,
        "success_rate": round(success_count / len(scored_labels), 6) if scored_labels else None,
        "score_source": "vlm_judge_generated_video"
        if success_label_generated
        else "none_blocked"
        if not rollouts or not visual_rollout_useful
        else "wam_generated_rollouts_pending_review",
        "blockers": [] if success_label_generated else success_label_blockers,
        "eval_ready_task_grounding_used": bool(eval_ready_grounding),
        "selected_task_target": eval_ready_grounding.get("selected_task_target"),
        "camera_calibration_quality_status": _mapping(
            eval_ready_grounding.get("camera_calibration_quality_gate")
        ).get("status"),
        "robot_fk_projection_status": generic_fk_projection_manifest.get("status"),
        "handle_proxy_state": _mapping(
            eval_ready_grounding.get("handle_proxy_state_check")
        ).get("handle_proxy_state"),
        "handle_proxy_on_candidate": bool(
            _mapping(eval_ready_grounding.get("handle_proxy_state_check")).get("on_candidate")
        ),
        "claim_boundary": {
            "score_source_is_generated_video_judge": success_label_generated,
            "visual_smoke_required_for_review_grade_policy_ranking": True,
            "visual_rollout_useful_for_task_success_review": visual_rollout_useful,
            "lightweight_state_proxy_does_not_prove_task_success": True,
            "score_does_not_prove_generated_world_rank_fidelity": True,
            "score_does_not_prove_forward_inverse_consistency": True,
            "score_above_review_grade_requires_consistency_and_calibration_anchors": True,
        },
    }

    reference_trajectory_steps = [
        {"timestamp": row.get("sim_time"), "position": row.get("root_position")}
        for row in locomotion_rows
        if isinstance(row.get("root_position"), Sequence)
        and not isinstance(row.get("root_position"), (str, bytes, bytearray))
    ] or [
        {
            "timestamp": row.get("timestamp"),
            "position": _mapping(row.get("normalized_action")).get("waypoint"),
        }
        for row in action_rows
        if isinstance(_mapping(row.get("normalized_action")).get("waypoint"), Sequence)
    ]
    wam_consistency_score = score_wam_rollout_set_consistency(
        rollouts=rollouts,
        reference={"trajectory": reference_trajectory_steps},
        generated_at=generated,
    )
    write_json(output_dir / "wam_consistency_score.json", wam_consistency_score)
    calibration_anchor_check = evaluate_wam_calibration_anchors(
        _load_json(input_dir / "policy_ranking_ladder_validation.json") or None,
        generated_at=generated,
    )
    write_json(output_dir / "wam_calibration_anchor_check.json", calibration_anchor_check)
    review_grade_evidence_ok = bool(success_label_generated and visual_rollout_useful)
    consistency_measured_ok = bool(
        wam_consistency_score.get("status") == "scored"
        and wam_consistency_score.get("passed") is True
    )
    anchors_ok = bool(
        calibration_anchor_check.get("anchors_present")
        and calibration_anchor_check.get("anchors_passed")
    )
    wam_score_claim_gate = apply_wam_score_claim_gate(
        requested_grade="calibrated_evaluator_grade"
        if review_grade_evidence_ok and consistency_measured_ok and anchors_ok
        else "review_grade"
        if review_grade_evidence_ok
        else "fixture_evaluator_only",
        consistency=wam_consistency_score,
        calibration_anchors=calibration_anchor_check,
        generated_at=generated,
    )
    write_json(output_dir / "wam_score_claim_gate.json", wam_score_claim_gate)
    scorecard["wam_score_claim"] = wam_score_claim_gate
    scorecard["wam_score_claim_grade"] = wam_score_claim_gate["granted_grade"]
    write_json(output_dir / "wam_policy_scorecard.json", scorecard)
    prediction_outcome_correlation_ledger = _build_prediction_outcome_correlation_ledger(
        generated_at=generated,
        input_dir=input_dir,
        output_dir=output_dir,
        rollouts=rollouts,
        success_labels=success_labels,
        scorecard=scorecard,
        consistency=consistency,
        visual_smoke=visual_smoke,
        grounding=eval_ready_grounding,
    )

    policy_requery_endpoint_readiness = _build_policy_requery_endpoint_readiness_manifest(
        generated_at=generated,
        input_dir=input_dir,
        visual_rollout_useful=visual_rollout_useful,
        single_step_policy_requery_frame_useful=single_step_policy_requery_frame_useful,
        visual_smoke_status=visual_smoke_status,
    )
    write_json(
        output_dir / "policy_requery_endpoint_readiness_manifest.json",
        policy_requery_endpoint_readiness,
    )
    wam_policy_requery = _run_wam_policy_requery(
        output_dir=output_dir,
        generated_at=generated,
        input_dir=input_dir,
        rollouts=rollouts,
        visual_rollout_useful=visual_rollout_useful,
        single_step_policy_requery_frame_useful=single_step_policy_requery_frame_useful,
        visual_smoke_status=visual_smoke_status,
        task_prompts=rollout_input_manifest["task_prompts"],
        timeout_seconds=timeout_seconds,
    )
    write_json(output_dir / "wam_policy_requery_manifest.json", wam_policy_requery)
    policy_requery_ran = bool(wam_policy_requery.get("single_step_wam_policy_requery_proven"))
    manipulation_loop_readiness = _build_manipulation_loop_readiness_manifest(
        generated_at=generated,
        input_dir=input_dir,
        attempts=attempts,
        matrix_runs=matrix_runs,
        action_rows=action_rows,
        unitree_controller_proof=unitree_controller_proof,
        openvla_provider_smoke_proof=openvla_provider_smoke_proof,
        rollouts=rollouts,
        visual_rollout_useful=visual_rollout_useful,
        visual_smoke_status=visual_smoke_status,
        wam_policy_requery=wam_policy_requery,
        policy_requery_ran=policy_requery_ran,
    )
    write_json(output_dir / "wam_manipulation_loop_readiness_manifest.json", manipulation_loop_readiness)

    wam_policy_loop_manifest = {
        "schema_version": "wam_policy_loop_manifest.v1",
        "generated_at": generated,
        "status": "completed" if rollout_input_manifest["status"] == "ready_for_model" else "blocked",
        "target_architecture": (
            "policy endpoint observes image/state -> policy emits action chunk -> WAM predicts "
            "next video/world observation -> policy observes generated next observation -> repeat "
            "until task termination -> success judge scores generated rollout"
        ),
        "actual_loop_mode": "offline_wam_generated_observation_policy_requery"
        if policy_requery_ran
        else "offline_action_conditioned_wam_evaluator",
        "source_mujoco_endpoint_eval_job_dir": str(input_dir),
        "policy_endpoint_actions_used_as_conditioning": bool(action_rows),
        "eval_ready_task_grounding_used": bool(eval_ready_grounding),
        "eval_ready_task_grounding_status": eval_ready_grounding.get("status"),
        "robot_fk_projection_available": generic_fk_projection_available,
        "camera_calibration_quality_status": _mapping(
            eval_ready_grounding.get("camera_calibration_quality_gate")
        ).get("status"),
        "handle_proxy_state": _mapping(
            eval_ready_grounding.get("handle_proxy_state_check")
        ).get("handle_proxy_state"),
        "selected_model_candidate": runtime_discovery.get("selected_candidate"),
        "model_command_executed_this_invocation": model_command_executed_this_invocation,
        "fresh_model_command_executed_this_invocation": fresh_model_command_executed_this_invocation,
        "learned_wam_model_ran": learned_wam_model_ran_or_imported,
        "learned_wam_model_ran_this_invocation": learned_wam_model_ran_this_invocation,
        "provider_learned_wam_model_ran": provider_learned_wam_model_ran,
        "provider_generated_video_is_model_output": provider_generated_video_is_model_output,
        "learned_wam_model_ran_or_imported": learned_wam_model_ran_or_imported,
        "learned_wam_model_output_available": learned_wam_model_output_available,
        "provider_output_replay_used": provider_output_replay_used,
        "wam_generated_rollout_status": generated_manifest["status"],
        "wam_generated_rollout_blockers": _string_list(generated_manifest.get("blockers")),
        "action_conditioned_video_rollout_generated": learned_wam_model_output_available,
        "action_conditioned_video_rollout_available": learned_wam_model_output_available,
        "valid_reviewable_generated_video_available": bool(rollouts and visual_rollout_useful),
        "generated_rollout_visual_smoke_status": visual_smoke_status,
        "generated_rollout_visual_quality_blockers": visual_quality_blockers,
        "generated_rollout_visually_useful_for_success_review": visual_rollout_useful,
        "single_step_policy_requery_visual_candidate_status": (
            single_step_policy_requery_visual_candidate.get("status")
        ),
        "single_step_policy_requery_frame_useful": single_step_policy_requery_frame_useful,
        "single_step_wam_policy_requery_visual_candidate": str(
            output_dir / "single_step_wam_policy_requery_visual_candidate.json"
        ),
        "closed_loop_policy_wam_interaction": policy_requery_ran,
        "policy_observes_wam_generated_next_observation": policy_requery_ran,
        "single_step_wam_policy_requery_proven": policy_requery_ran,
        "endpoint_action_returned_for_wam_generated_next_observation": bool(
            wam_policy_requery.get("endpoint_action_returned_for_wam_generated_next_observation")
        ),
        "real_vla_or_unitree_hand_policy_requery_used": bool(
            wam_policy_requery.get("real_vla_or_unitree_hand_policy_endpoint_used")
        ),
        "real_vla_or_unitree_hand_policy_endpoint_used": bool(
            wam_policy_requery.get("real_vla_or_unitree_hand_policy_endpoint_used")
        ),
        "unitree_g1_hand_policy_requery_used": bool(
            wam_policy_requery.get("unitree_g1_hand_policy_endpoint_used")
        ),
        "unitree_g1_hand_policy_endpoint_used": bool(
            wam_policy_requery.get("unitree_g1_hand_policy_endpoint_used")
        ),
        "unitree_g1_hand_policy_output_observed": bool(
            wam_policy_requery.get("unitree_g1_hand_policy_output_observed")
        ),
        "unitree_family_policy_output_observed_for_wam_requery": bool(
            wam_policy_requery.get("unitree_family_policy_output_observed_for_wam_requery")
        ),
        "unitree_hand_policy_requery_used": bool(
            wam_policy_requery.get("unitree_hand_policy_requery_used")
        ),
        "unitree_hand_policy_requery_output_observed": bool(
            wam_policy_requery.get("unitree_hand_policy_requery_output_observed")
        ),
        "policy_requery_provider_replay_used": bool(
            wam_policy_requery.get("policy_requery_provider_replay_used")
        ),
        "fresh_unitree_hand_policy_requery_inference_proven": bool(
            wam_policy_requery.get("fresh_unitree_hand_policy_requery_inference_proven")
        ),
        "policy_requery_provider_replay_is_not_fresh_policy_observation": bool(
            wam_policy_requery.get("policy_requery_provider_replay_is_not_fresh_policy_observation")
        ),
        "policy_requery_policy_id": wam_policy_requery.get("policy_requery_policy_id"),
        "g1_robot_policy_selection_contract": "unitree_native_policy_required_for_g1_claims",
        "g1_robot_policy_selected_family": manipulation_loop_readiness.get(
            "g1_robot_policy_selected_family"
        ),
        "openvla_selected_as_g1_robot_policy": False,
        "wam_rollout_selected_as_g1_robot_policy": False,
        "unitree_hand_manipulation_policy_used": bool(
            manipulation_loop_readiness.get("unitree_hand_manipulation_policy_used")
        ),
        "unitree_hand_policy_required_for_g1_manipulation": True,
        "full_closed_loop_episode_proven": False,
        "closed_loop_manipulation_policy_wam_interaction_ready": bool(
            manipulation_loop_readiness.get("closed_loop_manipulation_policy_wam_interaction_ready")
        ),
        "wam_manipulation_loop_readiness_status": manipulation_loop_readiness.get("status"),
        "wam_manipulation_loop_readiness_manifest": str(
            output_dir / "wam_manipulation_loop_readiness_manifest.json"
        ),
        "manipulation_loop_blockers": manipulation_loop_readiness.get("blockers", []),
        "wam_called_between_policy_action_chunks": policy_requery_ran,
        "every_frame_policy_wam_exchange": False,
        "wam_policy_requery_status": wam_policy_requery.get("status"),
        "wam_policy_requery_blockers": _string_list(wam_policy_requery.get("blockers")),
        "why_policy_requery_not_run": []
        if policy_requery_ran
        else _string_list(wam_policy_requery.get("blockers"))
        or visual_quality_blockers
        or generated_blockers
        or rollout_input_blockers,
        "wam_policy_requery_manifest": str(output_dir / "wam_policy_requery_manifest.json"),
        "policy_requery_endpoint_readiness_manifest": str(
            output_dir / "policy_requery_endpoint_readiness_manifest.json"
        ),
        "policy_requery_endpoint_readiness_status": policy_requery_endpoint_readiness.get(
            "status"
        ),
        "live_policy_requery_endpoint_ready": bool(
            policy_requery_endpoint_readiness.get("live_policy_requery_endpoint_ready")
        ),
        "selected_review_video_count": len(videos),
        "first_person_or_robot_pov_input_video_count": len(wam_input_videos),
        "diagnostic_review_video_count": len(diagnostic_review_videos),
        "third_person_overview_used_as_wam_input": False,
        "wam_input_video_blockers": rollout_input_blockers,
        "success_label_judge_configured": bool(configured_success_label_command),
        "success_label_judge_ran": bool(
            isinstance(success_label_command_result, Mapping)
            and success_label_command_result.get("status") == "completed"
        ),
        "wam_success_label_from_generated_video": success_label_generated,
        "wam_success_label_blockers": success_label_blockers,
        "why_wam_success_label_not_run": []
        if success_label_generated
        else success_label_blockers,
        "failure_diagnosis_status": generated_failure_labels.get("status"),
        "failure_diagnosis_coverage_complete": bool(
            generated_failure_labels.get("failure_diagnosis_coverage_complete")
        ),
        "failure_diagnosis_complete": bool(
            generated_failure_labels.get("failure_diagnosis_complete")
        ),
        "failure_diagnosis_blockers": _string_list(
            generated_failure_labels.get("failure_diagnosis_blockers")
        ),
        "failure_labels": str(output_dir / "failure_labels.json"),
        "forward_inverse_consistency_proven": forward_inverse_consistency_proven,
        "forward_inverse_consistency_blockers": _string_list(consistency.get("blockers")),
        "external_episode_consistency_scorer_blocked_by_visual_quality": bool(
            rollouts and not visual_rollout_useful
        ),
        "external_episode_consistency_scorer_ran": forward_inverse_scorer_ran,
        "external_episode_consistency_scorer_required": not forward_inverse_scorer_ran,
        "external_episode_consistency_scorer_id": consistency.get(
            "external_episode_consistency_scorer_id"
        ),
        "required_to_match_requested_closed_loop": [
            "policy endpoint that consumes generated WAM next observations",
            "reviewable WAM-generated next observation",
            "repeated loop scheduler that alternates policy action chunks with WAM rollout updates",
            "camera/state packet contract shared by policy and WAM",
            "VLM success judge command or provider key configured for final episode scoring",
        ],
        "truth_boundary": (
            "This lane consumes completed MuJoCo endpoint traces and asks a WAM adapter to generate "
            "or replay action-conditioned rollouts. A completed policy requery proves only a single "
            "WAM-generated-observation policy call, not a full repeated task loop."
        ),
    }
    write_json(output_dir / "wam_policy_loop_manifest.json", wam_policy_loop_manifest)

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
            "g1_projected_skeleton_trace_jsonl": str(g1_projected_skeleton_trace_path)
            if g1_projected_skeleton_trace_path.is_file()
            else None,
            "g1_projected_skeleton_manifest": str(
                input_dir / "g1_projected_skeleton_manifest.json"
            )
            if (input_dir / "g1_projected_skeleton_manifest.json").is_file()
            else None,
            "review_video_selection_manifest": str(
                input_dir / "review_video_selection_manifest.json"
            ),
            "eval_ready_task_grounding": eval_ready_grounding_artifacts.get(
                "eval_ready_task_grounding"
            ),
            "robot_fk_projection_manifest": eval_ready_grounding_artifacts.get(
                "robot_fk_projection_manifest"
            ),
            "robot_fk_projected_skeleton_trace_jsonl": eval_ready_grounding_artifacts.get(
                "robot_fk_projected_skeleton_trace"
            ),
            "camera_calibration_quality_gate": eval_ready_grounding_artifacts.get(
                "camera_calibration_quality_gate"
            ),
            "handle_proxy_state_check": eval_ready_grounding_artifacts.get(
                "handle_proxy_state_check"
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
                "wam_generated_rollout_visual_smoke",
                "wam_success_label_request",
                "wam_episode_consistency_request",
                "wam_consistency_checks",
                "wam_success_labels",
                "failure_labels",
                "wam_policy_scorecard",
                "wam_prediction_outcome_correlation_ledger",
                "wam_policy_requery_manifest",
                "policy_requery_endpoint_readiness_manifest",
                "single_step_wam_policy_requery_visual_candidate",
                "wam_policy_loop_manifest",
                "wam_manipulation_loop_readiness_manifest",
                "wam_evaluator_truth_boundary",
                "policy_model_truth_boundary",
                "policy_model_endpoint_readiness_manifest",
                "policy_model_endpoint_creation_plan",
                "policy_model_endpoint_probe_results",
                "openvla_provider_smoke_proof",
                "unitree_unifolm_provider_smoke_proof",
                "source_unitree_controller_proof",
                "policy_cloud_gpu_setup_manifest",
                "local_model_source_tree_discovery",
            ]
        },
    }
    write_json(output_dir / "wam_evaluator_trace_binding.json", trace_binding)

    real_model_endpoint_ready = bool(
        endpoint_creation_plan.get("can_create_real_model_endpoint_now")
    )
    model_http_wrapper_ready = bool(
        endpoint_creation_plan.get("can_create_http_wrapper_for_configured_commands_now")
    )
    success_label_judge_configured = bool(configured_success_label_command)
    success_label_judge_ran = bool(
        isinstance(success_label_command_result, Mapping)
        and success_label_command_result.get("status") == "completed"
    )
    model_endpoint_probe_passed = bool(
        endpoint_probe_results.get("can_claim_real_model_endpoint_after_probe")
    )
    truth_boundary = {
        "schema_version": "wam_evaluator_truth_boundary.v1",
        "generated_at": generated,
        "status": "completed",
        "mujoco_source_job": str(input_dir),
        "mujoco_evidence_is_simulator_only": True,
        "eval_ready_task_grounding_used": bool(eval_ready_grounding),
        "eval_ready_task_grounding_status": eval_ready_grounding.get("status"),
        "eval_ready_task_grounding_learned_rollout_request_ready": _mapping(
            eval_ready_grounding.get("readiness")
        ).get("learned_rollout_request_ready"),
        "selected_task_target": eval_ready_grounding.get("selected_task_target"),
        "camera_calibration_quality_gate": eval_ready_grounding.get(
            "camera_calibration_quality_gate"
        ),
        "robot_fk_projection_available": generic_fk_projection_available,
        "handle_proxy_state_check": eval_ready_grounding.get("handle_proxy_state_check"),
        "prediction_outcome_correlation_status": prediction_outcome_correlation_ledger.get(
            "status"
        ),
        "http_endpoint_wrapper_available": True,
        "model_http_wrapper_ready": model_http_wrapper_ready,
        "real_model_endpoint_ready": real_model_endpoint_ready,
        "model_endpoint_command_probe_passed": model_endpoint_probe_passed,
        "real_model_endpoint_probe_claim_ready": model_endpoint_probe_passed,
        "policy_model_endpoint_probe_results": str(
            output_dir / "policy_model_endpoint_probe_results.json"
        ),
        "real_model_endpoint_claim_blocked": not (
            real_model_endpoint_ready and model_endpoint_probe_passed
        ),
        "learned_wam_model_ran": learned_wam_model_ran_or_imported,
        "learned_wam_model_ran_this_invocation": learned_wam_model_ran_this_invocation,
        "model_command_executed_this_invocation": model_command_executed_this_invocation,
        "fresh_model_command_executed_this_invocation": fresh_model_command_executed_this_invocation,
        "provider_learned_wam_model_ran": provider_learned_wam_model_ran,
        "provider_generated_video_is_model_output": provider_generated_video_is_model_output,
        "learned_wam_model_ran_or_imported": learned_wam_model_ran_or_imported,
        "openvla_provider_smoke_proof": openvla_provider_smoke_proof,
        "openvla_provider_smoke_model_executed": bool(
            openvla_provider_smoke_proof.get("openvla_model_executed")
        ),
        "openvla_provider_smoke_job_dir": openvla_provider_smoke_proof.get("job_dir"),
        "openvla_policy_action_from_provider_smoke": openvla_provider_smoke_proof.get(
            "action"
        ),
        "openvla_provider_smoke_imported": openvla_provider_smoke_imported,
        "openvla_policy_action_command_imported": openvla_provider_smoke_imported,
        "unitree_unifolm_provider_smoke_proof": unitree_unifolm_provider_smoke_proof,
        "unitree_unifolm_provider_smoke_model_executed": bool(
            unitree_unifolm_provider_smoke_proof.get("unitree_unifolm_model_executed")
        ),
        "unitree_unifolm_provider_smoke_job_dir": (
            unitree_unifolm_provider_smoke_proof.get("job_dir")
        ),
        "unitree_unifolm_policy_action_from_provider_smoke": (
            unitree_unifolm_provider_smoke_proof.get("action")
        ),
        "source_unitree_controller_proof": unitree_controller_proof,
        "unitree_locomotion_policy_ran": unitree_locomotion_policy_ran,
        "unitree_locomotion_policy_used": bool(
            unitree_controller_proof.get("unitree_locomotion_policy_used")
        ),
        "unitree_locomotion_policy_kind": unitree_controller_proof.get(
            "unitree_locomotion_policy_kind"
        ),
        "realistic_navigation_policy_used": bool(
            unitree_controller_proof.get("realistic_navigation_policy_used")
        ),
        "realistic_navigation_policy_used_for_endpoint_rollouts": bool(
            unitree_controller_proof.get("realistic_navigation_policy_used_for_endpoint_rollouts")
        ),
        "freejoint_proxy_used": bool(unitree_controller_proof.get("freejoint_proxy_used")),
        "freejoint_proxy_used_for_endpoint_rollouts": bool(
            unitree_controller_proof.get("freejoint_proxy_used_for_endpoint_rollouts")
        ),
        "official_unitree_controller_used": bool(
            unitree_controller_proof.get("official_unitree_controller_used")
        ),
        "official_policy_execution_proven": bool(
            unitree_controller_proof.get("official_policy_execution_proven")
        ),
        "balanced_walking_controller_proven": bool(
            unitree_controller_proof.get("balanced_walking_controller_proven")
        ),
        "learned_wam_model_output_available": learned_wam_model_output_available,
        "provider_output_replay_used": provider_output_replay_used,
        "oscar_cosmos_openvla_unitree_model_ran": oscar_cosmos_openvla_unitree_model_ran,
        "wam_rollout_model_ran": learned_wam_model_ran_or_imported,
        "policy_action_model_command_ran": policy_action_model_command_ran,
        "policy_action_model_provider_smoke_imported": policy_action_model_provider_smoke_imported,
        "openvla_policy_action_command_ran": openvla_policy_action_command_ran,
        "unitree_policy_action_command_ran": unitree_policy_action_command_ran,
        "unitree_unifolm_policy_action_command_ran": unitree_policy_action_command_ran,
        "policy_action_model_execution_scope": (
            unitree_unifolm_provider_smoke_proof.get("model_execution_scope")
            if unitree_policy_action_command_ran
            else openvla_provider_smoke_proof.get("model_execution_scope")
        ),
        "endpoint_closed_loop_policy_proven": policy_requery_ran,
        "endpoint_closed_loop_policy_requery_proven": policy_requery_ran,
        "unitree_g1_hand_policy_output_observed": bool(
            wam_policy_requery.get("unitree_g1_hand_policy_output_observed")
        ),
        "unitree_g1_hand_policy_endpoint_used": bool(
            wam_policy_requery.get("unitree_g1_hand_policy_endpoint_used")
        ),
        "fresh_unitree_hand_policy_requery_inference_proven": bool(
            wam_policy_requery.get("fresh_unitree_hand_policy_requery_inference_proven")
        ),
        "policy_requery_provider_replay_used": bool(
            wam_policy_requery.get("policy_requery_provider_replay_used")
        ),
        "policy_requery_provider_replay_is_not_fresh_policy_observation": bool(
            wam_policy_requery.get("policy_requery_provider_replay_is_not_fresh_policy_observation")
        ),
        "full_closed_loop_episode_proven": False,
        "wam_policy_requery_status": wam_policy_requery.get("status"),
        "policy_requery_endpoint_readiness_status": policy_requery_endpoint_readiness.get(
            "status"
        ),
        "live_policy_requery_endpoint_ready": bool(
            policy_requery_endpoint_readiness.get("live_policy_requery_endpoint_ready")
        ),
        "g1_robot_policy_selection_contract": "unitree_native_policy_required_for_g1_claims",
        "g1_robot_policy_selected_family": manipulation_loop_readiness.get(
            "g1_robot_policy_selected_family"
        ),
        "openvla_selected_as_g1_robot_policy": False,
        "wam_rollout_selected_as_g1_robot_policy": False,
        "unitree_hand_manipulation_policy_used": bool(
            manipulation_loop_readiness.get("unitree_hand_manipulation_policy_used")
        ),
        "unitree_hand_policy_required_for_g1_manipulation": True,
        "closed_loop_manipulation_policy_wam_interaction_ready": bool(
            manipulation_loop_readiness.get("closed_loop_manipulation_policy_wam_interaction_ready")
        ),
        "wam_manipulation_loop_readiness_status": manipulation_loop_readiness.get("status"),
        "wam_manipulation_loop_readiness_manifest": str(
            output_dir / "wam_manipulation_loop_readiness_manifest.json"
        ),
        "manipulation_loop_blockers": manipulation_loop_readiness.get("blockers", []),
        "unitree_g1_dexterous_manipulation_proven": False,
        "selected_model_candidate": runtime_discovery.get("selected_candidate"),
        "wam_backend_strategy_manifest": str(output_dir / "wam_backend_strategy_manifest.json"),
        "preferred_configured_learned_wam_backend_candidate": backend_strategy_manifest.get(
            "preferred_configured_learned_wam_backend_candidate"
        ),
        "preferred_configured_backend_is_not_permanent_dependency": bool(
            backend_strategy_manifest.get(
                "preferred_configured_backend_is_not_permanent_dependency"
            )
        ),
        "cosmos3_preference_is_not_universal_grading_proof": bool(
            _mapping(backend_strategy_manifest.get("claim_boundary")).get(
                "cosmos3_preference_does_not_prove_universal_all_task_grading"
            )
        ),
        "model_choice_does_not_prove_generated_world_rank_fidelity": True,
        "action_conditioned_video_rollout_generated": learned_wam_model_output_available,
        "action_conditioned_video_rollout_available": learned_wam_model_output_available,
        "valid_reviewable_generated_video_available": bool(rollouts and visual_rollout_useful),
        "wam_success_label_from_generated_video": success_label_generated,
        "wam_success_label_judge_configured": success_label_judge_configured,
        "wam_success_label_judge_ran": success_label_judge_ran,
        "wam_success_label_requires_generated_rollout": True,
        "failure_diagnosis_status": generated_failure_labels.get("status"),
        "failure_diagnosis_coverage_complete": bool(
            generated_failure_labels.get("failure_diagnosis_coverage_complete")
        ),
        "failure_diagnosis_complete": bool(
            generated_failure_labels.get("failure_diagnosis_complete")
        ),
        "failure_diagnosis_blockers": _string_list(
            generated_failure_labels.get("failure_diagnosis_blockers")
        ),
        "failure_labels": str(output_dir / "failure_labels.json"),
        "generated_rollout_visual_smoke_status": visual_smoke_status,
        "generated_rollout_visually_useful_for_success_review": visual_rollout_useful,
        "single_step_policy_requery_frame_useful": single_step_policy_requery_frame_useful,
        "forward_inverse_consistency_proven": forward_inverse_consistency_proven,
        "external_episode_consistency_scorer_blocked_by_visual_quality": bool(
            rollouts and not visual_rollout_useful
        ),
        "external_episode_consistency_scorer_ran": forward_inverse_scorer_ran,
        "external_episode_consistency_scorer_required": not forward_inverse_scorer_ran,
        "external_episode_consistency_scorer_id": consistency.get(
            "external_episode_consistency_scorer_id"
        ),
        "fixture_policy_used_as_wam_model": False,
        "generated_outputs_are_raw_capture_evidence": False,
        "wam_rollout_is_robot_policy": False,
        "cosmos3_wam_never_auto_runs_without_explicit_adapter_and_gates": True,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "official_unitree_controller_proven": bool(
            unitree_controller_proof.get("official_unitree_controller_proven")
        ),
        "blockers": visual_quality_blockers
        if visual_quality_blockers
        else rollout_input_blockers
        if rollout_input_blockers
        else generated_blockers
        if generated_blockers
        else success_label_blockers,
        "what_is_needed_to_make_false_flags_true": {
            "real_model_endpoint_ready": endpoint_creation_plan.get(
                "minimum_user_supplied_inputs", []
            ),
            "action_conditioned_video_rollout_generated": [
                "real_model_endpoint_ready",
                "--allow-wam-model-run",
                f"{LOCAL_MODEL_GATE_ENV}=true",
                "model command returns rollout JSON with existing generated_video_path values",
            ],
            "wam_success_label_from_generated_video": [
                "action_conditioned_video_rollout_generated",
                "configured WAM success label command or human/VLM review adapter",
                f"{WAM_SUCCESS_LABEL_GATE_ENV}=true when automated labeling is used",
            ],
            "forward_inverse_consistency_proven": consistency[
                "what_is_needed_to_make_forward_inverse_consistency_true"
            ],
            "closed_loop_manipulation_policy": [
                "run a Unitree LeRobot or Unitree UnifoLM hand/manipulation policy through the HTTP endpoint",
                "produce a visually useful WAM-generated next observation",
                "decode model actions into Blueprint manipulation_contact or hand/arm commands",
                "alternate policy action chunks with WAM-generated next observations",
                "score generated and/or executed videos with a configured VLM judge",
            ],
        },
        "why_cannot_just_create_endpoints": endpoint_creation_plan.get(
            "why_cannot_just_create_missing_model_endpoints", []
        ),
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
        "wam_success_label_status": success_labels["status"],
        "http_endpoint_wrapper_available": True,
        "model_http_wrapper_ready": model_http_wrapper_ready,
        "real_model_endpoint_ready": real_model_endpoint_ready,
        "model_endpoint_command_probe_passed": model_endpoint_probe_passed,
        "real_model_endpoint_probe_claim_ready": model_endpoint_probe_passed,
        "real_model_endpoint_claim_blocked": not (
            real_model_endpoint_ready and model_endpoint_probe_passed
        ),
        "learned_wam_model_ran": learned_wam_model_ran_or_imported,
        "learned_wam_model_ran_this_invocation": learned_wam_model_ran_this_invocation,
        "model_command_executed_this_invocation": model_command_executed_this_invocation,
        "fresh_model_command_executed_this_invocation": fresh_model_command_executed_this_invocation,
        "provider_learned_wam_model_ran": provider_learned_wam_model_ran,
        "provider_generated_video_is_model_output": provider_generated_video_is_model_output,
        "learned_wam_model_ran_or_imported": learned_wam_model_ran_or_imported,
        "learned_wam_model_output_available": learned_wam_model_output_available,
        "provider_output_replay_used": provider_output_replay_used,
        "oscar_cosmos_openvla_unitree_model_ran": oscar_cosmos_openvla_unitree_model_ran,
        "wam_rollout_model_ran": learned_wam_model_ran_or_imported,
        "policy_action_model_command_ran": policy_action_model_command_ran,
        "policy_action_model_provider_smoke_imported": policy_action_model_provider_smoke_imported,
        "openvla_policy_action_command_ran": openvla_policy_action_command_ran,
        "openvla_policy_action_command_imported": openvla_provider_smoke_imported,
        "unitree_policy_action_command_ran": unitree_policy_action_command_ran,
        "unitree_unifolm_policy_action_command_ran": unitree_policy_action_command_ran,
        "unitree_locomotion_policy_ran": unitree_locomotion_policy_ran,
        "unitree_locomotion_policy_used": bool(
            unitree_controller_proof.get("unitree_locomotion_policy_used")
        ),
        "unitree_locomotion_policy_kind": unitree_controller_proof.get(
            "unitree_locomotion_policy_kind"
        ),
        "realistic_navigation_policy_used": bool(
            unitree_controller_proof.get("realistic_navigation_policy_used")
        ),
        "realistic_navigation_policy_used_for_endpoint_rollouts": bool(
            unitree_controller_proof.get("realistic_navigation_policy_used_for_endpoint_rollouts")
        ),
        "freejoint_proxy_used": bool(unitree_controller_proof.get("freejoint_proxy_used")),
        "freejoint_proxy_used_for_endpoint_rollouts": bool(
            unitree_controller_proof.get("freejoint_proxy_used_for_endpoint_rollouts")
        ),
        "official_unitree_controller_used": bool(
            unitree_controller_proof.get("official_unitree_controller_used")
        ),
        "official_unitree_controller_proven": bool(
            unitree_controller_proof.get("official_unitree_controller_proven")
        ),
        "official_policy_execution_proven": bool(
            unitree_controller_proof.get("official_policy_execution_proven")
        ),
        "balanced_walking_controller_proven": bool(
            unitree_controller_proof.get("balanced_walking_controller_proven")
        ),
        "openvla_provider_smoke_model_executed": bool(
            openvla_provider_smoke_proof.get("openvla_model_executed")
        ),
        "openvla_provider_smoke_job_dir": openvla_provider_smoke_proof.get("job_dir"),
        "unitree_unifolm_provider_smoke_model_executed": bool(
            unitree_unifolm_provider_smoke_proof.get("unitree_unifolm_model_executed")
        ),
        "unitree_unifolm_provider_smoke_job_dir": (
            unitree_unifolm_provider_smoke_proof.get("job_dir")
        ),
        "endpoint_closed_loop_policy_proven": policy_requery_ran,
        "endpoint_closed_loop_policy_requery_proven": policy_requery_ran,
        "unitree_g1_hand_policy_output_observed": bool(
            wam_policy_requery.get("unitree_g1_hand_policy_output_observed")
        ),
        "unitree_g1_hand_policy_endpoint_used": bool(
            wam_policy_requery.get("unitree_g1_hand_policy_endpoint_used")
        ),
        "fresh_unitree_hand_policy_requery_inference_proven": bool(
            wam_policy_requery.get("fresh_unitree_hand_policy_requery_inference_proven")
        ),
        "policy_requery_provider_replay_used": bool(
            wam_policy_requery.get("policy_requery_provider_replay_used")
        ),
        "policy_requery_provider_replay_is_not_fresh_policy_observation": bool(
            wam_policy_requery.get("policy_requery_provider_replay_is_not_fresh_policy_observation")
        ),
        "full_closed_loop_episode_proven": False,
        "wam_policy_requery_status": wam_policy_requery.get("status"),
        "policy_requery_endpoint_readiness_status": policy_requery_endpoint_readiness.get(
            "status"
        ),
        "live_policy_requery_endpoint_ready": bool(
            policy_requery_endpoint_readiness.get("live_policy_requery_endpoint_ready")
        ),
        "g1_robot_policy_selection_contract": "unitree_native_policy_required_for_g1_claims",
        "g1_robot_policy_selected_family": manipulation_loop_readiness.get(
            "g1_robot_policy_selected_family"
        ),
        "openvla_selected_as_g1_robot_policy": False,
        "wam_rollout_selected_as_g1_robot_policy": False,
        "unitree_hand_manipulation_policy_used": bool(
            manipulation_loop_readiness.get("unitree_hand_manipulation_policy_used")
        ),
        "unitree_hand_policy_required_for_g1_manipulation": True,
        "closed_loop_manipulation_policy_wam_interaction_ready": bool(
            manipulation_loop_readiness.get("closed_loop_manipulation_policy_wam_interaction_ready")
        ),
        "wam_manipulation_loop_readiness_status": manipulation_loop_readiness.get("status"),
        "manipulation_loop_blockers": manipulation_loop_readiness.get("blockers", []),
        "unitree_g1_dexterous_manipulation_proven": False,
        "selected_model_candidate": runtime_discovery.get("selected_candidate"),
        "preferred_configured_learned_wam_backend_candidate": backend_strategy_manifest.get(
            "preferred_configured_learned_wam_backend_candidate"
        ),
        "preferred_configured_backend_is_not_permanent_dependency": bool(
            backend_strategy_manifest.get(
                "preferred_configured_backend_is_not_permanent_dependency"
            )
        ),
        "cosmos3_wam_never_auto_runs_without_explicit_adapter_and_gates": True,
        "cosmos3_preference_is_not_universal_grading_proof": bool(
            _mapping(backend_strategy_manifest.get("claim_boundary")).get(
                "cosmos3_preference_does_not_prove_universal_all_task_grading"
            )
        ),
        "model_choice_does_not_prove_generated_world_rank_fidelity": True,
        "wam_success_label_from_generated_video": success_label_generated,
        "wam_success_label_judge_configured": success_label_judge_configured,
        "wam_success_label_judge_ran": success_label_judge_ran,
        "failure_diagnosis_status": generated_failure_labels.get("status"),
        "failure_diagnosis_coverage_complete": bool(
            generated_failure_labels.get("failure_diagnosis_coverage_complete")
        ),
        "failure_diagnosis_complete": bool(
            generated_failure_labels.get("failure_diagnosis_complete")
        ),
        "failure_diagnosis_blockers": _string_list(
            generated_failure_labels.get("failure_diagnosis_blockers")
        ),
        "wam_generated_rollout_visual_smoke_status": visual_smoke_status,
        "generated_rollout_visually_useful_for_success_review": visual_rollout_useful,
        "single_step_policy_requery_frame_useful": single_step_policy_requery_frame_useful,
        "forward_inverse_consistency_proven": forward_inverse_consistency_proven,
        "external_episode_consistency_scorer_blocked_by_visual_quality": bool(
            rollouts and not visual_rollout_useful
        ),
        "external_episode_consistency_scorer_ran": forward_inverse_scorer_ran,
        "external_episode_consistency_scorer_required": not forward_inverse_scorer_ran,
        "external_episode_consistency_scorer_id": consistency.get(
            "external_episode_consistency_scorer_id"
        ),
        "wam_score_claim_grade": wam_score_claim_gate["granted_grade"],
        "wam_score_above_review_grade_allowed": wam_score_claim_gate["max_allowed_grade"]
        == "calibrated_evaluator_grade",
        "wam_score_claim_gate_status": wam_score_claim_gate["status"],
        "wam_consistency_score": wam_consistency_score.get("consistency_score"),
        "wam_calibration_anchors_passed": bool(
            calibration_anchor_check.get("anchors_passed")
        ),
        "blockers": visual_quality_blockers
        if visual_quality_blockers
        else generated_blockers
        if generated_blockers
        else success_label_blockers,
        "artifact_paths": {
            "wam_model_runtime_discovery": str(output_dir / "wam_model_runtime_discovery.json"),
            "wam_backend_strategy_manifest": str(output_dir / "wam_backend_strategy_manifest.json"),
            "wam_rollout_input_manifest": str(output_dir / "wam_rollout_input_manifest.json"),
            "wam_action_conditioning_manifest": str(output_dir / "wam_action_conditioning_manifest.json"),
            "wam_generated_rollout_manifest": str(output_dir / "wam_generated_rollout_manifest.json"),
            "wam_generated_rollout_results": str(output_dir / "wam_generated_rollout_results.json"),
            "wam_generated_rollout_visual_smoke": str(output_dir / "wam_generated_rollout_visual_smoke.json"),
            "wam_success_label_request": str(output_dir / "wam_success_label_request.json"),
            "wam_episode_consistency_request": str(
                output_dir / "wam_episode_consistency_request.json"
            ),
            "wam_consistency_checks": str(output_dir / "wam_consistency_checks.json"),
            "wam_success_labels": str(output_dir / "wam_success_labels.json"),
            "failure_labels": str(output_dir / "failure_labels.json"),
            "wam_policy_scorecard": str(output_dir / "wam_policy_scorecard.json"),
            "wam_consistency_score": str(output_dir / "wam_consistency_score.json"),
            "wam_calibration_anchor_check": str(
                output_dir / "wam_calibration_anchor_check.json"
            ),
            "wam_score_claim_gate": str(output_dir / "wam_score_claim_gate.json"),
            "wam_prediction_outcome_correlation_ledger": str(
                output_dir / "wam_prediction_outcome_correlation_ledger.json"
            ),
            "wam_policy_requery_manifest": str(output_dir / "wam_policy_requery_manifest.json"),
            "eval_ready_task_grounding": eval_ready_grounding_artifacts.get(
                "eval_ready_task_grounding"
            ),
            "robot_fk_projection_manifest": eval_ready_grounding_artifacts.get(
                "robot_fk_projection_manifest"
            ),
            "robot_fk_projected_skeleton_trace": eval_ready_grounding_artifacts.get(
                "robot_fk_projected_skeleton_trace"
            ),
            "camera_calibration_quality_gate": eval_ready_grounding_artifacts.get(
                "camera_calibration_quality_gate"
            ),
            "handle_proxy_state_check": eval_ready_grounding_artifacts.get(
                "handle_proxy_state_check"
            ),
            "policy_requery_endpoint_readiness_manifest": str(
                output_dir / "policy_requery_endpoint_readiness_manifest.json"
            ),
            "single_step_wam_policy_requery_visual_candidate": str(
                output_dir / "single_step_wam_policy_requery_visual_candidate.json"
            ),
            "wam_policy_loop_manifest": str(output_dir / "wam_policy_loop_manifest.json"),
            "wam_manipulation_loop_readiness_manifest": str(
                output_dir / "wam_manipulation_loop_readiness_manifest.json"
            ),
            "wam_evaluator_trace_binding": str(output_dir / "wam_evaluator_trace_binding.json"),
            "wam_evaluator_truth_boundary": str(output_dir / "wam_evaluator_truth_boundary.json"),
            "policy_model_truth_boundary": str(output_dir / "policy_model_truth_boundary.json"),
            "policy_model_endpoint_readiness_manifest": str(
                output_dir / "policy_model_endpoint_readiness_manifest.json"
            ),
            "policy_model_endpoint_creation_plan": str(
                output_dir / "policy_model_endpoint_creation_plan.json"
            ),
            "policy_model_endpoint_probe_results": str(
                output_dir / "policy_model_endpoint_probe_results.json"
            ),
            "openvla_provider_smoke_proof": str(output_dir / "openvla_provider_smoke_proof.json"),
            "unitree_unifolm_provider_smoke_proof": str(
                output_dir / "unitree_unifolm_provider_smoke_proof.json"
            ),
            "source_unitree_controller_proof": str(
                output_dir / "source_unitree_controller_proof.json"
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
    parser.add_argument("--wam-success-label-command")
    parser.add_argument("--allow-wam-success-labeling", action="store_true")
    parser.add_argument("--wam-consistency-command")
    parser.add_argument("--allow-wam-consistency-scoring", action="store_true")
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
        wam_success_label_command=args.wam_success_label_command,
        allow_wam_success_labeling=args.allow_wam_success_labeling,
        wam_consistency_command=args.wam_consistency_command,
        allow_wam_consistency_scoring=args.allow_wam_consistency_scoring,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "status": summary.get("status"),
                "job_dir": summary.get("job_dir"),
                "wam_generated_rollout_status": summary.get("wam_generated_rollout_status"),
                "wam_success_label_status": summary.get("wam_success_label_status"),
                "wam_success_label_from_generated_video": summary.get(
                    "wam_success_label_from_generated_video"
                ),
                "learned_wam_model_ran": summary.get("learned_wam_model_ran"),
                "blockers": summary.get("blockers"),
            },
            sort_keys=True,
        )
    )
    return 0 if summary.get("status") == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
