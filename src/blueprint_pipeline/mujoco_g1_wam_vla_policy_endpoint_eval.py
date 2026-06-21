"""MuJoCo Unitree G1 WAM/VLA policy-endpoint evaluation lane.

This lane is intentionally simulator-only. It tests policy endpoint discovery,
observation/action contracts, action normalization, MuJoCo execution, contact
traces, and WAM-style evaluator scoring without requiring Isaac Sim, splat/PLY
visuals, cloud GPUs, or physical robot controls.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .g1_controlled_proof_setup import OFFICIAL_UNITREE_G1_POLICY_SOURCES
from .mujoco_g1_simulator_command import (
    DEFAULT_MENAGERIE_REF,
    _asset_source_manifest,
    _resolve_g1_model_root,
    _sha256,
    _write_g1_xml_with_absolute_meshes,
)


LANE_SCHEMA_VERSION = "mujoco_g1_wam_vla_policy_endpoint_eval.v1"
SCENARIO_MATRIX_SCHEMA_VERSION = "mujoco_g1_wam_vla_scenario_eval_matrix.v1"
OBSERVATION_SCHEMA_ID = "blueprint.mujoco_g1_wam_vla.observation_packet.v1"
ACTION_SCHEMA_ID = "blueprint.mujoco_g1_wam_vla.action.v1"
REFERENCE_FIXTURE_POLICY_ID = "reference_fixture_policy"
ROBOT_PROFILE_ID = "unitree_g1_mujoco_menagerie"
DEFAULT_STEPS_PER_EPISODE = 3000
DEFAULT_VIDEO_FRAME_STRIDE_STEPS = 8
DEFAULT_REVIEW_VIDEO_FPS = 60
DEFAULT_EXTEND_TERMINAL_FRAME_FOR_REVIEW = False
DEFAULT_RENDERED_VIDEO_EPISODE_LIMIT = 8
DEFAULT_MAX_CONTACT_TRACE_ROWS = 50000
DEFAULT_CONTACT_OBSERVATION_RECORD_LIMIT = 24
AVAILABLE_VIDEO_CAMERAS = ("third_person", "overhead", "robot_follow")
DEFAULT_VIDEO_CAMERAS = ("third_person",)
CONTROLLER_BACKENDS = ("auto", "freejoint_proxy", "unitree_rl_gym")
DEFAULT_CONTROLLER_BACKEND = "auto"
UNITREE_RL_GYM_SAME_SCENE_BACKEND_ID = "unitree_rl_gym_same_scene_lower_body_policy"
UNITREE_RL_GYM_CONTROLLER_COMMAND_LIMITS = {
    "max_forward_velocity_mps": 0.35,
    "max_reverse_velocity_mps": 0.10,
    "max_lateral_velocity_mps": 0.12,
    "max_yaw_rate_rad_s": 0.45,
}
UNITREE_RL_GYM_LEG_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
EXTRA_G1_POLICY_ONLINE_CANDIDATES = (
    {
        "name": "unitree_lerobot",
        "url": "https://github.com/unitreerobotics/unitree_lerobot",
        "candidate_use": "LeRobot-format Unitree data collection, policy training, and test path.",
    },
    {
        "name": "lerobot_unitree_g1",
        "url": "https://huggingface.co/docs/lerobot/unitree_g1",
        "candidate_use": "Hugging Face LeRobot Unitree G1 hardware/sim support path.",
    },
    {
        "name": "unitree_rl_mjlab",
        "url": "https://github.com/unitreerobotics/unitree_rl_mjlab",
        "candidate_use": "MuJoCo-backed Unitree G1 RL policy research candidate.",
    },
)

ENDPOINT_ENVS = (
    {
        "runtime": "wam",
        "endpoint_env": "WAM_POLICY_ENDPOINT_URL",
        "auth_file_env": "WAM_POLICY_AUTH_TOKEN_FILE",
    },
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

SAFETY_LIMITS = {
    "max_forward_velocity_mps": 0.75,
    "max_lateral_velocity_mps": 0.35,
    "max_yaw_rate_rad_s": 0.9,
    "max_waypoint_distance_m": 1.25,
    "max_episode_seconds": 8.0,
    "fall_root_height_m": 0.45,
    "goal_tolerance_m": 0.35,
    "object_displacement_success_m": 0.015,
    "stop_speed_mps": 0.06,
}

SPAWN_SPECS = [
    {
        "spawn_id": "doorway",
        "label": "Doorway approach",
        "pose_xy_yaw": [-0.22, 0.0, 0.0],
    },
    {
        "spawn_id": "side_aisle",
        "label": "Side aisle",
        "pose_xy_yaw": [-0.25, 0.55, 0.0],
    },
    {
        "spawn_id": "near_task_wall_or_table",
        "label": "Near task wall/table",
        "pose_xy_yaw": [-0.18, -0.32, 0.0],
    },
    {
        "spawn_id": "manipulation_stance",
        "label": "Manipulation stance",
        "pose_xy_yaw": [-0.18, -0.65, 0.0],
    },
    {
        "spawn_id": "blocked_or_occluded",
        "label": "Blocked/occluded route",
        "pose_xy_yaw": [-0.35, 0.92, 0.0],
        "expected_blockers": ["occluded_policy_context_variant"],
    },
]

TASK_SPECS = [
    {
        "task_id": "inspect_target",
        "scenario_id": "mujoco_g1_inspect_target",
        "prompt": "Inspect the target zone and report when it is visible.",
        "target_xy": [0.25, 0.16],
        "expected_action_types": ["inspect_look", "stop"],
    },
    {
        "task_id": "approach_target",
        "scenario_id": "mujoco_g1_approach_target",
        "prompt": "Approach the target marker and stop within the goal tolerance.",
        "target_xy": [0.52, 0.0],
        "expected_action_types": ["waypoint", "base_velocity", "stop"],
    },
    {
        "task_id": "route_around_obstruction",
        "scenario_id": "mujoco_g1_route_around_obstruction",
        "prompt": "Route around the obstruction without contacting it.",
        "target_xy": [0.72, 0.46],
        "route_waypoints": [[0.10, 0.76], [0.44, 0.76], [0.72, 0.46]],
        "expected_action_types": ["waypoint", "base_velocity", "stop"],
    },
    {
        "task_id": "contact_or_push_light_object",
        "scenario_id": "mujoco_g1_contact_or_push_light_object",
        "prompt": "Approach the lightweight object and push it forward slightly.",
        "target_xy": [0.50, -0.65],
        "object_id": "blueprint_light_object",
        "expected_action_types": ["manipulation_contact", "base_velocity", "stop"],
    },
    {
        "task_id": "stop_at_goal_and_report",
        "scenario_id": "mujoco_g1_stop_at_goal_and_report",
        "prompt": "Move to the goal, stop, and report task completion.",
        "target_xy": [0.42, -0.14],
        "expected_action_types": ["waypoint", "stop"],
    },
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_id(value: Any, *, fallback: str = "item") -> str:
    text = str(value or fallback).strip().lower()
    cleaned = "".join(char if char.isalnum() else "_" for char in text)
    return "_".join(part for part in cleaned.split("_") if part) or fallback


def _number(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _utc_timestamp_for_path() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _yaw_quat(yaw: float) -> list[float]:
    half = yaw * 0.5
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _yaw_from_quat(quat: Sequence[float]) -> float:
    w = float(quat[0])
    x = float(quat[1])
    y = float(quat[2])
    z = float(quat[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


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


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "y"}


def _runtime_endpoint_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in ENDPOINT_ENVS:
        endpoint = os.getenv(spec["endpoint_env"], "").strip()
        token_file_raw = os.getenv(spec["auth_file_env"], "").strip()
        token_file = Path(token_file_raw).expanduser() if token_file_raw else None
        rows.append(
            {
                "runtime": spec["runtime"],
                "endpoint_env": spec["endpoint_env"],
                "endpoint_url_configured": bool(endpoint),
                "endpoint_url": endpoint or None,
                "auth_file_env": spec["auth_file_env"],
                "auth_token_file_configured": bool(token_file_raw),
                "auth_token_file_path": str(token_file) if token_file else None,
                "auth_token_file_exists": bool(token_file and token_file.is_file()),
                "auth_token_file_size_bytes": token_file.stat().st_size
                if token_file and token_file.is_file()
                else None,
                "ready_for_endpoint_call": bool(endpoint and token_file and token_file.is_file()),
                "token_value_written_to_artifacts": False,
            }
        )
    return rows


def discover_policy_runtime(*, generated_at: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    endpoint_rows = _runtime_endpoint_rows()
    ready_rows = [row for row in endpoint_rows if row["ready_for_endpoint_call"]]
    missing_reasons: list[str] = []
    if not any(row["endpoint_url_configured"] for row in endpoint_rows):
        missing_reasons.append("blocked_missing_policy_endpoint")
    if any(row["endpoint_url_configured"] for row in endpoint_rows) and not ready_rows:
        missing_reasons.append("blocked_missing_policy_auth_token_file")
    provider_command_envs = [
        "BLUEPRINT_WAM_PROVIDER_COMMAND",
        "BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND",
        "BLUEPRINT_OSCAR_WAM_PROVIDER_COMMAND",
        "BLUEPRINT_COSMOS_WAM_COMMAND",
        "BLUEPRINT_OSCAR_WAM_COMMAND",
        "BLUEPRINT_OPENVLA_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
    ]
    local_runners = [
        {
            "runner_id": "blueprint_run_wam_fixture_evaluator",
            "command": "blueprint-run-wam-fixture-evaluator",
            "available_on_path": bool(shutil.which("blueprint-run-wam-fixture-evaluator")),
            "runtime_kind": "reference_fixture_policy",
        },
        {
            "runner_id": "blueprint_run_policy_autoresearch_mujoco_evaluator",
            "command": "blueprint-run-policy-autoresearch-mujoco-evaluator",
            "available_on_path": bool(
                shutil.which("blueprint-run-policy-autoresearch-mujoco-evaluator")
            ),
            "runtime_kind": "local_mujoco_policy_eval_support",
        },
        {
            "runner_id": "blueprint_g1_endpoint_reference_adapter",
            "command": "blueprint-g1-endpoint-reference-adapter",
            "available_on_path": bool(shutil.which("blueprint-g1-endpoint-reference-adapter")),
            "runtime_kind": "command_policy_reference_heuristic",
        },
        {
            "runner_id": "blueprint_run_oscar_cosmos_wam_evaluator",
            "command": "blueprint-run-oscar-cosmos-wam-evaluator",
            "available_on_path": bool(shutil.which("blueprint-run-oscar-cosmos-wam-evaluator")),
            "runtime_kind": "learned_wam_evaluator_contract",
        },
    ]
    discovery = {
        "schema_version": "policy_endpoint_discovery.v1",
        "generated_at": generated_at,
        "status": "endpoint_ready" if ready_rows else "blocked_missing_policy_endpoint",
        "endpoint_candidates": endpoint_rows,
        "selected_endpoint_runtime": ready_rows[0]["runtime"] if ready_rows else None,
        "selected_endpoint_env": ready_rows[0]["endpoint_env"] if ready_rows else None,
        "selected_auth_file_env": ready_rows[0]["auth_file_env"] if ready_rows else None,
        "blockers": missing_reasons,
        "fixture_reference_policy_fallback_available": True,
        "raw_tokens_written_to_artifacts": False,
    }
    auth_manifest = {
        "schema_version": "policy_endpoint_auth_manifest.v1",
        "generated_at": generated_at,
        "status": "auth_ready" if ready_rows else "blocked_or_not_configured",
        "auth_records": endpoint_rows,
        "file_based_secrets_only": True,
        "raw_token_values_persisted": False,
        "raw_token_hashes_persisted": False,
    }
    runtime_discovery = {
        "schema_version": "wam_vla_runtime_discovery.v1",
        "generated_at": generated_at,
        "status": "endpoint_ready" if ready_rows else "fixture_fallback_ready",
        "endpoint_runtimes": endpoint_rows,
        "local_model_package_runners": local_runners,
        "provider_command_contracts": [
            {
                "env": env_name,
                "configured": bool(os.getenv(env_name)),
                "value_redacted": "<configured>" if os.getenv(env_name) else None,
            }
            for env_name in provider_command_envs
        ],
        "fixture_reference_policy": {
            "policy_id": REFERENCE_FIXTURE_POLICY_ID,
            "available": True,
            "claim_boundary": "reference_fixture_policy_not_real_wam_vla",
        },
    }
    probe_results = {
        "schema_version": "policy_endpoint_probe_results.v1",
        "generated_at": generated_at,
        "status": "configured_for_per_observation_calls"
        if ready_rows
        else "blocked_missing_policy_endpoint",
        "endpoint_preflight_call_performed": False,
        "reason": "endpoint_calls_are_recorded_per_policy_observation"
        if ready_rows
        else "endpoint_url_or_auth_file_missing_fixture_path_will_run",
        "selected_endpoint_runtime": ready_rows[0]["runtime"] if ready_rows else None,
        "blockers": [] if ready_rows else ["blocked_missing_policy_endpoint"],
    }
    return discovery, runtime_discovery, auth_manifest, probe_results


def selected_endpoint(discovery: Mapping[str, Any]) -> dict[str, Any] | None:
    for row in discovery.get("endpoint_candidates", []) or []:
        if isinstance(row, Mapping) and row.get("ready_for_endpoint_call"):
            return dict(row)
    return None


def _derive_health_url(endpoint_url: str) -> str:
    if endpoint_url.endswith("/policy/action"):
        return endpoint_url[: -len("/policy/action")] + "/health"
    return endpoint_url.rstrip("/") + "/health"


def _probe_endpoint_health(
    *, endpoint_row: Mapping[str, Any] | None, timeout_seconds: float
) -> dict[str, Any]:
    if endpoint_row is None:
        return {
            "schema_version": "policy_endpoint_health_probe.v1",
            "status": "blocked",
            "endpoint_health_probe_performed": False,
            "blockers": ["blocked_missing_policy_endpoint"],
            "raw_token_persisted": False,
        }
    endpoint_url = str(endpoint_row.get("endpoint_url") or "")
    if not endpoint_url:
        return {
            "schema_version": "policy_endpoint_health_probe.v1",
            "status": "blocked",
            "endpoint_health_probe_performed": False,
            "blockers": ["blocked_missing_policy_endpoint"],
            "raw_token_persisted": False,
        }
    health_url = _derive_health_url(endpoint_url)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(health_url, timeout=timeout_seconds) as response:
            body = response.read()
        payload = json.loads(body.decode("utf-8"))
        return {
            "schema_version": "policy_endpoint_health_probe.v1",
            "status": "completed",
            "endpoint_health_probe_performed": True,
            "health_url": health_url,
            "http_status": int(getattr(response, "status", 0) or 0),
            "duration_seconds": round(time.monotonic() - started, 6),
            "health_payload_redacted": _redact(payload),
            "raw_token_persisted": False,
            "blockers": [],
        }
    except Exception as exc:
        return {
            "schema_version": "policy_endpoint_health_probe.v1",
            "status": "blocked",
            "endpoint_health_probe_performed": True,
            "health_url": health_url,
            "duration_seconds": round(time.monotonic() - started, 6),
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "raw_token_persisted": False,
            "blockers": ["policy_endpoint_health_probe_failed"],
        }


def build_policy_model_candidate_matrix(*, generated_at: str) -> dict[str, Any]:
    return {
        "schema_version": "policy_model_candidate_matrix.v1",
        "generated_at": generated_at,
        "status": "adapter_boundary_defined",
        "stable_contracts": [
            "oscar_wam",
            "cosmos_wam",
            "openvla_policy",
            "unitree_g1_policy",
            "command_policy",
        ],
        "candidates": [
            {
                "id": "command_policy",
                "runtime_role": "local_command_adapter",
                "default_local_command": "blueprint-g1-endpoint-reference-adapter",
                "available_on_path": bool(shutil.which("blueprint-g1-endpoint-reference-adapter")),
                "checkpoint_required": False,
                "claim_boundary": "heuristic_endpoint_plumbing_not_real_wam_vla",
            },
            {
                "id": "unitree_g1_policy",
                "runtime_role": "realistic_g1_locomotion_or_control_policy",
                "command_env": "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
                "checkpoint_env": "BLUEPRINT_UNITREE_G1_POLICY_CHECKPOINT",
                "configured": bool(
                    os.getenv("BLUEPRINT_UNITREE_G1_POLICY_COMMAND")
                    or os.getenv("BLUEPRINT_REALISTIC_G1_POLICY_COMMAND")
                ),
                "claim_boundary": "requires_controller_grade_runner_execution",
            },
            {
                "id": "openvla_policy",
                "runtime_role": "vla_or_imitation_policy_endpoint_candidate",
                "command_env": "BLUEPRINT_OPENVLA_POLICY_COMMAND",
                "checkpoint_env": "BLUEPRINT_OPENVLA_POLICY_CHECKPOINT",
                "configured": bool(os.getenv("BLUEPRINT_OPENVLA_POLICY_COMMAND")),
                "claim_boundary": "requires_model_endpoint_response_and_action_decoder",
            },
            {
                "id": "oscar_wam",
                "runtime_role": "action_conditioned_world_model_evaluator",
                "command_env": "BLUEPRINT_OSCAR_WAM_COMMAND",
                "checkpoint_env": "BLUEPRINT_OSCAR_WAM_CHECKPOINT",
                "configured": bool(
                    os.getenv("BLUEPRINT_OSCAR_WAM_COMMAND")
                    or os.getenv("BLUEPRINT_OSCAR_WAM_PROVIDER_COMMAND")
                ),
                "claim_boundary": "requires_generated_rollout_artifacts",
            },
            {
                "id": "cosmos_wam",
                "runtime_role": "world_video_rollout_or_review_substrate",
                "command_env": "BLUEPRINT_COSMOS_WAM_COMMAND",
                "checkpoint_env": "BLUEPRINT_COSMOS_WAM_CHECKPOINT",
                "configured": bool(
                    os.getenv("BLUEPRINT_COSMOS_WAM_COMMAND")
                    or os.getenv("BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND")
                ),
                "claim_boundary": "requires_generated_rollout_artifacts",
            },
        ],
    }


def build_policy_model_truth_boundary(*, generated_at: str) -> dict[str, Any]:
    return {
        "schema_version": "policy_model_truth_boundary.v1",
        "generated_at": generated_at,
        "mujoco_evidence_is_simulator_only": True,
        "endpoint_response_is_policy_plumbing_proof_only": True,
        "reference_command_policy_is_not_real_wam_vla": True,
        "real_wam_vla_proof_requires_actual_model_endpoint_response": True,
        "wam_rollouts_are_model_derived_support_artifacts": True,
        "oscar_cosmos_rollout_proven_by_this_lane": False,
        "openvla_policy_proven_by_this_lane": False,
        "unitree_g1_controller_proven_by_this_lane": False,
        "isaac_proof": False,
        "splat_ply_spz_proof": False,
        "physical_robot_readiness_proven": False,
        "deployment_readiness_proven": False,
        "raw_tokens_written_to_artifacts": False,
        "raw_token_hashes_written_to_artifacts": False,
    }


def _unitree_rl_gym_required_files(root: Path) -> dict[str, Path]:
    return {
        "deploy_mujoco_config": root / "deploy" / "deploy_mujoco" / "configs" / "g1.yaml",
        "pretrained_motion_policy": root / "deploy" / "pre_train" / "g1" / "motion.pt",
        "g1_mjcf": root / "resources" / "robots" / "g1_description" / "g1_12dof.xml",
        "g1_scene_mjcf": root / "resources" / "robots" / "g1_description" / "scene.xml",
    }


def _unitree_rl_gym_root_row(*, label: str, root: Path | None) -> dict[str, Any]:
    required = _unitree_rl_gym_required_files(root) if root is not None else {}
    missing = [
        name
        for name, path in required.items()
        if not path.expanduser().resolve().is_file()
    ]
    return {
        "label": label,
        "path": str(root.expanduser().resolve()) if root is not None else None,
        "exists": bool(root and root.expanduser().exists()),
        "required_files_present": bool(root and not missing),
        "required_files": {name: str(path) for name, path in required.items()},
        "missing_required_files": missing,
    }


def _default_unitree_rl_gym_root_candidates() -> list[tuple[str, Path]]:
    repo = _repo_root()
    candidates: list[tuple[str, Path]] = [
        ("workspace_unitree_rl_gym", repo.parent / "unitree_rl_gym"),
        ("workspace_unitree_rl_gym_pascal", repo.parent / "Unitree_RL_Gym"),
    ]
    for path in sorted(
        (repo / "robot_eval_jobs").glob("*/runtime_sources/unitree_rl_gym"),
        reverse=True,
    ):
        candidates.append(("robot_eval_job_runtime_source", path))
    return candidates


def _select_unitree_rl_gym_root(
    *,
    explicit_root: Path | None,
    discovery: Mapping[str, Any] | None = None,
) -> Path | None:
    if explicit_root is not None:
        row = _unitree_rl_gym_root_row(label="explicit", root=Path(explicit_root))
        return Path(row["path"]) if row["required_files_present"] else None
    for row in _mapping(discovery).get("unitree_rl_gym_root_candidates", []):
        if isinstance(row, Mapping) and row.get("required_files_present") and row.get("path"):
            return Path(str(row["path"]))
    for _label, candidate in _default_unitree_rl_gym_root_candidates():
        row = _unitree_rl_gym_root_row(label=_label, root=candidate)
        if row["required_files_present"]:
            return Path(str(row["path"]))
    return None


def discover_realistic_navigation_policy(*, generated_at: str) -> dict[str, Any]:
    candidate_envs = [
        "BLUEPRINT_REALISTIC_G1_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
        "UNITREE_G1_POLICY_COMMAND",
    ]
    root_envs = [
        "BLUEPRINT_UNITREE_G1_POLICY_ROOT",
        "BLUEPRINT_UNITREE_RL_GYM_ROOT",
        "UNITREE_G1_POLICY_ROOT",
    ]
    command_rows = []
    for env_name in candidate_envs:
        value = os.getenv(env_name, "").strip()
        executable = ""
        available = False
        if value:
            try:
                parts = shlex.split(value)
            except ValueError:
                parts = []
            executable = parts[0] if parts else ""
            available = bool(executable and (Path(executable).expanduser().is_file() or shutil.which(executable)))
        command_rows.append(
            {
                "env": env_name,
                "configured": bool(value),
                "available": available,
                "value_redacted": "<configured>" if value else None,
            }
        )
    root_rows = []
    for env_name in root_envs:
        value = os.getenv(env_name, "").strip()
        root = Path(value).expanduser() if value else None
        row = _unitree_rl_gym_root_row(label=env_name, root=root)
        root_rows.append({"env": env_name, "configured": bool(value), **row})
    available = any(row["available"] for row in command_rows)
    workspace_root = _repo_root().parent
    official_source_rows = [
        {
            "name": source.get("name"),
            "url": source.get("url"),
            "recommended_ref": source.get("recommended_ref"),
            "candidate_use": source.get("candidate_use"),
            "relevant_paths": source.get("relevant_paths", []),
        }
        for source in OFFICIAL_UNITREE_G1_POLICY_SOURCES
    ] + [dict(source) for source in EXTRA_G1_POLICY_ONLINE_CANDIDATES]
    local_checkout_rows = []
    for source in official_source_rows:
        name = str(source.get("name") or "").strip()
        if not name:
            continue
        candidate_path = workspace_root / name
        local_checkout_rows.append(
            {
                "name": name,
                "expected_local_path": str(candidate_path),
                "exists": candidate_path.exists(),
            }
        )
    unitree_rl_gym_root_candidates = [
        _unitree_rl_gym_root_row(label=label, root=path)
        for label, path in _default_unitree_rl_gym_root_candidates()
    ]
    root_available = any(row["required_files_present"] for row in root_rows) or any(
        row["required_files_present"] for row in unitree_rl_gym_root_candidates
    )
    blockers = []
    if not available and not root_available:
        blockers.append("blocked_missing_realistic_g1_navigation_policy")
    elif available and not root_available:
        blockers.append("blocked_controller_command_not_integrated_into_same_scene_endpoint_rollouts")
    return {
        "schema_version": "realistic_navigation_policy_discovery.v1",
        "generated_at": generated_at,
        "status": "candidate_available_for_endpoint_controller_selection"
        if available or root_available
        else "blocked_missing_controller_command",
        "pre_execution_discovery_only": True,
        "final_execution_truth_artifact": "controller_truth_boundary.json",
        "execution_truth_fields": "deferred_to_controller_truth_boundary_json",
        "realistic_navigation_policy_used": None,
        "realistic_navigation_policy_used_for_endpoint_rollouts": None,
        "freejoint_proxy_used": None,
        "freejoint_proxy_used_for_endpoint_rollouts": None,
        "official_unitree_controller_used": None,
        "balanced_walking_controller_proven": None,
        "candidate_command_envs": command_rows,
        "candidate_root_envs": root_rows,
        "official_online_candidates": official_source_rows,
        "local_checkout_candidates": local_checkout_rows,
        "unitree_rl_gym_root_candidates": unitree_rl_gym_root_candidates,
        "unitree_policy_root_available": root_available,
        "unitree_policy_root_ready_for_sidecar_execution": root_available,
        "unitree_policy_root_ready_for_same_scene_endpoint_controller": root_available,
        "same_scene_endpoint_controller_can_be_selected_with": "--controller-backend unitree_rl_gym"
        if root_available
        else None,
        "blockers": blockers,
        "next_upgrade_path": (
            "Run the Unitree RL Gym controller sidecar to prove non-fixture G1 locomotion, "
            "then build the task-conditioned bridge that maps Blueprint endpoint actions into "
            "the Unitree controller command stream for the same scene attempts."
        ),
        "claim_boundary": {
            "freejoint_proxy_is_not_realistic_navigation_policy": True,
            "official_unitree_controller_proof_requires_controller_grade_stack": True,
            "unitree_rl_gym_root_discovery_is_not_endpoint_task_control": True,
            "online_source_discovery_is_not_controller_execution_proof": True,
            "physical_robot_readiness_proven": False,
        },
    }


def _run_official_unitree_controller_sidecar(
    *,
    job_dir: Path,
    job_id: str,
    generated_at: str,
    unitree_rl_gym_root: Path | None,
    navigation_discovery: Mapping[str, Any],
    enabled: bool,
    max_steps: int,
    command_xyz: Sequence[float] | None = None,
) -> dict[str, Any]:
    selected_root = _select_unitree_rl_gym_root(
        explicit_root=unitree_rl_gym_root,
        discovery=navigation_discovery,
    )
    sidecar_dir = job_dir / "official_unitree_g1_policy_execution"
    manifest_path = sidecar_dir / "official_unitree_g1_policy_execution_manifest.json"
    if not enabled:
        return {
            "schema_version": "official_unitree_controller_sidecar.v1",
            "generated_at": generated_at,
            "status": "skipped",
            "enabled": False,
            "selected_unitree_rl_gym_root": str(selected_root) if selected_root else None,
            "manifest_path": str(manifest_path),
            "official_unitree_controller_used": False,
            "balanced_walking_controller_proven": False,
            "realistic_navigation_policy_used_for_endpoint_rollouts": False,
            "endpoint_task_control_used": False,
            "command_xyz": list(command_xyz) if command_xyz is not None else None,
            "blockers": ["skipped_official_unitree_controller_sidecar"],
        }
    if selected_root is None:
        return {
            "schema_version": "official_unitree_controller_sidecar.v1",
            "generated_at": generated_at,
            "status": "blocked",
            "enabled": True,
            "selected_unitree_rl_gym_root": None,
            "manifest_path": str(manifest_path),
            "official_unitree_controller_used": False,
            "balanced_walking_controller_proven": False,
            "realistic_navigation_policy_used_for_endpoint_rollouts": False,
            "endpoint_task_control_used": False,
            "command_xyz": list(command_xyz) if command_xyz is not None else None,
            "blockers": ["blocked_missing_unitree_rl_gym_root_or_required_files"],
        }
    try:
        from .unitree_g1_policy_execution import build_unitree_g1_policy_execution

        manifest = build_unitree_g1_policy_execution(
            capture_root=job_dir,
            unitree_rl_gym_root=selected_root,
            job_id=job_id,
            duration_seconds=max(0.2, float(max_steps) * 0.002),
            max_steps=max(1, int(max_steps)),
            output_dir=sidecar_dir,
            command_xyz=command_xyz,
        )
    except Exception as exc:
        return {
            "schema_version": "official_unitree_controller_sidecar.v1",
            "generated_at": generated_at,
            "status": "blocked",
            "enabled": True,
            "selected_unitree_rl_gym_root": str(selected_root),
            "manifest_path": str(manifest_path),
            "official_unitree_controller_used": False,
            "balanced_walking_controller_proven": False,
            "realistic_navigation_policy_used_for_endpoint_rollouts": False,
            "endpoint_task_control_used": False,
            "command_xyz": list(command_xyz) if command_xyz is not None else None,
            "blockers": ["blocked_official_unitree_controller_sidecar_failed"],
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    completed = manifest.get("status") == "completed"
    proof_boundary = _mapping(manifest.get("proof_boundary"))
    controller_proven = bool(
        completed and proof_boundary.get("non_default_policy_execution_trace_proven")
    )
    return {
        "schema_version": "official_unitree_controller_sidecar.v1",
        "generated_at": generated_at,
        "status": "completed" if controller_proven else "blocked",
        "enabled": True,
        "selected_unitree_rl_gym_root": str(selected_root),
        "manifest_path": str(manifest_path),
        "policy_id": manifest.get("policy_id"),
        "official_unitree_controller_used": controller_proven,
        "official_policy_execution_proven": controller_proven,
        "balanced_walking_controller_proven": controller_proven,
        "realistic_navigation_policy_used_for_endpoint_rollouts": False,
        "endpoint_task_control_used": False,
        "command_xyz": list(command_xyz)
        if command_xyz is not None
        else _mapping(manifest.get("metrics")).get("command_xyz"),
        "blockers": []
        if controller_proven
        else ["blocked_official_unitree_controller_sidecar_not_proven"],
        "claim_boundary": {
            "sidecar_execution_is_not_endpoint_task_control": True,
            "sidecar_execution_is_not_physical_robot_readiness": True,
            "task_conditioned_controller_bridge_required": True,
        },
    }


def _unitree_command_rows_from_endpoint_actions(
    action_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    command_rows: list[dict[str, Any]] = []
    for row in action_rows:
        action = _mapping(row.get("normalized_action"))
        if action.get("normalization_status") != "accepted" or row.get("rejected"):
            continue
        safe_command = _unitree_controller_safe_command(action)
        command_rows.append(
            {
                "episode_id": row.get("episode_id"),
                "scenario_eval_run_id": row.get("scenario_eval_run_id"),
                "task_id": row.get("task_id"),
                "spawn_id": row.get("spawn_id"),
                "step": row.get("step"),
                "sim_time_s": row.get("sim_time_s"),
                "source": row.get("source"),
                "action_type": action.get("action_type"),
                **safe_command,
                "target_waypoint": action.get("target_waypoint"),
            }
        )
    return command_rows


def _bounded_float(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


def _unitree_controller_safe_command_from_values(
    vx_mps: Any,
    vy_mps: Any,
    yaw_rate_rad_s: Any,
) -> dict[str, Any]:
    raw = [
        float(vx_mps or 0.0),
        float(vy_mps or 0.0),
        float(yaw_rate_rad_s or 0.0),
    ]
    limits = UNITREE_RL_GYM_CONTROLLER_COMMAND_LIMITS
    command = [
        _bounded_float(
            raw[0],
            -float(limits["max_reverse_velocity_mps"]),
            float(limits["max_forward_velocity_mps"]),
        ),
        _bounded_float(
            raw[1],
            -float(limits["max_lateral_velocity_mps"]),
            float(limits["max_lateral_velocity_mps"]),
        ),
        _bounded_float(
            raw[2],
            -float(limits["max_yaw_rate_rad_s"]),
            float(limits["max_yaw_rate_rad_s"]),
        ),
    ]
    rounded_raw = [round(float(value), 6) for value in raw]
    rounded_command = [round(float(value), 6) for value in command]
    return {
        "raw_endpoint_command_xyz": rounded_raw,
        "controller_command_xyz": rounded_command,
        "command_xyz": rounded_command,
        "controller_command_clamped": any(
            abs(float(before) - float(after)) > 1e-9
            for before, after in zip(raw, command)
        ),
        "controller_command_limits": dict(limits),
    }


def _unitree_controller_safe_command(action: Mapping[str, Any]) -> dict[str, Any]:
    return _unitree_controller_safe_command_from_values(
        action.get("vx_mps"),
        action.get("vy_mps"),
        action.get("yaw_rate_rad_s"),
    )


def _representative_unitree_command(command_rows: Sequence[Mapping[str, Any]]) -> list[float] | None:
    for row in command_rows:
        command = row.get("command_xyz")
        if (
            isinstance(command, Sequence)
            and not isinstance(command, (str, bytes))
            and len(command) == 3
        ):
            values = [float(command[0]), float(command[1]), float(command[2])]
            if any(abs(value) > 1e-6 for value in values):
                return [round(value, 6) for value in values]
    for row in command_rows:
        command = row.get("command_xyz")
        if (
            isinstance(command, Sequence)
            and not isinstance(command, (str, bytes))
            and len(command) == 3
        ):
            return [round(float(command[0]), 6), round(float(command[1]), 6), round(float(command[2]), 6)]
    return None


def _run_unitree_controller_replay_from_endpoint_actions(
    *,
    job_dir: Path,
    job_id: str,
    generated_at: str,
    unitree_rl_gym_root: Path | None,
    navigation_discovery: Mapping[str, Any],
    enabled: bool,
    max_steps: int,
    command_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    representative_command = _representative_unitree_command(command_rows)
    selected_root = _select_unitree_rl_gym_root(
        explicit_root=unitree_rl_gym_root,
        discovery=navigation_discovery,
    )
    replay_dir = job_dir / "unitree_endpoint_action_controller_replay"
    manifest_path = replay_dir / "official_unitree_g1_policy_execution_manifest.json"
    if not enabled:
        return {
            "schema_version": "unitree_endpoint_action_controller_replay.v1",
            "generated_at": generated_at,
            "status": "skipped",
            "enabled": False,
            "selected_unitree_rl_gym_root": str(selected_root) if selected_root else None,
            "representative_endpoint_command_xyz": representative_command,
            "endpoint_action_command_count": len(command_rows),
            "manifest_path": str(manifest_path),
            "official_unitree_controller_used": False,
            "balanced_walking_controller_proven": False,
            "endpoint_action_trace_bound_to_unitree_command_stream": bool(command_rows),
            "realistic_navigation_policy_used_for_endpoint_rollouts": False,
            "same_scene_controller_backend_integrated": False,
            "blockers": ["skipped_unitree_endpoint_action_controller_replay"],
        }
    if selected_root is None:
        return {
            "schema_version": "unitree_endpoint_action_controller_replay.v1",
            "generated_at": generated_at,
            "status": "blocked",
            "enabled": True,
            "selected_unitree_rl_gym_root": None,
            "representative_endpoint_command_xyz": representative_command,
            "endpoint_action_command_count": len(command_rows),
            "manifest_path": str(manifest_path),
            "official_unitree_controller_used": False,
            "balanced_walking_controller_proven": False,
            "endpoint_action_trace_bound_to_unitree_command_stream": bool(command_rows),
            "realistic_navigation_policy_used_for_endpoint_rollouts": False,
            "same_scene_controller_backend_integrated": False,
            "blockers": ["blocked_missing_unitree_rl_gym_root_or_required_files"],
        }
    if representative_command is None:
        return {
            "schema_version": "unitree_endpoint_action_controller_replay.v1",
            "generated_at": generated_at,
            "status": "blocked",
            "enabled": True,
            "selected_unitree_rl_gym_root": str(selected_root),
            "representative_endpoint_command_xyz": None,
            "endpoint_action_command_count": len(command_rows),
            "manifest_path": str(manifest_path),
            "official_unitree_controller_used": False,
            "balanced_walking_controller_proven": False,
            "endpoint_action_trace_bound_to_unitree_command_stream": bool(command_rows),
            "realistic_navigation_policy_used_for_endpoint_rollouts": False,
            "same_scene_controller_backend_integrated": False,
            "blockers": ["blocked_missing_endpoint_command_vector"],
        }
    try:
        from .unitree_g1_policy_execution import build_unitree_g1_policy_execution

        manifest = build_unitree_g1_policy_execution(
            capture_root=job_dir,
            unitree_rl_gym_root=selected_root,
            job_id=f"{job_id}_endpoint_action_replay",
            duration_seconds=max(0.2, float(max_steps) * 0.002),
            max_steps=max(1, int(max_steps)),
            output_dir=replay_dir,
            command_xyz=representative_command,
        )
    except Exception as exc:
        return {
            "schema_version": "unitree_endpoint_action_controller_replay.v1",
            "generated_at": generated_at,
            "status": "blocked",
            "enabled": True,
            "selected_unitree_rl_gym_root": str(selected_root),
            "representative_endpoint_command_xyz": representative_command,
            "endpoint_action_command_count": len(command_rows),
            "manifest_path": str(manifest_path),
            "official_unitree_controller_used": False,
            "balanced_walking_controller_proven": False,
            "endpoint_action_trace_bound_to_unitree_command_stream": bool(command_rows),
            "realistic_navigation_policy_used_for_endpoint_rollouts": False,
            "same_scene_controller_backend_integrated": False,
            "blockers": ["blocked_unitree_endpoint_action_controller_replay_failed"],
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    completed = manifest.get("status") == "completed"
    proof_boundary = _mapping(manifest.get("proof_boundary"))
    controller_proven = bool(
        completed and proof_boundary.get("non_default_policy_execution_trace_proven")
    )
    return {
        "schema_version": "unitree_endpoint_action_controller_replay.v1",
        "generated_at": generated_at,
        "status": "completed" if controller_proven else "blocked",
        "enabled": True,
        "selected_unitree_rl_gym_root": str(selected_root),
        "manifest_path": str(manifest_path),
        "policy_id": manifest.get("policy_id"),
        "representative_endpoint_command_xyz": representative_command,
        "endpoint_action_command_count": len(command_rows),
        "official_unitree_controller_used": controller_proven,
        "official_policy_execution_proven": controller_proven,
        "balanced_walking_controller_proven": controller_proven,
        "endpoint_action_trace_bound_to_unitree_command_stream": bool(command_rows),
        "realistic_navigation_policy_used_for_endpoint_rollouts": False,
        "same_scene_controller_backend_integrated": False,
        "blockers": []
        if controller_proven
        else ["blocked_unitree_endpoint_action_controller_replay_not_proven"],
        "claim_boundary": {
            "endpoint_action_replay_is_not_same_scene_task_control": True,
            "same_scene_controller_backend_still_required": True,
            "physical_robot_readiness_proven": False,
        },
    }


def build_unitree_controller_bridge_manifest(
    *,
    generated_at: str,
    command_rows: Sequence[Mapping[str, Any]],
    official_controller_sidecar: Mapping[str, Any],
    endpoint_replay: Mapping[str, Any],
    same_scene_controller: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    representative_command = _representative_unitree_command(command_rows)
    replay_proven = bool(endpoint_replay.get("official_unitree_controller_used"))
    sidecar_proven = bool(official_controller_sidecar.get("official_unitree_controller_used"))
    same_scene = _mapping(same_scene_controller)
    same_scene_integrated = bool(same_scene.get("same_scene_controller_backend_integrated"))
    blockers: list[str] = []
    if not command_rows:
        blockers.append("blocked_missing_endpoint_action_command_stream")
    if not replay_proven and not sidecar_proven and not same_scene_integrated:
        blockers.append("blocked_missing_unitree_controller_execution_proof")
    if not same_scene_integrated:
        blockers.append("blocked_same_scene_unitree_controller_bridge_not_integrated")
    if same_scene_integrated and not bool(same_scene.get("balanced_walking_controller_proven")):
        blockers.extend(str(item) for item in same_scene.get("blockers", []) or [])
    if same_scene_integrated:
        status = (
            "completed"
            if command_rows and not blockers
            else "completed_with_failures"
            if command_rows
            else "blocked"
        )
    elif command_rows and (replay_proven or sidecar_proven):
        status = "bridge_ready_for_implementation"
    else:
        status = "blocked"
    return {
        "schema_version": "unitree_controller_bridge_manifest.v1",
        "generated_at": generated_at,
        "status": status,
        "endpoint_action_trace_bound_to_unitree_command_stream": bool(command_rows),
        "endpoint_action_command_count": len(command_rows),
        "representative_endpoint_command_xyz": representative_command,
        "sample_endpoint_commands": [dict(row) for row in command_rows[:50]],
        "official_controller_sidecar_status": official_controller_sidecar.get("status"),
        "endpoint_action_controller_replay_status": endpoint_replay.get("status"),
        "same_scene_controller_backend_status": same_scene.get("status"),
        "official_unitree_controller_used": replay_proven or sidecar_proven or same_scene_integrated,
        "balanced_walking_controller_proven": bool(
            endpoint_replay.get("balanced_walking_controller_proven")
            or official_controller_sidecar.get("balanced_walking_controller_proven")
            or same_scene.get("balanced_walking_controller_proven")
        ),
        "realistic_navigation_policy_used_for_endpoint_rollouts": same_scene_integrated,
        "freejoint_proxy_used_for_endpoint_rollouts": not same_scene_integrated,
        "same_scene_controller_backend_integrated": same_scene_integrated,
        "next_bridge_required": (
            "Replace the freejoint proxy update in the MuJoCo endpoint loop with a "
            "Unitree controller backend that consumes these command vectors while "
            "stepping the same task scene."
        )
        if not same_scene_integrated
        else None,
        "blockers": blockers,
        "claim_boundary": {
            "command_binding_is_not_same_scene_controller_execution": not same_scene_integrated,
            "controller_sidecar_or_replay_is_not_physical_robot_readiness": True,
            "freejoint_proxy_is_not_realistic_navigation_policy": not same_scene_integrated,
        },
    }


def build_policy_endpoint_server_manifest(
    *,
    generated_at: str,
    selected_runtime: Mapping[str, Any] | None,
    health_probe: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "policy_endpoint_server_manifest.v1",
        "generated_at": generated_at,
        "status": "completed" if selected_runtime else "blocked_missing_policy_endpoint",
        "selected_endpoint_runtime": _mapping(selected_runtime).get("runtime"),
        "endpoint_url": _mapping(selected_runtime).get("endpoint_url"),
        "http_contract": {
            "health": {"method": "GET", "path": "/health"},
            "policy_action": {"method": "POST", "path": "/policy/action"},
        },
        "auth": {
            "type": "bearer_token_from_file",
            "auth_token_file_env": _mapping(selected_runtime).get("auth_file_env"),
            "auth_token_file_configured": bool(
                _mapping(selected_runtime).get("auth_token_file_configured")
            ),
            "auth_token_file_exists": bool(_mapping(selected_runtime).get("auth_token_file_exists")),
            "raw_token_values_persisted": False,
            "raw_token_hashes_persisted": False,
        },
        "health_probe": dict(health_probe),
        "claim_boundary": {
            "server_reachable_is_not_policy_quality_proof": True,
            "physical_robot_readiness_proven": False,
        },
    }


def build_policy_command_adapter_manifest(
    *, generated_at: str, action_rows: Sequence[Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    rows = list(action_rows or [])
    policy_ids = sorted(
        {
            str(row.get("policy_id"))
            for row in rows
            if row.get("source") == "endpoint_policy" and row.get("policy_id")
        }
    )
    return {
        "schema_version": "policy_command_adapter_manifest.v1",
        "generated_at": generated_at,
        "status": "completed" if rows else "defined",
        "adapter_families": [
            "command_policy",
            "unitree_g1_policy",
            "openvla_policy",
            "oscar_wam",
            "cosmos_wam",
        ],
        "default_reference_adapter_command": "blueprint-g1-endpoint-reference-adapter",
        "default_reference_adapter_available_on_path": bool(
            shutil.which("blueprint-g1-endpoint-reference-adapter")
        ),
        "observed_endpoint_policy_ids": policy_ids,
        "supported_action_types": [
            "waypoint",
            "base_velocity",
            "stop",
            "inspect_look",
            "manipulation_contact",
        ],
        "stdin_contract": {"observation": "Blueprint observation packet"},
        "stdout_contract": {"policy_id": "string", "action": "Blueprint supported action"},
        "raw_tokens_written_to_artifacts": False,
        "claim_boundary": {
            "reference_adapter_is_heuristic_endpoint_plumbing": True,
            "reference_adapter_is_not_real_wam_vla": True,
            "model_backends_replaceable": True,
        },
    }


def build_policy_endpoint_runtime_manifest(
    *,
    generated_at: str,
    selected_runtime: Mapping[str, Any] | None,
    endpoint_policy_used: bool,
    fixture_policy_used: bool,
    endpoint_invocation_count: int,
    endpoint_valid_action_count: int,
    rejected_policy_action_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": "policy_endpoint_runtime_manifest.v1",
        "generated_at": generated_at,
        "status": "completed" if endpoint_policy_used else "blocked_or_fixture_fallback",
        "selected_endpoint_runtime": _mapping(selected_runtime).get("runtime"),
        "selected_endpoint_env": _mapping(selected_runtime).get("endpoint_env"),
        "endpoint_url": _mapping(selected_runtime).get("endpoint_url"),
        "endpoint_policy_used": bool(endpoint_policy_used),
        "fixture_policy_used": bool(fixture_policy_used),
        "endpoint_invocation_count": int(endpoint_invocation_count),
        "endpoint_valid_action_count": int(endpoint_valid_action_count),
        "rejected_policy_action_count": int(rejected_policy_action_count),
        "raw_tokens_written_to_artifacts": False,
        "raw_token_hashes_written_to_artifacts": False,
        "claim_boundary": {
            "endpoint_invocation_is_not_model_quality_proof": True,
            "mujoco_evidence_is_simulator_only": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
        },
    }


def _observation_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": OBSERVATION_SCHEMA_ID,
        "type": "object",
        "required": [
            "episode_id",
            "scenario_id",
            "task_id",
            "spawn_id",
            "robot",
            "sim_time_s",
            "proprioception",
            "base_pose",
            "base_velocity",
            "contact_state",
            "route_task_state",
            "object_state",
            "task_prompt",
            "allowed_action_schema",
            "safety_limits",
        ],
        "properties": {
            "episode_id": {"type": "string"},
            "scenario_id": {"type": "string"},
            "task_id": {"type": "string"},
            "spawn_id": {"type": "string"},
            "robot": {"type": "object"},
            "sim_time_s": {"type": "number"},
            "proprioception": {"type": "object"},
            "base_pose": {"type": "object"},
            "base_velocity": {"type": "object"},
            "contact_state": {"type": "object"},
            "route_task_state": {"type": "object"},
            "object_state": {"type": "object"},
            "sensor_surrogates": {"type": "object"},
            "task_prompt": {"type": "string"},
            "allowed_action_schema": {"type": "object"},
            "safety_limits": {"type": "object"},
        },
    }


def _action_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": ACTION_SCHEMA_ID,
        "type": "object",
        "oneOf": [
            {
                "required": ["action_type", "linear_velocity_mps"],
                "properties": {
                    "action_type": {"const": "base_velocity"},
                    "linear_velocity_mps": {"type": "number"},
                    "lateral_velocity_mps": {"type": "number"},
                    "yaw_rate_rad_s": {"type": "number"},
                },
            },
            {
                "required": ["action_type", "target_yaw_rad"],
                "properties": {
                    "action_type": {"const": "heading_yaw"},
                    "target_yaw_rad": {"type": "number"},
                },
            },
            {
                "required": ["action_type", "waypoint"],
                "properties": {
                    "action_type": {"const": "waypoint"},
                    "waypoint": {"type": "array", "minItems": 2},
                },
            },
            {"required": ["action_type"], "properties": {"action_type": {"const": "stop"}}},
            {
                "required": ["action_type"],
                "properties": {"action_type": {"enum": ["inspect_look", "look"]}},
            },
            {
                "required": ["action_type"],
                "properties": {"action_type": {"const": "manipulation_contact"}},
            },
        ],
    }


def build_scenario_eval_matrix(
    *,
    job_id: str,
    generated_at: str,
    task_filter: Sequence[str] | None = None,
    spawn_filter: Sequence[str] | None = None,
    max_tasks: int | None = None,
    max_spawns: int | None = None,
) -> dict[str, Any]:
    task_ids = set(task_filter or [])
    spawn_ids = set(spawn_filter or [])
    tasks = [task for task in TASK_SPECS if not task_ids or task["task_id"] in task_ids]
    spawns = [spawn for spawn in SPAWN_SPECS if not spawn_ids or spawn["spawn_id"] in spawn_ids]
    if max_tasks is not None:
        tasks = tasks[: max(0, int(max_tasks))]
    if max_spawns is not None:
        spawns = spawns[: max(0, int(max_spawns))]
    runs: list[dict[str, Any]] = []
    for spawn in spawns:
        for task in tasks:
            run_index = len(runs) + 1
            run_id = f"{job_id}__{_safe_id(spawn['spawn_id'])}__{_safe_id(task['task_id'])}"
            start = [float(spawn["pose_xy_yaw"][0]), float(spawn["pose_xy_yaw"][1]), 0.79]
            target = [float(task["target_xy"][0]), float(task["target_xy"][1]), 0.79]
            runs.append(
                {
                    "schema_version": "mujoco_g1_wam_vla_scenario_eval_run.v1",
                    "scenario_eval_run_id": run_id,
                    "episode_id": f"episode_{run_index:04d}_{_safe_id(spawn['spawn_id'])}_{_safe_id(task['task_id'])}",
                    "scenario_variation_instance_id": f"{task['scenario_id']}:{spawn['spawn_id']}",
                    "scenario_id": task["scenario_id"],
                    "task_id": task["task_id"],
                    "spawn_id": spawn["spawn_id"],
                    "spawn_pose": start,
                    "spawn_yaw_rad": float(spawn["pose_xy_yaw"][2]),
                    "target_pose": target,
                    "route_waypoints": task.get("route_waypoints") or [target[:2]],
                    "task_prompt": task["prompt"],
                    "expected_action_types": list(task["expected_action_types"]),
                    "object_id": task.get("object_id"),
                    "expected_blockers": list(spawn.get("expected_blockers", [])),
                    "deterministic_seed": run_index * 9973,
                }
            )
    return {
        "schema_version": SCENARIO_MATRIX_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "completed" if runs else "blocked_empty_matrix",
        "spawn_count": len(spawns),
        "task_count": len(tasks),
        "scenario_eval_run_count": len(runs),
        "minimum_required_shape": "5_spawns_x_5_tasks_for_full_run",
        "runs": runs,
    }


def _write_scene_xml(*, g1_xml: Path, output_xml: Path) -> dict[str, Any]:
    scene = f"""<mujoco model="blueprint_mujoco_g1_wam_vla_policy_endpoint_eval">
  <include file="{g1_xml}"/>
  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.25 0.25 0.25" specular="0.4 0.4 0.4"/>
    <map znear="0.01" zfar="100"/>
    <global offwidth="640" offheight="360" azimuth="145" elevation="-20"/>
  </visual>
  <asset>
    <material name="blueprint_floor_mat" rgba="0.18 0.20 0.22 1"/>
    <material name="blueprint_obstacle_mat" rgba="0.75 0.15 0.10 1"/>
    <material name="blueprint_table_mat" rgba="0.25 0.45 0.80 1"/>
    <material name="blueprint_object_mat" rgba="0.98 0.78 0.20 1"/>
  </asset>
  <worldbody>
    <light name="blueprint_eval_key_light" pos="0 -4 6" dir="0 0 -1" directional="true"/>
    <geom name="blueprint_reference_floor" type="plane" size="5 5 0.05" material="blueprint_floor_mat" contype="1" conaffinity="1"/>
    <geom name="blueprint_route_obstruction" type="box" pos="0.28 0.52 0.34" size="0.10 0.28 0.34" material="blueprint_obstacle_mat" contype="1" conaffinity="1"/>
    <geom name="blueprint_task_wall_or_table" type="box" pos="0.70 -0.22 0.38" size="0.12 0.20 0.38" material="blueprint_table_mat" contype="1" conaffinity="1"/>
    <geom name="blueprint_occluder" type="box" pos="0.10 0.98 0.45" size="0.16 0.22 0.45" rgba="0.12 0.12 0.12 1" contype="1" conaffinity="1"/>
    <body name="blueprint_light_object" pos="0.36 -0.65 0.27">
      <freejoint name="blueprint_light_object_freejoint"/>
      <geom name="blueprint_light_object_geom" type="box" size="0.08 0.08 0.24" material="blueprint_object_mat" mass="0.25" contype="1" conaffinity="1" friction="0.7 0.02 0.002"/>
    </body>
  </worldbody>
</mujoco>
"""
    ensure_dir(output_xml.parent)
    output_xml.write_text(scene, encoding="utf-8")
    return {
        "schema_version": "mujoco_scene_manifest.v1",
        "status": "completed",
        "scene_xml": str(output_xml),
        "g1_include_xml": str(g1_xml),
        "visual_source_required": False,
        "splat_ply_spz_required": False,
        "scene_elements": [
            "blueprint_reference_floor",
            "blueprint_route_obstruction",
            "blueprint_task_wall_or_table",
            "blueprint_occluder",
            "blueprint_light_object",
        ],
        "manipulation_object_id": "blueprint_light_object",
    }


def _build_contact_metadata(model: Any, mujoco_module: Any) -> dict[int, dict[str, Any]]:
    geom_count = int(getattr(model, "ngeom", 0) or 0)
    if geom_count <= 0:
        try:
            geom_count = len(model.geom_bodyid)
        except Exception:
            geom_count = 0
    metadata: dict[int, dict[str, Any]] = {}
    for geom_id in range(max(0, geom_count)):
        geom_name = _mujoco_name(
            mujoco_module,
            model,
            mujoco_module.mjtObj.mjOBJ_GEOM,
            geom_id,
        )
        body_name = None
        try:
            body_id = int(model.geom_bodyid[geom_id])
            body_name = _mujoco_name(
                mujoco_module,
                model,
                mujoco_module.mjtObj.mjOBJ_BODY,
                body_id,
            )
        except Exception:
            body_id = -1
        names = {name for name in (geom_name, body_name) if name}
        metadata[geom_id] = {
            "geom_name": geom_name,
            "body_id": body_id,
            "body_name": body_name,
            "names": names,
        }
    return metadata


def _contact_metadata_for_geom(
    *,
    model: Any,
    mujoco_module: Any,
    contact_metadata: dict[int, dict[str, Any]] | None,
    geom_id: int,
) -> dict[str, Any]:
    if contact_metadata is not None and geom_id in contact_metadata:
        return contact_metadata[geom_id]
    geom_name = _mujoco_name(
        mujoco_module,
        model,
        mujoco_module.mjtObj.mjOBJ_GEOM,
        geom_id,
    )
    body_name = None
    try:
        body_id = int(model.geom_bodyid[geom_id])
        body_name = _mujoco_name(
            mujoco_module,
            model,
            mujoco_module.mjtObj.mjOBJ_BODY,
            body_id,
        )
    except Exception:
        body_id = -1
    return {
        "geom_name": geom_name,
        "body_id": body_id,
        "body_name": body_name,
        "names": {name for name in (geom_name, body_name) if name},
    }


def _contact_state(
    model: Any,
    data: Any,
    mujoco_module: Any,
    *,
    contact_metadata: dict[int, dict[str, Any]] | None = None,
    include_force: bool = True,
    record_limit: int | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    ncon = int(getattr(data, "ncon", 0) or 0)
    floor_contact_count = 0
    object_contact_count = 0
    obstacle_contact_count = 0
    left_foot_contact_count = 0
    right_foot_contact_count = 0
    normalized_limit = None if record_limit is None else max(0, int(record_limit))
    for index in range(int(getattr(data, "ncon", 0) or 0)):
        contact = data.contact[index]
        geom_ids = [int(contact.geom1), int(contact.geom2)]
        metadata = [
            _contact_metadata_for_geom(
                model=model,
                mujoco_module=mujoco_module,
                contact_metadata=contact_metadata,
                geom_id=geom_id,
            )
            for geom_id in geom_ids
        ]
        geom_names = [item.get("geom_name") for item in metadata]
        body_names = [item.get("body_name") for item in metadata]
        names = {
            name
            for item in metadata
            for name in set(item.get("names") or set())
            if name
        }
        floor_contact = "blueprint_reference_floor" in names
        object_contact = (
            "blueprint_light_object" in names
            or "blueprint_light_object_geom" in names
        )
        obstacle_contact = bool(
            names
            & {
                "blueprint_route_obstruction",
                "blueprint_task_wall_or_table",
                "blueprint_occluder",
            }
        )
        left_foot_contact = "left_ankle_roll_link" in names
        right_foot_contact = "right_ankle_roll_link" in names
        floor_contact_count += int(floor_contact)
        object_contact_count += int(object_contact)
        obstacle_contact_count += int(obstacle_contact)
        left_foot_contact_count += int(left_foot_contact)
        right_foot_contact_count += int(right_foot_contact)
        if normalized_limit is not None and len(records) >= normalized_limit:
            continue
        force = [0.0] * 6
        if include_force:
            try:
                force_vec = [0.0] * 6
                mujoco_module.mj_contactForce(model, data, index, force_vec)
                force = [round(float(value), 8) for value in force_vec]
            except Exception:
                force = [0.0] * 6
        records.append(
            {
                "contact_index": index,
                "geom_ids": geom_ids,
                "geom_names": geom_names,
                "body_names": body_names,
                "distance": round(float(contact.dist), 9),
                "position": [round(float(value), 6) for value in contact.pos],
                "contact_force_6d": force,
                "floor_contact": floor_contact,
                "object_contact": object_contact,
                "obstacle_contact": obstacle_contact,
                "left_foot_contact": left_foot_contact,
                "right_foot_contact": right_foot_contact,
            }
        )
    return {
        "contact_count": ncon,
        "floor_contact_count": floor_contact_count,
        "object_contact_count": object_contact_count,
        "obstacle_contact_count": obstacle_contact_count,
        "left_foot_contact_count": left_foot_contact_count,
        "right_foot_contact_count": right_foot_contact_count,
        "records": records,
        "record_count": len(records),
        "records_truncated": len(records) < ncon,
        "dropped_record_count": max(0, ncon - len(records)),
    }


def _contact_records(
    model: Any,
    data: Any,
    mujoco_module: Any,
    *,
    contact_metadata: dict[int, dict[str, Any]] | None = None,
    include_force: bool = True,
    record_limit: int | None = None,
) -> list[dict[str, Any]]:
    state = _contact_state(
        model,
        data,
        mujoco_module,
        contact_metadata=contact_metadata,
        include_force=include_force,
        record_limit=record_limit,
    )
    records = state["records"]
    return records


def _object_pose(data: Any, object_qpos: int | None) -> dict[str, Any]:
    if object_qpos is None:
        return {"available": False}
    return {
        "available": True,
        "object_id": "blueprint_light_object",
        "position": [round(float(value), 6) for value in data.qpos[object_qpos : object_qpos + 3]],
        "quaternion_wxyz": [
            round(float(value), 6) for value in data.qpos[object_qpos + 3 : object_qpos + 7]
        ],
    }


def _build_observation_packet(
    *,
    model: Any,
    data: Any,
    root_qpos: int,
    root_dof: int,
    object_qpos: int | None,
    run: Mapping[str, Any],
    step: int,
    contacts: Sequence[Mapping[str, Any]],
    contact_summary: Mapping[str, Any] | None = None,
    mujoco_version: str,
) -> dict[str, Any]:
    qpos = [round(float(value), 8) for value in data.qpos[:]]
    qvel = [round(float(value), 8) for value in data.qvel[:]]
    root_pos = [float(value) for value in data.qpos[root_qpos : root_qpos + 3]]
    root_quat = [float(value) for value in data.qpos[root_qpos + 3 : root_qpos + 7]]
    root_vel = [float(value) for value in data.qvel[root_dof : root_dof + 3]]
    root_ang = [float(value) for value in data.qvel[root_dof + 3 : root_dof + 6]]
    target = list(run.get("target_pose") or [0.0, 0.0, 0.79])
    contact_state = dict(contact_summary or {})
    return {
        "schema_version": OBSERVATION_SCHEMA_ID,
        "episode_id": run.get("episode_id"),
        "scenario_id": run.get("scenario_id"),
        "scenario_eval_run_id": run.get("scenario_eval_run_id"),
        "task_id": run.get("task_id"),
        "spawn_id": run.get("spawn_id"),
        "robot": {
            "robot_profile_id": ROBOT_PROFILE_ID,
            "model": "Unitree G1",
            "model_source": "google_deepmind_mujoco_menagerie",
            "simulator": "mujoco",
            "mujoco_version": mujoco_version,
        },
        "sim_time_s": round(float(data.time), 9),
        "step_index": step,
        "proprioception": {"qpos": qpos, "qvel": qvel, "nq": int(model.nq), "nv": int(model.nv)},
        "base_pose": {
            "position": [round(value, 6) for value in root_pos],
            "quaternion_wxyz": [round(value, 6) for value in root_quat],
            "yaw_rad": round(_yaw_from_quat(root_quat), 6),
        },
        "base_velocity": {
            "linear_xyz_mps": [round(value, 6) for value in root_vel],
            "angular_xyz_rad_s": [round(value, 6) for value in root_ang],
            "planar_speed_mps": round(math.hypot(root_vel[0], root_vel[1]), 6),
        },
        "contact_state": {
            "contact_count": int(contact_state.get("contact_count", len(contacts)) or 0),
            "left_foot_contact": bool(
                contact_state.get("left_foot_contact_count")
                if contact_state
                else any(contact.get("left_foot_contact") for contact in contacts)
            ),
            "right_foot_contact": bool(
                contact_state.get("right_foot_contact_count")
                if contact_state
                else any(contact.get("right_foot_contact") for contact in contacts)
            ),
            "floor_contact_count": int(
                contact_state.get(
                    "floor_contact_count",
                    sum(1 for contact in contacts if contact.get("floor_contact")),
                )
                or 0
            ),
            "object_contact_count": int(
                contact_state.get(
                    "object_contact_count",
                    sum(1 for contact in contacts if contact.get("object_contact")),
                )
                or 0
            ),
            "obstacle_contact_count": int(
                contact_state.get(
                    "obstacle_contact_count",
                    sum(1 for contact in contacts if contact.get("obstacle_contact")),
                )
                or 0
            ),
            "contacts_truncated": bool(contact_state.get("records_truncated", False)),
            "contacts": [dict(contact) for contact in contacts[:12]],
        },
        "route_task_state": {
            "target_pose": target,
            "target_error_m": round(math.dist(root_pos[:2], [float(target[0]), float(target[1])]), 6),
            "route_waypoints": run.get("route_waypoints") or [],
            "task_prompt": run.get("task_prompt"),
        },
        "object_state": _object_pose(data, object_qpos),
        "sensor_surrogates": {
            "camera_surrogates": ["third_person", "overhead", "robot_follow"],
            "visual_assets_required": False,
            "splat_ply_spz_required": False,
        },
        "task_prompt": run.get("task_prompt"),
        "allowed_action_schema": {"schema_id": ACTION_SCHEMA_ID, "supported_action_types": [
            "base_velocity",
            "heading_yaw",
            "waypoint",
            "stop",
            "inspect_look",
            "manipulation_contact",
        ]},
        "safety_limits": dict(SAFETY_LIMITS),
    }


def _fixture_policy_action(*, observation: Mapping[str, Any]) -> dict[str, Any]:
    task_id = str(observation.get("task_id") or "")
    spawn_id = str(observation.get("spawn_id") or "")
    step = int(observation.get("step_index") or 0)
    route = _mapping(observation.get("route_task_state"))
    target = route.get("target_pose") or [0.0, 0.0, 0.79]
    object_state = _mapping(observation.get("object_state"))
    if spawn_id == "blocked_or_occluded" and task_id == "inspect_target" and step == 0:
        return {
            "policy_id": REFERENCE_FIXTURE_POLICY_ID,
            "action": {"action_type": "base_velocity", "linear_velocity_mps": "fast"},
            "fixture_intent": "exercise_rejected_policy_action_path",
        }
    if task_id == "inspect_target":
        return {
            "policy_id": REFERENCE_FIXTURE_POLICY_ID,
            "action": {"action_type": "inspect_look", "yaw_rate_rad_s": 0.35},
        }
    if task_id == "contact_or_push_light_object":
        object_pos = object_state.get("position") or [0.36, -0.65, 0.27]
        return {
            "policy_id": REFERENCE_FIXTURE_POLICY_ID,
            "action": {
                "action_type": "manipulation_contact",
                "target_object_id": "blueprint_light_object",
                "waypoint": [float(object_pos[0]) + 0.18, float(object_pos[1]), 0.79],
            },
        }
    if task_id == "stop_at_goal_and_report" and float(route.get("target_error_m") or 9.0) < 0.38:
        return {
            "policy_id": REFERENCE_FIXTURE_POLICY_ID,
            "action": {"action_type": "stop", "report": "at_goal"},
        }
    if task_id == "route_around_obstruction":
        waypoints = route.get("route_waypoints") or []
        waypoint = waypoints[min(len(waypoints) - 1, max(0, step // 120))] if waypoints else target
        return {
            "policy_id": REFERENCE_FIXTURE_POLICY_ID,
            "action": {"action_type": "waypoint", "waypoint": waypoint},
        }
    return {
        "policy_id": REFERENCE_FIXTURE_POLICY_ID,
        "action": {"action_type": "waypoint", "waypoint": target},
    }


def _read_token(path: str | None) -> str | None:
    if not path:
        return None
    token_path = Path(path).expanduser()
    if not token_path.is_file():
        return None
    return token_path.read_text(encoding="utf-8").strip()


def _call_endpoint_action(
    *,
    endpoint_row: Mapping[str, Any] | None,
    observation: Mapping[str, Any],
    timeout_seconds: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if endpoint_row is None:
        return None, {
            "status": "blocked_missing_policy_endpoint",
            "endpoint_invoked": False,
            "blockers": ["blocked_missing_policy_endpoint"],
        }
    endpoint = str(endpoint_row.get("endpoint_url") or "")
    token = _read_token(str(endpoint_row.get("auth_token_file_path") or ""))
    if not endpoint or not token:
        return None, {
            "status": "blocked_missing_policy_endpoint_or_auth",
            "endpoint_invoked": False,
            "runtime": endpoint_row.get("runtime"),
            "blockers": ["blocked_missing_policy_endpoint_or_auth"],
        }
    request_payload = json.dumps({"observation": observation}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=request_payload,
        method="POST",
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {token}",
        },
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
            elapsed = round(time.monotonic() - started, 6)
            payload = json.loads(body.decode("utf-8"))
            return _mapping(payload), {
                "status": "completed",
                "endpoint_invoked": True,
                "runtime": endpoint_row.get("runtime"),
                "http_status": int(getattr(response, "status", 0) or 0),
                "duration_seconds": elapsed,
                "response_size_bytes": len(body),
                "response_json_keys": sorted(payload) if isinstance(payload, Mapping) else [],
                "raw_response_metadata_preserved": True,
                "raw_token_persisted": False,
            }
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return None, {
            "status": "failed",
            "endpoint_invoked": True,
            "runtime": endpoint_row.get("runtime"),
            "duration_seconds": round(time.monotonic() - started, 6),
            "blockers": ["policy_endpoint_call_failed"],
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "raw_token_persisted": False,
        }


def _extract_action(raw_payload: Any) -> Any:
    payload = _mapping(raw_payload)
    if "action" in payload:
        return payload.get("action")
    if "policy_action" in payload:
        return payload.get("policy_action")
    if "decision" in payload:
        decision = _mapping(payload.get("decision"))
        if "action" in decision:
            return decision["action"]
    return raw_payload


def normalize_policy_action(
    *,
    raw_payload: Any,
    observation: Mapping[str, Any],
    source: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    action = _extract_action(raw_payload)
    if not isinstance(action, Mapping):
        normalized = {
            "action_type": "stop",
            "vx_mps": 0.0,
            "vy_mps": 0.0,
            "yaw_rate_rad_s": 0.0,
            "source": source,
            "normalization_status": "rejected",
        }
        return normalized, {
            "reason": "policy_action_not_mapping",
            "raw_action_redacted": _redact(raw_payload),
            "observation_episode_id": observation.get("episode_id"),
        }
    action_type = str(action.get("action_type") or action.get("type") or "").strip()
    base_pose = _mapping(observation.get("base_pose"))
    base_pos = base_pose.get("position") or [0.0, 0.0, 0.79]
    yaw = float(base_pose.get("yaw_rad") or 0.0)
    rejected: dict[str, Any] | None = None
    vx = 0.0
    vy = 0.0
    yaw_rate = 0.0
    target_waypoint: list[float] | None = None
    if action_type == "base_velocity":
        linear = _number(action.get("linear_velocity_mps"))
        if linear is None:
            rejected = {"reason": "base_velocity_missing_numeric_linear_velocity"}
        else:
            lateral = _number(action.get("lateral_velocity_mps"), 0.0) or 0.0
            raw_yaw_rate = _number(action.get("yaw_rate_rad_s") or action.get("yaw_rate"), 0.0) or 0.0
            vx = max(-SAFETY_LIMITS["max_forward_velocity_mps"], min(SAFETY_LIMITS["max_forward_velocity_mps"], linear))
            vy = max(-SAFETY_LIMITS["max_lateral_velocity_mps"], min(SAFETY_LIMITS["max_lateral_velocity_mps"], lateral))
            yaw_rate = max(-SAFETY_LIMITS["max_yaw_rate_rad_s"], min(SAFETY_LIMITS["max_yaw_rate_rad_s"], raw_yaw_rate))
    elif action_type == "heading_yaw":
        target_yaw = _number(action.get("target_yaw_rad"))
        if target_yaw is None:
            rejected = {"reason": "heading_yaw_missing_numeric_target_yaw"}
        else:
            diff = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
            yaw_rate = max(-SAFETY_LIMITS["max_yaw_rate_rad_s"], min(SAFETY_LIMITS["max_yaw_rate_rad_s"], diff * 2.0))
    elif action_type == "waypoint":
        waypoint = action.get("waypoint")
        if not isinstance(waypoint, Sequence) or isinstance(waypoint, (str, bytes)) or len(waypoint) < 2:
            rejected = {"reason": "waypoint_missing_xy"}
        else:
            try:
                target_waypoint = [float(waypoint[0]), float(waypoint[1]), float(waypoint[2]) if len(waypoint) > 2 else 0.79]
                dx = target_waypoint[0] - float(base_pos[0])
                dy = target_waypoint[1] - float(base_pos[1])
                distance = math.hypot(dx, dy)
                if distance > SAFETY_LIMITS["max_waypoint_distance_m"]:
                    scale = SAFETY_LIMITS["max_waypoint_distance_m"] / distance
                    dx *= scale
                    dy *= scale
                    distance = SAFETY_LIMITS["max_waypoint_distance_m"]
                if distance > 1e-6:
                    speed = min(SAFETY_LIMITS["max_forward_velocity_mps"], max(0.12, distance * 2.2))
                    vx = speed * dx / distance
                    vy = speed * dy / distance
                    target_heading = math.atan2(dy, dx)
                    diff = math.atan2(math.sin(target_heading - yaw), math.cos(target_heading - yaw))
                    yaw_rate = max(-SAFETY_LIMITS["max_yaw_rate_rad_s"], min(SAFETY_LIMITS["max_yaw_rate_rad_s"], diff * 1.5))
            except (TypeError, ValueError):
                rejected = {"reason": "waypoint_contains_non_numeric_value"}
    elif action_type == "stop":
        vx = 0.0
        vy = 0.0
        yaw_rate = 0.0
    elif action_type in {"inspect_look", "look"}:
        yaw_rate = max(
            -SAFETY_LIMITS["max_yaw_rate_rad_s"],
            min(
                SAFETY_LIMITS["max_yaw_rate_rad_s"],
                _number(action.get("yaw_rate_rad_s"), 0.3) or 0.3,
            ),
        )
    elif action_type == "manipulation_contact":
        waypoint = action.get("waypoint")
        if isinstance(waypoint, Sequence) and not isinstance(waypoint, (str, bytes)) and len(waypoint) >= 2:
            target_waypoint = [float(waypoint[0]), float(waypoint[1]), float(waypoint[2]) if len(waypoint) > 2 else 0.79]
        else:
            object_state = _mapping(observation.get("object_state"))
            object_pos = object_state.get("position") or [0.36, -0.65, 0.27]
            target_waypoint = [float(object_pos[0]) + 0.18, float(object_pos[1]), 0.79]
        dx = target_waypoint[0] - float(base_pos[0])
        dy = target_waypoint[1] - float(base_pos[1])
        distance = max(1e-6, math.hypot(dx, dy))
        speed = min(SAFETY_LIMITS["max_forward_velocity_mps"], max(0.18, distance * 2.5))
        vx = speed * dx / distance
        vy = speed * dy / distance
    else:
        rejected = {"reason": "unsupported_policy_action_type", "action_type": action_type or None}
    if rejected is not None:
        return (
            {
                "action_type": "stop",
                "vx_mps": 0.0,
                "vy_mps": 0.0,
                "yaw_rate_rad_s": 0.0,
                "source": source,
                "normalization_status": "rejected",
            },
            {
                **rejected,
                "raw_action_redacted": _redact(action),
                "observation_episode_id": observation.get("episode_id"),
                "scenario_eval_run_id": observation.get("scenario_eval_run_id"),
            },
        )
    return (
        {
            "action_type": action_type,
            "vx_mps": round(float(vx), 6),
            "vy_mps": round(float(vy), 6),
            "yaw_rate_rad_s": round(float(yaw_rate), 6),
            "target_waypoint": target_waypoint,
            "source": source,
            "normalization_status": "accepted",
            "safety_limits_applied": dict(SAFETY_LIMITS),
        },
        None,
    )


def _set_joint_position_holds(model: Any, data: Any) -> None:
    for actuator_index in range(int(model.nu)):
        joint_id = int(model.actuator_trnid[actuator_index][0])
        if joint_id < 0:
            continue
        qpos_addr = int(model.jnt_qposadr[joint_id])
        data.ctrl[actuator_index] = data.qpos[qpos_addr]


def _mujoco_name(mujoco_module: Any, model: Any, obj_type: Any, index: int) -> str | None:
    try:
        return mujoco_module.mj_id2name(model, obj_type, int(index))
    except Exception:
        return None


def _actuator_id_by_name(mujoco_module: Any, model: Any, name: str) -> int:
    try:
        return int(mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_ACTUATOR, name))
    except Exception:
        return -1


def _joint_id_by_name(mujoco_module: Any, model: Any, name: str) -> int:
    try:
        return int(mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_JOINT, name))
    except Exception:
        return -1


class _SameSceneUnitreeRLGymController:
    def __init__(
        self,
        *,
        model: Any,
        mujoco_module: Any,
        root_qpos: int,
        root_dof: int,
        selected_root: Path,
        config: Mapping[str, Any],
        policy: Any,
        policy_path: Path,
        config_path: Path,
        leg_actuator_ids: Sequence[int],
        leg_qpos_addrs: Sequence[int],
        leg_qvel_addrs: Sequence[int],
        upper_hold_actuator_ids: Sequence[int],
        upper_hold_qpos_addrs: Sequence[int],
        actuator_output_mode: str,
    ) -> None:
        import numpy as np

        self.model = model
        self.mujoco_module = mujoco_module
        self.root_qpos = int(root_qpos)
        self.root_dof = int(root_dof)
        self.selected_root = selected_root
        self.config = dict(config)
        self.policy = policy
        self.policy_path = policy_path
        self.config_path = config_path
        self.leg_actuator_ids = [int(value) for value in leg_actuator_ids]
        self.leg_qpos_addrs = [int(value) for value in leg_qpos_addrs]
        self.leg_qvel_addrs = [int(value) for value in leg_qvel_addrs]
        self.upper_hold_actuator_ids = [int(value) for value in upper_hold_actuator_ids]
        self.upper_hold_qpos_addrs = [int(value) for value in upper_hold_qpos_addrs]
        self.actuator_output_mode = str(actuator_output_mode)
        self.simulation_dt = float(config["simulation_dt"])
        self.control_decimation = int(config["control_decimation"])
        self.kps = np.array(config["kps"], dtype=np.float32)
        self.kds = np.array(config["kds"], dtype=np.float32)
        self.default_angles = np.array(config["default_angles"], dtype=np.float32)
        self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        self.num_actions = int(config["num_actions"])
        self.num_obs = int(config["num_obs"])
        self.action_scale = float(config["action_scale"])
        self.dof_pos_scale = float(config["dof_pos_scale"])
        self.dof_vel_scale = float(config["dof_vel_scale"])
        self.ang_vel_scale = float(config["ang_vel_scale"])
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.upper_hold_targets: list[float] = []
        self.update_count = 0

    def reset(self, data: Any) -> None:
        import numpy as np

        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.upper_hold_targets = [
            float(data.qpos[qpos_addr]) for qpos_addr in self.upper_hold_qpos_addrs
        ]
        self.update_count = 0
        self.apply(data=data)

    def _gravity_orientation(self, quaternion: Any) -> Any:
        import numpy as np

        qw, qx, qy, qz = quaternion
        return np.array(
            [
                2 * (-qz * qx + qw * qy),
                -2 * (qz * qy + qw * qx),
                1 - 2 * (qw * qw + qz * qz),
            ],
            dtype=np.float32,
        )

    def apply(self, *, data: Any) -> None:
        import numpy as np

        if self.actuator_output_mode == "position_target":
            for action_index, actuator_id in enumerate(self.leg_actuator_ids):
                target = float(self.target_dof_pos[action_index])
                try:
                    low, high = self.model.actuator_ctrlrange[actuator_id]
                    if float(high) > float(low):
                        target = float(np.clip(target, float(low), float(high)))
                except Exception:
                    pass
                data.ctrl[actuator_id] = target
        else:
            q = np.array([float(data.qpos[addr]) for addr in self.leg_qpos_addrs], dtype=np.float32)
            dq = np.array([float(data.qvel[addr]) for addr in self.leg_qvel_addrs], dtype=np.float32)
            tau = (self.target_dof_pos - q) * self.kps + (np.zeros_like(self.kds) - dq) * self.kds
            for action_index, actuator_id in enumerate(self.leg_actuator_ids):
                data.ctrl[actuator_id] = float(tau[action_index])
        for actuator_id, target in zip(self.upper_hold_actuator_ids, self.upper_hold_targets):
            data.ctrl[actuator_id] = float(target)

    def step(
        self,
        *,
        data: Any,
        step: int,
        command_xyz: Sequence[float],
    ) -> dict[str, Any] | None:
        import numpy as np
        import torch

        command = np.array([float(value) for value in command_xyz[:3]], dtype=np.float32)
        update_row: dict[str, Any] | None = None
        if int(step) % max(1, self.control_decimation) == 0:
            qj = np.array([float(data.qpos[addr]) for addr in self.leg_qpos_addrs], dtype=np.float32)
            dqj = np.array([float(data.qvel[addr]) for addr in self.leg_qvel_addrs], dtype=np.float32)
            quat = np.array(
                [float(value) for value in data.qpos[self.root_qpos + 3 : self.root_qpos + 7]],
                dtype=np.float32,
            )
            omega = np.array(
                [float(value) for value in data.qvel[self.root_dof + 3 : self.root_dof + 6]],
                dtype=np.float32,
            )
            qj_scaled = (qj - self.default_angles) * self.dof_pos_scale
            dqj_scaled = dqj * self.dof_vel_scale
            gravity_orientation = self._gravity_orientation(quat)
            omega_scaled = omega * self.ang_vel_scale
            obs = np.zeros(self.num_obs, dtype=np.float32)
            period = 0.8
            sim_time = float(data.time)
            phase_value = sim_time % period / period
            obs[:3] = omega_scaled
            obs[3:6] = gravity_orientation
            obs[6:9] = command * self.cmd_scale
            obs[9 : 9 + self.num_actions] = qj_scaled
            obs[9 + self.num_actions : 9 + 2 * self.num_actions] = dqj_scaled
            obs[9 + 2 * self.num_actions : 9 + 3 * self.num_actions] = self.action
            obs[9 + 3 * self.num_actions : 9 + 3 * self.num_actions + 2] = np.array(
                [np.sin(2 * np.pi * phase_value), np.cos(2 * np.pi * phase_value)],
                dtype=np.float32,
            )
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                self.action = (
                    self.policy(obs_tensor).detach().cpu().numpy().squeeze().astype(np.float32)
                )
            self.target_dof_pos = self.action * self.action_scale + self.default_angles
            self.update_count += 1
            update_row = {
                "schema_version": "unitree_rl_gym_same_scene_controller_update.v1",
                "step": int(step),
                "sim_time_s": round(float(data.time), 9),
                "command_xyz": [round(float(value), 6) for value in command],
                "target_dof_pos": [round(float(value), 6) for value in self.target_dof_pos],
                "action": [round(float(value), 6) for value in self.action],
            }
        self.apply(data=data)
        return update_row


def _same_scene_unitree_controller_manifest(
    *,
    generated_at: str,
    status: str,
    selected_root: Path | None,
    blockers: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "same_scene_unitree_rl_gym_controller_backend.v1",
        "generated_at": generated_at,
        "status": status,
        "controller_backend": "unitree_rl_gym",
        "backend_id": UNITREE_RL_GYM_SAME_SCENE_BACKEND_ID,
        "selected_unitree_rl_gym_root": str(selected_root) if selected_root else None,
        "same_scene_controller_backend_integrated": status == "ready",
        "official_unitree_controller_used": status == "ready",
        "realistic_navigation_policy_used_for_endpoint_rollouts": status == "ready",
        "freejoint_proxy_used_for_endpoint_rollouts": status != "ready",
        "controller_command_limits": dict(UNITREE_RL_GYM_CONTROLLER_COMMAND_LIMITS),
        "balanced_walking_controller_proven": False,
        "blockers": list(blockers),
        "claim_boundary": {
            "same_scene_controller_loaded_is_not_physical_robot_readiness": True,
            "balanced_walking_requires_no_fall_rollout_evidence": True,
            "lower_body_policy_does_not_prove_dexterous_hand_vla": True,
        },
        **dict(extra or {}),
    }


def _create_same_scene_unitree_rl_gym_controller(
    *,
    model: Any,
    data: Any,
    mujoco_module: Any,
    root_qpos: int,
    root_dof: int,
    generated_at: str,
    unitree_rl_gym_root: Path | None,
    navigation_discovery: Mapping[str, Any],
    enabled: bool,
) -> tuple[_SameSceneUnitreeRLGymController | None, dict[str, Any]]:
    if not enabled:
        return None, _same_scene_unitree_controller_manifest(
            generated_at=generated_at,
            status="skipped",
            selected_root=None,
            blockers=["same_scene_unitree_controller_backend_not_selected"],
        )
    selected_root = _select_unitree_rl_gym_root(
        explicit_root=unitree_rl_gym_root,
        discovery=navigation_discovery,
    )
    if selected_root is None:
        return None, _same_scene_unitree_controller_manifest(
            generated_at=generated_at,
            status="blocked",
            selected_root=None,
            blockers=["blocked_missing_unitree_rl_gym_root_or_required_files"],
        )
    try:
        import torch

        from .unitree_g1_policy_execution import _read_yaml, _sha256

        config_path = selected_root / "deploy" / "deploy_mujoco" / "configs" / "g1.yaml"
        config = _read_yaml(config_path)
        policy_path = Path(
            str(config["policy_path"]).replace("{LEGGED_GYM_ROOT_DIR}", str(selected_root))
        )
        policy = torch.jit.load(str(policy_path), map_location="cpu")
        policy.eval()
    except Exception as exc:
        return None, _same_scene_unitree_controller_manifest(
            generated_at=generated_at,
            status="blocked",
            selected_root=selected_root,
            blockers=["blocked_unitree_rl_gym_policy_or_config_load_failed"],
            extra={"error_type": type(exc).__name__, "error": str(exc)[:500]},
        )

    leg_actuator_ids: list[int] = []
    leg_qpos_addrs: list[int] = []
    leg_qvel_addrs: list[int] = []
    missing: list[str] = []
    for name in UNITREE_RL_GYM_LEG_JOINT_NAMES:
        actuator_id = _actuator_id_by_name(mujoco_module, model, name)
        joint_id = _joint_id_by_name(mujoco_module, model, name)
        if actuator_id < 0 or joint_id < 0:
            missing.append(name)
            continue
        leg_actuator_ids.append(actuator_id)
        leg_qpos_addrs.append(int(model.jnt_qposadr[joint_id]))
        leg_qvel_addrs.append(int(model.jnt_dofadr[joint_id]))
    if missing:
        return None, _same_scene_unitree_controller_manifest(
            generated_at=generated_at,
            status="blocked",
            selected_root=selected_root,
            blockers=["blocked_same_scene_missing_required_leg_actuators_or_joints"],
            extra={"missing_joint_or_actuator_names": missing},
        )
    if int(config.get("num_actions", 0)) != len(UNITREE_RL_GYM_LEG_JOINT_NAMES):
        return None, _same_scene_unitree_controller_manifest(
            generated_at=generated_at,
            status="blocked",
            selected_root=selected_root,
            blockers=["blocked_unitree_policy_action_dimension_mismatch"],
            extra={
                "policy_num_actions": int(config.get("num_actions", 0)),
                "expected_leg_action_count": len(UNITREE_RL_GYM_LEG_JOINT_NAMES),
            },
        )

    actuator_output_mode = "torque"
    try:
        gain_values = [float(model.actuator_gainprm[actuator_id][0]) for actuator_id in leg_actuator_ids]
        bias_values = [float(model.actuator_biasprm[actuator_id][1]) for actuator_id in leg_actuator_ids]
        if all(abs(bias + gain) < max(1.0, abs(gain) * 0.05) and gain > 10.0 for gain, bias in zip(gain_values, bias_values)):
            actuator_output_mode = "position_target"
    except Exception:
        actuator_output_mode = "torque"

    upper_hold_actuator_ids: list[int] = []
    upper_hold_qpos_addrs: list[int] = []
    leg_set = set(leg_actuator_ids)
    for actuator_index in range(int(model.nu)):
        if actuator_index in leg_set:
            continue
        joint_id = int(model.actuator_trnid[actuator_index][0])
        if joint_id < 0:
            continue
        upper_hold_actuator_ids.append(actuator_index)
        upper_hold_qpos_addrs.append(int(model.jnt_qposadr[joint_id]))
    controller = _SameSceneUnitreeRLGymController(
        model=model,
        mujoco_module=mujoco_module,
        root_qpos=root_qpos,
        root_dof=root_dof,
        selected_root=selected_root,
        config=config,
        policy=policy,
        policy_path=policy_path,
        config_path=config_path,
        leg_actuator_ids=leg_actuator_ids,
        leg_qpos_addrs=leg_qpos_addrs,
        leg_qvel_addrs=leg_qvel_addrs,
        upper_hold_actuator_ids=upper_hold_actuator_ids,
        upper_hold_qpos_addrs=upper_hold_qpos_addrs,
        actuator_output_mode=actuator_output_mode,
    )
    return controller, _same_scene_unitree_controller_manifest(
        generated_at=generated_at,
        status="ready",
        selected_root=selected_root,
        blockers=[],
        extra={
            "config_path": str(config_path),
            "policy_path": str(policy_path),
            "config_sha256": _sha256(config_path),
            "policy_sha256": _sha256(policy_path),
            "leg_joint_names": list(UNITREE_RL_GYM_LEG_JOINT_NAMES),
            "leg_actuator_ids": leg_actuator_ids,
            "upper_hold_actuator_count": len(upper_hold_actuator_ids),
            "actuator_output_mode": actuator_output_mode,
            "control_decimation": int(config["control_decimation"]),
            "simulation_dt": float(config["simulation_dt"]),
            "policy_num_obs": int(config["num_obs"]),
            "policy_num_actions": int(config["num_actions"]),
        },
    )


def _camera_for(mujoco_module: Any, camera_id: str, root_position: Sequence[float], yaw: float) -> Any:
    camera = mujoco_module.MjvCamera()
    camera.type = mujoco_module.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [float(root_position[0]), float(root_position[1]), float(root_position[2]) + 0.55]
    if camera_id == "overhead":
        camera.distance = 4.8
        camera.azimuth = 0.0
        camera.elevation = -89.0
    elif camera_id == "robot_follow":
        camera.distance = 2.0
        camera.azimuth = math.degrees(yaw) + 180.0
        camera.elevation = -14.0
    else:
        camera.distance = 3.2
        camera.azimuth = 220.0
        camera.elevation = -18.0
    return camera


def _write_video_from_frames(*, frames_dir: Path, output_path: Path, fps: int) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {"path": str(output_path), "status": "blocked", "reason": "ffmpeg_unavailable"}
    ensure_dir(output_path.parent)
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    return {
        "path": str(output_path),
        "status": "complete"
        if result.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0
        else "blocked",
        "returncode": result.returncode,
        "stderr": (result.stderr or "")[-800:],
        "size_bytes": output_path.stat().st_size if output_path.is_file() else 0,
    }


def _episode_frame_steps(
    *,
    steps_per_episode: int,
    render_frame_count: int,
    video_frame_stride_steps: int,
) -> tuple[list[int], str, int]:
    steps = max(1, int(steps_per_episode))
    if int(render_frame_count) > 0:
        if int(render_frame_count) <= 1:
            return [0], "fixed_sample_count", steps
        stride = max(1, steps // max(1, int(render_frame_count) - 1))
        frame_steps = sorted(
            {min(steps - 1, step * stride) for step in range(int(render_frame_count))}
        )
        return frame_steps, "fixed_sample_count", stride
    stride = max(1, int(video_frame_stride_steps))
    frame_steps = list(range(0, steps, stride))
    if frame_steps[-1] != steps - 1:
        frame_steps.append(steps - 1)
    return frame_steps, "full_episode_stride", stride


def _video_output_fps(*, requested_fps: int, timestep: float, stride_steps: int) -> int:
    if int(requested_fps) > 0:
        return int(requested_fps)
    sim_seconds_per_frame = max(float(timestep) * max(1, int(stride_steps)), 1e-6)
    return max(1, int(round(1.0 / sim_seconds_per_frame)))


def _video_timing_contract(
    *,
    requested_fps: int,
    encoded_fps: int,
    timestep: float,
    stride_steps: int,
    physics_frame_count: int,
    encoded_frame_count: int,
) -> dict[str, Any]:
    sim_seconds_per_frame = max(float(timestep) * max(1, int(stride_steps)), 1e-9)
    expected_sim_time_fps = max(1, int(round(1.0 / sim_seconds_per_frame)))
    physics_duration_s = max(0.0, float(physics_frame_count) * sim_seconds_per_frame)
    encoded_duration_s = (
        max(0.0, float(encoded_frame_count) / max(1, int(encoded_fps)))
        if encoded_frame_count
        else 0.0
    )
    playback_scale = encoded_duration_s / physics_duration_s if physics_duration_s > 0 else None
    fixed_fps_forced = int(requested_fps) > 0
    slow_motion = bool(
        fixed_fps_forced
        and playback_scale is not None
        and playback_scale > 1.2
    )
    return {
        "requested_fps": int(requested_fps),
        "encoded_video_fps": int(encoded_fps),
        "expected_sim_time_fps_for_stride": expected_sim_time_fps,
        "sim_seconds_per_rendered_frame": round(sim_seconds_per_frame, 9),
        "physics_duration_s": round(physics_duration_s, 9),
        "encoded_duration_estimate_s": round(encoded_duration_s, 9),
        "playback_time_scale_vs_sim": round(playback_scale, 6)
        if playback_scale is not None
        else None,
        "fps_zero_used_for_sim_time_playback": not fixed_fps_forced,
        "fixed_fps_forced_by_user": fixed_fps_forced,
        "video_playback_may_look_slow_motion": slow_motion,
        "slow_motion_reason": (
            "fixed_fps_lower_than_mujoco_step_rate_for_captured_frames"
            if slow_motion
            else None
        ),
    }


def _ffprobe_video(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {"path": str(path), "status": "not_checked", "reason": "ffprobe_unavailable"}
    if not path.is_file():
        return {"path": str(path), "status": "not_checked", "reason": "missing_video"}
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,duration,width,height",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return {"path": str(path), "status": "blocked", "stderr": (result.stderr or "")[-500:]}
    payload = json.loads(result.stdout or "{}")
    stream = (payload.get("streams") or [{}])[0]
    frame_count = stream.get("nb_frames")
    frame_count_int = int(frame_count) if str(frame_count).isdigit() else None
    duration = float(stream.get("duration") or 0.0)
    return {
        "path": str(path),
        "status": "complete" if duration > 0 and (frame_count_int or 0) > 0 else "blocked",
        "duration_s": duration,
        "frame_count": frame_count_int,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
    }


def _counts_by_key(attempts: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        value = str(attempt.get(key) or "unknown")
        row = grouped.setdefault(
            value,
            {"id": value, "attempted": 0, "passed": 0, "failed": 0, "blocked": 0},
        )
        row["attempted"] += 1
        if attempt.get("status") == "blocked":
            row["blocked"] += 1
        elif attempt.get("success") is True:
            row["passed"] += 1
        else:
            row["failed"] += 1
    return [grouped[key] for key in sorted(grouped)]


def _score_attempt(
    *,
    run: Mapping[str, Any],
    final_error_m: float,
    final_speed_mps: float,
    fall_count: int,
    unsafe_collision_count: int,
    object_contact_count: int,
    object_displacement_m: float,
    rejected_action_count: int,
    action_types: Sequence[str],
) -> tuple[str, bool, list[str], dict[str, Any]]:
    task_id = str(run.get("task_id") or "")
    blocker_labels: list[str] = []
    if fall_count:
        blocker_labels.append("failure_g1_fell_in_mujoco")
    if unsafe_collision_count:
        blocker_labels.append("failure_unsafe_obstacle_contact")
    if rejected_action_count and run.get("spawn_id") == "blocked_or_occluded" and task_id == "inspect_target":
        return (
            "blocked",
            False,
            ["blocked_rejected_policy_action", *blocker_labels],
            {
                "navigation_success": False,
                "route_safety": unsafe_collision_count == 0,
                "action_validity": False,
                "blocked_reason": "fixture_malformed_action_preserved",
            },
        )
    route_safety = fall_count == 0 and unsafe_collision_count == 0
    navigation_success = final_error_m <= SAFETY_LIMITS["goal_tolerance_m"]
    stopped = final_speed_mps <= SAFETY_LIMITS["stop_speed_mps"]
    contact_success = (
        object_contact_count > 0
        and object_displacement_m >= SAFETY_LIMITS["object_displacement_success_m"]
    )
    if task_id == "inspect_target":
        success = route_safety and any(action in {"inspect_look", "look"} for action in action_types)
    elif task_id == "approach_target":
        success = route_safety and navigation_success
    elif task_id == "route_around_obstruction":
        success = route_safety and final_error_m <= 0.45
    elif task_id == "contact_or_push_light_object":
        success = fall_count == 0 and contact_success
    elif task_id == "stop_at_goal_and_report":
        success = route_safety and final_error_m <= 0.42 and stopped
    else:
        success = route_safety and navigation_success
    labels = list(blocker_labels)
    if not navigation_success and task_id in {"approach_target", "route_around_obstruction", "stop_at_goal_and_report"}:
        labels.append("failure_goal_not_reached")
    if task_id == "contact_or_push_light_object" and not contact_success:
        labels.append("failure_object_contact_or_displacement_not_validated")
    if task_id == "stop_at_goal_and_report" and not stopped:
        labels.append("failure_stop_not_validated")
    if rejected_action_count:
        labels.append("failure_policy_action_rejected")
    return (
        "completed" if success else "failed",
        success,
        labels,
        {
            "navigation_success": navigation_success,
            "task_progress": max(0.0, 1.0 - final_error_m),
            "route_safety": route_safety,
            "contact_collision_correctness": unsafe_collision_count == 0,
            "object_displacement_success": contact_success,
            "stopped_at_goal": stopped,
            "action_validity": rejected_action_count == 0,
        },
    )


def run_mujoco_g1_wam_vla_policy_endpoint_eval(
    *,
    job_dir: Path | None = None,
    job_root: Path | None = None,
    g1_model_root: Path | None = None,
    task_filter: Sequence[str] | None = None,
    spawn_filter: Sequence[str] | None = None,
    max_tasks: int | None = None,
    max_spawns: int | None = None,
    steps_per_episode: int = DEFAULT_STEPS_PER_EPISODE,
    policy_interval_steps: int = 20,
    render: bool = True,
    render_frame_count: int = 0,
    video_frame_stride_steps: int = DEFAULT_VIDEO_FRAME_STRIDE_STEPS,
    extend_terminal_frame_for_review: bool = DEFAULT_EXTEND_TERMINAL_FRAME_FOR_REVIEW,
    rendered_video_episode_limit: int | None = DEFAULT_RENDERED_VIDEO_EPISODE_LIMIT,
    video_cameras: Sequence[str] | None = None,
    fps: int = DEFAULT_REVIEW_VIDEO_FPS,
    endpoint_timeout_seconds: float = 8.0,
    max_contact_trace_rows: int = DEFAULT_MAX_CONTACT_TRACE_ROWS,
    allow_fetch_g1_assets: bool = False,
    menagerie_ref: str = DEFAULT_MENAGERIE_REF,
    unitree_rl_gym_root: Path | None = None,
    run_official_unitree_controller_sidecar: bool = False,
    unitree_controller_sidecar_steps: int = 400,
    unitree_controller_sidecar_command_xyz: Sequence[float] | None = None,
    run_unitree_controller_replay_from_endpoint_actions: bool = False,
    unitree_controller_replay_steps: int = 400,
    controller_backend: str = DEFAULT_CONTROLLER_BACKEND,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or utc_now_iso()
    if controller_backend not in CONTROLLER_BACKENDS:
        raise ValueError(
            f"controller_backend must be one of {', '.join(CONTROLLER_BACKENDS)}"
        )
    if job_dir is None:
        root = job_root or (_repo_root() / "robot_eval_jobs")
        job_dir = root / f"mujoco_g1_wam_vla_policy_endpoint_eval_{_utc_timestamp_for_path()}"
    job_dir = Path(job_dir).resolve()
    ensure_dir(job_dir)
    job_id = job_dir.name
    phase_rows: list[dict[str, Any]] = []

    def phase(phase_id: str, status: str, **extra: Any) -> None:
        phase_rows.append(
            {
                "generated_at": utc_now_iso(),
                "phase_id": phase_id,
                "status": status,
                **extra,
            }
        )
        _write_jsonl(job_dir / "mujoco_execution_phase_log.jsonl", phase_rows)

    phase("job_created", "completed", job_dir=str(job_dir))
    endpoint_discovery, runtime_discovery, auth_manifest, probe_results = discover_policy_runtime(
        generated_at=generated_at
    )
    selected_runtime = selected_endpoint(endpoint_discovery)
    endpoint_health_probe = _probe_endpoint_health(
        endpoint_row=selected_runtime,
        timeout_seconds=min(float(endpoint_timeout_seconds), 2.0),
    )
    probe_results = {
        **probe_results,
        "endpoint_health_probe": endpoint_health_probe,
        "endpoint_health_probe_performed": endpoint_health_probe.get(
            "endpoint_health_probe_performed"
        ),
    }
    write_json(job_dir / "policy_endpoint_discovery.json", endpoint_discovery)
    write_json(job_dir / "wam_vla_runtime_discovery.json", runtime_discovery)
    write_json(job_dir / "policy_endpoint_auth_manifest.json", auth_manifest)
    write_json(job_dir / "policy_endpoint_probe_results.json", probe_results)
    write_json(
        job_dir / "policy_endpoint_server_manifest.json",
        build_policy_endpoint_server_manifest(
            generated_at=generated_at,
            selected_runtime=selected_runtime,
            health_probe=endpoint_health_probe,
        ),
    )
    write_json(
        job_dir / "policy_model_candidate_matrix.json",
        build_policy_model_candidate_matrix(generated_at=generated_at),
    )
    write_json(
        job_dir / "policy_model_truth_boundary.json",
        build_policy_model_truth_boundary(generated_at=generated_at),
    )
    write_json(
        job_dir / "policy_command_adapter_manifest.json",
        build_policy_command_adapter_manifest(generated_at=generated_at),
    )
    write_json(
        job_dir / "policy_runtime_truth_boundary.json",
        {
            "schema_version": "policy_runtime_truth_boundary.v1",
            "generated_at": generated_at,
            "simulator_only": True,
            "real_wam_vla_claim_requires_endpoint_response": True,
            "reference_fixture_policy_is_not_real_wam_vla": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "official_policy_execution_proven": False,
            "isaac_runtime_proven_by_this_lane": False,
        },
    )
    write_json(job_dir / "wam_vla_observation_packet_schema.json", _observation_schema())
    write_json(job_dir / "wam_vla_action_schema.json", _action_schema())

    try:
        import mujoco  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent.
        runtime = {
            "schema_version": "mujoco_runtime_discovery.v1",
            "generated_at": generated_at,
            "status": "blocked",
            "mujoco_runtime_available": False,
            "blockers": ["mujoco_import_failed"],
            "error": str(exc),
        }
        write_json(job_dir / "mujoco_runtime_discovery.json", runtime)
        summary = {
            "schema_version": LANE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "job_dir": str(job_dir),
            "mujoco_runtime_available": False,
            "unitree_g1_loaded_in_mujoco": False,
            "blockers": ["mujoco_import_failed"],
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
        }
        write_json(job_dir / "mujoco_g1_wam_vla_policy_endpoint_eval_summary.json", summary)
        return summary

    runtime = {
        "schema_version": "mujoco_runtime_discovery.v1",
        "generated_at": generated_at,
        "status": "completed",
        "mujoco_runtime_available": True,
        "mujoco_version": getattr(mujoco, "__version__", None),
        "mujoco_module_file": getattr(mujoco, "__file__", None),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "renderer_requested": bool(render),
        "ffmpeg_available": bool(shutil.which("ffmpeg")),
        "ffprobe_available": bool(shutil.which("ffprobe")),
    }
    write_json(job_dir / "mujoco_runtime_discovery.json", runtime)
    phase("mujoco_runtime_discovered", "completed", mujoco_version=runtime["mujoco_version"])

    g1_root = _resolve_g1_model_root(
        explicit_root=g1_model_root,
        capture_root=job_dir,
        allow_fetch=allow_fetch_g1_assets,
        menagerie_ref=menagerie_ref,
    )
    g1_abs_xml = job_dir / "generated_mujoco" / "unitree_g1_absolute_meshes.xml"
    _write_g1_xml_with_absolute_meshes(g1_root / "g1.xml", g1_abs_xml)
    asset_manifest = {
        "schema_version": "unitree_g1_mujoco_asset_source_manifest.v1",
        "generated_at": generated_at,
        **_asset_source_manifest(g1_root),
        "resolved_g1_xml": str(g1_root / "g1.xml"),
        "generated_absolute_mesh_xml": str(g1_abs_xml),
        "unitree_g1_mujoco_model_source": "google_deepmind_mujoco_menagerie",
    }
    write_json(job_dir / "unitree_g1_mujoco_asset_source_manifest.json", asset_manifest)
    scene_manifest = _write_scene_xml(
        g1_xml=g1_abs_xml,
        output_xml=job_dir / "generated_mujoco" / "mujoco_g1_wam_vla_eval_scene.xml",
    )
    write_json(job_dir / "mujoco_scene_manifest.json", scene_manifest)
    navigation_discovery = discover_realistic_navigation_policy(generated_at=generated_at)
    write_json(job_dir / "realistic_navigation_policy_discovery.json", navigation_discovery)
    official_controller_sidecar = _run_official_unitree_controller_sidecar(
        job_dir=job_dir,
        job_id=job_id,
        generated_at=generated_at,
        unitree_rl_gym_root=unitree_rl_gym_root,
        navigation_discovery=navigation_discovery,
        enabled=run_official_unitree_controller_sidecar,
        max_steps=unitree_controller_sidecar_steps,
        command_xyz=unitree_controller_sidecar_command_xyz,
    )
    write_json(
        job_dir / "official_unitree_controller_sidecar_manifest.json",
        official_controller_sidecar,
    )
    official_controller_proven = bool(
        official_controller_sidecar.get("official_unitree_controller_used")
    )
    write_json(
        job_dir / "controller_truth_boundary.json",
        {
            "schema_version": "controller_truth_boundary.v1",
            "generated_at": generated_at,
            "status": "pending_execution",
            "controller_kind": None,
            "realistic_navigation_policy_used": None,
            "realistic_navigation_policy_used_for_endpoint_rollouts": None,
            "official_unitree_controller_sidecar_status": official_controller_sidecar.get("status"),
            "navigation_policy_kind": None,
            "final_execution_truth_pending": True,
            "final_execution_truth_written_after_summary": True,
            "realistic_unitree_walking_policy_required_for_navigation_claim": True,
            "continuous_mujoco_stepping": True,
            "root_pose_teleport_success_used": False,
            "official_unitree_controller_used": None,
            "official_policy_execution_proven": None,
            "training_grade_policy_rollout_proven": False,
            "balanced_walking_controller_proven": None,
            "official_unitree_controller_sidecar_command_xyz": official_controller_sidecar.get(
                "command_xyz"
            ),
            "freejoint_proxy_used": None,
            "blockers": navigation_discovery.get("blockers", []),
            "proof_boundary": (
                "Provisional artifact only. Final controller execution truth is written "
                "after episode rollout completion."
            ),
        },
    )

    matrix = build_scenario_eval_matrix(
        job_id=job_id,
        generated_at=generated_at,
        task_filter=task_filter,
        spawn_filter=spawn_filter,
        max_tasks=max_tasks,
        max_spawns=max_spawns,
    )
    write_json(job_dir / "scenario_eval_matrix.json", matrix)
    write_json(
        job_dir / "episode_spec_manifest.json",
        {
            "schema_version": "mujoco_g1_wam_vla_episode_spec_manifest.v1",
            "generated_at": generated_at,
            "job_id": job_id,
            "episode_count": matrix["scenario_eval_run_count"],
            "episodes": [
                {
                    "episode_id": run["episode_id"],
                    "scenario_eval_run_id": run["scenario_eval_run_id"],
                    "task_id": run["task_id"],
                    "spawn_id": run["spawn_id"],
                    "steps_per_episode": steps_per_episode,
                    "policy_interval_steps": policy_interval_steps,
                    "expected_sim_duration_seconds": round(float(steps_per_episode) * float(timestep), 6)
                    if "timestep" in locals()
                    else None,
                    "simulator": "mujoco",
                    "robot_profile_id": ROBOT_PROFILE_ID,
                }
                for run in matrix["runs"]
            ],
        },
    )

    model = mujoco.MjModel.from_xml_path(str(scene_manifest["scene_xml"]))
    data = mujoco.MjData(model)
    root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if root_joint_id < 0:
        raise RuntimeError("Unitree G1 floating_base_joint not found in MuJoCo model")
    root_qpos = int(model.jnt_qposadr[root_joint_id])
    root_dof = int(model.jnt_dofadr[root_joint_id])
    object_joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "blueprint_light_object_freejoint",
    )
    object_qpos = int(model.jnt_qposadr[object_joint_id]) if object_joint_id >= 0 else None
    stand_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    base_qpos = model.key_qpos[stand_key_id].copy() if stand_key_id >= 0 else model.qpos0.copy()
    root_z = float(base_qpos[root_qpos + 2])
    object_initial_pose = [0.36, -0.65, 0.27, 1.0, 0.0, 0.0, 0.0]
    timestep = float(model.opt.timestep)
    contact_metadata = _build_contact_metadata(model, mujoco)
    selected_unitree_root_for_auto = _select_unitree_rl_gym_root(
        explicit_root=unitree_rl_gym_root,
        discovery=navigation_discovery,
    )
    auto_controller_selection = controller_backend == "auto"
    controller_backend_for_setup = (
        "unitree_rl_gym"
        if auto_controller_selection and selected_unitree_root_for_auto is not None
        else controller_backend
    )
    if controller_backend_for_setup == "auto":
        controller_backend_for_setup = "freejoint_proxy"
    same_scene_controller, same_scene_controller_manifest = (
        _create_same_scene_unitree_rl_gym_controller(
            model=model,
            data=data,
            mujoco_module=mujoco,
            root_qpos=root_qpos,
            root_dof=root_dof,
            generated_at=generated_at,
            unitree_rl_gym_root=unitree_rl_gym_root,
            navigation_discovery=navigation_discovery,
            enabled=controller_backend_for_setup == "unitree_rl_gym",
        )
    )
    same_scene_controller_manifest = {
        **same_scene_controller_manifest,
        "requested_controller_backend": controller_backend,
        "resolved_controller_backend_for_setup": controller_backend_for_setup,
        "auto_controller_selection": auto_controller_selection,
        "auto_selected_unitree_rl_gym_root": str(selected_unitree_root_for_auto)
        if selected_unitree_root_for_auto is not None
        else None,
        "auto_selection_fell_back_to_freejoint_proxy": bool(
            auto_controller_selection and same_scene_controller is None
        ),
    }
    write_json(
        job_dir / "same_scene_unitree_controller_backend_manifest.json",
        same_scene_controller_manifest,
    )
    same_scene_controller_ready = same_scene_controller is not None
    if (
        controller_backend == "unitree_rl_gym"
        and not same_scene_controller_ready
    ):
        blocked_summary = {
            "schema_version": LANE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "job_id": job_id,
            "job_dir": str(job_dir),
            "mujoco_runtime_available": True,
            "unitree_g1_mujoco_model_source": asset_manifest["unitree_g1_mujoco_model_source"],
            "unitree_g1_mujoco_model_path": str(g1_root / "g1.xml"),
            "unitree_g1_loaded_in_mujoco": True,
            "requested_controller_backend": controller_backend,
            "resolved_controller_backend_for_setup": controller_backend_for_setup,
            "controller_backend": controller_backend_for_setup,
            "same_scene_unitree_controller_backend_integrated": False,
            "realistic_navigation_policy_used": False,
            "realistic_navigation_policy_used_for_endpoint_rollouts": False,
            "official_unitree_controller_used": False,
            "balanced_walking_controller_proven": False,
            "freejoint_proxy_used": False,
            "blockers": same_scene_controller_manifest.get("blockers", []),
            "attempted_episode_count": 0,
            "artifact_paths": {
                "same_scene_unitree_controller_backend_manifest": str(
                    job_dir / "same_scene_unitree_controller_backend_manifest.json"
                ),
                "realistic_navigation_policy_discovery": str(
                    job_dir / "realistic_navigation_policy_discovery.json"
                ),
                "mujoco_scene_manifest": str(job_dir / "mujoco_scene_manifest.json"),
            },
        }
        write_json(job_dir / "policy_evaluation_summary.json", blocked_summary)
        write_json(job_dir / "mujoco_g1_wam_vla_policy_endpoint_eval_summary.json", blocked_summary)
        phase("job_blocked", "blocked", blockers=blocked_summary["blockers"])
        return blocked_summary
    selected_controller_backend = (
        "unitree_rl_gym" if same_scene_controller_ready else "freejoint_proxy"
    )
    rendered_video_episode_cap = (
        None
        if rendered_video_episode_limit is None or int(rendered_video_episode_limit) <= 0
        else int(rendered_video_episode_limit)
    )
    selected_video_cameras = tuple(video_cameras or DEFAULT_VIDEO_CAMERAS)
    selected_review_playback_fps = max(1, int(fps) if int(fps) > 0 else 1)
    fixture_policy_used = selected_runtime is None
    endpoint_policy_valid_actions = 0
    endpoint_policy_decisions = 0

    renderer = None
    image_module = None
    if render:
        try:
            from PIL import Image

            renderer = mujoco.Renderer(model, height=360, width=640)
            image_module = Image
        except Exception as exc:  # pragma: no cover - renderer availability.
            renderer = None
            image_module = None
            write_json(
                job_dir / "blocked_video_renderer_unavailable.json",
                {
                    "schema_version": "blocked_video_renderer_unavailable.v1",
                    "generated_at": generated_at,
                    "status": "blocked",
                    "blockers": ["video_renderer_unavailable"],
                    "error": str(exc),
                },
            )

    locomotion_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    rejected_actions: list[dict[str, Any]] = []
    endpoint_attempt_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    manipulation_contacts: list[dict[str, Any]] = []
    contact_trace_row_limit = max(0, int(max_contact_trace_rows))
    contact_trace_total_count = 0
    contact_trace_dropped_count = 0
    contact_trace_truncated = False
    contact_aggregate_counts = {
        "floor_contact_count": 0,
        "object_contact_count": 0,
        "obstacle_contact_count": 0,
        "left_foot_contact_count": 0,
        "right_foot_contact_count": 0,
    }
    object_motion_rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    failure_labels: list[dict[str, Any]] = []
    blocked_attempts: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    poster_rows: list[dict[str, Any]] = []
    ffprobe_rows: list[dict[str, Any]] = []
    same_scene_controller_rows: list[dict[str, Any]] = []

    try:
        for episode_index, run in enumerate(matrix["runs"], start=1):
            phase("episode_started", "running", episode_id=run["episode_id"])
            data.qpos[:] = base_qpos
            data.qvel[:] = 0.0
            data.time = 0.0
            spawn_pose = list(run["spawn_pose"])
            data.qpos[root_qpos : root_qpos + 3] = [
                float(spawn_pose[0]),
                float(spawn_pose[1]),
                root_z,
            ]
            data.qpos[root_qpos + 3 : root_qpos + 7] = _yaw_quat(float(run.get("spawn_yaw_rad") or 0.0))
            if object_qpos is not None:
                data.qpos[object_qpos : object_qpos + 7] = object_initial_pose
                data.qvel[int(model.jnt_dofadr[object_joint_id]) : int(model.jnt_dofadr[object_joint_id]) + 6] = 0.0
            if same_scene_controller is not None:
                same_scene_controller.reset(data)
            else:
                _set_joint_position_holds(model, data)
            mujoco.mj_forward(model, data)
            initial_root = [float(value) for value in data.qpos[root_qpos : root_qpos + 3]]
            initial_object = _object_pose(data, object_qpos)
            active_action = {
                "action_type": "stop",
                "vx_mps": 0.0,
                "vy_mps": 0.0,
                "yaw_rate_rad_s": 0.0,
                "source": "initial_stop",
                "normalization_status": "accepted",
            }
            episode_action_types: list[str] = []
            episode_rejected = 0
            episode_contacts: list[dict[str, Any]] = []
            fall_count = 0
            unsafe_collision_count = 0
            object_contact_count = 0
            frame_steps: set[int] = set()
            frame_step_list: list[int] = []
            frame_index_by_step: dict[int, int] = {}
            video_render_mode = "not_requested"
            video_render_stride_steps = 0
            video_fps = max(1, int(fps) if int(fps) > 0 else 1)
            render_episode_video = renderer is not None and image_module is not None and (
                rendered_video_episode_cap is None or episode_index <= rendered_video_episode_cap
            )
            if render_episode_video:
                frame_step_list, video_render_mode, video_render_stride_steps = _episode_frame_steps(
                    steps_per_episode=steps_per_episode,
                    render_frame_count=render_frame_count,
                    video_frame_stride_steps=video_frame_stride_steps,
                )
                frame_steps = set(frame_step_list)
                frame_index_by_step = {frame_step: index for index, frame_step in enumerate(frame_step_list)}
                video_fps = _video_output_fps(
                    requested_fps=fps,
                    timestep=timestep,
                    stride_steps=video_render_stride_steps,
                )
                selected_review_playback_fps = video_fps
            for step in range(max(1, int(steps_per_episode))):
                current_contact_state = _contact_state(
                    model,
                    data,
                    mujoco,
                    contact_metadata=contact_metadata,
                    include_force=False,
                    record_limit=DEFAULT_CONTACT_OBSERVATION_RECORD_LIMIT,
                )
                current_contacts = current_contact_state["records"]
                if step % max(1, int(policy_interval_steps)) == 0:
                    observation = _build_observation_packet(
                        model=model,
                        data=data,
                        root_qpos=root_qpos,
                        root_dof=root_dof,
                        object_qpos=object_qpos,
                        run=run,
                        step=step,
                        contacts=current_contacts,
                        contact_summary=current_contact_state,
                        mujoco_version=str(getattr(mujoco, "__version__", "")),
                    )
                    raw_policy_payload, endpoint_meta = _call_endpoint_action(
                        endpoint_row=selected_runtime,
                        observation=observation,
                        timeout_seconds=endpoint_timeout_seconds,
                    )
                    source = "endpoint_policy"
                    if raw_policy_payload is None:
                        if selected_runtime is None:
                            raw_policy_payload = _fixture_policy_action(observation=observation)
                            source = "reference_fixture_policy"
                            endpoint_meta = {
                                **endpoint_meta,
                                "fixture_fallback_used": True,
                                "fixture_policy_id": REFERENCE_FIXTURE_POLICY_ID,
                            }
                        else:
                            raw_policy_payload = {
                                "policy_id": "endpoint_policy_blocked",
                                "action": {
                                    "action_type": "stop",
                                    "report": "endpoint_call_failed_no_fixture_fallback",
                                },
                                "endpoint_failure_preserved": True,
                            }
                            source = "endpoint_policy_failed"
                            endpoint_meta = {
                                **endpoint_meta,
                                "fixture_fallback_used": False,
                                "endpoint_failure_preserved": True,
                            }
                    else:
                        endpoint_policy_decisions += 1
                    normalized, rejected = normalize_policy_action(
                        raw_payload=raw_policy_payload,
                        observation=observation,
                        source=source,
                    )
                    if source == "endpoint_policy" and rejected is None:
                        endpoint_policy_valid_actions += 1
                    active_action = normalized
                    episode_action_types.append(str(normalized["action_type"]))
                    action_record = {
                        "schema_version": "normalized_policy_action.v1",
                        "generated_at": generated_at,
                        "episode_id": run["episode_id"],
                        "scenario_eval_run_id": run["scenario_eval_run_id"],
                        "task_id": run["task_id"],
                        "spawn_id": run["spawn_id"],
                        "step": step,
                        "sim_time_s": round(float(data.time), 9),
                        "source": source,
                        "policy_id": _mapping(raw_policy_payload).get("policy_id")
                        or REFERENCE_FIXTURE_POLICY_ID,
                        "raw_policy_response_redacted": _redact(raw_policy_payload),
                        "normalized_action": normalized,
                        "rejected": rejected is not None,
                    }
                    action_rows.append(action_record)
                    endpoint_attempt_rows.append(
                        {
                            "schema_version": "policy_endpoint_attempt.v1",
                            "generated_at": generated_at,
                            "episode_id": run["episode_id"],
                            "scenario_eval_run_id": run["scenario_eval_run_id"],
                            "task_id": run["task_id"],
                            "spawn_id": run["spawn_id"],
                            "step": step,
                            "source": source,
                            **endpoint_meta,
                            "normalized_action_status": normalized["normalization_status"],
                        }
                    )
                    if rejected is not None:
                        episode_rejected += 1
                        rejected_actions.append(
                            {
                                "schema_version": "rejected_policy_action.v1",
                                "generated_at": generated_at,
                                "episode_id": run["episode_id"],
                                "scenario_eval_run_id": run["scenario_eval_run_id"],
                                "task_id": run["task_id"],
                                "spawn_id": run["spawn_id"],
                                "step": step,
                                **rejected,
                            }
                        )
                if same_scene_controller is not None:
                    safe_controller_command = _unitree_controller_safe_command(active_action)
                    controller_update = same_scene_controller.step(
                        data=data,
                        step=step,
                        command_xyz=safe_controller_command["controller_command_xyz"],
                    )
                    if controller_update is not None:
                        same_scene_controller_rows.append(
                            {
                                **controller_update,
                                **safe_controller_command,
                                "episode_id": run["episode_id"],
                                "scenario_eval_run_id": run["scenario_eval_run_id"],
                                "task_id": run["task_id"],
                                "spawn_id": run["spawn_id"],
                                "active_policy_action_type": active_action.get("action_type"),
                            }
                        )
                else:
                    data.qvel[root_dof + 0] = float(active_action.get("vx_mps") or 0.0)
                    data.qvel[root_dof + 1] = float(active_action.get("vy_mps") or 0.0)
                    data.qvel[root_dof + 5] = float(active_action.get("yaw_rate_rad_s") or 0.0)
                    _set_joint_position_holds(model, data)
                mujoco.mj_step(model, data)
                remaining_contact_trace_rows = max(
                    0, contact_trace_row_limit - len(contact_rows)
                )
                step_contact_state = _contact_state(
                    model,
                    data,
                    mujoco,
                    contact_metadata=contact_metadata,
                    include_force=False,
                    record_limit=remaining_contact_trace_rows,
                )
                contacts = step_contact_state["records"]
                root_position = [float(value) for value in data.qpos[root_qpos : root_qpos + 3]]
                root_quat = [float(value) for value in data.qpos[root_qpos + 3 : root_qpos + 7]]
                root_velocity = [float(value) for value in data.qvel[root_dof : root_dof + 3]]
                step_unsafe = int(step_contact_state.get("obstacle_contact_count") or 0)
                step_object = int(step_contact_state.get("object_contact_count") or 0)
                contact_trace_total_count += int(step_contact_state.get("contact_count") or 0)
                contact_trace_dropped_count += int(
                    step_contact_state.get("dropped_record_count") or 0
                )
                contact_trace_truncated = contact_trace_truncated or bool(
                    step_contact_state.get("records_truncated")
                )
                for key in contact_aggregate_counts:
                    contact_aggregate_counts[key] += int(step_contact_state.get(key) or 0)
                unsafe_collision_count += step_unsafe
                object_contact_count += step_object
                if root_position[2] < SAFETY_LIMITS["fall_root_height_m"]:
                    fall_count += 1
                for contact in contacts:
                    row = {
                        **contact,
                        "episode_id": run["episode_id"],
                        "scenario_eval_run_id": run["scenario_eval_run_id"],
                        "task_id": run["task_id"],
                        "spawn_id": run["spawn_id"],
                        "step": step,
                        "sim_time_s": round(float(data.time), 9),
                    }
                    contact_rows.append(row)
                    episode_contacts.append(row)
                    if row.get("object_contact"):
                        manipulation_contacts.append(row)
                object_state = _object_pose(data, object_qpos)
                if object_state.get("available"):
                    object_motion_rows.append(
                        {
                            "schema_version": "object_motion_trace_row.v1",
                            "episode_id": run["episode_id"],
                            "scenario_eval_run_id": run["scenario_eval_run_id"],
                            "task_id": run["task_id"],
                            "spawn_id": run["spawn_id"],
                            "step": step,
                            "sim_time_s": round(float(data.time), 9),
                            **object_state,
                        }
                    )
                locomotion_rows.append(
                    {
                        "schema_version": "g1_mujoco_locomotion_trace_row.v1",
                        "episode_id": run["episode_id"],
                        "scenario_eval_run_id": run["scenario_eval_run_id"],
                        "task_id": run["task_id"],
                        "spawn_id": run["spawn_id"],
                        "step": step,
                        "sim_time_s": round(float(data.time), 9),
                        "root_position": [round(value, 6) for value in root_position],
                        "root_quaternion_wxyz": [round(value, 6) for value in root_quat],
                        "root_yaw_rad": round(_yaw_from_quat(root_quat), 6),
                        "root_linear_velocity_xyz_mps": [round(value, 6) for value in root_velocity],
                        "active_action": dict(active_action),
                        "controller_backend": selected_controller_backend,
                        "freejoint_proxy_used": selected_controller_backend == "freejoint_proxy",
                        "official_unitree_controller_used": selected_controller_backend == "unitree_rl_gym",
                        "contact_count": int(step_contact_state.get("contact_count") or 0),
                        "sampled_contact_record_count": len(contacts),
                        "contact_trace_truncated": bool(
                            step_contact_state.get("records_truncated")
                        ),
                        "object_contact_count": step_object,
                        "unsafe_obstacle_contact_count": step_unsafe,
                        "fall_detected": root_position[2] < SAFETY_LIMITS["fall_root_height_m"],
                    }
                )
                if render_episode_video and step in frame_steps:
                    for camera_id in selected_video_cameras:
                        frames_dir = job_dir / "mujoco_frames" / str(run["episode_id"]) / camera_id
                        ensure_dir(frames_dir)
                        camera = _camera_for(mujoco, camera_id, root_position, _yaw_from_quat(root_quat))
                        renderer.update_scene(data, camera=camera)
                        frame = renderer.render()
                        frame_index = frame_index_by_step[step]
                        frame_path = frames_dir / f"frame_{frame_index:04d}.png"
                        image_module.fromarray(frame).save(frame_path)
                if fall_count:
                    break
            final_root = [float(value) for value in data.qpos[root_qpos : root_qpos + 3]]
            final_velocity = [float(value) for value in data.qvel[root_dof : root_dof + 3]]
            final_speed = math.hypot(final_velocity[0], final_velocity[1])
            target = list(run.get("target_pose") or [0.0, 0.0, root_z])
            final_error = math.dist(final_root[:2], [float(target[0]), float(target[1])])
            final_object = _object_pose(data, object_qpos)
            object_displacement = 0.0
            if initial_object.get("available") and final_object.get("available"):
                object_displacement = math.dist(
                    list(initial_object["position"])[:2],
                    list(final_object["position"])[:2],
                )
            status, success, labels, evaluator_metrics = _score_attempt(
                run=run,
                final_error_m=final_error,
                final_speed_mps=final_speed,
                fall_count=fall_count,
                unsafe_collision_count=unsafe_collision_count,
                object_contact_count=object_contact_count,
                object_displacement_m=object_displacement,
                rejected_action_count=episode_rejected,
                action_types=episode_action_types,
            )
            media: dict[str, Any] = {}
            if render_episode_video and frame_steps:
                for camera_id in selected_video_cameras:
                    output_name = camera_id
                    frames_dir = job_dir / "mujoco_frames" / str(run["episode_id"]) / camera_id
                    video_path = job_dir / "mujoco_videos" / f"{run['episode_id']}__{output_name}.mp4"
                    poster_path = job_dir / "mujoco_posters" / f"{run['episode_id']}__{output_name}.png"
                    if frames_dir.is_dir():
                        frame_files = sorted(frames_dir.glob("frame_*.png"))
                        if frame_files:
                            physics_frame_count = len(frame_files)
                            terminal_frame_hold_count = 0
                            missing_terminal_frame_count = max(
                                0, len(frame_step_list) - len(frame_files)
                            )
                            if (
                                extend_terminal_frame_for_review
                                and missing_terminal_frame_count > 0
                            ):
                                last_frame = frame_files[-1]
                                for frame_index in range(len(frame_files), len(frame_step_list)):
                                    held_frame = frames_dir / f"frame_{frame_index:04d}.png"
                                    if not held_frame.is_file():
                                        shutil.copyfile(last_frame, held_frame)
                                        terminal_frame_hold_count += 1
                                frame_files = sorted(frames_dir.glob("frame_*.png"))
                            ensure_dir(poster_path.parent)
                            shutil.copyfile(frame_files[len(frame_files) // 2], poster_path)
                            video = _write_video_from_frames(
                                frames_dir=frames_dir,
                                output_path=video_path,
                                fps=video_fps,
                            )
                            timing = _video_timing_contract(
                                requested_fps=fps,
                                encoded_fps=video_fps,
                                timestep=timestep,
                                stride_steps=video_render_stride_steps,
                                physics_frame_count=physics_frame_count,
                                encoded_frame_count=len(frame_files),
                            )
                            video.update(
                                {
                                    "render_mode": video_render_mode,
                                    "rendered_frame_count": len(frame_files),
                                    "physics_rendered_frame_count": physics_frame_count,
                                    "requested_frame_count": len(frame_step_list),
                                    "missing_terminal_frame_count": missing_terminal_frame_count,
                                    "terminal_frame_hold_count": terminal_frame_hold_count,
                                    "terminal_frame_extended_for_review": terminal_frame_hold_count > 0,
                                    "terminal_failure_frame_hold_enabled": bool(
                                        extend_terminal_frame_for_review
                                    ),
                                    "early_termination_before_requested_frames": (
                                        missing_terminal_frame_count > 0
                                    ),
                                    "review_video_stops_at_terminal_failure": (
                                        missing_terminal_frame_count > 0
                                        and not extend_terminal_frame_for_review
                                    ),
                                    "video_fps": video_fps,
                                    "video_frame_stride_steps": video_render_stride_steps,
                                    "sim_timestep_s": round(float(timestep), 9),
                                    "sim_seconds_per_rendered_frame": round(
                                        float(timestep) * max(1, video_render_stride_steps),
                                        9,
                                    ),
                                    "playback_timing": timing,
                                    "video_playback_may_look_slow_motion": timing[
                                        "video_playback_may_look_slow_motion"
                                    ],
                                    "full_episode_video": (
                                        video_render_mode == "full_episode_stride"
                                        and missing_terminal_frame_count == 0
                                    ),
                                    "configured_full_episode_timeline_requested": (
                                        video_render_mode == "full_episode_stride"
                                    ),
                                }
                            )
                            probe = _ffprobe_video(video_path) if video.get("status") == "complete" else {
                                "path": str(video_path),
                                "status": "not_checked",
                                "reason": video.get("reason") or "video_not_complete",
                            }
                            media[output_name] = {"video": video, "ffprobe": probe, "poster": str(poster_path)}
                            video_rows.append({"episode_id": run["episode_id"], "camera": output_name, **video})
                            poster_rows.append(
                                {
                                    "episode_id": run["episode_id"],
                                    "camera": output_name,
                                    "path": str(poster_path),
                                    "size_bytes": poster_path.stat().st_size if poster_path.is_file() else 0,
                                }
                            )
                            ffprobe_rows.append(probe)
            media_videos = [
                camera_media.get("video")
                for camera_media in media.values()
                if isinstance(camera_media, Mapping)
                and isinstance(camera_media.get("video"), Mapping)
            ]
            media_stops_at_terminal_failure = any(
                bool(video.get("review_video_stops_at_terminal_failure"))
                for video in media_videos
            )
            media_full_episode_video = bool(media_videos) and all(
                bool(video.get("full_episode_video")) for video in media_videos
            )
            human_review_media_source = "no_review_video"
            if media_stops_at_terminal_failure:
                human_review_media_source = "terminal_failure_stopped_mujoco_review_video"
            elif media_full_episode_video:
                human_review_media_source = "full_episode_mujoco_video"
            elif media_videos:
                human_review_media_source = "sampled_mujoco_video"
            attempt = {
                "schema_version": "mujoco_g1_wam_vla_normalized_attempt.v1",
                "attempt_id": f"attempt_{_safe_id(run['episode_id'])}",
                "episode_id": run["episode_id"],
                "scenario_eval_run_id": run["scenario_eval_run_id"],
                "scenario_variation_instance_id": run["scenario_variation_instance_id"],
                "scenario_id": run["scenario_id"],
                "task_id": run["task_id"],
                "spawn_id": run["spawn_id"],
                "status": status,
                "success": success,
                "policy_id": REFERENCE_FIXTURE_POLICY_ID if fixture_policy_used else "endpoint_policy",
                "policy_runtime_source": "reference_fixture_policy"
                if fixture_policy_used
                else selected_runtime.get("runtime"),
                "fixture_policy_used": fixture_policy_used,
                "endpoint_policy_used": bool(not fixture_policy_used and endpoint_policy_valid_actions),
                "start_root_position": [round(value, 6) for value in initial_root],
                "final_root_position": [round(value, 6) for value in final_root],
                "target_pose": target,
                "metrics": {
                    "final_target_error_m": round(final_error, 6),
                    "final_speed_mps": round(final_speed, 6),
                    "fall_count": fall_count,
                    "unsafe_collision_contact_count": unsafe_collision_count,
                    "object_contact_count": object_contact_count,
                    "object_displacement_m": round(object_displacement, 6),
                    "rejected_policy_action_count": episode_rejected,
                    **evaluator_metrics,
                },
                "failure_label_ids": labels,
                "media": media,
                "video_analysis_binding": {
                    "video_review_expected": bool(media),
                    "automated_success_source": "structured_mujoco_trace_metrics",
                    "human_review_media_source": human_review_media_source,
                    "visual_model_success_classifier_used": False,
                },
                "claim_boundary": {
                    "mujoco_simulator_only": True,
                    "endpoint_policy_plumbing_proven": bool(
                        not fixture_policy_used and endpoint_policy_valid_actions
                    ),
                    "real_wam_vla_policy_proven": False,
                    "real_wam_vla_policy_requires_model_provenance": True,
                    "physical_robot_readiness_proven": False,
                    "deployment_readiness_proven": False,
                    "official_policy_execution_proven": selected_controller_backend == "unitree_rl_gym",
                    "same_scene_unitree_controller_backend_used": selected_controller_backend == "unitree_rl_gym",
                    "freejoint_proxy_used": selected_controller_backend == "freejoint_proxy",
                },
            }
            attempts.append(attempt)
            if labels:
                failure_labels.append(
                    {
                        "label_id": f"failure_{len(failure_labels) + 1:04d}",
                        "attempt_id": attempt["attempt_id"],
                        "episode_id": run["episode_id"],
                        "scenario_eval_run_id": run["scenario_eval_run_id"],
                        "task_id": run["task_id"],
                        "spawn_id": run["spawn_id"],
                        "failure_label_ids": labels,
                        "status": status,
                        "review_required": True,
                    }
                )
            if status == "blocked":
                blocked_attempts.append(attempt)
            phase("episode_completed", status, episode_id=run["episode_id"], success=success)
    except KeyboardInterrupt:
        interruption = {
            "schema_version": "mujoco_g1_wam_vla_policy_endpoint_eval_interruption.v1",
            "generated_at": utc_now_iso(),
            "status": "interrupted",
            "job_id": job_id,
            "job_dir": str(job_dir),
            "completed_attempt_count": len(attempts),
            "matrix_attempt_count": int(matrix.get("scenario_eval_run_count") or 0),
            "final_summary_written": False,
            "final_controller_truth_written": False,
            "truth_boundary": (
                "Interrupted runs are diagnostic only. Use controller truth fields only "
                "from a completed mujoco_g1_wam_vla_policy_endpoint_eval_summary.json."
            ),
        }
        write_json(job_dir / "run_interruption_status.json", interruption)
        write_json(
            job_dir / "controller_truth_boundary.json",
            {
                "schema_version": "controller_truth_boundary.v1",
                "generated_at": interruption["generated_at"],
                "status": "interrupted_before_final_execution_truth",
                "requested_controller_backend": controller_backend,
                "controller_backend": selected_controller_backend,
                "realistic_navigation_policy_used": None,
                "realistic_navigation_policy_used_for_endpoint_rollouts": None,
                "official_unitree_controller_used": None,
                "official_policy_execution_proven": None,
                "balanced_walking_controller_proven": None,
                "freejoint_proxy_used": None,
                "completed_attempt_count": len(attempts),
                "matrix_attempt_count": int(matrix.get("scenario_eval_run_count") or 0),
                "proof_boundary": interruption["truth_boundary"],
                "blockers": ["run_interrupted_before_final_controller_truth"],
            },
        )
        phase(
            "job_interrupted",
            "interrupted",
            completed_attempt_count=len(attempts),
            matrix_attempt_count=int(matrix.get("scenario_eval_run_count") or 0),
        )
        raise
    finally:
        if renderer is not None:
            renderer.close()

    unitree_endpoint_command_rows = _unitree_command_rows_from_endpoint_actions(action_rows)
    unitree_controller_clamped_command_count = sum(
        1 for row in unitree_endpoint_command_rows if row.get("controller_command_clamped")
    )
    write_json(
        job_dir / "unitree_endpoint_action_command_stream.json",
        {
            "schema_version": "unitree_endpoint_action_command_stream.v1",
            "generated_at": generated_at,
            "status": "completed" if unitree_endpoint_command_rows else "blocked",
            "source_action_trace": "normalized_policy_action_trace.jsonl",
            "endpoint_action_trace_bound_to_unitree_command_stream": bool(
                unitree_endpoint_command_rows
            ),
            "command_count": len(unitree_endpoint_command_rows),
            "controller_clamped_command_count": unitree_controller_clamped_command_count,
            "controller_command_limits": dict(UNITREE_RL_GYM_CONTROLLER_COMMAND_LIMITS),
            "representative_endpoint_command_xyz": _representative_unitree_command(
                unitree_endpoint_command_rows
            ),
            "commands": unitree_endpoint_command_rows,
            "claim_boundary": {
                "command_stream_binding_is_not_controller_execution": True,
                "same_scene_controller_backend_still_required": True,
                "physical_robot_readiness_proven": False,
            },
        },
    )
    endpoint_action_controller_replay = _run_unitree_controller_replay_from_endpoint_actions(
        job_dir=job_dir,
        job_id=job_id,
        generated_at=generated_at,
        unitree_rl_gym_root=unitree_rl_gym_root,
        navigation_discovery=navigation_discovery,
        enabled=run_unitree_controller_replay_from_endpoint_actions,
        max_steps=unitree_controller_replay_steps,
        command_rows=unitree_endpoint_command_rows,
    )
    write_json(
        job_dir / "unitree_endpoint_action_controller_replay_manifest.json",
        endpoint_action_controller_replay,
    )
    total_attempt_fall_count = sum(
        int(attempt.get("metrics", {}).get("fall_count", 0)) for attempt in attempts
    )
    same_scene_balanced = bool(
        same_scene_controller_ready and attempts and total_attempt_fall_count == 0
    )
    same_scene_controller_clamped_update_count = sum(
        1 for row in same_scene_controller_rows if row.get("controller_command_clamped")
    )
    if same_scene_controller_ready:
        same_scene_blockers = list(same_scene_controller_manifest.get("blockers", []))
        if total_attempt_fall_count:
            same_scene_blockers.append("blocked_same_scene_unitree_controller_rollout_fell")
        same_scene_controller_manifest = {
            **same_scene_controller_manifest,
            "status": "completed" if same_scene_balanced else "completed_with_failures",
            "rollout_attempt_count": len(attempts),
            "controller_update_count": len(same_scene_controller_rows),
            "endpoint_action_controller_clamped_command_count": unitree_controller_clamped_command_count,
            "same_scene_controller_clamped_update_count": same_scene_controller_clamped_update_count,
            "controller_command_limits": dict(UNITREE_RL_GYM_CONTROLLER_COMMAND_LIMITS),
            "fall_count": total_attempt_fall_count,
            "balanced_walking_controller_proven": same_scene_balanced,
            "blockers": same_scene_blockers,
        }
        write_json(
            job_dir / "same_scene_unitree_controller_backend_manifest.json",
            same_scene_controller_manifest,
        )
    unitree_controller_bridge_manifest = build_unitree_controller_bridge_manifest(
        generated_at=generated_at,
        command_rows=unitree_endpoint_command_rows,
        official_controller_sidecar=official_controller_sidecar,
        endpoint_replay=endpoint_action_controller_replay,
        same_scene_controller=same_scene_controller_manifest,
    )
    write_json(
        job_dir / "unitree_controller_bridge_manifest.json",
        unitree_controller_bridge_manifest,
    )

    _write_jsonl(job_dir / "normalized_policy_action_trace.jsonl", action_rows)
    _write_jsonl(job_dir / "policy_endpoint_attempt_trace.jsonl", endpoint_attempt_rows)
    _write_jsonl(job_dir / "policy_endpoint_invocation_trace.jsonl", endpoint_attempt_rows)
    _write_jsonl(job_dir / "g1_mujoco_locomotion_trace.jsonl", locomotion_rows)
    _write_jsonl(
        job_dir / "same_scene_unitree_controller_trace.jsonl",
        same_scene_controller_rows,
    )
    write_json(
        job_dir / "rejected_policy_actions.json",
        {
            "schema_version": "rejected_policy_actions.v1",
            "generated_at": generated_at,
            "rejected_policy_action_count": len(rejected_actions),
            "rejections": rejected_actions,
        },
    )
    write_json(
        job_dir / "policy_action_normalization_report.json",
        {
            "schema_version": "policy_action_normalization_report.v1",
            "generated_at": generated_at,
            "total_policy_action_count": len(action_rows),
            "accepted_policy_action_count": len(action_rows) - len(rejected_actions),
            "rejected_policy_action_count": len(rejected_actions),
            "action_validity_rate": round((len(action_rows) - len(rejected_actions)) / len(action_rows), 6)
            if action_rows
            else 0.0,
            "supported_action_types": [
                "base_velocity",
                "heading_yaw",
                "waypoint",
                "stop",
                "inspect_look",
                "manipulation_contact",
            ],
            "safety_limits": dict(SAFETY_LIMITS),
        },
    )
    write_json(
        job_dir / "normalized_attempt_trace.json",
        {
            "schema_version": "mujoco_g1_wam_vla_normalized_attempt_trace.v1",
            "generated_at": generated_at,
            "status": "completed" if attempts else "blocked_missing_attempts",
            "attempt_count": len(attempts),
            "successful_task_attempt_count": sum(1 for attempt in attempts if attempt["success"]),
            "failed_task_attempt_count": sum(
                1 for attempt in attempts if attempt["status"] == "failed"
            ),
            "blocked_task_attempt_count": sum(
                1 for attempt in attempts if attempt["status"] == "blocked"
            ),
            "attempts": attempts,
            "covered_scenario_eval_run_ids": sorted(
                str(attempt["scenario_eval_run_id"]) for attempt in attempts
            ),
            "claim_boundary": {
                "simulator_only_mujoco": True,
                "physical_robot_readiness_proven": False,
                "deployment_readiness_proven": False,
            },
        },
    )
    write_json(
        job_dir / "failure_labels.json",
        {
            "schema_version": "mujoco_g1_wam_vla_failure_labels.v1",
            "generated_at": generated_at,
            "status": "review_required" if failure_labels else "no_failures_labeled",
            "label_count": len(failure_labels),
            "labels": failure_labels,
            "failed_or_blocked_attempt_count": len(failure_labels),
        },
    )
    write_json(
        job_dir / "blocked_attempts.json",
        {
            "schema_version": "mujoco_g1_wam_vla_blocked_attempts.v1",
            "generated_at": generated_at,
            "blocked_attempt_count": len(blocked_attempts),
            "blocked_attempts": blocked_attempts,
        },
    )
    required_ids = sorted(str(run["scenario_eval_run_id"]) for run in matrix["runs"])
    covered_ids = sorted(str(attempt["scenario_eval_run_id"]) for attempt in attempts)
    write_json(
        job_dir / "scenario_matrix_coverage_report.json",
        {
            "schema_version": "scenario_matrix_coverage_report.v1",
            "generated_at": generated_at,
            "required_scenario_eval_run_count": len(required_ids),
            "covered_scenario_eval_run_count": len(covered_ids),
            "missing_scenario_eval_run_ids": sorted(set(required_ids) - set(covered_ids)),
            "scenario_eval_run_coverage_complete": required_ids == covered_ids and bool(required_ids),
            "attempt_count_matches_matrix_count": len(attempts) == len(required_ids),
        },
    )
    discontinuities = []
    by_episode: dict[str, list[Mapping[str, Any]]] = {}
    for row in locomotion_rows:
        by_episode.setdefault(str(row["episode_id"]), []).append(row)
    max_step_translation = 0.0
    for episode_id, rows in by_episode.items():
        previous: Mapping[str, Any] | None = None
        for row in rows:
            if previous is not None:
                dist = math.dist(
                    list(previous["root_position"])[:2],
                    list(row["root_position"])[:2],
                )
                max_step_translation = max(max_step_translation, dist)
                if dist > SAFETY_LIMITS["max_forward_velocity_mps"] * timestep * 5.0 + 0.02:
                    discontinuities.append(
                        {
                            "episode_id": episode_id,
                            "from_step": previous["step"],
                            "to_step": row["step"],
                            "translation_m": round(dist, 6),
                        }
                    )
            previous = row
    continuity_report = {
        "schema_version": "root_motion_continuity_report.v1",
        "generated_at": generated_at,
        "status": "validated" if locomotion_rows and not discontinuities else "blocked",
        "continuous_mujoco_stepping": True,
        "root_pose_teleport_success_used": False,
        "locomotion_continuity_validated": bool(locomotion_rows and not discontinuities),
        "trace_row_count": len(locomotion_rows),
        "max_step_translation_m": round(max_step_translation, 6),
        "discontinuity_count": len(discontinuities),
        "discontinuities": discontinuities[:25],
    }
    write_json(job_dir / "root_motion_continuity_report.json", continuity_report)
    foot_contact_trace = {
        "schema_version": "foot_contact_trace.v1",
        "generated_at": generated_at,
        "contact_row_count": contact_trace_total_count,
        "sampled_contact_row_count": len(contact_rows),
        "max_sampled_contact_trace_rows": contact_trace_row_limit,
        "contact_trace_truncated": contact_trace_truncated,
        "dropped_contact_trace_row_count": contact_trace_dropped_count,
        "left_foot_contact_count": contact_aggregate_counts["left_foot_contact_count"],
        "right_foot_contact_count": contact_aggregate_counts["right_foot_contact_count"],
        "floor_contact_count": contact_aggregate_counts["floor_contact_count"],
        "contact_detection_sampled": contact_trace_total_count > 0,
        "contacts": contact_rows[:500],
    }
    write_json(job_dir / "foot_contact_trace.json", foot_contact_trace)
    collision_report = {
        "schema_version": "collision_contact_report.v1",
        "generated_at": generated_at,
        "contact_detection_sampled": contact_trace_total_count > 0,
        "collision_dynamics_validated": contact_trace_total_count > 0,
        "contact_row_count": contact_trace_total_count,
        "sampled_contact_row_count": len(contact_rows),
        "max_sampled_contact_trace_rows": contact_trace_row_limit,
        "contact_trace_truncated": contact_trace_truncated,
        "dropped_contact_trace_row_count": contact_trace_dropped_count,
        "unsafe_obstacle_contact_count": contact_aggregate_counts[
            "obstacle_contact_count"
        ],
        "object_contact_count": contact_aggregate_counts["object_contact_count"],
        "floor_contact_count": contact_aggregate_counts["floor_contact_count"],
        "sample_contacts": contact_rows[:100],
    }
    write_json(job_dir / "collision_contact_report.json", collision_report)
    write_json(
        job_dir / "manipulation_scene_object_manifest.json",
        {
            "schema_version": "manipulation_scene_object_manifest.v1",
            "generated_at": generated_at,
            "status": "completed",
            "object_id": "blueprint_light_object",
            "object_geom": "blueprint_light_object_geom",
            "mujoco_freejoint": "blueprint_light_object_freejoint",
            "initial_pose": object_initial_pose,
            "do_not_fake_object_motion": True,
        },
    )
    write_json(
        job_dir / "manipulation_action_spec_manifest.json",
        {
            "schema_version": "manipulation_action_spec_manifest.v1",
            "generated_at": generated_at,
            "supported_action_type": "manipulation_contact",
            "current_action_effect": (
                "high_level_endpoint_action_normalized_to_unitree_lower_body_controller_command"
                if selected_controller_backend == "unitree_rl_gym"
                else "high_level_endpoint_action_normalized_to_base_proxy_contact_motion"
            ),
            "hand_end_effector_control_available": False,
            "dexterous_hand_policy_required_for_vla_manipulation_claim": True,
            "success_threshold_object_displacement_m": SAFETY_LIMITS[
                "object_displacement_success_m"
            ],
            "simulator_only": True,
        },
    )
    write_json(
        job_dir / "manipulation_contact_trace.json",
        {
            "schema_version": "manipulation_contact_trace.v1",
            "generated_at": generated_at,
            "status": "completed"
            if contact_aggregate_counts["object_contact_count"] > 0
            else "blocked",
            "manipulation_contact_count": contact_aggregate_counts["object_contact_count"],
            "sampled_manipulation_contact_count": len(manipulation_contacts),
            "contact_trace_truncated": contact_trace_truncated,
            "dropped_contact_trace_row_count": contact_trace_dropped_count,
            "contacts": manipulation_contacts[:500],
            "blockers": []
            if contact_aggregate_counts["object_contact_count"] > 0
            else ["blocked_manipulation_contact_asset_unavailable"],
        },
    )
    write_json(
        job_dir / "object_motion_trace.json",
        {
            "schema_version": "object_motion_trace.v1",
            "generated_at": generated_at,
            "status": "completed" if object_motion_rows else "blocked",
            "row_count": len(object_motion_rows),
            "rows": object_motion_rows[:1000],
        },
    )
    contact_attempts = [
        attempt for attempt in attempts if attempt["task_id"] == "contact_or_push_light_object"
    ]
    manipulation_successes = [
        attempt
        for attempt in contact_attempts
        if attempt["metrics"].get("object_displacement_m", 0) >= SAFETY_LIMITS[
            "object_displacement_success_m"
        ]
        and attempt["metrics"].get("object_contact_count", 0) > 0
    ]
    write_json(
        job_dir / "manipulation_success_evaluator_results.json",
        {
            "schema_version": "manipulation_success_evaluator_results.v1",
            "generated_at": generated_at,
            "status": "completed" if contact_attempts else "blocked",
            "attempt_count": len(contact_attempts),
            "success_count": len(manipulation_successes),
            "success_rate": round(len(manipulation_successes) / len(contact_attempts), 6)
            if contact_attempts
            else 0.0,
            "results": [
                {
                    "attempt_id": attempt["attempt_id"],
                    "scenario_eval_run_id": attempt["scenario_eval_run_id"],
                    "spawn_id": attempt["spawn_id"],
                    "object_displacement_m": attempt["metrics"].get("object_displacement_m"),
                    "object_contact_count": attempt["metrics"].get("object_contact_count"),
                    "passed": attempt in manipulation_successes,
                }
                for attempt in contact_attempts
            ],
        },
    )
    manipulation_validated = bool(manipulation_successes)
    manipulation_blockers: list[str] = []
    if not contact_attempts:
        manipulation_blockers.append("blocked_missing_contact_task_attempts")
    if contact_attempts and not manipulation_validated:
        manipulation_blockers.append("blocked_manipulation_contact_not_validated")
    manipulation_blockers.extend(
        [
            "blocked_dexterous_hand_policy_not_integrated",
            "blocked_real_vla_model_not_configured",
        ]
    )
    write_json(
        job_dir / "manipulation_endpoint_task_report.json",
        {
            "schema_version": "manipulation_endpoint_task_report.v1",
            "generated_at": generated_at,
            "status": "completed" if contact_attempts else "blocked",
            "endpoint_action_path_required": True,
            "endpoint_action_path_used": bool(
                contact_attempts and any(attempt.get("endpoint_policy_used") for attempt in contact_attempts)
            ),
            "manipulation_endpoint_path_used": bool(
                contact_attempts and any(attempt.get("endpoint_policy_used") for attempt in contact_attempts)
            ),
            "fixture_policy_used": bool(
                contact_attempts and any(attempt.get("fixture_policy_used") for attempt in contact_attempts)
            ),
            "attempt_count": len(contact_attempts),
            "object_contact_count": sum(
                int(attempt["metrics"].get("object_contact_count", 0)) for attempt in contact_attempts
            ),
            "max_object_displacement_m": max(
                [float(attempt["metrics"].get("object_displacement_m", 0.0)) for attempt in contact_attempts]
                or [0.0]
            ),
            "unsafe_collision_count": sum(
                int(attempt["metrics"].get("unsafe_collision_contact_count", 0))
                for attempt in contact_attempts
            ),
            "fall_count": sum(int(attempt["metrics"].get("fall_count", 0)) for attempt in contact_attempts),
            "successful_contact_attempt_count": len(manipulation_successes),
            "hand_end_effector_policy_used": False,
            "base_proxy_contact_path_used": bool(
                contact_attempts and selected_controller_backend == "freejoint_proxy"
            ),
            "lower_body_controller_contact_path_used": bool(
                contact_attempts and selected_controller_backend == "unitree_rl_gym"
            ),
            "task_requires_dexterous_hand_policy_for_vla_manipulation_claim": True,
            "claim_boundary": {
                "simulator_only": True,
                "contact_success_only_via_freejoint_proxy": bool(
                    manipulation_validated and selected_controller_backend == "freejoint_proxy"
                ),
                "contact_success_only_via_unitree_lower_body_controller": bool(
                    manipulation_validated and selected_controller_backend == "unitree_rl_gym"
                ),
                "dexterous_vla_manipulation_proven": False,
                "real_vla_model_ran": False,
                "physical_robot_readiness_proven": False,
            },
            "blockers": manipulation_blockers,
        },
    )
    write_json(
        job_dir / "manipulation_truth_boundary.json",
        {
            "schema_version": "manipulation_truth_boundary.v1",
            "generated_at": generated_at,
            "manipulation_contact_dynamics_validated": manipulation_validated,
            "simulator_only": True,
            "freejoint_proxy_used": selected_controller_backend == "freejoint_proxy",
            "unitree_lower_body_controller_used": selected_controller_backend == "unitree_rl_gym",
            "contact_success_only_via_freejoint_proxy": bool(
                manipulation_validated and selected_controller_backend == "freejoint_proxy"
            ),
            "contact_success_only_via_unitree_lower_body_controller": bool(
                manipulation_validated and selected_controller_backend == "unitree_rl_gym"
            ),
            "base_proxy_contact_path_used": bool(
                contact_attempts and selected_controller_backend == "freejoint_proxy"
            ),
            "lower_body_controller_contact_path_used": bool(
                contact_attempts and selected_controller_backend == "unitree_rl_gym"
            ),
            "hand_end_effector_policy_used": False,
            "dexterous_hand_policy_proven": False,
            "vla_manipulation_policy_proven": False,
            "real_wam_vla_model_ran": False,
            "physical_robot_readiness_proven": False,
            "blockers": manipulation_blockers,
        },
    )
    evaluator_rows = []
    for attempt in attempts:
        metrics = _mapping(attempt.get("metrics"))
        evaluator_rows.append(
            {
                "attempt_id": attempt["attempt_id"],
                "scenario_eval_run_id": attempt["scenario_eval_run_id"],
                "task_id": attempt["task_id"],
                "spawn_id": attempt["spawn_id"],
                "status": attempt["status"],
                "passed": attempt["success"],
                "navigation_success": metrics.get("navigation_success"),
                "task_progress": metrics.get("task_progress"),
                "route_safety": metrics.get("route_safety"),
                "contact_collision_correctness": metrics.get("contact_collision_correctness"),
                "object_displacement_success": metrics.get("object_displacement_success"),
                "stopped_at_goal": metrics.get("stopped_at_goal"),
                "policy_endpoint_responsiveness": attempt["fixture_policy_used"]
                or attempt["endpoint_policy_used"],
                "action_validity": metrics.get("action_validity"),
                "failure_label_ids": attempt.get("failure_label_ids"),
            }
        )
    write_json(
        job_dir / "wam_evaluator_thresholds.json",
        {
            "schema_version": "wam_evaluator_thresholds.v1",
            "generated_at": generated_at,
            "thresholds": dict(SAFETY_LIMITS),
            "simulator_backend": "mujoco",
            "proof_boundary": "WAM-style evaluator over MuJoCo traces, not real-world SRCC",
        },
    )
    write_json(
        job_dir / "wam_evaluator_trace_binding.json",
        {
            "schema_version": "wam_evaluator_trace_binding.v1",
            "generated_at": generated_at,
            "normalized_attempt_trace": "normalized_attempt_trace.json",
            "locomotion_trace_jsonl": "g1_mujoco_locomotion_trace.jsonl",
            "action_trace_jsonl": "normalized_policy_action_trace.jsonl",
            "contact_trace": "foot_contact_trace.json",
            "collision_report": "collision_contact_report.json",
            "video_generation_status": "video_generation_status.json",
            "video_analysis_manifest": "video_analysis_manifest.json",
            "scoring_source": "structured_mujoco_state_contact_action_traces",
            "video_role": "human_review_evidence_bound_to_same_episode_ids",
        },
    )
    write_json(
        job_dir / "wam_evaluator_results.json",
        {
            "schema_version": "wam_evaluator_results.v1",
            "generated_at": generated_at,
            "status": "completed",
            "wam_evaluator_trace_scored": True,
            "attempt_count": len(attempts),
            "passed_count": sum(1 for attempt in attempts if attempt["success"]),
            "failed_count": sum(1 for attempt in attempts if attempt["status"] == "failed"),
            "blocked_count": sum(1 for attempt in attempts if attempt["status"] == "blocked"),
            "results": evaluator_rows,
        },
    )
    write_json(
        job_dir / "evaluator_truth_boundary.json",
        {
            "schema_version": "evaluator_truth_boundary.v1",
            "generated_at": generated_at,
            "wam_evaluator_trace_scored": True,
            "simulator_only": True,
            "real_world_outcome_proven": False,
            "deployment_readiness_proven": False,
        },
    )
    video_status = {
        "schema_version": "video_generation_status.v1",
        "generated_at": generated_at,
        "status": "completed" if video_rows else "blocked",
        "video_count": len(video_rows),
        "poster_count": len(poster_rows),
        "render_contract": {
            "default_render_mode": "full_episode_stride"
            if int(render_frame_count) <= 0
            else "fixed_sample_count",
            "steps_per_episode": int(steps_per_episode),
            "mujoco_timestep_s": round(float(timestep), 9),
            "configured_episode_sim_duration_s": round(
                float(timestep) * max(1, int(steps_per_episode)),
                9,
            ),
            "render_frame_count": int(render_frame_count),
            "video_frame_stride_steps": int(video_frame_stride_steps),
            "default_review_video_stride_is_bounded_for_matrix_runs": (
                int(render_frame_count) <= 0
                and int(video_frame_stride_steps) == DEFAULT_VIDEO_FRAME_STRIDE_STEPS
            ),
            "fps_zero_encodes_sim_time_playback": int(fps) == 0,
            "fixed_fps_with_every_step_can_create_slow_motion": True,
            "every_sim_step_captured_for_selected_review_videos": int(video_frame_stride_steps)
            == 1,
            "terminal_failure_frame_hold_enabled": bool(extend_terminal_frame_for_review),
            "review_videos_stop_at_terminal_failure_by_default": not bool(
                extend_terminal_frame_for_review
            ),
            "rendered_video_episode_limit": rendered_video_episode_cap,
            "rendered_video_camera_ids": list(selected_video_cameras),
            "scored_episode_count_can_exceed_rendered_video_episode_count": True,
            "fps": int(fps),
            "default_review_video_fps": DEFAULT_REVIEW_VIDEO_FPS,
            "default_review_video_contract": (
                "stride-8 frame sampling encoded at 60fps unless --fps 0 requests exact simulator-time playback"
            ),
            "fps_zero_means_realtime_from_mujoco_timestep": True,
            "short_review_video_reason": (
                "videos stop at the actual terminal physics failure frame unless terminal-frame hold is enabled"
            ),
            "videos_are_for_human_review_of_scored_episodes": True,
            "automated_success_source": "structured_mujoco_trace_metrics",
        },
        "videos": video_rows,
        "posters": poster_rows,
        "ffprobe": ffprobe_rows,
        "blockers": [] if video_rows else ["blocked_video_renderer_unavailable"],
    }
    write_json(job_dir / "video_generation_status.json", video_status)
    write_json(
        job_dir / "video_analysis_manifest.json",
        {
            "schema_version": "video_analysis_manifest.v1",
            "generated_at": generated_at,
            "status": "completed" if attempts else "blocked",
            "analysis_boundary": {
                "videos_cover_configured_episode_timeline_when_full_episode_stride": (
                    int(render_frame_count) <= 0 and bool(extend_terminal_frame_for_review)
                ),
                "videos_stop_at_terminal_physics_failure_unless_hold_enabled": not bool(
                    extend_terminal_frame_for_review
                ),
                "videos_include_every_mujoco_step_for_selected_review_videos": int(
                    video_frame_stride_steps
                )
                == 1,
                "fps_zero_encodes_sim_time_playback": int(fps) == 0,
                "videos_are_review_evidence_for_the_same_attempt_ids": True,
                "automated_visual_success_classifier_used": False,
                "automated_success_source": "structured_mujoco_state_contact_action_traces",
                "manual_review_can_override_or_annotate_failure_labels": True,
            },
            "attempt_count": len(attempts),
            "attempts": [
                {
                    "attempt_id": attempt["attempt_id"],
                    "episode_id": attempt["episode_id"],
                    "scenario_eval_run_id": attempt["scenario_eval_run_id"],
                    "task_id": attempt["task_id"],
                    "spawn_id": attempt["spawn_id"],
                    "status": attempt["status"],
                    "success": attempt["success"],
                    "failure_label_ids": attempt.get("failure_label_ids", []),
                    "metrics": attempt.get("metrics", {}),
                    "media": attempt.get("media", {}),
                    "review_question": "Does the full MuJoCo video agree with the structured trace score for this task?",
                }
                for attempt in attempts
            ],
        },
    )
    required_count = int(matrix["scenario_eval_run_count"])
    ffprobe_by_path = {
        str(row.get("path")): dict(row)
        for row in ffprobe_rows
        if isinstance(row, Mapping) and row.get("path")
    }
    selected_review_videos = []
    for row in video_rows:
        video_path = str(row.get("path") or "")
        selected_review_videos.append(
            {
                "episode_id": row.get("episode_id"),
                "camera": row.get("camera"),
                "path": video_path,
                "status": row.get("status"),
                "full_episode_video": bool(row.get("full_episode_video")),
                "configured_full_episode_timeline_requested": bool(
                    row.get("configured_full_episode_timeline_requested")
                ),
                "physics_rendered_frame_count": int(row.get("physics_rendered_frame_count") or 0),
                "requested_frame_count": int(row.get("requested_frame_count") or 0),
                "missing_terminal_frame_count": int(row.get("missing_terminal_frame_count") or 0),
                "terminal_frame_hold_count": int(row.get("terminal_frame_hold_count") or 0),
                "terminal_frame_extended_for_review": bool(
                    row.get("terminal_frame_extended_for_review")
                ),
                "terminal_failure_frame_hold_enabled": bool(
                    row.get("terminal_failure_frame_hold_enabled")
                ),
                "early_termination_before_requested_frames": bool(
                    row.get("early_termination_before_requested_frames")
                ),
                "review_video_stops_at_terminal_failure": bool(
                    row.get("review_video_stops_at_terminal_failure")
                ),
                "playback_timing": row.get("playback_timing", {}),
                "video_playback_may_look_slow_motion": bool(
                    row.get("video_playback_may_look_slow_motion")
                ),
                "ffprobe": ffprobe_by_path.get(video_path, {}),
            }
        )
    write_json(
        job_dir / "review_video_selection_manifest.json",
        {
            "schema_version": "review_video_selection_manifest.v1",
            "generated_at": generated_at,
            "status": "completed" if selected_review_videos else "blocked",
            "selection_policy": {
                "default_camera": "third_person",
                "selected_camera_ids": list(selected_video_cameras),
                "rendered_video_episode_limit": rendered_video_episode_cap,
                "steps_per_episode": int(steps_per_episode),
                "mujoco_timestep_s": round(float(timestep), 9),
                "configured_episode_sim_duration_s": round(
                    float(timestep) * max(1, int(steps_per_episode)),
                    9,
                ),
                "video_frame_stride_steps": int(video_frame_stride_steps),
                "every_sim_step_captured": int(video_frame_stride_steps) == 1,
                "terminal_failure_frame_hold_enabled": bool(
                    extend_terminal_frame_for_review
                ),
                "review_videos_stop_at_terminal_failure_by_default": not bool(
                    extend_terminal_frame_for_review
                ),
                "playback_fps": int(selected_review_playback_fps),
                "default_review_video_fps": DEFAULT_REVIEW_VIDEO_FPS,
                "default_review_video_contract": (
                    "stride-8 frame sampling encoded at 60fps unless --fps 0 requests exact simulator-time playback"
                ),
                "fps_zero_encodes_sim_time_playback": int(fps) == 0,
                "fixed_fps_with_every_step_can_create_slow_motion": True,
                "rendered_attempts_are_subset_of_scored_attempts": True,
                "default_target_review_video_count": "5_to_10_attempts",
                "scored_all_matrix_rows": len(attempts) == required_count and required_count > 0,
            },
            "selected_review_video_count": len(selected_review_videos),
            "selected_review_videos": selected_review_videos,
            "blockers": [] if selected_review_videos else ["blocked_video_renderer_unavailable"],
        },
    )
    if not video_rows:
        write_json(
            job_dir / "blocked_video_renderer_unavailable.json",
            {
                "schema_version": "blocked_video_renderer_unavailable.v1",
                "generated_at": generated_at,
                "status": "blocked",
                "blockers": ["blocked_video_renderer_unavailable"],
            },
        )
    successful = sum(1 for attempt in attempts if attempt["success"])
    failed = sum(1 for attempt in attempts if attempt["status"] == "failed")
    blocked = sum(1 for attempt in attempts if attempt["status"] == "blocked")
    endpoint_policy_used = bool(endpoint_policy_valid_actions)
    endpoint_validity_rate = (
        round(endpoint_policy_valid_actions / endpoint_policy_decisions, 6)
        if endpoint_policy_decisions
        else None
    )
    endpoint_invocation_count = sum(1 for row in endpoint_attempt_rows if row.get("endpoint_invoked"))
    final_fixture_policy_used = fixture_policy_used or any(
        row.get("source") == "reference_fixture_policy" for row in action_rows
    )
    same_scene_integrated = bool(
        unitree_controller_bridge_manifest.get("same_scene_controller_backend_integrated")
    )
    same_scene_balanced_proven = bool(
        same_scene_controller_manifest.get("balanced_walking_controller_proven")
    )
    final_official_unitree_controller_used = bool(
        official_controller_proven or same_scene_integrated
    )
    final_balanced_walking_proven = bool(
        same_scene_balanced_proven
        or official_controller_sidecar.get("balanced_walking_controller_proven")
    )
    final_freejoint_proxy_used = not same_scene_integrated
    final_navigation_policy_kind = (
        UNITREE_RL_GYM_SAME_SCENE_BACKEND_ID
        if same_scene_integrated
        else "freejoint_velocity_proxy_with_g1_joint_position_holds"
    )
    write_json(
        job_dir / "controller_truth_boundary.json",
        {
            "schema_version": "controller_truth_boundary.v1",
            "generated_at": generated_at,
            "requested_controller_backend": controller_backend,
            "controller_backend": selected_controller_backend,
            "controller_kind": final_navigation_policy_kind,
            "realistic_navigation_policy_used": same_scene_integrated,
            "realistic_navigation_policy_used_for_endpoint_rollouts": same_scene_integrated,
            "official_unitree_controller_sidecar_status": official_controller_sidecar.get("status"),
            "same_scene_controller_backend_status": same_scene_controller_manifest.get("status"),
            "navigation_policy_kind": final_navigation_policy_kind,
            "continuous_mujoco_stepping": True,
            "root_pose_teleport_success_used": False,
            "official_unitree_controller_used": final_official_unitree_controller_used,
            "official_policy_execution_proven": final_official_unitree_controller_used,
            "training_grade_policy_rollout_proven": same_scene_integrated,
            "balanced_walking_controller_proven": final_balanced_walking_proven,
            "same_scene_unitree_controller_backend_integrated": same_scene_integrated,
            "same_scene_unitree_controller_rollout_fall_count": total_attempt_fall_count,
            "same_scene_unitree_controller_update_count": len(same_scene_controller_rows),
            "endpoint_action_controller_clamped_command_count": unitree_controller_clamped_command_count,
            "same_scene_controller_clamped_update_count": same_scene_controller_clamped_update_count,
            "controller_command_limits": dict(UNITREE_RL_GYM_CONTROLLER_COMMAND_LIMITS),
            "official_unitree_controller_sidecar_command_xyz": official_controller_sidecar.get(
                "command_xyz"
            ),
            "freejoint_proxy_used": final_freejoint_proxy_used,
            "blockers": same_scene_controller_manifest.get("blockers", [])
            if same_scene_integrated
            else same_scene_controller_manifest.get("blockers", [])
            or navigation_discovery.get("blockers", []),
            "proof_boundary": (
                "same-scene Unitree RL Gym policy execution in MuJoCo is still simulator-only "
                "and does not prove physical robot readiness, safety validation, or dexterous VLA manipulation"
            )
            if same_scene_integrated
            else "simulator policy endpoint plumbing and MuJoCo trace lane, not official Unitree locomotion proof",
        },
    )
    write_json(
        job_dir / "policy_endpoint_runtime_manifest.json",
        build_policy_endpoint_runtime_manifest(
            generated_at=generated_at,
            selected_runtime=selected_runtime,
            endpoint_policy_used=endpoint_policy_used,
            fixture_policy_used=final_fixture_policy_used,
            endpoint_invocation_count=endpoint_invocation_count,
            endpoint_valid_action_count=endpoint_policy_valid_actions,
            rejected_policy_action_count=len(rejected_actions),
        ),
    )
    write_json(
        job_dir / "policy_command_adapter_manifest.json",
        build_policy_command_adapter_manifest(
            generated_at=generated_at,
            action_rows=action_rows,
        ),
    )
    summary = {
        "schema_version": LANE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if attempts else "blocked",
        "job_id": job_id,
        "job_dir": str(job_dir),
        "mujoco_runtime_available": True,
        "unitree_g1_mujoco_model_source": asset_manifest["unitree_g1_mujoco_model_source"],
        "unitree_g1_mujoco_model_path": str(g1_root / "g1.xml"),
        "unitree_g1_loaded_in_mujoco": True,
        "policy_endpoint_runtime_proven": endpoint_policy_used,
        "wam_vla_runtime_proven": bool(
            endpoint_policy_used
            and selected_runtime
            and selected_runtime.get("runtime") in {"wam", "vla"}
            and selected_runtime.get("model_provenance_recorded") is True
        ),
        "fixture_policy_used": final_fixture_policy_used,
        "endpoint_policy_used": endpoint_policy_used,
        "endpoint_invocation_count": endpoint_invocation_count,
        "endpoint_policy_decision_count": endpoint_policy_decisions,
        "endpoint_valid_action_count": endpoint_policy_valid_actions,
        "endpoint_policy_action_validity_rate": endpoint_validity_rate,
        "locomotion_continuity_validated": continuity_report["locomotion_continuity_validated"],
        "collision_dynamics_validated": collision_report["collision_dynamics_validated"],
        "manipulation_contact_dynamics_validated": manipulation_validated,
        "wam_evaluator_trace_scored": True,
        "requested_controller_backend": controller_backend,
        "controller_backend": selected_controller_backend,
        "realistic_navigation_policy_used": same_scene_integrated,
        "realistic_navigation_policy_used_for_endpoint_rollouts": same_scene_integrated,
        "official_unitree_controller_sidecar_status": official_controller_sidecar.get("status"),
        "realistic_navigation_policy_sidecar_proven": official_controller_proven,
        "navigation_policy_kind": final_navigation_policy_kind,
        "freejoint_proxy_used": final_freejoint_proxy_used,
        "official_unitree_controller_used": final_official_unitree_controller_used,
        "official_policy_execution_proven": final_official_unitree_controller_used,
        "balanced_walking_controller_proven": final_balanced_walking_proven,
        "official_unitree_controller_sidecar_command_xyz": official_controller_sidecar.get(
            "command_xyz"
        ),
        "same_scene_unitree_controller_backend_status": same_scene_controller_manifest.get("status"),
        "same_scene_unitree_controller_rollout_fall_count": total_attempt_fall_count,
        "same_scene_unitree_controller_update_count": len(same_scene_controller_rows),
        "unitree_endpoint_action_command_count": len(unitree_endpoint_command_rows),
        "unitree_endpoint_action_controller_clamped_command_count": unitree_controller_clamped_command_count,
        "same_scene_controller_clamped_update_count": same_scene_controller_clamped_update_count,
        "unitree_controller_command_limits": dict(UNITREE_RL_GYM_CONTROLLER_COMMAND_LIMITS),
        "endpoint_action_trace_bound_to_unitree_command_stream": bool(
            unitree_endpoint_command_rows
        ),
        "unitree_controller_replay_status": endpoint_action_controller_replay.get("status"),
        "unitree_controller_replay_proven": bool(
            endpoint_action_controller_replay.get("official_unitree_controller_used")
        ),
        "same_scene_unitree_controller_backend_integrated": bool(
            unitree_controller_bridge_manifest.get("same_scene_controller_backend_integrated")
        ),
        "unitree_controller_bridge_blockers": unitree_controller_bridge_manifest.get(
            "blockers", []
        ),
        "realistic_navigation_policy_blockers": navigation_discovery.get("blockers", []),
        "attempted_episode_count": len(attempts),
        "successful_episode_count": successful,
        "failed_episode_count": failed,
        "blocked_episode_count": blocked,
        "rejected_policy_action_count": len(rejected_actions),
        "contact_row_count": contact_trace_total_count,
        "sampled_contact_row_count": len(contact_rows),
        "max_sampled_contact_trace_rows": contact_trace_row_limit,
        "contact_trace_truncated": contact_trace_truncated,
        "dropped_contact_trace_row_count": contact_trace_dropped_count,
        "scenario_eval_run_coverage_complete": len(attempts) == required_count and required_count > 0,
        "attempt_count_matches_matrix_count": len(attempts) == required_count,
        "pass_fail_by_task": _counts_by_key(attempts, "task_id"),
        "pass_fail_by_spawn": _counts_by_key(attempts, "spawn_id"),
        "isaac_runtime_available": False,
        "realistic_splat_visual_rendered": False,
        "physical_robot_readiness_proven": False,
        "deployment_readiness_proven": False,
        "artifact_paths": {
            "policy_endpoint_discovery": str(job_dir / "policy_endpoint_discovery.json"),
            "policy_endpoint_runtime_manifest": str(job_dir / "policy_endpoint_runtime_manifest.json"),
            "policy_endpoint_server_manifest": str(job_dir / "policy_endpoint_server_manifest.json"),
            "policy_command_adapter_manifest": str(job_dir / "policy_command_adapter_manifest.json"),
            "wam_vla_runtime_discovery": str(job_dir / "wam_vla_runtime_discovery.json"),
            "policy_endpoint_auth_manifest": str(job_dir / "policy_endpoint_auth_manifest.json"),
            "policy_endpoint_probe_results": str(job_dir / "policy_endpoint_probe_results.json"),
            "policy_endpoint_invocation_trace_jsonl": str(job_dir / "policy_endpoint_invocation_trace.jsonl"),
            "policy_model_candidate_matrix": str(job_dir / "policy_model_candidate_matrix.json"),
            "policy_model_truth_boundary": str(job_dir / "policy_model_truth_boundary.json"),
            "realistic_navigation_policy_discovery": str(job_dir / "realistic_navigation_policy_discovery.json"),
            "official_unitree_controller_sidecar_manifest": str(
                job_dir / "official_unitree_controller_sidecar_manifest.json"
            ),
            "unitree_endpoint_action_command_stream": str(
                job_dir / "unitree_endpoint_action_command_stream.json"
            ),
            "unitree_endpoint_action_controller_replay_manifest": str(
                job_dir / "unitree_endpoint_action_controller_replay_manifest.json"
            ),
            "unitree_controller_bridge_manifest": str(
                job_dir / "unitree_controller_bridge_manifest.json"
            ),
            "same_scene_unitree_controller_backend_manifest": str(
                job_dir / "same_scene_unitree_controller_backend_manifest.json"
            ),
            "same_scene_unitree_controller_trace_jsonl": str(
                job_dir / "same_scene_unitree_controller_trace.jsonl"
            ),
            "scenario_eval_matrix": str(job_dir / "scenario_eval_matrix.json"),
            "normalized_attempt_trace": str(job_dir / "normalized_attempt_trace.json"),
            "normalized_policy_action_trace_jsonl": str(job_dir / "normalized_policy_action_trace.jsonl"),
            "g1_mujoco_locomotion_trace_jsonl": str(job_dir / "g1_mujoco_locomotion_trace.jsonl"),
            "wam_evaluator_results": str(job_dir / "wam_evaluator_results.json"),
            "video_generation_status": str(job_dir / "video_generation_status.json"),
            "video_analysis_manifest": str(job_dir / "video_analysis_manifest.json"),
            "review_video_selection_manifest": str(job_dir / "review_video_selection_manifest.json"),
            "manipulation_endpoint_task_report": str(job_dir / "manipulation_endpoint_task_report.json"),
        },
        "recommendation": (
            "Use this as the fast local policy-endpoint plumbing evaluator before Isaac. "
            "Do not use it as Isaac, physical robot, deployment, safety, or official Unitree controller proof."
        ),
    }
    write_json(job_dir / "policy_evaluation_summary.json", summary)
    write_json(job_dir / "mujoco_g1_wam_vla_policy_endpoint_eval_summary.json", summary)
    phase("job_completed", "completed", attempted_episode_count=len(attempts))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path)
    parser.add_argument("--job-root", type=Path)
    parser.add_argument("--g1-model-root", type=Path)
    parser.add_argument("--task-filter", action="append", default=None)
    parser.add_argument("--spawn-filter", action="append", default=None)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-spawns", type=int)
    parser.add_argument("--steps-per-episode", type=int, default=DEFAULT_STEPS_PER_EPISODE)
    parser.add_argument("--policy-interval-steps", type=int, default=20)
    parser.add_argument(
        "--render-frame-count",
        type=int,
        default=0,
        help="Fixed sampled frame count. Use 0 for full-episode stride rendering.",
    )
    parser.add_argument("--video-frame-stride-steps", type=int, default=DEFAULT_VIDEO_FRAME_STRIDE_STEPS)
    parser.add_argument(
        "--capture-every-sim-step-review-video",
        action="store_true",
        help=(
            "Override video stride to 1 for high-fidelity review of selected episodes. "
            "This is useful for targeted debugging but can be slow for full matrix runs."
        ),
    )
    parser.add_argument(
        "--extend-terminal-frame-for-review",
        action="store_true",
        help=(
            "Pad early terminal physics failures with the last rendered frame. "
            "By default videos stop at the actual terminal failure frame."
        ),
    )
    parser.add_argument(
        "--rendered-video-episode-limit",
        type=int,
        default=DEFAULT_RENDERED_VIDEO_EPISODE_LIMIT,
        help="Maximum scored episodes to render. Use 0 to render every episode.",
    )
    parser.add_argument(
        "--video-camera",
        action="append",
        dest="video_cameras",
        choices=AVAILABLE_VIDEO_CAMERAS,
        default=None,
        help="Camera to render. Repeat for multiple cameras. Defaults to all cameras.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help=(
            "Review video FPS. Defaults to 60 with the default stride-8 frame sampling. "
            "When --capture-every-sim-step-review-video is used without an explicit --fps, "
            "the CLI uses 0 to derive exact simulator-time playback FPS from MuJoCo timestep "
            "and frame stride. Forcing 60 fps while capturing every 0.002s MuJoCo step creates "
            "slow-motion debug video."
        ),
    )
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--endpoint-timeout-seconds", type=float, default=8.0)
    parser.add_argument(
        "--max-contact-trace-rows",
        type=int,
        default=DEFAULT_MAX_CONTACT_TRACE_ROWS,
        help=(
            "Maximum sampled per-contact rows to persist. Aggregate contact counts remain "
            "complete; this cap only bounds trace artifact size and contact-heavy runtime."
        ),
    )
    parser.add_argument("--allow-fetch-g1-assets", action="store_true")
    parser.add_argument("--menagerie-ref", default=DEFAULT_MENAGERIE_REF)
    parser.add_argument("--unitree-rl-gym-root", type=Path)
    parser.add_argument(
        "--run-official-unitree-controller-sidecar",
        action="store_true",
        help=(
            "Run the local Unitree RL Gym G1 controller as a separate proof artifact. "
            "This does not replace the endpoint task rollout controller."
        ),
    )
    parser.add_argument("--unitree-controller-sidecar-steps", type=int, default=400)
    parser.add_argument(
        "--unitree-controller-sidecar-command-xyz",
        nargs=3,
        type=float,
        metavar=("VX_MPS", "VY_MPS", "YAW_RATE_RAD_S"),
        help="Optional command vector for the Unitree RL Gym controller sidecar.",
    )
    parser.add_argument(
        "--run-unitree-controller-replay-from-endpoint-actions",
        action="store_true",
        help=(
            "After endpoint actions are collected, replay a representative normalized "
            "endpoint command through the Unitree RL Gym controller as a bridge proof. "
            "This still does not replace the same-scene endpoint rollout controller."
        ),
    )
    parser.add_argument("--unitree-controller-replay-steps", type=int, default=400)
    parser.add_argument(
        "--controller-backend",
        choices=CONTROLLER_BACKENDS,
        default=DEFAULT_CONTROLLER_BACKEND,
        help=(
            "Physics controller backend for endpoint rollouts. "
            "auto selects the local Unitree RL Gym lower-body policy when its runtime snapshot "
            "is available and falls back to the legacy proxy otherwise. freejoint_proxy preserves "
            "the legacy fast lane; unitree_rl_gym fail-closes if the controller cannot load."
        ),
    )
    args = parser.parse_args(argv)
    effective_video_frame_stride_steps = (
        1 if args.capture_every_sim_step_review_video else args.video_frame_stride_steps
    )
    effective_fps = (
        int(args.fps)
        if args.fps is not None
        else (0 if args.capture_every_sim_step_review_video else DEFAULT_REVIEW_VIDEO_FPS)
    )
    summary = run_mujoco_g1_wam_vla_policy_endpoint_eval(
        job_dir=args.job_dir,
        job_root=args.job_root,
        g1_model_root=args.g1_model_root,
        task_filter=args.task_filter,
        spawn_filter=args.spawn_filter,
        max_tasks=args.max_tasks,
        max_spawns=args.max_spawns,
        steps_per_episode=args.steps_per_episode,
        policy_interval_steps=args.policy_interval_steps,
        render=not args.skip_render,
        render_frame_count=args.render_frame_count,
        video_frame_stride_steps=effective_video_frame_stride_steps,
        extend_terminal_frame_for_review=args.extend_terminal_frame_for_review,
        rendered_video_episode_limit=args.rendered_video_episode_limit,
        video_cameras=args.video_cameras,
        fps=effective_fps,
        endpoint_timeout_seconds=args.endpoint_timeout_seconds,
        max_contact_trace_rows=args.max_contact_trace_rows,
        allow_fetch_g1_assets=args.allow_fetch_g1_assets,
        menagerie_ref=args.menagerie_ref,
        unitree_rl_gym_root=args.unitree_rl_gym_root,
        run_official_unitree_controller_sidecar=args.run_official_unitree_controller_sidecar,
        unitree_controller_sidecar_steps=args.unitree_controller_sidecar_steps,
        unitree_controller_sidecar_command_xyz=args.unitree_controller_sidecar_command_xyz,
        run_unitree_controller_replay_from_endpoint_actions=(
            args.run_unitree_controller_replay_from_endpoint_actions
        ),
        unitree_controller_replay_steps=args.unitree_controller_replay_steps,
        controller_backend=args.controller_backend,
    )
    print(
        json.dumps(
            {
                "status": summary.get("status"),
                "job_dir": summary.get("job_dir"),
                "attempted_episode_count": summary.get("attempted_episode_count"),
                "successful_episode_count": summary.get("successful_episode_count"),
                "failed_episode_count": summary.get("failed_episode_count"),
                "blocked_episode_count": summary.get("blocked_episode_count"),
                "fixture_policy_used": summary.get("fixture_policy_used"),
                "endpoint_policy_used": summary.get("endpoint_policy_used"),
                "rejected_policy_action_count": summary.get("rejected_policy_action_count"),
            },
            sort_keys=True,
        )
    )
    return 0 if summary.get("status") == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
