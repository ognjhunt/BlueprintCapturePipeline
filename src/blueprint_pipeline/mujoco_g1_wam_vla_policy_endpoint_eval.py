"""MuJoCo Unitree G1 WAM/VLA policy-endpoint evaluation lane.

This lane is intentionally simulator-only. It tests policy endpoint discovery,
observation/action contracts, action normalization, MuJoCo execution, contact
traces, and WAM-style evaluator scoring without requiring Isaac Sim, splat/PLY
visuals, cloud GPUs, or physical robot controls.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from defusedxml import ElementTree as DefusedET

from .common import ensure_dir, utc_now_iso, write_json
from .g1_controlled_proof_setup import OFFICIAL_UNITREE_G1_POLICY_SOURCES
from .mujoco_g1_simulator_command import (
    DEFAULT_MENAGERIE_REF,
    _asset_source_manifest,
    _resolve_g1_model_root,
    _write_g1_xml_with_absolute_meshes,
)
from .policy_endpoint_boundary import build_policy_endpoint_boundary_manifest
from .provider_worker_contract import (
    PROVIDER_WORKER_CONTRACT_SCHEMA_VERSION,
    classify_policy_worker_command,
    write_provider_worker_contract,
)
from .unitree_lerobot_policy_runtime import (
    UnitreeLeRobotPolicyRuntimeConfig,
    build_policy_provider_registry_probe,
    build_unitree_policy_stack_installation_audit,
    run_unitree_lerobot_g1_policy_eval,
)
from .unitree_groot_n17_sonic_policy_runtime import (
    DEFAULT_EXPERIMENTAL_UNITREE_G1_SONIC_POLICY_CHECKPOINT,
    GROOT_ROOT_ENV,
    N17_CHECKPOINT_ENV,
    POLICY_COMMAND_ENV as GROOT_POLICY_COMMAND_ENV,
    POLICY_ID as GROOT_POLICY_ID,
    POLICY_SERVER_URL_ENV,
    SIM2SIM_COMMAND_ENV,
    SONIC_CHECKPOINT_ENV,
    WBC_ROOT_ENV,
    build_unitree_groot_n17_sonic_runtime_truth_boundary,
    probe_unitree_groot_n17_sonic_runtime,
    is_known_base_n17_without_unitree_g1_sonic_support,
    run_unitree_groot_n17_sonic_policy_runtime,
    select_unitree_g1_sonic_policy_checkpoint,
    unitree_g1_sonic_checkpoint_provenance,
)
from .unitree_groot_n17_sonic_policy_command_adapter import (
    PROVIDER_OUTPUT_ENV as GROOT_PROVIDER_OUTPUT_ENV,
    run_unitree_groot_n17_sonic_policy as run_groot_policy_command_adapter,
)
from .unitree_groot_n17_sonic_sim2sim_command import (
    run_unitree_groot_n17_sonic_sim2sim,
)
from .wam_generated_video_review import (
    validate_generated_mp4_for_review,
    visual_smoke_generated_rollouts_for_review,
)
from .wam_derived_observation_harness import (
    EXTERNAL_BACKEND_COMMAND_ENV,
    EXTERNAL_BACKEND_ENV_GATE,
    run_wam_derived_observation_harness_step,
    summarize_wam_derived_observation_artifacts,
    write_wam_derived_observation_artifacts,
)
from .wam_auxiliary_observation import (
    build_wam_auxiliary_observation_manifest,
    summarize_wam_auxiliary_observation_manifest,
)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    return value if math.isfinite(value) and value > 0.0 else float(default)


def _unitree_rl_gym_position_target_action_clip_abs() -> float:
    return _float_env(
        UNITREE_RL_GYM_POSITION_TARGET_ACTION_CLIP_ENV,
        DEFAULT_UNITREE_RL_GYM_POSITION_TARGET_ACTION_CLIP_ABS,
    )


LANE_SCHEMA_VERSION = "mujoco_g1_wam_vla_policy_endpoint_eval.v1"
SCENARIO_MATRIX_SCHEMA_VERSION = "mujoco_g1_wam_vla_scenario_eval_matrix.v1"
OBSERVATION_SCHEMA_ID = "blueprint.mujoco_g1_wam_vla.observation_packet.v1"
ACTION_SCHEMA_ID = "blueprint.mujoco_g1_wam_vla.action.v1"
REFERENCE_FIXTURE_POLICY_ID = "reference_fixture_policy"
ROBOT_PROFILE_ID = "unitree_g1_mujoco_menagerie"
DEFAULT_STEPS_PER_EPISODE = 3000
DEFAULT_WAM_LOOP_STEP_COUNT = 12
DEFAULT_WAM_GENERATION_TIMEOUT_SECONDS = 900.0
WAM_GENERATION_TIMEOUT_ENV = "BLUEPRINT_WAM_GENERATION_TIMEOUT_SECONDS"
DEFAULT_POLICY_ACTION_MODEL_COMMAND_TIMEOUT_SECONDS = 1800.0
POLICY_ACTION_MODEL_COMMAND_TIMEOUT_ENV = "BLUEPRINT_POLICY_ACTION_MODEL_COMMAND_TIMEOUT_SECONDS"
DEFAULT_VIDEO_FRAME_STRIDE_STEPS = 8
DEFAULT_REVIEW_VIDEO_FPS = 60
DEFAULT_EXTEND_TERMINAL_FRAME_FOR_REVIEW = False
DEFAULT_RENDERED_VIDEO_EPISODE_LIMIT = 8
DEFAULT_MAX_CONTACT_TRACE_ROWS = 50000
DEFAULT_CONTACT_OBSERVATION_RECORD_LIMIT = 24
EGOCENTRIC_VIDEO_CAMERAS = ("head_pov", "torso_pov", "robot_pov")
FIXED_G1_CAMERA_NAMES = {
    "head_pov": "blueprint_g1_head_pov",
    "torso_pov": "blueprint_g1_torso_pov",
    "robot_pov": "blueprint_g1_head_pov",
}
G1_PROJECTED_SKELETON_SCHEMA_ID = "blueprint.mujoco_g1.projected_upper_body_skeleton.v1"
G1_UPPER_BODY_LANDMARK_SPECS = (
    {"landmark_id": "left_shoulder", "body_name": "left_shoulder_pitch_link"},
    {"landmark_id": "left_elbow", "body_name": "left_elbow_link"},
    {"landmark_id": "left_wrist", "body_name": "left_wrist_yaw_link"},
    {
        "landmark_id": "left_hand",
        "body_name": "left_hand_palm_link",
        "fallback_body_name": "left_wrist_yaw_link",
        "fallback_local_offset_m": [0.082, 0.003, 0.0],
    },
    {"landmark_id": "right_shoulder", "body_name": "right_shoulder_pitch_link"},
    {"landmark_id": "right_elbow", "body_name": "right_elbow_link"},
    {"landmark_id": "right_wrist", "body_name": "right_wrist_yaw_link"},
    {
        "landmark_id": "right_hand",
        "body_name": "right_hand_palm_link",
        "fallback_body_name": "right_wrist_yaw_link",
        "fallback_local_offset_m": [0.082, -0.003, 0.0],
    },
)
G1_UPPER_BODY_SKELETON_SEGMENTS = (
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_hand"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_hand"),
)
AVAILABLE_VIDEO_CAMERAS = (
    "head_pov",
    "torso_pov",
    "robot_pov",
    "third_person",
    "overhead",
    "robot_follow",
)
DEFAULT_VIDEO_CAMERAS = ("head_pov", "torso_pov")
DIAGNOSTIC_VIDEO_CAMERAS = ("third_person", "robot_follow", "overhead")
PREFERRED_G1_POLICY_OBSERVATION_MJCF = "g1_with_hands.xml"
FALLBACK_G1_POLICY_OBSERVATION_MJCF = "g1.xml"
EGOCENTRIC_UPPER_BODY_OBSERVATION_POSE = {
    "left_shoulder_pitch_joint": -0.9,
    "right_shoulder_pitch_joint": -0.9,
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    "left_elbow_joint": 0.9,
    "right_elbow_joint": 0.9,
}
CONTROLLER_BACKENDS = ("auto", "freejoint_proxy", "unitree_rl_gym")
DEFAULT_CONTROLLER_BACKEND = "auto"
POLICY_ACTION_MODEL_COMMAND_GATE_ENV = "BLUEPRINT_ALLOW_POLICY_ACTION_MODEL_COMMAND"
SCENE_WAM_POLICY_EPISODE_PACKET_ENV = "BLUEPRINT_SCENE_WAM_POLICY_EPISODE_PACKET"
EXTERNAL_PHOTOREAL_OBSERVATION_FRAME_ENV = "BLUEPRINT_EXTERNAL_PHOTOREAL_OBSERVATION_FRAME"
EXTERNAL_PHOTOREAL_OBSERVATION_SOURCE_ENV = "BLUEPRINT_EXTERNAL_PHOTOREAL_OBSERVATION_SOURCE"
UNITREE_RL_GYM_SAME_SCENE_BACKEND_ID = "unitree_rl_gym_same_scene_lower_body_policy"
UNITREE_RL_GYM_CONTROLLER_COMMAND_LIMITS = {
    "max_forward_velocity_mps": 0.18,
    "max_reverse_velocity_mps": 0.06,
    "max_lateral_velocity_mps": 0.04,
    "max_yaw_rate_rad_s": 0.20,
}
UNITREE_RL_GYM_RESET_TO_POLICY_DEFAULT_LEG_POSE = True
UNITREE_RL_GYM_POSITION_TARGET_ACTION_CLIP_ENV = (
    "BLUEPRINT_UNITREE_RL_GYM_POSITION_TARGET_ACTION_CLIP_ABS"
)
DEFAULT_UNITREE_RL_GYM_POSITION_TARGET_ACTION_CLIP_ABS = 0.5
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


def _policy_action_model_command_timeout_seconds(endpoint_timeout_seconds: float) -> float:
    override = os.getenv(POLICY_ACTION_MODEL_COMMAND_TIMEOUT_ENV, "").strip()
    if override:
        return _float_env(
            POLICY_ACTION_MODEL_COMMAND_TIMEOUT_ENV,
            DEFAULT_POLICY_ACTION_MODEL_COMMAND_TIMEOUT_SECONDS,
        )
    try:
        endpoint_timeout = float(endpoint_timeout_seconds)
    except (TypeError, ValueError):
        endpoint_timeout = 0.0
    if math.isfinite(endpoint_timeout) and endpoint_timeout > 30.0:
        return endpoint_timeout
    return DEFAULT_POLICY_ACTION_MODEL_COMMAND_TIMEOUT_SECONDS


UNITREE_G1_SONIC_STATE_JOINT_GROUPS = {
    "left_leg": UNITREE_RL_GYM_LEG_JOINT_NAMES[:6],
    "right_leg": UNITREE_RL_GYM_LEG_JOINT_NAMES[6:],
    "waist": (
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ),
    "left_arm": (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ),
    "right_arm": (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ),
    "left_hand": (
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
    ),
    "right_hand": (
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
    ),
}
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
    {
        "name": "isaac_groot_n17",
        "url": "https://github.com/NVIDIA/Isaac-GR00T",
        "candidate_use": ("GR00T N1.7 VLA policy candidate for Unitree G1/SONIC action chunks."),
    },
    {
        "name": "groot_wholebodycontrol_sonic",
        "url": "https://github.com/NVlabs/GR00T-WholeBodyControl",
        "candidate_use": "SONIC whole-body control and MuJoCo Sim2Sim bridge for Unitree G1.",
    },
)
UNITREE_G1_MANIPULATION_POLICY_CANDIDATES = (
    {
        "id": GROOT_POLICY_ID,
        "name": "isaac_groot_n17_unitree_g1_sonic",
        "url": "https://github.com/NVIDIA/Isaac-GR00T",
        "runtime_role": "unitree_g1_sonic_groot_n17_vla_manipulation_policy",
        "expected_local_paths": ("Isaac-GR00T", "GR00T-WholeBodyControl"),
        "command_env": GROOT_POLICY_COMMAND_ENV,
        "checkpoint_env": N17_CHECKPOINT_ENV,
        "root_env": GROOT_ROOT_ENV,
        "extra_required_checkpoint_envs": (SONIC_CHECKPOINT_ENV,),
        "extra_required_root_envs": (WBC_ROOT_ENV,),
        "claim_boundary": (
            "requires Isaac-GR00T N1.7 checkpoint, SONIC checkpoint, PolicyServer/action "
            "wrapper, and simulator-only G1/SONIC execution"
        ),
    },
    {
        "id": "unitree_lerobot_g1_dex",
        "name": "unitree_lerobot",
        "url": "https://github.com/unitreerobotics/unitree_lerobot",
        "runtime_role": "g1_dexterous_or_gripper_imitation_policy",
        "expected_local_paths": ("unitree_lerobot",),
        "command_env": "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
        "checkpoint_env": "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT",
        "root_env": "BLUEPRINT_UNITREE_LEROBOT_ROOT",
        "claim_boundary": "requires task-specific trained policy plus hand/gripper action adapter",
    },
    {
        "id": "unitree_sim_isaaclab_g1_manipulation",
        "name": "unitree_sim_isaaclab",
        "url": "https://github.com/unitreerobotics/unitree_sim_isaaclab",
        "runtime_role": "isaaclab_g1_manipulation_policy_or_demo_weight",
        "expected_local_paths": ("unitree_sim_isaaclab",),
        "command_env": "BLUEPRINT_UNITREE_ISAACLAB_MANIPULATION_COMMAND",
        "checkpoint_env": "BLUEPRINT_UNITREE_ISAACLAB_MANIPULATION_CHECKPOINT",
        "root_env": "BLUEPRINT_UNITREE_ISAACLAB_ROOT",
        "claim_boundary": "requires IsaacLab runtime and task-specific manipulation policy execution",
    },
    {
        "id": "openvla_g1_manipulation",
        "name": "openvla_policy",
        "url": "https://github.com/openvla/openvla",
        "runtime_role": "vision_language_action_manipulation_policy",
        "expected_local_paths": ("openvla",),
        "command_env": "BLUEPRINT_OPENVLA_POLICY_COMMAND",
        "checkpoint_env": "BLUEPRINT_OPENVLA_POLICY_CHECKPOINT",
        "root_env": "BLUEPRINT_OPENVLA_POLICY_ROOT",
        "claim_boundary": "requires model endpoint response and Blueprint hand/action decoder",
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


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


UNITREE_ENDPOINT_HAND_POLICY_IDS = frozenset(
    {
        "unitree_lerobot_g1_policy",
        "unitree_unifolm_vla_policy",
        "unitree_unifolm_wma_policy",
        "unitree_groot_n17_sonic_policy",
    }
)

UNITREE_ENDPOINT_HAND_POLICY_REPLAY_IDS = frozenset(
    {
        "unitree_lerobot_g1_policy_provider_replay",
        "unitree_unifolm_vla_policy_provider_replay",
        "unitree_unifolm_wma_policy_provider_replay",
        "unitree_groot_n17_sonic_policy_provider_replay",
    }
)


def _unitree_endpoint_policy_response_summary(
    endpoint_policy_inner_responses: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    """Classify Unitree endpoint responses without treating replay as fresh policy use."""

    rows = [_mapping(row) for row in endpoint_policy_inner_responses]
    unitree_endpoint_hand_policy_output_observed = any(
        _string(row.get("policy_id"))
        in UNITREE_ENDPOINT_HAND_POLICY_IDS | UNITREE_ENDPOINT_HAND_POLICY_REPLAY_IDS
        or bool(_mapping(row.get("claim_boundary")).get("unitree_hand_manipulation_policy_used"))
        for row in rows
    )
    unitree_endpoint_provider_output_replay_used = any(
        bool(_mapping(row.get("claim_boundary")).get("provider_output_replay_used"))
        or _string(row.get("policy_id")).endswith("_provider_replay")
        for row in rows
    )
    unitree_endpoint_action_chunk_used = any(
        bool(_mapping(row.get("action")).get("unitree_unifolm_action_chunk_present"))
        or bool(_mapping(row.get("action")).get("unitree_unifolm_action_chunk"))
        or bool(_mapping(row.get("action")).get("unitree_groot_n17_sonic_action_chunk_present"))
        or bool(_mapping(row.get("action")).get("unitree_groot_n17_sonic_action_payload_present"))
        or bool(_mapping(row.get("action")).get("sonic_latent_action"))
        or bool(_mapping(row.get("action")).get("action_chunk"))
        for row in rows
    )
    unitree_endpoint_fresh_policy_action_command_ran = any(
        _string(row.get("policy_id")) in UNITREE_ENDPOINT_HAND_POLICY_IDS
        and not (
            bool(_mapping(row.get("claim_boundary")).get("provider_output_replay_used"))
            or _string(row.get("policy_id")).endswith("_provider_replay")
        )
        and (
            bool(row.get("unitree_policy_action_command_ran"))
            or bool(row.get("unitree_lerobot_policy_action_command_ran"))
            or bool(row.get("unitree_unifolm_policy_action_command_ran"))
            or bool(row.get("unitree_groot_n17_sonic_policy_action_command_ran"))
            or bool(
                _mapping(row.get("claim_boundary")).get("unitree_hand_manipulation_policy_used")
            )
        )
        for row in rows
    )
    return {
        "unitree_endpoint_hand_policy_output_observed": (
            unitree_endpoint_hand_policy_output_observed
        ),
        "unitree_endpoint_provider_output_replay_used": (
            unitree_endpoint_provider_output_replay_used
        ),
        "unitree_endpoint_action_chunk_used": unitree_endpoint_action_chunk_used,
        "unitree_endpoint_fresh_policy_action_command_ran": (
            unitree_endpoint_fresh_policy_action_command_ran
        ),
        "unitree_endpoint_hand_policy_used": (unitree_endpoint_fresh_policy_action_command_ran),
        "g1_robot_policy_selection_contract": "unitree_native_policy_required_for_g1_claims",
        "g1_robot_policy_selected_family": (
            "unitree_native_hand_policy_endpoint"
            if unitree_endpoint_fresh_policy_action_command_ran
            else None
        ),
        "unitree_hand_manipulation_policy_scope": (
            "endpoint_action_command" if unitree_endpoint_fresh_policy_action_command_ran else None
        ),
        "openvla_selected_as_g1_robot_policy": False,
        "wam_rollout_selected_as_g1_robot_policy": False,
    }


def _policy_action_provider_output_replay_used(
    *,
    policy_action_model_command_execution: Mapping[str, Any],
    robot_policy_wam_closed_loop_attempt: Mapping[str, Any],
) -> bool:
    return bool(
        policy_action_model_command_execution.get("provider_output_replay_used")
        or robot_policy_wam_closed_loop_attempt.get("provider_output_replay_used")
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        text = _string(value)
        return [text] if text else []
    if isinstance(value, Sequence):
        rows: list[str] = []
        for item in value:
            text = _string(item)
            if text:
                rows.append(text)
        return rows
    text = _string(value)
    return [text] if text else []


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if limit is not None and len(rows) >= limit:
                    break
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, Mapping):
                    rows.append(dict(value))
    except Exception:
        return rows
    return rows


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
        provenance = _endpoint_model_provenance(str(spec["runtime"]))
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
                **provenance,
            }
        )
    return rows


def _existing_env_path(env_name: str) -> Path | None:
    value = os.getenv(env_name, "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.exists() else None


def _endpoint_model_provenance(runtime: str) -> dict[str, Any]:
    command_env_by_runtime = {
        "wam": ("BLUEPRINT_OSCAR_WAM_COMMAND", "BLUEPRINT_COSMOS_WAM_COMMAND"),
        "vla": ("BLUEPRINT_OPENVLA_POLICY_COMMAND",),
        "team": ("BLUEPRINT_WAM_VLA_POLICY_COMMAND",),
    }
    checkpoint_env_by_runtime = {
        "wam": ("BLUEPRINT_OSCAR_WAM_CHECKPOINT", "BLUEPRINT_COSMOS_WAM_CHECKPOINT"),
        "vla": ("BLUEPRINT_OPENVLA_POLICY_CHECKPOINT",),
        "team": ("BLUEPRINT_POLICY_MODEL_CHECKPOINT",),
    }
    provider_output_env_by_runtime = {
        "wam": ("BLUEPRINT_OSCAR_WAM_PROVIDER_OUTPUT", "BLUEPRINT_COSMOS_WAM_PROVIDER_OUTPUT"),
        "vla": ("BLUEPRINT_OPENVLA_PROVIDER_OUTPUT",),
        "team": ("BLUEPRINT_POLICY_MODEL_PROVIDER_OUTPUT",),
    }
    configured_command_env = next(
        (
            env_name
            for env_name in command_env_by_runtime.get(runtime, ())
            if os.getenv(env_name, "").strip()
        ),
        None,
    )
    checkpoint_path = next(
        (
            path
            for env_name in checkpoint_env_by_runtime.get(runtime, ())
            for path in [_existing_env_path(env_name)]
            if path is not None
        ),
        None,
    )
    provider_output_path = next(
        (
            path
            for env_name in provider_output_env_by_runtime.get(runtime, ())
            for path in [_existing_env_path(env_name)]
            if path is not None
        ),
        None,
    )
    provenance_recorded = bool(configured_command_env and (checkpoint_path or provider_output_path))
    return {
        "model_command_env": configured_command_env,
        "model_command_configured": bool(configured_command_env),
        "model_checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "model_provider_output_path": str(provider_output_path) if provider_output_path else None,
        "model_provenance_recorded": provenance_recorded,
        "model_provenance_kind": (
            "provider_output_replay"
            if provider_output_path
            else "checkpoint"
            if checkpoint_path
            else None
        ),
        "model_provenance_claim_boundary": {
            "provider_output_replay_is_not_fresh_per_request_model_inference": bool(
                provider_output_path
            ),
            "model_provenance_is_not_task_success_proof": True,
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        },
    }


def discover_policy_runtime(
    *, generated_at: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
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
        "BLUEPRINT_UNITREE_LEROBOT_ROOT",
        "BLUEPRINT_UNITREE_LEROBOT_POLICY_PATH",
        "BLUEPRINT_UNITREE_POLICY_FAMILY",
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
            "runner_id": "blueprint_openvla_policy_command_adapter",
            "command": "blueprint-openvla-policy-command-adapter",
            "available_on_path": bool(shutil.which("blueprint-openvla-policy-command-adapter")),
            "runtime_kind": "openvla_policy_command_adapter_requires_checkpoint",
        },
        {
            "runner_id": "blueprint_run_unitree_lerobot_g1_policy_eval",
            "command": "blueprint-run-unitree-lerobot-g1-policy-eval",
            "available_on_path": bool(shutil.which("blueprint-run-unitree-lerobot-g1-policy-eval")),
            "runtime_kind": "unitree_lerobot_g1_sim_manipulation_policy_runtime",
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
    provider_registry = build_policy_provider_registry_probe(
        job_dir=_repo_root() / "robot_eval_jobs" / "unitree_lerobot_g1_policy_probe",
        generated_at=generated_at,
    )
    runtime_discovery = {
        "schema_version": "wam_vla_runtime_discovery.v1",
        "generated_at": generated_at,
        "status": "endpoint_ready" if ready_rows else "fixture_fallback_ready",
        "endpoint_runtimes": endpoint_rows,
        "policy_lane_provider_registry": provider_registry,
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
            "unitree_lerobot_g1",
            "unifolm_vla",
            "unifolm_wma",
            GROOT_POLICY_ID,
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
                "id": "unitree_lerobot_g1",
                "runtime_role": "unitree_g1_lerobot_sim_manipulation_or_loco_manip_policy",
                "root_env": "BLUEPRINT_UNITREE_LEROBOT_ROOT",
                "policy_path_env": "BLUEPRINT_UNITREE_LEROBOT_POLICY_PATH",
                "dataset_repo_env": "BLUEPRINT_UNITREE_LEROBOT_DATASET_REPO_ID",
                "default_runtime_mode": "probe",
                "configured": bool(
                    os.getenv("BLUEPRINT_UNITREE_LEROBOT_ROOT")
                    and os.getenv("BLUEPRINT_UNITREE_LEROBOT_POLICY_PATH")
                ),
                "claim_boundary": (
                    "separate LeRobot sim-eval provider; not official RL Gym locomotion "
                    "and not generated-world rank fidelity"
                ),
            },
            {
                "id": "openvla_policy",
                "runtime_role": "vla_or_imitation_policy_endpoint_candidate",
                "command_env": "BLUEPRINT_OPENVLA_POLICY_COMMAND",
                "checkpoint_env": "BLUEPRINT_OPENVLA_POLICY_CHECKPOINT",
                "default_adapter_command": "blueprint-openvla-policy-command-adapter",
                "default_adapter_available_on_path": bool(
                    shutil.which("blueprint-openvla-policy-command-adapter")
                ),
                "configured": bool(os.getenv("BLUEPRINT_OPENVLA_POLICY_COMMAND")),
                "claim_boundary": "requires_model_endpoint_response_and_action_decoder",
            },
            {
                "id": "openvla_endpoint",
                "runtime_role": "generic_vla_endpoint_comparison_only",
                "endpoint_env": "BLUEPRINT_UNITREE_OPENVLA_ENDPOINT_URL",
                "configured": bool(os.getenv("BLUEPRINT_UNITREE_OPENVLA_ENDPOINT_URL")),
                "g1_action_adapter_required": True,
                "claim_boundary": (
                    "generic_openvla_action_output_is_not_unitree_g1_control_without "
                    "an explicit G1 action adapter"
                ),
            },
            {
                "id": "unifolm_vla",
                "runtime_role": "unitree_native_vla_policy_candidate",
                "command_env": "BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND",
                "checkpoint_env": "BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT",
                "configured": bool(
                    os.getenv("BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND")
                    and os.getenv("BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT")
                    and os.getenv("BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT")
                ),
                "claim_boundary": "requires actual UnifoLM VLA command/checkpoint execution",
            },
            {
                "id": "unifolm_wma",
                "runtime_role": "unitree_native_world_model_action_candidate",
                "command_env": "BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND",
                "checkpoint_env": "BLUEPRINT_UNITREE_UNIFOLM_WMA_CHECKPOINT",
                "configured": bool(
                    os.getenv("BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND")
                    and os.getenv("BLUEPRINT_UNITREE_UNIFOLM_WMA_CHECKPOINT")
                ),
                "claim_boundary": "wam_world_model_used remains false until WMA runtime is invoked",
            },
            {
                "id": GROOT_POLICY_ID,
                "runtime_role": "unitree_native_groot_n17_sonic_vla_policy_candidate",
                "command_env": GROOT_POLICY_COMMAND_ENV,
                "checkpoint_env": N17_CHECKPOINT_ENV,
                "sonic_checkpoint_env": SONIC_CHECKPOINT_ENV,
                "source_root_env": GROOT_ROOT_ENV,
                "wbc_root_env": WBC_ROOT_ENV,
                "policy_server_url_env": POLICY_SERVER_URL_ENV,
                "sim2sim_command_env": SIM2SIM_COMMAND_ENV,
                "default_adapter_command": (
                    "blueprint-unitree-groot-n17-sonic-policy-command-adapter"
                ),
                "configured": bool(
                    os.getenv(GROOT_POLICY_COMMAND_ENV)
                    and os.getenv(N17_CHECKPOINT_ENV)
                    and os.getenv(SONIC_CHECKPOINT_ENV)
                ),
                "known_public_checkpoint_files": [
                    "nvidia/GR00T-N1.7-3B:<model repository root>",
                    (
                        DEFAULT_EXPERIMENTAL_UNITREE_G1_SONIC_POLICY_CHECKPOINT
                        + ":<experimental UNITREE_G1_SONIC GR00T N1.7 checkpoint>"
                    ),
                    "nvidia/GEAR-SONIC:gear_sonic_deploy/policy/model_encoder.onnx",
                    "nvidia/GEAR-SONIC:gear_sonic_deploy/policy/model_decoder.onnx",
                    "nvidia/GEAR-SONIC:sonic_release/last.pt",
                ],
                "embodiment_tag": "UNITREE_G1_SONIC",
                "expected_action_dimension": 78,
                "claim_boundary": {
                    "checkpoint_presence_is_not_endpoint_execution": True,
                    "groot_n17_does_not_replace_unitree_rl_gym_locomotion_proof": True,
                    "requires_policy_server_action_wrapper_and_sim2sim_execution": True,
                    "generated_world_rank_fidelity_result_proven": False,
                    "generated_world_policy_evaluation_scope_proven": False,
                    "non_ranking_operational_claim_proven": False,
                },
            },
            {
                "id": "oscar_wam",
                "runtime_role": "action_conditioned_world_model_rollout_generator",
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
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
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
    missing = [name for name, path in required.items() if not path.expanduser().resolve().is_file()]
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
        "BLUEPRINT_UNITREE_G1_POLICY_SOURCE_ROOT",
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
            available = bool(
                executable and (Path(executable).expanduser().is_file() or shutil.which(executable))
            )
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
        blockers.append(
            "blocked_controller_command_not_integrated_into_same_scene_endpoint_rollouts"
        )
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
            "generated_world_rank_fidelity_result_proven": False,
        },
    }


def _command_available(command_value: str) -> bool:
    if not command_value:
        return False
    try:
        parts = shlex.split(command_value)
    except ValueError:
        return False
    executable = parts[0] if parts else ""
    return bool(
        executable and (Path(executable).expanduser().is_file() or shutil.which(executable))
    )


POLICY_ACTION_MODEL_COMMAND_CANDIDATES = (
    {
        "candidate_id": GROOT_POLICY_ID,
        "command_envs": (GROOT_POLICY_COMMAND_ENV,),
        "default_command_value": (
            f"{shlex.quote(sys.executable)} -m "
            "blueprint_pipeline.unitree_groot_n17_sonic_policy_server_command"
        ),
        "checkpoint_envs": (N17_CHECKPOINT_ENV,),
        "extra_required_checkpoint_envs": (),
        "optional_checkpoint_envs": (SONIC_CHECKPOINT_ENV,),
        "source_root_env": GROOT_ROOT_ENV,
        "extra_required_root_envs": (WBC_ROOT_ENV,),
        "runtime_role": "unitree_groot_n17_sonic_policy_action_model",
    },
    {
        "candidate_id": "unitree_g1_policy",
        "command_envs": (
            "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
            "BLUEPRINT_REALISTIC_G1_POLICY_COMMAND",
        ),
        "checkpoint_envs": ("BLUEPRINT_UNITREE_G1_POLICY_CHECKPOINT",),
        "runtime_role": "unitree_g1_policy_action_model",
    },
    {
        "candidate_id": "unitree_lerobot_policy",
        "command_envs": ("BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",),
        "checkpoint_envs": ("BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT",),
        "runtime_role": "unitree_g1_manipulation_policy_action_model",
    },
    {
        "candidate_id": "unitree_unifolm_vla_policy",
        "command_envs": ("BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND",),
        "checkpoint_envs": ("BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT",),
        "extra_required_checkpoint_envs": ("BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT",),
        "runtime_role": "unitree_native_vla_policy_action_model",
    },
    {
        "candidate_id": "unitree_unifolm_wma_policy",
        "command_envs": ("BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND",),
        "checkpoint_envs": ("BLUEPRINT_UNITREE_UNIFOLM_WMA_CHECKPOINT",),
        "runtime_role": "unitree_native_wma_policy_action_model",
    },
)

UNITREE_POLICY_ACTION_MODEL_CANDIDATE_IDS = {
    "unitree_g1_policy",
    "unitree_lerobot_policy",
    "unitree_unifolm_vla_policy",
    "unitree_unifolm_wma_policy",
    GROOT_POLICY_ID,
}
UNITREE_MANIPULATION_POLICY_ACTION_MODEL_CANDIDATE_IDS = {
    "unitree_lerobot_policy",
    "unitree_unifolm_vla_policy",
    "unitree_unifolm_wma_policy",
    GROOT_POLICY_ID,
}

GENERIC_POLICY_ACTION_MODEL_COMPARISON_CANDIDATES = (
    {
        "candidate_id": "openvla_policy",
        "command_envs": ("BLUEPRINT_OPENVLA_POLICY_COMMAND",),
        "checkpoint_envs": ("BLUEPRINT_OPENVLA_POLICY_CHECKPOINT",),
        "runtime_role": "generic_vla_policy_comparison_only",
        "not_selected_reason": "generic_openvla_is_not_the_default_unitree_g1_policy_path",
    },
)


def _is_repo_id_reference(value: str) -> bool:
    text = value.strip()
    if not text or text.startswith(("/", "./", "../", "~")):
        return False
    parts = text.split("/")
    return (
        len(parts) >= 2
        and all(part.strip() for part in parts[:2])
        and not any(part in {".", ".."} for part in parts[:2])
        and " " not in text
    )


def _configured_checkpoint_reference(value: str) -> tuple[bool, str | None, bool, str | None]:
    text = value.strip()
    if not text:
        return False, None, False, None
    path = Path(text).expanduser()
    if path.exists():
        return True, str(path), True, "local_path"
    if _is_repo_id_reference(text):
        return True, text, False, "repo_id"
    return False, str(path), False, "missing_path"


def discover_policy_action_model_commands(*, generated_at: str) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    comparison_candidates: list[dict[str, Any]] = []

    def _candidate_row(spec: Mapping[str, Any], *, selectable: bool) -> dict[str, Any]:
        command_env = ""
        command_value = ""
        for env_name in spec["command_envs"]:
            value = os.getenv(str(env_name), "").strip()
            if value:
                command_env = str(env_name)
                command_value = value
                break
        command_default_applied = False
        default_command_value = _string(spec.get("default_command_value"))
        if not command_value and default_command_value:
            command_env = str(list(spec["command_envs"])[0])
            command_value = default_command_value
            command_default_applied = True
        checkpoint_env = ""
        checkpoint_value = ""
        for env_name in spec["checkpoint_envs"]:
            value = os.getenv(str(env_name), "").strip()
            if value:
                checkpoint_env = str(env_name)
                checkpoint_value = value
                break
        checkpoint_original_value = checkpoint_value
        checkpoint_default_applied = False
        checkpoint_selection_source = "configured_checkpoint"
        if spec["candidate_id"] == GROOT_POLICY_ID:
            (
                checkpoint_value,
                checkpoint_selection_source,
                checkpoint_default_applied,
            ) = select_unitree_g1_sonic_policy_checkpoint(checkpoint_original_value)
        (
            checkpoint_configured,
            checkpoint_reference,
            checkpoint_exists,
            checkpoint_reference_kind,
        ) = _configured_checkpoint_reference(checkpoint_value)
        command_available = _command_available(command_value)
        policy_worker_contract = classify_policy_worker_command(command_value)
        blockers: list[str] = []
        if not command_value:
            blockers.append("blocked_missing_policy_action_model_command")
        elif not command_available:
            blockers.append("blocked_policy_action_model_command_unavailable")
        if command_value and not policy_worker_contract.get("repeated_policy_loop_allowed"):
            blockers.extend(
                str(item) for item in policy_worker_contract.get("blockers", []) if str(item)
            )
        if not checkpoint_value:
            blockers.append("blocked_missing_policy_action_model_checkpoint")
        elif not checkpoint_configured:
            blockers.append("blocked_policy_action_model_checkpoint_missing")
        elif spec[
            "candidate_id"
        ] == GROOT_POLICY_ID and is_known_base_n17_without_unitree_g1_sonic_support(
            checkpoint_original_value
        ):
            # The base checkpoint is incompatible with UNITREE_G1_SONIC, but
            # this candidate can fall back to the experimental SONIC fine-tune.
            # Keep the fact visible without blocking admission.
            pass
        extra_checkpoint_rows = []
        for env_name in spec.get("extra_required_checkpoint_envs", ()):
            value = os.getenv(str(env_name), "").strip()
            configured, path_text, exists, reference_kind = _configured_checkpoint_reference(value)
            extra_checkpoint_rows.append(
                {
                    "checkpoint_env": str(env_name),
                    "checkpoint_configured": configured,
                    "checkpoint_exists": exists,
                    "checkpoint_path": path_text,
                    "checkpoint_reference_kind": reference_kind,
                }
            )
            if not value:
                blockers.append(f"blocked_missing_{env_name}")
            elif not configured:
                blockers.append(f"blocked_missing_path_for_{env_name}")
        optional_checkpoint_rows = []
        for env_name in spec.get("optional_checkpoint_envs", ()):
            value = os.getenv(str(env_name), "").strip()
            configured, path_text, exists, reference_kind = _configured_checkpoint_reference(value)
            optional_checkpoint_rows.append(
                {
                    "checkpoint_env": str(env_name),
                    "checkpoint_configured": configured,
                    "checkpoint_exists": exists,
                    "checkpoint_path": path_text,
                    "checkpoint_reference_kind": reference_kind,
                    "required_for_policy_action_admission": False,
                }
            )
        source_root_env = _string(spec.get("source_root_env"))
        source_root_value = os.getenv(source_root_env, "").strip() if source_root_env else ""
        source_root_path = Path(source_root_value).expanduser() if source_root_value else None
        source_root_exists = bool(source_root_path and source_root_path.exists())
        if source_root_value and not source_root_exists:
            blockers.append(f"blocked_missing_path_for_{source_root_env}")
        extra_root_rows = []
        for env_name in spec.get("extra_required_root_envs", ()):
            value = os.getenv(str(env_name), "").strip()
            path = Path(value).expanduser() if value else None
            exists = bool(path and path.exists())
            extra_root_rows.append(
                {
                    "root_env": str(env_name),
                    "root_configured": bool(value),
                    "root_path": str(path) if path else None,
                    "root_exists": exists,
                }
            )
            if value and not exists:
                blockers.append(f"blocked_missing_path_for_{env_name}")
        if spec["candidate_id"] == GROOT_POLICY_ID and command_default_applied:
            explicit_groot_setup = bool(
                checkpoint_original_value
                or source_root_value
                or any(row.get("root_configured") for row in extra_root_rows)
                or any(row.get("checkpoint_configured") for row in optional_checkpoint_rows)
            )
            if not explicit_groot_setup:
                blockers.append(
                    "blocked_missing_explicit_unitree_groot_n17_sonic_runtime_configuration"
                )
        ready = bool(
            command_value
            and command_available
            and checkpoint_configured
            and not blockers
            and selectable
        )
        row = {
            "candidate_id": spec["candidate_id"],
            "runtime_role": spec["runtime_role"],
            "unitree_specific_policy_candidate": selectable,
            "command_env": command_env or list(spec["command_envs"])[0],
            "command_configured": bool(command_value),
            "command_from_default": command_default_applied,
            "command_available": command_available,
            "command_value_redacted": "<configured>" if command_value else None,
            "command_value_for_execution": command_value if command_default_applied else None,
            "checkpoint_env": checkpoint_env or list(spec["checkpoint_envs"])[0],
            "checkpoint_configured": checkpoint_configured,
            "checkpoint_exists": checkpoint_exists,
            "checkpoint_path": checkpoint_reference,
            "checkpoint_reference_kind": checkpoint_reference_kind,
            "checkpoint_original_env_reference": checkpoint_original_value or None,
            "checkpoint_selection_source": checkpoint_selection_source,
            "checkpoint_default_applied": checkpoint_default_applied,
            "checkpoint_known_base_model_without_unitree_g1_sonic_support": bool(
                spec["candidate_id"] == GROOT_POLICY_ID
                and is_known_base_n17_without_unitree_g1_sonic_support(checkpoint_original_value)
            ),
            "checkpoint_provenance": unitree_g1_sonic_checkpoint_provenance(checkpoint_value)
            if spec["candidate_id"] == GROOT_POLICY_ID
            else None,
            "trusted_for_production": False,
            "task_specific_finetuning_required_for_admission": False,
            "unitree_g1_sonic_requires_finetuned_gr00t_checkpoint": bool(
                spec["candidate_id"] == GROOT_POLICY_ID
            ),
            "extra_required_checkpoints": extra_checkpoint_rows,
            "optional_checkpoints": optional_checkpoint_rows,
            "source_root_env": source_root_env or None,
            "source_root_configured": bool(source_root_value),
            "source_root_path": str(source_root_path) if source_root_path else None,
            "source_root_exists": source_root_exists,
            "extra_required_roots": extra_root_rows,
            "provider_worker_contract_schema_version": PROVIDER_WORKER_CONTRACT_SCHEMA_VERSION,
            "policy_worker_contract": policy_worker_contract,
            "policy_worker_invocation_kind": policy_worker_contract.get("invocation_kind"),
            "repeated_policy_loop_allowed": bool(
                policy_worker_contract.get("repeated_policy_loop_allowed")
            ),
            "provider_instance_launch_per_inference": policy_worker_contract.get(
                "provider_instance_launch_per_inference"
            ),
            "ready_for_policy_action_command": ready,
            "blockers": blockers,
        }
        if not selectable:
            row["ready_for_policy_action_command"] = False
            row["not_selected_reason"] = spec.get(
                "not_selected_reason",
                "generic_policy_comparison_candidate_not_selected_for_unitree_g1",
            )
            if command_value:
                row["blockers"] = sorted(
                    set(blockers + ["blocked_generic_policy_not_unitree_specific"])
                )
        return row

    for spec in POLICY_ACTION_MODEL_COMMAND_CANDIDATES:
        candidates.append(_candidate_row(spec, selectable=True))
    for spec in GENERIC_POLICY_ACTION_MODEL_COMPARISON_CANDIDATES:
        comparison_candidates.append(_candidate_row(spec, selectable=False))

    ready = sorted(
        [row for row in candidates if row["ready_for_policy_action_command"]],
        key=lambda row: bool(row.get("command_from_default")),
    )
    configured_manipulation_candidate = next(
        (
            row
            for row in candidates
            if row["candidate_id"] in UNITREE_MANIPULATION_POLICY_ACTION_MODEL_CANDIDATE_IDS
            and (
                (row.get("command_configured") and not row.get("command_from_default"))
                or (
                    row.get("checkpoint_configured")
                    and (
                        not row.get("checkpoint_default_applied")
                        or row.get("checkpoint_original_env_reference")
                    )
                )
                or row.get("source_root_configured")
                or any(
                    item.get("checkpoint_configured")
                    for item in row.get("extra_required_checkpoints", [])
                )
                or any(item.get("root_configured") for item in row.get("extra_required_roots", []))
            )
        ),
        None,
    )
    selected_candidate_id = (
        ready[0]["candidate_id"]
        if ready
        else (
            configured_manipulation_candidate["candidate_id"]
            if configured_manipulation_candidate
            else None
        )
    )
    selected_candidate_blockers = (
        list(configured_manipulation_candidate.get("blockers", []))
        if configured_manipulation_candidate and not ready
        else []
    )
    return {
        "schema_version": "policy_action_model_command_discovery.v1",
        "generated_at": generated_at,
        "provider_worker_contract_schema_version": PROVIDER_WORKER_CONTRACT_SCHEMA_VERSION,
        "status": "ready" if ready else "blocked_missing_unitree_policy_action_model_command",
        "selection_policy": "unitree_specific_policy_candidates_only",
        "selected_candidate_id": selected_candidate_id,
        "selected_candidate_ready_for_policy_action_command": bool(ready),
        "candidates": candidates,
        "generic_policy_comparison_candidates": comparison_candidates,
        "ready_candidate_count": len(ready),
        "blockers": []
        if ready
        else sorted(
            set(
                ["blocked_missing_unitree_specific_policy_action_model_command"]
                + selected_candidate_blockers
            )
        ),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


def _policy_action_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    action = payload.get("normalized_action") or payload.get("action")
    if isinstance(action, Mapping):
        return dict(action)
    for key in ("action_chunk", "actions", "action_vector", "joint_targets", "joint_positions"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return {
                "action_type": "manipulation_contact",
                "unitree_action_chunk_present": True,
                "unitree_raw_action_key": key,
                key: dict(value),
            }
        if isinstance(value, (list, tuple)):
            return {
                "action_type": "manipulation_contact",
                "unitree_action_chunk_present": True,
                "unitree_raw_action_key": key,
                key: list(value),
            }
    return None


def _policy_action_command_blockers(
    command_result: Mapping[str, Any],
    response_payload: Mapping[str, Any],
) -> list[str]:
    blockers = [str(item) for item in command_result.get("blockers", []) if str(item)]
    response_blockers = response_payload.get("blockers")
    if isinstance(response_blockers, Sequence) and not isinstance(
        response_blockers, (str, bytes, bytearray)
    ):
        blockers.extend(str(item) for item in response_blockers if str(item))
    elif response_blockers:
        blockers.append(str(response_blockers))
    status = _string(response_payload.get("status"))
    if status in {"blocked", "failed"} and not response_blockers:
        blockers.append(f"policy_action_model_command_output_{status}")
    return sorted(set(blockers))


def _write_blocked_policy_action_model_command_output(
    output_path: Path,
    *,
    generated_at: str,
    selected_candidate_id: str | None,
    blockers: Sequence[str],
    command_result: Mapping[str, Any] | None = None,
    response_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    blocked_output = {
        "schema_version": "policy_action_model_command_output.v1",
        "generated_at": generated_at,
        "status": "blocked",
        "selected_candidate_id": selected_candidate_id,
        "policy_action_model_command_ran": False,
        "action_payload_present": False,
        "unitree_policy_action_command_ran": False,
        "unitree_lerobot_policy_action_command_ran": False,
        "unitree_unifolm_policy_action_command_ran": False,
        "unitree_groot_n17_sonic_policy_action_command_ran": False,
        "unitree_specific_manipulation_candidate_ran": False,
        "openvla_policy_action_command_ran": False,
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "command_result": _redact(command_result or {}),
        "response_redacted": _redact(response_payload or {}),
        "claim_boundary": {
            "blocked_output_is_not_model_proof": True,
            "policy_action_command_is_model_contract_probe_not_wam_rollout": True,
            "policy_action_command_does_not_prove_task_success": True,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(output_path, blocked_output)
    return blocked_output


def _unitree_policy_action_execution_flags(
    *,
    ran: bool,
    selected_candidate_id: str | None,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    selected = selected_candidate_id or ""
    unitree_selected = selected in UNITREE_POLICY_ACTION_MODEL_CANDIDATE_IDS
    manipulation_selected = selected in UNITREE_MANIPULATION_POLICY_ACTION_MODEL_CANDIDATE_IDS
    provider_output_replay_used = bool(
        payload.get("provider_output_replay_used")
        or _mapping(payload.get("claim_boundary")).get("provider_output_replay_used")
    )
    unitree_g1_policy_ran = bool(
        ran and selected == "unitree_g1_policy" and not provider_output_replay_used
    )
    unitree_lerobot_policy_ran = bool(
        ran
        and selected == "unitree_lerobot_policy"
        and not provider_output_replay_used
        and payload.get("unitree_lerobot_policy_action_command_ran", True)
    )
    unitree_unifolm_policy_ran = bool(
        ran
        and selected in {"unitree_unifolm_vla_policy", "unitree_unifolm_wma_policy"}
        and not provider_output_replay_used
        and payload.get("unitree_unifolm_policy_action_command_ran", True)
    )
    unitree_groot_policy_ran = bool(
        ran
        and selected == GROOT_POLICY_ID
        and not provider_output_replay_used
        and payload.get("unitree_groot_n17_sonic_policy_action_command_ran", True)
    )
    unitree_policy_ran = bool(
        unitree_g1_policy_ran
        or unitree_lerobot_policy_ran
        or unitree_unifolm_policy_ran
        or unitree_groot_policy_ran
    )
    unitree_manipulation_policy_ran = bool(
        unitree_lerobot_policy_ran or unitree_unifolm_policy_ran or unitree_groot_policy_ran
    )
    return {
        "unitree_policy_action_command_ran": unitree_policy_ran,
        "unitree_lerobot_policy_action_command_ran": unitree_lerobot_policy_ran,
        "unitree_unifolm_policy_action_command_ran": unitree_unifolm_policy_ran,
        "unitree_groot_n17_sonic_policy_action_command_ran": unitree_groot_policy_ran,
        "unitree_manipulation_policy_action_command_ran": bool(
            manipulation_selected and unitree_manipulation_policy_ran
        ),
        "unitree_specific_policy_candidate_ran": bool(unitree_selected and unitree_policy_ran),
        "unitree_specific_manipulation_candidate_ran": bool(
            manipulation_selected and unitree_manipulation_policy_ran
        ),
    }


def _policy_action_model_frame_candidates(job_dir: Path) -> list[Path]:
    frame_root = job_dir / "policy_observation_frames"
    if not frame_root.is_dir():
        return []
    camera_rank = {
        "head_pov": 0,
        "blueprint_g1_head_pov": 0,
        "torso_pov": 1,
        "blueprint_g1_torso_pov": 1,
        "robot_pov": 2,
    }
    candidates = [
        path
        for path in frame_root.rglob("*.jpg")
        if path.is_file() and _policy_action_model_frame_camera_id(path) in camera_rank
    ]

    def rank(path: Path) -> tuple[int, str]:
        camera_id = _policy_action_model_frame_camera_id(path)
        return camera_rank.get(camera_id or "", 99), path.as_posix()

    return sorted(candidates, key=rank)


def _external_photoreal_observation_frame() -> tuple[Path | None, str | None]:
    frame_text = os.getenv(EXTERNAL_PHOTOREAL_OBSERVATION_FRAME_ENV, "").strip()
    if not frame_text:
        return None, None
    frame = Path(frame_text).expanduser()
    if not frame.is_file():
        return None, None
    source = os.getenv(EXTERNAL_PHOTOREAL_OBSERVATION_SOURCE_ENV, "").strip()
    return frame.resolve(), source or "external_photoreal_frame"


def _policy_action_model_frame_camera_id(path: Path | str | None) -> str | None:
    if path is None:
        return None
    text = Path(path).as_posix()
    for camera in (
        "blueprint_g1_head_pov",
        "blueprint_g1_torso_pov",
        "head_pov",
        "torso_pov",
        "robot_pov",
    ):
        if camera in text:
            return (
                "head_pov"
                if camera == "blueprint_g1_head_pov"
                else ("torso_pov" if camera == "blueprint_g1_torso_pov" else camera)
            )
    return None


def _unitree_g1_sonic_contract_probe_state() -> dict[str, list[float]]:
    return {
        "left_leg": [0.0] * 6,
        "right_leg": [0.0] * 6,
        "waist": [0.0] * 3,
        "left_arm": [0.0] * 7,
        "right_arm": [0.0] * 7,
        "left_hand": [0.0] * 7,
        "right_hand": [0.0] * 7,
        "projected_gravity": [0.0, 0.0, -1.0],
    }


def _first_unitree_g1_sonic_state_from_visual_trace(
    job_dir: Path | None,
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
    if job_dir is None:
        return None, None, {}
    trace_path = job_dir / "policy_visual_observation_trace.jsonl"
    if not trace_path.is_file():
        return None, None, {}
    try:
        with trace_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping) or not row.get("available"):
                    continue
                state = row.get("unitree_g1_sonic_state")
                if not isinstance(state, Mapping):
                    continue
                return (
                    dict(state),
                    _string(row.get("unitree_g1_sonic_state_source"))
                    or "simulated_mujoco_joint_groups",
                    {
                        "episode_id": row.get("episode_id"),
                        "scenario_eval_run_id": row.get("scenario_eval_run_id"),
                        "step": row.get("step"),
                        "camera_frame_path": row.get("camera_frame_path"),
                        "metadata": row.get("unitree_g1_sonic_state_metadata"),
                    },
                )
    except Exception:
        return None, None, {}
    return None, None, {}


def _scene_wam_policy_episode_packet_candidates(job_dir: Path | None) -> list[Path]:
    candidates: list[Path] = []
    env_value = os.getenv(SCENE_WAM_POLICY_EPISODE_PACKET_ENV, "").strip()
    if env_value:
        candidates.append(Path(env_value).expanduser())
    if job_dir is not None:
        candidates.extend(
            [
                job_dir / "scene_wam_policy_episode_packet.json",
                job_dir / "scene_episode_packet" / "scene_wam_policy_episode_packet.json",
            ]
        )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _resolve_artifact_path(value: Any, *, base_dir: Path) -> Path | None:
    text = _string(value)
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def _scene_policy_task_prompt(
    *,
    task_id: str | None,
    target_object_id: str | None,
) -> str:
    task = task_id or "scene manipulation task"
    target = target_object_id or "the target object"
    if task == "turn_on_sink_handle":
        return (
            "Attempt the simulator-only kitchen task: turn on the sink handle. "
            f"Use the Unitree G1/SONIC observation to actuate {target}, and return "
            "a Unitree G1 SONIC-compatible action chunk or control target."
        )
    return (
        f"Attempt the simulator-only scene task '{task}' on target '{target}'. "
        "Return a Unitree G1 SONIC-compatible action chunk or control target."
    )


def _scene_packet_policy_action_model_input(
    *,
    generated_at: str,
    job_dir: Path | None,
) -> dict[str, Any] | None:
    for packet_path in _scene_wam_policy_episode_packet_candidates(job_dir):
        if not packet_path.is_file():
            continue
        try:
            packet_value = json.loads(packet_path.read_text(encoding="utf-8"))
            packet = _mapping(packet_value)
            observation_path = _resolve_artifact_path(
                packet.get("initial_policy_observation_path"),
                base_dir=packet_path.parent,
            ) or (packet_path.parent / "initial_policy_observation.json")
            if not observation_path.is_file():
                continue
            observation_value = json.loads(observation_path.read_text(encoding="utf-8"))
            observation = _mapping(observation_value)
        except Exception:
            continue
        observation.setdefault("schema_version", "blueprint_policy_observation.v1")
        frame_path = _resolve_artifact_path(
            observation.get("camera_frame_path")
            or _mapping(observation.get("visual_observation")).get("camera_frame_path")
            or packet.get("initial_policy_observation_frame_path"),
            base_dir=observation_path.parent,
        )
        external_frame, external_source = _external_photoreal_observation_frame()
        external_photoreal = external_frame is not None
        if external_photoreal:
            frame_path = external_frame
        visual_observation = _mapping(observation.get("visual_observation"))
        frame_available = bool(frame_path and frame_path.is_file())
        if frame_path is not None:
            observation["camera_frame_path"] = str(frame_path)
            visual_observation["camera_frame_path"] = str(frame_path)
        capture_derived_synthetic = bool(
            visual_observation.get("capture_derived_robot_pov_synthesis_used")
            or observation.get("capture_derived_robot_pov_frame_path")
            or visual_observation.get("capture_derived_robot_pov_frame_path")
        )
        visual_observation["available"] = frame_available
        visual_observation.setdefault(
            "camera_id",
            _policy_action_model_frame_camera_id(frame_path) or "head_pov",
        )
        visual_observation["first_person_policy_observation_candidate"] = frame_available
        visual_observation["simulated_camera_view"] = (
            frame_available and not capture_derived_synthetic and not external_photoreal
        )
        visual_observation["capture_derived_robot_pov_synthesis_used"] = (
            frame_available and capture_derived_synthetic and not external_photoreal
        )
        visual_observation["synthesized_or_splatted_outputs_are_not_raw_capture_truth"] = True
        visual_observation["physical_robot_sensor_proof"] = False
        if external_photoreal:
            visual_observation["external_photoreal_observation_used"] = True
            visual_observation["photoreal_observation_source"] = external_source
        visual_observation["blockers"] = (
            [] if frame_available else ["scene_packet_policy_observation_frame_missing"]
        )
        observation["visual_observation"] = visual_observation
        task_id = _string(observation.get("task_id") or packet.get("task_id"))
        target_object_id = _string(
            observation.get("target_object_id") or packet.get("target_object_id")
        )
        task_prompt = _string(observation.get("task_prompt")) or _scene_policy_task_prompt(
            task_id=task_id,
            target_object_id=target_object_id,
        )
        observation["task_id"] = task_id or observation.get("task_id")
        observation["target_object_id"] = target_object_id or observation.get("target_object_id")
        observation["task_prompt"] = task_prompt
        observation.setdefault("unitree_g1_sonic_state", _unitree_g1_sonic_contract_probe_state())
        observation.setdefault(
            "unitree_g1_sonic_state_source",
            "scene_packet_contract_probe_zero_state",
        )
        observation["scene_wam_policy_episode_packet_path"] = str(packet_path)
        observation["initial_policy_observation_path"] = str(observation_path)
        claim_boundary = {
            "sample_input_is_scene_packet_not_task_success_evidence": True,
            "visual_frame_is_simulated_mujoco_policy_observation": (
                frame_available and not capture_derived_synthetic and not external_photoreal
            ),
            "visual_frame_is_capture_derived_synthetic_robot_pov": (
                frame_available and capture_derived_synthetic and not external_photoreal
            ),
            "visual_frame_is_raw_capture_truth": False,
            "synthesized_or_splatted_outputs_are_not_raw_capture_truth": True,
            "unitree_g1_sonic_state_is_simulated_observation": True,
            "unitree_g1_sonic_state_is_contract_probe": (
                observation.get("unitree_g1_sonic_state_source")
                == "scene_packet_contract_probe_zero_state"
            ),
            "task_specific_finetuning_required_for_admission": False,
            "policy_action_command_does_not_prove_generated_world_rank_fidelity": True,
        }
        if external_photoreal:
            claim_boundary["visual_frame_is_external_photoreal_handoff"] = True
            claim_boundary["mujoco_owns_physics_external_lane_owns_pixels"] = True
        return {
            "schema_version": "policy_action_model_command_input.v1",
            "generated_at": generated_at,
            "robot_profile_id": ROBOT_PROFILE_ID,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
            "task_prompt": task_prompt,
            "observation": observation,
            "allowed_action_types": [
                "waypoint",
                "base_velocity",
                "stop",
                "inspect_look",
                "manipulation_contact",
                "joint_targets",
                "action_chunk",
            ],
            "scene_wam_policy_episode_packet_path": str(packet_path),
            "claim_boundary": claim_boundary,
        }
    return None


def _policy_action_scene_task(job_dir: Path) -> dict[str, Any]:
    input_path = job_dir / "policy_action_model_command_input.json"
    if not input_path.is_file():
        return {}
    try:
        value = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    packet = _mapping(value)
    observation = _mapping(packet.get("observation"))
    return {
        "task_id": _string(packet.get("task_id") or observation.get("task_id")),
        "target_object_id": _string(
            packet.get("target_object_id") or observation.get("target_object_id")
        ),
        "scene_wam_policy_episode_packet_path": _string(
            packet.get("scene_wam_policy_episode_packet_path")
            or observation.get("scene_wam_policy_episode_packet_path")
        ),
    }


def _final_success_question_for_scene_task(scene_task: Mapping[str, Any]) -> tuple[str, str | None]:
    task_id = _string(scene_task.get("task_id"))
    target_object_id = _string(scene_task.get("target_object_id")).lower()
    if task_id == "turn_on_sink_handle" or (
        "sink" in target_object_id and "handle" in target_object_id
    ):
        return "Did the sink handle end up turned on?", "sink_handle_turned_on"
    return "Did the object/tote end up correctly placed?", None


def _sample_policy_action_model_input(
    *,
    generated_at: str,
    job_dir: Path | None = None,
) -> dict[str, Any]:
    scene_packet_input = _scene_packet_policy_action_model_input(
        generated_at=generated_at,
        job_dir=job_dir,
    )
    if scene_packet_input is not None:
        return scene_packet_input
    frame_path = None
    if job_dir is not None:
        frame_candidates = _policy_action_model_frame_candidates(job_dir)
        if frame_candidates:
            frame_path = str(frame_candidates[0].resolve())
    external_frame, external_source = _external_photoreal_observation_frame()
    external_photoreal = external_frame is not None
    if external_photoreal:
        frame_path = str(external_frame)
    visual_claim_boundary = {
        "simulated_camera_view": bool(frame_path) and not external_photoreal,
        "physical_robot_sensor_proof": False,
        "visual_observation_path_can_feed_vla_policy_endpoint": bool(frame_path),
    }
    visual_observation = {
        "available": bool(frame_path),
        "camera_frame_path": frame_path,
        "camera_id": _policy_action_model_frame_camera_id(frame_path),
        "first_person_policy_observation_candidate": bool(frame_path),
        "simulated_camera_view": bool(frame_path) and not external_photoreal,
        "physical_robot_sensor_proof": False,
        "blockers": [] if frame_path else ["policy_observation_frame_not_captured"],
        "claim_boundary": visual_claim_boundary,
    }
    if external_photoreal:
        visual_observation["external_photoreal_observation_used"] = True
        visual_observation["photoreal_observation_source"] = external_source
        visual_observation["synthesized_or_splatted_outputs_are_not_raw_capture_truth"] = True
    (
        captured_sonic_state,
        captured_sonic_state_source,
        captured_sonic_state_metadata,
    ) = _first_unitree_g1_sonic_state_from_visual_trace(job_dir)
    unitree_g1_sonic_state = captured_sonic_state or _unitree_g1_sonic_contract_probe_state()
    unitree_g1_sonic_state_source = (
        captured_sonic_state_source or "simulated_mujoco_contract_probe_zero_state"
    )
    unitree_g1_sonic_state_is_contract_probe = captured_sonic_state is None
    claim_boundary = {
        "sample_input_is_contract_probe_not_task_success_evidence": True,
        "visual_frame_is_simulated_mujoco_policy_observation": bool(frame_path)
        and not external_photoreal,
        "unitree_g1_sonic_state_is_simulated_observation": True,
        "unitree_g1_sonic_state_is_contract_probe": unitree_g1_sonic_state_is_contract_probe,
        "unitree_g1_sonic_state_derived_from_mujoco_qpos": not unitree_g1_sonic_state_is_contract_probe,
        "policy_action_command_does_not_prove_generated_world_rank_fidelity": True,
    }
    if external_photoreal:
        claim_boundary["visual_frame_is_external_photoreal_handoff"] = True
        claim_boundary["visual_frame_is_raw_capture_truth"] = False
        claim_boundary["mujoco_owns_physics_external_lane_owns_pixels"] = True
    return {
        "schema_version": "policy_action_model_command_input.v1",
        "generated_at": generated_at,
        "robot_profile_id": ROBOT_PROFILE_ID,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "action_schema_id": ACTION_SCHEMA_ID,
        "task_prompt": "Return one safe Unitree G1 action for a MuJoCo waypoint/manipulation evaluation packet.",
        "observation": {
            "camera_frame_path": frame_path,
            "visual_observation": visual_observation,
            "state": {
                "root_position": [0.0, 0.0, 0.79],
                "root_yaw_rad": 0.0,
                "target_waypoint": [1.0, 0.0],
                "nearest_object": "blueprint_light_object",
            },
            "unitree_g1_sonic_state": unitree_g1_sonic_state,
            "unitree_g1_sonic_state_source": unitree_g1_sonic_state_source,
            "unitree_g1_sonic_state_metadata": captured_sonic_state_metadata,
        },
        "allowed_action_types": [
            "waypoint",
            "base_velocity",
            "stop",
            "inspect_look",
            "manipulation_contact",
        ],
        "claim_boundary": claim_boundary,
    }


def _openvla_policy_execution_proof(payload: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    schema_version = _string(payload.get("schema_version"))
    allowed_schemas = {
        "openvla_policy_command_adapter.v1",
        "openvla_policy_provider_output.v1",
        "openvla_policy_provider_smoke.v1",
    }
    model_executed = bool(
        payload.get("openvla_model_executed")
        or payload.get("model_ran")
        or payload.get("provider_openvla_model_executed")
    )
    predict_action_invoked = bool(
        payload.get("openvla_predict_action_invoked")
        or payload.get("provider_openvla_policy_action_command_ran")
        or payload.get("openvla_policy_action_command_ran")
    )
    proof = {
        "schema_version": schema_version or None,
        "schema_version_allowed": schema_version in allowed_schemas,
        "openvla_model_executed": model_executed,
        "openvla_predict_action_invoked": predict_action_invoked,
        "provider_output_replay_used": bool(payload.get("provider_output_replay_used")),
    }
    blockers: list[str] = []
    if schema_version not in allowed_schemas:
        blockers.append("openvla_policy_command_output_schema_not_proven")
    if not model_executed:
        blockers.append("openvla_policy_command_missing_model_execution_proof")
    if not predict_action_invoked:
        blockers.append("openvla_policy_command_missing_predict_action_invocation_proof")
    return proof, blockers


def _groot_provider_replay_command_result(
    *,
    candidate: Mapping[str, Any],
    payload: Mapping[str, Any],
    started: float,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if str(candidate.get("candidate_id") or "") != GROOT_POLICY_ID:
        return None
    provider_output = _string(os.getenv(GROOT_PROVIDER_OUTPUT_ENV))
    if not provider_output:
        return None
    response_payload, exit_code = run_groot_policy_command_adapter(
        payload=payload,
        command=None,
        n17_checkpoint=None,
        sonic_checkpoint=None,
        provider_output=Path(provider_output).expanduser(),
    )
    command_result = {
        "status": "completed" if exit_code == 0 else "blocked",
        "returncode": exit_code,
        "duration_seconds": round(time.monotonic() - started, 6),
        "stdout_size_bytes": 0,
        "stderr_size_bytes": 0,
        "stderr_omitted_to_avoid_secret_leakage": False,
        "provider_output_replay_short_circuited": True,
        "blockers": []
        if exit_code == 0
        else _policy_action_command_blockers({"status": "blocked"}, response_payload),
    }
    return command_result, response_payload


def run_policy_action_model_command_contract(
    *,
    job_dir: Path,
    generated_at: str,
    allow_policy_action_model_command_run: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    discovery = discover_policy_action_model_commands(generated_at=generated_at)
    write_json(job_dir / "policy_action_model_command_discovery.json", discovery)
    input_path = job_dir / "policy_action_model_command_input.json"
    output_path = job_dir / "policy_action_model_command_output.json"
    sample_input = _sample_policy_action_model_input(generated_at=generated_at, job_dir=job_dir)
    write_json(
        input_path,
        sample_input,
    )
    ready_candidates = sorted(
        [row for row in discovery["candidates"] if row.get("ready_for_policy_action_command")],
        key=lambda row: (
            row.get("candidate_id") != discovery.get("selected_candidate_id"),
            bool(row.get("command_from_default")),
        ),
    )
    selected_contract_candidate = (
        ready_candidates[0]
        if ready_candidates
        else next(
            (
                row
                for row in discovery.get("candidates", [])
                if row.get("candidate_id") == discovery.get("selected_candidate_id")
            ),
            None,
        )
    )
    selected_contract_command = ""
    if isinstance(selected_contract_candidate, Mapping):
        selected_contract_command = (
            _string(selected_contract_candidate.get("command_value_for_execution"))
            or os.getenv(str(selected_contract_candidate.get("command_env") or ""), "").strip()
        )
    provider_worker_contract = write_provider_worker_contract(
        output_dir=job_dir,
        generated_at=generated_at,
        provider="provider_neutral",
        worker_role="unitree_policy_action_worker",
        policy_command=selected_contract_command,
    )
    provider_worker_contract_path = str(job_dir / "provider_worker_contract.json")
    blockers: list[str] = []
    if not allow_policy_action_model_command_run:
        blockers.append("missing_cli_allow_policy_action_model_command_run")
    if os.getenv(POLICY_ACTION_MODEL_COMMAND_GATE_ENV, "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "y",
    }:
        blockers.append(f"missing_env_{POLICY_ACTION_MODEL_COMMAND_GATE_ENV}")
    if not ready_candidates:
        blockers.extend(discovery.get("blockers", []))
    if blockers:
        _write_blocked_policy_action_model_command_output(
            output_path,
            generated_at=generated_at,
            selected_candidate_id=_string(discovery.get("selected_candidate_id")),
            blockers=blockers,
        )
        result = {
            "schema_version": "policy_action_model_command_execution.v1",
            "generated_at": generated_at,
            "status": "blocked",
            "policy_action_model_command_ran": False,
            "openvla_policy_action_command_ran": False,
            "openvla_model_executed": False,
            "openvla_predict_action_invoked": False,
            "unitree_policy_action_command_ran": False,
            "unitree_lerobot_policy_action_command_ran": False,
            "unitree_unifolm_policy_action_command_ran": False,
            "unitree_groot_n17_sonic_policy_action_command_ran": False,
            "unitree_manipulation_policy_action_command_ran": False,
            "unitree_specific_policy_candidate_ran": False,
            "unitree_specific_manipulation_candidate_ran": False,
            "provider_output_replay_used": False,
            "fresh_policy_action_model_executed_this_invocation": False,
            "selected_candidate_id": discovery.get("selected_candidate_id"),
            "discovery": discovery,
            "provider_worker_contract_path": provider_worker_contract_path,
            "provider_worker_contract": provider_worker_contract,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "blockers": sorted(set(blockers)),
            "claim_boundary": {
                "policy_action_command_is_model_contract_probe_not_wam_rollout": True,
                "generated_world_rank_fidelity_result_proven": False,
                "generated_world_policy_evaluation_scope_proven": False,
                "non_ranking_operational_claim_proven": False,
            },
        }
        write_json(job_dir / "policy_action_model_command_execution.json", result)
        return result
    selected = ready_candidates[0]
    command = (
        _string(selected.get("command_value_for_execution"))
        or os.getenv(str(selected["command_env"]), "").strip()
    )
    started = time.monotonic()
    env = {
        **os.environ,
        "BLUEPRINT_POLICY_ACTION_INPUT": str(input_path),
        "BLUEPRINT_POLICY_ACTION_OUTPUT": str(output_path),
        "BLUEPRINT_POLICY_MODEL_CANDIDATE": str(selected["candidate_id"]),
        "BLUEPRINT_POLICY_MODEL_CHECKPOINT": str(selected.get("checkpoint_path") or ""),
    }
    if (
        selected["candidate_id"] == GROOT_POLICY_ID
        and selected.get("command_from_default")
        and not env.get(POLICY_SERVER_URL_ENV)
    ):
        env[POLICY_SERVER_URL_ENV] = "tcp://127.0.0.1:5550"
    command_result: dict[str, Any]
    payload: dict[str, Any] = {}
    replay_result = _groot_provider_replay_command_result(
        candidate=selected,
        payload=sample_input,
        started=started,
    )
    if replay_result is not None:
        command_result, payload = replay_result
        write_json(output_path, payload)
    else:
        try:
            completed = subprocess.run(
                shlex.split(command),
                cwd=str(job_dir),
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
            command_result = {
                "status": "completed" if completed.returncode == 0 else "blocked",
                "returncode": completed.returncode,
                "duration_seconds": round(time.monotonic() - started, 6),
                "stdout_size_bytes": len(completed.stdout or ""),
                "stderr_size_bytes": len(completed.stderr or ""),
                "stderr_omitted_to_avoid_secret_leakage": bool(completed.stderr),
                "blockers": []
                if completed.returncode == 0
                else ["policy_action_model_command_nonzero_exit"],
            }
            if output_path.is_file():
                value = json.loads(output_path.read_text(encoding="utf-8"))
                payload = dict(value) if isinstance(value, Mapping) else {}
            elif completed.stdout.strip():
                value = json.loads(completed.stdout)
                payload = dict(value) if isinstance(value, Mapping) else {}
            if _string(payload.get("status")) in {"blocked", "failed"}:
                command_result["status"] = "blocked"
                command_result["blockers"] = _policy_action_command_blockers(
                    command_result,
                    payload,
                )
        except Exception as exc:
            command_result = {
                "status": "blocked",
                "duration_seconds": round(time.monotonic() - started, 6),
                "blockers": [f"policy_action_model_command_failed:{type(exc).__name__}"],
            }
    action_payload = _policy_action_payload(payload)
    action_present = isinstance(action_payload, Mapping)
    if command_result.get("status") == "completed" and not action_present:
        command_result["status"] = "blocked"
        command_result["blockers"] = ["policy_action_model_command_missing_action_payload"]
    openvla_execution_proof: dict[str, Any] = {}
    if command_result.get("status") == "completed" and selected["candidate_id"] == "openvla_policy":
        openvla_execution_proof, openvla_blockers = _openvla_policy_execution_proof(payload)
        if openvla_blockers:
            command_result["status"] = "blocked"
            command_result["blockers"] = openvla_blockers
    ran = command_result.get("status") == "completed"
    selected_candidate_id = str(selected["candidate_id"])
    if not output_path.is_file():
        if payload:
            write_json(output_path, payload)
        else:
            _write_blocked_policy_action_model_command_output(
                output_path,
                generated_at=generated_at,
                selected_candidate_id=selected_candidate_id,
                blockers=command_result.get("blockers", []),
                command_result=command_result,
            )
    unitree_execution_flags = _unitree_policy_action_execution_flags(
        ran=ran,
        selected_candidate_id=selected_candidate_id,
        payload=payload,
    )
    provider_output_replay_used = bool(
        payload.get("provider_output_replay_used")
        or _mapping(payload.get("claim_boundary")).get("provider_output_replay_used")
    )
    fresh_policy_action_model_executed_this_invocation = bool(
        ran
        and not provider_output_replay_used
        and (
            payload.get("model_ran")
            or payload.get("fresh_unitree_groot_n17_sonic_model_executed_this_invocation")
            or unitree_execution_flags.get("unitree_policy_action_command_ran")
            or (
                selected["candidate_id"] == "openvla_policy"
                and openvla_execution_proof.get("openvla_model_executed")
            )
        )
    )
    result = {
        "schema_version": "policy_action_model_command_execution.v1",
        "generated_at": generated_at,
        "status": "completed" if ran else "blocked",
        "policy_action_model_command_ran": ran,
        "openvla_policy_action_command_ran": bool(
            ran and selected["candidate_id"] == "openvla_policy"
        ),
        "openvla_model_executed": bool(
            ran
            and selected["candidate_id"] == "openvla_policy"
            and openvla_execution_proof.get("openvla_model_executed")
        ),
        "openvla_predict_action_invoked": bool(
            ran
            and selected["candidate_id"] == "openvla_policy"
            and openvla_execution_proof.get("openvla_predict_action_invoked")
        ),
        **unitree_execution_flags,
        "provider_output_replay_used": provider_output_replay_used,
        "fresh_policy_action_model_executed_this_invocation": (
            fresh_policy_action_model_executed_this_invocation
        ),
        "selected_candidate_id": selected_candidate_id,
        "selected_command_env": selected["command_env"],
        "provider_worker_contract_path": provider_worker_contract_path,
        "provider_worker_contract": provider_worker_contract,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "action_payload_present": action_present,
        "action_payload_redacted": dict(action_payload)
        if isinstance(action_payload, Mapping)
        else None,
        "openvla_execution_proof": openvla_execution_proof
        if selected["candidate_id"] == "openvla_policy"
        else None,
        "command_result": command_result,
        "blockers": [] if ran else command_result.get("blockers", []),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "claim_boundary": {
            "policy_action_command_is_model_contract_probe_not_wam_rollout": True,
            "policy_action_command_does_not_prove_task_success": True,
            "provider_output_replay_is_not_fresh_per_request_model_inference": (
                provider_output_replay_used
            ),
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
        },
    }
    write_json(job_dir / "policy_action_model_command_execution.json", result)
    return result


def _execute_policy_action_model_candidate(
    *,
    job_dir: Path,
    candidate: Mapping[str, Any],
    payload: Mapping[str, Any],
    input_path: Path,
    output_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    write_json(input_path, dict(payload))
    if output_path.exists():
        output_path.unlink()
    command = (
        _string(candidate.get("command_value_for_execution"))
        or os.getenv(str(candidate.get("command_env") or ""), "").strip()
    )
    selected_candidate_id = str(candidate.get("candidate_id") or "")
    started = time.monotonic()
    policy_worker_contract = _mapping(candidate.get("policy_worker_contract"))
    if not policy_worker_contract:
        policy_worker_contract = classify_policy_worker_command(command)
    if not policy_worker_contract.get("repeated_policy_loop_allowed"):
        blockers = [
            str(item) for item in policy_worker_contract.get("blockers", []) if str(item)
        ] or ["blocked_policy_worker_not_safe_for_repeated_loop"]
        command_result = {
            "status": "blocked",
            "duration_seconds": round(time.monotonic() - started, 6),
            "policy_worker_invocation_kind": policy_worker_contract.get("invocation_kind"),
            "provider_instance_launch_per_inference": policy_worker_contract.get(
                "provider_instance_launch_per_inference"
            ),
            "blockers": blockers,
        }
        _write_blocked_policy_action_model_command_output(
            output_path,
            generated_at=utc_now_iso(),
            selected_candidate_id=selected_candidate_id,
            blockers=blockers,
            command_result=command_result,
        )
        unitree_execution_flags = _unitree_policy_action_execution_flags(
            ran=False,
            selected_candidate_id=selected_candidate_id,
            payload={},
        )
        return {
            "status": "blocked",
            "selected_candidate_id": selected_candidate_id,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "policy_action_model_command_ran": False,
            "action_payload_present": False,
            "action_payload_redacted": None,
            "response_redacted": None,
            "command_result": command_result,
            "blockers": blockers,
            **unitree_execution_flags,
            "provider_output_replay_used": False,
            "fresh_policy_action_model_executed_this_invocation": False,
            "provider_worker_contract_schema_version": PROVIDER_WORKER_CONTRACT_SCHEMA_VERSION,
            "policy_worker_contract": policy_worker_contract,
            "policy_worker_invocation_kind": policy_worker_contract.get("invocation_kind"),
            "provider_instance_launch_per_inference": policy_worker_contract.get(
                "provider_instance_launch_per_inference"
            ),
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        }
    env = {
        **os.environ,
        "BLUEPRINT_POLICY_ACTION_INPUT": str(input_path),
        "BLUEPRINT_POLICY_ACTION_OUTPUT": str(output_path),
        "BLUEPRINT_POLICY_MODEL_CANDIDATE": selected_candidate_id,
        "BLUEPRINT_POLICY_MODEL_CHECKPOINT": str(candidate.get("checkpoint_path") or ""),
    }
    command_result: dict[str, Any]
    response_payload: dict[str, Any] = {}
    replay_result = _groot_provider_replay_command_result(
        candidate=candidate,
        payload=dict(payload),
        started=started,
    )
    if replay_result is not None:
        command_result, response_payload = replay_result
        write_json(output_path, response_payload)
    else:
        try:
            completed = subprocess.run(
                shlex.split(command),
                cwd=str(job_dir),
                env=env,
                input=json.dumps(dict(payload)),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
            command_result = {
                "status": "completed" if completed.returncode == 0 else "blocked",
                "returncode": completed.returncode,
                "duration_seconds": round(time.monotonic() - started, 6),
                "stdout_size_bytes": len(completed.stdout or ""),
                "stderr_size_bytes": len(completed.stderr or ""),
                "stderr_omitted_to_avoid_secret_leakage": bool(completed.stderr),
                "blockers": []
                if completed.returncode == 0
                else ["policy_action_model_command_nonzero_exit"],
            }
            if output_path.is_file():
                value = json.loads(output_path.read_text(encoding="utf-8"))
                response_payload = dict(value) if isinstance(value, Mapping) else {}
            elif completed.stdout.strip():
                value = json.loads(completed.stdout)
                response_payload = dict(value) if isinstance(value, Mapping) else {}
            if _string(response_payload.get("status")) in {"blocked", "failed"}:
                command_result["status"] = "blocked"
                command_result["blockers"] = _policy_action_command_blockers(
                    command_result,
                    response_payload,
                )
        except Exception as exc:
            command_result = {
                "status": "blocked",
                "duration_seconds": round(time.monotonic() - started, 6),
                "blockers": [f"policy_action_model_command_failed:{type(exc).__name__}"],
            }
    action_payload = _policy_action_payload(response_payload)
    action_present = isinstance(action_payload, Mapping)
    ran = bool(command_result.get("status") == "completed" and action_present)
    if command_result.get("status") == "completed" and not action_present:
        command_result["status"] = "blocked"
        command_result["blockers"] = ["policy_action_model_command_missing_action_payload"]
    if not output_path.is_file():
        if response_payload:
            write_json(output_path, response_payload)
        else:
            _write_blocked_policy_action_model_command_output(
                output_path,
                generated_at=utc_now_iso(),
                selected_candidate_id=selected_candidate_id,
                blockers=command_result.get("blockers", []),
                command_result=command_result,
            )
    unitree_execution_flags = _unitree_policy_action_execution_flags(
        ran=ran,
        selected_candidate_id=selected_candidate_id,
        payload=response_payload,
    )
    provider_output_replay_used = bool(
        response_payload.get("provider_output_replay_used")
        or _mapping(response_payload.get("claim_boundary")).get("provider_output_replay_used")
    )
    return {
        "status": "completed" if ran else "blocked",
        "selected_candidate_id": selected_candidate_id,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "policy_action_model_command_ran": ran,
        "action_payload_present": action_present,
        "action_payload_redacted": dict(action_payload) if action_payload else None,
        "response_redacted": _redact(response_payload) if response_payload else None,
        "command_result": command_result,
        "blockers": [] if ran else list(command_result.get("blockers", [])),
        **unitree_execution_flags,
        "provider_output_replay_used": provider_output_replay_used,
        "fresh_policy_action_model_executed_this_invocation": bool(
            ran
            and not provider_output_replay_used
            and (
                response_payload.get("model_ran")
                or response_payload.get(
                    "fresh_unitree_groot_n17_sonic_model_executed_this_invocation"
                )
                or unitree_execution_flags.get("unitree_policy_action_command_ran")
            )
        ),
        "provider_worker_contract_schema_version": PROVIDER_WORKER_CONTRACT_SCHEMA_VERSION,
        "policy_worker_contract": policy_worker_contract,
        "policy_worker_invocation_kind": policy_worker_contract.get("invocation_kind"),
        "provider_instance_launch_per_inference": policy_worker_contract.get(
            "provider_instance_launch_per_inference"
        ),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


def _initial_policy_action_loop_observation(
    *,
    generated_at: str,
    job_dir: Path,
) -> dict[str, Any]:
    sample = _sample_policy_action_model_input(generated_at=generated_at, job_dir=job_dir)
    observation = _mapping(sample.get("observation"))
    observation.setdefault("schema_version", "blueprint_policy_observation.v1")
    observation.setdefault("task_id", "contact_or_push_light_object")
    observation.setdefault(
        "task_prompt",
        "move the Unitree G1 hand toward the light object and make controlled contact",
    )
    return {
        "schema_version": "policy_action_model_command_input.v1",
        "generated_at": generated_at,
        "robot_profile_id": ROBOT_PROFILE_ID,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "action_schema_id": ACTION_SCHEMA_ID,
        "observation": observation,
        "allowed_action_types": list(sample.get("allowed_action_types") or []),
        "claim_boundary": {
            "policy_wam_loop_probe": True,
            "simulator_only": True,
            "physical_robot_sensor_proof": False,
        },
    }


def _declared_policy_observation_schema_for_wam_loop(
    selected_candidate_id: str | None,
) -> dict[str, Any]:
    """Return the observation fields the current policy loop is allowed to receive."""

    return {
        "schema_version": "wam_policy_observation_adapter_declared_schema.v1",
        "policy_id": selected_candidate_id,
        "schema_id": OBSERVATION_SCHEMA_ID,
        "modalities": ["rgb", "depth", "nominal_state"],
        "fields": [
            "schema_version",
            "camera_frame_path",
            "visual_observation",
            "task_id",
            "task_prompt",
            "target_object_id",
            "state",
            "proprioception",
            "unitree_g1_sonic_state",
            "base_pose",
            "base_velocity",
            "contact_state",
            "route_task_state",
            "object_state",
            "depth_frame_path",
            "allowed_action_schema",
            "safety_limits",
        ],
        "supports_depth": True,
        "supports_masks": False,
        "supports_state": True,
        "claim_boundary": {
            "policy_schema_requests_mujoco_render_pass_depth_when_available": True,
            "policy_schema_does_not_request_harness_masks_by_default": True,
            "harness_outputs_available_for_diagnostics_and_gating": True,
            "mujoco_render_pass_depth_co_registered_with_rgb": True,
        },
    }


def _wam_perception_harness_backend_config() -> dict[str, Any]:
    backend_kind = os.getenv("BLUEPRINT_WAM_PERCEPTION_HARNESS_BACKEND_KIND", "").strip()
    backend_command = os.getenv(EXTERNAL_BACKEND_COMMAND_ENV, "").strip()
    if not backend_kind and backend_command:
        backend_kind = "external_command"
    return {
        "backend_kind": backend_kind or "fixture",
        "backend_command": backend_command or None,
        "allow_external_backend": None,
        "env_gate": EXTERNAL_BACKEND_ENV_GATE,
        "command_env": EXTERNAL_BACKEND_COMMAND_ENV,
        "configured_for_external_backend": bool(backend_kind or backend_command),
    }


WAM_GENERATION_COMMAND_GATE_ENV = "BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER"
WAM_GENERATION_COMMAND_CANDIDATES = (
    {
        "backend_id": "oscar_wam",
        "command_envs": ("BLUEPRINT_OSCAR_WAM_COMMAND", "BLUEPRINT_OSCAR_WAM_PROVIDER_COMMAND"),
        "checkpoint_envs": ("BLUEPRINT_OSCAR_WAM_CHECKPOINT",),
    },
    {
        "backend_id": "cosmos_wam",
        "command_envs": (
            "BLUEPRINT_COSMOS_WAM_COMMAND",
            "BLUEPRINT_COSMOS_WAM_PROVIDER_COMMAND",
            "BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND",
        ),
        "checkpoint_envs": ("BLUEPRINT_COSMOS_WAM_CHECKPOINT", "BLUEPRINT_COSMOS3_WAM_CHECKPOINT"),
    },
)


def discover_wam_generation_command(*, generated_at: str) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    gate_enabled = _env_truthy(WAM_GENERATION_COMMAND_GATE_ENV)
    for spec in WAM_GENERATION_COMMAND_CANDIDATES:
        command_env = ""
        command_value = ""
        for env_name in spec["command_envs"]:
            value = os.getenv(str(env_name), "").strip()
            if value:
                command_env = str(env_name)
                command_value = value
                break
        checkpoint_env = ""
        checkpoint_value = ""
        checkpoint_configured = False
        checkpoint_exists = False
        checkpoint_reference = None
        checkpoint_reference_kind = None
        for env_name in spec["checkpoint_envs"]:
            value = os.getenv(str(env_name), "").strip()
            if value:
                checkpoint_env = str(env_name)
                checkpoint_value = value
                (
                    checkpoint_configured,
                    checkpoint_reference,
                    checkpoint_exists,
                    checkpoint_reference_kind,
                ) = _configured_checkpoint_reference(checkpoint_value)
                break
        command_available = _command_available(command_value)
        blockers: list[str] = []
        if not command_value:
            blockers.append("blocked_missing_wam_generation_command")
        elif not command_available:
            blockers.append("blocked_wam_generation_command_unavailable")
        if not checkpoint_value:
            blockers.append("blocked_missing_wam_model_checkpoint")
        elif not checkpoint_configured:
            blockers.append("blocked_wam_model_checkpoint_missing")
        if not gate_enabled:
            blockers.append(f"missing_env_{WAM_GENERATION_COMMAND_GATE_ENV}")
        ready = bool(command_value and command_available and gate_enabled)
        candidates.append(
            {
                "backend_id": spec["backend_id"],
                "command_env": command_env or list(spec["command_envs"])[0],
                "command_configured": bool(command_value),
                "command_available": command_available,
                "command_value_redacted": "<configured>" if command_value else None,
                "checkpoint_env": checkpoint_env or list(spec["checkpoint_envs"])[0],
                "checkpoint_configured": checkpoint_configured,
                "checkpoint_exists": checkpoint_exists,
                "checkpoint_reference": checkpoint_reference,
                "checkpoint_reference_kind": checkpoint_reference_kind,
                "ready_for_live_wam_generation": ready,
                "blockers": blockers,
            }
        )
    ready_candidates = [row for row in candidates if row["ready_for_live_wam_generation"]]
    selected = ready_candidates[0] if ready_candidates else None
    selected_candidate = selected or next(
        (row for row in candidates if row["command_configured"] or row["checkpoint_configured"]),
        None,
    )
    blockers = []
    if not ready_candidates:
        blockers.append("blocked_missing_live_wam_generation_command")
        for row in candidates:
            blockers.extend(str(item) for item in row.get("blockers", []) if str(item))
    return {
        "schema_version": "wam_generation_command_discovery.v1",
        "generated_at": generated_at,
        "status": "ready" if ready_candidates else "blocked",
        "selection_policy": "prefer_configured_live_oscar_or_cosmos_wam_command",
        "gate_env": WAM_GENERATION_COMMAND_GATE_ENV,
        "gate_enabled": gate_enabled,
        "selected_backend_id": selected_candidate.get("backend_id") if selected_candidate else None,
        "selected_command_env": selected_candidate.get("command_env")
        if selected_candidate
        else None,
        "selected_backend_ready_for_live_wam_generation": bool(
            selected and selected.get("ready_for_live_wam_generation")
        ),
        "ready_candidate_count": len(ready_candidates),
        "candidates": candidates,
        "ready_for_live_wam_generation": bool(ready_candidates),
        "blockers": sorted(set(blockers)),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "claim_boundary": {
            "wam_generation_command_is_world_model_or_evaluator_not_robot_policy": True,
            "provider_credentials_not_written_to_artifacts": True,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
        },
    }


def _candidate_path_from_payload(
    payload: Mapping[str, Any],
    *,
    output_path: Path,
    keys: Sequence[str],
) -> Path | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = output_path.parent / path
            return path
    return None


def _extract_generated_frame_from_video(
    video_path: Path, target_frame: Path
) -> tuple[bool, str | None]:
    try:
        import cv2  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"opencv_import_failed:{type(exc).__name__}"
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return False, "generated_video_unreadable_for_next_observation"
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return False, "generated_video_first_frame_unreadable_for_next_observation"
    ensure_dir(target_frame.parent)
    written = cv2.imwrite(str(target_frame), frame)
    return bool(written), None if written else "generated_video_first_frame_write_failed"


def _materialize_wam_generated_frame(
    *,
    payload: Mapping[str, Any],
    output_path: Path,
    target_frame: Path,
) -> tuple[Path | None, dict[str, Any]]:
    visual = _mapping(payload.get("visual_observation"))
    direct_frame = _candidate_path_from_payload(
        {
            **dict(payload),
            "visual_observation_camera_frame_path": visual.get("camera_frame_path"),
        },
        output_path=output_path,
        keys=(
            "generated_next_observation_frame_path",
            "camera_frame_path",
            "frame_path",
            "image_path",
            "visual_observation_camera_frame_path",
        ),
    )
    if direct_frame and direct_frame.is_file():
        ensure_dir(target_frame.parent)
        if direct_frame.resolve() != target_frame.resolve():
            shutil.copy2(direct_frame, target_frame)
        return target_frame, {
            "source_kind": "image_frame",
            "source_path": str(direct_frame),
            "materialized_frame_path": str(target_frame),
        }

    for item in payload.get("generated_frames") or []:
        if isinstance(item, str):
            frame = Path(item).expanduser()
        elif isinstance(item, Mapping):
            frame = (
                _candidate_path_from_payload(
                    item,
                    output_path=output_path,
                    keys=("path", "frame_path", "image_path", "camera_frame_path"),
                )
                or Path()
            )
        else:
            continue
        if frame.is_file():
            ensure_dir(target_frame.parent)
            shutil.copy2(frame, target_frame)
            return target_frame, {
                "source_kind": "generated_frame_list",
                "source_path": str(frame),
                "materialized_frame_path": str(target_frame),
            }

    rollout = next(
        (dict(item) for item in payload.get("rollouts") or [] if isinstance(item, Mapping)),
        {},
    )
    video_path = _candidate_path_from_payload(
        {**dict(payload), **rollout},
        output_path=output_path,
        keys=("generated_video_path", "video_path", "output_video_path"),
    )
    if video_path and video_path.is_file():
        ok, blocker = _extract_generated_frame_from_video(video_path, target_frame)
        return (
            target_frame if ok else None,
            {
                "source_kind": "generated_video_first_frame",
                "source_path": str(video_path),
                "materialized_frame_path": str(target_frame) if ok else None,
                "blocker": blocker,
            },
        )
    return None, {
        "source_kind": "missing_generated_frame_or_video",
        "blocker": "wam_command_output_missing_generated_next_observation_frame_or_video",
    }


def _wam_generation_action_numeric_values(
    action: Mapping[str, Any],
    *,
    limit: int = 24,
) -> list[float]:
    values: list[float] = []

    def collect(value: Any) -> None:
        if len(values) >= limit:
            return
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            numeric = float(value)
            if math.isfinite(numeric):
                values.append(numeric)
            return
        if isinstance(value, Mapping):
            for child in value.values():
                collect(child)
                if len(values) >= limit:
                    break
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                collect(child)
                if len(values) >= limit:
                    break

    for key in (
        "action_chunk",
        "action",
        "joint_targets",
        "actions",
        "action_vector",
        "joint_positions",
        "arm_targets",
        "hand_targets",
        "gripper_targets",
    ):
        if key in action:
            collect(action.get(key))
            if values:
                break
    if not values:
        collect(action)
    return values[:limit]


def _wam_skeleton_trace_candidates(job_dir: Path) -> list[Path]:
    candidates = [
        job_dir / "g1_projected_skeleton_trace.jsonl",
        job_dir / "robot_fk_projected_skeleton_trace.jsonl",
        job_dir / "simulation_automation" / "robot_fk_projected_skeleton_trace.jsonl",
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _same_existing_path(left: Any, right: Path) -> bool:
    text = _string(left)
    if not text:
        return False
    try:
        return Path(text).expanduser().resolve() == right.resolve()
    except Exception:
        return Path(text).expanduser() == right


def _projected_skeleton_points(
    row: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> dict[str, tuple[int, int]]:
    points: dict[str, tuple[int, int]] = {}
    for landmark in row.get("landmarks") or []:
        if not isinstance(landmark, Mapping):
            continue
        projection = _mapping(landmark.get("image_projection"))
        if not projection.get("available"):
            continue
        u = _number(projection.get("u_px"), None)
        v = _number(projection.get("v_px"), None)
        if u is None or v is None:
            continue
        x = max(0, min(width - 1, int(round(float(u)))))
        y = max(0, min(height - 1, int(round(float(v)))))
        landmark_id = _string(landmark.get("landmark_id"))
        if landmark_id:
            points[landmark_id] = (x, y)
    return points


def _select_wam_skeleton_conditioning(
    *,
    job_dir: Path,
    source_frame: Path,
) -> dict[str, Any]:
    best_row: dict[str, Any] | None = None
    best_trace_path: Path | None = None
    best_rank: tuple[int, str] | None = None
    inspected_paths: list[str] = []
    for trace_path in _wam_skeleton_trace_candidates(job_dir):
        if not trace_path.is_file():
            continue
        inspected_paths.append(str(trace_path))
        for index, row in enumerate(_read_jsonl(trace_path, limit=5000)):
            projected_count = int(row.get("projected_landmark_count") or 0)
            if projected_count <= 0:
                continue
            same_frame = _same_existing_path(row.get("camera_frame_path"), source_frame)
            camera_id = _string(row.get("camera_id"))
            rank = (
                0 if same_frame else 1,
                f"{0 - projected_count:06d}:{camera_id}:{index:06d}",
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_row = dict(row)
                best_trace_path = trace_path
    if best_row is None:
        return {
            "projected_skeleton_trace_used": False,
            "inspected_trace_paths": inspected_paths,
            "projected_landmark_count": 0,
            "segments": [],
            "landmarks": [],
            "blockers": ["projected_g1_skeleton_trace_not_available"],
            "claim_boundary": {
                "simulated_skeleton_conditioning_available": False,
                "physical_robot_proprioception_proof": False,
            },
        }
    landmarks = []
    for landmark in best_row.get("landmarks") or []:
        if not isinstance(landmark, Mapping):
            continue
        projection = _mapping(landmark.get("image_projection"))
        if not projection.get("available"):
            continue
        landmarks.append(
            {
                "landmark_id": landmark.get("landmark_id"),
                "u_px": projection.get("u_px"),
                "v_px": projection.get("v_px"),
                "depth_m_abs": projection.get("depth_m_abs"),
            }
        )
    return {
        "projected_skeleton_trace_used": True,
        "trace_path": str(best_trace_path) if best_trace_path else None,
        "source_camera_frame_path": best_row.get("camera_frame_path"),
        "episode_id": best_row.get("episode_id"),
        "scenario_eval_run_id": best_row.get("scenario_eval_run_id"),
        "step": best_row.get("step"),
        "camera_id": best_row.get("camera_id"),
        "projected_landmark_count": int(best_row.get("projected_landmark_count") or 0),
        "segments": [
            dict(item) for item in best_row.get("segments") or [] if isinstance(item, Mapping)
        ],
        "landmarks": landmarks[:16],
        "blockers": [],
        "claim_boundary": {
            "simulated_skeleton_conditioning_available": True,
            "derived_from_unitree_g1_mujoco_body_transforms": True,
            "physical_robot_proprioception_proof": False,
        },
    }


def _default_wam_action_conditioning(
    *,
    current_action: Mapping[str, Any],
    current_observation: Mapping[str, Any],
) -> dict[str, Any]:
    values = _wam_generation_action_numeric_values(current_action)
    observation_state = _mapping(current_observation.get("state"))
    proprioception = _mapping(current_observation.get("proprioception"))
    if not proprioception:
        proprioception = {
            key: value
            for key, value in observation_state.items()
            if key in {"joint_positions", "joint_velocities", "qpos", "qvel", "robot_pose"}
        }
    return {
        "source_policy_action_type": current_action.get("action_type"),
        "source_policy_action_summary": _policy_action_trace_summary(current_action),
        "numeric_action_value_count": len(values),
        "numeric_action_value_sample": [round(value, 6) for value in values[:8]],
        "proprioception_keys": sorted(str(key) for key in proprioception.keys()),
        "action_and_proprioception_conditioning_used": bool(values or proprioception),
        "claim_boundary": {
            "conditioned_on_policy_action_payload": bool(current_action),
            "conditioned_on_simulated_proprioception_if_available": bool(proprioception),
            "physical_robot_proprioception_proof": False,
        },
    }


def _render_default_wam_next_observation_frame(
    *,
    source_frame: Path,
    target_frame: Path,
    step_index: int,
    action_conditioning: Mapping[str, Any],
    skeleton_conditioning: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps  # type: ignore[import-untyped]
    except Exception as exc:
        return {
            "status": "blocked",
            "blockers": [f"pillow_import_failed:{type(exc).__name__}"],
            "source_frame_decoded": False,
        }

    source_frame_decoded = True
    try:
        image = Image.open(source_frame).convert("RGB")
        image = ImageOps.exif_transpose(image)
    except Exception:
        source_frame_decoded = False
        image = Image.new("RGB", (640, 480), (34, 38, 44))
    if image.width < 128 or image.height < 96:
        image = image.resize((max(128, image.width * 4), max(96, image.height * 4)))

    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    values = [
        float(value)
        for value in action_conditioning.get("numeric_action_value_sample", []) or []
        if isinstance(value, (int, float))
    ]
    magnitude = sum(abs(value) for value in values[:6]) / max(1, min(6, len(values)))
    dx = int(max(-0.32, min(0.32, sum(values[0::2]) * 0.08)) * width) if values else width // 10
    dy = (
        int(max(-0.24, min(0.24, sum(values[1::2]) * -0.08)) * height)
        if len(values) > 1
        else -height // 12
    )
    center = (width // 2, height // 2)
    arrow_end = (
        max(8, min(width - 9, center[0] + dx)),
        max(8, min(height - 9, center[1] + dy)),
    )
    tint_alpha = max(34, min(118, int(42 + magnitude * 90)))
    draw.rectangle((0, 0, width, height), fill=(20, 54, 68, tint_alpha))

    skeleton_points: dict[str, tuple[int, int]] = {}
    if skeleton_conditioning.get("projected_skeleton_trace_used"):
        skeleton_row = {
            "landmarks": [
                {
                    "landmark_id": landmark.get("landmark_id"),
                    "image_projection": {
                        "available": True,
                        "u_px": landmark.get("u_px"),
                        "v_px": landmark.get("v_px"),
                    },
                }
                for landmark in skeleton_conditioning.get("landmarks", []) or []
                if isinstance(landmark, Mapping)
            ]
        }
        skeleton_points = _projected_skeleton_points(skeleton_row, width=width, height=height)
        for segment in skeleton_conditioning.get("segments") or []:
            if not isinstance(segment, Mapping):
                continue
            start = skeleton_points.get(_string(segment.get("from")))
            end = skeleton_points.get(_string(segment.get("to")))
            if start and end:
                draw.line((start, end), fill=(72, 202, 228, 238), width=max(3, width // 180))
        for point in skeleton_points.values():
            radius = max(4, width // 160)
            draw.ellipse(
                (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
                fill=(249, 226, 116, 245),
            )
    wrist = (
        skeleton_points.get("left_wrist")
        or skeleton_points.get("right_wrist")
        or skeleton_points.get("left_hand")
        or skeleton_points.get("right_hand")
        or center
    )
    draw.line((wrist, arrow_end), fill=(255, 116, 85, 250), width=max(4, width // 140))
    radius = max(7, width // 90)
    draw.ellipse(
        (
            arrow_end[0] - radius,
            arrow_end[1] - radius,
            arrow_end[0] + radius,
            arrow_end[1] + radius,
        ),
        outline=(255, 245, 214, 250),
        width=max(2, width // 260),
    )
    try:
        font = ImageFont.load_default()
        label = f"default WAM support step {step_index}"
        draw.text((12, 10), label, fill=(240, 246, 250, 238), font=font)
    except Exception:
        pass
    output = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    ensure_dir(target_frame.parent)
    output.save(target_frame, format="JPEG", quality=92)
    return {
        "status": "completed",
        "blockers": [],
        "source_frame_decoded": source_frame_decoded,
        "source_frame_path": str(source_frame),
        "generated_frame_path": str(target_frame),
        "image_width": width,
        "image_height": height,
        "drawn_projected_skeleton_landmark_count": len(skeleton_points),
        "drawn_action_vector": True,
    }


def _write_default_wam_next_observation_video_segment(
    *,
    generated_frame: Path,
    target_video: Path,
    step_index: int,
    frame_count: int = 6,
    fps: int = 6,
) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-untyped]
    except Exception as exc:
        return {
            "status": "blocked",
            "blockers": [f"pillow_import_failed:{type(exc).__name__}"],
            "generated_video_segment_path": str(target_video),
        }
    frames_dir = target_video.parent / f"{target_video.stem}_frames"
    ensure_dir(frames_dir)
    try:
        base = Image.open(generated_frame).convert("RGB")
        font = ImageFont.load_default()
        for index in range(frame_count):
            frame = base.copy()
            draw = ImageDraw.Draw(frame)
            progress_x = int((index + 1) / max(1, frame_count) * frame.width)
            draw.rectangle(
                (0, frame.height - 8, progress_x, frame.height),
                fill=(255, 116, 85),
            )
            draw.text(
                (12, max(12, frame.height - 26)),
                f"default WAM segment {step_index}.{index + 1}",
                fill=(240, 246, 250),
                font=font,
            )
            frame.save(frames_dir / f"frame_{index + 1:04d}.png")
    except Exception as exc:
        return {
            "status": "blocked",
            "blockers": [f"default_wam_video_segment_frame_write_failed:{type(exc).__name__}"],
            "generated_video_segment_path": str(target_video),
            "frame_dir": str(frames_dir),
        }

    ffmpeg_result = _write_video_from_frames(
        frames_dir=frames_dir,
        output_path=target_video,
        fps=fps,
    )
    if ffmpeg_result.get("status") == "complete":
        return {
            "status": "completed",
            "blockers": [],
            "generated_video_segment_path": str(target_video),
            "frame_dir": str(frames_dir),
            "frame_count": frame_count,
            "fps": fps,
            "encoder": "ffmpeg_libx264",
            "size_bytes": ffmpeg_result.get("size_bytes"),
            "ffmpeg_result": ffmpeg_result,
        }

    try:
        import cv2  # type: ignore[import-untyped]
        import numpy as np  # type: ignore[import-untyped]
    except Exception as exc:
        return {
            "status": "blocked",
            "blockers": [
                "ffmpeg_video_segment_write_failed",
                f"opencv_video_segment_fallback_unavailable:{type(exc).__name__}",
            ],
            "generated_video_segment_path": str(target_video),
            "frame_dir": str(frames_dir),
            "ffmpeg_result": ffmpeg_result,
        }

    try:
        first = Image.open(frames_dir / "frame_0001.png").convert("RGB")
        width, height = first.size
        ensure_dir(target_video.parent)
        writer = cv2.VideoWriter(
            str(target_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )
        if not writer.isOpened():
            return {
                "status": "blocked",
                "blockers": ["opencv_video_writer_failed_for_default_wam_segment"],
                "generated_video_segment_path": str(target_video),
                "frame_dir": str(frames_dir),
                "ffmpeg_result": ffmpeg_result,
            }
        try:
            for index in range(frame_count):
                frame = Image.open(frames_dir / f"frame_{index + 1:04d}.png").convert("RGB")
                writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
    except Exception as exc:
        return {
            "status": "blocked",
            "blockers": [f"opencv_video_segment_write_failed:{type(exc).__name__}"],
            "generated_video_segment_path": str(target_video),
            "frame_dir": str(frames_dir),
            "ffmpeg_result": ffmpeg_result,
        }
    completed = target_video.is_file() and target_video.stat().st_size > 0
    return {
        "status": "completed" if completed else "blocked",
        "blockers": [] if completed else ["default_wam_video_segment_missing_after_write"],
        "generated_video_segment_path": str(target_video),
        "frame_dir": str(frames_dir),
        "frame_count": frame_count,
        "fps": fps,
        "encoder": "opencv_mp4v",
        "size_bytes": target_video.stat().st_size if target_video.is_file() else 0,
        "ffmpeg_result": ffmpeg_result,
    }


def _execute_default_local_wam_generation_step(
    *,
    job_dir: Path,
    loop_dir: Path,
    generated_at: str,
    step_index: int,
    source_frame: Path,
    target_frame: Path,
    current_action: Mapping[str, Any],
    current_observation: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    input_dir = loop_dir / "wam_generation_inputs"
    output_dir = loop_dir / "wam_generation_outputs"
    video_dir = loop_dir / "generated_next_observation_video_segments"
    input_path = input_dir / f"default_local_wam_step_{step_index:04d}_input.json"
    output_path = output_dir / f"default_local_wam_step_{step_index:04d}_output.json"
    video_path = video_dir / f"wam_generated_next_observation_step_{step_index:04d}.mp4"
    action_conditioning = _default_wam_action_conditioning(
        current_action=current_action,
        current_observation=current_observation,
    )
    skeleton_conditioning = _select_wam_skeleton_conditioning(
        job_dir=job_dir,
        source_frame=source_frame,
    )
    input_payload = {
        "schema_version": "default_local_wam_generation_step_input.v1",
        "generated_at": generated_at,
        "step_index": step_index,
        "wam_evaluator_backend": "blueprint_default_oscar_style_wam",
        "source_policy_observation_frame_path": str(source_frame),
        "source_policy_action": dict(current_action),
        "current_policy_observation": dict(current_observation),
        "action_conditioning": action_conditioning,
        "skeleton_conditioning": skeleton_conditioning,
        "requested_output": {
            "next_observation_frame_path": str(target_frame),
            "next_observation_video_segment_path": str(video_path),
            "action_conditioned_generation_required": True,
            "action_conditioned_video_segment_required": True,
        },
        "claim_boundary": {
            "default_local_wam_generator_used": True,
            "oscar_style_action_skeleton_conditioning": True,
            "learned_oscar_or_cosmos_model_ran": False,
            "generated_next_observation_is_not_raw_capture": True,
            "physical_robot_sensor_proof": False,
        },
    }
    write_json(input_path, input_payload)
    started = time.monotonic()
    materialization = _render_default_wam_next_observation_frame(
        source_frame=source_frame,
        target_frame=target_frame,
        step_index=step_index,
        action_conditioning=action_conditioning,
        skeleton_conditioning=skeleton_conditioning,
    )
    frame_completed = materialization.get("status") == "completed" and target_frame.is_file()
    video_materialization = (
        _write_default_wam_next_observation_video_segment(
            generated_frame=target_frame,
            target_video=video_path,
            step_index=step_index,
        )
        if frame_completed
        else {
            "status": "blocked",
            "blockers": ["default_wam_video_segment_not_attempted_without_frame"],
            "generated_video_segment_path": str(video_path),
        }
    )
    video_completed = video_materialization.get("status") == "completed" and video_path.is_file()
    completed = bool(frame_completed and video_completed)
    blockers = [str(item) for item in materialization.get("blockers", []) if str(item)]
    blockers.extend(str(item) for item in video_materialization.get("blockers", []) if str(item))
    if not frame_completed and not blockers:
        blockers.append("default_local_wam_generated_frame_missing")
    if frame_completed and not video_completed and not blockers:
        blockers.append("default_local_wam_generated_video_segment_missing")
    execution = {
        "schema_version": "wam_generation_command_step_execution.v1",
        "generated_at": generated_at,
        "step_index": step_index,
        "wam_evaluator_backend": "blueprint_default_oscar_style_wam",
        "selected_command_env": None,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "command_ran": False,
        "live_wam_generation_command_ran": False,
        "default_local_wam_generator_used": True,
        "duration_seconds": round(time.monotonic() - started, 6),
        "status": "completed" if completed else "blocked",
        "blockers": blockers,
    }
    output_summary = {
        "schema_version": "wam_generation_command_step_output.v1",
        "generated_at": generated_at,
        "step_index": step_index,
        "status": execution["status"],
        "wam_evaluator_backend": "blueprint_default_oscar_style_wam",
        "output_path": str(output_path),
        "materialization": materialization,
        "video_materialization": video_materialization,
        "wam_model_checkpoint_used": False,
        "action_conditioned_generation_ran": completed,
        "default_local_wam_generator_used": True,
        "live_wam_generation_command_ran": False,
        "learned_oscar_or_cosmos_model_ran": False,
        "action_conditioning": action_conditioning,
        "skeleton_conditioning": skeleton_conditioning,
        "blockers": blockers,
        "claim_boundary": {
            "generated_output_is_support_evidence_not_raw_capture": True,
            "default_local_wam_generator_is_not_live_oscar_or_cosmos_model": True,
            "learned_wam_checkpoint_invoked": False,
            "physical_robot_sensor_proof": False,
        },
    }
    write_json(
        output_path,
        {
            "schema_version": "default_local_wam_generation_step_output.v1",
            "generated_at": generated_at,
            "status": execution["status"],
            "generated_next_observation_frame_path": str(target_frame) if completed else None,
            "generated_next_observation_video_path": str(video_path) if completed else None,
            "wam_generation_step_output_summary": output_summary,
            "action_conditioning": action_conditioning,
            "skeleton_conditioning": skeleton_conditioning,
            "materialization": materialization,
            "video_materialization": video_materialization,
            "blockers": blockers,
        },
    )
    if not completed:
        return None, execution, output_summary
    generated_observation = {
        "schema_version": "wam_generated_next_observation.v1",
        "generated_at": generated_at,
        "generated_observation_index": step_index,
        "observation_source": "default_local_oscar_style_wam_next_observation",
        "wam_evaluator_backend": "blueprint_default_oscar_style_wam",
        "wam_model_checkpoint_used": False,
        "action_conditioned_generation_ran": True,
        "default_local_wam_generator_used": True,
        "live_wam_generation_command_ran": False,
        "learned_oscar_or_cosmos_model_ran": False,
        "generated_next_observation_frame_path": str(target_frame),
        "generated_next_observation_video_path": str(video_path),
        "source_policy_action": dict(current_action),
        "action_conditioning": action_conditioning,
        "skeleton_conditioning": skeleton_conditioning,
        "visual_observation": {
            "available": True,
            "camera_frame_path": str(target_frame),
            "generated_video_segment_path": str(video_path),
            "camera_id": "wam_generated_next_observation",
            "wam_generated_observation": True,
            "simulated_camera_view": True,
            "physical_robot_sensor_proof": False,
        },
        "state": {
            "generated_step_index": step_index,
            "previous_action_type": current_action.get("action_type"),
        },
        "claim_boundary": {
            "generated_observation_is_evaluator_output_not_raw_capture": True,
            "generated_observation_from_default_local_wam_generator": True,
            "generated_observation_from_live_wam_command": False,
            "default_local_wam_generator_is_not_live_oscar_or_cosmos_model": True,
            "learned_wam_checkpoint_invoked": False,
            "frame_copy_placeholder_until_live_wam_model_configured": False,
            "support_evidence_only": True,
            "physical_robot_sensor_proof": False,
        },
    }
    return generated_observation, execution, output_summary


def _execute_live_wam_generation_step(
    *,
    loop_dir: Path,
    generated_at: str,
    discovery: Mapping[str, Any],
    step_index: int,
    source_frame: Path,
    target_frame: Path,
    current_action: Mapping[str, Any],
    current_observation: Mapping[str, Any],
    timeout_seconds: float,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    selected_backend_id = _string(discovery.get("selected_backend_id"))
    selected_command_env = _string(discovery.get("selected_command_env"))
    command = os.getenv(selected_command_env, "").strip()
    input_dir = loop_dir / "wam_generation_inputs"
    output_dir = loop_dir / "wam_generation_outputs"
    input_path = input_dir / f"wam_generation_step_{step_index:04d}_input.json"
    output_path = output_dir / f"wam_generation_step_{step_index:04d}_output.json"
    ensure_dir(output_path.parent)
    auxiliary_observation = build_wam_auxiliary_observation_manifest(
        output_dir=input_dir / f"wam_auxiliary_observation_step_{step_index:04d}",
        source_image_path=source_frame,
        policy_observation=current_observation,
        source_policy_action=current_action,
        generated_at=generated_at,
        source_kind=_string(current_observation.get("source_kind"))
        or _string(_mapping(current_observation.get("visual_observation")).get("source_kind"))
        or None,
        camera_id=_string(_mapping(current_observation.get("visual_observation")).get("camera_id"))
        or _string(current_observation.get("camera_id"))
        or None,
        robot_profile_id=_string(current_observation.get("robot_profile_id")) or None,
        task_id=_string(current_observation.get("task_id")) or None,
        target_object_id=_string(current_observation.get("target_object_id")) or None,
    )
    auxiliary_observation_summary = summarize_wam_auxiliary_observation_manifest(
        auxiliary_observation
    )
    input_payload = {
        "schema_version": "wam_generation_step_input.v1",
        "generated_at": generated_at,
        "step_index": step_index,
        "wam_evaluator_backend": selected_backend_id,
        "source_policy_observation_frame_path": str(source_frame),
        "source_policy_action": dict(current_action),
        "current_policy_observation": dict(current_observation),
        "wam_auxiliary_observation_manifest_path": auxiliary_observation["manifest_path"],
        "auxiliary_observation": auxiliary_observation_summary,
        "requested_output": {
            "next_observation_frame_path": str(target_frame),
            "action_conditioned_generation_required": True,
        },
        "claim_boundary": {
            "wam_generation_is_not_robot_policy": True,
            "generated_next_observation_is_not_raw_capture": True,
            "physical_robot_sensor_proof": False,
        },
    }
    write_json(input_path, input_payload)
    if output_path.exists():
        output_path.unlink()
    started = time.monotonic()
    env = {
        **os.environ,
        "BLUEPRINT_WAM_ROLLOUT_INPUT": str(input_path),
        "BLUEPRINT_WAM_ROLLOUT_OUTPUT": str(output_path),
        "BLUEPRINT_WAM_PROVIDER_INPUT": str(input_path),
        "BLUEPRINT_WAM_PROVIDER_OUTPUT": str(output_path),
        "BLUEPRINT_WAM_PROVIDER_SUBSTRATE": selected_backend_id,
        "BLUEPRINT_WAM_GENERATION_STEP_INDEX": str(step_index),
    }
    execution = {
        "schema_version": "wam_generation_command_step_execution.v1",
        "generated_at": generated_at,
        "step_index": step_index,
        "wam_evaluator_backend": selected_backend_id,
        "selected_command_env": selected_command_env,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "command_ran": False,
        "status": "blocked",
        "blockers": [],
        "wam_auxiliary_observation_manifest_path": auxiliary_observation["manifest_path"],
    }
    payload: dict[str, Any] = {}
    try:
        completed = subprocess.run(
            shlex.split(command),
            cwd=str(loop_dir),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        execution.update(
            {
                "command_ran": True,
                "returncode": completed.returncode,
                "duration_seconds": round(time.monotonic() - started, 6),
                "stdout_size_bytes": len(completed.stdout or ""),
                "stderr_size_bytes": len(completed.stderr or ""),
                "stdout_omitted_to_avoid_secret_leakage": bool(completed.stdout),
                "stderr_omitted_to_avoid_secret_leakage": bool(completed.stderr),
                "status": "completed" if completed.returncode == 0 else "blocked",
                "blockers": []
                if completed.returncode == 0
                else ["wam_generation_command_nonzero_exit"],
            }
        )
        if output_path.is_file():
            value = json.loads(output_path.read_text(encoding="utf-8"))
            payload = dict(value) if isinstance(value, Mapping) else {}
        elif completed.stdout.strip():
            value = json.loads(completed.stdout)
            payload = dict(value) if isinstance(value, Mapping) else {}
    except Exception as exc:
        execution.update(
            {
                "duration_seconds": round(time.monotonic() - started, 6),
                "blockers": [f"wam_generation_command_failed:{type(exc).__name__}"],
            }
        )

    response_blockers = payload.get("blockers")
    if isinstance(response_blockers, Sequence) and not isinstance(
        response_blockers, (str, bytes, bytearray)
    ):
        execution["blockers"] = sorted(
            set(list(execution.get("blockers", [])) + [str(item) for item in response_blockers])
        )
    elif response_blockers:
        execution["blockers"] = sorted(
            set(list(execution.get("blockers", [])) + [str(response_blockers)])
        )
    if _string(payload.get("status")) in {"blocked", "failed"}:
        execution["status"] = "blocked"
        if not execution.get("blockers"):
            execution["blockers"] = ["wam_generation_command_output_blocked"]

    materialized_frame, materialization = _materialize_wam_generated_frame(
        payload=payload,
        output_path=output_path,
        target_frame=target_frame,
    )
    if execution.get("status") == "completed" and materialized_frame is None:
        execution["status"] = "blocked"
        execution["blockers"] = sorted(
            set(
                list(execution.get("blockers", []))
                + [str(materialization.get("blocker") or "wam_generated_frame_missing")]
            )
        )
    output_summary = {
        "schema_version": "wam_generation_command_step_output.v1",
        "generated_at": generated_at,
        "step_index": step_index,
        "status": execution["status"],
        "wam_evaluator_backend": selected_backend_id,
        "output_path": str(output_path),
        "payload_redacted": _redact(payload) if payload else {},
        "materialization": materialization,
        "wam_auxiliary_observation": auxiliary_observation_summary,
        "wam_auxiliary_observation_manifest_path": auxiliary_observation["manifest_path"],
        "wam_model_checkpoint_used": any(
            bool(row.get("checkpoint_configured"))
            for row in discovery.get("candidates", [])
            if row.get("backend_id") == selected_backend_id
        ),
        "action_conditioned_generation_ran": bool(execution["status"] == "completed"),
        "default_local_wam_generator_used": False,
        "live_wam_generation_command_ran": bool(
            execution.get("command_ran") and execution.get("status") == "completed"
        ),
        "learned_oscar_or_cosmos_model_ran": bool(execution["status"] == "completed"),
        "blockers": execution.get("blockers", []),
    }
    if execution["status"] != "completed" or materialized_frame is None:
        return None, execution, output_summary
    generated_observation = {
        "schema_version": "wam_generated_next_observation.v1",
        "generated_at": generated_at,
        "generated_observation_index": step_index,
        "observation_source": "live_wam_generation_command_next_observation",
        "wam_evaluator_backend": selected_backend_id,
        "wam_model_checkpoint_used": output_summary["wam_model_checkpoint_used"],
        "action_conditioned_generation_ran": True,
        "default_local_wam_generator_used": False,
        "live_wam_generation_command_ran": True,
        "learned_oscar_or_cosmos_model_ran": True,
        "generated_next_observation_frame_path": str(materialized_frame),
        "source_policy_action": dict(current_action),
        "visual_observation": {
            "available": True,
            "camera_frame_path": str(materialized_frame),
            "camera_id": "wam_generated_next_observation",
            "wam_generated_observation": True,
            "simulated_camera_view": True,
            "physical_robot_sensor_proof": False,
        },
        "state": {
            "generated_step_index": step_index,
            "previous_action_type": current_action.get("action_type"),
        },
        "claim_boundary": {
            "generated_observation_is_evaluator_output_not_raw_capture": True,
            "generated_observation_from_live_wam_command": True,
            "generated_observation_from_default_local_wam_generator": False,
            "frame_copy_placeholder_until_live_wam_model_configured": False,
            "physical_robot_sensor_proof": False,
        },
    }
    return generated_observation, execution, output_summary


def _write_wam_generation_command_artifacts(
    *,
    loop_dir: Path,
    generated_at: str,
    discovery: Mapping[str, Any],
    execution_steps: Sequence[Mapping[str, Any]],
    output_steps: Sequence[Mapping[str, Any]],
    structural_generation_count: int,
    default_generation_count: int,
    blockers: Sequence[str],
) -> None:
    live_success_count = sum(
        1
        for row in output_steps
        if row.get("action_conditioned_generation_ran")
        and row.get("live_wam_generation_command_ran")
    )
    default_success_count = sum(
        1
        for row in output_steps
        if row.get("action_conditioned_generation_ran")
        and row.get("default_local_wam_generator_used")
    )
    action_conditioned_success_count = live_success_count + default_success_count
    command_ran_count = sum(1 for row in execution_steps if row.get("command_ran"))
    write_json(
        loop_dir / "wam_generation_command_execution.json",
        {
            "schema_version": "wam_generation_command_execution.v1",
            "generated_at": generated_at,
            "status": "completed" if action_conditioned_success_count else "blocked",
            "selected_backend_id": discovery.get("selected_backend_id"),
            "selected_command_env": discovery.get("selected_command_env"),
            "command_step_count": len(execution_steps),
            "command_ran_count": command_ran_count,
            "live_wam_generation_success_count": live_success_count,
            "default_wam_generation_success_count": default_success_count,
            "action_conditioned_generation_success_count": action_conditioned_success_count,
            "default_local_wam_generator_used": bool(default_success_count),
            "structural_fallback_generation_count": structural_generation_count,
            "default_local_generation_count": default_generation_count,
            "steps": [dict(row) for row in execution_steps],
            "blockers": sorted(set(str(item) for item in blockers if str(item))),
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
            "claim_boundary": {
                "wam_generation_command_is_not_robot_policy": True,
                "generated_world_rank_fidelity_result_proven": False,
                "generated_world_policy_evaluation_scope_proven": False,
                "non_ranking_operational_claim_proven": False,
            },
        },
    )
    write_json(
        loop_dir / "wam_generation_command_output.json",
        {
            "schema_version": "wam_generation_command_output.v1",
            "generated_at": generated_at,
            "status": "completed" if action_conditioned_success_count else "blocked",
            "wam_evaluator_backend": (
                discovery.get("selected_backend_id")
                if live_success_count
                else (
                    "blueprint_default_oscar_style_wam"
                    if default_success_count
                    else discovery.get("selected_backend_id")
                )
            ),
            "wam_model_checkpoint_used": any(
                bool(row.get("wam_model_checkpoint_used")) for row in output_steps
            ),
            "action_conditioned_generation_ran": bool(action_conditioned_success_count),
            "live_generated_next_observation_count": live_success_count,
            "default_generated_next_observation_count": default_success_count,
            "action_conditioned_generated_next_observation_count": (
                action_conditioned_success_count
            ),
            "default_local_wam_generator_used": bool(default_success_count),
            "learned_oscar_or_cosmos_model_ran": bool(live_success_count),
            "structural_fallback_generation_count": structural_generation_count,
            "default_local_generation_count": default_generation_count,
            "outputs": [dict(row) for row in output_steps],
            "blockers": sorted(set(str(item) for item in blockers if str(item))),
            "claim_boundary": {
                "generated_outputs_are_not_raw_capture_evidence": True,
                "wam_generation_output_is_not_robot_policy": True,
                "local_structural_wam_generator_is_not_live_oscar_or_cosmos_model": (
                    structural_generation_count > 0
                ),
                "default_local_wam_generator_is_not_live_oscar_or_cosmos_model": bool(
                    default_success_count
                ),
                "default_local_outputs_are_support_evidence_only": bool(default_success_count),
                "learned_wam_checkpoint_invoked": bool(live_success_count),
                "generated_world_rank_fidelity_result_proven": False,
                "generated_world_policy_evaluation_scope_proven": False,
                "non_ranking_operational_claim_proven": False,
            },
        },
    )


def _policy_action_trace_summary(action: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = _mapping(action)
    chunk = None
    for key in (
        "action_chunk",
        "action",
        "joint_targets",
        "actions",
        "action_vector",
        "joint_positions",
    ):
        chunk = payload.get(key)
        if chunk is not None:
            break
    if isinstance(chunk, Mapping) and chunk.get("action_chunk") is not None:
        chunk = chunk.get("action_chunk")
    if isinstance(chunk, Mapping):
        chunk_length = len(chunk)
        chunk_sample = dict(list(chunk.items())[:6])
    elif isinstance(chunk, (list, tuple)):
        chunk_length = len(chunk)
        chunk_sample = list(chunk[:6])
    else:
        chunk_length = None
        chunk_sample = None
    return {
        "action_type": payload.get("action_type"),
        "action_keys": sorted(str(key) for key in payload.keys()),
        "action_chunk_length": chunk_length,
        "action_chunk_sample": chunk_sample,
        "arm_targets_present": "arm_targets" in payload,
        "hand_targets_present": "hand_targets" in payload,
        "gripper_targets_present": "gripper_targets" in payload,
        "joint_targets_present": "joint_targets" in payload,
    }


def _trace_html_image_src(frame_path: Any, *, relative_to: Path) -> str | None:
    if not isinstance(frame_path, str) or not frame_path:
        return None
    try:
        return os.path.relpath(Path(frame_path), relative_to)
    except ValueError:
        return frame_path


def _write_robot_policy_wam_side_by_side_trace_html(
    *,
    html_path: Path,
    rows: Sequence[Mapping[str, Any]],
    selected_candidate_id: str,
) -> None:
    html_dir = html_path.parent
    cards: list[str] = []
    for row in rows:
        policy_src = _trace_html_image_src(row.get("policy_pov_frame_path"), relative_to=html_dir)
        wam_src = _trace_html_image_src(
            row.get("wam_generated_next_observation_frame_path"), relative_to=html_dir
        )
        action_summary = html.escape(
            json.dumps(row.get("policy_action_summary") or {}, indent=2, sort_keys=True)
        )
        next_action_summary = html.escape(
            json.dumps(row.get("next_policy_action_summary") or {}, indent=2, sort_keys=True)
        )
        policy_img = (
            f'<img src="{html.escape(policy_src)}" alt="Policy POV frame">'
            if policy_src
            else '<div class="missing">missing policy POV frame</div>'
        )
        wam_img = (
            f'<img src="{html.escape(wam_src)}" alt="WAM generated next observation">'
            if wam_src
            else '<div class="missing">missing WAM generated next observation</div>'
        )
        cards.append(
            "\n".join(
                [
                    '<section class="transition">',
                    f"<h2>Transition {html.escape(str(row.get('transition_index')))}</h2>",
                    '<div class="grid">',
                    "<div><h3>Policy POV Input</h3>",
                    policy_img,
                    f"<p>source: {html.escape(str(row.get('policy_observation_source')))}</p></div>",
                    "<div><h3>GR00T/SONIC Action Summary</h3>",
                    f"<pre>{action_summary}</pre></div>",
                    "<div><h3>WAM Generated Next Observation</h3>",
                    wam_img,
                    (f"<p>backend: {html.escape(str(row.get('wam_evaluator_backend')))}</p></div>"),
                    "<div><h3>Next Policy Call</h3>",
                    (
                        f"<p>status: {html.escape(str(row.get('next_policy_call_status')))}</p>"
                        f"<p>provider replay: "
                        f"{html.escape(str(row.get('next_policy_call_provider_output_replay_used')))}</p>"
                    ),
                    f"<pre>{next_action_summary}</pre></div>",
                    "</div>",
                    "</section>",
                ]
            )
        )
    page = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>Robot Policy WAM Side By Side Trace</title>",
            "<style>",
            "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;background:#f6f7f9;color:#16181d}",
            "h1{font-size:24px;margin:0 0 8px} h2{font-size:18px;margin:0 0 12px} h3{font-size:13px;margin:0 0 8px;color:#3b4250}",
            ".meta{margin:0 0 20px;color:#4b5563}.transition{background:white;border:1px solid #d9dee7;border-radius:8px;margin:0 0 18px;padding:16px}",
            ".grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start}img{width:100%;height:auto;border:1px solid #d9dee7;border-radius:6px;background:#111}",
            "pre{white-space:pre-wrap;word-break:break-word;background:#111827;color:#f9fafb;border-radius:6px;padding:12px;font-size:12px;line-height:1.4;margin:0}",
            "p{font-size:12px;color:#4b5563;margin:8px 0 0}.missing{border:1px dashed #9ca3af;border-radius:6px;padding:32px;text-align:center;color:#6b7280}",
            "@media (max-width:900px){.grid{grid-template-columns:1fr}}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Robot Policy WAM Side By Side Trace</h1>",
            (
                f'<p class="meta">candidate: {html.escape(selected_candidate_id)} | '
                f"transitions: {len(rows)} | simulator structural debug artifact only</p>"
            ),
            *cards,
            "</body>",
            "</html>",
        ]
    )
    html_path.write_text(page, encoding="utf-8")


def _write_robot_policy_wam_side_by_side_trace(
    *,
    loop_dir: Path,
    generated_at: str,
    selected_candidate_id: str,
    policy_calls: Sequence[Mapping[str, Any]],
    generated_observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    trace_path = loop_dir / "robot_policy_wam_side_by_side_trace.jsonl"
    manifest_path = loop_dir / "robot_policy_wam_side_by_side_trace_manifest.json"
    html_path = loop_dir / "robot_policy_wam_side_by_side_trace.html"
    rows: list[dict[str, Any]] = []
    for index, generated_observation in enumerate(generated_observations, start=1):
        policy_call = _mapping(policy_calls[index - 1]) if index - 1 < len(policy_calls) else {}
        next_policy_call = _mapping(policy_calls[index]) if index < len(policy_calls) else {}
        generated_visual = _mapping(generated_observation.get("visual_observation"))
        rows.append(
            {
                "schema_version": "robot_policy_wam_side_by_side_trace_row.v1",
                "generated_at": generated_at,
                "transition_index": index,
                "selected_candidate_id": selected_candidate_id,
                "policy_observation_step_index": policy_call.get("step_index"),
                "policy_observation_source": policy_call.get("observation_source"),
                "policy_pov_frame_path": policy_call.get("policy_observation_frame_path"),
                "policy_action_summary": _policy_action_trace_summary(
                    _mapping(policy_call.get("action_payload_redacted"))
                ),
                "policy_action_output_path": policy_call.get("output_path"),
                "wam_generated_next_observation_index": generated_observation.get(
                    "generated_observation_index"
                ),
                "wam_generated_next_observation_source": generated_observation.get(
                    "observation_source"
                ),
                "wam_generated_next_observation_frame_path": generated_visual.get(
                    "camera_frame_path"
                ),
                "wam_evaluator_backend": generated_observation.get("wam_evaluator_backend"),
                "next_policy_call_step_index": next_policy_call.get("step_index"),
                "next_policy_call_input_path": next_policy_call.get("input_path"),
                "next_policy_call_output_path": next_policy_call.get("output_path"),
                "next_policy_call_status": next_policy_call.get("status"),
                "next_policy_action_summary": _policy_action_trace_summary(
                    _mapping(next_policy_call.get("action_payload_redacted"))
                ),
                "next_policy_call_unitree_policy_action_command_ran": bool(
                    next_policy_call.get("unitree_policy_action_command_ran")
                ),
                "next_policy_call_provider_output_replay_used": bool(
                    next_policy_call.get("provider_output_replay_used")
                ),
                "claim_boundary": {
                    "side_by_side_trace_is_structural_debug_artifact": True,
                    "wam_generated_observation_is_not_raw_capture": True,
                    "physical_robot_sensor_proof": False,
                    "generated_world_rank_fidelity_result_proven": False,
                },
            }
        )
    _write_jsonl(trace_path, rows)
    _write_robot_policy_wam_side_by_side_trace_html(
        html_path=html_path,
        rows=rows,
        selected_candidate_id=selected_candidate_id,
    )
    manifest = {
        "schema_version": "robot_policy_wam_side_by_side_trace_manifest.v1",
        "generated_at": generated_at,
        "status": "completed" if rows else "blocked",
        "selected_candidate_id": selected_candidate_id,
        "trace_path": str(trace_path),
        "trace_html_path": str(html_path),
        "transition_count": len(rows),
        "policy_call_count": len(policy_calls),
        "generated_next_observation_count": len(generated_observations),
        "row_contract": (
            "policy POV frame -> policy action chunk summary -> WAM/generated next "
            "observation frame -> next policy call"
        ),
        "claim_boundary": {
            "side_by_side_trace_is_not_task_success": True,
            "simulator_only": True,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def run_robot_policy_wam_closed_loop_attempt(
    *,
    job_dir: Path,
    generated_at: str,
    policy_action_model_command_execution: Mapping[str, Any],
    loop_step_count: int = DEFAULT_WAM_LOOP_STEP_COUNT,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    discovery = _mapping(policy_action_model_command_execution.get("discovery"))
    if not discovery and (job_dir / "policy_action_model_command_discovery.json").is_file():
        discovery = _mapping(
            json.loads(
                (job_dir / "policy_action_model_command_discovery.json").read_text(encoding="utf-8")
            )
        )
    selected_candidate_id = _string(
        policy_action_model_command_execution.get("selected_candidate_id")
        or discovery.get("selected_candidate_id")
    )
    candidates = [
        _mapping(row) for row in discovery.get("candidates", []) or [] if isinstance(row, Mapping)
    ]
    selected = next(
        (
            row
            for row in candidates
            if row.get("candidate_id") == selected_candidate_id
            and row.get("ready_for_policy_action_command")
        ),
        None,
    )
    loop_dir = job_dir / "robot_policy_wam_closed_loop"
    generated_dir = loop_dir / "generated_next_observations"
    policy_call_dir = loop_dir / "policy_calls"
    harness_dir = loop_dir / "wam_derived_observation_harness"
    ensure_dir(generated_dir)
    ensure_dir(policy_call_dir)
    ensure_dir(harness_dir)
    trace_path = loop_dir / "robot_policy_wam_loop_trace.jsonl"
    generated_observation_trace_path = loop_dir / "wam_generated_next_observations.jsonl"
    wam_generation_discovery = discover_wam_generation_command(generated_at=generated_at)
    write_json(loop_dir / "wam_generation_command_discovery.json", wam_generation_discovery)
    harness_backend_config = _wam_perception_harness_backend_config()
    blockers: list[str] = []
    if selected is None:
        blockers.extend(
            discovery.get("blockers", []) or ["blocked_missing_unitree_policy_action_model_command"]
        )
    if (
        selected_candidate_id
        and selected_candidate_id not in UNITREE_MANIPULATION_POLICY_ACTION_MODEL_CANDIDATE_IDS
    ):
        blockers.append("blocked_selected_policy_is_not_unitree_manipulation_action_command")
    if not policy_action_model_command_execution.get("policy_action_model_command_ran"):
        blockers.extend(
            str(item)
            for item in policy_action_model_command_execution.get("blockers", [])
            or ["blocked_initial_unitree_policy_action_command_not_run"]
        )
    initial_action = _mapping(policy_action_model_command_execution.get("action_payload_redacted"))
    if not initial_action:
        blockers.append("blocked_initial_unitree_policy_action_missing")
    if blockers:
        harness_artifacts = write_wam_derived_observation_artifacts(
            output_dir=harness_dir,
            generated_at=generated_at,
            steps=[],
            adapter_reports=[],
        )
        harness_summary = summarize_wam_derived_observation_artifacts(harness_artifacts)
        _write_wam_generation_command_artifacts(
            loop_dir=loop_dir,
            generated_at=generated_at,
            discovery=wam_generation_discovery,
            execution_steps=[],
            output_steps=[],
            structural_generation_count=0,
            default_generation_count=0,
            blockers=list(wam_generation_discovery.get("blockers", []))
            + ["blocked_policy_loop_did_not_reach_wam_generation"],
        )
        side_by_side_manifest = _write_robot_policy_wam_side_by_side_trace(
            loop_dir=loop_dir,
            generated_at=generated_at,
            selected_candidate_id=selected_candidate_id or "",
            policy_calls=[],
            generated_observations=[],
        )
        manifest = {
            "schema_version": "robot_policy_wam_closed_loop_attempt.v1",
            "generated_at": generated_at,
            "status": "blocked",
            "selected_candidate_id": selected_candidate_id or None,
            "wam_evaluator_in_control_loop": False,
            "policy_observes_wam_generated_next_observation": False,
            "unitree_policy_action_command_ran": bool(
                policy_action_model_command_execution.get("unitree_policy_action_command_ran")
            ),
            "unitree_lerobot_policy_action_command_ran": bool(
                policy_action_model_command_execution.get(
                    "unitree_lerobot_policy_action_command_ran"
                )
            ),
            "unitree_unifolm_policy_action_command_ran": bool(
                policy_action_model_command_execution.get(
                    "unitree_unifolm_policy_action_command_ran"
                )
            ),
            "unitree_groot_n17_sonic_policy_action_command_ran": bool(
                policy_action_model_command_execution.get(
                    "unitree_groot_n17_sonic_policy_action_command_ran"
                )
            ),
            "repeated_policy_calls_count": 1
            if policy_action_model_command_execution.get("policy_action_model_command_ran")
            else 0,
            "generated_next_observation_count": 0,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
            "accepted_anchor_manipulation_success_proven": False,
            "blockers": sorted(set(blockers)),
            "trace_path": str(trace_path),
            "generated_next_observation_trace": str(generated_observation_trace_path),
            "wam_derived_observation_manifest": harness_summary["artifact_paths"].get(
                "wam_derived_observation_manifest"
            ),
            "wam_derived_observation_bundle": harness_summary["artifact_paths"].get(
                "wam_derived_observation_bundle"
            ),
            "wam_perception_harness_checks": harness_summary["artifact_paths"].get(
                "wam_perception_harness_checks"
            ),
            "wam_policy_observation_adapter_report": harness_summary["artifact_paths"].get(
                "wam_policy_observation_adapter_report"
            ),
            "wam_perception_harness_validation_report": harness_summary["artifact_paths"].get(
                "wam_perception_harness_validation_report"
            ),
            "wam_false_success_reduction_metrics": harness_summary["artifact_paths"].get(
                "wam_false_success_reduction_metrics"
            ),
            "wam_perception_harness_review_report": harness_summary["artifact_paths"].get(
                "wam_perception_harness_review_report"
            ),
            "wam_derived_observation_step_count": harness_summary.get("step_count"),
            "wam_derived_observation_early_termination_recommended": (
                harness_summary.get("early_termination_recommended")
            ),
            "wam_perception_harness_validation_status": harness_summary.get(
                "validation_status"
            ),
            "wam_false_success_reduction_status": harness_summary.get(
                "false_success_reduction_status"
            ),
            "wam_perception_harness_backend_config": {
                "backend_kind": harness_backend_config.get("backend_kind"),
                "env_gate": harness_backend_config.get("env_gate"),
                "command_env": harness_backend_config.get("command_env"),
                "configured_for_external_backend": harness_backend_config.get(
                    "configured_for_external_backend"
                ),
                "raw_credentials_written_to_artifacts": False,
            },
            "wam_generation_command_discovery": str(
                loop_dir / "wam_generation_command_discovery.json"
            ),
            "wam_generation_command_execution": str(
                loop_dir / "wam_generation_command_execution.json"
            ),
            "wam_generation_command_output": str(loop_dir / "wam_generation_command_output.json"),
            "live_wam_generation_command_ran": False,
            "action_conditioned_generation_ran": False,
            "live_wam_generation_success_count": 0,
            "default_wam_generation_success_count": 0,
            "default_local_wam_generator_used": False,
            "structural_wam_generation_count": 0,
            "side_by_side_trace_manifest": str(
                loop_dir / "robot_policy_wam_side_by_side_trace_manifest.json"
            ),
            "side_by_side_trace_path": str(loop_dir / "robot_policy_wam_side_by_side_trace.jsonl"),
            "side_by_side_trace_html_path": str(
                loop_dir / "robot_policy_wam_side_by_side_trace.html"
            ),
            "side_by_side_transition_count": int(
                side_by_side_manifest.get("transition_count") or 0
            ),
            "claim_boundary": {
                "simulator_only": True,
                "closed_loop_requires_repeated_unitree_manipulation_policy_calls": True,
                "wam_generated_observations_are_model_or_evaluator_outputs_not_raw_capture": True,
                "generated_world_rank_fidelity_result_proven": False,
                "generated_world_policy_evaluation_scope_proven": False,
            },
        }
        write_json(loop_dir / "robot_policy_wam_closed_loop_attempt.json", manifest)
        return manifest

    max_calls = max(2, int(loop_step_count))
    policy_calls: list[dict[str, Any]] = [
        {
            "step_index": 0,
            "observation_source": "initial_mujoco_policy_observation",
            "status": "completed",
            "selected_candidate_id": selected_candidate_id,
            "action_payload_redacted": initial_action,
            "policy_action_model_command_ran": bool(
                policy_action_model_command_execution.get("policy_action_model_command_ran")
            ),
            "unitree_policy_action_command_ran": bool(
                policy_action_model_command_execution.get("unitree_policy_action_command_ran")
            ),
            "unitree_lerobot_policy_action_command_ran": bool(
                policy_action_model_command_execution.get(
                    "unitree_lerobot_policy_action_command_ran"
                )
            ),
            "unitree_unifolm_policy_action_command_ran": bool(
                policy_action_model_command_execution.get(
                    "unitree_unifolm_policy_action_command_ran"
                )
            ),
            "unitree_groot_n17_sonic_policy_action_command_ran": bool(
                policy_action_model_command_execution.get(
                    "unitree_groot_n17_sonic_policy_action_command_ran"
                )
            ),
            "unitree_manipulation_policy_action_command_ran": bool(
                policy_action_model_command_execution.get(
                    "unitree_manipulation_policy_action_command_ran"
                )
            ),
            "unitree_specific_policy_candidate_ran": bool(
                policy_action_model_command_execution.get("unitree_specific_policy_candidate_ran")
            ),
            "unitree_specific_manipulation_candidate_ran": bool(
                policy_action_model_command_execution.get(
                    "unitree_specific_manipulation_candidate_ran"
                )
            ),
            "provider_output_replay_used": bool(
                policy_action_model_command_execution.get("provider_output_replay_used")
            ),
            "fresh_policy_action_model_executed_this_invocation": bool(
                policy_action_model_command_execution.get(
                    "fresh_policy_action_model_executed_this_invocation"
                )
            ),
        }
    ]
    generated_observations: list[dict[str, Any]] = []
    current_action = initial_action
    base_packet = _initial_policy_action_loop_observation(
        generated_at=generated_at,
        job_dir=job_dir,
    )
    current_observation = _mapping(base_packet.get("observation"))
    initial_visual = _mapping(current_observation.get("visual_observation"))
    if initial_visual.get("capture_derived_robot_pov_synthesis_used"):
        policy_calls[0]["observation_source"] = "initial_capture_derived_robot_pov_observation"
        policy_calls[0]["capture_derived_robot_pov_synthesis_used"] = True
    if initial_visual.get("external_photoreal_observation_used"):
        policy_calls[0]["observation_source"] = "initial_external_photoreal_observation"
        policy_calls[0]["photoreal_observation_source"] = initial_visual.get(
            "photoreal_observation_source"
        )
    source_frame_text = _mapping(current_observation.get("visual_observation")).get(
        "camera_frame_path"
    ) or current_observation.get("camera_frame_path")
    source_frame = Path(str(source_frame_text)).expanduser() if source_frame_text else None
    if source_frame is not None:
        policy_calls[0]["policy_observation_frame_path"] = str(source_frame)
    wam_execution_steps: list[dict[str, Any]] = []
    wam_output_steps: list[dict[str, Any]] = []
    derived_observation_steps: list[dict[str, Any]] = []
    policy_adapter_reports: list[dict[str, Any]] = []
    harness_artifacts: dict[str, Any] | None = None
    structural_wam_generation_count = 0
    live_wam_generation_success_count = 0
    default_wam_generation_success_count = 0
    declared_policy_observation_schema = _declared_policy_observation_schema_for_wam_loop(
        selected_candidate_id
    )
    for step_index in range(1, max_calls):
        if source_frame is None or not source_frame.is_file():
            blockers.append("blocked_wam_loop_source_policy_frame_missing")
            break
        generated_frame = (
            generated_dir / f"wam_generated_next_observation_step_{step_index:04d}.jpg"
        )
        if wam_generation_discovery.get("ready_for_live_wam_generation"):
            generated_observation, execution_step, output_step = _execute_live_wam_generation_step(
                loop_dir=loop_dir,
                generated_at=generated_at,
                discovery=wam_generation_discovery,
                step_index=step_index,
                source_frame=source_frame,
                target_frame=generated_frame,
                current_action=current_action,
                current_observation=current_observation,
                timeout_seconds=timeout_seconds,
            )
            wam_execution_steps.append(execution_step)
            wam_output_steps.append(output_step)
            if generated_observation is None:
                blockers.extend(str(item) for item in execution_step.get("blockers", []))
                if not execution_step.get("blockers"):
                    blockers.append("blocked_live_wam_generation_command_failed")
                break
            live_wam_generation_success_count += 1
        else:
            generated_observation, execution_step, output_step = (
                _execute_default_local_wam_generation_step(
                    job_dir=job_dir,
                    loop_dir=loop_dir,
                    generated_at=generated_at,
                    step_index=step_index,
                    source_frame=source_frame,
                    target_frame=generated_frame,
                    current_action=current_action,
                    current_observation=current_observation,
                )
            )
            wam_execution_steps.append(execution_step)
            wam_output_steps.append(output_step)
            if generated_observation is None:
                blockers.extend(str(item) for item in execution_step.get("blockers", []))
                if not execution_step.get("blockers"):
                    blockers.append("blocked_default_local_wam_generation_failed")
                break
            default_wam_generation_success_count += 1
        generated_observations.append(generated_observation)
        harness_result = run_wam_derived_observation_harness_step(
            output_dir=harness_dir,
            generated_at=generated_at,
            step_index=step_index,
            source_generated_frame_path=generated_frame,
            source_generated_video_path=generated_observation.get(
                "generated_next_observation_video_path"
            ),
            source_wam_rollout_id=f"wam_policy_loop_step_{step_index:04d}",
            transition_id=f"policy_wam_transition_{step_index:04d}",
            source_policy_action=current_action,
            action_history=[
                _mapping(row.get("action_payload_redacted"))
                for row in policy_calls
                if _mapping(row.get("action_payload_redacted"))
            ],
            current_policy_observation=current_observation,
            skeleton_conditioning=_mapping(generated_observation.get("skeleton_conditioning")),
            controller_limits=_mapping(current_observation.get("safety_limits")),
            previous_steps=derived_observation_steps,
            previous_adapter_reports=policy_adapter_reports,
            backend_kind=str(harness_backend_config["backend_kind"]),
            backend_command=harness_backend_config.get("backend_command"),
            allow_external_backend=harness_backend_config.get("allow_external_backend"),
            policy_id=selected_candidate_id,
            declared_policy_observation_schema=declared_policy_observation_schema,
        )
        harness_artifacts = harness_result
        derived_observation_steps.append(dict(harness_result["step_record"]))
        policy_adapter_reports.append(dict(harness_result["policy_adapter_report"]))
        harness_summary = summarize_wam_derived_observation_artifacts(harness_result)
        generated_observation["wam_derived_observation"] = {
            "manifest_path": harness_summary["artifact_paths"].get(
                "wam_derived_observation_manifest"
            ),
            "bundle_path": harness_summary["artifact_paths"].get(
                "wam_derived_observation_bundle"
            ),
            "checks_path": harness_summary["artifact_paths"].get(
                "wam_perception_harness_checks"
            ),
            "adapter_report_path": harness_summary["artifact_paths"].get(
                "wam_policy_observation_adapter_report"
            ),
            "validation_report_path": harness_summary["artifact_paths"].get(
                "wam_perception_harness_validation_report"
            ),
            "review_report_path": harness_summary["artifact_paths"].get(
                "wam_perception_harness_review_report"
            ),
            "step_index": step_index,
            "status": harness_result["step_record"].get("status"),
            "overall_confidence": _mapping(
                harness_result["step_record"].get("uncertainty")
            ).get("overall_confidence"),
            "early_termination_recommended": _mapping(
                harness_result["step_record"].get("uncertainty")
            ).get("early_termination_recommended"),
            "policy_adapter_safe_for_policy_requery": _mapping(
                harness_result["policy_adapter_report"]
            ).get("safe_for_policy_requery"),
        }
        if generated_observations:
            generated_observations[-1] = generated_observation
        if _mapping(harness_result["step_record"].get("uncertainty")).get(
            "early_termination_recommended"
        ):
            blockers.extend(
                str(item)
                for item in harness_result["step_record"].get("blockers", [])
                if str(item)
            )
            if not harness_result["step_record"].get("blockers"):
                blockers.append("blocked_wam_derived_observation_reliability_too_low")
            break
        packet = {
            **base_packet,
            "generated_at": generated_at,
            "observation": dict(harness_result["adapted_policy_observation"]),
        }
        result = _execute_policy_action_model_candidate(
            job_dir=job_dir,
            candidate=selected,
            payload=packet,
            input_path=policy_call_dir / f"policy_call_{step_index:04d}_input.json",
            output_path=policy_call_dir / f"policy_call_{step_index:04d}_output.json",
            timeout_seconds=timeout_seconds,
        )
        policy_calls.append(
            {
                "step_index": step_index,
                "observation_source": "wam_generated_next_observation",
                "policy_observation_frame_path": str(generated_frame),
                "wam_generated_next_observation_index": step_index,
                **result,
            }
        )
        if result.get("status") != "completed":
            blockers.extend(str(item) for item in result.get("blockers", []))
            break
        current_action = _mapping(result.get("action_payload_redacted"))
        current_observation = _mapping(packet.get("observation"))
        source_frame = generated_frame
    wam_artifact_blockers = list(blockers)
    action_conditioned_generation_success_count = (
        live_wam_generation_success_count + default_wam_generation_success_count
    )
    if action_conditioned_generation_success_count == 0:
        wam_artifact_blockers.extend(
            str(item) for item in wam_generation_discovery.get("blockers", [])
        )
        wam_artifact_blockers.append("blocked_wam_action_conditioned_generation_not_run")
    _write_wam_generation_command_artifacts(
        loop_dir=loop_dir,
        generated_at=generated_at,
        discovery=wam_generation_discovery,
        execution_steps=wam_execution_steps,
        output_steps=wam_output_steps,
        structural_generation_count=structural_wam_generation_count,
        default_generation_count=default_wam_generation_success_count,
        blockers=wam_artifact_blockers,
    )
    _write_jsonl(trace_path, policy_calls)
    _write_jsonl(generated_observation_trace_path, generated_observations)
    structural_action_responses = sum(1 for row in policy_calls if row.get("status") == "completed")
    replay_action_responses = sum(
        1
        for row in policy_calls
        if row.get("status") == "completed" and row.get("provider_output_replay_used")
    )
    repeated_policy_calls = sum(
        1
        for row in policy_calls
        if row.get("status") == "completed"
        and row.get("unitree_policy_action_command_ran")
        and not row.get("provider_output_replay_used")
    )
    generated_count = len(generated_observations)
    if structural_action_responses >= 2 and generated_count >= 1 and repeated_policy_calls < 2:
        blockers.append("blocked_repeated_fresh_unitree_policy_calls_not_proven")
    if generated_count >= 1 and action_conditioned_generation_success_count == 0:
        blockers.append("blocked_wam_action_conditioned_generation_not_run")
    completed = bool(
        repeated_policy_calls >= 2
        and action_conditioned_generation_success_count >= 1
        and generated_count >= 1
        and not blockers
    )
    side_by_side_manifest = _write_robot_policy_wam_side_by_side_trace(
        loop_dir=loop_dir,
        generated_at=generated_at,
        selected_candidate_id=selected_candidate_id,
        policy_calls=policy_calls,
        generated_observations=generated_observations,
    )
    if harness_artifacts is None:
        harness_artifacts = write_wam_derived_observation_artifacts(
            output_dir=harness_dir,
            generated_at=generated_at,
            steps=derived_observation_steps,
            adapter_reports=policy_adapter_reports,
        )
    harness_summary = summarize_wam_derived_observation_artifacts(harness_artifacts)
    manifest = {
        "schema_version": "robot_policy_wam_closed_loop_attempt.v1",
        "generated_at": generated_at,
        "status": "completed" if completed else "blocked",
        "selected_candidate_id": selected_candidate_id,
        "requested_loop_step_count": int(loop_step_count),
        "wam_evaluator_in_control_loop": completed,
        "policy_observes_wam_generated_next_observation": completed,
        "unitree_policy_action_command_ran": any(
            row.get("unitree_policy_action_command_ran") for row in policy_calls
        ),
        "unitree_lerobot_policy_action_command_ran": any(
            row.get("unitree_lerobot_policy_action_command_ran") for row in policy_calls
        ),
        "unitree_unifolm_policy_action_command_ran": any(
            row.get("unitree_unifolm_policy_action_command_ran") for row in policy_calls
        ),
        "unitree_groot_n17_sonic_policy_action_command_ran": any(
            row.get("unitree_groot_n17_sonic_policy_action_command_ran") for row in policy_calls
        ),
        "repeated_policy_calls_count": repeated_policy_calls,
        "fresh_policy_action_call_count": repeated_policy_calls,
        "structural_policy_action_response_count": structural_action_responses,
        "provider_output_replay_action_response_count": replay_action_responses,
        "provider_output_replay_used": replay_action_responses > 0,
        "generated_next_observation_count": generated_count,
        "live_wam_generation_command_ran": any(
            row.get("command_ran") for row in wam_execution_steps
        ),
        "action_conditioned_generation_ran": bool(action_conditioned_generation_success_count),
        "live_wam_generation_success_count": live_wam_generation_success_count,
        "default_wam_generation_success_count": default_wam_generation_success_count,
        "action_conditioned_generation_success_count": (
            action_conditioned_generation_success_count
        ),
        "default_local_wam_generator_used": bool(default_wam_generation_success_count),
        "learned_oscar_or_cosmos_model_ran": bool(live_wam_generation_success_count),
        "structural_wam_generation_count": structural_wam_generation_count,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "non_ranking_operational_claim_proven": False,
        "accepted_anchor_manipulation_success_proven": False,
        "policy_call_trace_path": str(trace_path),
        "generated_next_observation_trace": str(generated_observation_trace_path),
        "wam_derived_observation_manifest": harness_summary["artifact_paths"].get(
            "wam_derived_observation_manifest"
        ),
        "wam_derived_observation_bundle": harness_summary["artifact_paths"].get(
            "wam_derived_observation_bundle"
        ),
        "wam_perception_harness_checks": harness_summary["artifact_paths"].get(
            "wam_perception_harness_checks"
        ),
        "wam_policy_observation_adapter_report": harness_summary["artifact_paths"].get(
            "wam_policy_observation_adapter_report"
        ),
        "wam_perception_harness_validation_report": harness_summary["artifact_paths"].get(
            "wam_perception_harness_validation_report"
        ),
        "wam_false_success_reduction_metrics": harness_summary["artifact_paths"].get(
            "wam_false_success_reduction_metrics"
        ),
        "wam_perception_harness_review_report": harness_summary["artifact_paths"].get(
            "wam_perception_harness_review_report"
        ),
        "wam_derived_observation_steps": harness_summary["artifact_paths"].get(
            "wam_derived_observation_steps"
        ),
        "wam_derived_observation_step_count": harness_summary.get("step_count"),
        "wam_derived_observation_early_termination_recommended": (
            harness_summary.get("early_termination_recommended")
        ),
        "wam_derived_observation_success_scoring_blocked": harness_summary.get(
            "success_scoring_blocked"
        ),
        "wam_policy_observation_adapter_safe_for_policy_requery": harness_summary.get(
            "policy_adapter_safe_for_policy_requery"
        ),
        "wam_perception_harness_backend_config": {
            "backend_kind": harness_backend_config.get("backend_kind"),
            "env_gate": harness_backend_config.get("env_gate"),
            "command_env": harness_backend_config.get("command_env"),
            "configured_for_external_backend": harness_backend_config.get(
                "configured_for_external_backend"
            ),
            "raw_credentials_written_to_artifacts": False,
        },
        "wam_perception_harness_validation_status": harness_summary.get(
            "validation_status"
        ),
        "wam_false_success_reduction_status": harness_summary.get(
            "false_success_reduction_status"
        ),
        "wam_false_success_reduction_rate": harness_summary.get(
            "false_success_reduction_rate"
        ),
        "wam_generation_command_discovery": str(loop_dir / "wam_generation_command_discovery.json"),
        "wam_generation_command_execution": str(loop_dir / "wam_generation_command_execution.json"),
        "wam_generation_command_output": str(loop_dir / "wam_generation_command_output.json"),
        "policy_call_output_dir": str(policy_call_dir),
        "generated_next_observation_dir": str(generated_dir),
        "side_by_side_trace_manifest": str(
            loop_dir / "robot_policy_wam_side_by_side_trace_manifest.json"
        ),
        "side_by_side_trace_path": str(loop_dir / "robot_policy_wam_side_by_side_trace.jsonl"),
        "side_by_side_trace_html_path": str(loop_dir / "robot_policy_wam_side_by_side_trace.html"),
        "side_by_side_transition_count": int(side_by_side_manifest.get("transition_count") or 0),
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "simulator_only": True,
            "wam_is_next_observation_generator_not_robot_policy": True,
            "unitree_policy_is_robot_policy": True,
            "generated_observations_are_not_raw_capture": True,
            "wam_derived_observations_are_not_real_sensors": True,
            "wam_derived_depth_is_not_sensor_depth": True,
            "wam_derived_masks_are_not_physical_truth": True,
            "wam_derived_contact_likelihood_is_not_physical_contact_proof": True,
            "local_structural_wam_generator_is_not_live_oscar_or_cosmos_model": (
                structural_wam_generation_count > 0
            ),
            "default_local_wam_generator_is_not_live_oscar_or_cosmos_model": (
                default_wam_generation_success_count > 0
            ),
            "default_local_outputs_are_support_evidence_only": (
                default_wam_generation_success_count > 0
            ),
            "learned_wam_checkpoint_invoked": live_wam_generation_success_count > 0,
            "frame_copy_placeholder_until_live_wam_model_configured": (
                structural_wam_generation_count > 0
            ),
            "provider_output_replay_is_not_fresh_per_observation_policy_execution": (
                replay_action_responses > 0
            ),
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
        },
    }
    write_json(loop_dir / "robot_policy_wam_closed_loop_attempt.json", manifest)
    return manifest


def discover_unitree_manipulation_policy(*, generated_at: str) -> dict[str, Any]:
    """Discover configured G1 hand/gripper manipulation policy candidates.

    This is intentionally discovery-only. The MuJoCo endpoint lane can prove
    lower-body contact traces, but a dexterous/VLA manipulation claim requires
    a separate command/checkpoint that controls hands or grippers.
    """

    workspace_root = _repo_root().parent
    candidate_rows: list[dict[str, Any]] = []
    ready_candidates: list[dict[str, Any]] = []
    for candidate in UNITREE_G1_MANIPULATION_POLICY_CANDIDATES:
        command_env = str(candidate["command_env"])
        checkpoint_env = str(candidate["checkpoint_env"])
        root_env = str(candidate["root_env"])
        command_value = os.getenv(command_env, "").strip()
        checkpoint_value = os.getenv(checkpoint_env, "").strip()
        root_value = os.getenv(root_env, "").strip()
        (
            checkpoint_ready,
            checkpoint_reference,
            checkpoint_exists,
            checkpoint_reference_kind,
        ) = _configured_checkpoint_reference(checkpoint_value)
        root_path = Path(root_value).expanduser() if root_value else None
        local_checkout_rows = []
        for relative_path in candidate.get("expected_local_paths", ()):
            path = workspace_root / str(relative_path)
            local_checkout_rows.append(
                {
                    "path": str(path),
                    "exists": path.exists(),
                }
            )
        command_ready = _command_available(command_value)
        root_ready = bool(root_path and root_path.exists())
        extra_checkpoint_rows = []
        for env_name in candidate.get("extra_required_checkpoint_envs", ()):
            value = os.getenv(str(env_name), "").strip()
            configured, path_text, exists, reference_kind = _configured_checkpoint_reference(value)
            extra_checkpoint_rows.append(
                {
                    "checkpoint_env": str(env_name),
                    "checkpoint_configured": configured,
                    "checkpoint_exists": exists,
                    "checkpoint_path": path_text,
                    "checkpoint_reference_kind": reference_kind,
                }
            )
        extra_root_rows = []
        for env_name in candidate.get("extra_required_root_envs", ()):
            value = os.getenv(str(env_name), "").strip()
            path = Path(value).expanduser() if value else None
            extra_root_rows.append(
                {
                    "root_env": str(env_name),
                    "root_configured": bool(value),
                    "root_exists": bool(path and path.exists()),
                    "root_path": str(path) if path else None,
                }
            )
        extra_checkpoints_ready = all(
            row.get("checkpoint_configured") for row in extra_checkpoint_rows
        )
        extra_roots_ready = all(
            row.get("root_configured") and row.get("root_exists") for row in extra_root_rows
        )
        candidate_ready = bool(
            command_ready
            and checkpoint_ready
            and (root_ready or not root_value)
            and extra_checkpoints_ready
            and extra_roots_ready
        )
        row = {
            "id": candidate["id"],
            "name": candidate["name"],
            "url": candidate["url"],
            "runtime_role": candidate["runtime_role"],
            "command_env": command_env,
            "command_configured": bool(command_value),
            "command_available": command_ready,
            "command_value_redacted": "<configured>" if command_value else None,
            "checkpoint_env": checkpoint_env,
            "checkpoint_configured": bool(checkpoint_value),
            "checkpoint_exists": checkpoint_exists,
            "checkpoint_path": checkpoint_reference,
            "checkpoint_reference_kind": checkpoint_reference_kind,
            "extra_required_checkpoints": extra_checkpoint_rows,
            "root_env": root_env,
            "root_configured": bool(root_value),
            "root_exists": root_ready,
            "root_path": str(root_path) if root_path else None,
            "extra_required_roots": extra_root_rows,
            "local_checkout_candidates": local_checkout_rows,
            "ready_for_hand_or_gripper_policy_execution": candidate_ready,
            "claim_boundary": candidate["claim_boundary"],
            "missing_requirements": [
                requirement
                for requirement, missing in (
                    ("runnable_manipulation_policy_command", not command_ready),
                    ("local_or_mounted_policy_checkpoint", not checkpoint_ready),
                    ("configured_runtime_root", bool(root_value) and not root_ready),
                    (
                        "extra_required_checkpoint",
                        any(not row.get("checkpoint_configured") for row in extra_checkpoint_rows),
                    ),
                    (
                        "extra_required_runtime_root",
                        any(
                            not (row.get("root_configured") and row.get("root_exists"))
                            for row in extra_root_rows
                        ),
                    ),
                )
                if missing
            ],
        }
        candidate_rows.append(row)
        if candidate_ready:
            ready_candidates.append(row)
    blockers = []
    if not ready_candidates:
        blockers.extend(
            [
                "blocked_dexterous_hand_policy_not_integrated",
                "blocked_missing_unitree_or_vla_manipulation_command_or_checkpoint",
            ]
        )
    return {
        "schema_version": "unitree_g1_manipulation_policy_discovery.v1",
        "generated_at": generated_at,
        "status": "candidate_ready" if ready_candidates else "blocked_missing_hand_policy_runtime",
        "pre_execution_discovery_only": True,
        "selected_candidate_id": ready_candidates[0]["id"] if ready_candidates else None,
        "candidate_count": len(candidate_rows),
        "ready_candidate_count": len(ready_candidates),
        "candidates": candidate_rows,
        "unitree_lerobot_or_isaaclab_manipulation_policy_used": False,
        "unitree_hand_manipulation_policy_used": False,
        "hand_end_effector_control_available_in_current_mujoco_lane": False,
        "current_mujoco_manipulation_policy_kind": "contact_trace_proxy_only",
        "can_claim_vla_or_dexterous_manipulation": False,
        "blockers": blockers,
        "what_would_make_this_true": [
            "Configure a runnable manipulation policy command via the candidate command env.",
            "Configure a local or mounted checkpoint via the candidate checkpoint env.",
            "Add a Blueprint hand/gripper action decoder and execute it in the task scene.",
            "Record generated action traces and task success/failure evidence from that execution.",
        ],
        "claim_boundary": {
            "lower_body_locomotion_policy_does_not_prove_hand_manipulation": True,
            "contact_trace_proxy_is_not_dexterous_vla_manipulation": True,
            "isaaclab_or_lerobot_candidate_discovery_is_not_execution_proof": True,
            "generated_world_rank_fidelity_result_proven": False,
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
            "sidecar_execution_is_not_generated_world_rank_fidelity": True,
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
                "velocity_frame": action.get("velocity_frame"),
                "world_velocity_xy_mps": [
                    round(float(action.get("vx_mps") or 0.0), 6),
                    round(float(action.get("vy_mps") or 0.0), 6),
                ],
                "controller_velocity_xy_mps": [
                    round(float(action.get("controller_vx_mps", action.get("vx_mps")) or 0.0), 6),
                    round(float(action.get("controller_vy_mps", action.get("vy_mps")) or 0.0), 6),
                ],
                "target_waypoint": action.get("target_waypoint"),
            }
        )
    return command_rows


def _bounded_float(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


def _world_xy_velocity_to_body_frame(
    vx_mps: float, vy_mps: float, yaw_rad: float
) -> tuple[float, float]:
    cos_yaw = math.cos(float(yaw_rad))
    sin_yaw = math.sin(float(yaw_rad))
    body_vx = cos_yaw * float(vx_mps) + sin_yaw * float(vy_mps)
    body_vy = -sin_yaw * float(vx_mps) + cos_yaw * float(vy_mps)
    return body_vx, body_vy


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
            abs(float(before) - float(after)) > 1e-9 for before, after in zip(raw, command)
        ),
        "controller_command_limits": dict(limits),
    }


def _unitree_controller_safe_command(action: Mapping[str, Any]) -> dict[str, Any]:
    return _unitree_controller_safe_command_from_values(
        action.get("controller_vx_mps", action.get("vx_mps")),
        action.get("controller_vy_mps", action.get("vy_mps")),
        action.get("yaw_rate_rad_s"),
    )


def _representative_unitree_command(
    command_rows: Sequence[Mapping[str, Any]],
) -> list[float] | None:
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
            return [
                round(float(command[0]), 6),
                round(float(command[1]), 6),
                round(float(command[2]), 6),
            ]
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
            "generated_world_rank_fidelity_result_proven": False,
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
        "official_unitree_controller_used": replay_proven
        or sidecar_proven
        or same_scene_integrated,
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
            "controller_sidecar_or_replay_is_not_generated_world_rank_fidelity": True,
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
            "auth_token_file_exists": bool(
                _mapping(selected_runtime).get("auth_token_file_exists")
            ),
            "raw_token_values_persisted": False,
            "raw_token_hashes_persisted": False,
        },
        "health_probe": dict(health_probe),
        "claim_boundary": {
            "server_reachable_is_not_policy_quality_proof": True,
            "generated_world_rank_fidelity_result_proven": False,
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
    reference_policy_observed = any(
        policy_id in {"reference_fixture_policy", "blueprint_g1_endpoint_reference_adapter"}
        or policy_id.startswith("reference_")
        for policy_id in policy_ids
    )
    unitree_policy_observed = any(policy_id.startswith("unitree_") for policy_id in policy_ids)
    unitree_provider_replay_observed = any(
        policy_id.startswith("unitree_") and policy_id.endswith("_provider_replay")
        for policy_id in policy_ids
    )
    openvla_policy_observed = any(policy_id.startswith("openvla") for policy_id in policy_ids)
    wam_policy_observed = any(
        policy_id.startswith("oscar_") or policy_id.startswith("cosmos_")
        for policy_id in policy_ids
    )
    return {
        "schema_version": "policy_command_adapter_manifest.v1",
        "generated_at": generated_at,
        "status": "completed" if rows else "defined",
        "adapter_families": [
            "command_policy",
            "provider_worker_policy",
            "unitree_g1_policy",
            GROOT_POLICY_ID,
            "unitree_lerobot_policy",
            "unitree_unifolm_policy",
            "openvla_policy",
            "oscar_wam",
            "cosmos_wam",
        ],
        "default_reference_adapter_command": "blueprint-g1-endpoint-reference-adapter",
        "default_reference_adapter_available_on_path": bool(
            shutil.which("blueprint-g1-endpoint-reference-adapter")
        ),
        "openvla_policy_adapter_command": "blueprint-openvla-policy-command-adapter",
        "openvla_policy_adapter_available_on_path": bool(
            shutil.which("blueprint-openvla-policy-command-adapter")
        ),
        "provider_worker_policy_adapter_command": (
            "blueprint-provider-worker-policy-command-adapter"
        ),
        "provider_worker_policy_adapter_available_on_path": bool(
            shutil.which("blueprint-provider-worker-policy-command-adapter")
        ),
        "provider_worker_policy_adapter_contract": {
            "requires_readyz_before_infer": True,
            "does_not_allocate_provider": True,
            "reuses_already_allocated_worker": True,
        },
        "unitree_groot_n17_sonic_policy_adapter_command": (
            "blueprint-unitree-groot-n17-sonic-policy-command-adapter"
        ),
        "unitree_groot_n17_sonic_policy_adapter_available_on_path": bool(
            shutil.which("blueprint-unitree-groot-n17-sonic-policy-command-adapter")
        ),
        "observed_endpoint_policy_ids": policy_ids,
        "observed_adapter_families": {
            "reference_policy_observed": reference_policy_observed,
            "unitree_policy_observed": unitree_policy_observed,
            "unitree_provider_output_replay_observed": unitree_provider_replay_observed,
            "openvla_policy_observed": openvla_policy_observed,
            "wam_policy_observed": wam_policy_observed,
        },
        "observed_policy_truth_boundary": {
            "unitree_provider_output_replay_is_not_fresh_per_observation_inference": (
                unitree_provider_replay_observed
            ),
            "openvla_observed_policy_is_not_default_unitree_g1_policy": (openvla_policy_observed),
            "wam_observed_policy_is_evaluator_support_not_g1_robot_policy": (wam_policy_observed),
            "reference_policy_observed_is_endpoint_plumbing_only": (reference_policy_observed),
        },
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
            "observed_policy_ids_drive_adapter_family_fields": True,
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
    unitree_endpoint_policy_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    unitree_summary = _mapping(unitree_endpoint_policy_summary)
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
        "g1_robot_policy_selection_contract": unitree_summary.get(
            "g1_robot_policy_selection_contract",
            "unitree_native_policy_required_for_g1_claims",
        ),
        "g1_robot_policy_selected_family": unitree_summary.get("g1_robot_policy_selected_family"),
        "unitree_hand_manipulation_policy_scope": unitree_summary.get(
            "unitree_hand_manipulation_policy_scope"
        ),
        "openvla_selected_as_g1_robot_policy": bool(
            unitree_summary.get("openvla_selected_as_g1_robot_policy", False)
        ),
        "wam_rollout_selected_as_g1_robot_policy": bool(
            unitree_summary.get("wam_rollout_selected_as_g1_robot_policy", False)
        ),
        "unitree_endpoint_hand_policy_output_observed": bool(
            unitree_summary.get("unitree_endpoint_hand_policy_output_observed", False)
        ),
        "unitree_endpoint_hand_policy_used": bool(
            unitree_summary.get("unitree_endpoint_hand_policy_used", False)
        ),
        "unitree_endpoint_provider_output_replay_used": bool(
            unitree_summary.get("unitree_endpoint_provider_output_replay_used", False)
        ),
        "unitree_endpoint_fresh_policy_action_command_ran": bool(
            unitree_summary.get("unitree_endpoint_fresh_policy_action_command_ran", False)
        ),
        "unitree_endpoint_action_chunk_used": bool(
            unitree_summary.get("unitree_endpoint_action_chunk_used", False)
        ),
        "raw_tokens_written_to_artifacts": False,
        "raw_token_hashes_written_to_artifacts": False,
        "claim_boundary": {
            "endpoint_invocation_is_not_model_quality_proof": True,
            "unitree_endpoint_provider_replay_is_not_fresh_hand_policy_inference": bool(
                unitree_summary.get("unitree_endpoint_provider_output_replay_used", False)
            ),
            "openvla_is_not_selected_as_g1_robot_policy": True,
            "wam_rollout_is_not_selected_as_g1_robot_policy": True,
            "mujoco_evidence_is_simulator_only": True,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
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
                    "velocity_frame": {"enum": ["robot_base", "world_xy"]},
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
                    "max_speed_mps": {"type": "number"},
                    "approach_speed_mps": {"type": "number"},
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
    <body name="blueprint_light_object" pos="0.36 -0.65 0.24">
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


def _select_g1_policy_observation_mjcf(g1_root: Path) -> tuple[Path, dict[str, Any]]:
    preferred = g1_root / PREFERRED_G1_POLICY_OBSERVATION_MJCF
    fallback = g1_root / FALLBACK_G1_POLICY_OBSERVATION_MJCF
    selected = preferred if preferred.is_file() else fallback
    hand_mesh_names = (
        "left_hand_palm_link.STL",
        "right_hand_palm_link.STL",
        "left_hand_index_0_link.STL",
        "right_hand_index_0_link.STL",
    )
    hand_meshes_present = all((g1_root / "assets" / name).is_file() for name in hand_mesh_names)
    return selected, {
        "preferred_g1_mjcf": str(preferred),
        "fallback_g1_mjcf": str(fallback),
        "selected_g1_mjcf": str(selected),
        "selected_g1_mjcf_name": selected.name,
        "hands_capable_g1_mjcf_available": preferred.is_file(),
        "hands_capable_g1_mjcf_selected": selected == preferred,
        "hand_meshes_present": hand_meshes_present,
        "hand_mesh_probe_names": list(hand_mesh_names),
        "claim_boundary": {
            "hands_capable_mjcf_improves_simulated_egocentric_visual_observation": (
                selected == preferred
            ),
            "hands_capable_mjcf_does_not_prove_dexterous_hand_policy_execution": True,
            "hands_capable_mjcf_is_not_physical_robot_sensor_proof": True,
        },
    }


def _add_g1_fixed_egocentric_cameras(g1_xml: Path) -> dict[str, Any]:
    camera_specs = [
        {
            "camera_id": "head_pov",
            "name": FIXED_G1_CAMERA_NAMES["head_pov"],
            "body": "torso_link",
            "mount": "upper_torso_head_proxy",
            "pos": "0.10 0 0.225",
            "xyaxes": "0 -1 0 0.174 0 0.985",
        },
        {
            "camera_id": "torso_pov",
            "name": FIXED_G1_CAMERA_NAMES["torso_pov"],
            "body": "torso_link",
            "mount": "torso_forward_proxy",
            "pos": "0.10 0 0.115",
            "xyaxes": "0 -1 0 0.342 0 0.940",
        },
    ]
    try:
        tree = DefusedET.parse(g1_xml)
        root = tree.getroot()
    except Exception as exc:
        return {
            "schema_version": "g1_fixed_egocentric_camera_injection.v1",
            "status": "blocked",
            "g1_xml": str(g1_xml),
            "blockers": ["blocked_g1_xml_parse_failed"],
            "error": str(exc),
            "mounted_camera_count": 0,
            "mounted_cameras": [],
        }
    mounted: list[dict[str, Any]] = []
    blockers: list[str] = []
    existing_names = {
        camera.get("name") for camera in root.findall(".//camera") if camera.get("name")
    }
    for spec in camera_specs:
        if spec["name"] in existing_names:
            mounted.append({**spec, "status": "already_present"})
            continue
        body = root.find(f".//body[@name='{spec['body']}']")
        if body is None:
            blockers.append(f"blocked_missing_camera_mount_body:{spec['body']}")
            continue
        ET.SubElement(
            body,
            "camera",
            {
                "name": spec["name"],
                "mode": "fixed",
                "pos": spec["pos"],
                "xyaxes": spec["xyaxes"],
                "fovy": "75",
            },
        )
        mounted.append({**spec, "status": "mounted"})
    try:
        tree.write(g1_xml, encoding="unicode")
    except Exception as exc:
        return {
            "schema_version": "g1_fixed_egocentric_camera_injection.v1",
            "status": "blocked",
            "g1_xml": str(g1_xml),
            "blockers": ["blocked_g1_xml_camera_write_failed"],
            "error": str(exc),
            "mounted_camera_count": 0,
            "mounted_cameras": mounted,
        }
    return {
        "schema_version": "g1_fixed_egocentric_camera_injection.v1",
        "status": "completed" if not blockers else "blocked",
        "g1_xml": str(g1_xml),
        "fixed_camera_names": dict(FIXED_G1_CAMERA_NAMES),
        "mounted_camera_count": len(mounted),
        "mounted_cameras": mounted,
        "blockers": blockers,
        "truth_boundary": {
            "camera_mounted_in_mujoco_g1_mjcf": not blockers and bool(mounted),
            "camera_is_simulated_sensor_view": True,
            "camera_is_not_physical_robot_sensor_proof": True,
        },
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
        names = {name for item in metadata for name in set(item.get("names") or set()) if name}
        floor_contact = "blueprint_reference_floor" in names
        object_contact = "blueprint_light_object" in names or "blueprint_light_object_geom" in names
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


def _projected_gravity_from_quat(quat: Sequence[float]) -> list[float]:
    qw, qx, qy, qz = [float(value) for value in quat[:4]]
    return [
        round(float(2 * (-qz * qx + qw * qy)), 8),
        round(float(-2 * (qz * qy + qw * qx)), 8),
        round(float(1 - 2 * (qw * qw + qz * qz)), 8),
    ]


def _joint_qpos_values(
    *,
    mujoco_module: Any,
    model: Any,
    data: Any,
    joint_names: Sequence[str],
) -> tuple[list[float], list[str], list[int]]:
    values: list[float] = []
    missing: list[str] = []
    qpos_addrs: list[int] = []
    for joint_name in joint_names:
        joint_id = _joint_id_by_name(mujoco_module, model, joint_name)
        if joint_id < 0:
            missing.append(joint_name)
            continue
        qpos_addr = int(model.jnt_qposadr[joint_id])
        qpos_addrs.append(qpos_addr)
        values.append(round(float(data.qpos[qpos_addr]), 8))
    return values, missing, qpos_addrs


def _build_unitree_g1_sonic_state_from_mujoco(
    *,
    mujoco_module: Any,
    model: Any,
    data: Any,
    root_qpos: int,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    state: dict[str, list[float]] = {}
    missing: list[str] = []
    qpos_addrs: dict[str, list[int]] = {}
    for group_name, joint_names in UNITREE_G1_SONIC_STATE_JOINT_GROUPS.items():
        values, group_missing, group_addrs = _joint_qpos_values(
            mujoco_module=mujoco_module,
            model=model,
            data=data,
            joint_names=joint_names,
        )
        state[group_name] = values
        missing.extend(group_missing)
        qpos_addrs[group_name] = group_addrs
    root_quat = [float(value) for value in data.qpos[root_qpos + 3 : root_qpos + 7]]
    state["projected_gravity"] = _projected_gravity_from_quat(root_quat)
    metadata = {
        "schema_version": "unitree_g1_sonic_state_metadata.v1",
        "state_source": "simulated_mujoco_qpos_joint_groups",
        "required_joint_groups": {
            group_name: list(joint_names)
            for group_name, joint_names in UNITREE_G1_SONIC_STATE_JOINT_GROUPS.items()
        },
        "qpos_addresses": qpos_addrs,
        "missing_joint_names": sorted(set(missing)),
        "complete": not missing,
        "root_quaternion_wxyz": root_quat,
        "claim_boundary": {
            "simulated_mujoco_state": True,
            "physical_robot_proprioception": False,
            "state_extraction_is_not_policy_execution": True,
        },
    }
    return state, metadata


def _mujoco_object_id(mujoco_module: Any, model: Any, obj_kind: Any, name: str) -> int:
    try:
        return int(mujoco_module.mj_name2id(model, obj_kind, name))
    except Exception:
        return -1


def _matrix3_rows(values: Sequence[Any]) -> list[list[float]]:
    items = [float(value) for value in values[:9]]
    if len(items) < 9:
        items.extend([0.0] * (9 - len(items)))
    return [items[0:3], items[3:6], items[6:9]]


def _mat3_mul_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [
        float(matrix[row][0]) * float(vector[0])
        + float(matrix[row][1]) * float(vector[1])
        + float(matrix[row][2]) * float(vector[2])
        for row in range(3)
    ]


def _g1_body_landmark_position(
    *,
    mujoco_module: Any,
    model: Any,
    data: Any,
    body_name: str,
    local_offset_m: Sequence[float] | None = None,
) -> dict[str, Any]:
    body_id = _mujoco_object_id(
        mujoco_module,
        model,
        mujoco_module.mjtObj.mjOBJ_BODY,
        body_name,
    )
    if body_id < 0:
        return {
            "available": False,
            "body_name": body_name,
            "blockers": [f"missing_g1_body:{body_name}"],
        }
    origin = [float(value) for value in data.xpos[body_id][:3]]
    offset = [float(value) for value in (local_offset_m or [0.0, 0.0, 0.0])[:3]]
    if len(offset) < 3:
        offset.extend([0.0] * (3 - len(offset)))
    world = origin
    if any(abs(value) > 1e-12 for value in offset):
        body_xmat = _matrix3_rows(data.xmat[body_id])
        world_offset = _mat3_mul_vec(body_xmat, offset)
        world = [origin[index] + world_offset[index] for index in range(3)]
    return {
        "available": True,
        "body_name": body_name,
        "body_id": body_id,
        "local_offset_m": [round(value, 6) for value in offset],
        "world_xyz_m": [round(value, 6) for value in world],
    }


def _g1_body_landmark_position_from_spec(
    *,
    mujoco_module: Any,
    model: Any,
    data: Any,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    primary = _g1_body_landmark_position(
        mujoco_module=mujoco_module,
        model=model,
        data=data,
        body_name=str(spec["body_name"]),
        local_offset_m=spec.get("local_offset_m"),
    )
    if primary.get("available"):
        primary["landmark_source"] = "primary_g1_body"
        return primary
    fallback_body = _string(spec.get("fallback_body_name"))
    if not fallback_body:
        primary["landmark_source"] = "primary_g1_body_missing"
        return primary
    fallback = _g1_body_landmark_position(
        mujoco_module=mujoco_module,
        model=model,
        data=data,
        body_name=fallback_body,
        local_offset_m=spec.get("fallback_local_offset_m"),
    )
    if fallback.get("available"):
        fallback["landmark_source"] = "fallback_g1_body_with_local_offset"
        fallback["preferred_body_name"] = str(spec["body_name"])
        fallback["fallback_for_missing_preferred_body"] = True
        return fallback
    fallback["blockers"] = sorted(
        set(_string_list(primary.get("blockers")) + _string_list(fallback.get("blockers")))
    )
    fallback["landmark_source"] = "preferred_and_fallback_g1_bodies_missing"
    fallback["preferred_body_name"] = str(spec["body_name"])
    return fallback


def _fixed_camera_projection_context(
    *,
    mujoco_module: Any,
    model: Any,
    data: Any,
    camera_id: str,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    fixed_camera_name = FIXED_G1_CAMERA_NAMES.get(camera_id)
    if not fixed_camera_name:
        return {
            "available": False,
            "camera_id": camera_id,
            "blockers": [f"camera_not_fixed_g1_camera:{camera_id}"],
        }
    camera_obj_id = _mujoco_object_id(
        mujoco_module,
        model,
        mujoco_module.mjtObj.mjOBJ_CAMERA,
        fixed_camera_name,
    )
    if camera_obj_id < 0:
        return {
            "available": False,
            "camera_id": camera_id,
            "fixed_mujoco_camera_name": fixed_camera_name,
            "blockers": [f"missing_fixed_g1_camera:{fixed_camera_name}"],
        }
    try:
        fovy_deg = float(model.cam_fovy[camera_obj_id])
        cam_xpos = [float(value) for value in data.cam_xpos[camera_obj_id][:3]]
        cam_xmat = _matrix3_rows(data.cam_xmat[camera_obj_id])
    except Exception as exc:
        return {
            "available": False,
            "camera_id": camera_id,
            "fixed_mujoco_camera_name": fixed_camera_name,
            "camera_obj_id": camera_obj_id,
            "blockers": ["fixed_g1_camera_projection_metadata_unavailable"],
            "error": str(exc),
        }
    focal_px = 0.5 * float(image_height) / math.tan(math.radians(fovy_deg) / 2.0)
    return {
        "available": True,
        "camera_id": camera_id,
        "fixed_mujoco_camera_name": fixed_camera_name,
        "camera_obj_id": camera_obj_id,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "fovy_deg": round(fovy_deg, 6),
        "focal_length_px": round(focal_px, 6),
        "camera_world_xyz_m": [round(value, 6) for value in cam_xpos],
        "camera_xmat_row_major": [[round(value, 8) for value in row] for row in cam_xmat],
        "projection_method": "mujoco_fixed_camera_pinhole_from_data_cam_xpos_xmat",
    }


def _project_world_xyz_to_camera_pixel(
    *,
    world_xyz_m: Sequence[float],
    projection_context: Mapping[str, Any],
) -> dict[str, Any]:
    if not projection_context.get("available"):
        return {
            "available": False,
            "blockers": list(
                projection_context.get("blockers") or ["projection_context_unavailable"]
            ),
        }
    cam_pos = [float(value) for value in projection_context.get("camera_world_xyz_m", [])[:3]]
    if len(cam_pos) < 3:
        return {"available": False, "blockers": ["camera_world_position_unavailable"]}
    cam_xmat = projection_context.get("camera_xmat_row_major")
    if not (
        isinstance(cam_xmat, Sequence)
        and len(cam_xmat) >= 3
        and all(isinstance(row, Sequence) and len(row) >= 3 for row in cam_xmat[:3])
    ):
        return {"available": False, "blockers": ["camera_orientation_unavailable"]}
    world = [float(value) for value in world_xyz_m[:3]]
    if len(world) < 3:
        return {"available": False, "blockers": ["world_point_unavailable"]}
    delta = [world[index] - cam_pos[index] for index in range(3)]
    rows = [[float(value) for value in row[:3]] for row in cam_xmat[:3]]
    columns = [[rows[0][index], rows[1][index], rows[2][index]] for index in range(3)]
    camera_local = [
        sum(delta[index] * columns[axis][index] for index in range(3)) for axis in range(3)
    ]
    depth = abs(float(camera_local[2]))
    if depth <= 1e-9:
        return {
            "available": False,
            "camera_local_xyz": [round(value, 6) for value in camera_local],
            "blockers": ["camera_projection_depth_near_zero"],
        }
    width = int(projection_context.get("image_width") or 0)
    height = int(projection_context.get("image_height") or 0)
    focal_px = float(projection_context.get("focal_length_px") or 0.0)
    if width <= 0 or height <= 0 or focal_px <= 0.0:
        return {
            "available": False,
            "camera_local_xyz": [round(value, 6) for value in camera_local],
            "blockers": ["camera_projection_intrinsics_unavailable"],
        }
    u = width * 0.5 + focal_px * float(camera_local[0]) / depth
    v = height * 0.5 - focal_px * float(camera_local[1]) / depth
    return {
        "available": True,
        "u_px": round(u, 3),
        "v_px": round(v, 3),
        "depth_m_abs": round(depth, 6),
        "camera_local_xyz": [round(value, 6) for value in camera_local],
        "inside_image": bool(0.0 <= u < width and 0.0 <= v < height),
        "projection_depth_sign_abs_used": True,
    }


def _build_g1_projected_skeleton_trace_row(
    *,
    mujoco_module: Any,
    model: Any,
    data: Any,
    run: Mapping[str, Any],
    step: int,
    visual_observation: Mapping[str, Any],
) -> dict[str, Any]:
    camera_id = str(visual_observation.get("camera_id") or "head_pov")
    image_width = int(visual_observation.get("image_width") or 0)
    image_height = int(visual_observation.get("image_height") or 0)
    projection_context = _fixed_camera_projection_context(
        mujoco_module=mujoco_module,
        model=model,
        data=data,
        camera_id=camera_id,
        image_width=image_width,
        image_height=image_height,
    )
    landmarks: list[dict[str, Any]] = []
    blockers: list[str] = []
    for spec in G1_UPPER_BODY_LANDMARK_SPECS:
        landmark = {
            "landmark_id": spec["landmark_id"],
            **_g1_body_landmark_position_from_spec(
                mujoco_module=mujoco_module,
                model=model,
                data=data,
                spec=spec,
            ),
        }
        if landmark.get("available"):
            projection = _project_world_xyz_to_camera_pixel(
                world_xyz_m=landmark.get("world_xyz_m", []),
                projection_context=projection_context,
            )
            landmark["image_projection"] = projection
        else:
            blockers.extend(str(item) for item in landmark.get("blockers") or [])
        landmarks.append(landmark)
    projected_count = sum(
        1 for landmark in landmarks if _mapping(landmark.get("image_projection")).get("available")
    )
    if not projection_context.get("available"):
        blockers.extend(str(item) for item in projection_context.get("blockers") or [])
    if projected_count <= 0:
        blockers.append("no_g1_upper_body_landmarks_projected_into_camera")
    return {
        "schema_version": G1_PROJECTED_SKELETON_SCHEMA_ID,
        "status": "completed" if not blockers else "warning_partial_projection",
        "episode_id": run.get("episode_id"),
        "scenario_eval_run_id": run.get("scenario_eval_run_id"),
        "task_id": run.get("task_id"),
        "spawn_id": run.get("spawn_id"),
        "step": int(step),
        "sim_time_s": round(float(data.time), 9),
        "camera_id": camera_id,
        "camera_frame_path": visual_observation.get("camera_frame_path"),
        "visual_observation_available": bool(visual_observation.get("available")),
        "projection_context": projection_context,
        "landmarks": landmarks,
        "segments": [{"from": start, "to": end} for start, end in G1_UPPER_BODY_SKELETON_SEGMENTS],
        "available_landmark_count": sum(1 for landmark in landmarks if landmark.get("available")),
        "projected_landmark_count": projected_count,
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "uses_unitree_g1_mujoco_body_transforms": True,
            "uses_simulated_fixed_head_or_torso_camera_projection": bool(
                projection_context.get("available")
            ),
            "simulated_g1_kinematic_skeleton_available": projected_count > 0,
            "not_hand_drawn_stick_figure": True,
            "not_physical_robot_sensor_proof": True,
            "not_dexterous_hand_policy_execution": True,
            "projection_depth_sign_abs_used_for_robust_review_overlay": True,
        },
    }


def _g1_projected_skeleton_manifest(
    *,
    generated_at: str,
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    completed_rows = [
        row
        for row in rows
        if row.get("status") == "completed" or int(row.get("projected_landmark_count") or 0) > 0
    ]
    blockers: list[str] = []
    if not rows:
        blockers.append("blocked_no_policy_visual_observation_steps")
    if rows and not completed_rows:
        blockers.append("blocked_no_g1_projected_skeleton_rows")
    return {
        "schema_version": "g1_projected_skeleton_trace_manifest.v1",
        "generated_at": generated_at,
        "status": "completed" if completed_rows else "blocked",
        "trace_jsonl": str(output_path),
        "row_count": len(rows),
        "projectable_row_count": len(completed_rows),
        "landmark_ids": [str(spec["landmark_id"]) for spec in G1_UPPER_BODY_LANDMARK_SPECS],
        "segments": [{"from": start, "to": end} for start, end in G1_UPPER_BODY_SKELETON_SEGMENTS],
        "blockers": blockers,
        "claim_boundary": {
            "derived_from_unitree_g1_mujoco_body_transforms": True,
            "derived_from_head_or_torso_sim_camera_metadata": True,
            "simulated_g1_arm_hand_state_available_for_wam_conditioning": bool(completed_rows),
            "not_physical_robot_sensor_proof": True,
            "not_wam_generated_output": True,
            "not_success_review_label": True,
        },
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
    visual_observation: Mapping[str, Any] | None = None,
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
    visual = dict(visual_observation or {})
    if not visual:
        visual = {
            "available": False,
            "camera_frame_path": None,
            "camera_id": None,
            "blockers": ["policy_observation_frame_not_captured"],
            "claim_boundary": {
                "visual_observation_required_for_real_vla_policy": True,
                "missing_visual_observation_blocks_openvla_or_cosmos_policy_proof": True,
            },
        }
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
            "target_error_m": round(
                math.dist(root_pos[:2], [float(target[0]), float(target[1])]), 6
            ),
            "route_waypoints": run.get("route_waypoints") or [],
            "task_prompt": run.get("task_prompt"),
        },
        "object_state": _object_pose(data, object_qpos),
        "visual_observation": visual,
        "unitree_g1_sonic_state": visual.get("unitree_g1_sonic_state"),
        "unitree_g1_sonic_state_source": visual.get("unitree_g1_sonic_state_source"),
        "unitree_g1_sonic_state_metadata": visual.get("unitree_g1_sonic_state_metadata"),
        "sensor_surrogates": {
            "camera_surrogates": [
                "third_person",
                "overhead",
                "robot_follow",
                "robot_pov",
                "torso_pov",
            ],
            "visual_assets_required": False,
            "splat_ply_spz_required": False,
            "camera_frame_path": visual.get("camera_frame_path"),
            "first_person_policy_observation_candidate": bool(
                visual.get("first_person_policy_observation_candidate")
            ),
        },
        "task_prompt": run.get("task_prompt"),
        "allowed_action_schema": {
            "schema_id": ACTION_SCHEMA_ID,
            "supported_action_types": [
                "base_velocity",
                "heading_yaw",
                "waypoint",
                "stop",
                "inspect_look",
                "manipulation_contact",
            ],
        },
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
        object_pos = object_state.get("position") or [0.36, -0.65, 0.24]
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
    velocity_frame = "robot_base"
    target_waypoint: list[float] | None = None
    if action_type == "base_velocity":
        linear = _number(action.get("linear_velocity_mps"))
        if linear is None:
            rejected = {"reason": "base_velocity_missing_numeric_linear_velocity"}
        else:
            lateral = _number(action.get("lateral_velocity_mps"), 0.0) or 0.0
            raw_yaw_rate = (
                _number(action.get("yaw_rate_rad_s") or action.get("yaw_rate"), 0.0) or 0.0
            )
            requested_frame = str(action.get("velocity_frame") or "robot_base")
            velocity_frame = "world_xy" if requested_frame == "world_xy" else "robot_base"
            vx = max(
                -SAFETY_LIMITS["max_forward_velocity_mps"],
                min(SAFETY_LIMITS["max_forward_velocity_mps"], linear),
            )
            vy = max(
                -SAFETY_LIMITS["max_lateral_velocity_mps"],
                min(SAFETY_LIMITS["max_lateral_velocity_mps"], lateral),
            )
            yaw_rate = max(
                -SAFETY_LIMITS["max_yaw_rate_rad_s"],
                min(SAFETY_LIMITS["max_yaw_rate_rad_s"], raw_yaw_rate),
            )
    elif action_type == "heading_yaw":
        target_yaw = _number(action.get("target_yaw_rad"))
        if target_yaw is None:
            rejected = {"reason": "heading_yaw_missing_numeric_target_yaw"}
        else:
            diff = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
            yaw_rate = max(
                -SAFETY_LIMITS["max_yaw_rate_rad_s"],
                min(SAFETY_LIMITS["max_yaw_rate_rad_s"], diff * 2.0),
            )
    elif action_type == "waypoint":
        waypoint = action.get("waypoint")
        if (
            not isinstance(waypoint, Sequence)
            or isinstance(waypoint, (str, bytes))
            or len(waypoint) < 2
        ):
            rejected = {"reason": "waypoint_missing_xy"}
        else:
            try:
                target_waypoint = [
                    float(waypoint[0]),
                    float(waypoint[1]),
                    float(waypoint[2]) if len(waypoint) > 2 else 0.79,
                ]
                dx = target_waypoint[0] - float(base_pos[0])
                dy = target_waypoint[1] - float(base_pos[1])
                distance = math.hypot(dx, dy)
                if distance > SAFETY_LIMITS["max_waypoint_distance_m"]:
                    scale = SAFETY_LIMITS["max_waypoint_distance_m"] / distance
                    dx *= scale
                    dy *= scale
                    distance = SAFETY_LIMITS["max_waypoint_distance_m"]
                if distance > 1e-6:
                    requested_speed = _number(
                        action.get("max_speed_mps") or action.get("approach_speed_mps")
                    )
                    if requested_speed is not None:
                        speed = min(
                            SAFETY_LIMITS["max_forward_velocity_mps"],
                            max(0.0, requested_speed),
                            max(0.0, distance * 2.2),
                        )
                    else:
                        speed = min(
                            SAFETY_LIMITS["max_forward_velocity_mps"], max(0.12, distance * 2.2)
                        )
                    vx = speed * dx / distance
                    vy = speed * dy / distance
                    velocity_frame = "world_xy"
                    target_heading = math.atan2(dy, dx)
                    diff = math.atan2(
                        math.sin(target_heading - yaw), math.cos(target_heading - yaw)
                    )
                    yaw_rate = max(
                        -SAFETY_LIMITS["max_yaw_rate_rad_s"],
                        min(SAFETY_LIMITS["max_yaw_rate_rad_s"], diff * 1.5),
                    )
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
        if (
            isinstance(waypoint, Sequence)
            and not isinstance(waypoint, (str, bytes))
            and len(waypoint) >= 2
        ):
            target_waypoint = [
                float(waypoint[0]),
                float(waypoint[1]),
                float(waypoint[2]) if len(waypoint) > 2 else 0.79,
            ]
        else:
            object_state = _mapping(observation.get("object_state"))
            object_pos = object_state.get("position") or [0.36, -0.65, 0.24]
            target_waypoint = [float(object_pos[0]) + 0.18, float(object_pos[1]), 0.79]
        dx = target_waypoint[0] - float(base_pos[0])
        dy = target_waypoint[1] - float(base_pos[1])
        distance = max(1e-6, math.hypot(dx, dy))
        velocity_frame = "world_xy"
        requested_speed = _number(action.get("approach_speed_mps") or action.get("max_speed_mps"))
        if requested_speed is not None:
            speed = min(
                SAFETY_LIMITS["max_forward_velocity_mps"],
                max(0.0, requested_speed),
                max(0.0, distance * 2.5),
            )
        else:
            speed = min(SAFETY_LIMITS["max_forward_velocity_mps"], max(0.18, distance * 2.5))
        vx = speed * dx / distance
        vy = speed * dy / distance
        target_heading = math.atan2(dy, dx)
        diff = math.atan2(math.sin(target_heading - yaw), math.cos(target_heading - yaw))
        yaw_rate = max(
            -SAFETY_LIMITS["max_yaw_rate_rad_s"],
            min(SAFETY_LIMITS["max_yaw_rate_rad_s"], diff * 1.2),
        )
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
    if velocity_frame == "world_xy":
        controller_vx, controller_vy = _world_xy_velocity_to_body_frame(vx, vy, yaw)
    else:
        controller_vx, controller_vy = vx, vy
    return (
        {
            "action_type": action_type,
            "vx_mps": round(float(vx), 6),
            "vy_mps": round(float(vy), 6),
            "velocity_frame": velocity_frame,
            "controller_vx_mps": round(float(controller_vx), 6),
            "controller_vy_mps": round(float(controller_vy), 6),
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


def _apply_egocentric_upper_body_observation_pose(
    *,
    model: Any,
    mujoco_module: Any,
    data: Any,
    generated_at: str,
) -> dict[str, Any]:
    applied: list[dict[str, Any]] = []
    missing: list[str] = []
    for joint_name, target_rad in EGOCENTRIC_UPPER_BODY_OBSERVATION_POSE.items():
        joint_id = _joint_id_by_name(mujoco_module, model, joint_name)
        if joint_id < 0:
            missing.append(joint_name)
            continue
        qpos_addr = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_addr] = float(target_rad)
        try:
            qvel_addr = int(model.jnt_dofadr[joint_id])
            data.qvel[qvel_addr] = 0.0
        except Exception:
            qvel_addr = None
        applied.append(
            {
                "joint_name": joint_name,
                "qpos_addr": qpos_addr,
                "qvel_addr": qvel_addr,
                "target_rad": float(target_rad),
            }
        )
    return {
        "schema_version": "egocentric_upper_body_observation_pose.v1",
        "generated_at": generated_at,
        "status": "completed" if applied and not missing else "partial" if applied else "blocked",
        "pose_id": "g1_hands_forward_egocentric_observation_pose",
        "applied_joint_count": len(applied),
        "applied_joints": applied,
        "missing_joint_names": missing,
        "hands_or_end_effectors_expected_in_egocentric_torso_view": bool(applied),
        "hand_end_effector_policy_used": False,
        "pose_role": "camera_observation_framing_support",
        "blockers": [f"missing_observation_pose_joint:{name}" for name in missing],
        "claim_boundary": {
            "upper_body_pose_is_support_framing_not_hand_policy_execution": True,
            "hand_end_effector_policy_used": False,
            "dexterous_manipulation_policy_proven": False,
            "physical_robot_sensor_proof": False,
        },
    }


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
        self.raw_policy_action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.upper_hold_targets: list[float] = []
        self.update_count = 0
        self.policy_action_clipped_count = 0
        self.max_raw_policy_action_abs = 0.0
        self.max_applied_policy_action_abs = 0.0

    def reset(self, data: Any) -> None:
        import numpy as np

        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.raw_policy_action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        if UNITREE_RL_GYM_RESET_TO_POLICY_DEFAULT_LEG_POSE:
            for qpos_addr, qvel_addr, default_angle in zip(
                self.leg_qpos_addrs,
                self.leg_qvel_addrs,
                self.default_angles,
            ):
                data.qpos[qpos_addr] = float(default_angle)
                data.qvel[qvel_addr] = 0.0
        self.upper_hold_targets = [
            float(data.qpos[qpos_addr]) for qpos_addr in self.upper_hold_qpos_addrs
        ]
        self.update_count = 0
        self.policy_action_clipped_count = 0
        self.max_raw_policy_action_abs = 0.0
        self.max_applied_policy_action_abs = 0.0
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
            dq = np.array(
                [float(data.qvel[addr]) for addr in self.leg_qvel_addrs], dtype=np.float32
            )
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
            qj = np.array(
                [float(data.qpos[addr]) for addr in self.leg_qpos_addrs], dtype=np.float32
            )
            dqj = np.array(
                [float(data.qvel[addr]) for addr in self.leg_qvel_addrs], dtype=np.float32
            )
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
                raw_action = (
                    self.policy(obs_tensor).detach().cpu().numpy().squeeze().astype(np.float32)
                )
            clip_abs = None
            action_clipped = False
            if self.actuator_output_mode == "position_target":
                clip_abs = _unitree_rl_gym_position_target_action_clip_abs()
                self.action = np.clip(raw_action, -clip_abs, clip_abs).astype(np.float32)
                action_clipped = bool(np.any(np.abs(raw_action - self.action) > 1e-6))
            else:
                self.action = raw_action
            self.raw_policy_action = raw_action.astype(np.float32)
            if action_clipped:
                self.policy_action_clipped_count += 1
            raw_abs = (
                float(np.max(np.abs(self.raw_policy_action)))
                if self.raw_policy_action.size
                else 0.0
            )
            applied_abs = float(np.max(np.abs(self.action))) if self.action.size else 0.0
            self.max_raw_policy_action_abs = max(self.max_raw_policy_action_abs, raw_abs)
            self.max_applied_policy_action_abs = max(
                self.max_applied_policy_action_abs,
                applied_abs,
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
                "raw_policy_action": [round(float(value), 6) for value in self.raw_policy_action],
                "policy_action_clipped": action_clipped,
                "policy_action_clip_abs": clip_abs,
                "actuator_output_mode": self.actuator_output_mode,
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
        "reset_to_policy_default_leg_pose": UNITREE_RL_GYM_RESET_TO_POLICY_DEFAULT_LEG_POSE,
        "balanced_walking_controller_proven": False,
        "blockers": list(blockers),
        "claim_boundary": {
            "same_scene_controller_loaded_is_not_generated_world_rank_fidelity": True,
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
        gain_values = [
            float(model.actuator_gainprm[actuator_id][0]) for actuator_id in leg_actuator_ids
        ]
        bias_values = [
            float(model.actuator_biasprm[actuator_id][1]) for actuator_id in leg_actuator_ids
        ]
        if all(
            abs(bias + gain) < max(1.0, abs(gain) * 0.05) and gain > 10.0
            for gain, bias in zip(gain_values, bias_values)
        ):
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
            "position_target_action_clip_abs": (
                _unitree_rl_gym_position_target_action_clip_abs()
                if actuator_output_mode == "position_target"
                else None
            ),
            "position_target_action_clip_env": (
                UNITREE_RL_GYM_POSITION_TARGET_ACTION_CLIP_ENV
                if actuator_output_mode == "position_target"
                else None
            ),
            "position_target_action_clip_default_abs": (
                DEFAULT_UNITREE_RL_GYM_POSITION_TARGET_ACTION_CLIP_ABS
                if actuator_output_mode == "position_target"
                else None
            ),
            "position_target_action_clip_reason": (
                "Blueprint task scenes use the MuJoCo Menagerie G1 position actuators; "
                "the Unitree RL Gym policy was trained/deployed against a torque-controlled "
                "G1 XML, so same-scene position-target adaptation conservatively clips "
                "policy outputs before applying actuator targets."
                if actuator_output_mode == "position_target"
                else None
            ),
            "control_decimation": int(config["control_decimation"]),
            "simulation_dt": float(config["simulation_dt"]),
            "policy_num_obs": int(config["num_obs"]),
            "policy_num_actions": int(config["num_actions"]),
        },
    )


def _camera_for(
    mujoco_module: Any, camera_id: str, root_position: Sequence[float], yaw: float
) -> Any:
    camera = mujoco_module.MjvCamera()
    camera.type = mujoco_module.mjtCamera.mjCAMERA_FREE
    forward_x = math.cos(float(yaw))
    forward_y = math.sin(float(yaw))
    root_x = float(root_position[0])
    root_y = float(root_position[1])
    root_z = float(root_position[2])
    camera.lookat[:] = [root_x, root_y, root_z + 0.55]
    if camera_id == "overhead":
        camera.distance = 4.8
        camera.azimuth = 0.0
        camera.elevation = -89.0
    elif camera_id == "robot_follow":
        camera.distance = 2.0
        camera.azimuth = math.degrees(yaw) + 180.0
        camera.elevation = -14.0
    elif camera_id in {"head_pov", "robot_pov"}:
        eye_z = root_z + 1.23
        look_distance = 1.15
        camera.lookat[:] = [
            root_x + forward_x * look_distance,
            root_y + forward_y * look_distance,
            eye_z,
        ]
        camera.distance = look_distance
        camera.azimuth = math.degrees(yaw) + 180.0
        camera.elevation = 0.0
    elif camera_id == "torso_pov":
        eye_z = root_z + 0.92
        look_distance = 1.05
        camera.lookat[:] = [
            root_x + forward_x * look_distance,
            root_y + forward_y * look_distance,
            eye_z - 0.03,
        ]
        camera.distance = look_distance
        camera.azimuth = math.degrees(yaw) + 180.0
        camera.elevation = -2.0
    else:
        camera.distance = 3.2
        camera.azimuth = 220.0
        camera.elevation = -18.0
    return camera


def _has_fixed_camera(mujoco_module: Any, model: Any, camera_id: str) -> bool:
    camera_name = FIXED_G1_CAMERA_NAMES.get(camera_id)
    if not camera_name:
        return False
    try:
        return (
            int(mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_CAMERA, camera_name))
            >= 0
        )
    except Exception:
        return False


def _camera_for_render(
    mujoco_module: Any,
    model: Any,
    camera_id: str,
    root_position: Sequence[float],
    yaw: float,
) -> Any:
    if _has_fixed_camera(mujoco_module, model, camera_id):
        return FIXED_G1_CAMERA_NAMES[camera_id]
    return _camera_for(mujoco_module, camera_id, root_position, yaw)


def _camera_contract(camera_id: str, *, fixed_camera_used: bool) -> dict[str, Any]:
    egocentric = camera_id in EGOCENTRIC_VIDEO_CAMERAS
    return {
        "camera_id": camera_id,
        "camera_mount": (
            "g1_head_fixed_mjcf_camera"
            if camera_id in {"head_pov", "robot_pov"}
            else "g1_torso_fixed_mjcf_camera"
            if camera_id == "torso_pov"
            else "external_review_camera"
        ),
        "fixed_mujoco_camera_used": bool(fixed_camera_used),
        "fixed_mujoco_camera_name": FIXED_G1_CAMERA_NAMES.get(camera_id),
        "egocentric_sensor_view": bool(egocentric),
        "first_person_policy_observation_candidate": bool(egocentric),
        "hands_or_end_effectors_expected_in_view": bool(egocentric),
        "hands_or_end_effectors_expected_due_to_observation_pose": bool(egocentric),
        "fallback_free_camera_used": bool(egocentric and not fixed_camera_used),
        "camera_truth_boundary": {
            "simulated_camera_view": True,
            "physical_robot_sensor_proof": False,
            "third_person_overview_is_diagnostic_not_policy_observation": not egocentric,
        },
    }


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


def _capture_policy_visual_observation(
    *,
    job_dir: Path,
    renderer: Any,
    image_module: Any,
    mujoco_module: Any,
    model: Any,
    data: Any,
    run: Mapping[str, Any],
    step: int,
    camera_id: str,
    root_position: Sequence[float],
    yaw: float,
    root_qpos: int,
) -> dict[str, Any]:
    if renderer is None or image_module is None:
        return {
            "schema_version": "policy_visual_observation.v1",
            "available": False,
            "episode_id": run.get("episode_id"),
            "scenario_eval_run_id": run.get("scenario_eval_run_id"),
            "task_id": run.get("task_id"),
            "spawn_id": run.get("spawn_id"),
            "step": step,
            "camera_id": camera_id,
            "camera_frame_path": None,
            "blockers": ["policy_observation_renderer_unavailable"],
            "claim_boundary": {
                "visual_observation_required_for_real_vla_policy": True,
                "missing_visual_observation_blocks_openvla_or_cosmos_policy_proof": True,
            },
        }
    try:
        fixed_camera_used = _has_fixed_camera(mujoco_module, model, camera_id)
        camera = _camera_for_render(mujoco_module, model, camera_id, root_position, yaw)
        renderer.update_scene(data, camera=camera)
        frame = renderer.render()
        frame_dir = job_dir / "policy_observation_frames" / str(run.get("episode_id")) / camera_id
        ensure_dir(frame_dir)
        frame_path = frame_dir / f"step_{int(step):06d}.jpg"
        image_module.fromarray(frame).save(frame_path, quality=85)
        depth_available = False
        depth_frame_path: str | None = None
        depth_encoding: str | None = None
        depth_min_m: float | None = None
        depth_max_m: float | None = None
        depth_is_render_pass = False
        depth_blockers: list[str] = []
        if hasattr(renderer, "enable_depth_rendering"):
            try:
                import numpy as np

                depth_path = frame_dir / f"step_{int(step):06d}_depth.npy"
                depth_mode_enabled = False
                try:
                    # The renderer is shared with review-video capture; always restore RGB mode so
                    # later renderer.render() calls keep returning RGB frames.
                    renderer.enable_depth_rendering(model, True)
                    depth_mode_enabled = True
                    renderer.update_scene(data, camera=camera)
                    depth = renderer.render()
                finally:
                    if depth_mode_enabled:
                        try:
                            renderer.enable_depth_rendering(model, False)
                        except Exception:  # noqa: BLE001
                            pass
                depth_arr = np.asarray(depth, dtype=np.float32)
                np.save(depth_path, depth_arr)
                finite = depth_arr[np.isfinite(depth_arr)]
                depth_min_m = float(finite.min()) if finite.size else None
                depth_max_m = float(finite.max()) if finite.size else None
                depth_available = True
                depth_frame_path = str(depth_path)
                depth_encoding = "npy_float32_meters"
                depth_is_render_pass = True
            except Exception:
                depth_blockers.append("policy_observation_depth_pass_unavailable")
        else:
            depth_blockers.append("policy_observation_depth_pass_unavailable")
        unitree_g1_sonic_state, unitree_g1_sonic_state_metadata = (
            _build_unitree_g1_sonic_state_from_mujoco(
                mujoco_module=mujoco_module,
                model=model,
                data=data,
                root_qpos=root_qpos,
            )
        )
        return {
            "schema_version": "policy_visual_observation.v1",
            "available": True,
            "episode_id": run.get("episode_id"),
            "scenario_eval_run_id": run.get("scenario_eval_run_id"),
            "task_id": run.get("task_id"),
            "spawn_id": run.get("spawn_id"),
            "step": int(step),
            "camera_id": camera_id,
            "camera_frame_path": str(frame_path),
            "image_width": int(frame.shape[1]),
            "image_height": int(frame.shape[0]),
            "image_encoding": "jpeg",
            "depth_available": bool(depth_available),
            "depth_frame_path": depth_frame_path,
            "depth_encoding": depth_encoding,
            "depth_min_m": depth_min_m,
            "depth_max_m": depth_max_m,
            "depth_is_render_pass": bool(depth_is_render_pass),
            "fixed_mujoco_camera_used": bool(fixed_camera_used),
            "fixed_mujoco_camera_name": FIXED_G1_CAMERA_NAMES.get(camera_id),
            "egocentric_sensor_view": camera_id in EGOCENTRIC_VIDEO_CAMERAS,
            "first_person_policy_observation_candidate": camera_id in EGOCENTRIC_VIDEO_CAMERAS,
            "physical_robot_sensor_proof": False,
            "unitree_g1_sonic_state": unitree_g1_sonic_state,
            "unitree_g1_sonic_state_source": "simulated_mujoco_qpos_joint_groups",
            "unitree_g1_sonic_state_metadata": unitree_g1_sonic_state_metadata,
            "blockers": depth_blockers,
            "claim_boundary": {
                "simulated_camera_view": True,
                "physical_robot_sensor_proof": False,
                "visual_observation_path_can_feed_vla_policy_endpoint": True,
                "unitree_g1_sonic_state_is_simulated_mujoco_state": True,
                "unitree_g1_sonic_state_is_physical_proprioception": False,
                "mujoco_render_pass_depth_is_simulator_geometry_not_physical_sensor": True,
            },
        }
    except Exception as exc:
        return {
            "schema_version": "policy_visual_observation.v1",
            "available": False,
            "episode_id": run.get("episode_id"),
            "scenario_eval_run_id": run.get("scenario_eval_run_id"),
            "task_id": run.get("task_id"),
            "spawn_id": run.get("spawn_id"),
            "step": step,
            "camera_id": camera_id,
            "camera_frame_path": None,
            "blockers": ["policy_observation_frame_capture_failed"],
            "error_type": type(exc).__name__,
            "error": str(exc)[:300],
            "claim_boundary": {
                "visual_observation_required_for_real_vla_policy": True,
                "missing_visual_observation_blocks_openvla_or_cosmos_policy_proof": True,
            },
        }


def _segmentation_claim_boundary() -> dict[str, bool]:
    return {
        "simulated_segmentation_view": True,
        "physical_robot_sensor_proof": False,
        "mujoco_segmentation_is_diagnostic_not_default_policy_input": True,
        "segmentation_is_mujoco_evidence_not_isaac_evidence": True,
    }


def _segmentation_unavailable_observation(
    *,
    run: Mapping[str, Any],
    step: int,
    camera_id: str,
    blockers: Sequence[str],
    error_type: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": "policy_segmentation_observation.v1",
        "available": False,
        "episode_id": run.get("episode_id"),
        "scenario_eval_run_id": run.get("scenario_eval_run_id"),
        "task_id": run.get("task_id"),
        "spawn_id": run.get("spawn_id"),
        "step": int(step),
        "camera_id": camera_id,
        "segmentation_backend": "mujoco_renderer_native",
        "segmentation_mask_path": None,
        "instances": [],
        "instance_count": 0,
        "blockers": list(blockers),
        "claim_boundary": _segmentation_claim_boundary(),
    }
    if error_type:
        row["error_type"] = error_type
    if error:
        row["error"] = error[:300]
    return row


def _capture_segmentation_observation(
    *,
    job_dir: Path,
    renderer: Any,
    image_module: Any,
    mujoco_module: Any,
    model: Any,
    data: Any,
    run: Mapping[str, Any],
    step: int,
    camera_id: str,
    root_position: Sequence[float],
    yaw: float,
    contact_metadata: dict[int, dict[str, Any]] | None,
) -> dict[str, Any]:
    if renderer is None:
        return _segmentation_unavailable_observation(
            run=run,
            step=step,
            camera_id=camera_id,
            blockers=["policy_observation_renderer_unavailable"],
        )
    if not hasattr(renderer, "enable_segmentation_rendering"):
        return _segmentation_unavailable_observation(
            run=run,
            step=step,
            camera_id=camera_id,
            blockers=["policy_segmentation_unsupported_renderer"],
        )
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependency guard.
        return _segmentation_unavailable_observation(
            run=run,
            step=step,
            camera_id=camera_id,
            blockers=["policy_segmentation_numpy_unavailable"],
            error_type=type(exc).__name__,
            error=str(exc),
        )
    try:
        fixed_camera_used = _has_fixed_camera(mujoco_module, model, camera_id)
        camera = _camera_for_render(mujoco_module, model, camera_id, root_position, yaw)
        segmentation_enabled = False
        try:
            renderer.enable_segmentation_rendering()
            segmentation_enabled = True
            renderer.update_scene(data, camera=camera)
            segmentation = renderer.render()
        finally:
            if segmentation_enabled and hasattr(renderer, "disable_segmentation_rendering"):
                renderer.disable_segmentation_rendering()
        seg = np.asarray(segmentation)
        if seg.ndim != 3 or seg.shape[2] < 2:
            return _segmentation_unavailable_observation(
                run=run,
                step=step,
                camera_id=camera_id,
                blockers=["policy_segmentation_invalid_render_shape"],
            )
        object_ids = seg[..., 0].astype(np.int64, copy=False)
        object_types = seg[..., 1].astype(np.int64, copy=False)
        geom_type = int(mujoco_module.mjtObj.mjOBJ_GEOM)
        geom_mask = object_types == geom_type
        instances: list[dict[str, Any]] = []
        for geom_id in sorted(int(value) for value in np.unique(object_ids[geom_mask])):
            if geom_id < 0:
                continue
            pixel_count = int(np.count_nonzero(geom_mask & (object_ids == geom_id)))
            if pixel_count <= 0:
                continue
            metadata = _contact_metadata_for_geom(
                model=model,
                mujoco_module=mujoco_module,
                contact_metadata=contact_metadata,
                geom_id=geom_id,
            )
            instances.append(
                {
                    "geom_id": geom_id,
                    "geom_name": metadata.get("geom_name"),
                    "body_id": metadata.get("body_id"),
                    "body_name": metadata.get("body_name"),
                    "pixel_count": pixel_count,
                }
            )
        mask_path_text: str | None = None
        mask_blockers: list[str] = []
        if image_module is not None:
            try:
                mask_dir = (
                    job_dir
                    / "policy_segmentation_frames"
                    / str(run.get("episode_id"))
                    / camera_id
                )
                ensure_dir(mask_dir)
                mask_path = mask_dir / f"step_{int(step):06d}.png"
                image_module.fromarray(object_ids.astype(np.uint16)).save(mask_path)
                mask_path_text = str(mask_path)
            except Exception as exc:  # noqa: BLE001
                mask_blockers.append(f"policy_segmentation_mask_save_failed:{type(exc).__name__}")
        return {
            "schema_version": "policy_segmentation_observation.v1",
            "available": True,
            "episode_id": run.get("episode_id"),
            "scenario_eval_run_id": run.get("scenario_eval_run_id"),
            "task_id": run.get("task_id"),
            "spawn_id": run.get("spawn_id"),
            "step": int(step),
            "camera_id": camera_id,
            "image_width": int(seg.shape[1]),
            "image_height": int(seg.shape[0]),
            "fixed_mujoco_camera_used": bool(fixed_camera_used),
            "fixed_mujoco_camera_name": FIXED_G1_CAMERA_NAMES.get(camera_id),
            "segmentation_backend": "mujoco_renderer_native",
            "segmentation_mask_path": mask_path_text,
            "instances": instances,
            "instance_count": len(instances),
            "blockers": mask_blockers,
            "claim_boundary": _segmentation_claim_boundary(),
        }
    except Exception as exc:
        return _segmentation_unavailable_observation(
            run=run,
            step=step,
            camera_id=camera_id,
            blockers=["policy_segmentation_capture_failed"],
            error_type=type(exc).__name__,
            error=str(exc),
        )


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
    slow_motion = bool(fixed_fps_forced and playback_scale is not None and playback_scale > 1.2)
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
            "fixed_fps_lower_than_mujoco_step_rate_for_captured_frames" if slow_motion else None
        ),
    }


def _review_video_sampling_contract(
    *,
    fps: int,
    timestep: float,
    video_frame_stride_steps: int,
    render_frame_count: int,
    extend_terminal_frame_for_review: bool,
) -> dict[str, Any]:
    stride = max(1, int(video_frame_stride_steps))
    expected_sim_time_fps = _video_output_fps(
        requested_fps=0,
        timestep=timestep,
        stride_steps=stride,
    )
    captures_every_step = stride == 1
    fixed_fps = int(fps) > 0
    default_stride = int(render_frame_count) <= 0 and stride == DEFAULT_VIDEO_FRAME_STRIDE_STEPS
    nominal_realtime_review_mp4 = bool(default_stride and int(fps) == DEFAULT_REVIEW_VIDEO_FPS)
    if nominal_realtime_review_mp4:
        sampling_mode = "nominal_realtime_stride_review"
    elif captures_every_step and not fixed_fps:
        sampling_mode = "every_sim_step_sim_time_review"
    elif captures_every_step and fixed_fps:
        sampling_mode = "every_sim_step_fixed_fps_debug_slow_motion"
    elif int(render_frame_count) > 0:
        sampling_mode = "fixed_sample_count_review"
    else:
        sampling_mode = "custom_stride_review"
    return {
        "schema_version": "review_video_sampling_contract.v1",
        "sampling_mode": sampling_mode,
        "sample_every_n_sim_steps": stride,
        "captures_every_mujoco_step": captures_every_step,
        "captures_bounded_stride_frames": not captures_every_step,
        "mujoco_timestep_s": round(float(timestep), 9),
        "sim_seconds_per_rendered_frame": round(float(timestep) * stride, 9),
        "expected_sim_time_fps_for_stride": expected_sim_time_fps,
        "requested_or_default_fps": int(fps),
        "default_review_video_fps": DEFAULT_REVIEW_VIDEO_FPS,
        "nominal_realtime_review_mp4": nominal_realtime_review_mp4,
        "recommended_for_matrix_runs": nominal_realtime_review_mp4,
        "every_frame_at_fixed_60fps_is_debug_slow_motion": bool(captures_every_step and fixed_fps),
        "why_not_every_frame_by_default": (
            "MuJoCo steps at 0.002s by default; encoding every step at fixed 60fps "
            "turns simulator time into slow-motion review video. The default samples "
            "every 8 sim steps and encodes at 60fps, which is close to real-time."
        ),
        "terminal_failure_frame_hold_enabled": bool(extend_terminal_frame_for_review),
        "review_video_stops_at_terminal_failure_by_default": not bool(
            extend_terminal_frame_for_review
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
    if (
        rejected_action_count
        and run.get("spawn_id") == "blocked_or_occluded"
        and task_id == "inspect_target"
    ):
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
        success = route_safety and any(
            action in {"inspect_look", "look"} for action in action_types
        )
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
    if not navigation_success and task_id in {
        "approach_target",
        "route_around_obstruction",
        "stop_at_goal_and_report",
    }:
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
    policy_lane: str = "auto",
    unitree_lerobot_mode: str = "probe",
    allow_policy_action_model_command_run: bool = False,
    wam_loop_step_count: int = DEFAULT_WAM_LOOP_STEP_COUNT,
    wam_generation_timeout_seconds: float | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or utc_now_iso()
    effective_wam_generation_timeout_seconds = (
        float(wam_generation_timeout_seconds)
        if wam_generation_timeout_seconds and float(wam_generation_timeout_seconds) > 0.0
        else _float_env(WAM_GENERATION_TIMEOUT_ENV, DEFAULT_WAM_GENERATION_TIMEOUT_SECONDS)
    )
    if controller_backend not in CONTROLLER_BACKENDS:
        raise ValueError(f"controller_backend must be one of {', '.join(CONTROLLER_BACKENDS)}")
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
        job_dir / "policy_endpoint_boundary_manifest.json",
        build_policy_endpoint_boundary_manifest(
            generated_at=generated_at,
            endpoint_discovery=endpoint_discovery,
            selected_runtime=selected_runtime,
            fixture_policy_used=selected_runtime is None,
            policy_execution_manifest_path=job_dir / "policy_execution_manifest.json",
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
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "official_policy_execution_proven": False,
            "isaac_runtime_proven_by_this_lane": False,
        },
    )
    unitree_lerobot_config = UnitreeLeRobotPolicyRuntimeConfig.from_env(
        job_dir=job_dir,
        mode=unitree_lerobot_mode,
        timeout_seconds=min(float(endpoint_timeout_seconds), 30.0),
    )
    unitree_lerobot_runtime_summary = run_unitree_lerobot_g1_policy_eval(
        job_dir=job_dir,
        config=unitree_lerobot_config,
        generated_at=generated_at,
    )
    unitree_groot_n17_sonic_runtime_summary = run_unitree_groot_n17_sonic_policy_runtime(
        job_dir=job_dir,
        generated_at=generated_at,
    )
    unitree_stack_installation_audit = build_unitree_policy_stack_installation_audit(
        job_dir=job_dir,
        generated_at=generated_at,
        config=unitree_lerobot_config,
    )
    write_json(
        job_dir / "unitree_policy_stack_installation_audit.json",
        unitree_stack_installation_audit,
    )
    write_json(
        job_dir / "unitree_policy_provider_registry_probe.json",
        build_policy_provider_registry_probe(
            job_dir=job_dir,
            generated_at=generated_at,
            config=unitree_lerobot_config,
        ),
    )
    write_json(job_dir / "wam_vla_observation_packet_schema.json", _observation_schema())
    write_json(job_dir / "wam_vla_action_schema.json", _action_schema())
    policy_action_model_command_execution = run_policy_action_model_command_contract(
        job_dir=job_dir,
        generated_at=generated_at,
        allow_policy_action_model_command_run=allow_policy_action_model_command_run,
        timeout_seconds=_policy_action_model_command_timeout_seconds(endpoint_timeout_seconds),
    )
    if policy_action_model_command_execution.get("selected_candidate_id") == GROOT_POLICY_ID:
        unitree_groot_n17_sonic_audit = probe_unitree_groot_n17_sonic_runtime(
            generated_at=generated_at
        )
        unitree_groot_n17_sonic_truth = build_unitree_groot_n17_sonic_runtime_truth_boundary(
            audit=unitree_groot_n17_sonic_audit,
            policy_action_command_result=policy_action_model_command_execution,
        )
        unitree_groot_n17_sonic_runtime_summary = {
            **unitree_groot_n17_sonic_runtime_summary,
            "unitree_groot_n17_sonic_policy_action_command_ran": bool(
                policy_action_model_command_execution.get(
                    "unitree_groot_n17_sonic_policy_action_command_ran"
                )
            ),
            "unitree_policy_action_command_ran": bool(
                policy_action_model_command_execution.get("unitree_policy_action_command_ran")
            ),
            "unitree_specific_manipulation_candidate_ran": bool(
                policy_action_model_command_execution.get(
                    "unitree_specific_manipulation_candidate_ran"
                )
            ),
            "openvla_policy_action_command_ran": False,
            "truth_boundary_path": str(
                job_dir / "unitree_groot_n17_sonic_policy_runtime_truth_boundary.json"
            ),
        }
        write_json(
            job_dir / "unitree_groot_n17_sonic_installation_audit.json",
            unitree_groot_n17_sonic_audit,
        )
        write_json(
            job_dir / "unitree_groot_n17_sonic_policy_runtime_truth_boundary.json",
            unitree_groot_n17_sonic_truth,
        )
        write_json(
            job_dir / "unitree_groot_n17_sonic_policy_runtime_summary.json",
            unitree_groot_n17_sonic_runtime_summary,
        )

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
            "policy_lane": policy_lane,
            "unitree_lerobot_runtime_status": unitree_lerobot_runtime_summary.get("status"),
            "unitree_lerobot_runtime_configured": unitree_lerobot_runtime_summary.get(
                "unitree_lerobot_runtime_configured"
            ),
            "unitree_lerobot_sim_inference_proven": unitree_lerobot_runtime_summary.get(
                "unitree_lerobot_sim_inference_proven"
            ),
            "unitree_groot_n17_sonic_runtime_status": (
                unitree_groot_n17_sonic_runtime_summary.get("status")
            ),
            "unitree_groot_n17_sonic_policy_configured": (
                unitree_groot_n17_sonic_runtime_summary.get(
                    "unitree_groot_n17_sonic_policy_configured"
                )
            ),
            "blockers": ["mujoco_import_failed"],
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
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
    g1_source_xml, g1_mjcf_selection = _select_g1_policy_observation_mjcf(g1_root)
    g1_abs_xml = job_dir / "generated_mujoco" / "unitree_g1_absolute_meshes.xml"
    _write_g1_xml_with_absolute_meshes(g1_source_xml, g1_abs_xml)
    g1_fixed_camera_manifest = _add_g1_fixed_egocentric_cameras(g1_abs_xml)
    write_json(job_dir / "g1_fixed_egocentric_camera_manifest.json", g1_fixed_camera_manifest)
    asset_manifest = {
        "schema_version": "unitree_g1_mujoco_asset_source_manifest.v1",
        "generated_at": generated_at,
        **_asset_source_manifest(g1_root),
        "resolved_g1_xml": str(g1_source_xml),
        "g1_mjcf_selection": g1_mjcf_selection,
        "hands_capable_g1_mjcf_selected": bool(
            g1_mjcf_selection.get("hands_capable_g1_mjcf_selected")
        ),
        "generated_absolute_mesh_xml": str(g1_abs_xml),
        "g1_fixed_egocentric_camera_manifest": str(
            job_dir / "g1_fixed_egocentric_camera_manifest.json"
        ),
        "g1_fixed_egocentric_cameras_mounted": bool(
            g1_fixed_camera_manifest.get("status") == "completed"
        ),
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
    manipulation_policy_discovery = discover_unitree_manipulation_policy(generated_at=generated_at)
    write_json(
        job_dir / "unitree_g1_manipulation_policy_discovery.json",
        manipulation_policy_discovery,
    )
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
    timestep: float | None = None
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
                    "expected_sim_duration_seconds": (
                        round(float(steps_per_episode) * timestep, 6)
                        if timestep is not None
                        else None
                    ),
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
    object_initial_pose = [0.36, -0.65, 0.24, 1.0, 0.0, 0.0, 0.0]
    timestep = float(model.opt.timestep)
    contact_metadata = _build_contact_metadata(model, mujoco)
    unitree_groot_n17_sonic_sim2sim_execution: dict[str, Any] = {
        "schema_version": "unitree_groot_n17_sonic_sim2sim_execution.v1",
        "generated_at": generated_at,
        "status": "skipped",
        "policy_id": GROOT_POLICY_ID,
        "selected_candidate_id": policy_action_model_command_execution.get("selected_candidate_id"),
        "unitree_groot_n17_sonic_sim2sim_command_ran": False,
        "unitree_groot_n17_sonic_action_chunk_consumed": False,
        "blockers": ["skipped_unitree_groot_n17_sonic_policy_action_command_not_completed"],
        "claim_boundary": {
            "simulator_only": True,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
            "accepted_anchor_manipulation_success_proven": False,
        },
    }
    if policy_action_model_command_execution.get(
        "selected_candidate_id"
    ) == GROOT_POLICY_ID and policy_action_model_command_execution.get(
        "unitree_groot_n17_sonic_policy_action_command_ran"
    ):
        phase("unitree_groot_n17_sonic_sim2sim_started", "running")
        unitree_groot_n17_sonic_sim2sim_execution = run_unitree_groot_n17_sonic_sim2sim(
            job_dir=job_dir,
            policy_action_output=job_dir / "policy_action_model_command_output.json",
            scene_xml=Path(str(scene_manifest["scene_xml"])),
            steps=min(40, max(1, int(steps_per_episode))),
            action_hold_steps=10,
            render_video=bool(render),
            generated_at=generated_at,
        )
        phase(
            "unitree_groot_n17_sonic_sim2sim_completed",
            str(unitree_groot_n17_sonic_sim2sim_execution.get("status") or "blocked"),
            sim2sim_command_ran=bool(
                unitree_groot_n17_sonic_sim2sim_execution.get(
                    "unitree_groot_n17_sonic_sim2sim_command_ran"
                )
            ),
        )
        unitree_groot_n17_sonic_audit = probe_unitree_groot_n17_sonic_runtime(
            generated_at=generated_at
        )
        unitree_groot_n17_sonic_truth = build_unitree_groot_n17_sonic_runtime_truth_boundary(
            audit=unitree_groot_n17_sonic_audit,
            policy_action_command_result=policy_action_model_command_execution,
            sim2sim_result=unitree_groot_n17_sonic_sim2sim_execution,
        )
        unitree_groot_n17_sonic_runtime_summary = {
            **unitree_groot_n17_sonic_runtime_summary,
            "ready_for_sim2sim": unitree_groot_n17_sonic_audit["ready_for_sim2sim"],
            "unitree_groot_n17_sonic_policy_action_command_ran": bool(
                policy_action_model_command_execution.get(
                    "unitree_groot_n17_sonic_policy_action_command_ran"
                )
            ),
            "unitree_policy_action_command_ran": bool(
                policy_action_model_command_execution.get("unitree_policy_action_command_ran")
            ),
            "unitree_specific_manipulation_candidate_ran": bool(
                policy_action_model_command_execution.get(
                    "unitree_specific_manipulation_candidate_ran"
                )
            ),
            "openvla_policy_action_command_ran": False,
            "sim2sim_command_ran": bool(
                unitree_groot_n17_sonic_sim2sim_execution.get(
                    "unitree_groot_n17_sonic_sim2sim_command_ran"
                )
            ),
            "unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim": bool(
                unitree_groot_n17_sonic_sim2sim_execution.get(
                    "unitree_groot_n17_sonic_action_chunk_consumed"
                )
            ),
            "sim2sim_execution_path": str(
                job_dir / "unitree_groot_n17_sonic_sim2sim_execution.json"
            ),
        }
        write_json(
            job_dir / "unitree_groot_n17_sonic_installation_audit.json",
            unitree_groot_n17_sonic_audit,
        )
        write_json(
            job_dir / "unitree_groot_n17_sonic_policy_runtime_truth_boundary.json",
            unitree_groot_n17_sonic_truth,
        )
        write_json(
            job_dir / "unitree_groot_n17_sonic_policy_runtime_summary.json",
            unitree_groot_n17_sonic_runtime_summary,
        )
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
    if controller_backend == "unitree_rl_gym" and not same_scene_controller_ready:
        blocked_summary = {
            "schema_version": LANE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "job_id": job_id,
            "job_dir": str(job_dir),
            "mujoco_runtime_available": True,
            "unitree_g1_mujoco_model_source": asset_manifest["unitree_g1_mujoco_model_source"],
            "unitree_g1_mujoco_model_path": str(g1_source_xml),
            "unitree_g1_hands_capable_mjcf_selected": bool(
                g1_mjcf_selection.get("hands_capable_g1_mjcf_selected")
            ),
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
                "policy_endpoint_boundary_manifest": str(
                    job_dir / "policy_endpoint_boundary_manifest.json"
                ),
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
    policy_visual_observation_rows: list[dict[str, Any]] = []
    policy_segmentation_observation_rows: list[dict[str, Any]] = []
    g1_projected_skeleton_rows: list[dict[str, Any]] = []
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
    review_video_sampling = _review_video_sampling_contract(
        fps=fps,
        timestep=timestep,
        video_frame_stride_steps=video_frame_stride_steps,
        render_frame_count=render_frame_count,
        extend_terminal_frame_for_review=extend_terminal_frame_for_review,
    )
    egocentric_observation_pose_manifest: dict[str, Any] = {
        "schema_version": "egocentric_upper_body_observation_pose.v1",
        "generated_at": generated_at,
        "status": "not_run",
        "pose_id": "g1_hands_forward_egocentric_observation_pose",
        "hand_end_effector_policy_used": False,
        "claim_boundary": {
            "upper_body_pose_is_support_framing_not_hand_policy_execution": True,
            "hand_end_effector_policy_used": False,
            "dexterous_manipulation_policy_proven": False,
            "physical_robot_sensor_proof": False,
        },
    }

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
            data.qpos[root_qpos + 3 : root_qpos + 7] = _yaw_quat(
                float(run.get("spawn_yaw_rad") or 0.0)
            )
            if object_qpos is not None:
                data.qpos[object_qpos : object_qpos + 7] = object_initial_pose
                data.qvel[
                    int(model.jnt_dofadr[object_joint_id]) : int(model.jnt_dofadr[object_joint_id])
                    + 6
                ] = 0.0
            egocentric_observation_pose_manifest = _apply_egocentric_upper_body_observation_pose(
                model=model,
                mujoco_module=mujoco,
                data=data,
                generated_at=generated_at,
            )
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
            render_episode_video = (
                renderer is not None
                and image_module is not None
                and (
                    rendered_video_episode_cap is None
                    or episode_index <= rendered_video_episode_cap
                )
            )
            if render_episode_video:
                frame_step_list, video_render_mode, video_render_stride_steps = (
                    _episode_frame_steps(
                        steps_per_episode=steps_per_episode,
                        render_frame_count=render_frame_count,
                        video_frame_stride_steps=video_frame_stride_steps,
                    )
                )
                frame_steps = set(frame_step_list)
                frame_index_by_step = {
                    frame_step: index for index, frame_step in enumerate(frame_step_list)
                }
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
                    current_root_pos = [
                        float(value) for value in data.qpos[root_qpos : root_qpos + 3]
                    ]
                    current_root_quat = [
                        float(value) for value in data.qpos[root_qpos + 3 : root_qpos + 7]
                    ]
                    current_yaw = _yaw_from_quat(current_root_quat)
                    visual_observation = _capture_policy_visual_observation(
                        job_dir=job_dir,
                        renderer=renderer,
                        image_module=image_module,
                        mujoco_module=mujoco,
                        model=model,
                        data=data,
                        run=run,
                        step=step,
                        camera_id="head_pov",
                        root_position=current_root_pos,
                        yaw=current_yaw,
                        root_qpos=root_qpos,
                    )
                    segmentation_observation = _capture_segmentation_observation(
                        job_dir=job_dir,
                        renderer=renderer,
                        image_module=image_module,
                        mujoco_module=mujoco,
                        model=model,
                        data=data,
                        run=run,
                        step=step,
                        camera_id="head_pov",
                        root_position=current_root_pos,
                        yaw=current_yaw,
                        contact_metadata=contact_metadata,
                    )
                    visual_observation["segmentation_observation"] = (
                        segmentation_observation
                    )
                    policy_segmentation_observation_rows.append(segmentation_observation)
                    policy_visual_observation_rows.append(visual_observation)
                    try:
                        g1_projected_skeleton_rows.append(
                            _build_g1_projected_skeleton_trace_row(
                                mujoco_module=mujoco,
                                model=model,
                                data=data,
                                run=run,
                                step=step,
                                visual_observation=visual_observation,
                            )
                        )
                    except Exception as exc:
                        g1_projected_skeleton_rows.append(
                            {
                                "schema_version": G1_PROJECTED_SKELETON_SCHEMA_ID,
                                "status": "blocked",
                                "episode_id": run.get("episode_id"),
                                "scenario_eval_run_id": run.get("scenario_eval_run_id"),
                                "task_id": run.get("task_id"),
                                "spawn_id": run.get("spawn_id"),
                                "step": int(step),
                                "camera_id": visual_observation.get("camera_id"),
                                "camera_frame_path": visual_observation.get("camera_frame_path"),
                                "landmarks": [],
                                "segments": [
                                    {"from": start, "to": end}
                                    for start, end in G1_UPPER_BODY_SKELETON_SEGMENTS
                                ],
                                "available_landmark_count": 0,
                                "projected_landmark_count": 0,
                                "blockers": ["g1_projected_skeleton_trace_row_build_failed"],
                                "error": str(exc),
                                "claim_boundary": {
                                    "uses_unitree_g1_mujoco_body_transforms": False,
                                    "simulated_g1_kinematic_skeleton_available": False,
                                    "not_hand_drawn_stick_figure": True,
                                    "not_physical_robot_sensor_proof": True,
                                    "not_dexterous_hand_policy_execution": True,
                                },
                            }
                        )
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
                        visual_observation=visual_observation,
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
                            "visual_observation_available": bool(
                                visual_observation.get("available")
                            ),
                            "visual_observation_camera_id": visual_observation.get("camera_id"),
                            "visual_observation_frame_path": visual_observation.get(
                                "camera_frame_path"
                            ),
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
                remaining_contact_trace_rows = max(0, contact_trace_row_limit - len(contact_rows))
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
                        "root_linear_velocity_xyz_mps": [
                            round(value, 6) for value in root_velocity
                        ],
                        "active_action": dict(active_action),
                        "controller_backend": selected_controller_backend,
                        "freejoint_proxy_used": selected_controller_backend == "freejoint_proxy",
                        "official_unitree_controller_used": selected_controller_backend
                        == "unitree_rl_gym",
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
                        camera = _camera_for_render(
                            mujoco,
                            model,
                            camera_id,
                            root_position,
                            _yaw_from_quat(root_quat),
                        )
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
                    video_path = (
                        job_dir / "mujoco_videos" / f"{run['episode_id']}__{output_name}.mp4"
                    )
                    poster_path = (
                        job_dir / "mujoco_posters" / f"{run['episode_id']}__{output_name}.png"
                    )
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
                                    **_camera_contract(
                                        output_name,
                                        fixed_camera_used=_has_fixed_camera(
                                            mujoco, model, output_name
                                        ),
                                    ),
                                    "rendered_frame_count": len(frame_files),
                                    "physics_rendered_frame_count": physics_frame_count,
                                    "requested_frame_count": len(frame_step_list),
                                    "missing_terminal_frame_count": missing_terminal_frame_count,
                                    "terminal_frame_hold_count": terminal_frame_hold_count,
                                    "terminal_frame_extended_for_review": terminal_frame_hold_count
                                    > 0,
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
                                    "review_video_sampling_mode": review_video_sampling[
                                        "sampling_mode"
                                    ],
                                    "nominal_realtime_review_mp4": bool(
                                        review_video_sampling["nominal_realtime_review_mp4"]
                                    ),
                                    "captures_every_mujoco_step": bool(
                                        review_video_sampling["captures_every_mujoco_step"]
                                    ),
                                    "why_not_every_frame_by_default": review_video_sampling[
                                        "why_not_every_frame_by_default"
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
                            probe = (
                                _ffprobe_video(video_path)
                                if video.get("status") == "complete"
                                else {
                                    "path": str(video_path),
                                    "status": "not_checked",
                                    "reason": video.get("reason") or "video_not_complete",
                                }
                            )
                            video_review_validation = (
                                validate_generated_mp4_for_review(video_path)
                                if video.get("status") == "complete"
                                else {
                                    "schema_version": "wam_generated_video_review_validation.v1",
                                    "status": "blocked",
                                    "path": str(video_path),
                                    "exists": video_path.is_file(),
                                    "size_bytes": video_path.stat().st_size
                                    if video_path.is_file()
                                    else 0,
                                    "blockers": ["generated_video_not_complete"],
                                }
                            )
                            video.update(
                                {
                                    "generated_video_review_validation": video_review_validation,
                                    "decode_valid_for_review": video_review_validation.get("status")
                                    == "completed",
                                }
                            )
                            media[output_name] = {
                                "video": video,
                                "ffprobe": probe,
                                "poster": str(poster_path),
                            }
                            video_identity = {
                                "episode_id": run["episode_id"],
                                "scenario_eval_run_id": run["scenario_eval_run_id"],
                                "task_id": run["task_id"],
                                "spawn_id": run["spawn_id"],
                                "camera": output_name,
                            }
                            video_rows.append({**video_identity, **video})
                            poster_rows.append(
                                {
                                    **video_identity,
                                    "path": str(poster_path),
                                    "size_bytes": poster_path.stat().st_size
                                    if poster_path.is_file()
                                    else 0,
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
                bool(video.get("review_video_stops_at_terminal_failure")) for video in media_videos
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
                "policy_id": REFERENCE_FIXTURE_POLICY_ID
                if fixture_policy_used
                else "endpoint_policy",
                "policy_runtime_source": "reference_fixture_policy"
                if fixture_policy_used
                else selected_runtime.get("runtime"),
                "fixture_policy_used": fixture_policy_used,
                "endpoint_policy_used": bool(
                    not fixture_policy_used and endpoint_policy_valid_actions
                ),
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
                    "generated_world_rank_fidelity_result_proven": False,
                    "generated_world_policy_evaluation_scope_proven": False,
                    "official_policy_execution_proven": selected_controller_backend
                    == "unitree_rl_gym",
                    "same_scene_unitree_controller_backend_used": selected_controller_backend
                    == "unitree_rl_gym",
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
                "generated_world_rank_fidelity_result_proven": False,
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
    same_scene_policy_action_clipped_update_count = sum(
        1 for row in same_scene_controller_rows if row.get("policy_action_clipped")
    )
    same_scene_max_raw_policy_action_abs = max(
        (
            max(abs(float(value)) for value in row.get("raw_policy_action", []) or [0.0])
            for row in same_scene_controller_rows
        ),
        default=0.0,
    )
    same_scene_max_applied_policy_action_abs = max(
        (
            max(abs(float(value)) for value in row.get("action", []) or [0.0])
            for row in same_scene_controller_rows
        ),
        default=0.0,
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
            "same_scene_policy_action_clipped_update_count": same_scene_policy_action_clipped_update_count,
            "same_scene_max_raw_policy_action_abs": round(
                float(same_scene_max_raw_policy_action_abs), 6
            ),
            "same_scene_max_applied_policy_action_abs": round(
                float(same_scene_max_applied_policy_action_abs), 6
            ),
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
    _write_jsonl(
        job_dir / "policy_visual_observation_trace.jsonl",
        policy_visual_observation_rows,
    )
    g1_projected_skeleton_trace_path = job_dir / "g1_projected_skeleton_trace.jsonl"
    _write_jsonl(g1_projected_skeleton_trace_path, g1_projected_skeleton_rows)
    g1_projected_skeleton_manifest = _g1_projected_skeleton_manifest(
        generated_at=generated_at,
        rows=g1_projected_skeleton_rows,
        output_path=g1_projected_skeleton_trace_path,
    )
    write_json(
        job_dir / "g1_projected_skeleton_manifest.json",
        g1_projected_skeleton_manifest,
    )
    write_json(
        job_dir / "policy_visual_observation_manifest.json",
        {
            "schema_version": "policy_visual_observation_manifest.v1",
            "generated_at": generated_at,
            "status": "completed"
            if any(row.get("available") for row in policy_visual_observation_rows)
            else "blocked_no_policy_visual_observations",
            "camera_id": "head_pov",
            "observation_count": len(policy_visual_observation_rows),
            "available_observation_count": sum(
                1 for row in policy_visual_observation_rows if row.get("available")
            ),
            "first_available_frame_path": next(
                (
                    str(row.get("camera_frame_path"))
                    for row in policy_visual_observation_rows
                    if row.get("available") and row.get("camera_frame_path")
                ),
                None,
            ),
            "blockers": []
            if any(row.get("available") for row in policy_visual_observation_rows)
            else ["policy_observation_renderer_unavailable"],
            "claim_boundary": {
                "simulated_camera_observation_available_for_vla_policy": any(
                    row.get("available") for row in policy_visual_observation_rows
                ),
                "physical_robot_sensor_proof": False,
                "visual_observation_does_not_prove_vla_model_execution": True,
                "g1_projected_skeleton_trace_available_for_wam_conditioning": bool(
                    g1_projected_skeleton_manifest.get("status") == "completed"
                ),
                "g1_projected_skeleton_trace_is_simulated_state_not_physical_proprioception": True,
            },
        },
    )
    write_json(
        job_dir / "policy_segmentation_observations.json",
        {
            "schema_version": "policy_segmentation_observation_manifest.v1",
            "generated_at": generated_at,
            "status": "completed"
            if any(row.get("available") for row in policy_segmentation_observation_rows)
            else "blocked_no_policy_segmentation_observations",
            "camera_id": "head_pov",
            "observation_count": len(policy_segmentation_observation_rows),
            "available_observation_count": sum(
                1 for row in policy_segmentation_observation_rows if row.get("available")
            ),
            "segmentation_backend": "mujoco_renderer_native",
            "observations": policy_segmentation_observation_rows,
            "blockers": []
            if any(row.get("available") for row in policy_segmentation_observation_rows)
            else sorted(
                {
                    blocker
                    for row in policy_segmentation_observation_rows
                    for blocker in list(row.get("blockers") or [])
                }
            )
            or ["policy_segmentation_unsupported_renderer"],
            "claim_boundary": {
                "simulated_segmentation_view": any(
                    row.get("available") for row in policy_segmentation_observation_rows
                ),
                "physical_robot_sensor_proof": False,
                "mujoco_segmentation_is_diagnostic_not_default_policy_input": True,
                "segmentation_is_mujoco_evidence_not_isaac_evidence": True,
            },
        },
    )
    phase(
        "policy_action_model_command_reused_for_wam_loop",
        str(policy_action_model_command_execution.get("status") or "blocked"),
        policy_action_model_command_ran=bool(
            policy_action_model_command_execution.get("policy_action_model_command_ran")
        ),
    )
    if (
        policy_action_model_command_execution.get("selected_candidate_id") == GROOT_POLICY_ID
        and policy_action_model_command_execution.get(
            "unitree_groot_n17_sonic_policy_action_command_ran"
        )
        and not unitree_groot_n17_sonic_sim2sim_execution.get(
            "unitree_groot_n17_sonic_sim2sim_command_ran"
        )
    ):
        phase("unitree_groot_n17_sonic_sim2sim_started", "running")
        unitree_groot_n17_sonic_sim2sim_execution = run_unitree_groot_n17_sonic_sim2sim(
            job_dir=job_dir,
            policy_action_output=job_dir / "policy_action_model_command_output.json",
            scene_xml=Path(str(scene_manifest["scene_xml"])),
            steps=min(40, max(1, int(steps_per_episode))),
            action_hold_steps=10,
            render_video=bool(render),
            generated_at=generated_at,
        )
        phase(
            "unitree_groot_n17_sonic_sim2sim_completed",
            str(unitree_groot_n17_sonic_sim2sim_execution.get("status") or "blocked"),
            sim2sim_command_ran=bool(
                unitree_groot_n17_sonic_sim2sim_execution.get(
                    "unitree_groot_n17_sonic_sim2sim_command_ran"
                )
            ),
        )
        unitree_groot_n17_sonic_audit = probe_unitree_groot_n17_sonic_runtime(
            generated_at=generated_at
        )
        unitree_groot_n17_sonic_truth = build_unitree_groot_n17_sonic_runtime_truth_boundary(
            audit=unitree_groot_n17_sonic_audit,
            policy_action_command_result=policy_action_model_command_execution,
            sim2sim_result=unitree_groot_n17_sonic_sim2sim_execution,
        )
        unitree_groot_n17_sonic_runtime_summary = {
            **unitree_groot_n17_sonic_runtime_summary,
            "ready_for_sim2sim": unitree_groot_n17_sonic_audit["ready_for_sim2sim"],
            "unitree_groot_n17_sonic_policy_action_command_ran": bool(
                policy_action_model_command_execution.get(
                    "unitree_groot_n17_sonic_policy_action_command_ran"
                )
            ),
            "unitree_policy_action_command_ran": bool(
                policy_action_model_command_execution.get("unitree_policy_action_command_ran")
            ),
            "unitree_specific_manipulation_candidate_ran": bool(
                policy_action_model_command_execution.get(
                    "unitree_specific_manipulation_candidate_ran"
                )
            ),
            "openvla_policy_action_command_ran": False,
            "sim2sim_command_ran": bool(
                unitree_groot_n17_sonic_sim2sim_execution.get(
                    "unitree_groot_n17_sonic_sim2sim_command_ran"
                )
            ),
            "unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim": bool(
                unitree_groot_n17_sonic_sim2sim_execution.get(
                    "unitree_groot_n17_sonic_action_chunk_consumed"
                )
            ),
            "sim2sim_execution_path": str(
                job_dir / "unitree_groot_n17_sonic_sim2sim_execution.json"
            ),
        }
        write_json(
            job_dir / "unitree_groot_n17_sonic_installation_audit.json",
            unitree_groot_n17_sonic_audit,
        )
        write_json(
            job_dir / "unitree_groot_n17_sonic_policy_runtime_truth_boundary.json",
            unitree_groot_n17_sonic_truth,
        )
        write_json(
            job_dir / "unitree_groot_n17_sonic_policy_runtime_summary.json",
            unitree_groot_n17_sonic_runtime_summary,
        )
    robot_policy_wam_closed_loop_attempt = run_robot_policy_wam_closed_loop_attempt(
        job_dir=job_dir,
        generated_at=generated_at,
        policy_action_model_command_execution=policy_action_model_command_execution,
        loop_step_count=wam_loop_step_count,
        timeout_seconds=effective_wam_generation_timeout_seconds,
    )
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
            "action_validity_rate": round(
                (len(action_rows) - len(rejected_actions)) / len(action_rows), 6
            )
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
                "generated_world_rank_fidelity_result_proven": False,
                "generated_world_policy_evaluation_scope_proven": False,
            },
        },
    )
    backend_failure_labels: list[dict[str, Any]] = []
    if not policy_action_model_command_execution.get("policy_action_model_command_ran"):
        backend_failure_labels.append(
            {
                "label_id": f"failure_{len(failure_labels) + len(backend_failure_labels) + 1:04d}",
                "attempt_id": "policy_action_model_command",
                "failure_label_ids": [
                    "blocked_policy_action_model_command_not_run",
                    *[
                        str(blocker)
                        for blocker in policy_action_model_command_execution.get("blockers", [])
                    ],
                ],
                "status": "blocked",
                "review_required": True,
                "claim_boundary": {
                    "backend_blocker_label_not_task_success_score": True,
                    "generated_world_rank_fidelity_result_proven": False,
                    "generated_world_policy_evaluation_scope_proven": False,
                },
            }
        )
    if robot_policy_wam_closed_loop_attempt.get("status") == "blocked":
        backend_failure_labels.append(
            {
                "label_id": f"failure_{len(failure_labels) + len(backend_failure_labels) + 1:04d}",
                "attempt_id": "robot_policy_wam_closed_loop",
                "failure_label_ids": [
                    "blocked_robot_policy_wam_closed_loop",
                    *[
                        str(blocker)
                        for blocker in robot_policy_wam_closed_loop_attempt.get("blockers", [])
                    ],
                ],
                "status": "blocked",
                "review_required": True,
                "claim_boundary": {
                    "backend_blocker_label_not_task_success_score": True,
                    "wam_evaluator_is_not_robot_policy": True,
                    "generated_world_rank_fidelity_result_proven": False,
                    "generated_world_policy_evaluation_scope_proven": False,
                },
            }
        )
    all_failure_labels = failure_labels + backend_failure_labels
    write_json(
        job_dir / "failure_labels.json",
        {
            "schema_version": "mujoco_g1_wam_vla_failure_labels.v1",
            "generated_at": generated_at,
            "status": "review_required" if all_failure_labels else "no_failures_labeled",
            "label_count": len(all_failure_labels),
            "backend_blocker_label_count": len(backend_failure_labels),
            "labels": all_failure_labels,
            "failed_or_blocked_attempt_count": len(all_failure_labels),
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
            "scenario_eval_run_coverage_complete": required_ids == covered_ids
            and bool(required_ids),
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
        "unsafe_obstacle_contact_count": contact_aggregate_counts["obstacle_contact_count"],
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
            "unitree_g1_manipulation_policy_discovery": str(
                job_dir / "unitree_g1_manipulation_policy_discovery.json"
            ),
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
        if attempt["metrics"].get("object_displacement_m", 0)
        >= SAFETY_LIMITS["object_displacement_success_m"]
        and attempt["metrics"].get("object_contact_count", 0) > 0
    ]
    endpoint_policy_responses = [
        _mapping(row.get("raw_policy_response_redacted"))
        for row in action_rows
        if row.get("source") == "endpoint_policy"
    ]
    endpoint_policy_inner_responses = [
        _mapping(_mapping(row.get("endpoint_metadata")).get("raw_response_redacted")) or row
        for row in endpoint_policy_responses
    ]
    unitree_groot_n17_sonic_sim2sim_command_ran = bool(
        unitree_groot_n17_sonic_sim2sim_execution.get("unitree_groot_n17_sonic_sim2sim_command_ran")
    )
    unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim = bool(
        unitree_groot_n17_sonic_sim2sim_execution.get(
            "unitree_groot_n17_sonic_action_chunk_consumed"
        )
    )
    unitree_groot_n17_sonic_policy_chunk_integrated_contact_rollout = bool(
        unitree_groot_n17_sonic_sim2sim_execution.get(
            "policy_action_chunk_integrated_into_contact_rollout"
        )
        or unitree_groot_n17_sonic_sim2sim_execution.get(
            "policy_chunk_integrated_contact_rollout_success"
        )
    )
    unitree_endpoint_policy_summary = _unitree_endpoint_policy_response_summary(
        endpoint_policy_inner_responses
    )
    unitree_endpoint_hand_policy_output_observed = unitree_endpoint_policy_summary[
        "unitree_endpoint_hand_policy_output_observed"
    ]
    policy_action_provider_output_replay_used = _policy_action_provider_output_replay_used(
        policy_action_model_command_execution=policy_action_model_command_execution,
        robot_policy_wam_closed_loop_attempt=robot_policy_wam_closed_loop_attempt,
    )
    unitree_endpoint_provider_output_replay_used = bool(
        unitree_endpoint_policy_summary["unitree_endpoint_provider_output_replay_used"]
        or policy_action_provider_output_replay_used
    )
    unitree_endpoint_action_chunk_used = unitree_endpoint_policy_summary[
        "unitree_endpoint_action_chunk_used"
    ]
    unitree_endpoint_fresh_policy_action_command_ran = unitree_endpoint_policy_summary[
        "unitree_endpoint_fresh_policy_action_command_ran"
    ]
    unitree_endpoint_hand_policy_used = unitree_endpoint_policy_summary[
        "unitree_endpoint_hand_policy_used"
    ]
    g1_robot_policy_selected_family = unitree_endpoint_policy_summary[
        "g1_robot_policy_selected_family"
    ]
    unitree_hand_manipulation_policy_scope = unitree_endpoint_policy_summary[
        "unitree_hand_manipulation_policy_scope"
    ]
    unitree_action_chunk_consumed_by_any_sim_path = bool(
        unitree_endpoint_action_chunk_used
        or unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim
    )
    write_json(
        job_dir / "manipulation_success_evaluator_results.json",
        {
            "schema_version": "manipulation_success_evaluator_results.v1",
            "generated_at": generated_at,
            "status": "completed" if contact_attempts else "blocked",
            "attempt_count": len(contact_attempts),
            "success_count": len(manipulation_successes),
            "policy_chunk_integrated_success_count": int(
                unitree_groot_n17_sonic_policy_chunk_integrated_contact_rollout
            ),
            "success_rate": round(len(manipulation_successes) / len(contact_attempts), 6)
            if contact_attempts
            else 0.0,
            "unitree_groot_n17_sonic_policy_chunk_integrated_contact_rollout": (
                unitree_groot_n17_sonic_policy_chunk_integrated_contact_rollout
            ),
            "unitree_groot_n17_sonic_object_robot_contact_count": int(
                unitree_groot_n17_sonic_sim2sim_execution.get("object_robot_contact_count") or 0
            ),
            "unitree_groot_n17_sonic_object_displacement_m": (
                unitree_groot_n17_sonic_sim2sim_execution.get("object_displacement_m")
            ),
            "unitree_groot_n17_sonic_object_horizontal_displacement_m": (
                unitree_groot_n17_sonic_sim2sim_execution.get("object_horizontal_displacement_m")
            ),
            "unitree_groot_n17_sonic_object_displacement_success_axis": (
                unitree_groot_n17_sonic_sim2sim_execution.get("object_displacement_success_axis")
            ),
            "unitree_groot_n17_sonic_object_displacement_without_robot_contact": bool(
                unitree_groot_n17_sonic_sim2sim_execution.get(
                    "object_displacement_without_robot_contact"
                )
            ),
            "unitree_groot_n17_sonic_contact_rollout_blockers": list(
                unitree_groot_n17_sonic_sim2sim_execution.get("contact_rollout_blockers") or []
            ),
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
    manipulation_validated = bool(
        manipulation_successes or unitree_groot_n17_sonic_policy_chunk_integrated_contact_rollout
    )
    manipulation_blockers: list[str] = []
    if not contact_attempts:
        manipulation_blockers.append("blocked_missing_contact_task_attempts")
    if contact_attempts and not manipulation_validated:
        manipulation_blockers.append("blocked_manipulation_contact_not_validated")
    if (
        not unitree_endpoint_hand_policy_used
        and not unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim
    ):
        manipulation_blockers.extend(
            [
                "blocked_dexterous_hand_policy_not_integrated",
                "blocked_real_vla_model_not_configured",
            ]
        )
    if unitree_endpoint_provider_output_replay_used:
        manipulation_blockers.append(
            "blocked_unitree_hand_policy_endpoint_used_provider_output_replay_not_fresh_per_observation"
        )
    if (
        unitree_endpoint_hand_policy_used or unitree_groot_n17_sonic_sim2sim_command_ran
    ) and not unitree_action_chunk_consumed_by_any_sim_path:
        manipulation_blockers.append("blocked_unitree_hand_policy_endpoint_missing_action_chunk")
    if (
        unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim
        and not unitree_endpoint_action_chunk_used
        and not unitree_groot_n17_sonic_policy_chunk_integrated_contact_rollout
    ):
        manipulation_blockers.append(
            "blocked_gr00t_sonic_chunk_not_integrated_into_contact_task_rollout"
        )
    write_json(
        job_dir / "manipulation_endpoint_task_report.json",
        {
            "schema_version": "manipulation_endpoint_task_report.v1",
            "generated_at": generated_at,
            "status": "completed" if contact_attempts else "blocked",
            "endpoint_action_path_required": True,
            "endpoint_action_path_used": bool(
                contact_attempts
                and any(attempt.get("endpoint_policy_used") for attempt in contact_attempts)
            ),
            "manipulation_endpoint_path_used": bool(
                contact_attempts
                and any(attempt.get("endpoint_policy_used") for attempt in contact_attempts)
            ),
            "fixture_policy_used": bool(
                contact_attempts
                and any(attempt.get("fixture_policy_used") for attempt in contact_attempts)
            ),
            "attempt_count": len(contact_attempts),
            "object_contact_count": sum(
                int(attempt["metrics"].get("object_contact_count", 0))
                for attempt in contact_attempts
            ),
            "max_object_displacement_m": max(
                [
                    float(attempt["metrics"].get("object_displacement_m", 0.0))
                    for attempt in contact_attempts
                ]
                or [0.0]
            ),
            "unsafe_collision_count": sum(
                int(attempt["metrics"].get("unsafe_collision_contact_count", 0))
                for attempt in contact_attempts
            ),
            "fall_count": sum(
                int(attempt["metrics"].get("fall_count", 0)) for attempt in contact_attempts
            ),
            "successful_contact_attempt_count": len(manipulation_successes),
            "hand_end_effector_policy_used": False,
            "g1_robot_policy_selection_contract": "unitree_native_policy_required_for_g1_claims",
            "g1_robot_policy_selected_family": g1_robot_policy_selected_family,
            "unitree_hand_manipulation_policy_scope": unitree_hand_manipulation_policy_scope,
            "openvla_selected_as_g1_robot_policy": False,
            "wam_rollout_selected_as_g1_robot_policy": False,
            "unitree_endpoint_hand_policy_output_observed": (
                unitree_endpoint_hand_policy_output_observed
            ),
            "unitree_endpoint_hand_policy_used": unitree_endpoint_hand_policy_used,
            "unitree_endpoint_provider_output_replay_used": (
                unitree_endpoint_provider_output_replay_used
            ),
            "unitree_endpoint_fresh_policy_action_command_ran": (
                unitree_endpoint_fresh_policy_action_command_ran
            ),
            "unitree_endpoint_action_chunk_used": unitree_endpoint_action_chunk_used,
            "unitree_groot_n17_sonic_sim2sim_command_ran": (
                unitree_groot_n17_sonic_sim2sim_command_ran
            ),
            "unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim": (
                unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim
            ),
            "unitree_groot_n17_sonic_policy_chunk_integrated_contact_rollout": (
                unitree_groot_n17_sonic_policy_chunk_integrated_contact_rollout
            ),
            "unitree_groot_n17_sonic_object_robot_contact_count": int(
                unitree_groot_n17_sonic_sim2sim_execution.get("object_robot_contact_count") or 0
            ),
            "unitree_groot_n17_sonic_object_horizontal_displacement_m": (
                unitree_groot_n17_sonic_sim2sim_execution.get("object_horizontal_displacement_m")
            ),
            "unitree_groot_n17_sonic_contact_rollout_blockers": list(
                unitree_groot_n17_sonic_sim2sim_execution.get("contact_rollout_blockers") or []
            ),
            "unitree_action_chunk_consumed_by_any_sim_path": (
                unitree_action_chunk_consumed_by_any_sim_path
            ),
            "base_proxy_contact_path_used": bool(
                contact_attempts and selected_controller_backend == "freejoint_proxy"
            ),
            "lower_body_controller_contact_path_used": bool(
                contact_attempts and selected_controller_backend == "unitree_rl_gym"
            ),
            "task_requires_dexterous_hand_policy_for_vla_manipulation_claim": True,
            "unitree_g1_manipulation_policy_discovery": str(
                job_dir / "unitree_g1_manipulation_policy_discovery.json"
            ),
            "claim_boundary": {
                "simulator_only": True,
                "contact_success_only_via_freejoint_proxy": bool(
                    manipulation_validated and selected_controller_backend == "freejoint_proxy"
                ),
                "contact_success_only_via_unitree_lower_body_controller": bool(
                    manipulation_validated and selected_controller_backend == "unitree_rl_gym"
                ),
                "dexterous_vla_manipulation_proven": False,
                "unitree_hand_policy_output_observed": (
                    unitree_endpoint_hand_policy_output_observed
                ),
                "unitree_hand_policy_endpoint_used": unitree_endpoint_hand_policy_used,
                "unitree_hand_policy_provider_output_replay_used": (
                    unitree_endpoint_provider_output_replay_used
                ),
                "unitree_endpoint_action_chunk_used": unitree_endpoint_action_chunk_used,
                "unitree_groot_n17_sonic_sim2sim_command_ran": (
                    unitree_groot_n17_sonic_sim2sim_command_ran
                ),
                "unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim": (
                    unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim
                ),
                "real_vla_model_ran": unitree_endpoint_fresh_policy_action_command_ran,
                "generated_world_rank_fidelity_result_proven": False,
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
            "unitree_endpoint_hand_policy_output_observed": (
                unitree_endpoint_hand_policy_output_observed
            ),
            "unitree_endpoint_hand_policy_used": unitree_endpoint_hand_policy_used,
            "unitree_endpoint_provider_output_replay_used": (
                unitree_endpoint_provider_output_replay_used
            ),
            "unitree_endpoint_fresh_policy_action_command_ran": (
                unitree_endpoint_fresh_policy_action_command_ran
            ),
            "unitree_endpoint_action_chunk_used": unitree_endpoint_action_chunk_used,
            "unitree_groot_n17_sonic_sim2sim_command_ran": (
                unitree_groot_n17_sonic_sim2sim_command_ran
            ),
            "unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim": (
                unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim
            ),
            "unitree_action_chunk_consumed_by_any_sim_path": (
                unitree_action_chunk_consumed_by_any_sim_path
            ),
            "dexterous_hand_policy_proven": False,
            "vla_manipulation_policy_proven": False,
            "real_wam_vla_model_ran": unitree_endpoint_fresh_policy_action_command_ran,
            "unitree_g1_manipulation_policy_discovery": str(
                job_dir / "unitree_g1_manipulation_policy_discovery.json"
            ),
            "generated_world_rank_fidelity_result_proven": False,
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
            "generated_world_policy_evaluation_scope_proven": False,
        },
    )
    generated_video_review_validations = [
        dict(row["generated_video_review_validation"])
        for row in video_rows
        if isinstance(row.get("generated_video_review_validation"), Mapping)
    ]
    generated_rollout_review_rows = [
        {
            "rollout_id": (
                f"{row.get('episode_id')}__{row.get('camera')}"
                if row.get("episode_id") and row.get("camera")
                else f"rendered_rollout_{index:04d}"
            ),
            "generated_video_path": str(row.get("path") or ""),
            "model_family": "unitree_rl_gym"
            if selected_controller_backend == "unitree_rl_gym"
            else "mujoco_controller_trace",
            "episode_id": row.get("episode_id"),
            "scenario_eval_run_id": row.get("scenario_eval_run_id"),
            "task_id": row.get("task_id"),
            "spawn_id": row.get("spawn_id"),
            "camera": row.get("camera"),
        }
        for index, row in enumerate(video_rows, start=1)
        if row.get("path")
    ]
    generated_rollout_visual_smoke = visual_smoke_generated_rollouts_for_review(
        rollouts=generated_rollout_review_rows,
        output_dir=job_dir,
        generated_at=generated_at,
    )
    write_json(job_dir / "generated_rollout_visual_smoke.json", generated_rollout_visual_smoke)
    generated_rollout_visual_smoke_status = str(
        generated_rollout_visual_smoke.get("status") or "not_applicable_missing_rollouts"
    )
    generated_rollout_visually_useful = bool(
        generated_rollout_visual_smoke.get("claim_boundary", {}).get(
            "visual_rollout_useful_for_task_success_review"
        )
    )
    generated_videos_decode_valid_for_review = bool(video_rows) and all(
        row.get("status") == "completed" for row in generated_video_review_validations
    )
    write_json(
        job_dir / "egocentric_upper_body_observation_pose_manifest.json",
        egocentric_observation_pose_manifest,
    )
    video_status = {
        "schema_version": "video_generation_status.v1",
        "generated_at": generated_at,
        "status": "completed" if video_rows else "blocked",
        "video_count": len(video_rows),
        "poster_count": len(poster_rows),
        "generated_video_review_validation_count": len(generated_video_review_validations),
        "generated_videos_decode_valid_for_review": generated_videos_decode_valid_for_review,
        "generated_rollout_visual_smoke_status": generated_rollout_visual_smoke_status,
        "generated_rollout_visually_useful_for_success_review": generated_rollout_visually_useful,
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
            "egocentric_video_camera_ids": list(EGOCENTRIC_VIDEO_CAMERAS),
            "fixed_g1_camera_names": dict(FIXED_G1_CAMERA_NAMES),
            "egocentric_upper_body_observation_pose": egocentric_observation_pose_manifest,
            "scored_episode_count_can_exceed_rendered_video_episode_count": True,
            "fps": int(fps),
            "default_review_video_fps": DEFAULT_REVIEW_VIDEO_FPS,
            "default_review_video_contract": (
                "stride-8 frame sampling encoded at 60fps unless --fps 0 requests exact simulator-time playback"
            ),
            "fps_zero_means_realtime_from_mujoco_timestep": True,
            "review_video_sampling": review_video_sampling,
            "nominal_realtime_review_mp4": bool(
                review_video_sampling["nominal_realtime_review_mp4"]
            ),
            "review_video_sampling_mode": review_video_sampling["sampling_mode"],
            "why_not_every_frame_by_default": review_video_sampling[
                "why_not_every_frame_by_default"
            ],
            "short_review_video_reason": (
                "videos stop at the actual terminal physics failure frame unless terminal-frame hold is enabled"
            ),
            "videos_are_for_human_review_of_scored_episodes": True,
            "automated_success_source": "structured_mujoco_trace_metrics",
        },
        "videos": video_rows,
        "posters": poster_rows,
        "ffprobe": ffprobe_rows,
        "generated_video_review_validations": generated_video_review_validations,
        "generated_rollout_visual_smoke": str(job_dir / "generated_rollout_visual_smoke.json"),
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
                "nominal_realtime_review_mp4": bool(
                    review_video_sampling["nominal_realtime_review_mp4"]
                ),
                "review_video_sampling_mode": review_video_sampling["sampling_mode"],
                "review_video_sampling_contract": review_video_sampling,
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
                "scenario_eval_run_id": row.get("scenario_eval_run_id"),
                "task_id": row.get("task_id"),
                "spawn_id": row.get("spawn_id"),
                "camera": row.get("camera"),
                "camera_mount": row.get("camera_mount"),
                "fixed_mujoco_camera_used": bool(row.get("fixed_mujoco_camera_used")),
                "fixed_mujoco_camera_name": row.get("fixed_mujoco_camera_name"),
                "egocentric_sensor_view": bool(row.get("egocentric_sensor_view")),
                "first_person_policy_observation_candidate": bool(
                    row.get("first_person_policy_observation_candidate")
                ),
                "hands_or_end_effectors_expected_in_view": bool(
                    row.get("hands_or_end_effectors_expected_in_view")
                ),
                "hands_or_end_effectors_expected_due_to_observation_pose": bool(
                    row.get("hands_or_end_effectors_expected_due_to_observation_pose")
                ),
                "fallback_free_camera_used": bool(row.get("fallback_free_camera_used")),
                "camera_truth_boundary": row.get("camera_truth_boundary", {}),
                "path": video_path,
                "status": row.get("status"),
                "decode_valid_for_review": bool(row.get("decode_valid_for_review")),
                "generated_video_review_validation": row.get(
                    "generated_video_review_validation", {}
                ),
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
                "review_video_sampling_mode": row.get("review_video_sampling_mode"),
                "nominal_realtime_review_mp4": bool(row.get("nominal_realtime_review_mp4")),
                "captures_every_mujoco_step": bool(row.get("captures_every_mujoco_step")),
                "why_not_every_frame_by_default": row.get("why_not_every_frame_by_default"),
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
                "default_camera": DEFAULT_VIDEO_CAMERAS[0],
                "default_camera_role": "egocentric_robot_policy_observation_candidate",
                "selected_camera_ids": list(selected_video_cameras),
                "egocentric_video_camera_ids": list(EGOCENTRIC_VIDEO_CAMERAS),
                "diagnostic_video_camera_ids": list(DIAGNOSTIC_VIDEO_CAMERAS),
                "fixed_g1_camera_names": dict(FIXED_G1_CAMERA_NAMES),
                "egocentric_upper_body_observation_pose": egocentric_observation_pose_manifest,
                "third_person_overview_is_diagnostic_not_policy_observation": True,
                "hands_or_end_effectors_visible_requires_egocentric_observation_pose_or_hand_policy": True,
                "rendered_video_episode_limit": rendered_video_episode_cap,
                "steps_per_episode": int(steps_per_episode),
                "mujoco_timestep_s": round(float(timestep), 9),
                "configured_episode_sim_duration_s": round(
                    float(timestep) * max(1, int(steps_per_episode)),
                    9,
                ),
                "video_frame_stride_steps": int(video_frame_stride_steps),
                "every_sim_step_captured": int(video_frame_stride_steps) == 1,
                "terminal_failure_frame_hold_enabled": bool(extend_terminal_frame_for_review),
                "review_videos_stop_at_terminal_failure_by_default": not bool(
                    extend_terminal_frame_for_review
                ),
                "playback_fps": int(selected_review_playback_fps),
                "default_review_video_fps": DEFAULT_REVIEW_VIDEO_FPS,
                "default_review_video_contract": (
                    "stride-8 frame sampling encoded at 60fps unless --fps 0 requests exact simulator-time playback"
                ),
                "review_video_sampling": review_video_sampling,
                "nominal_realtime_review_mp4": bool(
                    review_video_sampling["nominal_realtime_review_mp4"]
                ),
                "review_video_sampling_mode": review_video_sampling["sampling_mode"],
                "why_not_every_frame_by_default": review_video_sampling[
                    "why_not_every_frame_by_default"
                ],
                "fps_zero_encodes_sim_time_playback": int(fps) == 0,
                "fixed_fps_with_every_step_can_create_slow_motion": True,
                "rendered_attempts_are_subset_of_scored_attempts": True,
                "default_target_review_video_count": "5_to_10_attempts",
                "scored_all_matrix_rows": len(attempts) == required_count and required_count > 0,
            },
            "selected_review_video_count": len(selected_review_videos),
            "generated_rollout_visual_smoke_status": generated_rollout_visual_smoke_status,
            "generated_rollout_visually_useful_for_success_review": generated_rollout_visually_useful,
            "selected_review_videos": selected_review_videos,
            "generated_video_review_validations": generated_video_review_validations,
            "generated_rollout_visual_smoke": str(job_dir / "generated_rollout_visual_smoke.json"),
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
    endpoint_invocation_count = sum(
        1 for row in endpoint_attempt_rows if row.get("endpoint_invoked")
    )
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
    unitree_lower_body_locomotion_policy_used = bool(same_scene_integrated)
    unitree_locomotion_policy_config_path = (
        same_scene_controller_manifest.get("config_path")
        if unitree_lower_body_locomotion_policy_used
        else None
    )
    unitree_locomotion_policy_checkpoint_path = (
        same_scene_controller_manifest.get("policy_path")
        if unitree_lower_body_locomotion_policy_used
        else None
    )
    unitree_hand_manipulation_policy_used = bool(
        unitree_endpoint_hand_policy_used or unitree_groot_n17_sonic_sim2sim_command_ran
    )
    unitree_hand_manipulation_policy_kind = (
        "unitree_unifolm_endpoint_action_command"
        if unitree_endpoint_hand_policy_used
        else GROOT_POLICY_ID
        if unitree_groot_n17_sonic_sim2sim_command_ran
        else None
    )
    generated_rollout_review_blockers = sorted(
        {
            str(blocker)
            for validation in generated_video_review_validations
            for blocker in validation.get("blockers", [])
        }
        | {str(blocker) for blocker in generated_rollout_visual_smoke.get("blockers", [])}
    )
    unitree_generated_rollout_review_completed = bool(
        unitree_lower_body_locomotion_policy_used
        and generated_videos_decode_valid_for_review
        and generated_rollout_visually_useful
    )
    unitree_generated_rollout_review_manifest = {
        "schema_version": "unitree_generated_rollout_review_manifest.v1",
        "generated_at": generated_at,
        "status": "completed"
        if unitree_generated_rollout_review_completed
        else "blocked"
        if unitree_lower_body_locomotion_policy_used
        else "not_applicable_no_unitree_policy_rollout",
        "model_family": "unitree_rl_gym",
        "unitree_lower_body_locomotion_policy_ran": unitree_lower_body_locomotion_policy_used,
        "unitree_locomotion_policy_kind": final_navigation_policy_kind
        if unitree_lower_body_locomotion_policy_used
        else None,
        "unitree_locomotion_policy_checkpoint_path": unitree_locomotion_policy_checkpoint_path,
        "unitree_locomotion_policy_config_path": unitree_locomotion_policy_config_path,
        "generated_video_review_validation_count": len(generated_video_review_validations),
        "generated_videos_decode_valid_for_review": generated_videos_decode_valid_for_review,
        "generated_rollout_visual_smoke_status": generated_rollout_visual_smoke_status,
        "generated_rollout_visually_useful_for_success_review": generated_rollout_visually_useful,
        "visual_smoke": generated_rollout_visual_smoke,
        "video_decode_validations": generated_video_review_validations,
        "blockers": generated_rollout_review_blockers
        if unitree_lower_body_locomotion_policy_used
        else [],
        "claim_boundary": {
            "unitree_lower_body_locomotion_policy_ran": unitree_lower_body_locomotion_policy_used,
            "generated_rollout_visually_useful_for_success_review": bool(
                unitree_generated_rollout_review_completed
            ),
            "visual_smoke_is_not_forward_inverse_consistency": True,
            "simulator_only_not_generated_world_rank_fidelity": True,
            "not_openvla_or_cosmos_execution": True,
            "not_dexterous_manipulation_policy_proof": True,
        },
    }
    write_json(
        job_dir / "unitree_generated_rollout_review_manifest.json",
        unitree_generated_rollout_review_manifest,
    )
    write_json(
        job_dir / "video_review_status.json",
        {
            "schema_version": "video_review_status.v1",
            "generated_at": generated_at,
            "status": unitree_generated_rollout_review_manifest["status"],
            "source_manifest": str(job_dir / "unitree_generated_rollout_review_manifest.json"),
            "selected_review_video_count": len(selected_review_videos),
            "generated_videos_decode_valid_for_review": generated_videos_decode_valid_for_review,
            "generated_rollout_visually_useful_for_success_review": (
                generated_rollout_visually_useful
            ),
            "blockers": list(unitree_generated_rollout_review_manifest.get("blockers", [])),
            "claim_boundary": {
                "video_review_is_not_task_success_proof": True,
                "generated_world_rank_fidelity_result_proven": False,
                "generated_world_policy_evaluation_scope_proven": False,
                "non_ranking_operational_claim_proven": False,
                "accepted_anchor_manipulation_success_proven": False,
            },
        },
    )
    final_success_policy_command_ran = bool(
        policy_action_model_command_execution.get("unitree_manipulation_policy_action_command_ran")
        or unitree_endpoint_fresh_policy_action_command_ran
    )
    final_success_policy_action_consumed_by_sim = bool(
        unitree_endpoint_action_chunk_used
        or unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim
    )
    final_success_policy_chunk_integrated_into_contact_rollout = bool(
        manipulation_validated
        and final_success_policy_command_ran
        and (
            unitree_endpoint_action_chunk_used
            or unitree_groot_n17_sonic_policy_chunk_integrated_contact_rollout
        )
    )
    final_success_unitree_endpoint_action_available = bool(
        final_success_policy_command_ran
        or unitree_endpoint_hand_policy_used
        or unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim
    )
    final_success_blockers = sorted(
        {
            *[str(blocker) for blocker in manipulation_blockers],
            *[str(blocker) for blocker in robot_policy_wam_closed_loop_attempt.get("blockers", [])],
        }
    )
    if not final_success_unitree_endpoint_action_available:
        final_success_blockers.append("blocked_unitree_manipulation_action_command_not_run")
    elif unitree_endpoint_provider_output_replay_used:
        final_success_blockers.append(
            "blocked_unitree_manipulation_action_was_provider_output_replay_not_fresh_per_observation"
        )
    if (
        final_success_policy_action_consumed_by_sim
        and not final_success_policy_chunk_integrated_into_contact_rollout
    ):
        final_success_blockers.append(
            "blocked_policy_action_chunk_not_integrated_into_successful_contact_rollout"
        )
    scene_task = _policy_action_scene_task(job_dir)
    final_success_question, scene_task_success_field = _final_success_question_for_scene_task(
        scene_task
    )
    final_success_judge = {
        "schema_version": "final_success_judge.v1",
        "generated_at": generated_at,
        "status": "completed" if final_success_policy_command_ran else "blocked",
        "final_question": final_success_question,
        "answer": "yes"
        if final_success_policy_chunk_integrated_into_contact_rollout
        else "not_proven",
        "scene_task_id": scene_task.get("task_id") or None,
        "scene_target_object_id": scene_task.get("target_object_id") or None,
        "scene_wam_policy_episode_packet_path": (
            scene_task.get("scene_wam_policy_episode_packet_path") or None
        ),
        "object_or_tote_correctly_placed": bool(
            final_success_policy_chunk_integrated_into_contact_rollout
        ),
        "success_proven": bool(final_success_policy_chunk_integrated_into_contact_rollout),
        "score": 1.0 if final_success_policy_chunk_integrated_into_contact_rollout else 0.0,
        "structured_contact_or_push_success": bool(manipulation_validated),
        "policy_action_chunk_consumed_by_sim": final_success_policy_action_consumed_by_sim,
        "policy_action_chunk_integrated_into_contact_rollout": (
            final_success_policy_chunk_integrated_into_contact_rollout
        ),
        "unitree_manipulation_policy_action_command_ran": final_success_policy_command_ran,
        "g1_robot_policy_selection_contract": "unitree_native_policy_required_for_g1_claims",
        "g1_robot_policy_selected_family": g1_robot_policy_selected_family,
        "unitree_hand_manipulation_policy_scope": unitree_hand_manipulation_policy_scope,
        "openvla_selected_as_g1_robot_policy": False,
        "wam_rollout_selected_as_g1_robot_policy": False,
        "unitree_endpoint_hand_policy_output_observed": (
            unitree_endpoint_hand_policy_output_observed
        ),
        "unitree_endpoint_hand_policy_used": unitree_endpoint_hand_policy_used,
        "unitree_endpoint_provider_output_replay_used": (
            unitree_endpoint_provider_output_replay_used
        ),
        "unitree_endpoint_action_chunk_used": unitree_endpoint_action_chunk_used,
        "unitree_groot_n17_sonic_sim2sim_command_ran": (
            unitree_groot_n17_sonic_sim2sim_command_ran
        ),
        "unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim": (
            unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim
        ),
        "unitree_groot_n17_sonic_policy_chunk_integrated_contact_rollout": (
            unitree_groot_n17_sonic_policy_chunk_integrated_contact_rollout
        ),
        "unitree_groot_n17_sonic_object_robot_contact_count": int(
            unitree_groot_n17_sonic_sim2sim_execution.get("object_robot_contact_count") or 0
        ),
        "unitree_groot_n17_sonic_object_horizontal_displacement_m": (
            unitree_groot_n17_sonic_sim2sim_execution.get("object_horizontal_displacement_m")
        ),
        "unitree_groot_n17_sonic_contact_rollout_blockers": list(
            unitree_groot_n17_sonic_sim2sim_execution.get("contact_rollout_blockers") or []
        ),
        "unitree_groot_n17_sonic_policy_action_command_ran": bool(
            policy_action_model_command_execution.get(
                "unitree_groot_n17_sonic_policy_action_command_ran"
            )
        ),
        "wam_evaluator_in_control_loop": bool(
            robot_policy_wam_closed_loop_attempt.get("wam_evaluator_in_control_loop")
        ),
        "requested_wam_loop_step_count": int(wam_loop_step_count),
        "policy_observes_wam_generated_next_observation": bool(
            robot_policy_wam_closed_loop_attempt.get(
                "policy_observes_wam_generated_next_observation"
            )
        ),
        "video_review_status": unitree_generated_rollout_review_manifest["status"],
        "vlm_success_judge_used": False,
        "human_success_judge_used": False,
        "structured_mujoco_trace_judge_used": True,
        "blockers": sorted(set(final_success_blockers)),
        "claim_boundary": {
            "simulator_only": True,
            "structured_contact_trace_is_not_accepted_anchor_success": True,
            "success_not_proven_without_unitree_manipulation_action_command": True,
            "success_not_proven_without_policy_chunk_integrated_contact_rollout": True,
            "provider_output_replay_is_not_fresh_per_observation_policy_execution": (
                unitree_endpoint_provider_output_replay_used
            ),
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
            "accepted_anchor_manipulation_success_proven": False,
        },
    }
    if scene_task_success_field:
        final_success_judge[scene_task_success_field] = bool(
            final_success_policy_chunk_integrated_into_contact_rollout
        )
    write_json(job_dir / "final_success_judge.json", final_success_judge)
    write_json(
        job_dir / "claim_boundary.json",
        {
            "schema_version": "closed_loop_claim_boundary.v1",
            "generated_at": generated_at,
            "simulator_only": True,
            "policy_lane": policy_lane,
            "locomotion_proof_is_separate_from_manipulation_proof": True,
            "groot_n17_sonic_is_candidate_not_proven_unless_action_command_runs": True,
            "unitree_manipulation_policy_action_command_ran": final_success_policy_command_ran,
            "g1_robot_policy_selection_contract": "unitree_native_policy_required_for_g1_claims",
            "g1_robot_policy_selected_family": g1_robot_policy_selected_family,
            "unitree_hand_manipulation_policy_scope": unitree_hand_manipulation_policy_scope,
            "openvla_selected_as_g1_robot_policy": False,
            "wam_rollout_selected_as_g1_robot_policy": False,
            "unitree_endpoint_hand_policy_output_observed": (
                unitree_endpoint_hand_policy_output_observed
            ),
            "unitree_endpoint_hand_policy_used": unitree_endpoint_hand_policy_used,
            "unitree_endpoint_provider_output_replay_used": (
                unitree_endpoint_provider_output_replay_used
            ),
            "unitree_endpoint_action_chunk_used": unitree_endpoint_action_chunk_used,
            "unitree_groot_n17_sonic_sim2sim_command_ran": (
                unitree_groot_n17_sonic_sim2sim_command_ran
            ),
            "unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim": (
                unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim
            ),
            "unitree_groot_n17_sonic_policy_chunk_integrated_contact_rollout": (
                unitree_groot_n17_sonic_policy_chunk_integrated_contact_rollout
            ),
            "unitree_groot_n17_sonic_object_robot_contact_count": int(
                unitree_groot_n17_sonic_sim2sim_execution.get("object_robot_contact_count") or 0
            ),
            "unitree_groot_n17_sonic_object_horizontal_displacement_m": (
                unitree_groot_n17_sonic_sim2sim_execution.get("object_horizontal_displacement_m")
            ),
            "unitree_groot_n17_sonic_contact_rollout_blockers": list(
                unitree_groot_n17_sonic_sim2sim_execution.get("contact_rollout_blockers") or []
            ),
            "policy_action_chunk_consumed_by_sim": final_success_policy_action_consumed_by_sim,
            "policy_action_chunk_integrated_into_contact_rollout": (
                final_success_policy_chunk_integrated_into_contact_rollout
            ),
            "wam_evaluator_in_control_loop": bool(
                robot_policy_wam_closed_loop_attempt.get("wam_evaluator_in_control_loop")
            ),
            "policy_observes_wam_generated_next_observation": bool(
                robot_policy_wam_closed_loop_attempt.get(
                    "policy_observes_wam_generated_next_observation"
                )
            ),
            "object_or_tote_correctly_placed": bool(
                final_success_policy_chunk_integrated_into_contact_rollout
            ),
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
            "accepted_anchor_manipulation_success_proven": False,
            "not_claimed_as_openvla_oscar_cosmos_policy_proof": True,
            "blockers": final_success_judge["blockers"],
        },
    )
    robot_policy_wam_loop_manifest = {
        "schema_version": "robot_policy_wam_loop_manifest.v1",
        "generated_at": generated_at,
        "status": "completed" if attempts else "blocked",
        "target_architecture": (
            "scene plus robot observation -> policy endpoint action chunk -> simulator/controller "
            "execution -> WAM next-world evaluator -> policy observes generated next observation -> "
            "repeat -> VLM/human success judge"
        ),
        "actual_loop_mode": (
            "unitree_policy_wam_generated_observation_closed_loop"
            if robot_policy_wam_closed_loop_attempt.get("status") == "completed"
            else "mujoco_policy_endpoint_execution_with_offline_wam_trace_package"
        ),
        "scene_source": "procedural_mujoco_task_scene",
        "scene_runtime_backend": "mujoco",
        "robot_loaded_in_scene": True,
        "robot_model": "unitree_g1_mjcf",
        "robot_model_path": str(g1_source_xml),
        "hands_capable_g1_mjcf_selected": bool(
            g1_mjcf_selection.get("hands_capable_g1_mjcf_selected")
        ),
        "g1_robot_policy_selection_contract": "unitree_native_policy_required_for_g1_claims",
        "g1_robot_policy_selected_family": g1_robot_policy_selected_family,
        "unitree_hand_manipulation_policy_scope": unitree_hand_manipulation_policy_scope,
        "openvla_selected_as_g1_robot_policy": False,
        "wam_rollout_selected_as_g1_robot_policy": False,
        "policy_endpoint_used": endpoint_policy_used,
        "policy_action_model_command_ran": bool(
            policy_action_model_command_execution.get("policy_action_model_command_ran")
        ),
        "openvla_policy_action_command_ran": bool(
            policy_action_model_command_execution.get("openvla_policy_action_command_ran")
        ),
        "unitree_policy_action_command_ran": bool(
            policy_action_model_command_execution.get("unitree_policy_action_command_ran")
        ),
        "unitree_lerobot_policy_action_command_ran": bool(
            policy_action_model_command_execution.get("unitree_lerobot_policy_action_command_ran")
        ),
        "unitree_unifolm_policy_action_command_ran": bool(
            policy_action_model_command_execution.get("unitree_unifolm_policy_action_command_ran")
        ),
        "unitree_endpoint_hand_policy_output_observed": (
            unitree_endpoint_hand_policy_output_observed
        ),
        "unitree_endpoint_hand_policy_used": unitree_endpoint_hand_policy_used,
        "unitree_endpoint_provider_output_replay_used": (
            unitree_endpoint_provider_output_replay_used
        ),
        "unitree_endpoint_action_chunk_used": unitree_endpoint_action_chunk_used,
        "unitree_endpoint_fresh_policy_action_command_ran": (
            unitree_endpoint_fresh_policy_action_command_ran
        ),
        "unitree_groot_n17_sonic_policy_action_command_ran": bool(
            policy_action_model_command_execution.get(
                "unitree_groot_n17_sonic_policy_action_command_ran"
            )
        ),
        "unitree_groot_n17_sonic_sim2sim_command_ran": (
            unitree_groot_n17_sonic_sim2sim_command_ran
        ),
        "unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim": (
            unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim
        ),
        "unitree_unifolm_endpoint_policy_action_command_ran": (
            unitree_endpoint_fresh_policy_action_command_ran
        ),
        "real_vla_or_unitree_hand_policy_endpoint_used": unitree_endpoint_hand_policy_used,
        "policy_action_chunk_consumed_by_sim": final_success_policy_action_consumed_by_sim,
        "policy_action_chunk_integrated_into_contact_rollout": (
            final_success_policy_chunk_integrated_into_contact_rollout
        ),
        "unitree_groot_n17_sonic_sim2sim_execution": str(
            job_dir / "unitree_groot_n17_sonic_sim2sim_execution.json"
        ),
        "unitree_manipulation_policy_action_command_ran": bool(
            policy_action_model_command_execution.get(
                "unitree_manipulation_policy_action_command_ran"
            )
        ),
        "policy_action_model_command_selected_candidate_id": policy_action_model_command_execution.get(
            "selected_candidate_id"
        ),
        "fixture_policy_used": final_fixture_policy_used,
        "endpoint_invocation_count": endpoint_invocation_count,
        "selected_policy_runtime": selected_runtime,
        "policy_action_contract": [
            "waypoint",
            "base_velocity",
            "stop",
            "inspect_look",
            "manipulation_contact",
        ],
        "controller_backend": selected_controller_backend,
        "navigation_policy_kind": final_navigation_policy_kind,
        "unitree_lower_body_locomotion_policy_used": unitree_lower_body_locomotion_policy_used,
        "unitree_locomotion_policy_checkpoint_path": unitree_locomotion_policy_checkpoint_path,
        "unitree_locomotion_policy_config_path": unitree_locomotion_policy_config_path,
        "unitree_hand_manipulation_policy_used": unitree_hand_manipulation_policy_used,
        "unitree_hand_manipulation_policy_kind": unitree_hand_manipulation_policy_kind,
        "unitree_generated_rollout_review_status": unitree_generated_rollout_review_manifest[
            "status"
        ],
        "unitree_generated_rollout_visually_useful_for_success_review": bool(
            unitree_generated_rollout_review_completed
        ),
        "generated_rollout_visual_smoke_status": generated_rollout_visual_smoke_status,
        "generated_rollout_visually_useful_for_success_review": generated_rollout_visually_useful,
        "generated_videos_decode_valid_for_review": generated_videos_decode_valid_for_review,
        "manipulation_policy_kind": "contact_trace_proxy_only",
        "policy_observation_source": (
            "mujoco_structured_state_plus_simulated_egocentric_frame_when_renderer_available"
        ),
        "policy_visual_observation_count": len(policy_visual_observation_rows),
        "policy_visual_observation_available_count": sum(
            1 for row in policy_visual_observation_rows if row.get("available")
        ),
        "mujoco_segmentation_diagnostic_available": any(
            row.get("available") for row in policy_segmentation_observation_rows
        ),
        "segmentation_backend": "mujoco_renderer_native",
        "policy_segmentation_observation_count": len(policy_segmentation_observation_rows),
        "policy_segmentation_observation_available_count": sum(
            1 for row in policy_segmentation_observation_rows if row.get("available")
        ),
        "policy_segmentation_observation_manifest": str(
            job_dir / "policy_segmentation_observations.json"
        ),
        "policy_visual_observation_trace": str(job_dir / "policy_visual_observation_trace.jsonl"),
        "policy_visual_observation_manifest": str(
            job_dir / "policy_visual_observation_manifest.json"
        ),
        "g1_projected_skeleton_trace": str(g1_projected_skeleton_trace_path),
        "g1_projected_skeleton_manifest": str(job_dir / "g1_projected_skeleton_manifest.json"),
        "g1_projected_skeleton_trace_status": g1_projected_skeleton_manifest.get("status"),
        "camera_views_rendered_for_review": list(selected_video_cameras),
        "egocentric_camera_views_available": list(EGOCENTRIC_VIDEO_CAMERAS),
        "egocentric_upper_body_observation_pose_manifest": str(
            job_dir / "egocentric_upper_body_observation_pose_manifest.json"
        ),
        "hands_or_end_effectors_expected_in_egocentric_view": bool(
            egocentric_observation_pose_manifest.get(
                "hands_or_end_effectors_expected_in_egocentric_torso_view"
            )
        ),
        "first_person_policy_observation_candidate_available": any(
            bool(row.get("first_person_policy_observation_candidate")) for row in video_rows
        ),
        "wam_evaluator_in_control_loop": bool(
            robot_policy_wam_closed_loop_attempt.get("wam_evaluator_in_control_loop")
        ),
        "policy_observes_wam_generated_next_observation": bool(
            robot_policy_wam_closed_loop_attempt.get(
                "policy_observes_wam_generated_next_observation"
            )
        ),
        "repeated_policy_calls_count": int(
            robot_policy_wam_closed_loop_attempt.get("repeated_policy_calls_count") or 0
        ),
        "generated_next_observation_count": int(
            robot_policy_wam_closed_loop_attempt.get("generated_next_observation_count") or 0
        ),
        "robot_policy_wam_closed_loop_attempt_status": robot_policy_wam_closed_loop_attempt.get(
            "status"
        ),
        "robot_policy_wam_closed_loop_attempt_blockers": robot_policy_wam_closed_loop_attempt.get(
            "blockers", []
        ),
        "robot_policy_wam_closed_loop_attempt": str(
            job_dir / "robot_policy_wam_closed_loop" / "robot_policy_wam_closed_loop_attempt.json"
        ),
        "robot_policy_wam_loop_trace": str(
            job_dir / "robot_policy_wam_closed_loop" / "robot_policy_wam_loop_trace.jsonl"
        ),
        "wam_generated_next_observation_trace": str(
            job_dir / "robot_policy_wam_closed_loop" / "wam_generated_next_observations.jsonl"
        ),
        "robot_policy_wam_side_by_side_trace_manifest": str(
            job_dir
            / "robot_policy_wam_closed_loop"
            / "robot_policy_wam_side_by_side_trace_manifest.json"
        ),
        "robot_policy_wam_side_by_side_trace_jsonl": str(
            job_dir / "robot_policy_wam_closed_loop" / "robot_policy_wam_side_by_side_trace.jsonl"
        ),
        "robot_policy_wam_side_by_side_trace_html": str(
            job_dir / "robot_policy_wam_closed_loop" / "robot_policy_wam_side_by_side_trace.html"
        ),
        "robot_policy_wam_side_by_side_transition_count": int(
            robot_policy_wam_closed_loop_attempt.get("side_by_side_transition_count") or 0
        ),
        "oscar_cosmos_wam_evaluator_role": (
            "offline evaluator package consumer; run blueprint-run-oscar-cosmos-wam-evaluator "
            "against this job_dir to generate WAM rollout/evaluator artifacts"
        ),
        "automated_success_source": "structured_mujoco_state_contact_action_traces",
        "vlm_success_judge_in_this_lane": False,
        "required_to_match_requested_closed_loop": [
            "real scene ingestion boundary for USD/MJCF/PLY/SPZ inputs",
            "task-specific VLA or Unitree manipulation policy endpoint with checkpoint/runtime",
            "WAM evaluator API called between policy action chunks",
            "policy adapter that can consume WAM-generated next observations",
            "VLM success judge configured over selected generated or MuJoCo episode videos",
        ],
        "truth_boundary": (
            "This run proves local MuJoCo execution and, only when the loop-attempt status is "
            "completed, repeated Unitree-specific policy calls over evaluator-generated next "
            "observations. Ranking proof requires the scoped generated-world "
            "policy-evaluation rank-fidelity gate."
        ),
    }
    write_json(job_dir / "robot_policy_wam_loop_manifest.json", robot_policy_wam_loop_manifest)
    write_json(
        job_dir / "controller_truth_boundary.json",
        {
            "schema_version": "controller_truth_boundary.v1",
            "generated_at": generated_at,
            "requested_controller_backend": controller_backend,
            "controller_backend": selected_controller_backend,
            "controller_kind": final_navigation_policy_kind,
            "unitree_lower_body_locomotion_policy_used": unitree_lower_body_locomotion_policy_used,
            "unitree_locomotion_policy_kind": final_navigation_policy_kind
            if unitree_lower_body_locomotion_policy_used
            else None,
            "unitree_locomotion_policy_config_path": unitree_locomotion_policy_config_path,
            "unitree_locomotion_policy_checkpoint_path": unitree_locomotion_policy_checkpoint_path,
            "unitree_hand_manipulation_policy_used": unitree_hand_manipulation_policy_used,
            "unitree_endpoint_hand_policy_output_observed": (
                unitree_endpoint_hand_policy_output_observed
            ),
            "unitree_endpoint_hand_policy_used": unitree_endpoint_hand_policy_used,
            "unitree_endpoint_provider_output_replay_used": (
                unitree_endpoint_provider_output_replay_used
            ),
            "unitree_endpoint_action_chunk_used": unitree_endpoint_action_chunk_used,
            "unitree_groot_n17_sonic_sim2sim_command_ran": (
                unitree_groot_n17_sonic_sim2sim_command_ran
            ),
            "unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim": (
                unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim
            ),
            "policy_action_chunk_consumed_by_sim": final_success_policy_action_consumed_by_sim,
            "policy_action_chunk_integrated_into_contact_rollout": (
                final_success_policy_chunk_integrated_into_contact_rollout
            ),
            "unitree_hand_manipulation_policy_kind": unitree_hand_manipulation_policy_kind,
            "unitree_lerobot_or_isaaclab_manipulation_policy_used": False,
            "policy_action_model_command_ran": bool(
                policy_action_model_command_execution.get("policy_action_model_command_ran")
            ),
            "unitree_policy_action_command_ran": bool(
                policy_action_model_command_execution.get("unitree_policy_action_command_ran")
            ),
            "unitree_lerobot_policy_action_command_ran": bool(
                policy_action_model_command_execution.get(
                    "unitree_lerobot_policy_action_command_ran"
                )
            ),
            "unitree_unifolm_policy_action_command_ran": bool(
                policy_action_model_command_execution.get(
                    "unitree_unifolm_policy_action_command_ran"
                )
            ),
            "unitree_groot_n17_sonic_policy_action_command_ran": bool(
                policy_action_model_command_execution.get(
                    "unitree_groot_n17_sonic_policy_action_command_ran"
                )
            ),
            "unitree_groot_n17_sonic_sim2sim_execution": str(
                job_dir / "unitree_groot_n17_sonic_sim2sim_execution.json"
            ),
            "wam_evaluator_in_control_loop": bool(
                robot_policy_wam_closed_loop_attempt.get("wam_evaluator_in_control_loop")
            ),
            "policy_observes_wam_generated_next_observation": bool(
                robot_policy_wam_closed_loop_attempt.get(
                    "policy_observes_wam_generated_next_observation"
                )
            ),
            "repeated_policy_calls_count": int(
                robot_policy_wam_closed_loop_attempt.get("repeated_policy_calls_count") or 0
            ),
            "generated_next_observation_count": int(
                robot_policy_wam_closed_loop_attempt.get("generated_next_observation_count") or 0
            ),
            "unitree_g1_manipulation_policy_discovery": str(
                job_dir / "unitree_g1_manipulation_policy_discovery.json"
            ),
            "manipulation_policy_kind": "contact_trace_proxy_only",
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
            "same_scene_policy_action_clipped_update_count": same_scene_policy_action_clipped_update_count,
            "same_scene_max_raw_policy_action_abs": round(
                float(same_scene_max_raw_policy_action_abs), 6
            ),
            "same_scene_max_applied_policy_action_abs": round(
                float(same_scene_max_applied_policy_action_abs), 6
            ),
            "controller_command_limits": dict(UNITREE_RL_GYM_CONTROLLER_COMMAND_LIMITS),
            "official_unitree_controller_sidecar_command_xyz": official_controller_sidecar.get(
                "command_xyz"
            ),
            "freejoint_proxy_used": final_freejoint_proxy_used,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
            "blockers": same_scene_controller_manifest.get("blockers", [])
            if same_scene_integrated
            else same_scene_controller_manifest.get("blockers", [])
            or navigation_discovery.get("blockers", []),
            "proof_boundary": (
                "same-scene Unitree RL Gym policy execution in MuJoCo is still simulator-only "
                "and needs the generated-world policy-evaluation rank-fidelity gate before ranking claims"
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
            unitree_endpoint_policy_summary=unitree_endpoint_policy_summary,
        ),
    )
    write_json(
        job_dir / "policy_endpoint_boundary_manifest.json",
        build_policy_endpoint_boundary_manifest(
            generated_at=generated_at,
            endpoint_discovery=endpoint_discovery,
            selected_runtime=selected_runtime,
            endpoint_policy_used=endpoint_policy_used,
            fixture_policy_used=final_fixture_policy_used,
            endpoint_invocation_count=endpoint_invocation_count,
            endpoint_valid_action_count=endpoint_policy_valid_actions,
            rejected_policy_action_count=len(rejected_actions),
            policy_execution_manifest_path=job_dir / "policy_execution_manifest.json",
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
        "policy_lane": policy_lane,
        "mujoco_runtime_available": True,
        "unitree_g1_mujoco_model_source": asset_manifest["unitree_g1_mujoco_model_source"],
        "unitree_g1_mujoco_model_path": str(g1_source_xml),
        "unitree_g1_hands_capable_mjcf_selected": bool(
            g1_mjcf_selection.get("hands_capable_g1_mjcf_selected")
        ),
        "unitree_g1_loaded_in_mujoco": True,
        "policy_endpoint_runtime_proven": endpoint_policy_used,
        "policy_action_model_command_ran": bool(
            policy_action_model_command_execution.get("policy_action_model_command_ran")
        ),
        "openvla_policy_action_command_ran": bool(
            policy_action_model_command_execution.get("openvla_policy_action_command_ran")
        ),
        "unitree_policy_action_command_ran": bool(
            policy_action_model_command_execution.get("unitree_policy_action_command_ran")
        ),
        "unitree_lerobot_policy_action_command_ran": bool(
            policy_action_model_command_execution.get("unitree_lerobot_policy_action_command_ran")
        ),
        "unitree_unifolm_policy_action_command_ran": bool(
            policy_action_model_command_execution.get("unitree_unifolm_policy_action_command_ran")
        ),
        "unitree_groot_n17_sonic_policy_action_command_ran": bool(
            policy_action_model_command_execution.get(
                "unitree_groot_n17_sonic_policy_action_command_ran"
            )
        ),
        "unitree_groot_n17_sonic_sim2sim_command_ran": (
            unitree_groot_n17_sonic_sim2sim_command_ran
        ),
        "unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim": (
            unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim
        ),
        "policy_action_chunk_consumed_by_sim": final_success_policy_action_consumed_by_sim,
        "policy_action_chunk_integrated_into_contact_rollout": (
            final_success_policy_chunk_integrated_into_contact_rollout
        ),
        "unitree_manipulation_policy_action_command_ran": bool(
            policy_action_model_command_execution.get(
                "unitree_manipulation_policy_action_command_ran"
            )
        ),
        "policy_action_model_command_selected_candidate_id": policy_action_model_command_execution.get(
            "selected_candidate_id"
        ),
        "wam_evaluator_in_control_loop": bool(
            robot_policy_wam_closed_loop_attempt.get("wam_evaluator_in_control_loop")
        ),
        "policy_observes_wam_generated_next_observation": bool(
            robot_policy_wam_closed_loop_attempt.get(
                "policy_observes_wam_generated_next_observation"
            )
        ),
        "repeated_policy_calls_count": int(
            robot_policy_wam_closed_loop_attempt.get("repeated_policy_calls_count") or 0
        ),
        "generated_next_observation_count": int(
            robot_policy_wam_closed_loop_attempt.get("generated_next_observation_count") or 0
        ),
        "robot_policy_wam_closed_loop_attempt_status": robot_policy_wam_closed_loop_attempt.get(
            "status"
        ),
        "robot_policy_wam_closed_loop_attempt_blockers": robot_policy_wam_closed_loop_attempt.get(
            "blockers", []
        ),
        "whole_unitree_policy_stack_installed": unitree_stack_installation_audit[
            "whole_unitree_policy_stack_installed"
        ],
        "unitree_policy_stack_installation_status": unitree_stack_installation_audit["status"],
        "unitree_policy_stack_installation_blockers": unitree_stack_installation_audit["blockers"],
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
        "policy_visual_observation_count": len(policy_visual_observation_rows),
        "policy_visual_observation_available_count": sum(
            1 for row in policy_visual_observation_rows if row.get("available")
        ),
        "mujoco_segmentation_diagnostic_available": any(
            row.get("available") for row in policy_segmentation_observation_rows
        ),
        "segmentation_backend": "mujoco_renderer_native",
        "policy_segmentation_observation_count": len(policy_segmentation_observation_rows),
        "policy_segmentation_observation_available_count": sum(
            1 for row in policy_segmentation_observation_rows if row.get("available")
        ),
        "policy_segmentation_observation_manifest": str(
            job_dir / "policy_segmentation_observations.json"
        ),
        "g1_projected_skeleton_trace_status": g1_projected_skeleton_manifest.get("status"),
        "g1_projected_skeleton_trace_row_count": int(
            g1_projected_skeleton_manifest.get("row_count") or 0
        ),
        "g1_projected_skeleton_projectable_row_count": int(
            g1_projected_skeleton_manifest.get("projectable_row_count") or 0
        ),
        "simulated_g1_projected_skeleton_available_for_wam_conditioning": bool(
            g1_projected_skeleton_manifest.get("status") == "completed"
        ),
        "policy_visual_observation_available_for_vla": any(
            row.get("available") for row in policy_visual_observation_rows
        ),
        "locomotion_continuity_validated": continuity_report["locomotion_continuity_validated"],
        "collision_dynamics_validated": collision_report["collision_dynamics_validated"],
        "manipulation_contact_dynamics_validated": manipulation_validated,
        "wam_evaluator_trace_scored": True,
        "requested_controller_backend": controller_backend,
        "controller_backend": selected_controller_backend,
        "unitree_lower_body_locomotion_policy_used": unitree_lower_body_locomotion_policy_used,
        "unitree_locomotion_policy_kind": final_navigation_policy_kind
        if unitree_lower_body_locomotion_policy_used
        else None,
        "unitree_locomotion_policy_config_path": unitree_locomotion_policy_config_path,
        "unitree_locomotion_policy_checkpoint_path": unitree_locomotion_policy_checkpoint_path,
        "g1_robot_policy_selection_contract": "unitree_native_policy_required_for_g1_claims",
        "g1_robot_policy_selected_family": g1_robot_policy_selected_family,
        "unitree_hand_manipulation_policy_scope": unitree_hand_manipulation_policy_scope,
        "openvla_selected_as_g1_robot_policy": False,
        "wam_rollout_selected_as_g1_robot_policy": False,
        "unitree_endpoint_hand_policy_output_observed": (
            unitree_endpoint_hand_policy_output_observed
        ),
        "unitree_endpoint_hand_policy_used": unitree_endpoint_hand_policy_used,
        "unitree_endpoint_provider_output_replay_used": (
            unitree_endpoint_provider_output_replay_used
        ),
        "unitree_endpoint_fresh_policy_action_command_ran": (
            unitree_endpoint_fresh_policy_action_command_ran
        ),
        "unitree_endpoint_action_chunk_used": unitree_endpoint_action_chunk_used,
        "unitree_endpoint_provider_replay_is_not_fresh_hand_policy_inference": (
            unitree_endpoint_provider_output_replay_used
        ),
        "unitree_hand_manipulation_policy_used": unitree_hand_manipulation_policy_used,
        "unitree_hand_manipulation_policy_kind": unitree_hand_manipulation_policy_kind,
        "unitree_lerobot_or_isaaclab_manipulation_policy_used": False,
        "unitree_lerobot_runtime_status": unitree_lerobot_runtime_summary.get("status"),
        "unitree_lerobot_runtime_configured": unitree_lerobot_runtime_summary.get(
            "unitree_lerobot_runtime_configured"
        ),
        "unitree_lerobot_command_built": unitree_lerobot_runtime_summary.get(
            "unitree_lerobot_command_built"
        ),
        "unitree_lerobot_sim_inference_attempted": unitree_lerobot_runtime_summary.get(
            "unitree_lerobot_sim_inference_attempted"
        ),
        "unitree_lerobot_sim_inference_proven": unitree_lerobot_runtime_summary.get(
            "unitree_lerobot_sim_inference_proven"
        ),
        "unitree_lerobot_truth_boundary": unitree_lerobot_runtime_summary.get(
            "truth_boundary_path"
        ),
        "unitree_lerobot_handoff_manifest": unitree_lerobot_runtime_summary.get(
            "handoff_manifest_path"
        ),
        "unitree_groot_n17_sonic_runtime_status": (
            unitree_groot_n17_sonic_runtime_summary.get("status")
        ),
        "unitree_groot_n17_sonic_policy_configured": (
            unitree_groot_n17_sonic_runtime_summary.get("unitree_groot_n17_sonic_policy_configured")
        ),
        "unitree_groot_n17_sonic_ready_for_policy_action_command": (
            unitree_groot_n17_sonic_runtime_summary.get("ready_for_policy_action_command")
        ),
        "unitree_groot_n17_sonic_ready_for_sim2sim": (
            unitree_groot_n17_sonic_runtime_summary.get("ready_for_sim2sim")
        ),
        "unitree_groot_n17_sonic_installation_audit": (
            unitree_groot_n17_sonic_runtime_summary.get("installation_audit_path")
        ),
        "unitree_groot_n17_sonic_truth_boundary": (
            unitree_groot_n17_sonic_runtime_summary.get("truth_boundary_path")
        ),
        "unitree_g1_manipulation_policy_discovery": str(
            job_dir / "unitree_g1_manipulation_policy_discovery.json"
        ),
        "manipulation_policy_kind": "contact_trace_proxy_only",
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
        "same_scene_unitree_controller_backend_status": same_scene_controller_manifest.get(
            "status"
        ),
        "same_scene_unitree_controller_rollout_fall_count": total_attempt_fall_count,
        "same_scene_unitree_controller_update_count": len(same_scene_controller_rows),
        "unitree_generated_rollout_review_status": unitree_generated_rollout_review_manifest[
            "status"
        ],
        "unitree_generated_rollout_visually_useful_for_success_review": bool(
            unitree_generated_rollout_review_completed
        ),
        "generated_rollout_visual_smoke_status": generated_rollout_visual_smoke_status,
        "generated_rollout_visually_useful_for_success_review": generated_rollout_visually_useful,
        "generated_videos_decode_valid_for_review": generated_videos_decode_valid_for_review,
        "generated_video_review_validation_count": len(generated_video_review_validations),
        "final_success_judge_status": final_success_judge["status"],
        "final_success_judge_score": final_success_judge["score"],
        "object_or_tote_correctly_placed": final_success_judge["object_or_tote_correctly_placed"],
        "final_success_judge_answer": final_success_judge["answer"],
        "unitree_endpoint_action_command_count": len(unitree_endpoint_command_rows),
        "unitree_endpoint_action_controller_clamped_command_count": unitree_controller_clamped_command_count,
        "same_scene_controller_clamped_update_count": same_scene_controller_clamped_update_count,
        "same_scene_policy_action_clipped_update_count": same_scene_policy_action_clipped_update_count,
        "same_scene_max_raw_policy_action_abs": round(
            float(same_scene_max_raw_policy_action_abs), 6
        ),
        "same_scene_max_applied_policy_action_abs": round(
            float(same_scene_max_applied_policy_action_abs), 6
        ),
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
        "scenario_eval_run_coverage_complete": len(attempts) == required_count
        and required_count > 0,
        "attempt_count_matches_matrix_count": len(attempts) == required_count,
        "pass_fail_by_task": _counts_by_key(attempts, "task_id"),
        "pass_fail_by_spawn": _counts_by_key(attempts, "spawn_id"),
        "isaac_runtime_available": False,
        "realistic_splat_visual_rendered": False,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "artifact_paths": {
            "policy_endpoint_discovery": str(job_dir / "policy_endpoint_discovery.json"),
            "policy_endpoint_runtime_manifest": str(
                job_dir / "policy_endpoint_runtime_manifest.json"
            ),
            "policy_endpoint_boundary_manifest": str(
                job_dir / "policy_endpoint_boundary_manifest.json"
            ),
            "policy_endpoint_server_manifest": str(
                job_dir / "policy_endpoint_server_manifest.json"
            ),
            "policy_command_adapter_manifest": str(
                job_dir / "policy_command_adapter_manifest.json"
            ),
            "wam_vla_runtime_discovery": str(job_dir / "wam_vla_runtime_discovery.json"),
            "policy_endpoint_auth_manifest": str(job_dir / "policy_endpoint_auth_manifest.json"),
            "policy_endpoint_probe_results": str(job_dir / "policy_endpoint_probe_results.json"),
            "policy_endpoint_invocation_trace_jsonl": str(
                job_dir / "policy_endpoint_invocation_trace.jsonl"
            ),
            "policy_visual_observation_manifest": str(
                job_dir / "policy_visual_observation_manifest.json"
            ),
            "policy_visual_observation_trace_jsonl": str(
                job_dir / "policy_visual_observation_trace.jsonl"
            ),
            "policy_segmentation_observation_manifest": str(
                job_dir / "policy_segmentation_observations.json"
            ),
            "g1_projected_skeleton_trace_jsonl": str(g1_projected_skeleton_trace_path),
            "g1_projected_skeleton_manifest": str(job_dir / "g1_projected_skeleton_manifest.json"),
            "policy_model_candidate_matrix": str(job_dir / "policy_model_candidate_matrix.json"),
            "policy_model_truth_boundary": str(job_dir / "policy_model_truth_boundary.json"),
            "policy_action_model_command_discovery": str(
                job_dir / "policy_action_model_command_discovery.json"
            ),
            "policy_action_model_command_execution": str(
                job_dir / "policy_action_model_command_execution.json"
            ),
            "policy_action_model_command_output": str(
                job_dir / "policy_action_model_command_output.json"
            ),
            "unitree_policy_provider_registry_probe": str(
                job_dir / "unitree_policy_provider_registry_probe.json"
            ),
            "unitree_policy_stack_installation_audit": str(
                job_dir / "unitree_policy_stack_installation_audit.json"
            ),
            "unitree_lerobot_g1_runtime_probe": str(
                job_dir / "unitree_lerobot_g1_runtime_probe.json"
            ),
            "unitree_lerobot_g1_policy_runtime_summary": str(
                job_dir / "unitree_lerobot_g1_policy_runtime_summary.json"
            ),
            "unitree_lerobot_g1_policy_runtime_truth_boundary": str(
                job_dir / "unitree_lerobot_g1_policy_runtime_truth_boundary.json"
            ),
            "unitree_lerobot_g1_policy_handoff_manifest": str(
                job_dir / "unitree_lerobot_g1_policy_handoff" / "robot_team_handoff_manifest.json"
            ),
            "unitree_groot_n17_sonic_installation_audit": str(
                job_dir / "unitree_groot_n17_sonic_installation_audit.json"
            ),
            "unitree_groot_n17_sonic_policy_runtime_summary": str(
                job_dir / "unitree_groot_n17_sonic_policy_runtime_summary.json"
            ),
            "unitree_groot_n17_sonic_policy_runtime_truth_boundary": str(
                job_dir / "unitree_groot_n17_sonic_policy_runtime_truth_boundary.json"
            ),
            "unitree_groot_n17_sonic_sim2sim_execution": str(
                job_dir / "unitree_groot_n17_sonic_sim2sim_execution.json"
            ),
            "unitree_groot_n17_sonic_sim2sim_action_trace_jsonl": str(
                job_dir / "unitree_groot_n17_sonic_sim2sim_action_trace.jsonl"
            ),
            "unitree_groot_n17_sonic_sim2sim_controller_truth": str(
                job_dir / "unitree_groot_n17_sonic_sim2sim_controller_truth.json"
            ),
            "unitree_groot_n17_sonic_sim2sim_review_video": str(
                job_dir / "unitree_groot_n17_sonic_sim2sim_review.mp4"
            ),
            "realistic_navigation_policy_discovery": str(
                job_dir / "realistic_navigation_policy_discovery.json"
            ),
            "unitree_g1_manipulation_policy_discovery": str(
                job_dir / "unitree_g1_manipulation_policy_discovery.json"
            ),
            "official_unitree_controller_sidecar_manifest": str(
                job_dir / "official_unitree_controller_sidecar_manifest.json"
            ),
            "unitree_endpoint_action_command_stream": str(
                job_dir / "unitree_endpoint_action_command_stream.json"
            ),
            "egocentric_upper_body_observation_pose_manifest": str(
                job_dir / "egocentric_upper_body_observation_pose_manifest.json"
            ),
            "unitree_endpoint_action_controller_replay_manifest": str(
                job_dir / "unitree_endpoint_action_controller_replay_manifest.json"
            ),
            "unitree_controller_bridge_manifest": str(
                job_dir / "unitree_controller_bridge_manifest.json"
            ),
            "robot_policy_wam_loop_manifest": str(job_dir / "robot_policy_wam_loop_manifest.json"),
            "final_success_judge": str(job_dir / "final_success_judge.json"),
            "claim_boundary": str(job_dir / "claim_boundary.json"),
            "manipulation_success_evaluator_results": str(
                job_dir / "manipulation_success_evaluator_results.json"
            ),
            "robot_policy_wam_closed_loop_attempt": str(
                job_dir
                / "robot_policy_wam_closed_loop"
                / "robot_policy_wam_closed_loop_attempt.json"
            ),
            "robot_policy_wam_loop_trace_jsonl": str(
                job_dir / "robot_policy_wam_closed_loop" / "robot_policy_wam_loop_trace.jsonl"
            ),
            "wam_generated_next_observations_jsonl": str(
                job_dir / "robot_policy_wam_closed_loop" / "wam_generated_next_observations.jsonl"
            ),
            "robot_policy_wam_side_by_side_trace_manifest": str(
                job_dir
                / "robot_policy_wam_closed_loop"
                / "robot_policy_wam_side_by_side_trace_manifest.json"
            ),
            "robot_policy_wam_side_by_side_trace_jsonl": str(
                job_dir
                / "robot_policy_wam_closed_loop"
                / "robot_policy_wam_side_by_side_trace.jsonl"
            ),
            "robot_policy_wam_side_by_side_trace_html": str(
                job_dir
                / "robot_policy_wam_closed_loop"
                / "robot_policy_wam_side_by_side_trace.html"
            ),
            "same_scene_unitree_controller_backend_manifest": str(
                job_dir / "same_scene_unitree_controller_backend_manifest.json"
            ),
            "same_scene_unitree_controller_trace_jsonl": str(
                job_dir / "same_scene_unitree_controller_trace.jsonl"
            ),
            "unitree_generated_rollout_review_manifest": str(
                job_dir / "unitree_generated_rollout_review_manifest.json"
            ),
            "generated_rollout_visual_smoke": str(job_dir / "generated_rollout_visual_smoke.json"),
            "scenario_eval_matrix": str(job_dir / "scenario_eval_matrix.json"),
            "normalized_attempt_trace": str(job_dir / "normalized_attempt_trace.json"),
            "normalized_policy_action_trace_jsonl": str(
                job_dir / "normalized_policy_action_trace.jsonl"
            ),
            "g1_mujoco_locomotion_trace_jsonl": str(job_dir / "g1_mujoco_locomotion_trace.jsonl"),
            "wam_evaluator_results": str(job_dir / "wam_evaluator_results.json"),
            "video_generation_status": str(job_dir / "video_generation_status.json"),
            "video_analysis_manifest": str(job_dir / "video_analysis_manifest.json"),
            "review_video_selection_manifest": str(
                job_dir / "review_video_selection_manifest.json"
            ),
            "manipulation_endpoint_task_report": str(
                job_dir / "manipulation_endpoint_task_report.json"
            ),
        },
        "recommendation": (
            "Use this as the fast local policy-endpoint plumbing evaluator before Isaac. "
            "When controller_backend is unitree_rl_gym and controller_truth_boundary proves "
            "same-scene integration, it is simulator-only Unitree RL Gym locomotion proof. "
            "Do not use it as Isaac, physical robot, deployment, safety, or dexterous VLA proof."
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
        "--wam-loop-step-count",
        type=int,
        default=DEFAULT_WAM_LOOP_STEP_COUNT,
        help=(
            "Policy/WAM closed-loop calls to run. Defaults to 12 so the simulator "
            "artifact shows a multi-call policy/WAM loop rather than a three-call smoke test."
        ),
    )
    parser.add_argument(
        "--render-frame-count",
        type=int,
        default=0,
        help="Fixed sampled frame count. Use 0 for full-episode stride rendering.",
    )
    parser.add_argument(
        "--video-frame-stride-steps", type=int, default=DEFAULT_VIDEO_FRAME_STRIDE_STEPS
    )
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
        help=(
            "Camera to render. Repeat for multiple cameras. Defaults to head_pov and "
            "torso_pov egocentric policy-observation candidates."
        ),
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
        "--wam-generation-timeout-seconds",
        type=float,
        default=None,
        help=(
            "Timeout for live WAM generation commands. Defaults to "
            f"${WAM_GENERATION_TIMEOUT_ENV} or {DEFAULT_WAM_GENERATION_TIMEOUT_SECONDS:.0f}s. "
            "This is separate from short policy endpoint HTTP timeouts."
        ),
    )
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
    parser.add_argument(
        "--policy-lane",
        default="auto",
        choices=(
            "auto",
            "official_unitree_rl_gym",
            "unitree_lerobot_g1",
            GROOT_POLICY_ID,
            "openvla_endpoint",
            "unifolm_vla",
            "unifolm_wma",
            "unsupported",
        ),
    )
    parser.add_argument(
        "--unitree-lerobot-mode",
        default="probe",
        choices=("probe", "dry_run", "sim_eval", "not_configured"),
        help="Unitree LeRobot G1 provider mode. Probe is the default and never runs inference.",
    )
    parser.add_argument("--allow-policy-action-model-command-run", action="store_true")
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
        wam_generation_timeout_seconds=args.wam_generation_timeout_seconds,
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
        policy_lane=args.policy_lane,
        unitree_lerobot_mode=args.unitree_lerobot_mode,
        allow_policy_action_model_command_run=args.allow_policy_action_model_command_run,
        wam_loop_step_count=args.wam_loop_step_count,
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
