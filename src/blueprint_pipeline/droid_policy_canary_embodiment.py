"""Exact DROID embodiment contract for the internal policy canary.

The DROID checkpoints run on Franka hardware, but "Franka + Robotiq" is not a
complete policy compatibility claim.  The checkpoint also binds the official
DROID reset, cameras, tool frame, gripper convention, and control surface.
This module keeps that product distinction explicit and applies the parts that
must be identical before a learned policy is queried.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import re
from typing import Any

from .decision_evidence_contracts import canonical_digest


DROID_POLICY_CANARY_PRESET_ID = "droid_franka_panda_robotiq_2f85_v1"
DROID_EMBODIMENT_ID = "droid_franka_robotiq_2f85"
DROID_ARENA_SOURCE_REVISION = "8b4a3a47fc53de23e8205089d71109a2e2348acd"
DROID_ARENA_SOURCE_BLOB = "2c8566a48bed83760e204b979f76d2e58ebc8a66"
DROID_ARENA_SOURCE_PATH = "isaaclab_arena/embodiments/droid/droid.py"
DROID_NATIVE_RESET_JOINTS_RAD = {
    "panda_joint1": 0.0,
    "panda_joint2": -math.pi / 5.0,
    "panda_joint3": 0.0,
    "panda_joint4": -4.0 * math.pi / 5.0,
    "panda_joint5": 0.0,
    "panda_joint6": 3.0 * math.pi / 5.0,
    "panda_joint7": 0.0,
}


def _subject_label(task_spec: Mapping[str, Any]) -> str:
    identity = str(
        task_spec.get("source_subject_identity")
        or task_spec.get("subject_asset_id")
        or task_spec.get("task_id")
        or "task object"
    ).lower()
    if "mug" in identity or "cup" in identity:
        return "mug"
    words = [
        word
        for word in re.split(r"[^a-z0-9]+", identity)
        if word and word not in {"scene", "replacement", "object"} and not word.isdigit()
    ]
    label = " ".join(words[-2:]) if words else "task object"
    return "task object" if label == "task" else label


def concrete_droid_task_instruction(task_spec: Mapping[str, Any]) -> str:
    """Return language that names both the visible subject and destination."""

    subject = str(task_spec.get("instruction_subject_label") or "").strip()
    if not subject:
        subject = _subject_label(task_spec)
    target = str(task_spec.get("visible_target_label") or "").strip()
    if not target:
        target = "green target marker"
    verb = str(task_spec.get("instruction_verb") or "").strip()
    strategy = str(task_spec.get("manipulation_strategy") or "")
    if not verb:
        verb = "Push" if strategy == "planar_push" else "Move"
    return f"{verb} the {subject} onto the {target}."


def apply_droid_policy_canary_profile(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Bind one resolved scene plan to the official DROID policy interface."""

    value = deepcopy(dict(plan))
    robot = value.get("robot")
    task_spec = value.get("task_spec")
    if not isinstance(robot, dict) or not isinstance(task_spec, dict):
        raise ValueError("droid_policy_canary_scene_plan_invalid")
    robot["joint_reset_positions_rad"] = dict(DROID_NATIVE_RESET_JOINTS_RAD)
    task_spec["prompt"] = concrete_droid_task_instruction(task_spec)
    target_position = task_spec.get("target_position_world_m")
    marker = (
        {
            "shape": "flat_green_disc",
            "radius_m": 0.06,
            "non_colliding": True,
            "position_world_m": list(target_position),
        }
        if isinstance(target_position, list) and len(target_position) == 3
        else None
    )
    profile: dict[str, Any] = {
        "schema_version": "droid_policy_canary_embodiment_profile.v1",
        "robot_preset_id": DROID_POLICY_CANARY_PRESET_ID,
        "embodiment_id": DROID_EMBODIMENT_ID,
        "arena_source": {
            "revision": DROID_ARENA_SOURCE_REVISION,
            "git_blob_sha1": DROID_ARENA_SOURCE_BLOB,
            "path": DROID_ARENA_SOURCE_PATH,
        },
        "embodiment_class": "DroidAbsoluteJointPositionEmbodiment",
        "action_contract": {
            "arena_control_variant": "droid_abs_joint_pos",
            "executed_policy_head": "joint_position",
            "retained_auxiliary_policy_heads": ["eef_9d"],
            "official_gr00t_config_path": (
                "isaaclab_arena_gr00t/policy/config/"
                "droid_manip_gr00t_closedloop_config.yaml"
            ),
            "selection_rule": (
                "official_arena_droid_abs_joint_pos_executes_joint_head;"
                "eef_head_retained_as_diagnostic_evidence"
            ),
        },
        "preserve_official_policy_camera_calibration": True,
        "preserve_official_reset_joint_positions": True,
        "visible_target_marker": marker,
        "policy_camera_roles": ["external", "wrist"],
        "review_only_camera_roles": ["overview"],
        "profile_digest": "",
    }
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )
    value["policy_canary_embodiment_profile"] = profile
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return value


__all__ = [
    "DROID_ARENA_SOURCE_BLOB",
    "DROID_ARENA_SOURCE_PATH",
    "DROID_ARENA_SOURCE_REVISION",
    "DROID_EMBODIMENT_ID",
    "DROID_NATIVE_RESET_JOINTS_RAD",
    "DROID_POLICY_CANARY_PRESET_ID",
    "apply_droid_policy_canary_profile",
    "concrete_droid_task_instruction",
]
