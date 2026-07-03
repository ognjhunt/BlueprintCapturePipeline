#!/usr/bin/env python3
"""Isaac Sim GPU runner: MuJoCo-parity G1 walk-to-target eval in the sim-ready Lightwheel kitchen.

Runs inside Isaac's python (``/isaac-sim/python.sh``) on the GPU worker. Self-contained: the
policy module ``isaac_g1_policy.py`` is shipped alongside this script in the bundle. Per
navigation scenario it drives the SAME deterministic walk-to-target controller as the MuJoCo
lane — proposing collision-checked candidate root poses, probing each via a PhysX overlap
query, kinematically placing the G1, and RTX-rendering overview + robot-POV frames into MP4 —
and records the MuJoCo-schema trace. Emits ``isaac_g1_kitchen_parity_result.json`` (same
task-outcome contract) + traces + MP4s and uploads the out dir via the provider signed-PUT.

Honesty boundary: Stage A is a *kinematic* navigation preview (parity with MuJoCo's preview
controller), RTX-rendered on Isaac. It is not dynamic locomotion and not a learned policy; the
GR00T N1.7 SONIC stage swaps the policy (``--policy groot_sonic``) without changing this harness.

The Isaac-API calls (boot, stage, PhysX overlap, Replicator render) are GPU-only and verified
on the worker, not locally; the non-Isaac helpers are unit-tested in the repo.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import shlex
import os
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence


def _log(msg: str) -> None:
    """Flushed, timestamped progress line so the heartbeat-uploaded console shows exactly how
    far the runner got (Isaac ops between scene-load and render give no output otherwise)."""
    print(f"[parity {time.strftime('%H:%M:%S')}] {msg}", flush=True)

# --- policy import: bundle dir on the worker, package in the repo (tests) ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import isaac_g1_policy as policy_mod  # bundle (worker)
except Exception:  # noqa: BLE001
    from blueprint_pipeline import isaac_g1_policy as policy_mod  # repo (tests)

RESULT_SCHEMA_VERSION = "isaac_g1_kitchen_parity_result.v1"
DRY_RENDER_SOURCE_MARKER = "dry_render_preview"
DRY_RENDER_SOURCE_HEADER = "X-Blueprint-Render-Source"
DRY_RENDER_NOTE_HEADER = "X-Blueprint-Render-Note"
DRY_RENDER_NOT_RENDERED_NOTE = (
    "NOT a rendered frame: CPU-only dry-render preview of stance/camera/projection math."
)
# robot footprint half-extent (m) for the PhysX overlap probe (approx G1 standing bbox)
# Unitree publishes G1 standing dimensions as 1320 x 450 x 200 mm and arm span as ~0.45 m.
# The tuple is local robot-frame half extent: +x forward depth, +y lateral width, +z vertical.
ROBOT_FOOTPRINT_HALF_EXTENT = (0.12, 0.23, 0.62)
ROBOT_PELVIS_HEIGHT_M = 0.79
DEFAULT_ISAAC_LEG_KP = 100.0
DEFAULT_ISAAC_LEG_KD = 2.0
ROOT_FALL_VERTICAL_DROP_M = 0.25
ROOT_DRIFT_VERTICAL_DROP_M = 0.05
ROOT_DRIFT_DISPLACEMENT_M = 0.10
PLACEMENT_GROUND_TRUTH_MAX_FOOTPRINT_CENTER_DELTA_M = 0.10
TASK_STANCE_SCHEMA_VERSION = "task_stance_plan.v1"
TASK_STANCE_TARGET_KEYS = (
    "task_target_position_xyz",
    "manipulation_target_position_xyz",
    "target_object_position_xyz",
    "look_at_position_xyz",
)
TASK_STANCE_TARGET_BBOX_KEY_PAIRS = (
    ("task_target_bbox_min_xyz", "task_target_bbox_max_xyz"),
    ("target_object_bbox_min_xyz", "target_object_bbox_max_xyz"),
)
TASK_STANCE_FALLBACK_TARGET_KEYS = ("raw_target_position_xyz", "target")
TASK_STANCE_TARGET_OBJECT_KEYS = ("target_object_id", "task_target_object_id", "object_id")
TASK_STANCE_TARGET_OBJECT_LIST_KEYS = (
    "target_object_ids",
    "task_target_object_ids",
    "object_ids",
    "target_object_aliases",
)
TASK_STANCE_AFFORDANCE_OBJECT_LIST_KEYS = (
    "affordance_object_ids",
    "manipulation_affordance_object_ids",
    "task_affordance_object_ids",
)
TASK_STANCE_AFFORDANCE_KEYS = (
    "task_affordance_xyz",
    "manipulation_affordance_xyz",
    "affordance_position_xyz",
)
TASK_STANCE_APPROACH_KEYS = (
    "approach_position_xyz",
    "robot_start_position_xyz",
    "raw_spawn_position_xyz",
    "start",
)
TASK_STANCE_ANGLE_OFFSETS_DEG = (0, -15, 15, -30, 30, -45, 45, -60, 60, -90, 90, -120, 120, 180)
TASK_STANCE_DEFAULT_SURFACE_STANDOFF_M = 0.85
TASK_STANCE_DEFAULT_VALIDATION_STANDOFF_RANGE_M = (0.4, 1.2)
TASK_STANCE_CLOSE_REACH_TARGET_TOKENS = (
    "faucet",
    "tap",
    "sink",
    "stove",
    "stovetop",
    "cooktop",
    "burner",
    "knob",
    "door",
    "drawer",
    "handle",
    "hatch",
    "lid",
    "panel",
    "cabinet",
    "cupboard",
    "refrigerator",
    "fridge",
    "freezer",
    "dishwasher",
    "oven",
    "microwave",
    "washer",
    "dryer",
)
TASK_STANCE_CLOSE_REACH_ACTION_TOKENS = (
    "open",
    "close",
    "pull",
    "push",
    "slide",
    "grasp",
    "grab",
    "reach",
    "turn",
    "unlatch",
    "latch",
)
TASK_STANCE_CLOSE_REACH_GAP_RANGE_M = (0.08, 0.72)
MANIPULATION_READY_ARM_SELECTIONS = ("right", "left", "both")
G1_APPROX_ARM_SPAN_M = 0.45
G1_APPROX_SHOULDER_FORWARD_OFFSET_M = 0.0
G1_APPROX_SHOULDER_LATERAL_OFFSET_M = 0.16
G1_APPROX_SHOULDER_ABOVE_ROOT_M = 0.29
MANIPULATION_SEED_MAX_EFFECTOR_TO_AFFORDANCE_M = 0.35
MANIPULATION_SEED_MAX_SHOULDER_TO_AFFORDANCE_M = (
    G1_APPROX_ARM_SPAN_M + MANIPULATION_SEED_MAX_EFFECTOR_TO_AFFORDANCE_M
)
MANIPULATION_RENDERED_SEED_SHOULDER_MARGIN_M = 0.10
MANIPULATION_RENDERED_SEED_EFFECTOR_MARGIN_M = 0.05
MANIPULATION_STANCE_APPROX_SHOULDER_MARGIN_M = 0.10
MANIPULATION_STANCE_APPROX_EFFECTOR_MARGIN_M = 0.10
VISIBLE_REACH_FINAL_MAX_EFFECTOR_TO_AFFORDANCE_M = 0.12
MANIPULATION_ENDPOINT_AFFORDANCE_AIM_START_FRACTION = 0.82
MANIPULATION_HIGH_REACH_MIN_AFFORDANCE_ABOVE_SHOULDER_M = 0.22
MANIPULATION_HIGH_REACH_MAX_SEED_Z_ABOVE_SHOULDER_M = 0.38
MANIPULATION_HIGH_REACH_SEED_HEIGHT_FRACTION = 0.75
MANIPULATION_READY_ARM_JOINT_DELTAS = {
    "left": {
        "left_shoulder_pitch_joint": -0.85,
        "left_shoulder_roll_joint": 0.15,
        "left_shoulder_yaw_joint": 0.10,
        "left_elbow_joint": -0.23,
        "left_wrist_roll_joint": -0.10,
        "left_wrist_pitch_joint": -0.15,
    },
    "right": {
        "right_shoulder_pitch_joint": -0.85,
        "right_shoulder_roll_joint": -0.15,
        "right_shoulder_yaw_joint": -0.10,
        "right_elbow_joint": -0.23,
        "right_wrist_roll_joint": 0.10,
        "right_wrist_pitch_joint": -0.15,
    },
}
ACTIVE_ROBOT_PROFILE = None  # set by apply_robot_profile(); None = built-in G1 defaults
GROOT_POLICY_COMMAND_ENV = "PARITY_GROOT_POLICY_COMMAND"
GROOT_POLICY_COMMAND_TIMEOUT_ENV = "PARITY_GROOT_POLICY_COMMAND_TIMEOUT_SECONDS"
GROOT_POLICY_INITIAL_FRAME_ENV = "PARITY_GROOT_POLICY_INITIAL_FRAME"
UNITREE_G1_SONIC_NEUTRAL_STATE = {
    "left_leg": [0.0] * 6,
    "right_leg": [0.0] * 6,
    "waist": [0.0] * 3,
    "left_arm": [0.0] * 7,
    "right_arm": [0.0] * 7,
    "left_hand": [0.0] * 7,
    "right_hand": [0.0] * 7,
    "projected_gravity": [0.0, 0.0, -1.0],
}


def _robot_profile_module():
    """Dual-path import mirroring _resolve_task_target_via_scene_placement: the
    provider bundle ships scene_placement flat; the repo/tests have the package."""
    try:
        from scene_placement import robot_profile  # type: ignore
        return robot_profile
    except ImportError:
        try:
            from blueprint_pipeline.scene_placement import robot_profile  # type: ignore
            return robot_profile
        except ImportError:
            return None


def resolve_robot_profile_from_args(args):
    """--robot-profile-json > --robot-id > registry default (unitree_g1).

    Returns None when scene_placement is not importable (degraded worker):
    apply_robot_profile(None) is a no-op and the built-in G1 constants stand.
    """
    rp = _robot_profile_module()
    if rp is None:
        return None
    profile_json = getattr(args, "robot_profile_json", None)
    if profile_json:
        return rp.robot_profile_from_json_file(profile_json)
    return rp.get_robot_profile(getattr(args, "robot_id", None) or rp.DEFAULT_ROBOT_ID)


def apply_robot_profile(profile) -> None:
    """Point every robot-scale module constant at ``profile`` so placement,
    reach gating, and the dry-render skeleton stop assuming the G1. A None
    profile keeps the G1 defaults (worker without scene_placement)."""
    global ACTIVE_ROBOT_PROFILE, ROBOT_FOOTPRINT_HALF_EXTENT, ROBOT_PELVIS_HEIGHT_M
    global G1_APPROX_ARM_SPAN_M, G1_APPROX_SHOULDER_FORWARD_OFFSET_M
    global G1_APPROX_SHOULDER_LATERAL_OFFSET_M, G1_APPROX_SHOULDER_ABOVE_ROOT_M
    global MANIPULATION_SEED_MAX_EFFECTOR_TO_AFFORDANCE_M
    global MANIPULATION_SEED_MAX_SHOULDER_TO_AFFORDANCE_M
    global TASK_STANCE_DEFAULT_VALIDATION_STANDOFF_RANGE_M
    global TASK_STANCE_CLOSE_REACH_GAP_RANGE_M
    global MANIPULATION_READY_ARM_JOINT_DELTAS, _NOMINAL_G1_REST_OFFSETS
    if profile is None:
        return
    ACTIVE_ROBOT_PROFILE = profile
    ROBOT_FOOTPRINT_HALF_EXTENT = tuple(profile.footprint_half_extent_xyz)
    ROBOT_PELVIS_HEIGHT_M = float(profile.pelvis_height_m)
    G1_APPROX_ARM_SPAN_M = float(profile.arm_span_m)
    G1_APPROX_SHOULDER_FORWARD_OFFSET_M = float(profile.shoulder_forward_offset_m)
    G1_APPROX_SHOULDER_LATERAL_OFFSET_M = float(profile.shoulder_lateral_offset_m)
    G1_APPROX_SHOULDER_ABOVE_ROOT_M = float(profile.shoulder_above_root_m)
    MANIPULATION_SEED_MAX_EFFECTOR_TO_AFFORDANCE_M = float(profile.max_effector_to_affordance_m)
    MANIPULATION_SEED_MAX_SHOULDER_TO_AFFORDANCE_M = (
        G1_APPROX_ARM_SPAN_M + MANIPULATION_SEED_MAX_EFFECTOR_TO_AFFORDANCE_M
    )
    TASK_STANCE_DEFAULT_VALIDATION_STANDOFF_RANGE_M = tuple(profile.standoff_range_m)
    # The close-reach gap ceiling is an arm-reach envelope: scale it with the
    # profile's arm span (G1 baseline 0.45m span -> 0.72m max gap).
    TASK_STANCE_CLOSE_REACH_GAP_RANGE_M = (
        TASK_STANCE_CLOSE_REACH_GAP_RANGE_M[0],
        max(0.72, round(0.72 * (G1_APPROX_ARM_SPAN_M / 0.45), 4)),
    )
    if profile.manipulation_ready_arm_joint_deltas:
        MANIPULATION_READY_ARM_JOINT_DELTAS = dict(profile.manipulation_ready_arm_joint_deltas)
    if profile.link_rest_offsets:
        _NOMINAL_G1_REST_OFFSETS = tuple(profile.link_rest_offsets)


MANIPULATION_ARM_LINK_NAME_TOKENS = (
    "shoulder",
    "upper_arm",
    "upperarm",
    "forearm",
    "lower_arm",
    "lowerarm",
    "elbow",
    "wrist",
    "hand",
    "palm",
    "finger",
    "gripper",
)
MANIPULATION_ARM_POSE_MIN_LINK_MOVE_M = 0.02
MANIPULATION_POV_MAX_CAMERA_PITCH_DOWN_DEG = 26.0
MANIPULATION_POV_CAMERA_PITCH_EPSILON_DEG = 0.1
MANIPULATION_POV_HEAD_FORWARD_PITCH_DOWN_DEG = 24.0
MANIPULATION_POV_MIN_VFOV_DEG = 110.0
MANIPULATION_REACH_BLOCKER_SET = {
    "manipulation_pov_affordance_outside_g1_reach_envelope",
    "manipulation_pov_effector_too_far_from_affordance",
    "manipulation_pov_reach_feasibility_unverified",
}
DEFAULT_RENDER_STEP_WATCHDOG_SECONDS = 180.0
ROBOT_VISUAL_MESH_MISSING_BLOCKER = "robot_visual_mesh_missing"
ROBOT_REVIEW_VISUAL_PROXY_USED_BLOCKER = "robot_review_visual_proxy_used"
REVIEW_TASK_SUCCESS_EVIDENCE_SCHEMA_VERSION = "isaac_g1_review_task_success_evidence.v1"
REVIEW_CAMERA_EVIDENCE_SCHEMA_VERSION = "isaac_g1_review_camera_evidence.v1"
# The 2026-07-02 render-noise audit (docs/G1_RENDER_NOISE_AUDIT.md) diagnosed
# render_budget_sample_starvation: default-budget 64-spp manipulation POV frames came back
# starved/black (variants B/C, dark_pixel_ratio 1.0) while the same scene at 384 spp was clean
# (variants D/E), at ~11s/frame render cost. 384 is the audit-proven clean budget.
DEFAULT_PATH_TRACING_MIN_SAMPLES_PER_PIXEL = 384
DEFAULT_PATH_TRACING_MAX_SAMPLES_PER_PIXEL = 512


# ============================ testable helpers (no isaacsim) ============================

def load_request(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_scenarios(request: Mapping[str, Any]) -> list[dict]:
    """Normalize scenarios to {scenario_id, route_points:[[x,y,z],...], start, target, instruction}.
    Accepts explicit route_points, spawn_position_xyz + target_position_xyz, or a task-only
    manipulation scenario whose target is resolved later from the loaded scene."""
    out: list[dict] = []
    for raw in request.get("scenarios", []) or []:
        sid = str(raw.get("scenario_id") or raw.get("id") or f"scenario_{len(out)+1}")
        instruction = str(raw.get("instruction") or raw.get("description") or "").strip()
        route = raw.get("route_points") or raw.get("waypoints")
        start = raw.get("spawn_position_xyz") or raw.get("start") or (route[0] if route else None)
        target = raw.get("target_position_xyz") or raw.get("target") or (route[-1] if route else None)
        if start is None or target is None:
            if not instruction:
                continue
            out.append({
                "scenario_id": sid,
                "route_points": [],
                "instruction": instruction,
                "scenario_eval_run_id": raw.get("scenario_eval_run_id"),
                "floor_z_hint": float(raw.get("floor_z_hint", 0.05)),
                "task_target_deferred": True,
                "deferred_task_resolution": "scene_placement_task_label",
            })
            for key in (
                "task_target_position_xyz",
                "manipulation_target_position_xyz",
                "target_object_position_xyz",
                "look_at_position_xyz",
                "task_affordance_xyz",
                "manipulation_affordance_xyz",
                "affordance_position_xyz",
                "task_target_bbox_min_xyz",
                "task_target_bbox_max_xyz",
                "target_object_bbox_min_xyz",
                "target_object_bbox_max_xyz",
                "approach_position_xyz",
                "robot_start_position_xyz",
                "stance_distance_candidates_m",
                "preferred_stance_distance_m",
                "min_stance_distance_m",
                "max_stance_distance_m",
                "target_object_id",
                "task_target_object_id",
                "object_id",
                "target_object_ids",
                "task_target_object_ids",
                "object_ids",
                "target_object_aliases",
                "affordance_object_ids",
                "manipulation_affordance_object_ids",
                "task_affordance_object_ids",
                "target_object_label",
                "task_target_object_label",
                "task_id",
                "task",
                "task_description",
                "description",
                "task_instruction",
                "task_success_contract",
                "success_contract",
            ):
                if key in raw:
                    out[-1][key] = raw[key]
            continue
        raw_start = [float(c) for c in start]
        raw_target = [float(c) for c in target]
        start = [float(c) for c in start]
        target = [float(c) for c in target]
        if not route:
            route = [start, target]
        route = [[float(c) for c in p] for p in route]
        # lift the navigation route to pelvis height so the root trace is realistic
        route = [[p[0], p[1], ROBOT_PELVIS_HEIGHT_M] for p in route]
        out.append({
            "scenario_id": sid,
            "route_points": route,
            "start": [start[0], start[1], ROBOT_PELVIS_HEIGHT_M],
            "target": [target[0], target[1], ROBOT_PELVIS_HEIGHT_M],
            "instruction": instruction,
            "scenario_eval_run_id": raw.get("scenario_eval_run_id"),
            "raw_spawn_position_xyz": raw_start,
            "raw_target_position_xyz": raw_target,
            "floor_z_hint": float(raw.get("floor_z_hint", raw_start[2] if len(raw_start) > 2 else 0.0)),
        })
        for key in (
            "task_target_position_xyz",
            "manipulation_target_position_xyz",
            "target_object_position_xyz",
            "look_at_position_xyz",
            "task_affordance_xyz",
            "manipulation_affordance_xyz",
            "affordance_position_xyz",
            "task_target_bbox_min_xyz",
            "task_target_bbox_max_xyz",
            "target_object_bbox_min_xyz",
            "target_object_bbox_max_xyz",
            "approach_position_xyz",
            "robot_start_position_xyz",
            "stance_distance_candidates_m",
            "preferred_stance_distance_m",
            "min_stance_distance_m",
            "max_stance_distance_m",
            "target_object_id",
            "task_target_object_id",
            "object_id",
            "target_object_ids",
            "task_target_object_ids",
            "object_ids",
            "target_object_aliases",
            "affordance_object_ids",
            "manipulation_affordance_object_ids",
            "task_affordance_object_ids",
            "target_object_label",
            "task_target_object_label",
            "task_id",
            "task",
            "task_description",
            "description",
            "task_instruction",
            "task_success_contract",
            "success_contract",
        ):
            if key in raw:
                out[-1][key] = raw[key]
    return out


# ---------------- render-noise audit (testable, no isaacsim) ----------------
# Spec: G1 Textured Robot Render Noise Audit. Same dynamic path as the normal seed render
# (task string -> target resolution -> task stance -> robot pose -> camera contract), then one
# raw PNG per material/render variant with everything recorded in manifests. The canonical
# variant plan lives in blueprint_pipeline.g1_render_noise_audit and is shipped in
# request["render_noise_audit"]; this fallback keeps the worker self-contained.

RENDER_NOISE_AUDIT_DIR_NAME = "render_noise_audit"
RENDER_NOISE_AUDIT_RESULT_NAME = "render_noise_audit_result.json"
RENDER_NOISE_AUDIT_RESULT_SCHEMA_VERSION = "g1_render_noise_audit_worker_result.v1"
RENDER_NOISE_AUDIT_RUN_SCHEMA_VERSION = "g1_render_noise_audit_worker_run.v1"
RENDER_NOISE_AUDIT_CAM_PATH = "/World/Cameras/render_noise_audit_pov"
RENDER_NOISE_AUDIT_BOOST_LIGHT_PATH = "/World/RenderNoiseAuditBoostLight"
DEFAULT_AUDIT_HIGH_SAMPLES_PER_PIXEL = 384
DEFAULT_AUDIT_WARMUP_FRAMES = 8
DEFAULT_AUDIT_PER_VARIANT_SETTLE_FRAMES = 3
DEFAULT_AUDIT_BOOST_LIGHT_INTENSITY = 4500.0
# Audit steps are path traced (up to DEFAULT_AUDIT_HIGH_SAMPLES_PER_PIXEL spp) and the first
# warmup frame additionally pays cold shader compile, so the generic realtime-step watchdog
# (DEFAULT_RENDER_STEP_WATCHDOG_SECONDS) is far too tight for them: the 2026-07-02 GPU run was
# killed at `audit:warmup:0` after 180s before rendering a single variant.
DEFAULT_AUDIT_RENDER_STEP_WATCHDOG_SECONDS = 900.0
_AUDIT_MATERIAL_MONOTONIC_RANK = {
    "textured_original": 0,
    "simplified_diffuse": 1,
    "white_proxy": 2,
}
_AUDIT_END_EFFECTOR_ROLES = {"hand", "palm", "finger", "gripper", "wrist"}


def default_render_noise_audit_variants() -> list[dict]:
    """Fallback copy of the spec's minimum variant matrix (A-G)."""
    return [
        {"variant_id": "A", "label": "white_proxy_denoised_default_budget",
         "robot_material": "white_proxy", "denoiser_enabled": True,
         "render_budget": "current_default", "lighting_boost": False,
         "purpose": "known clean proxy baseline", "exploratory": False},
        {"variant_id": "B", "label": "textured_raw_default_budget",
         "robot_material": "textured_original", "denoiser_enabled": False,
         "render_budget": "current_default", "lighting_boost": False,
         "purpose": "raw textured-noise baseline", "exploratory": False},
        {"variant_id": "C", "label": "textured_denoised_default_budget",
         "robot_material": "textured_original", "denoiser_enabled": True,
         "render_budget": "current_default", "lighting_boost": False,
         "purpose": "denoiser regression check", "exploratory": False},
        {"variant_id": "D", "label": "textured_raw_high_budget",
         "robot_material": "textured_original", "denoiser_enabled": False,
         "render_budget": "high", "lighting_boost": False,
         "purpose": "test sample starvation", "exploratory": False},
        {"variant_id": "E", "label": "textured_denoised_high_budget",
         "robot_material": "textured_original", "denoiser_enabled": True,
         "render_budget": "high", "lighting_boost": False,
         "purpose": "test denoiser with enough samples", "exploratory": False},
        {"variant_id": "F", "label": "simplified_diffuse_denoised_default_budget",
         "robot_material": "simplified_diffuse", "denoiser_enabled": True,
         "render_budget": "current_default", "lighting_boost": False,
         "purpose": "test whether PBR/specular maps are unstable", "exploratory": False},
        {"variant_id": "G", "label": "textured_denoised_default_budget_bright_lighting",
         "robot_material": "textured_original", "denoiser_enabled": True,
         "render_budget": "current_default", "lighting_boost": True,
         "purpose": "test shadow/underexposure", "exploratory": False},
    ]


def audit_variant_execution_order(variants: Sequence[Mapping[str, Any]]) -> list[str]:
    """Material application is monotonic (authored -> simplified -> white proxy) so authored
    materials never need to be un-authored mid-run; textured variants render first."""
    ordered = sorted(
        variants,
        key=lambda v: (
            _AUDIT_MATERIAL_MONOTONIC_RANK.get(str(v.get("robot_material")), 3),
            str(v.get("variant_id")),
        ),
    )
    return [str(v.get("variant_id")) for v in ordered]


def render_noise_audit_plan_from_request(request: Mapping[str, Any]) -> dict:
    """Variant plan for the audit: request-shipped plan when present, fallback matrix otherwise."""
    shipped = request.get("render_noise_audit")
    if isinstance(shipped, Mapping) and shipped.get("variants"):
        variants = [dict(v) for v in shipped.get("variants") if isinstance(v, Mapping)]
        order = [str(v) for v in (shipped.get("execution_order") or [])]
        known = {str(v.get("variant_id")) for v in variants}
        if not order or set(order) != known:
            order = audit_variant_execution_order(variants)
        return {
            "schema_version": str(shipped.get("schema_version") or "g1_render_noise_audit_variant_plan.v1"),
            "variants": variants,
            "execution_order": order,
            "source": "request",
        }
    variants = default_render_noise_audit_variants()
    return {
        "schema_version": "g1_render_noise_audit_variant_plan.v1",
        "variants": variants,
        "execution_order": audit_variant_execution_order(variants),
        "source": "runner_default_matrix",
    }


def audit_samples_per_pixel(render_budget: str, *, default_spp: int, high_spp: int) -> int:
    budget = str(render_budget or "current_default").strip().lower()
    spp = high_spp if budget == "high" else default_spp
    return max(1, min(512, int(spp)))


def audit_arm_visibility_from_pov_geometry(pov_geometry: Mapping[str, Any]) -> dict:
    """Spec visibility fields from the pose-constant manipulation POV geometry record.

    Projection-based evidence only: it proves the posed USD arm links land inside the frame,
    not that the rendered pixels are readable; the per-variant robot pixel mask (when the
    instance annotator resolves) is the pixel-level complement.
    """
    by_arm = pov_geometry.get("arm_roles_in_frame_by_arm") or {}

    def _roles(side: str) -> set[str]:
        return {str(role) for role in (by_arm.get(side) or [])}

    left_roles = _roles("left")
    right_roles = _roles("right")
    return {
        "left_arm_visible": bool(left_roles),
        "right_arm_visible": bool(right_roles),
        "left_end_effector_visible": bool(left_roles & _AUDIT_END_EFFECTOR_ROLES),
        "right_end_effector_visible": bool(right_roles & _AUDIT_END_EFFECTOR_ROLES),
        "both_end_effectors_visible": bool(
            left_roles & _AUDIT_END_EFFECTOR_ROLES and right_roles & _AUDIT_END_EFFECTOR_ROLES
        ),
        "target_in_frame": bool(pov_geometry.get("target_in_frame")),
        "arm_roles_in_frame_by_arm": {side: sorted(_roles(side)) for side in ("left", "right")},
        "evidence_source": "projected_usd_arm_link_geometry",
        "claim_boundary": (
            "Projected-geometry visibility evidence for the shared audit pose. It is not "
            "pixel-level visual proof unless the per-variant robot pixel mask agrees."
        ),
    }


def _optional_xyz(value) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        vals = [float(v) for v in value]
    except Exception:  # noqa: BLE001
        return None
    if len(vals) < 2:
        return None
    if len(vals) == 2:
        vals.append(ROBOT_PELVIS_HEIGHT_M)
    return (vals[0], vals[1], vals[2])


def _target_bounds_for_scenario(
    scenario: Mapping[str, Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    for min_key, max_key in TASK_STANCE_TARGET_BBOX_KEY_PAIRS:
        bbox_min = _optional_xyz(scenario.get(min_key))
        bbox_max = _optional_xyz(scenario.get(max_key))
        if bbox_min is None or bbox_max is None:
            continue
        if any(bbox_max[i] < bbox_min[i] for i in range(3)):
            continue
        return bbox_min, bbox_max
    return None


def _target_bounds_for_stance_plan(
    stance_plan: Mapping[str, Any] | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if not stance_plan:
        return None
    bounds = stance_plan.get("task_target_bounds")
    if isinstance(bounds, Mapping):
        bbox_min = _optional_xyz(bounds.get("bbox_min_xyz"))
        bbox_max = _optional_xyz(bounds.get("bbox_max_xyz"))
        if bbox_min is not None and bbox_max is not None:
            return bbox_min, bbox_max
    return _target_bounds_for_scenario(stance_plan)


def _surface_affordance_point_for_stance(
    stance_plan: Mapping[str, Any] | None,
    root_pose: Sequence[float] | None,
) -> tuple[float, float, float] | None:
    """Target face nearest the stance, for camera/reach focus.

    Object resolution should return the whole object centroid, but manipulation needs the reachable
    surface, not a point inside an appliance/cabinet volume. This projects the resolved target point
    onto the bbox face nearest the robot. It is scene-agnostic: no kitchen or object coordinates.
    """
    if not stance_plan or root_pose is None:
        return None
    affordance = stance_plan.get("task_affordance_xyz")
    if affordance is not None:
        point = _optional_xyz(affordance)
        if point is not None:
            return point
    target = _optional_xyz(stance_plan.get("task_target_xyz"))
    if target is None:
        return None
    bounds = _target_bounds_for_stance_plan(stance_plan)
    if bounds is None:
        return target
    bbox_min, bbox_max = bounds
    cx, cy, cz = target
    dx = float(root_pose[0]) - cx
    dy = float(root_pose[1]) - cy
    x = min(max(cx, bbox_min[0]), bbox_max[0])
    y = min(max(cy, bbox_min[1]), bbox_max[1])
    z = min(max(cz, bbox_min[2]), bbox_max[2])
    if abs(dx) >= abs(dy):
        x = bbox_max[0] if dx >= 0.0 else bbox_min[0]
    else:
        y = bbox_max[1] if dy >= 0.0 else bbox_min[1]
    return (float(x), float(y), float(z))


def _half_extent_along_bounds(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    direction_xy: tuple[float, float],
) -> float:
    if bounds is None:
        return 0.0
    bbox_min, bbox_max = bounds
    hx = max(0.0, 0.5 * (bbox_max[0] - bbox_min[0]))
    hy = max(0.0, 0.5 * (bbox_max[1] - bbox_min[1]))
    ux, uy = direction_xy
    return abs(float(ux)) * hx + abs(float(uy)) * hy


def _surface_offset_from_focus_along_bounds(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    focus: Sequence[float],
    direction_xy: tuple[float, float],
) -> float:
    """Distance from a focus point to the target footprint surface along an outward ray.

    Coarse fixture centers are fine for navigation, but manipulation tasks often resolve a fine
    affordance near one edge of the fixture. Sampling around the fixture centroid can put the robot
    close to the sink/cabinet while still far from the actual handle/knob. When a fine affordance is
    available, use it as the stance focus and only add the distance needed to exit the coarse target
    footprint before applying the requested robot standoff.
    """
    if bounds is None:
        return 0.0
    bbox_min, bbox_max = bounds
    px, py = float(focus[0]), float(focus[1])
    ux, uy = float(direction_xy[0]), float(direction_xy[1])
    inside_xy = (
        float(bbox_min[0]) <= px <= float(bbox_max[0])
        and float(bbox_min[1]) <= py <= float(bbox_max[1])
    )
    if not inside_xy:
        return 0.0
    exits: list[float] = []
    if abs(ux) > 1e-9:
        boundary_x = float(bbox_max[0]) if ux > 0.0 else float(bbox_min[0])
        t = (boundary_x - px) / ux
        if t >= 0.0:
            exits.append(t)
    if abs(uy) > 1e-9:
        boundary_y = float(bbox_max[1]) if uy > 0.0 else float(bbox_min[1])
        t = (boundary_y - py) / uy
        if t >= 0.0:
            exits.append(t)
    if exits:
        return max(0.0, min(exits))
    return _half_extent_along_bounds(bounds, direction_xy)


def _xy_rect_overlap_and_gap(
    a_min: Sequence[float],
    a_max: Sequence[float],
    b_min: Sequence[float],
    b_max: Sequence[float],
) -> dict[str, float | bool]:
    """2D rectangle relation in the scene floor plane.

    This is deliberately pure so placement validation can be tested without Isaac. ``overlap_area``
    catches visual/geometry interpenetration; ``gap_m`` catches near-zero clearance when the boxes do
    not overlap.
    """
    overlap_x = min(float(a_max[0]), float(b_max[0])) - max(float(a_min[0]), float(b_min[0]))
    overlap_y = min(float(a_max[1]), float(b_max[1])) - max(float(a_min[1]), float(b_min[1]))
    overlaps = overlap_x > 0.0 and overlap_y > 0.0
    if overlaps:
        gap = 0.0
        area = overlap_x * overlap_y
    else:
        sep_x = max(float(b_min[0]) - float(a_max[0]), float(a_min[0]) - float(b_max[0]), 0.0)
        sep_y = max(float(b_min[1]) - float(a_max[1]), float(a_min[1]) - float(b_max[1]), 0.0)
        gap = math.hypot(sep_x, sep_y)
        area = 0.0
    return {
        "overlaps_xy": bool(overlaps),
        "overlap_area_xy_m2": round(float(area), 6),
        "gap_m": round(float(gap), 6),
    }


def _placement_validation_passed(validation: Mapping[str, Any] | None) -> bool:
    if validation is None:
        return True
    blockers = validation.get("blockers")
    if isinstance(blockers, Sequence) and not isinstance(blockers, (str, bytes)) and blockers:
        return False
    return str(validation.get("status") or "accepted").lower() in {
        "accepted",
        "passed",
        "valid",
        "clear",
        "ok",
    }


def _placement_corrected_root_pose(
    validation: Mapping[str, Any] | None,
) -> tuple[float, float, float] | None:
    if not isinstance(validation, Mapping):
        return None
    diagnostics = validation.get("place_root_diagnostics")
    if not isinstance(diagnostics, Mapping):
        ground_truth = validation.get("ground_truth_placement")
        if isinstance(ground_truth, Mapping):
            diagnostics = ground_truth.get("place_root_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return None
    corrected = diagnostics.get("corrected_root_translation_xyz")
    xyz = _optional_xyz(corrected)
    if xyz is None:
        return None
    return (float(xyz[0]), float(xyz[1]), float(xyz[2]))


def _task_stance_angle_priority(offset_deg: float) -> int:
    """Bucket placement rays by approach fidelity before falling back to backside poses."""
    offset = abs(float(offset_deg))
    if offset <= 45.0:
        return 0
    if offset <= 90.0:
        return 1
    if offset <= 120.0:
        return 2
    return 3


def _task_stance_selection_key(record: Mapping[str, Any]) -> tuple[float, ...]:
    """Sort key for accepted task-stance candidates.

    Candidate generation tries near distances before far distances, but an accepted pose on the
    opposite side of the target can appear before a farther pose on the intended approach ray.
    For manipulation tasks with a resolved fine affordance, rank by approximate G1 reach first,
    then use approach/standoff as tie-breakers. This keeps unreachable but visually plausible
    fixture stances from being fed downstream.
    """
    try:
        offset = abs(float(record.get("angle_offset_deg", 180.0)))
    except Exception:  # noqa: BLE001
        offset = 180.0
    try:
        distance = float(record.get("standoff_from_target_surface_m", float("inf")))
    except Exception:  # noqa: BLE001
        distance = float("inf")
    reach = record.get("reachability_estimate")
    if isinstance(reach, Mapping):
        best = reach.get("best_reach_arm_estimate")
        best = best if isinstance(best, Mapping) else {}
        try:
            best_shoulder = float(best.get("shoulder_to_affordance_m", reach.get("nearest_shoulder_to_affordance_m", float("inf"))))
        except Exception:  # noqa: BLE001
            best_shoulder = float("inf")
        try:
            best_effector = float(best.get("seed_effector_to_affordance_m", reach.get("nearest_seed_effector_to_affordance_m", float("inf"))))
        except Exception:  # noqa: BLE001
            best_effector = float("inf")
        try:
            max_shoulder = float(reach.get("max_shoulder_to_affordance_m", float("inf")))
        except Exception:  # noqa: BLE001
            max_shoulder = float("inf")
        try:
            max_effector = float(reach.get("max_seed_effector_to_affordance_m", float("inf")))
        except Exception:  # noqa: BLE001
            max_effector = float("inf")
        try:
            nearest_shoulder = float(reach.get("nearest_shoulder_to_affordance_m", float("inf")))
        except Exception:  # noqa: BLE001
            nearest_shoulder = float("inf")
        try:
            nearest_effector = float(reach.get("nearest_seed_effector_to_affordance_m", float("inf")))
        except Exception:  # noqa: BLE001
            nearest_effector = float("inf")
        shoulder_overage = max(
            0.0,
            best_shoulder - float(MANIPULATION_SEED_MAX_SHOULDER_TO_AFFORDANCE_M),
        )
        effector_overage = max(
            0.0,
            best_effector - float(MANIPULATION_SEED_MAX_EFFECTOR_TO_AFFORDANCE_M),
        )
        return (
            0.0 if reach.get("status") == "PASS" else 1.0,
            shoulder_overage,
            effector_overage,
            best_shoulder,
            best_effector,
            nearest_shoulder,
            nearest_effector,
            max_shoulder,
            max_effector,
            float(_task_stance_angle_priority(offset)) if bool(record.get("approach_bias_enabled")) else 0.0,
            distance,
            offset,
        )
    if not bool(record.get("approach_bias_enabled")):
        return (
            0.0,
            distance,
            offset,
        )
    return (
        float(_task_stance_angle_priority(offset)),
        distance,
        offset,
    )


def _task_stance_affordance_for_scenario(
    scenario: Mapping[str, Any] | None,
) -> tuple[float, float, float] | None:
    if not scenario:
        return None
    for key in TASK_STANCE_AFFORDANCE_KEYS:
        point = _optional_xyz(scenario.get(key))
        if point is not None:
            return point
    return None


def _approx_g1_shoulder_points_for_root(
    pose: Sequence[float],
    yaw: float,
) -> dict[str, tuple[float, float, float]]:
    """Approximate G1 shoulder centers from pelvis/root pose for pre-selection reach scoring.

    The authoritative reach gate later uses the actual USD link geometry in the rendered seed.
    This lightweight model is only for ranking collision-free candidate stances before the robot
    is finally placed and posed.
    """
    root = (float(pose[0]), float(pose[1]), float(pose[2]))
    fx, fy = math.cos(float(yaw)), math.sin(float(yaw))
    lx, ly = -fy, fx
    base = (
        root[0] + fx * G1_APPROX_SHOULDER_FORWARD_OFFSET_M,
        root[1] + fy * G1_APPROX_SHOULDER_FORWARD_OFFSET_M,
        root[2] + G1_APPROX_SHOULDER_ABOVE_ROOT_M,
    )
    return {
        "left": (
            base[0] + lx * G1_APPROX_SHOULDER_LATERAL_OFFSET_M,
            base[1] + ly * G1_APPROX_SHOULDER_LATERAL_OFFSET_M,
            base[2],
        ),
        "right": (
            base[0] - lx * G1_APPROX_SHOULDER_LATERAL_OFFSET_M,
            base[1] - ly * G1_APPROX_SHOULDER_LATERAL_OFFSET_M,
            base[2],
        ),
    }


def _seed_reach_blockers(
    *,
    shoulder_to_affordance_m: float,
    effector_to_affordance_m: float,
    shoulder_margin_m: float = 0.0,
    effector_margin_m: float = 0.0,
) -> list[str]:
    """Return seed-reach blockers using G1-scale dimensions without pretending to solve IK.

    The hand/wrist-to-affordance distance is the hard visual-conditioning signal. The shoulder
    distance is a gross body-envelope guard derived from G1 arm span plus the allowed seed-effector
    neighborhood; otherwise a neutral straight-arm seed can be rejected even when the hand is close
    enough for review-quality conditioning. Margins exist because the dry runner's nominal skeleton
    and final USD link geometry are seed evidence, not a full kinematic reach solver.
    """
    blockers: list[str] = []
    shoulder_limit = (
        float(MANIPULATION_SEED_MAX_SHOULDER_TO_AFFORDANCE_M)
        + max(0.0, float(shoulder_margin_m))
    )
    effector_limit = (
        float(MANIPULATION_SEED_MAX_EFFECTOR_TO_AFFORDANCE_M)
        + max(0.0, float(effector_margin_m))
    )
    if float(shoulder_to_affordance_m) > shoulder_limit:
        blockers.append("manipulation_pov_affordance_outside_g1_reach_envelope")
    if float(effector_to_affordance_m) > effector_limit:
        blockers.append("manipulation_pov_effector_too_far_from_affordance")
    return blockers


def _task_stance_reachability_estimate(
    pose: Sequence[float],
    yaw: float,
    affordance: Sequence[float] | None,
) -> dict[str, Any] | None:
    if affordance is None:
        return None
    aff = (float(affordance[0]), float(affordance[1]), float(affordance[2]))
    shoulders = _approx_g1_shoulder_points_for_root(pose, yaw)
    by_arm: dict[str, dict[str, Any]] = {}
    shoulder_distances: list[float] = []
    effector_distances: list[float] = []
    blockers: set[str] = set()
    passing_arms: list[str] = []
    for side, shoulder in shoulders.items():
        seed_effector = _manipulation_seed_arm_target_for_shoulder(
            shoulder,
            aff,
            forward_yaw=float(yaw),
        )
        shoulder_m = math.sqrt(sum((shoulder[i] - aff[i]) ** 2 for i in range(3)))
        effector_m = math.sqrt(sum((seed_effector[i] - aff[i]) ** 2 for i in range(3)))
        arm_blockers = _seed_reach_blockers(
            shoulder_to_affordance_m=shoulder_m,
            effector_to_affordance_m=effector_m,
            shoulder_margin_m=MANIPULATION_STANCE_APPROX_SHOULDER_MARGIN_M,
            effector_margin_m=MANIPULATION_STANCE_APPROX_EFFECTOR_MARGIN_M,
        )
        blockers.update(arm_blockers)
        if not arm_blockers:
            passing_arms.append(side)
        shoulder_distances.append(shoulder_m)
        effector_distances.append(effector_m)
        by_arm[side] = {
            "status": "PASS" if not arm_blockers else "FAIL",
            "blockers": arm_blockers,
            "approx_shoulder_xyz": [round(float(v), 6) for v in shoulder],
            "approx_seed_effector_xyz": [round(float(v), 6) for v in seed_effector],
            "shoulder_to_affordance_m": round(float(shoulder_m), 4),
            "seed_effector_to_affordance_m": round(float(effector_m), 4),
        }
    nearest_shoulder = min(shoulder_distances) if shoulder_distances else float("inf")
    max_shoulder = max(shoulder_distances) if shoulder_distances else float("inf")
    nearest_effector = min(effector_distances) if effector_distances else float("inf")
    max_effector = max(effector_distances) if effector_distances else float("inf")
    best_reach_arm = None
    if by_arm:
        best_reach_arm = min(
            by_arm,
            key=lambda side: (
                float(by_arm[side].get("shoulder_to_affordance_m", float("inf"))),
                float(by_arm[side].get("seed_effector_to_affordance_m", float("inf"))),
            ),
        )
    return {
        "status": "PASS" if passing_arms else "FAIL",
        "blockers": [] if passing_arms else sorted(blockers),
        "required_passing_arm_count": 1,
        "passing_arms": passing_arms,
        "best_reach_arm": best_reach_arm,
        "best_reach_arm_estimate": by_arm.get(best_reach_arm) if best_reach_arm else None,
        "target_affordance_xyz": [round(float(v), 6) for v in aff],
        "nearest_shoulder_to_affordance_m": round(float(nearest_shoulder), 4),
        "max_shoulder_to_affordance_m": round(float(max_shoulder), 4),
        "nearest_seed_effector_to_affordance_m": round(float(nearest_effector), 4),
        "max_seed_effector_to_affordance_m": round(float(max_effector), 4),
        "required_max_shoulder_to_affordance_m": round(
            float(MANIPULATION_SEED_MAX_SHOULDER_TO_AFFORDANCE_M),
            4,
        ),
        "approx_preselection_shoulder_margin_m": round(
            float(MANIPULATION_STANCE_APPROX_SHOULDER_MARGIN_M),
            4,
        ),
        "required_max_seed_effector_to_affordance_m": round(
            float(MANIPULATION_SEED_MAX_EFFECTOR_TO_AFFORDANCE_M),
            4,
        ),
        "approx_preselection_effector_margin_m": round(
            float(MANIPULATION_STANCE_APPROX_EFFECTOR_MARGIN_M),
            4,
        ),
        "by_arm": by_arm,
        "claim_boundary": (
            "Candidate reachability is an approximate G1-dimension pre-selection score. "
            "It is intentionally looser than the rendered seed gate so approximate shoulder "
            "placement does not reject a pose before USD/rendered arm geometry is checked. "
            "It requires at least one manipulation arm to be plausibly close; both-hand "
            "visibility remains a separate seed-framing gate. The rendered manipulation POV "
            "geometry gate remains the authority."
        ),
    }


def _synthetic_target_resolution_from_scenario(
    scenario: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build target-resolution evidence from explicit scenario target bounds.

    Some callers already provide a coarse target position/bounds, then ask the stage resolver only
    for a fine handle/knob/pull. The affordance still has to be scoped to the coarse fixture, so this
    creates the small target-resolution shape that ``_scope_affordance_resolution_to_target`` expects.
    """
    bounds = _target_bounds_for_scenario(scenario)
    if bounds is None:
        return None
    target = task_stance_target_for_scenario(
        scenario,
        manipulation_look_at=None,
        allow_navigation_target_fallback=False,
    )
    if target is None:
        bbox_min, bbox_max = bounds
        target = (
            0.5 * (float(bbox_min[0]) + float(bbox_max[0])),
            0.5 * (float(bbox_min[1]) + float(bbox_max[1])),
            0.5 * (float(bbox_min[2]) + float(bbox_max[2])),
        )
    object_ids = task_stance_target_object_ids_for_scenario(scenario)
    target_id = object_ids[0] if object_ids else str(
        scenario.get("target_object_id")
        or scenario.get("task_target_object_id")
        or scenario.get("object_id")
        or "scenario_target"
    )
    bbox_min, bbox_max = bounds
    return {
        "status": "resolved",
        "source": "scenario_target_bounds",
        "selected": {
            "target_object_id": target_id,
            "target_object_label": str(
                scenario.get("target_object_label")
                or scenario.get("task_target_object_label")
                or target_id
            ),
            "prim_path": str(scenario.get("prim_path") or ""),
            "center_xyz": [round(float(v), 6) for v in target],
            "bbox_min_xyz": [round(float(v), 6) for v in bbox_min],
            "bbox_max_xyz": [round(float(v), 6) for v in bbox_max],
        },
    }


def _rounded_xyz(value: Sequence[float], *, digits: int = 6) -> list[float]:
    return [round(float(value[i]), digits) for i in range(3)]


def _xy_focus_overlap(
    obj: Any,
    focus_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    *,
    margin_m: float,
) -> bool:
    if focus_bounds is None:
        return True
    try:
        bbox_min = getattr(obj, "bbox_min")
        bbox_max = getattr(obj, "bbox_max")
        focus_min, focus_max = focus_bounds
        return not (
            float(bbox_max[0]) < float(focus_min[0]) - float(margin_m)
            or float(bbox_min[0]) > float(focus_max[0]) + float(margin_m)
            or float(bbox_max[1]) < float(focus_min[1]) - float(margin_m)
            or float(bbox_min[1]) > float(focus_max[1]) + float(margin_m)
        )
    except Exception:  # noqa: BLE001
        return True


def _aligned_box_min_max_center_size(box) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return min/max/center/size for USD aligned boxes across pxr API variants.

    ``Gf.Range3d`` exposes ``GetMin``/``GetMax``/``GetSize`` but not ``GetCenter`` in the Isaac
    worker build. Some local fakes and older code paths expose ``GetCenter``/``GetSize`` only. Keep
    both forms supported so placement validation cannot fail before it evaluates the actual geometry.
    """
    get_min = getattr(box, "GetMin", None)
    get_max = getattr(box, "GetMax", None)
    if callable(get_min) and callable(get_max):
        bmin_raw = get_min()
        bmax_raw = get_max()
        bmin = [float(bmin_raw[i]) for i in range(3)]
        bmax = [float(bmax_raw[i]) for i in range(3)]
        center = [0.5 * (bmin[i] + bmax[i]) for i in range(3)]
        size = [bmax[i] - bmin[i] for i in range(3)]
        return bmin, bmax, center, size
    center_raw = box.GetCenter()
    size_raw = box.GetSize()
    center = [float(center_raw[i]) for i in range(3)]
    size = [float(size_raw[i]) for i in range(3)]
    bmin = [center[i] - 0.5 * size[i] for i in range(3)]
    bmax = [center[i] + 0.5 * size[i] for i in range(3)]
    return bmin, bmax, center, size


def _with_xyz(scenario: Mapping[str, Any], key: str, xyz: Sequence[float]) -> dict[str, Any]:
    out = dict(scenario)
    out[key] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    return out


def _unit_xy_from(a: Sequence[float], b: Sequence[float]) -> tuple[float, float] | None:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return None
    return (dx / mag, dy / mag)


def _target_text_for_semantic_stance(scenario: Mapping[str, Any] | None) -> str:
    if not scenario:
        return ""
    values: list[str] = []
    for key in (
        "target_object_id",
        "task_target_object_id",
        "target_object_label",
        "task_target_object_label",
        "object_id",
        "prim_path",
    ):
        value = scenario.get(key)
        if value is not None:
            values.append(str(value))
    selected = scenario.get("selected")
    if isinstance(selected, Mapping):
        for key in ("target_object_id", "target_object_label", "prim_path"):
            value = selected.get(key)
            if value is not None:
                values.append(str(value))
    return " ".join(values).replace("_", " ").replace("-", " ").lower()


def _task_text_for_semantic_stance(scenario: Mapping[str, Any] | None) -> str:
    if not scenario:
        return ""
    values: list[str] = []
    for key in (
        "instruction",
        "task",
        "task_description",
        "description",
        "task_instruction",
    ):
        value = scenario.get(key)
        if value is not None:
            values.append(str(value))
    return " ".join(values).replace("_", " ").replace("-", " ").lower()


def _is_close_reach_task_target(scenario: Mapping[str, Any] | None) -> bool:
    """True when task intent plus target identity imply a close hand-on-surface stance.

    This is semantic and scene-agnostic: the target still comes from scene resolution, and the final
    pose still comes from bounds/clearance validation. The classifier only switches the reach envelope
    from counter-working distance to closer articulated-surface distance.
    """
    task_text = _task_text_for_semantic_stance(scenario)
    target_text = _target_text_for_semantic_stance(scenario)
    if not task_text or not target_text:
        return False
    has_action = any(token in task_text for token in TASK_STANCE_CLOSE_REACH_ACTION_TOKENS)
    has_target = any(token in target_text for token in TASK_STANCE_CLOSE_REACH_TARGET_TOKENS)
    return bool(has_action and has_target)


def _uses_tight_control_surface_standoff(scenario: Mapping[str, Any] | None) -> bool:
    task_text = _task_text_for_semantic_stance(scenario)
    target_text = _target_text_for_semantic_stance(scenario)
    text = f"{task_text} {target_text}"
    control_tokens = (
        "faucet",
        "tap",
        "knob",
        "dial",
        "control",
        "burner",
        "stove",
        "stovetop",
        "cooktop",
    )
    excluded_tokens = ("refrigerator", "fridge", "door", "drawer")
    return any(token in text for token in control_tokens) and not any(
        token in text for token in excluded_tokens
    )


def _close_reach_surface_standoff_candidates(
    scenario: Mapping[str, Any] | None = None,
) -> list[float]:
    half_xy = max(float(ROBOT_FOOTPRINT_HALF_EXTENT[0]), float(ROBOT_FOOTPRINT_HALF_EXTENT[1]))
    clearances = []
    if _uses_tight_control_surface_standoff(scenario):
        clearances.extend((
            max(0.02, half_xy * 0.07),
            max(0.04, half_xy * 0.14),
        ))
    clearances.extend((
        max(0.12, half_xy * 0.40),
        max(0.18, half_xy * 0.65),
        max(0.27, half_xy * 0.95),
        max(0.40, half_xy * 1.40),
        max(0.55, half_xy * 2.00),
    ))
    defaults = (TASK_STANCE_DEFAULT_SURFACE_STANDOFF_M, TASK_STANCE_DEFAULT_SURFACE_STANDOFF_M * 1.25)
    return sorted({round(half_xy + float(gap), 4) for gap in tuple(clearances) + defaults})


def _validation_standoff_range_for_scenario(
    scenario: Mapping[str, Any] | None,
) -> tuple[float, float]:
    if _is_close_reach_task_target(scenario):
        return TASK_STANCE_CLOSE_REACH_GAP_RANGE_M
    return TASK_STANCE_DEFAULT_VALIDATION_STANDOFF_RANGE_M


def task_stance_distance_candidates(scenario: Mapping[str, Any] | None = None) -> list[float]:
    """Generic robot-profile stance distances around a task target.

    The defaults are derived from the robot footprint, not from a kitchen/sink coordinate. Scenario
    metadata can override them when a task compiler knows a better reach/standoff envelope.
    """
    scenario = scenario or {}
    raw = scenario.get("stance_distance_candidates_m")
    # An empty/exhausted explicit ladder means "derive", never "no candidates".
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) > 0:
        distances = sorted({round(float(v), 4) for v in raw if float(v) > 0.0})
    elif _is_close_reach_task_target(scenario):
        distances = _close_reach_surface_standoff_candidates(scenario)
    else:
        half_xy = max(float(ROBOT_FOOTPRINT_HALF_EXTENT[0]), float(ROBOT_FOOTPRINT_HALF_EXTENT[1]))
        base = max(TASK_STANCE_DEFAULT_SURFACE_STANDOFF_M, half_xy * 3.0)
        # Keep the default samples dense around the useful counter-working band. In the Lightwheel
        # kitchen, the first non-clipping centered sink stance can be only a few centimeters farther
        # out than the closest sampled point, while the next coarse jump is already too far to reach.
        distances = [round(base * scale, 4) for scale in (1.0, 1.25, 1.55, 1.65, 1.9, 2.3)]
    min_d = scenario.get("min_stance_distance_m")
    max_d = scenario.get("max_stance_distance_m")
    if min_d is not None:
        distances = [d for d in distances if d >= float(min_d)]
    if max_d is not None:
        distances = [d for d in distances if d <= float(max_d)]
    preferred = scenario.get("preferred_stance_distance_m")
    if preferred is not None:
        pref = float(preferred)
        distances.sort(key=lambda d: (abs(d - pref), d))
    return distances


def task_stance_target_for_scenario(
    scenario: Mapping[str, Any],
    manipulation_look_at=None,
    *,
    allow_navigation_target_fallback: bool = True,
) -> tuple[float, float, float] | None:
    if manipulation_look_at is not None:
        return _optional_xyz(manipulation_look_at)
    for key in TASK_STANCE_TARGET_KEYS:
        target = _optional_xyz(scenario.get(key))
        if target is not None:
            return target
    if allow_navigation_target_fallback:
        for key in TASK_STANCE_FALLBACK_TARGET_KEYS:
            target = _optional_xyz(scenario.get(key))
            if target is not None:
                return target
    return None


def task_stance_target_object_ids_for_scenario(scenario: Mapping[str, Any]) -> list[str]:
    """Ordered target id/name aliases to search in the USD stage."""
    out: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        out.append(text)

    for key in TASK_STANCE_TARGET_OBJECT_KEYS:
        if scenario.get(key):
            add(scenario.get(key))
    for key in TASK_STANCE_TARGET_OBJECT_LIST_KEYS:
        values = scenario.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for value in values:
                add(value)
        elif values:
            add(values)
    return out


def task_stance_affordance_object_ids_for_scenario(scenario: Mapping[str, Any]) -> list[str]:
    """Ordered fine-grained manipulation affordance aliases to search in the USD stage."""
    out: list[str] = []
    seen: set[str] = set()
    for key in TASK_STANCE_AFFORDANCE_OBJECT_LIST_KEYS:
        values = scenario.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            iterable = values
        elif values:
            iterable = [values]
        else:
            iterable = []
        for value in iterable:
            text = str(value or "").strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return out


def plan_task_stance(
    *,
    scenario: Mapping[str, Any],
    manipulation_look_at=None,
    probe_collision=None,
    floor_z_hint: float | None = None,
    robot_footprint_half_extent: Sequence[float] | None = None,
    placement_validator=None,
) -> dict[str, Any]:
    """Select a task start pose around a target object without scene-specific coordinates.

    The target is the object/workspace to face, not the root pose. Candidate root poses are sampled
    around the target, biased toward the scenario's approach/start side, and accepted only when the
    provided scene collision probe reports no hits. When object bounds are available, stance
    distances are measured from the target footprint surface, not the center point; this prevents a
    robot from being accepted visually inside a sink/counter just because the pelvis center is a
    short radius from a small faucet or basin centroid.
    """
    if robot_footprint_half_extent is None:
        # Resolve at call time so apply_robot_profile() overrides reach the planner.
        robot_footprint_half_extent = ROBOT_FOOTPRINT_HALF_EXTENT
    target = task_stance_target_for_scenario(scenario, manipulation_look_at)
    if target is None:
        return {
            "schema_version": TASK_STANCE_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["missing_task_stance_target"],
            "candidates": [],
        }
    approach = None
    for key in TASK_STANCE_APPROACH_KEYS:
        approach = _optional_xyz(scenario.get(key))
        if approach is not None:
            break
    primary = _unit_xy_from(approach, target) if approach is not None else None
    if primary is None:
        primary = (-1.0, 0.0)
    primary_angle = math.atan2(primary[1], primary[0])
    target_bounds = _target_bounds_for_scenario(scenario)
    affordance_target = _task_stance_affordance_for_scenario(scenario)
    floor_z = float(
        floor_z_hint
        if floor_z_hint is not None
        else scenario.get("floor_z_hint", 0.0)
    )
    root_z = floor_z + ROBOT_PELVIS_HEIGHT_M
    distances = task_stance_distance_candidates(scenario)
    stance_focus = affordance_target or target
    stance_focus_source = "task_affordance_xyz" if affordance_target is not None else "task_target_xyz"
    if not distances:
        return {
            "schema_version": TASK_STANCE_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["empty_task_stance_distance_candidates"],
            "task_target_xyz": [round(float(v), 6) for v in target],
            "candidates": [],
        }
    probe = probe_collision or (lambda pose, yaw: 0)
    candidates: list[dict[str, Any]] = []
    accepted_candidate_indices: list[int] = []
    rejected_by_placement_validation = 0
    rejected_by_reachability = 0
    for distance_m in distances:
        for offset_deg in TASK_STANCE_ANGLE_OFFSETS_DEG:
            angle = primary_angle + math.radians(float(offset_deg))
            ux, uy = math.cos(angle), math.sin(angle)
            target_surface_offset = (
                _surface_offset_from_focus_along_bounds(
                    target_bounds,
                    stance_focus,
                    (ux, uy),
                )
                if affordance_target is not None
                else _half_extent_along_bounds(target_bounds, (ux, uy))
            )
            center_distance_m = target_surface_offset + float(distance_m)
            pose = (
                float(stance_focus[0]) + ux * center_distance_m,
                float(stance_focus[1]) + uy * center_distance_m,
                root_z,
            )
            yaw = math.atan2(float(stance_focus[1]) - pose[1], float(stance_focus[0]) - pose[0])
            collision_count = int(probe(pose, yaw))
            record = {
                "candidate_kind": "task_stance",
                "pose": [round(float(v), 6) for v in pose],
                "yaw": round(float(yaw), 6),
                "distance_to_target_m": round(float(center_distance_m), 6),
                "standoff_from_target_surface_m": round(float(distance_m), 6),
                "target_surface_offset_m": round(float(target_surface_offset), 6),
                "stance_focus_xyz": [round(float(v), 6) for v in stance_focus],
                "stance_focus_source": stance_focus_source,
                "angle_offset_deg": int(offset_deg),
                "approach_bias_enabled": bool(approach is not None),
                "scene_collision_contact_count": collision_count,
            }
            reachability_estimate = _task_stance_reachability_estimate(
                pose,
                yaw,
                affordance_target,
            )
            if reachability_estimate is not None:
                record["reachability_estimate"] = reachability_estimate
            candidates.append(record)
            if collision_count == 0:
                if placement_validator is not None:
                    try:
                        validation = placement_validator(pose, yaw, record)
                    except Exception as exc:  # noqa: BLE001 - fail closed on validator failures
                        validation = {
                            "status": "blocked",
                            "blockers": ["placement_validator_error"],
                            "error": repr(exc),
                        }
                    record["placement_validation"] = validation
                    if not _placement_validation_passed(validation):
                        rejected_by_placement_validation += 1
                        continue
                    corrected_root_pose = _placement_corrected_root_pose(validation)
                    if reachability_estimate is not None and corrected_root_pose is not None:
                        corrected_reachability = _task_stance_reachability_estimate(
                            corrected_root_pose,
                            yaw,
                            affordance_target,
                        )
                        if corrected_reachability is not None:
                            record["pre_placement_reachability_estimate"] = reachability_estimate
                            corrected_reachability["pose_source"] = (
                                "placement_corrected_root_translation_xyz"
                            )
                            corrected_reachability["pre_placement_reachability_status"] = (
                                reachability_estimate.get("status")
                            )
                            record["reachability_estimate"] = corrected_reachability
                            record["placement_corrected_root_pose"] = [
                                round(float(v), 6) for v in corrected_root_pose
                            ]
                            reachability_estimate = corrected_reachability
                if (
                    reachability_estimate is not None
                    and reachability_estimate.get("status") != "PASS"
                ):
                    rejected_by_reachability += 1
                    continue
                accepted_candidate_indices.append(len(candidates) - 1)
    if accepted_candidate_indices:
        selected_candidate_index = min(
            accepted_candidate_indices,
            key=lambda idx: (_task_stance_selection_key(candidates[idx]), idx),
        )
        record = candidates[selected_candidate_index]
        accepted = {
            "schema_version": TASK_STANCE_SCHEMA_VERSION,
            "status": "accepted",
            "task_target_xyz": [round(float(v), 6) for v in target],
            "stance_focus_xyz": [round(float(v), 6) for v in stance_focus],
            "stance_focus_source": stance_focus_source,
            "approach_point_xyz": (
                [round(float(v), 6) for v in approach] if approach is not None else None
            ),
            "robot_footprint_half_extent": [round(float(v), 6) for v in robot_footprint_half_extent],
            "floor_z_hint": round(floor_z, 6),
            "accepted_pose": record["pose"],
            "accepted_yaw": record["yaw"],
            "selected_candidate_index": selected_candidate_index,
            "accepted_candidate_count": len(accepted_candidate_indices),
            "placement_validation_rejected_candidate_count": rejected_by_placement_validation,
            "reachability_rejected_candidate_count": rejected_by_reachability,
            "stance_selection_key": [
                round(float(v), 6) for v in _task_stance_selection_key(record)
            ],
            "stance_selection_strategy": (
                "when a fine affordance is resolved, validated candidates are sorted by approximate "
                "G1 reachability before approach/standoff tie-breakers; otherwise by standoff "
                "distance when no real approach hint exists, or approach-angle bucket, standoff "
                "distance, then absolute angle offset"
            ),
            "candidates": candidates,
            "claim_boundary": (
                "Task stance is selected from scene collision probes around the task target. "
                "It is placement evidence, not full dynamic locomotion or manipulation success."
            ),
        }
        if affordance_target is not None:
            accepted["task_affordance_xyz"] = [round(float(v), 6) for v in affordance_target]
            accepted["reachability_selection_enabled"] = True
            accepted["reach_seed_gate"] = {
                "status": "PASS",
                "requirement": (
                    "At least one approximate G1 manipulation arm seed can plausibly reach "
                    "the resolved task affordance from the selected base pose."
                ),
                "selected_candidate_reachability": record.get("reachability_estimate"),
            }
        if target_bounds is not None:
            accepted["task_target_bounds"] = {
                "bbox_min_xyz": [round(float(v), 6) for v in target_bounds[0]],
                "bbox_max_xyz": [round(float(v), 6) for v in target_bounds[1]],
            }
        if "placement_validation" in record:
            accepted["placement_validation"] = record["placement_validation"]
        return accepted
    blocked = {
        "schema_version": TASK_STANCE_SCHEMA_VERSION,
        "status": "blocked",
        "blockers": [
            (
                "no_reach_seed_task_stance_candidate"
                if rejected_by_reachability
                else (
                    "no_validated_task_stance_candidate"
                    if rejected_by_placement_validation
                    else "no_collision_free_task_stance_candidate"
                )
            )
        ],
        "task_target_xyz": [round(float(v), 6) for v in target],
        "stance_focus_xyz": [round(float(v), 6) for v in stance_focus],
        "stance_focus_source": stance_focus_source,
        "approach_point_xyz": [round(float(v), 6) for v in approach] if approach is not None else None,
        "robot_footprint_half_extent": [round(float(v), 6) for v in robot_footprint_half_extent],
        "floor_z_hint": round(floor_z, 6),
        "candidates": candidates,
        "placement_validation_rejected_candidate_count": rejected_by_placement_validation,
        "reachability_rejected_candidate_count": rejected_by_reachability,
    }
    if affordance_target is not None:
        blocked["task_affordance_xyz"] = [round(float(v), 6) for v in affordance_target]
        blocked["reachability_selection_enabled"] = True
    if target_bounds is not None:
        blocked["task_target_bounds"] = {
            "bbox_min_xyz": [round(float(v), 6) for v in target_bounds[0]],
            "bbox_max_xyz": [round(float(v), 6) for v in target_bounds[1]],
        }
    return blocked


def assemble_collision_summary(*, actions: Sequence[Mapping[str, Any]],
                               rejected_probe_total: int, response_event_total: int) -> dict:
    """Build the collision_summary that compute_task_outcome consumes from the per-step trace."""
    committed = sum(int(a.get("scene_collision_contact_count") or 0) for a in actions)
    return {
        "robot_scene_contact_event_count": committed,
        "rejected_scene_collision_probe_count": int(rejected_probe_total),
        "near_miss_event_count": int(rejected_probe_total),
        "collision_response_event_count": int(response_event_total),
        "clearance_threshold_m": policy_mod.TASK_CLEARANCE_THRESHOLD_M,
    }


def mp4_command(frames_glob: str, fps: int, out_path: str) -> list[str]:
    """ffmpeg command to assemble numbered PNG frames into an MP4 (yuv420p, web-playable)."""
    return ["ffmpeg", "-y", "-framerate", str(fps), "-pattern_type", "glob", "-i", frames_glob,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]


def _status_passed(value: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(value, Mapping)
        and str(value.get("status") or "").strip().upper() in {"PASS", "PASSED"}
    )


def _reach_feasibility_passed(value: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(value, Mapping)
        and str(value.get("status") or "").strip().upper() in {"PASS", "PASSED"}
    )


def _reach_feasibility_blockers(value: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    blockers: list[str] = []
    for key in ("blockers", "all_arm_blockers"):
        blockers.extend(str(b) for b in (value.get(key) or []) if b)
    by_arm = value.get("by_arm")
    if isinstance(by_arm, Mapping):
        for arm_report in by_arm.values():
            if isinstance(arm_report, Mapping):
                blockers.extend(str(b) for b in (arm_report.get("blockers") or []) if b)
    return blockers


def _nearest_effector_distance_to_affordance_m(value: Mapping[str, Any] | None) -> float | None:
    if not isinstance(value, Mapping):
        return None
    distances: list[float] = []

    def collect(mapping: Mapping[str, Any] | None) -> None:
        if not isinstance(mapping, Mapping):
            return
        for raw in mapping.values():
            try:
                distances.append(float(raw))
            except Exception:  # noqa: BLE001
                pass

    collect(value.get("effector_distance_to_affordance_m"))
    by_arm = value.get("effector_distance_to_affordance_m_by_arm")
    if isinstance(by_arm, Mapping):
        for arm_distances in by_arm.values():
            collect(arm_distances if isinstance(arm_distances, Mapping) else None)
    return min(distances) if distances else None


def _pov_geometry_reach_feasibility_evidence(
    pov_geometry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    frames_checked = 0
    passing_frame_indices: list[int] = []
    final_frame_index: int | None = None
    final_frame_passed: bool | None = None
    final_frame_nearest_effector_m: float | None = None
    final_frame_close_enough: bool | None = None

    if not isinstance(pov_geometry, Mapping):
        blockers.append("visible_reach_pov_geometry_missing")
    else:
        frames = pov_geometry.get("frames")
        if isinstance(frames, Sequence) and not isinstance(frames, (str, bytes, bytearray)):
            frame_rows = [
                (idx, frame)
                for idx, frame in enumerate(frames)
                if isinstance(frame, Mapping)
            ]
            frames_checked = len(frame_rows)
            for idx, frame in frame_rows:
                reach = (
                    frame.get("reach_feasibility")
                    if isinstance(frame.get("reach_feasibility"), Mapping)
                    else None
                )
                if _reach_feasibility_passed(reach):
                    passing_frame_indices.append(idx)
            if frame_rows:
                final_frame_index, final_frame = frame_rows[-1]
                final_reach = (
                    final_frame.get("reach_feasibility")
                    if isinstance(final_frame.get("reach_feasibility"), Mapping)
                    else None
                )
                final_frame_passed = _reach_feasibility_passed(final_reach)
                final_frame_nearest_effector_m = _nearest_effector_distance_to_affordance_m(
                    final_frame
                )
                if not final_frame_passed:
                    blockers.append("visible_reach_final_frame_reach_feasibility_not_passed")
                    blockers.extend(_reach_feasibility_blockers(final_reach))
                if final_frame_nearest_effector_m is None:
                    blockers.append("visible_reach_final_frame_effector_distance_missing")
                    final_frame_close_enough = False
                else:
                    final_frame_close_enough = (
                        final_frame_nearest_effector_m
                        <= float(VISIBLE_REACH_FINAL_MAX_EFFECTOR_TO_AFFORDANCE_M)
                    )
                    if not final_frame_close_enough:
                        blockers.append(
                            "visible_reach_final_frame_effector_not_close_enough"
                        )
            else:
                blockers.append("visible_reach_reach_feasibility_frames_missing")
        else:
            reach = (
                pov_geometry.get("reach_feasibility")
                if isinstance(pov_geometry.get("reach_feasibility"), Mapping)
                else None
            )
            final_frame_passed = _reach_feasibility_passed(reach)
            final_frame_nearest_effector_m = _nearest_effector_distance_to_affordance_m(
                pov_geometry
            )
            if not final_frame_passed:
                blockers.append("visible_reach_reach_feasibility_not_passed")
                blockers.extend(_reach_feasibility_blockers(reach))
            if final_frame_nearest_effector_m is None:
                blockers.append("visible_reach_final_frame_effector_distance_missing")
                final_frame_close_enough = False
            else:
                final_frame_close_enough = (
                    final_frame_nearest_effector_m
                    <= float(VISIBLE_REACH_FINAL_MAX_EFFECTOR_TO_AFFORDANCE_M)
                )
                if not final_frame_close_enough:
                    blockers.append("visible_reach_final_frame_effector_not_close_enough")

    if not passing_frame_indices and frames_checked:
        blockers.append("visible_reach_no_frame_with_passing_reach_feasibility")

    blockers = sorted({str(b) for b in blockers if b})
    passed = not blockers
    return {
        "schema_version": "visible_reach_feasibility_evidence.v1",
        "status": "PASS" if passed else "FAIL",
        "blockers": blockers,
        "frames_checked": frames_checked,
        "passing_frame_count": len(passing_frame_indices),
        "passing_frame_indices": passing_frame_indices[:20],
        "final_frame_index": final_frame_index,
        "final_frame_reach_feasibility_passed": final_frame_passed,
        "final_frame_nearest_effector_to_affordance_m": (
            round(float(final_frame_nearest_effector_m), 4)
            if final_frame_nearest_effector_m is not None
            else None
        ),
        "max_final_effector_to_affordance_m": round(
            float(VISIBLE_REACH_FINAL_MAX_EFFECTOR_TO_AFFORDANCE_M),
            4,
        ),
        "final_frame_effector_close_enough": final_frame_close_enough,
        "requirement": (
            "Visible reach success requires the final sampled manipulation frame to have passing "
            "reach feasibility and a hand/wrist close enough to the task affordance to be visually "
            "credible. Arm visibility/framing alone is not enough."
        ),
    }


def _review_grade_task_success_evidence(outcome: Mapping[str, Any]) -> dict[str, Any]:
    """Separate trace success from review-grade visible robot task success."""
    blockers: list[str] = []
    trace_task_success = bool(outcome.get("task_success"))
    if not trace_task_success:
        blockers.append("trace_task_success_false")

    camera_evidence = (
        outcome.get("review_camera_evidence")
        if isinstance(outcome.get("review_camera_evidence"), Mapping)
        else {}
    )
    if not camera_evidence:
        blockers.append("review_camera_evidence_missing")

    camera_mode = str(camera_evidence.get("robot_pov_camera_mode") or "").strip()
    if camera_mode == "root_follow":
        blockers.append("robot_pov_is_root_follow_camera_not_head_pov")
    elif camera_mode and camera_mode != "robot_mounted_manipulation":
        blockers.append(f"robot_pov_camera_mode_not_review_grade:{camera_mode}")

    if camera_evidence.get("visible_embodied_robot_action_evidence") is not True:
        blockers.append("visible_embodied_robot_action_not_proven")

    robot_visual = (
        outcome.get("robot_visual_geometry")
        if isinstance(outcome.get("robot_visual_geometry"), Mapping)
        else {}
    )
    if robot_visual and not _status_passed(robot_visual):
        blockers.extend(str(b) for b in (robot_visual.get("blockers") or []))
        blockers.append("robot_visual_geometry_not_review_ready")

    manipulation_pov = (
        outcome.get("manipulation_pov_geometry")
        if isinstance(outcome.get("manipulation_pov_geometry"), Mapping)
        else {}
    )
    if manipulation_pov and not _status_passed(manipulation_pov):
        blockers.extend(str(b) for b in (manipulation_pov.get("blockers") or []))
        blockers.append("manipulation_pov_geometry_not_review_ready")

    if str(outcome.get("task_success_contract") or "").strip().lower() == "visible_reach_to_affordance":
        reach_contract = (
            outcome.get("visible_reach_to_affordance_success")
            if isinstance(outcome.get("visible_reach_to_affordance_success"), Mapping)
            else {}
        )
        if not _status_passed(reach_contract):
            blockers.append("visible_reach_success_contract_not_passed")
            blockers.extend(str(b) for b in (reach_contract.get("blockers") or []))

    blockers = sorted({str(b) for b in blockers if b})
    review_task_success = bool(trace_task_success and not blockers)
    return {
        "schema_version": REVIEW_TASK_SUCCESS_EVIDENCE_SCHEMA_VERSION,
        "status": "PASS" if review_task_success else "FAIL",
        "review_task_success": review_task_success,
        "trace_task_success": trace_task_success,
        "blockers": blockers,
        "camera_evidence": dict(camera_evidence),
        "claim_boundary": (
            "Review-grade task success requires both the internal trace success and visible embodied "
            "robot-action evidence in the review media. A root-position trace or camera-only motion "
            "does not prove review-grade task success."
        ),
    }


def _scenario_task_success_contract(scenario: Mapping[str, Any]) -> str:
    return (
        str(
            scenario.get("task_success_contract")
            or scenario.get("success_contract")
            or "root_navigation_to_target"
        )
        .strip()
        .lower()
        or "root_navigation_to_target"
    )


def _apply_visible_reach_to_affordance_success_contract(
    outcome: dict[str, Any],
    *,
    placement_validation: Mapping[str, Any] | None,
    pov_geometry: Mapping[str, Any] | None,
    robot_visual_ready: bool,
    temporal_conditioning: Mapping[str, Any] | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    reach_feasibility_evidence = _pov_geometry_reach_feasibility_evidence(pov_geometry)
    if not _status_passed(placement_validation):
        blockers.append("visible_reach_placement_validation_not_passed")
        if isinstance(placement_validation, Mapping):
            blockers.extend(str(b) for b in (placement_validation.get("blockers") or []))
    if not _status_passed(pov_geometry):
        blockers.append("visible_reach_pov_geometry_not_passed")
        if isinstance(pov_geometry, Mapping):
            blockers.extend(str(b) for b in (pov_geometry.get("blockers") or []))
    if not robot_visual_ready:
        blockers.append("visible_reach_robot_visual_geometry_not_review_ready")
    if not _status_passed(temporal_conditioning):
        blockers.append("visible_reach_temporal_conditioning_not_passed")
        if isinstance(temporal_conditioning, Mapping):
            blockers.extend(str(b) for b in (temporal_conditioning.get("blockers") or []))
    if not _status_passed(reach_feasibility_evidence):
        blockers.append("visible_reach_reach_feasibility_not_passed")
        blockers.extend(str(b) for b in (reach_feasibility_evidence.get("blockers") or []))
    blockers = sorted({str(b) for b in blockers if b})
    passed = not blockers
    contract_result = {
        "schema_version": "visible_reach_to_affordance_success_contract.v1",
        "status": "PASS" if passed else "FAIL",
        "task_success_contract": "visible_reach_to_affordance",
        "blockers": blockers,
        "required_evidence": {
            "placement_validation_passed": _status_passed(placement_validation),
            "manipulation_pov_geometry_passed": _status_passed(pov_geometry),
            "robot_visual_geometry_review_ready": bool(robot_visual_ready),
            "temporal_reach_conditioning_passed": _status_passed(temporal_conditioning),
            "reach_feasibility_passed": _status_passed(reach_feasibility_evidence),
        },
        "reach_feasibility_evidence": reach_feasibility_evidence,
        "claim_boundary": (
            "This contract proves a visible reach-toward-affordance review task only. It does not "
            "prove faucet state change, contact force, water flow, physical reach, learned policy "
            "quality, or deployment readiness."
        ),
    }
    outcome["visible_reach_to_affordance_success"] = contract_result
    outcome["task_success_contract"] = "visible_reach_to_affordance"
    if passed:
        outcome["task_success"] = True
        outcome["task_status"] = "passed"
        outcome["failure_mode_ids"] = []
        outcome["failure_reason"] = None
        outcome["goal_reached"] = True
        outcome["endpoint_clean"] = True
    else:
        outcome["task_success"] = False
        outcome["task_status"] = "failed_task_criteria"
        outcome["failure_mode_ids"] = blockers
        outcome["failure_reason"] = ",".join(blockers)
    return contract_result


def build_result(*, scenarios: Sequence[Mapping[str, Any]], outcomes: Sequence[Mapping[str, Any]],
                 policy_id: str, kitchen_usd: str, g1_usd: str | None,
                 blockers: Sequence[str],
                 physics_articulation_contact_reports: Sequence[Mapping[str, Any]] | None = None,
                 segmentation_summary: Mapping[str, Any] | None = None,
                 actuator_output_mode: str | None = None,
                 authored_target_contact_material: Any | None = None) -> dict:
    passed = sum(1 for o in outcomes if o.get("task_success"))
    scenario_rows: list[dict[str, Any]] = []
    review_grade_passed = 0
    for s, o in zip(scenarios, outcomes):
        review_evidence = _review_grade_task_success_evidence(o)
        if review_evidence["review_task_success"]:
            review_grade_passed += 1
        scenario_rows.append({
            "scenario_id": s.get("scenario_id"),
            **dict(o),
            "review_task_success": review_evidence["review_task_success"],
            "review_task_success_evidence": review_evidence,
        })
    result_blockers = list(blockers)
    if scenarios and not outcomes and not result_blockers:
        result_blockers.append("scenario_execution_returned_no_outcomes")
    status = "completed" if outcomes and not result_blockers else "blocked"
    contact_summary = summarize_physics_articulation_contact_reports(
        physics_articulation_contact_reports or []
    )
    proof_boundary = (
        "Isaac RTX-rendered kinematic walk-to-target preview (parity with the MuJoCo preview "
        "controller). Not dynamic locomotion, not a learned policy, not deployment readiness."
    )
    if contact_summary["all_have_support_contact_evidence"]:
        proof_boundary = (
            "Isaac RTX-rendered kinematic walk-to-target preview plus opt-in PhysX articulation "
            "standing/contact settle samples. This upgrades the standing placement evidence to "
            "physics-stepped support/contact evidence, but it is still not full dynamic locomotion, "
            "not a learned balance controller, and not deployment readiness."
        )
        if contact_summary.get("any_physics_integrated"):
            proof_boundary += (
                " The G1 root integrated under gravity during the settle "
                f"(max vertical drop {float(contact_summary.get('max_root_vertical_drop_m') or 0.0):.3f} m); "
                "this is Isaac physics evidence only and still not full dynamic locomotion, "
                "learned balance, or deployment readiness."
            )
    elif contact_summary["scenario_count"] > 0:
        proof_boundary = (
            "Isaac RTX-rendered kinematic walk-to-target preview plus opt-in PhysX articulation "
            "standing/contact settle samples. The physics settle completed, but support-contact "
            "events were not observed, so this does not prove support contact, full dynamic "
            "locomotion, learned balance control, or deployment readiness."
        )
    contact_reports = list(physics_articulation_contact_reports or [])
    reported_actuator_modes = [
        str(r.get("actuator_output_mode"))
        for r in contact_reports
        if r.get("actuator_output_mode")
    ]
    if actuator_output_mode is None:
        if "effort" in reported_actuator_modes:
            actuator_output_mode = "effort"
        elif "position_target_fallback" in reported_actuator_modes:
            actuator_output_mode = "position_target_fallback"
        else:
            actuator_output_mode = "position_target"
    if actuator_output_mode == "effort":
        proof_boundary += (
            " Effort-drive samples use a physics torque/effort PD drive ported from the MuJoCo "
            "law; this is not learned balance control, not task success proof, not "
            "MuJoCo-equivalent contact-force proof, and not safety validation."
        )
    camera_contract_frames = sum(
        int(o.get("per_frame_camera_contract_frames") or 0) for o in outcomes
    )
    camera_contract_intrinsics_frames = sum(
        int(o.get("per_frame_camera_contract_available_intrinsics_frames") or 0)
        for o in outcomes
    )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "policy_id": policy_id,
        "kitchen_usd": kitchen_usd,
        "g1_usd": g1_usd,
        "scenario_count": len(scenarios),
        "scenarios_executed": len(outcomes),
        "scenarios_passed": passed,
        "review_grade_scenarios_passed": review_grade_passed,
        "review_grade_success_claim_boundary": (
            "Use review_grade_scenarios_passed/review_task_success for human-showable task success. "
            "scenarios_passed/task_success is the internal trace outcome and can be true even when "
            "the media does not visibly show the robot performing the task."
        ),
        "rendered_by_isaac_rtx": True,
        "per_frame_camera_contract_emitted": camera_contract_frames > 0,
        "per_frame_camera_contract_frames": camera_contract_frames,
        "per_frame_camera_contract_available_intrinsics_frames": camera_contract_intrinsics_frames,
        "actuator_output_mode": actuator_output_mode,
        "blockers": result_blockers,
        "scenarios": scenario_rows,
        "proof_boundary": proof_boundary,
    }
    if authored_target_contact_material is not None:
        result["authored_target_contact_material"] = authored_target_contact_material
    if contact_summary["scenario_count"] > 0:
        result["physics_articulation_standing_contact_summary"] = contact_summary
        result["physics_articulation_standing_contact_reports"] = [
            dict(report) for report in contact_reports
        ]
    if segmentation_summary is not None:
        seg_summary = dict(segmentation_summary)
        labeled_prim_count = int(seg_summary.get("labeled_prim_count") or 0)
        instance_mask_frames = int(seg_summary.get("instance_mask_frames") or 0)
        result["segmentation_pass"] = {
            "schema_version": "isaac_g1_kitchen_parity_segmentation.v1",
            "simulator_backend": "isaac_replicator",
            "native_segmentation_proven": bool(
                labeled_prim_count > 0 and instance_mask_frames > 0
            ),
            "labeled_prim_count": labeled_prim_count,
            "instance_mask_frames": instance_mask_frames,
            "semantic_mask_frames": int(seg_summary.get("semantic_mask_frames") or 0),
            "id_label_path": seg_summary.get("id_label_path"),
            "sample_labels": list(seg_summary.get("sample_labels") or [])[:40],
            "blockers": list(seg_summary.get("blockers") or []),
            "source": "replicator_instance_semantic_annotator",
        }
    return result


def summarize_physics_articulation_contact_reports(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scenario_count = len(reports)
    completed = [r for r in reports if r.get("status") == "completed"]
    contact_records = sum(int(r.get("contact_event_count") or 0) for r in reports)
    support_records = sum(int(r.get("support_contact_event_count") or 0) for r in reports)
    any_physics_integrated = bool(
        scenario_count and all(bool(r.get("physics_integrated")) for r in reports)
    )
    gravity_on_all = bool(scenario_count and all(bool(r.get("gravity_on")) for r in reports))
    max_root_vertical_drop_m = max(
        (float(r.get("root_vertical_drop_m") or 0.0) for r in reports),
        default=0.0,
    )
    verdict_counts: dict[str, int] = {}
    for report in reports:
        verdict = str(report.get("dynamic_settle_verdict") or "unknown")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    return {
        "scenario_count": scenario_count,
        "completed_scenario_count": len(completed),
        "contact_event_count": contact_records,
        "support_contact_event_count": support_records,
        "any_physics_integrated": any_physics_integrated,
        "gravity_on_all": gravity_on_all,
        "max_root_vertical_drop_m": round(float(max_root_vertical_drop_m), 6),
        "verdict_counts": verdict_counts,
        "all_completed": bool(scenario_count and len(completed) == scenario_count),
        "all_have_support_contact_evidence": bool(
            scenario_count and all(int(r.get("support_contact_event_count") or 0) > 0 for r in reports)
        ),
        "root_pose_teleport_during_physics_settle": any(
            bool(r.get("root_pose_teleport_during_physics_settle")) for r in reports
        ),
        "claim_boundary": (
            "PhysX articulation standing/contact settle evidence only. This does not prove full "
            "dynamic locomotion, learned balance, task manipulation success, or deployment readiness."
        ),
    }


def upload_zip(out_dir: Path, put_url: str | None) -> int | None:
    if not put_url:
        return None
    import urllib.request
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(out_dir).as_posix())
    req = urllib.request.Request(put_url, data=buf.getvalue(), method="PUT",
                                 headers={"Content-Type": "application/zip"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return int(getattr(r, "status", 200))


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """(w, x, y, z) for a rotation about +Z."""
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def _norm(v):
    m = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) or 1.0
    return (v[0] / m, v[1] / m, v[2] / m)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)) -> tuple[float, float, float, float]:
    """USD-camera look-at orientation as (w, x, y, z). The camera views along its local -Z with
    +Y up; we build the basis [x, y, z] with z = -forward and convert to a quaternion."""
    forward = _norm((target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]))
    zc = (-forward[0], -forward[1], -forward[2])            # camera local +Z (out of screen)
    xc = _norm(_cross(up, zc))
    if xc == (0.0, 0.0, 0.0):                               # up parallel to view dir
        xc = _norm(_cross((0.0, 1.0, 0.0), zc))
    yc = _cross(zc, xc)
    m00, m01, m02 = xc[0], yc[0], zc[0]
    m10, m11, m12 = xc[1], yc[1], zc[1]
    m20, m21, m22 = xc[2], yc[2], zc[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return (w, x, y, z)


def project_point_to_pixel(world_pt, eye, target, up, vfov_deg: float, width: int, height: int):
    """Pinhole-project a world point into the camera image. Returns (u, v, depth) in pixels if the
    point is in front of the camera and within frame, else None. Used to build the G1 skeleton
    landmarks (joint world positions -> 2D image landmarks) for OSCAR conditioning."""
    fwd = _norm((target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]))
    right = _norm(_cross(fwd, up))
    if right == (0.0, 0.0, 0.0):
        right = _norm(_cross(fwd, (0.0, 1.0, 0.0)))
    tup = _cross(right, fwd)
    rel = (world_pt[0] - eye[0], world_pt[1] - eye[1], world_pt[2] - eye[2])
    z = rel[0] * fwd[0] + rel[1] * fwd[1] + rel[2] * fwd[2]
    if z <= 1e-6:
        return None
    x = rel[0] * right[0] + rel[1] * right[1] + rel[2] * right[2]
    y = rel[0] * tup[0] + rel[1] * tup[1] + rel[2] * tup[2]
    f = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    u = width / 2.0 + f * (x / z)
    v = height / 2.0 - f * (y / z)
    if 0.0 <= u < width and 0.0 <= v < height:
        return (u, v, z)
    return None


def scene_framing(scenarios: Sequence[Mapping[str, Any]]) -> tuple[tuple[float, float, float], float]:
    """Center + radius of all scenario route points, for the static overview camera."""
    pts = [p for sc in scenarios for p in sc.get("route_points", [])]
    if not pts:
        return (0.0, 0.0, ROBOT_PELVIS_HEIGHT_M), 4.0
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    radius = max((math.hypot(p[0] - cx, p[1] - cy) for p in pts), default=2.0)
    return (cx, cy, ROBOT_PELVIS_HEIGHT_M), max(2.5, radius)


def _materialize_deferred_task_route(
    scenario: dict[str, Any],
    *,
    stance_plan: Mapping[str, Any],
    root_pose: Sequence[float],
    look_at: Sequence[float] | None,
) -> None:
    """Fill legacy navigation fields from the dynamically resolved manipulation stance.

    Task-only scenarios intentionally enter without scene coordinates. Once USD/scene-placement has
    resolved the target and accepted a clear stance, the legacy policy/outcome code can use this
    dynamically-derived route without any placeholder coordinates becoming evidence.
    """
    target = _optional_xyz(look_at) if look_at is not None else None
    if target is None:
        target = _optional_xyz(stance_plan.get("task_target_xyz"))
    if target is None:
        return
    root = [float(root_pose[0]), float(root_pose[1]), float(root_pose[2])]
    nav_target = [float(target[0]), float(target[1]), ROBOT_PELVIS_HEIGHT_M]
    floor_z = float(root[2]) - ROBOT_PELVIS_HEIGHT_M
    scenario["start"] = root
    scenario["target"] = nav_target
    scenario["route_points"] = [root, nav_target]
    scenario["raw_spawn_position_xyz"] = [root[0], root[1], floor_z]
    scenario["raw_target_position_xyz"] = [float(target[0]), float(target[1]), float(target[2])]
    scenario["task_target_deferred"] = False
    scenario["deferred_task_resolution"] = "materialized_from_task_stance_plan"


def follow_cam_pose(root_pose, yaw, *, back: float = 2.2, up: float = 1.6):
    """Eye + target for a robot-POV follow camera: behind and above the root, looking ahead."""
    fx, fy = math.cos(yaw), math.sin(yaw)
    eye = (root_pose[0] - fx * back, root_pose[1] - fy * back, root_pose[2] + up)
    target = (root_pose[0] + fx * 1.5, root_pose[1] + fy * 1.5, root_pose[2] + 0.2)
    return eye, target


def verify_cam_pose(root_pose, yaw, *, back: float = 1.4, up: float = 1.1, side: float = -0.8,
                    look_at=None):
    """3rd-person VERIFICATION camera: pulled back behind + above + to the side so the WHOLE robot AND
    the workspace it faces are both in frame — proves where the robot is actually standing (vs the
    egocentric POV, which shows only what the robot looks at)."""
    fx, fy = math.cos(yaw), math.sin(yaw)
    if look_at is not None:
        try:
            dx = float(look_at[0]) - float(root_pose[0])
            dy = float(look_at[1]) - float(root_pose[1])
            d = math.hypot(dx, dy)
            if d > 1e-4:
                fx, fy = dx / d, dy / d
        except Exception:  # noqa: BLE001
            pass
    px, py = -fy, fx  # perpendicular to facing for a 3/4 angle that reveals body-vs-target gap
    eye = (root_pose[0] - fx * back + px * side, root_pose[1] - fy * back + py * side, root_pose[2] + up)
    target = (root_pose[0] + fx * 0.45, root_pose[1] + fy * 0.45, root_pose[2] + 0.25)  # robot torso/front
    return eye, target


def manipulation_cam_pose(
    root_pose,
    yaw,
    *,
    eye_forward: float = 0.12,
    eye_height: float = 1.35,
    target_forward: float = 0.6,
    target_height: float = 0.9,
    look_at=None,
    shoulder_side: float = 0.38,
    reach_arm: str = "right",
):
    """Eye + target for an EGOCENTRIC manipulation POV: from the robot's head, looking forward
    at the task workspace directly in front of the robot and between the arms.

    Unlike ``follow_cam_pose`` (a chase shot behind+above, framing the whole robot walking across the
    room) this frames the local task region. Heights are absolute so the view sits at head level and
    looks at manipulation height — the in-distribution, coherent view a manipulation WAM can actually
    predict, instead of a room-scale navigation scene it collapses to blur on.

    ``look_at`` (a fixed world x,y,z — e.g. the affordance/handle surface) pins the target so the
    workspace stays centered regardless of the policy's noisy final yaw; without it the target is
    derived yaw-relative (forward of the robot)."""
    fx, fy = math.cos(yaw), math.sin(yaw)
    if look_at is not None:
        affordance = (float(look_at[0]), float(look_at[1]), float(look_at[2]))
        rx, ry = fy, -fx
        arm_norm = str(reach_arm or "right").strip().lower()
        if arm_norm == "left":
            active_side = -1.0
        elif arm_norm == "both":
            active_side = 0.0
        else:
            active_side = 1.0
        # Keep the camera at a head-mounted seed pose. The target still blends slightly toward the
        # active shoulder so the hand/forearm remain in the policy seed instead of a handle-only crop.
        shoulder_hint = (
            root_pose[0] + rx * float(shoulder_side) * active_side * 0.65,
            root_pose[1] + ry * float(shoulder_side) * active_side * 0.65,
            min(max(float(eye_height) - 0.10, affordance[2] + 0.04), affordance[2] + 0.22),
        )
        target_blend = 0.30 if active_side else 0.14
        target = (
            affordance[0] * (1.0 - target_blend) + shoulder_hint[0] * target_blend,
            affordance[1] * (1.0 - target_blend) + shoulder_hint[1] * target_blend,
            affordance[2] * (1.0 - target_blend) + shoulder_hint[2] * target_blend,
        )
        eye = (
            root_pose[0] + fx * float(eye_forward),
            root_pose[1] + fy * float(eye_forward),
            max(float(eye_height), affordance[2] + 0.32),
        )
    else:
        eye = (root_pose[0] + fx * eye_forward, root_pose[1] + fy * eye_forward, eye_height)
        target = (root_pose[0] + fx * target_forward, root_pose[1] + fy * target_forward, target_height)
    return eye, target


def _weighted_xyz(points: Sequence[tuple[Sequence[float], float]]) -> tuple[float, float, float] | None:
    total = 0.0
    acc = [0.0, 0.0, 0.0]
    for xyz, weight in points:
        try:
            w = float(weight)
            vals = [float(xyz[i]) for i in range(3)]
        except Exception:  # noqa: BLE001
            continue
        if w <= 0.0:
            continue
        total += w
        for i in range(3):
            acc[i] += vals[i] * w
    if total <= 0.0:
        return None
    return (acc[0] / total, acc[1] / total, acc[2] / total)


def _manipulation_camera_target_with_arm_context(
    affordance,
    arm_points: Mapping[str, Sequence[float]] | None,
) -> tuple[float, float, float]:
    """Aim between the affordance and the active arm, not only at the handle.

    A pure handle-centered ray can crop the forearm out of an egocentric frame. This target remains
    task anchored while pulling the camera aim toward the wrist/elbow chain when those robot links are
    available from the USD asset.
    """
    aff = (float(affordance[0]), float(affordance[1]), float(affordance[2]))
    pts: list[tuple[Sequence[float], float]] = [(aff, 0.38)]
    if arm_points:
        if arm_points.get("hand") is not None:
            pts.append((arm_points["hand"], 0.26))
        if arm_points.get("wrist") is not None:
            pts.append((arm_points["wrist"], 0.24))
        if arm_points.get("elbow") is not None:
            pts.append((arm_points["elbow"], 0.12))
    return _weighted_xyz(pts) or aff


def _camera_pitch_down_deg(eye, target) -> float:
    horizontal_to_target_m = math.hypot(
        float(target[0]) - float(eye[0]),
        float(target[1]) - float(eye[1]),
    )
    return math.degrees(math.atan2(
        max(0.0, float(eye[2]) - float(target[2])),
        max(horizontal_to_target_m, 1e-6),
    ))


def _target_raised_to_max_pitch_down(eye, target, max_pitch_down_deg: float) -> tuple[float, float, float]:
    """Raise a look-at point just enough to keep a robot-head seed from becoming a down-looking crop."""
    tgt = (float(target[0]), float(target[1]), float(target[2]))
    horizontal_to_target_m = math.hypot(tgt[0] - float(eye[0]), tgt[1] - float(eye[1]))
    if horizontal_to_target_m <= 1e-6:
        return tgt
    min_target_z = float(eye[2]) - math.tan(math.radians(float(max_pitch_down_deg))) * horizontal_to_target_m
    if tgt[2] >= min_target_z:
        return tgt
    return (tgt[0], tgt[1], min_target_z)


def _unit_xy_from_yaw(yaw: float | None) -> tuple[float, float] | None:
    if yaw is None:
        return None
    try:
        fx = math.cos(float(yaw))
        fy = math.sin(float(yaw))
    except Exception:  # noqa: BLE001
        return None
    norm = math.hypot(fx, fy)
    if norm <= 1e-9:
        return None
    return (fx / norm, fy / norm)


def _unit_xy_from_points(origin, target) -> tuple[float, float] | None:
    try:
        dx = float(target[0]) - float(origin[0])
        dy = float(target[1]) - float(origin[1])
    except Exception:  # noqa: BLE001
        return None
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return None
    return (dx / norm, dy / norm)


def _manipulation_seed_arm_target_for_shoulder(
    shoulder,
    affordance,
    *,
    forward_yaw: float | None = None,
    forward_xy: Sequence[float] | None = None,
) -> tuple[float, float, float]:
    """Forward-ready arm seed target for the initial policy/WAM observation.

    This is intentionally not a contact/reach target. The resolved affordance remains useful for the
    camera and geometry report, but the initial robot pose should only show both arms held forward
    from their own shoulders so the policy/WAM evaluator can produce the action. If the caller does
    not provide a robot yaw or forward vector, use a fixed +x seed rather than aiming the seed at the
    affordance.
    """
    shoulder_z = float(shoulder[2])
    direction: tuple[float, float] | None = None
    if forward_xy is not None:
        try:
            fx = float(forward_xy[0])
            fy = float(forward_xy[1])
            norm = math.hypot(fx, fy)
            if norm > 1e-9:
                direction = (fx / norm, fy / norm)
        except Exception:  # noqa: BLE001
            direction = None
    if direction is None:
        direction = _unit_xy_from_yaw(forward_yaw)
    if direction is None:
        direction = (1.0, 0.0)
    affordance_z = float(affordance[2])
    if (
        affordance_z - shoulder_z
        >= float(MANIPULATION_HIGH_REACH_MIN_AFFORDANCE_ABOVE_SHOULDER_M)
    ):
        forward_seed_z = min(
            shoulder_z + float(MANIPULATION_HIGH_REACH_MAX_SEED_Z_ABOVE_SHOULDER_M),
            shoulder_z
            + (affordance_z - shoulder_z)
            * float(MANIPULATION_HIGH_REACH_SEED_HEIGHT_FRACTION),
        )
    else:
        forward_seed_z = shoulder_z - max(0.04, min(0.10, float(ROBOT_FOOTPRINT_HALF_EXTENT[2]) * 0.12))
    forward_distance_m = max(0.32, min(0.48, float(G1_APPROX_ARM_SPAN_M)))
    return (
        float(shoulder[0]) + direction[0] * forward_distance_m,
        float(shoulder[1]) + direction[1] * forward_distance_m,
        forward_seed_z,
    )


def _manipulation_arm_target_for_reach_fraction(
    shoulder,
    affordance,
    reach_frac: float,
    *,
    forward_yaw: float | None = None,
    forward_xy: Sequence[float] | None = None,
) -> tuple[float, float, float]:
    """Return the generic arm target for the current reach phase.

    Early frames stay in the forward-ready seed posture used for policy conditioning. Near the end of
    a review rollout, the target blends to the resolved task affordance so the final sampled frame can
    be judged as a visible endpoint attempt instead of just a plausible arm seed.
    """
    seed_target = _manipulation_seed_arm_target_for_shoulder(
        shoulder,
        affordance,
        forward_yaw=forward_yaw,
        forward_xy=forward_xy,
    )
    try:
        aff = (float(affordance[0]), float(affordance[1]), float(affordance[2]))
    except Exception:  # noqa: BLE001
        return seed_target
    frac = max(0.0, min(1.0, float(reach_frac)))
    start = max(0.0, min(0.99, float(MANIPULATION_ENDPOINT_AFFORDANCE_AIM_START_FRACTION)))
    if frac <= start:
        return seed_target
    blend = (frac - start) / (1.0 - start)
    blend = blend * blend * (3.0 - 2.0 * blend)
    return (
        seed_target[0] * (1.0 - blend) + aff[0] * blend,
        seed_target[1] * (1.0 - blend) + aff[1] * blend,
        seed_target[2] * (1.0 - blend) + aff[2] * blend,
    )


def _projection_dict(px) -> dict[str, Any] | None:
    if px is None:
        return None
    return {
        "available": True,
        "u_px": round(float(px[0]), 2),
        "v_px": round(float(px[1]), 2),
        "depth_m": round(float(px[2]), 4),
    }


def _normalize_reach_arm_selection(arm: str) -> str:
    selection = str(arm or "right").strip().lower()
    return selection if selection in {"left", "right", "both"} else "right"


def _manipulation_seed_arm_visibility(
    *,
    available_roles: Sequence[str],
    roles_in_frame: set[str],
    useful_roles_in_frame: set[str],
) -> dict[str, Any]:
    """Validate initial seed arm visibility without overfitting to one link role name.

    For the WAM/policy seed we need a clear task POV with the hand/arm visible, not proof that a
    specific ``forearm``/``elbow`` role was projected in the useful band. A visible hand+wrist chain
    is sufficient arm evidence; a lone hand/fingertip is not.
    """
    available = {str(role) for role in available_roles}
    effector_roles = roles_in_frame.intersection({"hand", "wrist"})
    useful_effector_roles = useful_roles_in_frame.intersection({"hand", "wrist"})
    arm_chain_roles = roles_in_frame.intersection({"elbow", "wrist", "hand"})
    useful_arm_chain_roles = useful_roles_in_frame.intersection({"elbow", "wrist", "hand"})
    blockers: list[str] = []
    if not available:
        blockers.append("manipulation_pov_arm_links_unavailable")
    if not effector_roles:
        blockers.append("manipulation_pov_arm_not_in_frame")
    elif not useful_effector_roles:
        blockers.append("manipulation_pov_effector_not_usefully_in_frame")
    if len(arm_chain_roles) < 2:
        blockers.append("manipulation_pov_arm_chain_not_in_frame")
    return {
        "status": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "effector_roles_in_frame": sorted(effector_roles),
        "useful_effector_roles_in_frame": sorted(useful_effector_roles),
        "arm_chain_roles_in_frame": sorted(arm_chain_roles),
        "useful_arm_chain_roles_in_frame": sorted(useful_arm_chain_roles),
        "requirement": (
            "Initial manipulation seed requires visible hand/wrist arm-chain evidence and the "
            "task affordance in frame; it does not require a specifically named forearm/elbow role."
        ),
    }


def _required_manipulation_arms(arm: str) -> tuple[str, ...]:
    selection = _normalize_reach_arm_selection(arm)
    return ("left", "right") if selection == "both" else (selection,)


def _manipulation_pov_geometry_single(
    *,
    arm_points: Mapping[str, Sequence[float]] | None,
    affordance,
    eye,
    target,
    up=(0.0, 0.0, 1.0),
    vfov_deg: float,
    width: int,
    height: int,
    arm: str = "right",
) -> dict[str, Any]:
    """Machine-check that the actual USD arm links are visible in the head POV.

    The articulated skeleton trace is optional and may be disabled for crash-safe kinematic renders,
    so manipulation media needs an independent USD-link projection check. The target affordance and
    task-side hand/wrist arm chain must be visible. Endpoint success is handled by the final-frame
    distance gate; this projection check alone is not contact or task-completion proof.
    """
    blockers: list[str] = []
    aff = (float(affordance[0]), float(affordance[1]), float(affordance[2]))
    target_px = project_point_to_pixel(aff, eye, target, up, vfov_deg, width, height)
    target_projection = _projection_dict(target_px)
    target_margin_px = None
    pitch_down_deg = _camera_pitch_down_deg(eye, target)
    if target_px is not None:
        u_px = float(target_px[0])
        v_px = float(target_px[1])
        target_margin_px = min(u_px, float(width) - u_px, v_px, float(height) - v_px)
    projected: list[dict[str, Any]] = []
    available_roles = sorted(str(k) for k in (arm_points or {}).keys())
    for role in ("shoulder", "elbow", "wrist", "hand"):
        pt = (arm_points or {}).get(role)
        if pt is None:
            continue
        px = project_point_to_pixel(
            (float(pt[0]), float(pt[1]), float(pt[2])),
            eye,
            target,
            up,
            vfov_deg,
            width,
            height,
        )
        proj = _projection_dict(px)
        if proj is None:
            continue
        projected.append({
            "landmark_id": f"{arm}_{role}_link",
            "link_role": role,
            "image_projection": proj,
        })

    roles_in_frame = {str(item["link_role"]) for item in projected}
    useful_projected = []
    min_margin_px = min(float(width), float(height)) * 0.05
    max_useful_v_px = float(height) * 0.84
    for item in projected:
        proj = item.get("image_projection") or {}
        try:
            u_px = float(proj.get("u_px"))
            v_px = float(proj.get("v_px"))
        except Exception:  # noqa: BLE001
            continue
        margin = min(u_px, float(width) - u_px, v_px, float(height) - v_px)
        if margin >= min_margin_px and v_px <= max_useful_v_px:
            useful_projected.append(item)
    useful_roles_in_frame = {str(item["link_role"]) for item in useful_projected}
    seed_arm_visibility = _manipulation_seed_arm_visibility(
        available_roles=available_roles,
        roles_in_frame=roles_in_frame,
        useful_roles_in_frame=useful_roles_in_frame,
    )
    if not available_roles:
        blockers.append("manipulation_pov_arm_links_unavailable")
    if target_px is None:
        blockers.append("manipulation_pov_target_not_in_frame")
    elif target_margin_px is not None and target_margin_px < min(float(width), float(height)) * 0.06:
        blockers.append("manipulation_pov_target_near_frame_edge")
    if pitch_down_deg > (
        MANIPULATION_POV_MAX_CAMERA_PITCH_DOWN_DEG
        + MANIPULATION_POV_CAMERA_PITCH_EPSILON_DEG
    ):
        blockers.append("manipulation_pov_camera_pitched_down_too_far")
    blockers.extend(str(b) for b in (seed_arm_visibility.get("blockers") or []))

    effector_distances: dict[str, float] = {}
    for role in ("wrist", "hand"):
        pt = (arm_points or {}).get(role)
        if pt is None:
            continue
        effector_distances[role] = round(
            math.sqrt(sum((float(pt[i]) - aff[i]) ** 2 for i in range(3))),
            4,
        )
    reach_feasibility: dict[str, Any] = {
        "status": "unverified",
        "blockers": ["manipulation_pov_reach_feasibility_unverified"],
        "g1_approx_arm_span_m": round(float(G1_APPROX_ARM_SPAN_M), 4),
        "max_shoulder_to_affordance_m": round(
            float(MANIPULATION_SEED_MAX_SHOULDER_TO_AFFORDANCE_M),
            4,
        ),
        "max_effector_to_affordance_m": round(
            float(MANIPULATION_SEED_MAX_EFFECTOR_TO_AFFORDANCE_M),
            4,
        ),
    }
    shoulder_for_reach = (arm_points or {}).get("shoulder")
    if shoulder_for_reach is not None and effector_distances:
        shoulder_affordance_m = math.sqrt(
            sum((float(shoulder_for_reach[i]) - aff[i]) ** 2 for i in range(3))
        )
        nearest_effector_to_affordance_m = min(float(v) for v in effector_distances.values())
        reach_blockers = _seed_reach_blockers(
            shoulder_to_affordance_m=shoulder_affordance_m,
            effector_to_affordance_m=nearest_effector_to_affordance_m,
            shoulder_margin_m=MANIPULATION_RENDERED_SEED_SHOULDER_MARGIN_M,
            effector_margin_m=MANIPULATION_RENDERED_SEED_EFFECTOR_MARGIN_M,
        )
        reach_feasibility = {
            "status": "PASS" if not reach_blockers else "FAIL",
            "blockers": reach_blockers,
            "shoulder_to_affordance_m": round(float(shoulder_affordance_m), 4),
            "nearest_effector_to_affordance_m": round(float(nearest_effector_to_affordance_m), 4),
            "g1_approx_arm_span_m": round(float(G1_APPROX_ARM_SPAN_M), 4),
            "max_shoulder_to_affordance_m": round(
                float(MANIPULATION_SEED_MAX_SHOULDER_TO_AFFORDANCE_M),
                4,
            ),
            "rendered_seed_shoulder_margin_m": round(
                float(MANIPULATION_RENDERED_SEED_SHOULDER_MARGIN_M),
                4,
            ),
            "max_effector_to_affordance_m": round(
                float(MANIPULATION_SEED_MAX_EFFECTOR_TO_AFFORDANCE_M),
                4,
            ),
            "rendered_seed_effector_margin_m": round(
                float(MANIPULATION_RENDERED_SEED_EFFECTOR_MARGIN_M),
                4,
            ),
            "claim_boundary": (
                "Reach feasibility is a conservative static geometry gate using the Unitree G1 "
                "size scale and USD link positions. It is not contact proof, inverse-kinematics "
                "proof, force-control proof, task completion, or physical robot validation."
            ),
            "required_for_seed_geometry": True,
        }
    arm_extension: dict[str, Any] = {
        "status": "unverified",
        "blockers": ["manipulation_pov_arm_extension_unverified"],
    }
    if arm_points and (arm_points.get("shoulder") is not None):
        shoulder_pt = arm_points["shoulder"]
        spans = []
        for role in ("wrist", "hand"):
            pt = arm_points.get(role)
            if pt is None:
                continue
            spans.append(math.sqrt(sum((float(pt[i]) - float(shoulder_pt[i])) ** 2 for i in range(3))))
        effector_pt = (arm_points.get("hand") or arm_points.get("wrist"))
        if effector_pt is not None:
            shoulder = tuple(float(shoulder_pt[i]) for i in range(3))
            effector = tuple(float(effector_pt[i]) for i in range(3))
            arm_len = math.sqrt(sum((effector[i] - shoulder[i]) ** 2 for i in range(3)))
            horizontal_extension_m = math.sqrt(
                (effector[0] - shoulder[0]) ** 2
                + (effector[1] - shoulder[1]) ** 2
            )
            horizontal_extension_ratio = horizontal_extension_m / arm_len if arm_len > 1e-6 else 0.0
            vertical_drop_ratio = (
                abs(effector[2] - shoulder[2]) / arm_len if arm_len > 1e-6 else 1.0
            )
            extension_blockers: list[str] = []
            # Reject the bad class where the arm is visible but hanging nearly vertical. Endpoint
            # closeness to the affordance is measured separately by the final-frame contract.
            if horizontal_extension_ratio < 0.35 or vertical_drop_ratio > 0.85:
                extension_blockers.append("manipulation_pov_arm_not_extended_forward")
            if arm_len < 0.12:
                extension_blockers.append("manipulation_pov_arm_extension_too_short")
            arm_extension = {
                "status": "PASS" if not extension_blockers else "FAIL",
                "blockers": extension_blockers,
                "shoulder_to_effector_m": round(float(arm_len), 4),
                "horizontal_extension_m": round(float(horizontal_extension_m), 4),
                "horizontal_extension_ratio": round(float(horizontal_extension_ratio), 4),
                "vertical_drop_ratio": round(float(vertical_drop_ratio), 4),
                "claim_boundary": (
                    "Forward extension checks visible arm posture only. It does not prove contact, "
                    "force-control behavior, task completion, or physical robot readiness."
                ),
            }
    if arm_extension.get("status") != "PASS":
        blockers.extend(str(b) for b in (arm_extension.get("blockers") or []))
    if reach_feasibility.get("status") != "PASS":
        blockers.extend(str(b) for b in (reach_feasibility.get("blockers") or []))

    return {
        "schema_version": "manipulation_pov_geometry.v1",
        "status": "PASS" if not blockers else "FAIL",
        "blockers": sorted(set(blockers)),
        "camera": "robot_pov",
        "reach_arm": arm,
        "target_affordance_xyz": [round(float(v), 6) for v in aff],
        "target_in_frame": target_px is not None,
        "target_projection": target_projection,
        "target_margin_px": round(float(target_margin_px), 2) if target_margin_px is not None else None,
        "camera_pitch_down_deg": round(float(pitch_down_deg), 2),
        "available_arm_link_roles": available_roles,
        "arm_roles_in_frame": sorted(roles_in_frame),
        "arm_roles_usefully_in_frame": sorted(useful_roles_in_frame),
        "arm_landmarks_in_frame": len(projected),
        "arm_landmarks_usefully_in_frame": len(useful_projected),
        "seed_arm_visibility": seed_arm_visibility,
        "effector_distance_to_affordance_m": effector_distances,
        "effector_distance_is_metadata_only": False,
        "reach_feasibility": reach_feasibility,
        "arm_extension": arm_extension,
        "projected_landmarks": projected,
        "claim_boundary": (
            "This checks camera framing and posed USD robot-link geometry against the resolved task "
            "affordance. It is not contact proof, force-control proof, physical validation, or "
            "deployment readiness."
        ),
    }


def _manipulation_pov_geometry(
    *,
    arm_points: Mapping[str, Sequence[float]] | None,
    affordance,
    eye,
    target,
    up=(0.0, 0.0, 1.0),
    vfov_deg: float,
    width: int,
    height: int,
    arm: str = "right",
    arm_points_by_arm: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
) -> dict[str, Any]:
    """Validate a manipulation seed POV for one or more requested arms."""
    selection = _normalize_reach_arm_selection(arm)
    required_arms = _required_manipulation_arms(selection)
    if len(required_arms) == 1:
        side = required_arms[0]
        side_points = (arm_points_by_arm or {}).get(side) if arm_points_by_arm else None
        report = _manipulation_pov_geometry_single(
            arm_points=side_points or arm_points,
            affordance=affordance,
            eye=eye,
            target=target,
            up=up,
            vfov_deg=vfov_deg,
            width=width,
            height=height,
            arm=side,
        )
        report["reach_arm"] = selection
        report["required_arms"] = [side]
        return report

    per_arm: dict[str, dict[str, Any]] = {}
    for side in required_arms:
        per_arm[side] = _manipulation_pov_geometry_single(
            arm_points=(arm_points_by_arm or {}).get(side) or {},
            affordance=affordance,
            eye=eye,
            target=target,
            up=up,
            vfov_deg=vfov_deg,
            width=width,
            height=height,
            arm=side,
        )
    primary = per_arm.get("right") or next(iter(per_arm.values()))
    non_reach_blockers_by_arm = {
        side: [
            str(blocker)
            for blocker in (report.get("blockers") or [])
            if str(blocker) not in MANIPULATION_REACH_BLOCKER_SET
        ]
        for side, report in per_arm.items()
    }
    side_failures = [
        side for side, arm_blockers in non_reach_blockers_by_arm.items() if arm_blockers
    ]
    blockers = sorted({
        str(blocker)
        for arm_blockers in non_reach_blockers_by_arm.values()
        for blocker in arm_blockers
    } | {
        f"manipulation_pov_{side}_arm_seed_failed"
        for side in side_failures
    })
    target_px = primary.get("target_projection")
    projected = [
        landmark
        for report in per_arm.values()
        for landmark in (report.get("projected_landmarks") or [])
    ]
    roles_in_frame = sorted({
        str(role)
        for report in per_arm.values()
        for role in (report.get("arm_roles_in_frame") or [])
    })
    useful_roles_in_frame = sorted({
        str(role)
        for report in per_arm.values()
        for role in (report.get("arm_roles_usefully_in_frame") or [])
    })
    available_roles = sorted({
        str(role)
        for report in per_arm.values()
        for role in (report.get("available_arm_link_roles") or [])
    })
    extension_by_arm = {
        side: report.get("arm_extension")
        for side, report in per_arm.items()
    }
    seed_arm_visibility_by_arm = {
        side: report.get("seed_arm_visibility")
        for side, report in per_arm.items()
    }
    seed_arm_visibility_blockers = sorted({
        str(blocker)
        for visibility in seed_arm_visibility_by_arm.values()
        if isinstance(visibility, Mapping)
        for blocker in (visibility.get("blockers") or [])
    })
    extension_blockers = sorted({
        str(blocker)
        for extension in extension_by_arm.values()
        if isinstance(extension, Mapping)
        for blocker in (extension.get("blockers") or [])
    })
    reach_feasibility_by_arm = {
        side: report.get("reach_feasibility")
        for side, report in per_arm.items()
    }
    reach_passing_arms = sorted(
        side
        for side, reach in reach_feasibility_by_arm.items()
        if isinstance(reach, Mapping) and reach.get("status") == "PASS"
    )
    all_reach_feasibility_blockers = sorted({
        str(blocker)
        for reach in reach_feasibility_by_arm.values()
        if isinstance(reach, Mapping)
        for blocker in (reach.get("blockers") or [])
    })
    reach_feasibility_blockers = [] if reach_passing_arms else all_reach_feasibility_blockers
    return {
        "schema_version": "manipulation_pov_geometry.v1",
        "status": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "camera": "robot_pov",
        "reach_arm": selection,
        "required_arms": list(required_arms),
        "target_affordance_xyz": primary.get("target_affordance_xyz"),
        "target_in_frame": bool(primary.get("target_in_frame")),
        "target_projection": target_px,
        "target_margin_px": primary.get("target_margin_px"),
        "camera_pitch_down_deg": primary.get("camera_pitch_down_deg"),
        "available_arm_link_roles": available_roles,
        "available_arm_link_roles_by_arm": {
            side: report.get("available_arm_link_roles") or []
            for side, report in per_arm.items()
        },
        "arm_roles_in_frame": roles_in_frame,
        "arm_roles_usefully_in_frame": useful_roles_in_frame,
        "arm_roles_in_frame_by_arm": {
            side: report.get("arm_roles_in_frame") or []
            for side, report in per_arm.items()
        },
        "arm_roles_usefully_in_frame_by_arm": {
            side: report.get("arm_roles_usefully_in_frame") or []
            for side, report in per_arm.items()
        },
        "arm_landmarks_in_frame": len(projected),
        "arm_landmarks_usefully_in_frame": sum(
            int(report.get("arm_landmarks_usefully_in_frame") or 0)
            for report in per_arm.values()
        ),
        "seed_arm_visibility": {
            "status": "PASS" if not seed_arm_visibility_blockers else "FAIL",
            "blockers": seed_arm_visibility_blockers,
            "by_arm": seed_arm_visibility_by_arm,
            "requirement": (
                "Each requested arm must provide visible hand/wrist arm-chain evidence in the "
                "task POV. This is seed framing evidence, not task completion."
            ),
        },
        "effector_distance_to_affordance_m": (
            primary.get("effector_distance_to_affordance_m") or {}
        ),
        "effector_distance_to_affordance_m_by_arm": {
            side: report.get("effector_distance_to_affordance_m") or {}
            for side, report in per_arm.items()
        },
        "effector_distance_is_metadata_only": False,
        "reach_feasibility": {
            "status": "PASS" if not reach_feasibility_blockers else "FAIL",
            "blockers": reach_feasibility_blockers,
            "all_arm_blockers": all_reach_feasibility_blockers,
            "required_passing_arm_count": 1,
            "passing_arms": reach_passing_arms,
            "by_arm": reach_feasibility_by_arm,
            "g1_approx_arm_span_m": round(float(G1_APPROX_ARM_SPAN_M), 4),
            "required_for_seed_geometry": True,
            "claim_boundary": (
                "Reach feasibility is a conservative static geometry gate using the Unitree G1 "
                "size scale and USD link positions. It requires at least one manipulation arm to "
                "be close enough to the resolved affordance while visibility and forward extension "
                "remain separate hard gates. This is not contact proof, inverse-kinematics proof, "
                "force-control proof, task completion, or physical robot validation. SAM3/DA3 can "
                "refine affordance/mask/depth evidence for this gate when configured."
            ),
        },
        "arm_extension": {
            "status": "PASS" if not extension_blockers else "FAIL",
            "blockers": extension_blockers,
            "by_arm": extension_by_arm,
            "claim_boundary": (
                "Forward extension checks visible arm posture only. It does not prove contact, "
                "force-control behavior, task completion, or physical robot readiness."
            ),
        },
        "projected_landmarks": projected,
        "per_arm_geometry": per_arm,
        "claim_boundary": (
            "This checks camera framing and posed USD robot-link geometry against the resolved task "
            "affordance. It is not contact proof, physical reach validation, task completion, or "
            "deployment readiness."
        ),
    }


def _fraction_from_histogram(hist: Sequence[int], indexes: range) -> float:
    total = float(sum(hist))
    if total <= 0.0:
        return 0.0
    return float(sum(hist[i] for i in indexes)) / total


def _image_luma_extreme_fractions(gray_img, box) -> dict[str, float]:
    crop = gray_img.crop(tuple(int(v) for v in box))
    hist = crop.histogram()
    return {
        "dark_fraction": round(_fraction_from_histogram(hist, range(0, 14)), 6),
        "bright_fraction": round(_fraction_from_histogram(hist, range(242, 256)), 6),
    }


def _pov_seed_frame_quality(frame_path: Path | str) -> dict[str, Any]:
    """Detect obvious camera self-occlusion/clipping in a saved robot POV seed frame.

    This deliberately uses broad image statistics, not object-specific semantics: a robot-head seed
    with a large near-black edge wedge is usually a camera inside/behind robot geometry or a clipped
    near-field body part. Dark task objects in the center are allowed; edge occlusion is not.
    """
    try:
        from PIL import Image  # type: ignore
        img = Image.open(frame_path).convert("L")
    except Exception as exc:  # noqa: BLE001
        return {
            "schema_version": "manipulation_pov_seed_frame_quality.v1",
            "status": "FAIL",
            "blockers": ["manipulation_pov_frame_unreadable"],
            "error": repr(exc),
        }
    w, h = img.size
    edge_w = max(1, int(round(w * 0.16)))
    lower_y = max(0, int(round(h * 0.45)))
    regions = {
        "left_edge": (0, 0, edge_w, h),
        "right_edge": (max(0, w - edge_w), 0, w, h),
        "left_lower_edge": (0, lower_y, edge_w, h),
        "right_lower_edge": (max(0, w - edge_w), lower_y, w, h),
    }
    metrics = {
        name: _image_luma_extreme_fractions(img, box)
        for name, box in regions.items()
    }
    edge_dark = max(metrics["left_edge"]["dark_fraction"], metrics["right_edge"]["dark_fraction"])
    lower_edge_dark = max(
        metrics["left_lower_edge"]["dark_fraction"],
        metrics["right_lower_edge"]["dark_fraction"],
    )
    blockers: list[str] = []
    if edge_dark > 0.38 or lower_edge_dark > 0.46:
        blockers.append("manipulation_pov_edge_self_occlusion")
    return {
        "schema_version": "manipulation_pov_seed_frame_quality.v1",
        "status": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "frame_path": str(frame_path),
        "image_size_px": [int(w), int(h)],
        "edge_band_fraction": 0.16,
        "regions": metrics,
        "max_edge_dark_fraction": round(float(edge_dark), 6),
        "max_lower_edge_dark_fraction": round(float(lower_edge_dark), 6),
        "claim_boundary": (
            "Image statistics only catch gross edge self-occlusion/clipping in the seed POV. "
            "They do not validate task success or object contact."
        ),
    }


def _select_manipulation_camera_target_for_visible_arm(
    affordance,
    arm_points: Mapping[str, Sequence[float]] | None,
    eye,
    initial_target,
    *,
    vfov_deg: float,
    width: int,
    height: int,
    arm: str = "right",
    arm_points_by_arm: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Pick a task-anchored look-at that keeps the active forearm in the head POV when possible."""
    aff = (float(affordance[0]), float(affordance[1]), float(affordance[2]))
    candidates: list[tuple[str, tuple[float, float, float]]] = [
        ("affordance_arm_context", tuple(float(v) for v in initial_target)),
        ("affordance", aff),
    ]
    preferred_pitch = min(
        float(MANIPULATION_POV_HEAD_FORWARD_PITCH_DOWN_DEG),
        float(MANIPULATION_POV_MAX_CAMERA_PITCH_DOWN_DEG),
    )

    def add_pitch_limited(name: str, candidate: Sequence[float]) -> None:
        limited = _target_raised_to_max_pitch_down(eye, candidate, preferred_pitch)
        if any(abs(float(limited[i]) - float(candidate[i])) > 1e-5 for i in range(3)):
            candidates.append((name, limited))

    add_pitch_limited("head_forward_pitch_limited_arm_context", initial_target)
    add_pitch_limited("head_forward_pitch_limited_affordance", aff)
    if arm_points:
        hand = arm_points.get("hand")
        wrist = arm_points.get("wrist")
        elbow = arm_points.get("elbow")
        shoulder = arm_points.get("shoulder")
        forward_z = max(float(aff[2]), float(eye[2]) - 0.18)
        if shoulder is not None:
            try:
                forward_seed = _manipulation_seed_arm_target_for_shoulder(shoulder, aff)
                forward_z = max(forward_z, float(forward_seed[2]))
            except Exception:  # noqa: BLE001
                pass
        forward_z = min(forward_z, float(eye[2]) - 0.03)
        if forward_z > float(aff[2]):
            candidates.append(("head_forward_affordance", (aff[0], aff[1], forward_z)))
        if hand is not None or wrist is not None:
            pts: list[tuple[Sequence[float], float]] = [(aff, 0.22)]
            if hand is not None:
                pts.append((hand, 0.34))
            if wrist is not None:
                pts.append((wrist, 0.30))
            if elbow is not None:
                pts.append((elbow, 0.14))
            weighted = _weighted_xyz(pts)
            if weighted is not None:
                candidates.append(("forearm_weighted", weighted))
                add_pitch_limited("head_forward_pitch_limited_forearm_context", weighted)
                if forward_z > float(weighted[2]):
                    candidates.append((
                        "head_forward_forearm_context",
                        (float(weighted[0]), float(weighted[1]), forward_z),
                    ))
            task_context_pts: list[tuple[Sequence[float], float]] = [(aff, 0.70)]
            if hand is not None:
                task_context_pts.append((hand, 0.18))
            if wrist is not None:
                task_context_pts.append((wrist, 0.12))
            task_context = _weighted_xyz(task_context_pts)
            if task_context is not None and forward_z > float(task_context[2]):
                add_pitch_limited("head_forward_pitch_limited_task_context", task_context)
                candidates.append((
                    "head_forward_task_context",
                    (float(task_context[0]), float(task_context[1]), forward_z),
                ))
        if wrist is not None and hand is not None:
            weighted = _weighted_xyz([(aff, 0.35), (wrist, 0.30), (hand, 0.35)])
            if weighted is not None:
                candidates.append(("effector_weighted", weighted))
                add_pitch_limited("head_forward_pitch_limited_effector_context", weighted)
                if forward_z > float(weighted[2]):
                    candidates.append((
                        "head_forward_effector_context",
                        (float(weighted[0]), float(weighted[1]), forward_z),
                    ))

    best_name = candidates[0][0]
    best_target = candidates[0][1]
    best_score = -1.0
    scored: list[dict[str, Any]] = []
    for name, candidate in candidates:
        geom = _manipulation_pov_geometry(
            arm_points=arm_points,
            arm_points_by_arm=arm_points_by_arm,
            affordance=aff,
            eye=eye,
            target=candidate,
            vfov_deg=vfov_deg,
            width=width,
            height=height,
            arm=arm,
        )
        geom_blockers = set(str(b) for b in (geom.get("blockers") or []))
        geom_pitch_down_deg = float(geom.get("camera_pitch_down_deg") or 0.0)
        roles = set(geom.get("arm_roles_usefully_in_frame") or geom.get("arm_roles_in_frame") or [])
        score = 0.0
        if geom.get("target_in_frame"):
            score += 6.0
        if roles.intersection({"hand", "wrist"}):
            score += 8.0
        score += 2.0 * len(roles.intersection({"elbow", "wrist", "hand"}))
        if len(roles.intersection({"elbow", "wrist", "hand"})) >= 2:
            score += 5.0
        if geom.get("status") == "PASS":
            score += 10.0
        else:
            score -= 1.5 * len(geom.get("blockers") or [])
        if "manipulation_pov_camera_pitched_down_too_far" in geom_blockers:
            score -= 30.0
        target_proj = geom.get("target_projection") or {}
        if target_proj:
            u = float(target_proj.get("u_px") or 0.0)
            v = float(target_proj.get("v_px") or 0.0)
            margin = min(u, float(width) - u, v, float(height) - v)
            score += max(0.0, min(6.0, margin / max(float(height), 1.0) * 20.0))
            score -= (abs(u - float(width) * 0.5) / max(float(width), 1.0)) * 2.0
            score -= (abs(v - float(height) * 0.5) / max(float(height), 1.0)) * 2.0
            useful_v_min = float(height) * 0.16
            useful_v_max = float(height) * 0.58
            if v < useful_v_min:
                score -= min(14.0, (useful_v_min - v) / max(float(height), 1.0) * 40.0)
            elif v > useful_v_max:
                score -= min(8.0, (v - useful_v_max) / max(float(height), 1.0) * 24.0)
        pitch_down_deg = geom_pitch_down_deg
        score -= min(8.0, max(0.0, pitch_down_deg - 18.0) / 6.0)
        if name.startswith("head_forward_"):
            score += 2.0
        if name.startswith("head_forward_pitch_limited_"):
            score += 3.0
        if score > best_score:
            best_name = name
            best_target = candidate
            best_score = score
        scored.append({
            "candidate": name,
            "score": round(score, 3),
            "status": geom.get("status"),
            "target_in_frame": geom.get("target_in_frame"),
            "target_projection": geom.get("target_projection"),
            "target_margin_px": geom.get("target_margin_px"),
            "pitch_down_deg": geom.get("camera_pitch_down_deg"),
            "arm_roles_in_frame": geom.get("arm_roles_in_frame"),
            "arm_roles_usefully_in_frame": geom.get("arm_roles_usefully_in_frame"),
            "selection_allowed": "manipulation_pov_camera_pitched_down_too_far" not in geom_blockers,
            "blockers": sorted(geom_blockers),
        })
    return best_target, {
        "selected_camera_target": best_name,
        "camera_target_score": round(best_score, 3),
        "camera_target_candidates": scored,
    }


# ============================ Isaac-only (GPU worker) ============================

def _boot_sim(headless: bool = True):
    from isaacsim import SimulationApp  # type: ignore
    return SimulationApp({"headless": headless, "renderer": "RayTracedLighting"})


def _extension_toggle():
    try:
        from isaacsim.core.utils.extensions import enable_extension, disable_extension  # type: ignore
    except Exception:  # noqa: BLE001
        from omni.isaac.core.utils.extensions import enable_extension, disable_extension  # type: ignore
    return enable_extension, disable_extension


def _enable_and_import_replicator():
    """Enable the Replicator extension (needed for render products) and import it. Must be
    called AFTER SimulationApp boots — omni.* modules are not importable before Kit starts."""
    enable_extension, _ = _extension_toggle()
    enable_extension("omni.replicator.core")
    import omni.replicator.core as rep  # type: ignore
    return rep


def _disable_physics_cooking() -> None:
    """Disable ONLY the PhysX collision-*cooking* extension (not the physx core, which the RTX
    renderer depends on), so the 47-object kitchen's SDF/convex cooking can't block the render.
    Also push every collision approximation to the cheapest box via carb settings, so any residual
    cooking is trivial. The kinematic preview needs no kitchen physics anyway."""
    _, disable_extension = _extension_toggle()
    try:
        disable_extension("omni.physx.cooking")
    except Exception:  # noqa: BLE001
        pass
    try:
        import carb  # type: ignore
        s = carb.settings.get_settings()
        s.set_bool("/physics/cooking/ujitsoCollisionCooking", False)
        s.set_bool("/persistent/physics/visualizationDisplayColliders", False)
        s.set_bool("/physics/collisionConeCustomGeometry", False)
    except Exception:  # noqa: BLE001
        pass


def _open_stage(usd_path: str, *, timeout_s: float = 90.0):
    import omni.usd  # type: ignore
    ctx = omni.usd.get_context()
    open_result = ctx.open_stage(usd_path)
    stage = ctx.get_stage()
    if stage is not None:
        return stage
    deadline = time.monotonic() + float(timeout_s)
    updates = 0
    last_update_error = None
    while time.monotonic() < deadline:
        try:
            import omni.kit.app  # type: ignore
            omni.kit.app.get_app().update()
            updates += 1
        except Exception as exc:  # noqa: BLE001
            last_update_error = repr(exc)
            time.sleep(0.1)
        stage = ctx.get_stage()
        if stage is not None:
            return stage
    raise RuntimeError(
        "isaac_stage_open_failed:"
        f"path={usd_path}:open_result={open_result!r}:updates={updates}:"
        f"last_update_error={last_update_error}"
    )


def _kitchen_usd_resolution_candidates(kitchen_usd: str) -> list[Path]:
    """Return local KitchenRoom.usd candidates for both supported extracted zip layouts."""
    raw = str(kitchen_usd or "").strip()
    if not raw or "://" in raw or raw.startswith("omniverse:"):
        return []
    path = Path(raw)
    candidates: list[Path] = []

    def add(candidate: Path) -> None:
        if candidate not in candidates:
            candidates.append(candidate)

    if path.is_absolute():
        add(path)
        parts = path.parts
        if "kitchen" in parts:
            root = Path(*parts[: parts.index("kitchen") + 1])
            add(root / "KitchenRoom.usd")
            add(root / "Collected_KitchenRoom" / "KitchenRoom.usd")
            try:
                for found in sorted(root.rglob("KitchenRoom.usd"), key=lambda p: (len(p.parts), str(p))):
                    add(found)
            except OSError:
                pass
    return candidates


def _resolve_existing_kitchen_usd(kitchen_usd: str) -> tuple[str, dict[str, Any]]:
    """Resolve the actual extracted kitchen USD path without assuming one zip root layout."""
    raw = str(kitchen_usd or "").strip()
    candidates = _kitchen_usd_resolution_candidates(raw)
    existing = next((p for p in candidates if p.is_file()), None)
    resolved = str(existing) if existing is not None else raw
    return resolved, {
        "schema_version": "kitchen_usd_resolution.v1",
        "requested_kitchen_usd": raw,
        "resolved_kitchen_usd": resolved,
        "resolved_from_existing_candidate": existing is not None and resolved != raw,
        "requested_exists": Path(raw).is_file() if raw and "://" not in raw and not raw.startswith("omniverse:") else None,
        "candidate_paths": [str(p) for p in candidates],
        "existing_candidate_paths": [str(p) for p in candidates if p.is_file()],
        "claim_boundary": (
            "Kitchen USD resolution only records which extracted local USD path the Isaac worker "
            "opened. It does not validate rendered quality, task success, or physical readiness."
        ),
    }


def _task_description_for_scenario(scenario: Mapping[str, Any]) -> str:
    """Best-effort natural-language task string for the scenario (for task->object resolution)."""
    for key in ("instruction", "task", "task_description", "description", "task_instruction"):
        val = scenario.get(key)
        if val:
            return str(val).strip()
    return ""


def _resolve_task_target_via_scene_placement(stage, scenario: Mapping[str, Any]) -> dict[str, Any] | None:
    """Resolve the task's target OBJECT from the scene when no explicit object id/coords are given.

    Unlike :func:`_resolve_task_target_from_stage` (which needs the caller to name the prim id), this
    enumerates EVERY object in the scene via the swappable ``scene_placement`` spatial index and maps
    the task description ("turn on the faucet") onto one of them. No scene-specific coordinates and no
    foreknowledge of the prim id — it works for any task in any USD site. The package is imported
    lazily + optionally so a worker without it (or without pxr) degrades to the id-driven path.
    """
    task = _task_description_for_scenario(scenario)
    if not task:
        return None
    # The bundle dir (where scene_placement is shipped) is added to sys.path at module load, but
    # Isaac's SimulationApp boot rewrites sys.path and drops it — and this resolver runs AFTER boot.
    # Re-add the runner's own dir so the bundle-first import below can still find the package.
    bundle_dir = str(Path(__file__).resolve().parent)
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)
    try:
        from scene_placement import (  # type: ignore # worker: flat package in the bundle dir
            UsdSceneSpatialIndex,
            resolve_target_by_label,
        )
    except Exception as inner_exc:  # noqa: BLE001
        try:
            from blueprint_pipeline.scene_placement import (  # type: ignore # repo / tests
                UsdSceneSpatialIndex,
                resolve_target_by_label,
            )
        except Exception:  # noqa: BLE001
            # Surface the INNER (bundle) import error, not the repo-fallback's — that's the one
            # that matters on the worker. Include sys.path head for diagnosis.
            return {"status": "blocked", "blockers": ["scene_placement_unavailable"],
                    "error": repr(inner_exc), "bundle_dir": bundle_dir, "sys_path_head": sys.path[:6]}
    try:
        index = UsdSceneSpatialIndex(stage=stage)
        objects = list(index.objects())
    except Exception as exc:  # noqa: BLE001
        return {"status": "blocked", "blockers": ["scene_placement_index_failed"], "error": repr(exc)}
    if not objects:
        return {"status": "blocked", "blockers": ["scene_placement_no_objects"], "task": task}
    target = resolve_target_by_label(task, objects)
    if target is None:
        return {
            "status": "blocked",
            "blockers": ["scene_placement_no_task_match"],
            "task": task,
            "object_labels": sorted({o.label for o in objects})[:40],
        }
    size = target.size()
    prim_path = ""
    extra = getattr(target, "extra", None)
    if isinstance(extra, Mapping):
        prim_path = str(extra.get("prim_path", "") or "")
    return {
        "status": "resolved",
        "source": "scene_placement_task_label",
        "selected": {
            "target_object_id": target.id,
            "target_object_label": target.label,
            "prim_path": prim_path,
            "match_kind": "task_label",
            "center_xyz": [round(float(c), 6) for c in target.centroid],
            "size_xyz": [round(float(s), 6) for s in size],
            "bbox_min_xyz": [round(float(c), 6) for c in target.bbox_min],
            "bbox_max_xyz": [round(float(c), 6) for c in target.bbox_max],
            "footprint_center_xy": [
                round(float(c), 6) for c in target.footprint_center()
            ],
        },
        "task": task,
        "objects_considered": len(objects),
    }


def _resolve_task_target_from_stage(
    stage,
    scenario: Mapping[str, Any],
    *,
    allow_scene_placement_fallback: bool = True,
) -> dict[str, Any] | None:
    """Resolve a task target from USD prim bounds when a scene/task compiler provides an object id.

    This is the generic fallback for sites that do not ship a separate object-location JSON. It is
    intentionally object-id driven; it does not know about kitchens, sinks, dishwashers, or counters.
    When no object id is supplied (or the id isn't found), it defers to
    :func:`_resolve_task_target_via_scene_placement`, which maps the task description onto a scene
    object — so a scenario with only a natural-language task still resolves a target dynamically.
    """
    target_ids = task_stance_target_object_ids_for_scenario(scenario)
    if not target_ids:
        # No explicit object id given — derive the target object from the task description via the
        # scene_placement spatial index (enumerate scene objects, map task -> object). Dynamic path.
        return _resolve_task_target_via_scene_placement(stage, scenario)
    try:
        from pxr import Usd, UsdGeom  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"status": "blocked", "blockers": ["usd_target_bounds_unavailable"], "error": repr(exc)}
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=True)
    matches: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        prim_name = str(prim.GetName())
        path_segments = [segment.lower() for segment in prim_path.split("/") if segment]
        matched_id = next(
            (
                tid
                for tid in target_ids
                if tid.lower() == prim_name.lower() or tid.lower() in path_segments
            ),
            None,
        )
        matched_priority = target_ids.index(matched_id) if matched_id in target_ids else len(target_ids)
        match_kind = "exact_prim_name_or_path_segment"
        if not matched_id:
            text = f"{prim_path} {prim_name}".lower()
            matched_id = next((tid for tid in target_ids if tid.lower() in text), None)
            matched_priority = target_ids.index(matched_id) if matched_id in target_ids else len(target_ids)
            match_kind = "ancestor_or_text_match"
        if not matched_id:
            continue
        try:
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if box.IsEmpty():
                continue
            bbox_min, bbox_max, center, size = _aligned_box_min_max_center_size(box)
        except Exception:  # noqa: BLE001
            continue
        center_xyz = [round(float(center[i]), 6) for i in range(3)]
        size_xyz = [round(float(size[i]), 6) for i in range(3)]
        bbox_min_xyz = [round(float(bbox_min[i]), 6) for i in range(3)]
        bbox_max_xyz = [round(float(bbox_max[i]), 6) for i in range(3)]
        matches.append({
            "target_object_id": matched_id,
            "target_object_priority": matched_priority,
            "prim_path": prim_path,
            "match_kind": match_kind,
            "path_depth": len(path_segments),
            "center_xyz": center_xyz,
            "size_xyz": size_xyz,
            "bbox_min_xyz": bbox_min_xyz,
            "bbox_max_xyz": bbox_max_xyz,
            "footprint_center_xy": [round(0.5 * (bbox_min_xyz[i] + bbox_max_xyz[i]), 6)
                                    for i in range(2)],
            "volume_proxy": round(float(size[0] * size[1] * size[2]), 9),
        })
    if not matches:
        if not allow_scene_placement_fallback:
            return {
                "status": "blocked",
                "blockers": ["target_object_id_not_found_in_usd_stage"],
                "target_object_ids": target_ids,
                "scene_placement_fallback": "disabled",
            }
        # The supplied object id(s) weren't found as prims — fall back to task->object resolution
        # over the full scene catalog before giving up.
        sp = _resolve_task_target_via_scene_placement(stage, scenario)
        if sp is not None and sp.get("status") == "resolved":
            return sp
        return {
            "status": "blocked",
            "blockers": ["target_object_id_not_found_in_usd_stage"],
            "target_object_ids": target_ids,
            "scene_placement_fallback": sp,
        }
    matches.sort(key=lambda item: (
        int(item.get("target_object_priority", len(target_ids))),
        0 if item["match_kind"] == "exact_prim_name_or_path_segment" else 1,
        item["path_depth"],
        -float(item["volume_proxy"]),
        len(item["prim_path"]),
    ))
    return {
        "status": "resolved",
        "source": "usd_prim_bounds",
        "selected": matches[0],
        "matches_considered": matches[:50],
    }


def _resolution_selected(resolution: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(resolution, Mapping):
        return {}
    selected = resolution.get("selected")
    return selected if isinstance(selected, Mapping) else {}


def _point_bbox_xy_gap_m(point: Sequence[float], bbox_min: Sequence[float], bbox_max: Sequence[float]) -> float:
    dx = max(float(bbox_min[0]) - float(point[0]), 0.0, float(point[0]) - float(bbox_max[0]))
    dy = max(float(bbox_min[1]) - float(point[1]), 0.0, float(point[1]) - float(bbox_max[1]))
    return math.hypot(dx, dy)


def _point_in_bbox_xy(
    point: Sequence[float],
    bbox_min: Sequence[float],
    bbox_max: Sequence[float],
    *,
    margin_m: float = 0.0,
) -> bool:
    return (
        float(bbox_min[0]) - margin_m <= float(point[0]) <= float(bbox_max[0]) + margin_m
        and float(bbox_min[1]) - margin_m <= float(point[1]) <= float(bbox_max[1]) + margin_m
    )


def _point_in_bbox_xyz(
    point: Sequence[float],
    bbox_min: Sequence[float],
    bbox_max: Sequence[float],
    *,
    margin_m: float = 0.0,
) -> bool:
    return (
        float(bbox_min[0]) - margin_m <= float(point[0]) <= float(bbox_max[0]) + margin_m
        and float(bbox_min[1]) - margin_m <= float(point[1]) <= float(bbox_max[1]) + margin_m
        and float(bbox_min[2]) - margin_m <= float(point[2]) <= float(bbox_max[2]) + margin_m
    )


def _scope_affordance_resolution_to_target(
    affordance_resolution: Mapping[str, Any] | None,
    target_resolution: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Prefer fine affordances that belong to the resolved coarse fixture.

    A room can contain many handles/knobs/spouts. Once the coarse fixture has been resolved
    (stovetop, sink, top cabinet), a fine affordance must be a descendant of that fixture or
    spatially overlap its footprint. Otherwise it is probably a semantically-correct but task-wrong
    affordance, e.g. a coffee-machine knob for a stovetop task.
    """
    if affordance_resolution is None:
        return None
    if affordance_resolution.get("status") != "resolved":
        return dict(affordance_resolution)
    target_selected = _resolution_selected(target_resolution)
    if not target_selected:
        return dict(affordance_resolution)
    target_path = str(target_selected.get("prim_path") or "").rstrip("/")
    target_center = _optional_xyz(target_selected.get("center_xyz"))
    target_bbox_min = _optional_xyz(target_selected.get("bbox_min_xyz"))
    target_bbox_max = _optional_xyz(target_selected.get("bbox_max_xyz"))
    raw_matches: list[Mapping[str, Any]] = []
    selected = _resolution_selected(affordance_resolution)
    if selected:
        raw_matches.append(selected)
    matches = affordance_resolution.get("matches_considered")
    if isinstance(matches, Sequence) and not isinstance(matches, (str, bytes)):
        raw_matches.extend(m for m in matches if isinstance(m, Mapping))
    deduped: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for match in raw_matches:
        key = str(match.get("prim_path") or json.dumps(dict(match), sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(match)
    if not deduped:
        return dict(affordance_resolution)

    scored: list[tuple[tuple[float, ...], Mapping[str, Any], dict[str, Any]]] = []
    for match in deduped:
        prim_path = str(match.get("prim_path") or "").rstrip("/")
        center = _optional_xyz(match.get("center_xyz"))
        same_subtree = bool(
            target_path
            and prim_path
            and (prim_path == target_path or prim_path.startswith(target_path + "/"))
        )
        in_target_xy = False
        in_target_xyz = False
        xy_gap_m = float("inf")
        distance_m = float("inf")
        if center is not None and target_bbox_min is not None and target_bbox_max is not None:
            in_target_xy = _point_in_bbox_xy(center, target_bbox_min, target_bbox_max, margin_m=0.08)
            in_target_xyz = _point_in_bbox_xyz(center, target_bbox_min, target_bbox_max, margin_m=0.08)
            xy_gap_m = _point_bbox_xy_gap_m(center, target_bbox_min, target_bbox_max)
        if center is not None and target_center is not None:
            distance_m = math.sqrt(sum((float(center[i]) - float(target_center[i])) ** 2 for i in range(3)))
        if not same_subtree and not in_target_xyz:
            continue
        scope_evidence = {
            "same_target_subtree": same_subtree,
            "inside_target_xy": in_target_xy,
            "inside_target_xyz": in_target_xyz,
            "xy_gap_to_target_bbox_m": (
                round(float(xy_gap_m), 4) if math.isfinite(xy_gap_m) else None
            ),
            "distance_to_target_center_m": (
                round(float(distance_m), 4) if math.isfinite(distance_m) else None
            ),
        }
        scored.append((
            (
                0.0 if same_subtree else 1.0,
                0.0 if in_target_xyz else 1.0,
                0.0 if in_target_xy else 1.0,
                float(match.get("target_object_priority", 999)),
                xy_gap_m,
                distance_m,
                float(match.get("path_depth", 999)),
            ),
            match,
            scope_evidence,
        ))
    if not scored:
        out = dict(affordance_resolution)
        out["status"] = "blocked"
        if "selected" in out:
            out["unscoped_selected"] = out.pop("selected")
        blockers = list(out.get("blockers") or [])
        if "affordance_not_scoped_to_target_fixture" not in blockers:
            blockers.append("affordance_not_scoped_to_target_fixture")
        out["blockers"] = blockers
        out["scope_filter"] = {
            "status": "blocked",
            "target_prim_path": target_path or None,
            "target_center_xyz": list(target_center) if target_center is not None else None,
            "target_bbox_min_xyz": list(target_bbox_min) if target_bbox_min is not None else None,
            "target_bbox_max_xyz": list(target_bbox_max) if target_bbox_max is not None else None,
            "unscoped_selected_prim_path": selected.get("prim_path") if selected else None,
            "matches_checked": len(deduped),
        }
        return out
    scored.sort(key=lambda item: item[0])
    best = dict(scored[0][1])
    best["scope_evidence"] = scored[0][2]
    out = dict(affordance_resolution)
    out["selected"] = best
    out["matches_considered"] = [dict(item[1]) for item in scored[:20]]
    out["scope_filter"] = {
        "status": "scoped_to_target_fixture",
        "target_prim_path": target_path or None,
        "selected_prim_path": best.get("prim_path"),
        "matches_checked": len(deduped),
        "matches_kept": len(scored),
    }
    return out


def _scenario_is_top_cabinet_task(scenario: Mapping[str, Any] | None) -> bool:
    if not scenario:
        return False
    if bool(scenario.get("derive_handleless_upper_cabinet_affordance")):
        return True
    text_parts: list[str] = []
    for key in ("task_id", "scenario_id"):
        value = scenario.get(key)
        if value:
            text_parts.append(str(value))
    text = " ".join(text_parts).lower()
    return (
        "top_cabinet" in text
        or "topcabinet" in text
    )


def _derive_handleless_upper_cabinet_affordance_resolution(
    *,
    target_resolution: Mapping[str, Any] | None,
    affordance_resolution: Mapping[str, Any] | None,
    scenario: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Create a scoped affordance for upper cabinets that have modeled doors but no handles.

    Some Lightwheel kitchen USDs model upper-cabinet doors/glass panels without pull handles. For the
    high-reach preflight, silently using the broad cabinet center is worse than blocking: it points
    the reach gate at the middle of a tall cabinet volume. This derives a lower-front door-edge
    contact zone from the resolved cabinet bounds and marks it as derived evidence, not a detected
    handle/pull.
    """
    if not _scenario_is_top_cabinet_task(scenario):
        return None
    target_selected = _resolution_selected(target_resolution)
    if not target_selected:
        return None
    target_id_text = " ".join(
        str(target_selected.get(key) or "")
        for key in ("target_object_id", "target_object_label", "prim_path")
    ).lower()
    if not any(token in target_id_text for token in ("topcabinet", "top_cabinet", "cabinet")):
        return None
    bbox_min = _optional_xyz(target_selected.get("bbox_min_xyz"))
    bbox_max = _optional_xyz(target_selected.get("bbox_max_xyz"))
    center = _optional_xyz(target_selected.get("center_xyz"))
    if bbox_min is None or bbox_max is None or center is None:
        return None
    size_z = max(0.0, float(bbox_max[2]) - float(bbox_min[2]))
    lower_edge_lift_m = max(0.08, min(0.16, size_z * 0.12))
    affordance_xyz = (
        float(center[0]),
        float(bbox_min[1]),
        float(bbox_min[2]) + lower_edge_lift_m,
    )
    half_x = 0.06
    half_y = 0.01
    half_z = 0.04
    selected = {
        "target_object_id": "derived_upper_cabinet_lower_front_edge",
        "target_object_priority": 0,
        "prim_path": target_selected.get("prim_path"),
        "match_kind": "derived_from_scoped_top_cabinet_bounds",
        "path_depth": target_selected.get("path_depth"),
        "center_xyz": [round(float(v), 6) for v in affordance_xyz],
        "size_xyz": [round(half_x * 2.0, 6), round(half_y * 2.0, 6), round(half_z * 2.0, 6)],
        "bbox_min_xyz": [
            round(float(affordance_xyz[0]) - half_x, 6),
            round(float(affordance_xyz[1]) - half_y, 6),
            round(float(affordance_xyz[2]) - half_z, 6),
        ],
        "bbox_max_xyz": [
            round(float(affordance_xyz[0]) + half_x, 6),
            round(float(affordance_xyz[1]) + half_y, 6),
            round(float(affordance_xyz[2]) + half_z, 6),
        ],
        "footprint_center_xy": [round(float(affordance_xyz[0]), 6), round(float(affordance_xyz[1]), 6)],
        "volume_proxy": round((half_x * 2.0) * (half_y * 2.0) * (half_z * 2.0), 9),
        "scope_evidence": {
            "same_target_subtree": True,
            "inside_target_xy": True,
            "inside_target_xyz": True,
            "xy_gap_to_target_bbox_m": 0.0,
            "derived_from_target_prim_path": target_selected.get("prim_path"),
            "source": "target_bbox_lower_front_edge",
        },
        "derived_affordance": True,
        "claim_boundary": (
            "This is a USD-bounds-derived lower/front upper-cabinet door-edge affordance used only "
            "when no scoped handle/pull prim exists. It is not a detected handle, contact proof, "
            "or manipulation-success evidence."
        ),
    }
    return {
        "status": "resolved",
        "source": "usd_target_bounds_derived_affordance",
        "selected": selected,
        "matches_considered": [selected],
        "unresolved_affordance_resolution": dict(affordance_resolution or {}),
        "scope_filter": {
            "status": "derived_from_scoped_target_fixture",
            "target_prim_path": target_selected.get("prim_path"),
            "selected_prim_path": target_selected.get("prim_path"),
            "reason": "upper_cabinet_has_no_scoped_handle_or_pull_prim",
        },
        "claim_boundary": selected["claim_boundary"],
    }


def _scene_placement_stand_plan(
    target_resolution,
    probe,
    *,
    floor_z: float = 0.05,
    scenario: Mapping[str, Any] | None = None,
    placement_validator=None,
):
    """Place the robot for a scene_placement-resolved target.

    When a scenario is supplied, the runner uses the approach-biased task-stance planner with the
    resolved target bounds, so the robot stands on the aisle/start side with distance measured from
    the target surface. Without scenario context, it falls back to the generic open-side
    ``compute_stand_pose`` solver. Returns ``None`` when the resolution lacks geometry, so the
    caller can fall back to plain ``plan_task_stance``.
    """
    sel = (target_resolution or {}).get("selected") or {}
    center = sel.get("center_xyz")
    size = sel.get("size_xyz")
    if not center or not size or len(center) != 3 or len(size) != 3:
        return None
    cx, cy, cz = (float(c) for c in center)
    sx, sy, sz = (abs(float(s)) for s in size)
    bbox_min = sel.get("bbox_min_xyz") or [cx - sx / 2.0, cy - sy / 2.0, cz - sz / 2.0]
    bbox_max = sel.get("bbox_max_xyz") or [cx + sx / 2.0, cy + sy / 2.0, cz + sz / 2.0]
    if scenario is not None:
        stance_scenario = dict(scenario)
        stance_scenario["target_object_position_xyz"] = [cx, cy, cz]
        stance_scenario["target_object_bbox_min_xyz"] = bbox_min
        stance_scenario["target_object_bbox_max_xyz"] = bbox_max
        if sel.get("target_object_id") is not None:
            stance_scenario.setdefault("target_object_id", sel.get("target_object_id"))
        if sel.get("target_object_label") is not None:
            stance_scenario.setdefault("target_object_label", sel.get("target_object_label"))
        plan = plan_task_stance(
            scenario=stance_scenario,
            probe_collision=probe,
            floor_z_hint=floor_z,
            placement_validator=placement_validator,
        )
        plan["source"] = "scene_placement_surface_offset_task_stance"
        if plan.get("status") == "accepted":
            idx = int(plan.get("selected_candidate_index") or 0)
            candidates = plan.get("candidates") or []
            chosen = candidates[idx] if 0 <= idx < len(candidates) else {}
            plan["stand_clear"] = True
            plan["standoff_m"] = chosen.get("standoff_from_target_surface_m")
            plan["notes"] = (
                "approach-biased scene target stance; distance measured from target "
                "footprint surface"
            )
        return plan
    bundle_dir = str(Path(__file__).resolve().parent)
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)
    try:
        try:
            from scene_placement import SceneObject, compute_stand_pose  # worker bundle
        except Exception:  # noqa: BLE001
            from blueprint_pipeline.scene_placement import SceneObject, compute_stand_pose  # repo/tests
    except Exception as exc:  # noqa: BLE001
        return {"status": "blocked", "blockers": ["scene_placement_unavailable"], "error": repr(exc)}
    target = SceneObject(
        id=str(sel.get("target_object_id") or "target"),
        label=str(sel.get("target_object_label") or ""),
        bbox_min=(float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])),
        bbox_max=(float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])),
        centroid=(cx, cy, cz),
        source="usd",
    )
    pose = compute_stand_pose(
        target, probe=probe, pelvis_height=ROBOT_PELVIS_HEIGHT_M, floor_z=floor_z,
        standing_distance=TASK_STANCE_DEFAULT_SURFACE_STANDOFF_M, include_diagonals=True,
    )
    return {
        "schema_version": TASK_STANCE_SCHEMA_VERSION,
        # compute_stand_pose always yields a non-clipping best-effort pose (closest clear spot, or
        # the farthest probed spot when boxed in), so this is always usable placement.
        "status": "accepted",
        "source": "scene_placement_compute_stand_pose",
        "accepted_pose": [round(float(v), 6) for v in pose.position],
        "accepted_yaw": round(float(pose.yaw), 6),
        "task_target_xyz": [round(cx, 6), round(cy, 6), round(cz, 6)],
        "stand_clear": bool(pose.clear),
        "standoff_m": round(float(pose.standoff_m), 6),
        "task_target_bounds": {
            "bbox_min_xyz": [round(float(v), 6) for v in bbox_min],
            "bbox_max_xyz": [round(float(v), 6) for v in bbox_max],
        },
        "candidates": [],
        "notes": pose.notes,
    }


def _plan_task_stance_for_stage(
    *,
    stage,
    scenario: Mapping[str, Any],
    manipulation_look_at,
    probe,
    no_collision_probe: bool,
    robot_prim_path: str | None = None,
) -> dict[str, Any]:
    stance_scenario = dict(scenario)
    target_resolution = None
    affordance_resolution = None
    explicit_target = task_stance_target_for_scenario(
        stance_scenario,
        manipulation_look_at,
        allow_navigation_target_fallback=False,
    )
    target_ids = task_stance_target_object_ids_for_scenario(stance_scenario)
    target_bounds_present = _target_bounds_for_scenario(stance_scenario) is not None
    if explicit_target is None or target_ids or not target_bounds_present:
        target_resolution = _resolve_task_target_from_stage(stage, stance_scenario)
        if target_resolution and target_resolution.get("status") == "resolved":
            selected = (
                target_resolution.get("selected")
                if isinstance(target_resolution.get("selected"), Mapping)
                else {}
            )
            if explicit_target is None and selected.get("center_xyz") is not None:
                stance_scenario = _with_xyz(
                    stance_scenario,
                    "target_object_position_xyz",
                    selected["center_xyz"],
                )
            if (
                not target_bounds_present
                and selected.get("bbox_min_xyz") is not None
                and selected.get("bbox_max_xyz") is not None
            ):
                stance_scenario["target_object_bbox_min_xyz"] = selected["bbox_min_xyz"]
                stance_scenario["target_object_bbox_max_xyz"] = selected["bbox_max_xyz"]
            if selected.get("target_object_id") is not None:
                stance_scenario.setdefault("target_object_id", selected.get("target_object_id"))
            if selected.get("target_object_label") is not None:
                stance_scenario.setdefault("target_object_label", selected.get("target_object_label"))

    scope_target_resolution = (
        target_resolution
        if target_resolution and target_resolution.get("status") == "resolved"
        else _synthetic_target_resolution_from_scenario(stance_scenario)
    )
    affordance_ids = task_stance_affordance_object_ids_for_scenario(stance_scenario)
    affordance_resolution_failed = False
    if affordance_ids:
        affordance_scenario = dict(stance_scenario)
        for key in TASK_STANCE_TARGET_OBJECT_KEYS:
            affordance_scenario.pop(key, None)
        affordance_scenario["target_object_ids"] = affordance_ids
        affordance_resolution = _resolve_task_target_from_stage(
            stage,
            affordance_scenario,
            allow_scene_placement_fallback=False,
        )
        affordance_resolution = _scope_affordance_resolution_to_target(
            affordance_resolution,
            scope_target_resolution,
        )
        selected_affordance = (
            affordance_resolution.get("selected")
            if isinstance(affordance_resolution, Mapping)
            else None
        )
        if not (
            isinstance(selected_affordance, Mapping)
            and affordance_resolution.get("status") == "resolved"
            and selected_affordance.get("center_xyz") is not None
        ):
            derived_affordance_resolution = _derive_handleless_upper_cabinet_affordance_resolution(
                target_resolution=scope_target_resolution,
                affordance_resolution=affordance_resolution,
                scenario=stance_scenario,
            )
            if derived_affordance_resolution is not None:
                affordance_resolution = derived_affordance_resolution
        selected_affordance = (
            affordance_resolution.get("selected")
            if isinstance(affordance_resolution, Mapping)
            else None
        )
        if (
            isinstance(selected_affordance, Mapping)
            and affordance_resolution.get("status") == "resolved"
            and selected_affordance.get("center_xyz") is not None
        ):
            stance_scenario = _with_xyz(
                stance_scenario,
                "task_affordance_xyz",
                selected_affordance["center_xyz"],
            )
        else:
            affordance_resolution_failed = True

    def _with_affordance_resolution(plan: dict[str, Any]) -> dict[str, Any]:
        if affordance_resolution is None:
            return plan
        plan["affordance_resolution"] = affordance_resolution
        selected = affordance_resolution.get("selected") if isinstance(affordance_resolution, Mapping) else None
        if isinstance(selected, Mapping) and affordance_resolution.get("status") == "resolved":
            if selected.get("center_xyz") is not None:
                plan["task_affordance_xyz"] = selected.get("center_xyz")
            if selected.get("bbox_min_xyz") is not None and selected.get("bbox_max_xyz") is not None:
                plan["task_affordance_bounds"] = {
                    "bbox_min_xyz": selected.get("bbox_min_xyz"),
                    "bbox_max_xyz": selected.get("bbox_max_xyz"),
                }
            plan["affordance_focus_source"] = (
                "usd_target_bounds_derived_affordance"
                if selected.get("derived_affordance")
                else "usd_affordance_object_alias"
            )
        elif plan.get("task_affordance_xyz") is not None:
            plan["affordance_focus_source"] = "target_fixture_center_after_unscoped_affordance_rejected"
        return plan
    if affordance_resolution_failed:
        return _with_affordance_resolution({
            "schema_version": TASK_STANCE_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["task_affordance_resolution_failed"],
            "target_resolution": target_resolution,
            "claim_boundary": (
                "This manipulation task declared fine affordance aliases. The stance planner must "
                "not fall back to a broad fixture center when those affordances are unresolved or "
                "belong to another fixture."
            ),
        })
    if no_collision_probe:
        return _with_affordance_resolution({
            "schema_version": TASK_STANCE_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["task_stance_collision_probe_disabled"],
            "target_resolution": target_resolution,
            "claim_boundary": (
                "Task stance placement requires a scene collision/clearance probe; "
                "without it the runner must not claim the robot is standing clear."
            ),
        })
    target_bounds = _target_bounds_for_scenario(stance_scenario)
    validation_floor_z = float(stance_scenario.get("floor_z_hint", 0.05) or 0.05)
    validation_scene_objects = (
        _placement_obstacles_for_stage(stage, focus_bounds=target_bounds)
        if target_bounds is not None
        else []
    )
    validation_target_object = (
        _target_object_from_stance_plan(
            {
                "task_target_xyz": stance_scenario.get("target_object_position_xyz")
                or stance_scenario.get("task_target_position_xyz"),
                "task_target_bounds": (
                    {
                        "bbox_min_xyz": list(target_bounds[0]),
                        "bbox_max_xyz": list(target_bounds[1]),
                    }
                    if target_bounds is not None
                    else None
                ),
                "target_resolution": target_resolution,
            }
        )
        if target_bounds is not None
        else None
    )
    validation_standoff_range = _validation_standoff_range_for_scenario(stance_scenario)
    placement_validator = (
        _placement_validator_for_stage(
            stage,
            robot_prim_path,
            target_bounds,
            target_object=validation_target_object,
            scene_objects=validation_scene_objects,
            floor_z=validation_floor_z,
            standoff_range=validation_standoff_range,
        )
        if robot_prim_path and target_bounds is not None
        else None
    )
    # For a scene_placement-resolved target, keep the task object bounds while honoring the
    # scenario approach/start side. That places sink/faucet tasks in the counter-to-island aisle
    # rather than accepting a too-near point around the fixture centroid.
    if target_resolution and str(target_resolution.get("source") or "").startswith("scene_placement"):
        sp_plan = _scene_placement_stand_plan(
            target_resolution, probe,
            floor_z=float(stance_scenario.get("floor_z_hint", 0.05) or 0.05),
            scenario=stance_scenario,
            placement_validator=placement_validator,
        )
        if sp_plan is not None:
            sp_plan["target_resolution"] = target_resolution
            return _with_affordance_resolution(sp_plan)
    stance_plan = plan_task_stance(
        scenario=stance_scenario,
        manipulation_look_at=manipulation_look_at,
        probe_collision=probe,
        floor_z_hint=stance_scenario.get("floor_z_hint"),
        placement_validator=placement_validator,
    )
    if target_resolution is not None:
        stance_plan["target_resolution"] = target_resolution
    return _with_affordance_resolution(stance_plan)


def _resolve_asset_uri(value: str) -> str:
    """Resolve a relative Isaac asset path (e.g. 'Isaac/Robots/Unitree/G1/g1.usd') against the
    Isaac assets root on the worker. Absolute paths / URIs pass through unchanged."""
    if "://" in value or value.startswith("/") or value.startswith("omniverse:"):
        return value
    try:
        from isaacsim.storage.native import get_assets_root_path  # type: ignore
        root = get_assets_root_path()
        if root:
            return root.rstrip("/") + "/" + value.lstrip("/")
    except Exception:  # noqa: BLE001
        pass
    return value


def _g1_visual_asset_candidates(value: str) -> list[str]:
    """Return likely visual USD candidates without tying the harness to a scene.

    The caller supplies the robot asset URI/path. For multiphysics ``.usda`` inputs, prefer the
    same-directory ``.usd`` visual sibling and the Isaac-6 short ``Unitree/G1`` visual sibling before
    falling back to the physics-only ``.usda`` composition. Non-``.usda`` inputs are tried exactly
    first. The list is ordered and deduped.
    """
    candidates: list[str] = []

    def add(candidate: str | None) -> None:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    raw = str(value or "").strip()
    lower = raw.lower()
    short_candidates: list[str] = []
    if "/Isaac/Robots/Unitree/G1/" in raw:
        short_candidates.append(raw.replace("/Isaac/Robots/Unitree/G1/", "/Unitree/G1/"))
    if raw.startswith("Isaac/Robots/Unitree/G1/"):
        short_candidates.append(raw.replace("Isaac/Robots/Unitree/G1/", "Unitree/G1/", 1))
    if lower.endswith(".usda"):
        add(raw[:-5] + ".usd")
        for short in short_candidates:
            if short.lower().endswith(".usda"):
                add(short[:-5] + ".usd")
            add(short)
        add(raw)
    else:
        add(raw)
        for short in short_candidates:
            add(short)
    return candidates


def _bind_g1(stage, g1_usd: str, prim_path: str = "/World/G1"):
    """Reference the official Isaac G1 USD and verify it is a controllable, collidable articulation."""
    from pxr import UsdPhysics  # type: ignore
    g1_prim = stage.DefinePrim(prim_path, "Xform")
    g1_prim.GetReferences().AddReference(g1_usd)
    load_error = None
    try:
        stage.Load(prim_path)
        payload_load_status = "loaded"
    except Exception as exc:  # noqa: BLE001
        payload_load_status = "load_failed"
        load_error = repr(exc)
    art_count = collision_count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_count += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_count += 1
    return {
        "prim_path": prim_path,
        "controllable_articulation_detected": art_count > 0,
        "collision_enabled_verified": collision_count > 0,
        "articulation_root_api_prim_count": art_count,
        "collision_api_prim_count": collision_count,
        "g1_usd": g1_usd,
        "payload_load_status": payload_load_status,
        "payload_load_error": load_error,
    }


def _robot_visual_geometry_missing(diag: Mapping[str, Any] | None) -> bool:
    """True when the robot subtree has no actual renderable Gprim/Mesh surface."""
    if not diag:
        return True
    blockers = {str(b) for b in (diag.get("blockers") or [])}
    if ROBOT_VISUAL_MESH_MISSING_BLOCKER in blockers:
        return True
    try:
        return int(diag.get("gprim_count") or 0) <= 0
    except Exception:  # noqa: BLE001
        return True


def _bind_g1_with_visual_fallback(stage, g1_usd: str, prim_path: str = "/World/G1"):
    """Bind G1 and prefer a composition that exposes renderable visual surfaces.

    Link projections and articulation probes can pass against a physics-only composition. This helper
    keeps those signals separate from visual readiness by trying ordered visual candidates and returning
    the first one whose robot subtree has drawable Gprims. If every candidate remains physics-only, the
    final binding carries a fail-closed visual diagnostic.
    """
    attempts: list[dict[str, Any]] = []
    last_binding: dict[str, Any] | None = None
    best_nonvisual_candidate: str | None = None
    best_nonvisual_resolved: str | None = None
    for candidate in _g1_visual_asset_candidates(g1_usd):
        resolved = _resolve_asset_uri(candidate)
        try:
            if stage.GetPrimAtPath(prim_path).IsValid():
                stage.RemovePrim(prim_path)
        except Exception:  # noqa: BLE001
            pass
        binding = _bind_g1(stage, resolved, prim_path=prim_path)
        diag = _robot_render_visibility_diagnostics(stage, prim_path)
        binding["robot_render_diagnostics"] = diag
        binding["requested_g1_usd"] = g1_usd
        binding["candidate_g1_usd"] = candidate
        binding["resolved_g1_usd"] = resolved
        binding["visual_candidate_attempts"] = attempts
        attempt = {
            "candidate_g1_usd": candidate,
            "resolved_g1_usd": resolved,
            "status": diag.get("status"),
            "blockers": diag.get("blockers", []),
            "gprim_count": diag.get("gprim_count"),
            "mesh_count": diag.get("mesh_count"),
            "payload_load_status": binding.get("payload_load_status"),
        }
        attempts.append(attempt)
        last_binding = binding
        if not _robot_visual_geometry_missing(diag):
            binding["visual_binding_status"] = "renderable_robot_geometry_found"
            binding["visual_candidate_attempts"] = attempts
            return binding
        if best_nonvisual_candidate is None and (
            binding.get("controllable_articulation_detected")
            or binding.get("collision_enabled_verified")
            or int(binding.get("articulation_root_api_prim_count") or 0) > 0
            or int(binding.get("collision_api_prim_count") or 0) > 0
        ):
            best_nonvisual_candidate = candidate
            best_nonvisual_resolved = resolved
    assert last_binding is not None
    if best_nonvisual_resolved and best_nonvisual_resolved != last_binding.get("resolved_g1_usd"):
        try:
            if stage.GetPrimAtPath(prim_path).IsValid():
                stage.RemovePrim(prim_path)
        except Exception:  # noqa: BLE001
            pass
        last_binding = _bind_g1(stage, best_nonvisual_resolved, prim_path=prim_path)
        diag = _robot_render_visibility_diagnostics(stage, prim_path)
        last_binding["robot_render_diagnostics"] = diag
        last_binding["requested_g1_usd"] = g1_usd
        last_binding["candidate_g1_usd"] = best_nonvisual_candidate
        last_binding["resolved_g1_usd"] = best_nonvisual_resolved
    last_binding["visual_binding_status"] = "blocked_missing_renderable_robot_geometry"
    last_binding["visual_candidate_attempts"] = attempts
    last_binding["selected_nonvisual_candidate_reason"] = (
        "preserved_articulation_or_collision_candidate_when_no_renderable_gprims_found"
        if best_nonvisual_resolved
        else "no_candidate_exposed_renderable_gprims_or_articulation_collision"
    )
    return last_binding


def _setup_g1_articulation(prim_path: str):
    """Create + initialize an Isaac Articulation on the bound G1 so we can drive its joints
    (the procedural walk gait) and read its link world poses (the skeleton). Returns
    (articulation, dof_index_by_name, default_joint_positions, link_names). GPU-only."""
    from isaacsim.core.prims import SingleArticulation  # type: ignore
    art = SingleArticulation(prim_path=prim_path, name="g1")
    art.initialize()
    dof_names = list(art.dof_names or [])
    dof_index = {n: i for i, n in enumerate(dof_names)}
    import numpy as np  # type: ignore
    default = np.asarray(art.get_joint_positions()).astype("float32")
    link_names = list(getattr(art, "body_names", []) or [])
    return art, dof_index, default, link_names


def manipulation_ready_arm_joint_deltas(arm: str = "both") -> dict[str, float]:
    """Joint deltas that raise G1 forearms into a first-person manipulation-ready pose.

    The values are relative to the standing keyframe, so the pose is portable across Isaac and
    MuJoCo G1 assets without hard-coding absolute default qpos values.
    """
    selection = str(arm or "both").strip().lower()
    if selection not in MANIPULATION_READY_ARM_SELECTIONS:
        raise ValueError(f"unknown manipulation arm selection: {arm!r}")
    sides = ("left", "right") if selection == "both" else (selection,)
    out: dict[str, float] = {}
    for side in sides:
        out.update(MANIPULATION_READY_ARM_JOINT_DELTAS[side])
    return out


def _apply_joint_deltas(targets, default, dof_index, deltas: Mapping[str, float]) -> list[str]:
    applied: list[str] = []
    for name, delta in deltas.items():
        idx = dof_index.get(name)
        if idx is not None and idx < len(targets) and idx < len(default):
            targets[idx] = default[idx] + float(delta)
            applied.append(name)
    return applied


def _apply_named_joint_targets(targets, dof_index, named_targets: Mapping[str, Any] | None) -> list[str]:
    applied: list[str] = []
    if not isinstance(named_targets, Mapping):
        return applied
    for name, value in named_targets.items():
        idx = dof_index.get(str(name))
        if idx is None or idx >= len(targets):
            continue
        try:
            targets[idx] = float(value)
        except (TypeError, ValueError):
            continue
        applied.append(str(name))
    return applied


def _joint_targets_for_pose(
    default,
    dof_index,
    *,
    phase,
    moving,
    manipulation_ready: bool = False,
    manipulation_reach_arm: str = "both",
):
    import numpy as np  # type: ignore
    targets = np.array(default, dtype="float32", copy=True)
    _apply_joint_deltas(targets, default, dof_index, policy_mod.gait_joint_deltas(phase, moving))
    if manipulation_ready:
        _apply_joint_deltas(
            targets,
            default,
            dof_index,
            manipulation_ready_arm_joint_deltas(manipulation_reach_arm),
        )
    return targets


def _pd_leg_joint_efforts(
    target_q,
    q,
    dq,
    *,
    kp=DEFAULT_ISAAC_LEG_KP,
    kd=DEFAULT_ISAAC_LEG_KD,
):
    """Port the MuJoCo PD effort law without importing MuJoCo configuration."""
    import numpy as np  # type: ignore

    target_arr = np.asarray(target_q, dtype="float32")
    q_arr = np.asarray(q, dtype="float32")
    dq_arr = np.asarray(dq, dtype="float32")
    kp_arr = np.asarray(kp, dtype="float32")
    kd_arr = np.asarray(kd, dtype="float32")
    tau = (target_arr - q_arr) * kp_arr + (np.zeros_like(dq_arr) - dq_arr) * kd_arr
    return np.asarray(tau, dtype="float32")


def _apply_articulation_joint_targets(art, targets):
    """Prefer Isaac's articulation action path, falling back to direct joint state writes on older
    worker images. The return value is persisted in the contact report for auditability."""
    try:
        from isaacsim.core.utils.types import ArticulationAction  # type: ignore
        art.apply_action(ArticulationAction(joint_positions=targets))
        return "articulation_action_position_targets"
    except Exception:  # noqa: BLE001
        art.set_joint_positions(targets)
        return "direct_joint_state_position_set"


def _apply_articulation_joint_efforts(art, efforts):
    """Apply generalized joint efforts, with a direct setter fallback for older worker images."""
    try:
        from isaacsim.core.utils.types import ArticulationAction  # type: ignore
        art.apply_action(ArticulationAction(joint_efforts=efforts))
        return "articulation_action_joint_efforts"
    except Exception:  # noqa: BLE001
        art.set_joint_efforts(efforts)
        return "direct_joint_effort_set"


def _drive_g1_walk(
    art,
    dof_index,
    default,
    *,
    root_pose,
    yaw,
    phase,
    moving,
    manipulation_ready: bool = False,
    manipulation_reach_arm: str = "both",
    policy_joint_targets: Mapping[str, Any] | None = None,
):
    """Set the G1 root world pose + joint positions = standing + gait deltas (kinematic pose)."""
    import numpy as np  # type: ignore
    w, x, y, z = yaw_to_quat(float(yaw))
    art.set_world_pose(position=np.asarray(root_pose, dtype="float32"),
                       orientation=np.asarray([w, x, y, z], dtype="float32"))
    targets = _joint_targets_for_pose(
        default,
        dof_index,
        phase=phase,
        moving=moving,
        manipulation_ready=manipulation_ready,
        manipulation_reach_arm=manipulation_reach_arm,
    )
    _apply_named_joint_targets(targets, dof_index, policy_joint_targets)
    art.set_joint_positions(targets)
    return targets


def _g1_skeleton_world_positions(art, link_names):
    """World-space positions of the G1 links (the skeleton landmarks before projection)."""
    import numpy as np  # type: ignore
    positions, _ = art.get_link_world_poses()
    positions = np.asarray(positions)
    return [(link_names[i] if i < len(link_names) else f"link_{i}",
             (float(positions[i][0]), float(positions[i][1]), float(positions[i][2])))
            for i in range(len(positions))]


def _project_skeleton(skeleton_world, *, eye, target, up, vfov_deg, width, height):
    """Project G1 link world positions into the camera -> OSCAR-schema landmark list. Each landmark
    is {landmark_id, image_projection:{available,u_px,v_px,depth_m}} (the exact shape the OSCAR WAM
    input-package materialization reads)."""
    landmarks = []
    for name, wp in skeleton_world:
        px = project_point_to_pixel(wp, eye, target, up, vfov_deg, width, height)
        if px is not None:
            landmarks.append({"landmark_id": name, "image_projection": {
                "available": True, "u_px": round(px[0], 2), "v_px": round(px[1], 2),
                "depth_m": round(px[2], 4)}})
    return landmarks


def _g1_link_rest_offsets(stage, prim_path: str):
    """Pure-USD G1 skeleton: rest-pose offset (in the root frame) of each link prim under the G1.
    No physics/tensor-view (which gets invalidated on this G1 USD) — just the link transforms.
    Returns [(name, (dx,dy,dz)), ...]. Per-step world = root_pose + Rz(yaw) @ offset."""
    from pxr import Usd, UsdGeom  # type: ignore
    xc = UsdGeom.XformCache()
    root_prim = stage.GetPrimAtPath(prim_path)
    rt = xc.GetLocalToWorldTransform(root_prim).ExtractTranslation()
    root = (float(rt[0]), float(rt[1]), float(rt[2]))
    offs = []
    for prim in Usd.PrimRange(root_prim):
        name = prim.GetName()
        if "link" not in name.lower() or not prim.IsA(UsdGeom.Xformable):
            continue
        t = xc.GetLocalToWorldTransform(prim).ExtractTranslation()
        offs.append((name, (float(t[0]) - root[0], float(t[1]) - root[1], float(t[2]) - root[2])))
    return offs


def _rest_skeleton_world(offsets, root_pose, yaw):
    """Place the rest-pose link offsets at the robot's per-step root pose (translate + Z-rotate)."""
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    out = []
    for name, (ox, oy, oz) in offsets:
        out.append((name, (root_pose[0] + cy * ox - sy * oy,
                           root_pose[1] + sy * ox + cy * oy,
                           root_pose[2] + oz)))
    return out


def skeleton_world_for_frame(*, art_ctx, rest_offsets, root_pose, yaw):
    """Return the best available G1 skeleton for a rendered frame.

    Some Isaac worker images expose a controllable G1 articulation with valid joints but no body
    names. Reading link poses in that state can invalidate the PhysX tensor view, so fall back to
    the USD rest-offset skeleton unless the articulation has usable link names.
    """
    if art_ctx is not None and art_ctx.get("link_names"):
        try:
            return _g1_skeleton_world_positions(art_ctx["art"], art_ctx["link_names"])
        except Exception as exc:  # noqa: BLE001
            _log(f"G1 articulation skeleton read failed ({exc!r}); using USD skeleton fallback")
    if rest_offsets is not None:
        return _rest_skeleton_world(rest_offsets, root_pose, yaw)
    return []


def compute_arm_reach_skeleton(
    skeleton,
    target,
    reach_frac,
    *,
    arm: str = "right",
    forward_yaw: float | None = None,
):
    """Re-pose one arm of a world-space skeleton into the current manipulation reach phase.

    The walk policy never moves the arms, so the skeleton (OSCAR's action conditioning) just shows a
    rigid robot. This rotates the arm chain about the shoulder so the hand travels from its rest spot
    to a shoulder-relative reach target as ``reach_frac`` goes 0->1. Early frames preserve the
    forward-ready seed used for policy conditioning; endpoint frames aim at the resolved affordance for
    visible review. Each arm link keeps its rest fractional distance from the shoulder, and the reach
    is clamped to the arm's length so it never overstretches. Pure geometry, GPU-independent.

    ``skeleton`` is ``[(name, (x,y,z)), ...]``; returns the same shape with the arm links re-placed.
    """
    if target is None or reach_frac <= 0.0:
        return skeleton
    if str(arm).lower() == "both":
        out = skeleton
        for side in ("left", "right"):
            out = compute_arm_reach_skeleton(
                out,
                target,
                reach_frac,
                arm=side,
                forward_yaw=forward_yaw,
            )
        return out
    arm_keys = ("shoulder", "elbow", "wrist", "hand")
    prefix = f"{arm}_"
    arm_pts = [(n, p) for n, p in skeleton if n.startswith(prefix) and any(k in n for k in arm_keys)]
    sh = [p for n, p in arm_pts if "shoulder" in n]
    hand = [p for n, p in arm_pts if "hand" in n]
    if not sh or not hand:
        return skeleton

    def centroid(ps):
        return tuple(sum(c) / len(ps) for c in zip(*ps))

    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def add(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def scale(a, s):
        return (a[0] * s, a[1] * s, a[2] * s)

    def length(a):
        return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

    shoulder = centroid(sh)
    hand_rest = centroid(hand)
    arm_len = length(sub(hand_rest, shoulder)) or 1e-6
    reach_target = _manipulation_arm_target_for_reach_fraction(
        shoulder,
        target,
        reach_frac,
        forward_yaw=forward_yaw,
    )
    to_target = sub(reach_target, shoulder)
    tlen = length(to_target) or 1e-6
    reach_dist = min(arm_len, tlen)
    hand_reach = add(shoulder, scale(to_target, reach_dist / tlen))  # clamped along shoulder->target
    frac = max(0.0, min(1.0, float(reach_frac)))
    hand_now = add(scale(hand_rest, 1.0 - frac), scale(hand_reach, frac))
    out = []
    for n, p in skeleton:
        if n.startswith(prefix) and any(k in n for k in arm_keys):
            f = length(sub(p, shoulder)) / arm_len  # rest fractional distance along the arm
            out.append((n, add(shoulder, scale(sub(hand_now, shoulder), f))))
        else:
            out.append((n, p))
    return out


MANIPULATION_POV_REACH_RAMP_START_FRACTION = 0.65


def manipulation_reach_fraction_for_frame(
    alpha: float,
    *,
    manipulation_cam: bool,
    frame_count: int,
) -> float:
    """Return temporal reach conditioning for a rendered manipulation frame.

    Manipulation-stand clips keep the pelvis fixed by design. The reach pose must still vary across
    multi-frame clips so rendered review videos and OSCAR conditioning are not silent seed-only
    repeats. Start from a visible forward-ready arm for robot POV framing, then finish at full reach.
    """
    if frame_count <= 1:
        return 1.0 if manipulation_cam else 0.0
    a = max(0.0, min(1.0, float(alpha)))
    if not manipulation_cam:
        return a
    start = float(MANIPULATION_POV_REACH_RAMP_START_FRACTION)
    return max(0.0, min(1.0, start + (1.0 - start) * a))


def arm_reach_rotation(shoulder, rest_elbow, target, reach_frac):
    """Axis (unit xyz) + angle (radians) of the kinematic SHOULDER rotation that swings the rest
    upper-arm bone (shoulder->rest_elbow) toward the target (shoulder->target), scaled by reach_frac.

    Axis-agnostic: it derives the rotation from the rest bone and the desired bone direction, so there
    is NO hardcoded joint axis to inspect on the G1 USD. Applied about the shoulder pivot it points the
    upper arm at the object; the elbow/wrist/hand follow rigidly. Pure geometry, GPU-independent."""
    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def length(a):
        return math.sqrt(dot(a, a))

    def norm(a):
        L = length(a) or 1e-9
        return (a[0] / L, a[1] / L, a[2] / L)

    rest = norm(sub(rest_elbow, shoulder))
    want = norm(sub(target, shoulder))
    d = max(-1.0, min(1.0, dot(rest, want)))
    angle = math.acos(d) * max(0.0, min(1.0, float(reach_frac)))
    axis = cross(rest, want)
    if length(axis) < 1e-6:
        axis = (0.0, 0.0, 1.0)  # rest ~parallel/antiparallel to want -> arbitrary axis (angle ~0/pi)
    return norm(axis), angle


def _find_arm_link(links: dict, *keys: str):
    """First Xformable link prim whose name contains ALL keys (case-insensitive)."""
    for name, prim in links.items():
        low = name.lower()
        if all(k.lower() in low for k in keys):
            return prim
    return None


def _is_manipulation_arm_link_name(name: str, side: str) -> bool:
    """Return whether a robot link name belongs to the requested manipulation arm side."""
    low = str(name or "").lower()
    side_low = str(side or "").lower()
    return (
        "link" in low
        and side_low in low
        and any(token in low for token in MANIPULATION_ARM_LINK_NAME_TOKENS)
    )


def _arm_link_prims_for_side(links: Mapping[str, Any], side: str) -> list[Any]:
    out = [
        prim
        for name, prim in links.items()
        if _is_manipulation_arm_link_name(name, side)
    ]
    out.sort(key=lambda prim: str(prim.GetPath()))
    return out


def _pose_arm_kinematic_usd(
    stage,
    prim_path: str,
    target,
    *,
    arm: str = "right",
    reach_frac: float = 1.0,
    forward_yaw: float | None = None,
) -> int:
    """Kinematically pose the G1 arm(s) into the current manipulation reach phase.

    Pure USD: rotate the requested arm link set about the shoulder pivot so the
    shoulder->effector direction points into the current reach target. Some G1 USD variants do not place
    elbow/wrist/hand links below the shoulder link in an ordinary transform hierarchy, so rotating
    only the shoulder can report success while the visible/measured hand stays in rest pose. This path
    therefore authors target world transforms for every actual side-arm link prim and then verifies
    that the measured effector link moved. Endpoint aiming is review geometry, not contact or task
    completion proof. No physics tensor view is used.
    """
    from pxr import Usd, UsdGeom, Gf  # type: ignore
    sides = ("left", "right") if arm == "both" else (arm,)
    root = stage.GetPrimAtPath(prim_path)
    links = {p.GetName(): p for p in Usd.PrimRange(root)
             if p.IsA(UsdGeom.Xformable) and "link" in p.GetName().lower()}
    posed = 0
    for side in sides:
        shoulder = (_find_arm_link(links, side, "shoulder", "pitch")
                    or _find_arm_link(links, side, "shoulder"))
        # Align shoulder->effector with the current reach target. The effector is not translated
        # beyond its arm span; final distance remains a measured success-gate input.
        effector = (_find_arm_link(links, side, "hand") or _find_arm_link(links, side, "palm")
                    or _find_arm_link(links, side, "wrist") or _find_arm_link(links, side, "elbow"))
        if shoulder is None or effector is None:
            continue
        xc = UsdGeom.XformCache()  # fresh cache per arm (previous arm's mutation invalidated it)
        sh_w = xc.GetLocalToWorldTransform(shoulder)
        el_w = xc.GetLocalToWorldTransform(effector)
        sp = sh_w.ExtractTranslation()
        ep = el_w.ExtractTranslation()
        shoulder_xyz = (float(sp[0]), float(sp[1]), float(sp[2]))
        reach_target = _manipulation_arm_target_for_reach_fraction(
            shoulder_xyz,
            target,
            reach_frac,
            forward_yaw=forward_yaw,
        )
        axis, angle = arm_reach_rotation(
            shoulder_xyz,
            (float(ep[0]), float(ep[1]), float(ep[2])),
            reach_target,
            reach_frac,
        )
        if angle < 1e-4:
            continue
        rot = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(*axis), math.degrees(angle)))
        pivot = Gf.Vec3d(sp[0], sp[1], sp[2])
        # Rotate the link world transforms about the shoulder pivot (USD row-vector convention).
        m_pivot = Gf.Matrix4d().SetTranslate(-pivot) * rot * Gf.Matrix4d().SetTranslate(pivot)
        arm_link_prims = _arm_link_prims_for_side(links, side)
        if shoulder not in arm_link_prims:
            arm_link_prims.append(shoulder)
        if effector not in arm_link_prims:
            arm_link_prims.append(effector)
        arm_link_prims = sorted(
            {str(prim.GetPath()): prim for prim in arm_link_prims}.values(),
            key=lambda prim: str(prim.GetPath()),
        )
        old_world: dict[str, Any] = {}
        old_parent_world: dict[str, Any] = {}
        for prim in arm_link_prims:
            path = str(prim.GetPath())
            old_world[path] = xc.GetLocalToWorldTransform(prim)
            old_parent_world[path] = xc.GetLocalToWorldTransform(prim.GetParent())
        target_world = {
            path: matrix * m_pivot
            for path, matrix in old_world.items()
        }
        for prim in sorted(arm_link_prims, key=lambda p: str(p.GetPath()).count("/")):
            path = str(prim.GetPath())
            parent_path = str(prim.GetParent().GetPath())
            parent_world = target_world.get(parent_path) or old_parent_world[path]
            new_local = target_world[path] * parent_world.GetInverse()
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(new_local)
        moved_xc = UsdGeom.XformCache()
        moved = moved_xc.GetLocalToWorldTransform(effector).ExtractTranslation()
        moved_dist = math.sqrt(
            (float(moved[0]) - float(ep[0])) ** 2
            + (float(moved[1]) - float(ep[1])) ** 2
            + (float(moved[2]) - float(ep[2])) ** 2
        )
        if moved_dist < MANIPULATION_ARM_POSE_MIN_LINK_MOVE_M:
            continue
        posed += 1
    return posed


def _capture_robot_neutral_descendant_xforms(stage, prim_path: str) -> dict[str, Any]:
    """Capture robot descendant local transforms before any per-task arm posing mutates the stage.

    Warm workers reuse one USD stage for many jobs. Any pure-USD reach pose must therefore be applied
    from the same neutral robot seed every time, otherwise arm/link transforms compound across jobs.
    Root placement is intentionally excluded; it is re-authored from the dynamic stance each frame.
    """
    from pxr import Usd, UsdGeom  # type: ignore
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        return {}
    xc = UsdGeom.XformCache()
    neutral: dict[str, Any] = {}
    for prim in Usd.PrimRange(root):
        if prim.GetPath() == root.GetPath():
            continue
        try:
            if not prim.IsA(UsdGeom.Xformable):
                continue
            parent_world = xc.GetLocalToWorldTransform(prim.GetParent())
            world = xc.GetLocalToWorldTransform(prim)
            neutral[str(prim.GetPath())] = world * parent_world.GetInverse()
        except Exception:  # noqa: BLE001
            continue
    return neutral


def _restore_robot_neutral_descendant_xforms(stage, neutral_xforms: Mapping[str, Any]) -> int:
    """Restore descendant local transforms captured by
    :func:`_capture_robot_neutral_descendant_xforms`.
    """
    if not neutral_xforms:
        return 0
    from pxr import UsdGeom  # type: ignore
    restored = 0
    for path, local_matrix in neutral_xforms.items():
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        try:
            if not prim.IsA(UsdGeom.Xformable):
                continue
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(local_matrix)
            restored += 1
        except Exception:  # noqa: BLE001
            continue
    return restored


def _setup_articulated_g1(prim_path: str, *, gravity_z: float = 0.0):
    """Create a physics SimulationContext (gravity OFF, so the kinematic walk pose holds without the
    G1 collapsing), play it, and initialize the G1 articulation for joint driving + link readback.
    Returns a context dict. GPU-only."""
    from isaacsim.core.api import SimulationContext  # type: ignore
    ctx = SimulationContext(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
    ctx.initialize_physics()
    try:
        ctx.get_physics_context().set_gravity(float(gravity_z))
    except Exception:  # noqa: BLE001
        pass
    art, dof_index, default, link_names = _setup_g1_articulation(prim_path)
    ctx.play()
    return {"ctx": ctx, "art": art, "dof_index": dof_index, "default": default,
            "link_names": link_names, "dof_count": len(dof_index), "gravity_z": float(gravity_z)}


def _setup_physics_context_only(*, gravity_z: float = -9.81):
    """Create/play a PhysX SimulationContext without creating a SingleArticulation tensor view.

    The official G1 USD currently invalidates Isaac's tensor view when this runner drives or reads
    the articulation through SingleArticulation. Dynamic-standing contact proof only needs the
    authored USD articulation, gravity, collisions, and contact reporting, so this mode avoids the
    tensor API entirely.
    """
    from isaacsim.core.api import SimulationContext  # type: ignore
    ctx = SimulationContext(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
    ctx.initialize_physics()
    try:
        ctx.get_physics_context().set_gravity(float(gravity_z))
    except Exception:  # noqa: BLE001
        pass
    ctx.play()
    return {
        "ctx": ctx,
        "art": None,
        "dof_index": {},
        "default": [],
        "link_names": [],
        "dof_count": 0,
        "gravity_z": float(gravity_z),
        "tensor_view_used": False,
    }


def _sim_step(ctx, *, render: bool = False) -> None:
    try:
        ctx.step(render=render)
    except TypeError:
        ctx.step()


def _safe_articulation_world_pose(art) -> dict[str, Any]:
    try:
        pos, quat = art.get_world_pose()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": repr(exc)}
    try:
        return {
            "available": True,
            "position_xyz": [round(float(v), 6) for v in pos],
            "orientation_wxyz": [round(float(v), 6) for v in quat],
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": repr(exc)}


def _safe_usd_root_world_pose(stage, prim_path: str) -> dict[str, Any]:
    try:
        from pxr import UsdGeom  # type: ignore
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return {"available": False, "error": "prim_not_found"}
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        pos = transform.ExtractTranslation()
        return {
            "available": True,
            "position_xyz": [round(float(pos[i]), 6) for i in range(3)],
            "source": "usd_xform_cache",
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": repr(exc)}


def _root_displacement_metrics(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        if not before.get("available") or not after.get("available"):
            raise ValueError("root_pose_unavailable")
        before_pos = before.get("position_xyz")
        after_pos = after.get("position_xyz")
        if (
            not isinstance(before_pos, Sequence)
            or isinstance(before_pos, (str, bytes))
            or not isinstance(after_pos, Sequence)
            or isinstance(after_pos, (str, bytes))
            or len(before_pos) < 3
            or len(after_pos) < 3
        ):
            raise ValueError("root_position_xyz_unavailable")
        delta = [float(after_pos[index]) - float(before_pos[index]) for index in range(3)]
        displacement = math.sqrt(sum(value * value for value in delta))
        vertical_drop = max(0.0, float(before_pos[2]) - float(after_pos[2]))
        return {
            "available": True,
            "root_displacement_m": round(float(displacement), 6),
            "root_vertical_drop_m": round(float(vertical_drop), 6),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "root_displacement_m": 0.0,
            "root_vertical_drop_m": 0.0,
            "error": str(exc),
        }


def _path_from_encoded_sdf(value) -> str:
    try:
        from pxr import PhysicsSchemaTools  # type: ignore
        return str(PhysicsSchemaTools.intToSdfPath(int(value)))
    except Exception:  # noqa: BLE001
        return str(value)


def _vec3_to_list(value) -> list[float] | None:
    try:
        return [round(float(value[i]), 6) for i in range(3)]
    except Exception:  # noqa: BLE001
        try:
            return [round(float(getattr(value, axis)), 6) for axis in ("x", "y", "z")]
        except Exception:  # noqa: BLE001
            return None


def _enable_contact_reports(stage, robot_prim_path: str, *, threshold: float = 0.0) -> dict[str, Any]:
    """Apply PhysX contact-report API to the articulation/root and likely foot links.

    This is best-effort because Isaac worker images and G1 USD variants differ. A failure should
    block only the contact report, not the render path.
    """
    try:
        from pxr import PhysxSchema, Usd, UsdPhysics  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"status": "unavailable", "error": repr(exc), "enabled_paths": []}
    root = stage.GetPrimAtPath(robot_prim_path)
    if not root or not root.IsValid():
        return {"status": "unavailable", "error": "robot_prim_not_found", "enabled_paths": []}
    enabled: list[str] = []
    candidates = []
    for prim in Usd.PrimRange(root):
        name = prim.GetName().lower()
        if (
            prim.GetPath() == root.GetPath()
            or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            or prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or ("foot" in name and prim.HasAPI(UsdPhysics.CollisionAPI))
        ):
            candidates.append(prim)
    for prim in candidates:
        try:
            api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            api.CreateThresholdAttr().Set(float(threshold))
            enabled.append(str(prim.GetPath()))
        except Exception:  # noqa: BLE001
            continue
    return {"status": "enabled" if enabled else "unavailable", "enabled_paths": enabled}


def _contact_report_records(robot_prim_path: str, *, max_records: int = 40) -> list[dict[str, Any]]:
    try:
        from omni.physx import get_physx_simulation_interface  # type: ignore
    except Exception:  # noqa: BLE001
        return []
    try:
        report = get_physx_simulation_interface().get_contact_report()
    except Exception:  # noqa: BLE001
        return []
    if not report:
        return []
    try:
        headers, data = report[0], report[1] if len(report) > 1 else []
    except Exception:  # noqa: BLE001
        return []
    records: list[dict[str, Any]] = []
    for header in list(headers)[:max_records]:
        actor0 = _path_from_encoded_sdf(getattr(header, "actor0", ""))
        actor1 = _path_from_encoded_sdf(getattr(header, "actor1", ""))
        collider0 = _path_from_encoded_sdf(getattr(header, "collider0", actor0))
        collider1 = _path_from_encoded_sdf(getattr(header, "collider1", actor1))
        joined = " ".join((actor0, actor1, collider0, collider1)).lower()
        if robot_prim_path.lower() not in joined and "/world/g1" not in joined:
            continue
        offset = int(getattr(header, "contact_data_offset", 0) or 0)
        count = int(getattr(header, "num_contact_data", 0) or 0)
        samples = []
        for sample in list(data)[offset: offset + min(count, 3)]:
            samples.append({
                "position_xyz": _vec3_to_list(getattr(sample, "position", None)),
                "normal_xyz": _vec3_to_list(getattr(sample, "normal", None)),
                "impulse": (
                    round(float(getattr(sample, "impulse", 0.0) or 0.0), 6)
                    if hasattr(sample, "impulse") else None
                ),
            })
        records.append({
            "actor0": actor0,
            "actor1": actor1,
            "collider0": collider0,
            "collider1": collider1,
            "contact_data_count": count,
            "samples": samples,
        })
    return records


def _is_support_contact(record: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(record.get(key) or "").lower()
        for key in ("actor0", "actor1", "collider0", "collider1")
    )
    return ("foot" in text or "ankle" in text or "toe" in text) and (
        "floor" in text or "ground" in text or "room" in text or "kitchen" in text
    )


def _settle_dynamic_standing_contacts(
    *,
    stage,
    art_ctx,
    robot_prim_path: str,
    root_pose,
    yaw,
    phase,
    moving,
    settle_steps: int,
    scenario_id: str,
    manipulation_ready: bool = False,
    manipulation_reach_arm: str = "both",
    root_pose_seeded_before_tensor_view: bool = True,
    effort_drive: bool = False,
    effort_kp=DEFAULT_ISAAC_LEG_KP,
    effort_kd=DEFAULT_ISAAC_LEG_KD,
) -> dict[str, Any]:
    """Run a bounded PhysX standing/contact settle without mutating the G1 USD xform after the
    articulation tensor view exists.

    The policy route remains kinematic. This mode upgrades each sampled placement by stepping the
    real articulation against the scene with gravity and contact reporting; it is not a full dynamic
    walking controller.
    """
    art = art_ctx.get("art")
    ctx = art_ctx["ctx"]
    tensor_view_used = art is not None
    targets = None
    command_mode = "usd_physx_articulation_default_drives_no_tensor_view"
    actuator_output_mode = "position_target"
    effort_blockers: list[str] = []
    if effort_drive and not tensor_view_used:
        actuator_output_mode = "position_target_fallback"
        effort_blockers.append("effort_drive_requested_without_tensor_view")
    if tensor_view_used:
        targets = _joint_targets_for_pose(
            art_ctx["default"],
            art_ctx["dof_index"],
            phase=phase,
            moving=moving,
            manipulation_ready=manipulation_ready,
            manipulation_reach_arm=manipulation_reach_arm,
        )
        # Do not call art.set_world_pose() here. On the official G1 USD that invalidates the
        # PhysX tensor view with the same failure as mutating the USD root xform after initialization.
        # Dynamic-standing mode pre-seeds the root USD transform before the physics context is played.
        if effort_drive:
            try:
                command_mode = _apply_articulation_joint_efforts(
                    art,
                    _pd_leg_joint_efforts(
                        targets,
                        art.get_joint_positions(),
                        art.get_joint_velocities(),
                        kp=effort_kp,
                        kd=effort_kd,
                    ),
                )
                actuator_output_mode = "effort"
            except Exception as exc:  # noqa: BLE001
                effort_blockers.append(f"effort_drive_initial_command_failed:{exc!r}")
                command_mode = _apply_articulation_joint_targets(art, targets)
                actuator_output_mode = "position_target_fallback"
        else:
            command_mode = _apply_articulation_joint_targets(art, targets)
    before = (
        _safe_articulation_world_pose(art)
        if tensor_view_used else _safe_usd_root_world_pose(stage, robot_prim_path)
    )
    contact_setup = _enable_contact_reports(stage, robot_prim_path)
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    executed = 0
    for _ in range(max(0, int(settle_steps))):
        try:
            if tensor_view_used:
                if effort_drive and not effort_blockers:
                    try:
                        command_mode = _apply_articulation_joint_efforts(
                            art,
                            _pd_leg_joint_efforts(
                                targets,
                                art.get_joint_positions(),
                                art.get_joint_velocities(),
                                kp=effort_kp,
                                kd=effort_kd,
                            ),
                        )
                        actuator_output_mode = "effort"
                    except Exception as exc:  # noqa: BLE001
                        effort_blockers.append(f"effort_drive_step_command_failed:{exc!r}")
                        command_mode = _apply_articulation_joint_targets(art, targets)
                        actuator_output_mode = "position_target_fallback"
                else:
                    _apply_articulation_joint_targets(art, targets)
            _sim_step(ctx, render=False)
            executed += 1
            if len(records) < 80:
                records.extend(_contact_report_records(robot_prim_path, max_records=20))
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))
            break
    after = (
        _safe_articulation_world_pose(art)
        if tensor_view_used else _safe_usd_root_world_pose(stage, robot_prim_path)
    )
    metrics = _root_displacement_metrics(before, after)
    gravity_on = float(art_ctx.get("gravity_z") or 0.0) < 0.0
    physics_integrated = bool(executed > 0 and metrics.get("available") and gravity_on)
    root_displacement_m = float(metrics.get("root_displacement_m") or 0.0)
    root_vertical_drop_m = float(metrics.get("root_vertical_drop_m") or 0.0)
    if not metrics.get("available"):
        dynamic_settle_verdict = "unknown"
    elif root_vertical_drop_m > ROOT_FALL_VERTICAL_DROP_M:
        dynamic_settle_verdict = "fell"
    elif (
        root_vertical_drop_m >= ROOT_DRIFT_VERTICAL_DROP_M
        or root_displacement_m > ROOT_DRIFT_DISPLACEMENT_M
    ):
        dynamic_settle_verdict = "drifted"
    elif executed == 0:
        dynamic_settle_verdict = "no_motion"
    else:
        dynamic_settle_verdict = "stable"
    support_records = [r for r in records if _is_support_contact(r)]
    report = {
        "schema_version": "isaac_g1_physics_articulation_standing_contact_report.v1",
        "status": "completed" if executed == max(0, int(settle_steps)) and not errors else "blocked",
        "scenario_id": scenario_id,
        "gravity_z": art_ctx.get("gravity_z"),
        "gravity_on": gravity_on,
        "physics_integrated": physics_integrated,
        "root_displacement_m": root_displacement_m,
        "root_vertical_drop_m": root_vertical_drop_m,
        "dynamic_settle_verdict": dynamic_settle_verdict,
        "tensor_view_used": tensor_view_used,
        "requested_settle_steps": int(settle_steps),
        "executed_settle_steps": executed,
        "seed_root_pose_xyz": [round(float(v), 6) for v in root_pose],
        "seed_root_yaw_rad": round(float(yaw), 6),
        "root_pose_seeded_before_tensor_view": bool(root_pose_seeded_before_tensor_view),
        "root_pose_seeded_once_before_settle": False,
        "root_pose_teleport_during_physics_settle": False,
        "usd_root_xform_mutated_after_tensor_view": False,
        "joint_command_mode": command_mode,
        "contact_report_setup": contact_setup,
        "contact_event_count": len(records),
        "support_contact_event_count": len(support_records),
        "sample_contact_records": records[:20],
        "root_pose_before_settle": before,
        "root_pose_after_settle": after,
        "errors": errors,
        "claim_boundary": (
            "Physics articulation standing/contact settle for this sampled placement only; not "
            "full dynamic walking, learned balance control, task success, safety validation, or "
            "deployment readiness."
        ),
    }
    if effort_drive:
        def _jsonable_gain(value):  # noqa: ANN001
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return [float(v) for v in value]
            try:
                return float(value)
            except Exception:  # noqa: BLE001
                return repr(value)

        report.update({
            "actuator_output_mode": actuator_output_mode,
            "pd_gains": {
                "kp": _jsonable_gain(effort_kp),
                "kd": _jsonable_gain(effort_kd),
            },
            "effort_drive_blockers": effort_blockers,
        })
    return report


def _overlap_probe(robot_prim_path: str, ground_prim_path: str = "/World/GroundPlane"):
    """Return probe(pose, yaw) -> scene-collision hit count using a PhysX box overlap of the
    robot footprint at the candidate pose, excluding the robot's own prims and the ground."""
    from omni.physx import get_physx_scene_query_interface  # type: ignore
    import carb  # type: ignore

    sqi = get_physx_scene_query_interface()
    hx, hy, hz = ROBOT_FOOTPRINT_HALF_EXTENT

    def probe(pose, yaw) -> int:
        hits = {"n": 0}

        def report(hit):  # noqa: ANN001
            path = str(getattr(hit, "collision", "") or getattr(hit, "rigid_body", ""))
            if not path.startswith(robot_prim_path) and not path.startswith(ground_prim_path):
                hits["n"] += 1
            return True  # keep scanning

        w, x, y, z = yaw_to_quat(float(yaw))
        sqi.overlap_box(
            carb.Float3(hx, hy, hz),
            carb.Float3(float(pose[0]), float(pose[1]), float(pose[2])),
            carb.Float4(x, y, z, w),  # PhysX quat order is (x,y,z,w)
            report, False,
        )
        return hits["n"]

    return probe


def _set_root_xform(stage, prim_path: str, pose, yaw) -> None:
    from pxr import UsdGeom, Gf  # type: ignore

    xform = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2])))
    xform.AddRotateZOp().Set(math.degrees(float(yaw)))


def _matrix4_to_rows(matrix) -> list[list[float]]:
    rows: list[list[float]] = []
    for i in range(4):
        rows.append([round(float(matrix[i][j]), 6) for j in range(4)])
    return rows


def _xform_op_record(op) -> dict[str, Any]:
    try:
        value = op.Get()
    except Exception as exc:  # noqa: BLE001
        value = repr(exc)
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        try:
            value_out: Any = [round(float(value[i]), 6) for i in range(len(value))]
        except Exception:  # noqa: BLE001
            value_out = str(value)
    else:
        try:
            value_out = round(float(value), 6)
        except Exception:  # noqa: BLE001
            value_out = str(value)
    return {
        "op_name": str(op.GetOpName()),
        "op_type": str(op.GetOpType()),
        "is_inverse": bool(op.IsInverseOp()),
        "value": value_out,
    }


def _prim_transform_snapshot(stage, prim_path: str) -> dict[str, Any]:
    from pxr import UsdGeom  # type: ignore

    prim = stage.GetPrimAtPath(prim_path)
    valid = bool(prim and prim.IsValid())
    snap: dict[str, Any] = {"prim_path": prim_path, "valid": valid}
    if not valid:
        return snap
    try:
        xform = UsdGeom.Xformable(prim)
        snap["xform_ops"] = [_xform_op_record(op) for op in xform.GetOrderedXformOps()]
    except Exception as exc:  # noqa: BLE001
        snap["xform_ops_error"] = repr(exc)
    try:
        cache = UsdGeom.XformCache()
        snap["local_to_world_matrix"] = _matrix4_to_rows(cache.GetLocalToWorldTransform(prim))
    except Exception as exc:  # noqa: BLE001
        snap["local_to_world_error"] = repr(exc)
    return snap


def _root_transform_diagnostics(stage, prim_path: str) -> dict[str, Any]:
    prim = stage.GetPrimAtPath(prim_path)
    parent_path = ""
    if prim and prim.IsValid():
        parent = prim.GetParent()
        if parent and parent.IsValid():
            parent_path = str(parent.GetPath())
    return {
        "root": _prim_transform_snapshot(stage, prim_path),
        "parent": _prim_transform_snapshot(stage, parent_path) if parent_path else None,
        "pseudo_root": _prim_transform_snapshot(stage, str(stage.GetPseudoRoot().GetPath())),
    }


def _robot_upright_report(stage, prim_path: str, *, max_tilt_deg: float = 12.0) -> dict[str, Any]:
    """Verify the placed robot root is upright enough for a policy seed frame."""
    if not hasattr(stage, "GetPrimAtPath"):
        return {
            "schema_version": "robot_upright_report.v1",
            "status": "unverified",
            "blockers": [],
            "prim_path": prim_path,
            "reason": "stage_object_does_not_expose_usd_prim_lookup",
        }
    try:
        from pxr import UsdGeom, Gf  # type: ignore
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return {
                "schema_version": "robot_upright_report.v1",
                "status": "blocked",
                "blockers": ["robot_upright_prim_unavailable"],
                "prim_path": prim_path,
            }
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        try:
            up = matrix.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
            up_vec = (float(up[0]), float(up[1]), float(up[2]))
        except Exception:  # noqa: BLE001
            up_vec = (float(matrix[0][2]), float(matrix[1][2]), float(matrix[2][2]))
        norm = math.sqrt(sum(v * v for v in up_vec)) or 1e-9
        up_unit = tuple(v / norm for v in up_vec)
        cos_tilt = max(-1.0, min(1.0, up_unit[2]))
        tilt_deg = math.degrees(math.acos(cos_tilt))
        blockers = [] if tilt_deg <= float(max_tilt_deg) else ["robot_root_not_upright"]
        return {
            "schema_version": "robot_upright_report.v1",
            "status": "passed" if not blockers else "blocked",
            "blockers": blockers,
            "prim_path": prim_path,
            "root_up_vector_world": [round(float(v), 6) for v in up_unit],
            "tilt_deg": round(float(tilt_deg), 4),
            "max_tilt_deg": round(float(max_tilt_deg), 4),
            "claim_boundary": (
                "Upright validation checks root orientation for an initial policy seed only; it is not "
                "dynamic balance, locomotion, safety, or deployment validation."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "schema_version": "robot_upright_report.v1",
            "status": "blocked",
            "blockers": ["robot_upright_report_failed"],
            "prim_path": prim_path,
            "error": repr(exc),
        }


def _world_bbox_for_prim(stage, prim_path: str) -> dict[str, list[float]] | None:
    """Compute an aligned world bbox for a prim after its current stage transform is applied."""
    from pxr import Usd, UsdGeom  # type: ignore

    prim = stage.GetPrimAtPath(prim_path)
    is_valid = getattr(prim, "IsValid", None)
    if prim is None or (callable(is_valid) and not is_valid()):
        return None
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if box.IsEmpty():
        return None
    bbox_min, bbox_max, center, size = _aligned_box_min_max_center_size(box)
    return {
        "bbox_min_xyz": _rounded_xyz(bbox_min),
        "bbox_max_xyz": _rounded_xyz(bbox_max),
        "center_xyz": _rounded_xyz(center),
        "size_xyz": _rounded_xyz(size),
    }


def _footprint_center_xy_from_bbox(bbox: Mapping[str, Sequence[float]]) -> list[float]:
    bmin = bbox["bbox_min_xyz"]
    bmax = bbox["bbox_max_xyz"]
    return [
        round(0.5 * (float(bmin[0]) + float(bmax[0])), 6),
        round(0.5 * (float(bmin[1]) + float(bmax[1])), 6),
    ]


def _xy_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _place_root(
    stage,
    prim_path: str,
    pose,
    yaw,
    *,
    align_footprint_center: bool = True,
    max_xy_error_m: float = PLACEMENT_GROUND_TRUTH_MAX_FOOTPRINT_CENTER_DELTA_M,
) -> dict[str, Any]:
    """Place the G1 root and align its actual USD footprint center to ``pose``.

    The G1 asset root is not assumed to be the pelvis/footprint center. We first apply the requested
    transform, query the actual world AABB with ``UsdGeom.BBoxCache``, and, when the footprint center
    misses the requested pose by more than ``max_xy_error_m``, translate the root by that measured
    world-frame offset. Callers that ignore the return value still get corrected placement; the
    diagnostics are written into ``placement_validation.json`` by the manipulation runner.
    """
    requested_pose = (float(pose[0]), float(pose[1]), float(pose[2]))
    requested_xy = [requested_pose[0], requested_pose[1]]
    _set_root_xform(stage, prim_path, requested_pose, yaw)
    diagnostics: dict[str, Any] = {
        "schema_version": "place_root_diagnostics.v1",
        "status": "placed",
        "prim_path": prim_path,
        "requested_pose_xyz": _rounded_xyz(requested_pose),
        "requested_yaw_rad": round(float(yaw), 6),
        "max_xy_error_m": round(float(max_xy_error_m), 6),
        "footprint_center_alignment_enabled": bool(align_footprint_center),
        "correction_applied": False,
    }
    if not align_footprint_center:
        return diagnostics
    initial_bbox = _world_bbox_for_prim(stage, prim_path)
    if initial_bbox is None:
        diagnostics["status"] = "bbox_unavailable"
        diagnostics["blockers"] = ["placed_robot_bbox_unavailable"]
        diagnostics["xform_diagnostics"] = _root_transform_diagnostics(stage, prim_path)
        return diagnostics
    initial_center = _footprint_center_xy_from_bbox(initial_bbox)
    initial_error = _xy_distance(initial_center, requested_xy)
    diagnostics["initial_world_aabb"] = initial_bbox
    diagnostics["initial_footprint_center_xy"] = initial_center
    diagnostics["initial_xy_error_m"] = round(float(initial_error), 6)
    if initial_error <= max_xy_error_m:
        diagnostics["final_world_aabb"] = initial_bbox
        diagnostics["final_footprint_center_xy"] = initial_center
        diagnostics["final_xy_error_m"] = diagnostics["initial_xy_error_m"]
        return diagnostics

    dx = float(initial_center[0]) - requested_pose[0]
    dy = float(initial_center[1]) - requested_pose[1]
    corrected_pose = (requested_pose[0] - dx, requested_pose[1] - dy, requested_pose[2])
    diagnostics["status"] = "corrected"
    diagnostics["correction_applied"] = True
    diagnostics["measured_offset_xy_m"] = [round(dx, 6), round(dy, 6)]
    diagnostics["corrected_root_translation_xyz"] = _rounded_xyz(corrected_pose)
    diagnostics["xform_diagnostics_before_correction"] = _root_transform_diagnostics(stage, prim_path)
    _set_root_xform(stage, prim_path, corrected_pose, yaw)
    final_bbox = _world_bbox_for_prim(stage, prim_path)
    if final_bbox is None:
        diagnostics["status"] = "blocked"
        diagnostics["blockers"] = ["placed_robot_bbox_unavailable_after_correction"]
        diagnostics["xform_diagnostics_after_correction"] = _root_transform_diagnostics(stage, prim_path)
        return diagnostics
    final_center = _footprint_center_xy_from_bbox(final_bbox)
    final_error = _xy_distance(final_center, requested_xy)
    diagnostics["final_world_aabb"] = final_bbox
    diagnostics["final_footprint_center_xy"] = final_center
    diagnostics["final_xy_error_m"] = round(float(final_error), 6)
    diagnostics["xform_diagnostics_after_correction"] = _root_transform_diagnostics(stage, prim_path)
    if final_error > max_xy_error_m:
        diagnostics["status"] = "blocked"
        diagnostics["blockers"] = ["placed_robot_footprint_center_mismatch_after_correction"]
    return diagnostics


def _placement_validator_for_stage(
    stage,
    robot_prim_path: str,
    target_bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    *,
    target_object=None,
    scene_objects: Sequence[Any] | None = None,
    floor_z: float | None = None,
    max_root_to_bbox_center_xy_m: float = PLACEMENT_GROUND_TRUTH_MAX_FOOTPRINT_CENTER_DELTA_M,
    min_target_gap_m: float = 0.05,
    standoff_tolerance_m: float = 0.2,
    standoff_range: tuple[float, float] | None = None,
):
    """Validate the placed G1 geometry, not just the planned root pose.

    This catches the kitchen failure mode where the stance planner reports an aisle root but the
    referenced G1 asset lands with its visible/collidable bbox in the sink or cabinet footprint.
    """
    target_min, target_max = target_bounds
    target_center = (
        0.5 * (float(target_min[0]) + float(target_max[0])),
        0.5 * (float(target_min[1]) + float(target_max[1])),
        0.5 * (float(target_min[2]) + float(target_max[2])),
    )
    validation_standoff_range = (
        tuple(float(v) for v in standoff_range)
        if standoff_range is not None
        else TASK_STANCE_DEFAULT_VALIDATION_STANDOFF_RANGE_M
    )

    def validate(pose, yaw, record: Mapping[str, Any] | None = None) -> dict[str, Any]:
        root_diagnostics = _place_root(stage, robot_prim_path, pose, yaw)
        bbox = _world_bbox_for_prim(stage, robot_prim_path)
        result: dict[str, Any] = {
            "status": "accepted",
            "blockers": [],
            "robot_prim_path": robot_prim_path,
            "planned_pose": _rounded_xyz(pose),
            "planned_yaw": round(float(yaw), 6),
            "max_root_to_bbox_center_xy_m": round(float(max_root_to_bbox_center_xy_m), 6),
            "min_target_gap_m": round(float(min_target_gap_m), 6),
            "validation_standoff_range_m": [
                round(float(validation_standoff_range[0]), 6),
                round(float(validation_standoff_range[1]), 6),
            ],
            "place_root_diagnostics": root_diagnostics,
        }
        if bbox is None:
            result["status"] = "blocked"
            result["blockers"] = ["placed_robot_bbox_unavailable"]
            return result
        result["placed_robot_bbox"] = bbox
        bbox_center = bbox["center_xyz"]
        root_to_center_xy = math.hypot(
            float(bbox_center[0]) - float(pose[0]),
            float(bbox_center[1]) - float(pose[1]),
        )
        relation = _xy_rect_overlap_and_gap(
            bbox["bbox_min_xyz"],
            bbox["bbox_max_xyz"],
            target_min,
            target_max,
        )
        result["root_to_robot_bbox_center_xy_m"] = round(float(root_to_center_xy), 6)
        result["target_bbox_relation"] = relation
        blockers: list[str] = []
        deterministic_geometry_ok: bool | None = None
        if target_object is not None and scene_objects is not None:
            try:
                from blueprint_pipeline.scene_placement import validate_stand_pose  # type: ignore
            except Exception:
                from scene_placement import validate_stand_pose  # type: ignore
            deterministic_floor_z = (
                float(floor_z)
                if floor_z is not None
                else float(pose[2]) - ROBOT_PELVIS_HEIGHT_M
            )
            obstacles = list(scene_objects)
            if not any(
                str(getattr(o, "id", "")) == str(getattr(target_object, "id", ""))
                for o in obstacles
            ):
                obstacles.append(target_object)
            verdict = validate_stand_pose(
                tuple(float(v) for v in pose),
                float(yaw),
                target_object,
                obstacles,
                floor_z=deterministic_floor_z,
                footprint_half_extent=ROBOT_FOOTPRINT_HALF_EXTENT,
                pelvis_height=ROBOT_PELVIS_HEIGHT_M,
                max_facing_error_deg=30.0,
                standoff_range=validation_standoff_range,
                standoff_obstacles=_find_standoff_fixtures(obstacles, target_object),
            )
            adjusted_verdict, suppressed_clips = _adjust_verdict_for_broad_aabb_false_positives(
                verdict=verdict,
                obstacles=obstacles,
                target_obj=target_object,
                record=record,
            )
            if suppressed_clips:
                result["deterministic_geometry_raw"] = _placement_verdict_to_dict(verdict)
                result["deterministic_geometry_adjustments"] = {
                    "suppressed_broad_aabb_clips": suppressed_clips,
                    "claim_boundary": (
                        "Suppression is allowed only after PhysX reports zero contacts; it corrects "
                        "coarse USD AABB occupancy, not physical collision or task success."
                    ),
                }
                verdict = adjusted_verdict
            result["deterministic_geometry"] = _placement_verdict_to_dict(verdict)
            deterministic_geometry_ok = bool(verdict.ok)
            if not verdict.ok:
                blockers.append("placement_geometry_invalid")
        if root_to_center_xy > max_root_to_bbox_center_xy_m:
            blockers.append("placed_robot_bbox_center_far_from_root_pose")
        if relation["overlaps_xy"]:
            scene_collision_count = None
            if record is not None:
                try:
                    scene_collision_count = int(record.get("scene_collision_contact_count") or 0)
                except Exception:  # noqa: BLE001
                    scene_collision_count = None
            visual_bbox_overlap_allowed = bool(
                deterministic_geometry_ok is True
                and scene_collision_count == 0
            )
            if visual_bbox_overlap_allowed:
                result["target_bbox_relation"] = {
                    **relation,
                    "hard_blocker": False,
                    "reason": (
                        "full_robot_visual_aabb_overlaps_fixture_but_floor_footprint_and_scene_probe_are_clear"
                    ),
                }
            else:
                blockers.append("placed_robot_bbox_overlaps_target_bbox")
        else:
            required_gap = float(min_target_gap_m)
            if record is not None:
                try:
                    planned_standoff = float(record.get("standoff_from_target_surface_m"))
                except Exception:  # noqa: BLE001
                    planned_standoff = 0.0
                if planned_standoff > 0.0:
                    dx = target_center[0] - float(pose[0])
                    dy = target_center[1] - float(pose[1])
                    mag = math.hypot(dx, dy)
                    if mag > 1e-6:
                        robot_half_extent = _half_extent_along_bounds(
                            (
                                tuple(float(v) for v in bbox["bbox_min_xyz"]),
                                tuple(float(v) for v in bbox["bbox_max_xyz"]),
                            ),
                            (dx / mag, dy / mag),
                        )
                        required_gap = max(
                            required_gap,
                            planned_standoff - robot_half_extent - float(standoff_tolerance_m),
                        )
                        result["robot_half_extent_toward_target_m"] = round(
                            float(robot_half_extent), 6
                        )
            result["required_target_gap_m"] = round(float(required_gap), 6)
            if float(relation["gap_m"]) < required_gap:
                blockers.append("placed_robot_target_gap_below_threshold")
        if blockers:
            result["status"] = "blocked"
            result["blockers"] = blockers
        return result

    return validate


def _is_robot_scene_object(obj) -> bool:
    obj_id = str(getattr(obj, "id", "") or "").strip().lower()
    label = str(getattr(obj, "label", "") or "").strip().lower()
    text = f"{obj_id} {label}".lower()
    if obj_id in {"g", "g1", "unitree_g1", "proxy_body"}:
        return True
    if label in {"g", "g1", "unitree g1", "proxy body"}:
        return True
    return any(token in text for token in ("g1", "unitree", "robot", "placementdebug", "proxy_body"))


def _scene_objects_for_stage(stage) -> list[Any]:
    """Grouped scene-placement object catalog from the current USD stage.

    This stays grouped because it is used for target resolution and human-readable artifact samples:
    a sink assembly should be one target object. Use :func:`_placement_obstacles_for_stage` for clip
    validation, where grouped counter/cabinet AABBs are too coarse.
    """
    bundle_dir = str(Path(__file__).resolve().parent)
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)
    try:
        try:
            from scene_placement import UsdSceneSpatialIndex  # type: ignore
        except Exception:  # noqa: BLE001
            from blueprint_pipeline.scene_placement import UsdSceneSpatialIndex  # type: ignore
        objects = list(UsdSceneSpatialIndex(stage=stage).objects())
    except Exception:
        return []
    filtered = []
    for obj in objects:
        if _is_robot_scene_object(obj):
            continue
        filtered.append(obj)
    return filtered


def _semantic_label_from_prim_name(name: str) -> str:
    try:
        from blueprint_pipeline.scene_placement.usd_index import _clean_label  # type: ignore

        label = str(_clean_label(name))
    except Exception:  # noqa: BLE001
        label = str(name).replace("_", " ").strip().lower()
        while label and label[-1].isdigit():
            label = label[:-1].rstrip("_- .")
    return label or "scene_object"


def _author_scene_semantic_labels(
    stage,
    *,
    robot_prim_path: str | None,
    keep_substrings: Sequence[str] = (),
) -> dict[str, Any]:
    try:
        from pxr import UsdGeom  # type: ignore
    except Exception:
        UsdGeom = None  # type: ignore[assignment]
    add_semantics = None
    try:
        from semantics.schema_editor import add_prim_semantics  # type: ignore

        add_semantics = add_prim_semantics
    except Exception:
        try:
            from omni.isaac.core.utils.semantics import add_update_semantics  # type: ignore

            add_semantics = add_update_semantics
        except Exception as exc:
            return {
                "schema_version": "isaac_scene_semantic_label_authoring.v1",
                "labeled_prim_count": 0,
                "sample_labels": [],
                "keep_substrings": list(keep_substrings),
                "blockers": ["isaac_semantics_authoring_api_unavailable"],
                "error": repr(exc),
            }
    blockers: list[str] = []
    sample_labels: list[dict[str, str]] = []
    labeled_count = 0
    robot_path = str(robot_prim_path or "")
    skip_tokens = ("g1", "unitree", "robot", "placementdebug")
    try:
        prims = list(stage.Traverse())
    except Exception as exc:
        return {
            "schema_version": "isaac_scene_semantic_label_authoring.v1",
            "labeled_prim_count": 0,
            "sample_labels": [],
            "keep_substrings": list(keep_substrings),
            "blockers": ["isaac_stage_traversal_unavailable"],
            "error": repr(exc),
        }
    for prim in prims:
        try:
            prim_path = str(prim.GetPath())
            prim_name = str(prim.GetName())
        except Exception:
            continue
        text = f"{prim_path} {prim_name}".lower()
        if robot_path and prim_path.startswith(robot_path):
            continue
        if any(token in text for token in skip_tokens):
            continue
        if UsdGeom is not None:
            try:
                if not (prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.Imageable)):
                    continue
            except Exception:
                continue
        label = _semantic_label_from_prim_name(prim_name)
        try:
            try:
                add_semantics(prim, semantic_label=label, type_label="class")
            except TypeError:
                try:
                    add_semantics(prim, label, "class")
                except TypeError:
                    add_semantics(prim, label)
            labeled_count += 1
            if len(sample_labels) < 40:
                sample_labels.append({"prim_path": prim_path, "semantic_label": label})
        except Exception as exc:  # noqa: BLE001
            blockers.append(f"semantic_label_authoring_failed:{type(exc).__name__}")
    return {
        "schema_version": "isaac_scene_semantic_label_authoring.v1",
        "labeled_prim_count": labeled_count,
        "sample_labels": sample_labels,
        "keep_substrings": list(keep_substrings),
        "blockers": sorted(set(blockers)),
    }


def _placement_obstacles_for_stage(
    stage,
    *,
    focus_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    focus_margin_m: float = 2.5,
) -> list[Any]:
    """Validation-only obstacle catalog from USD.

    Target resolution wants one object per named assembly, but placement clipping wants the opposite:
    a broad cabinet/counter assembly AABB can cover open aisle floor. Prefer fine leaf-Gprim boxes
    when the index exposes them, and fall back to grouped objects on older bundles. When target bounds
    are known, keep only the target neighborhood so validation cannot spend the startup window walking
    unrelated decorative geometry.
    """
    bundle_dir = str(Path(__file__).resolve().parent)
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)
    try:
        try:
            from scene_placement import UsdSceneSpatialIndex  # type: ignore
        except Exception:  # noqa: BLE001
            from blueprint_pipeline.scene_placement import UsdSceneSpatialIndex  # type: ignore
        index = UsdSceneSpatialIndex(stage=stage)
        obstacle_boxes = getattr(index, "obstacle_boxes", None)
        objects = list(obstacle_boxes() if callable(obstacle_boxes) else index.objects())
    except Exception:
        objects = []
    filtered = []
    for obj in objects:
        if _is_robot_scene_object(obj):
            continue
        if not _xy_focus_overlap(obj, focus_bounds, margin_m=focus_margin_m):
            continue
        filtered.append(obj)
    shell = [
        obj
        for obj in _placement_shell_obstacles_for_stage(stage)
        if _xy_focus_overlap(obj, focus_bounds, margin_m=focus_margin_m)
    ]
    return filtered + shell


def _placement_shell_obstacles_for_stage(stage) -> list[Any]:
    """Validation-only wall/window/door obstacles.

    ``UsdSceneSpatialIndex`` intentionally excludes the structural shell for object targeting. Robot
    placement needs the opposite for walls: a pose outside or through the room shell is invalid even
    when it does not overlap a movable object. Keep floor/ground/ceiling/lights excluded so standing
    on the floor remains valid.
    """
    traverse = getattr(stage, "Traverse", None)
    if not callable(traverse):
        return []
    try:
        from pxr import Usd, UsdGeom  # type: ignore
    except Exception:
        return []
    try:
        from blueprint_pipeline.scene_placement import SceneObject  # type: ignore
    except Exception:
        try:
            from scene_placement import SceneObject  # type: ignore
        except Exception:
            return []
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    try:
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=True)
    except TypeError:
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    out = []
    seen: set[str] = set()
    include_words = ("wall", "window")
    exclude_words = (
        "floor",
        "ground",
        "ceiling",
        "light",
        "g1",
        "unitree",
        "robot",
        "placementdebug",
        "cabinet",
        "dishwasher",
        "oven",
        "fridge",
        "refrigerator",
    )
    traverse = getattr(stage, "Traverse", None)
    if not callable(traverse):
        return []
    for prim in traverse():
        prim_path = str(prim.GetPath())
        text = f"{prim_path} {prim.GetName()}".lower()
        if not any(word in text for word in include_words):
            continue
        if any(word in text for word in exclude_words):
            continue
        try:
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if box.IsEmpty():
                continue
            bbox_min, bbox_max, center, _size = _aligned_box_min_max_center_size(box)
        except Exception:  # noqa: BLE001
            continue
        obj_id = _safe_shell_obstacle_id(prim_path.strip("/") or str(prim.GetName()))
        size_x = abs(float(bbox_max[0]) - float(bbox_min[0]))
        size_y = abs(float(bbox_max[1]) - float(bbox_min[1]))
        if min(size_x, size_y) > 0.35:
            is_wall_mesh = False
            mesh_type = getattr(UsdGeom, "Mesh", None)
            if mesh_type is not None:
                try:
                    is_wall_mesh = bool(prim.IsA(mesh_type))
                except Exception:  # noqa: BLE001
                    is_wall_mesh = False
            if is_wall_mesh and ("wall" in text or "wallcollider" in text):
                for edge_id, edge_min, edge_max in _synthesized_room_edge_shell_boxes(
                    obj_id,
                    bbox_min,
                    bbox_max,
                ):
                    if edge_id in seen:
                        continue
                    seen.add(edge_id)
                    out.append(SceneObject(
                        id=edge_id,
                        label=f"{str(prim.GetName()) or 'wall'}_{edge_id.rsplit('_', 1)[-1]}",
                        bbox_min=edge_min,
                        bbox_max=edge_max,
                        centroid=(
                            0.5 * (edge_min[0] + edge_max[0]),
                            0.5 * (edge_min[1] + edge_max[1]),
                            0.5 * (edge_min[2] + edge_max[2]),
                        ),
                        source="usd_shell",
                    ))
            continue
        if obj_id in seen:
            continue
        seen.add(obj_id)
        out.append(SceneObject(
            id=obj_id,
            label=str(prim.GetName()) or "shell_obstacle",
            bbox_min=(float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])),
            bbox_max=(float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])),
            centroid=(float(center[0]), float(center[1]), float(center[2])),
            source="usd_shell",
        ))
    return out


def _synthesized_room_edge_shell_boxes(
    obj_id: str,
    bbox_min: Sequence[float],
    bbox_max: Sequence[float],
    *,
    thickness_m: float = 0.08,
) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """Thin wall-edge proxies for broad room wall meshes.

    Some kitchen assets author multiple wall planes as one connected mesh, so its single AABB spans
    most of the room and cannot be used as a clip obstacle. Dropping it loses the side/back wall
    clearance check entirely. These proxies represent only the AABB shell edges, which is enough to
    reject poses pressed into the side wall or behind the counter without treating the whole room as
    occupied.
    """
    min_x, min_y, min_z = (float(v) for v in bbox_min)
    max_x, max_y, max_z = (float(v) for v in bbox_max)
    t = abs(float(thickness_m))
    return [
        (
            f"{obj_id}_xmin",
            (min_x - 0.5 * t, min_y, min_z),
            (min_x + 0.5 * t, max_y, max_z),
        ),
        (
            f"{obj_id}_xmax",
            (max_x - 0.5 * t, min_y, min_z),
            (max_x + 0.5 * t, max_y, max_z),
        ),
        (
            f"{obj_id}_ymin",
            (min_x, min_y - 0.5 * t, min_z),
            (max_x, min_y + 0.5 * t, max_z),
        ),
        (
            f"{obj_id}_ymax",
            (min_x, max_y - 0.5 * t, min_z),
            (max_x, max_y + 0.5 * t, max_z),
        ),
    ]


def _safe_shell_obstacle_id(value: str) -> str:
    text = (value or "shell_obstacle").strip().lower().replace("/", "_")
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)
    return cleaned.strip("_") or "shell_obstacle"


def _target_object_from_stance_plan(stance_plan: Mapping[str, Any]):
    try:
        from blueprint_pipeline.scene_placement import SceneObject  # type: ignore
    except Exception:
        try:
            from scene_placement import SceneObject  # type: ignore
        except Exception:
            return None
    selected = {}
    target_resolution = stance_plan.get("target_resolution")
    if isinstance(target_resolution, Mapping):
        raw_selected = target_resolution.get("selected")
        if isinstance(raw_selected, Mapping):
            selected = dict(raw_selected)
    bounds = stance_plan.get("task_target_bounds")
    bbox_min = bbox_max = None
    if isinstance(bounds, Mapping):
        bbox_min = bounds.get("bbox_min_xyz")
        bbox_max = bounds.get("bbox_max_xyz")
    if bbox_min is None:
        bbox_min = selected.get("bbox_min_xyz")
    if bbox_max is None:
        bbox_max = selected.get("bbox_max_xyz")
    center = (
        selected.get("center_xyz")
        or stance_plan.get("task_target_xyz")
        or selected.get("centroid")
    )
    if bbox_min is None or bbox_max is None or center is None:
        return None
    label = str(selected.get("target_object_label") or selected.get("label") or "target")
    obj_id = str(selected.get("target_object_id") or selected.get("id") or label or "target")
    return SceneObject(
        id=obj_id,
        label=label,
        bbox_min=(float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])),
        bbox_max=(float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])),
        centroid=(float(center[0]), float(center[1]), float(center[2])),
        source="stance_plan",
    )


def _scene_object_to_dict(obj) -> dict[str, Any]:
    return {
        "id": str(getattr(obj, "id", "")),
        "label": str(getattr(obj, "label", "")),
        "bbox_min_xyz": _rounded_xyz(getattr(obj, "bbox_min")),
        "bbox_max_xyz": _rounded_xyz(getattr(obj, "bbox_max")),
        "centroid_xyz": _rounded_xyz(getattr(obj, "centroid")),
        "source": str(getattr(obj, "source", "") or ""),
    }


def _scene_object_xy_size_area(obj) -> tuple[float, float, float]:
    try:
        bbox_min = getattr(obj, "bbox_min")
        bbox_max = getattr(obj, "bbox_max")
        sx = max(0.0, float(bbox_max[0]) - float(bbox_min[0]))
        sy = max(0.0, float(bbox_max[1]) - float(bbox_min[1]))
        return sx, sy, sx * sy
    except Exception:  # noqa: BLE001
        return 0.0, 0.0, 0.0


def _is_structural_or_target_obstacle(obj, target_obj) -> bool:
    obj_id = str(getattr(obj, "id", "") or "")
    target_id = str(getattr(target_obj, "id", "") or "")
    if obj_id and target_id and obj_id == target_id:
        return True
    text = f"{obj_id} {getattr(obj, 'label', '')} {getattr(obj, 'source', '')}".lower()
    return any(token in text for token in ("wall", "window", "floor", "ground", "ceiling", "usd_shell"))


def _broad_aabb_false_positive_clip_ids(
    *,
    verdict,
    obstacles: Sequence[Any],
    target_obj,
    record: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    try:
        contact_count = int((record or {}).get("scene_collision_contact_count", 1))
    except Exception:  # noqa: BLE001
        contact_count = 1
    if contact_count != 0:
        return []
    clipped = list(getattr(verdict, "clipping", []) or [])
    if not clipped:
        return []
    by_id = {str(getattr(obj, "id", "") or ""): obj for obj in obstacles}
    footprint_area = (2.0 * float(ROBOT_FOOTPRINT_HALF_EXTENT[0])) * (
        2.0 * float(ROBOT_FOOTPRINT_HALF_EXTENT[1])
    )
    broad_area_threshold = max(2.0, footprint_area * 12.0)
    broad_span_threshold = max(
        1.0,
        max(float(ROBOT_FOOTPRINT_HALF_EXTENT[0]), float(ROBOT_FOOTPRINT_HALF_EXTENT[1])) * 4.0,
    )
    suppressed: list[dict[str, Any]] = []
    for obj_id, overlap_area in clipped:
        obj = by_id.get(str(obj_id))
        if obj is None or _is_structural_or_target_obstacle(obj, target_obj):
            continue
        sx, sy, area = _scene_object_xy_size_area(obj)
        if area < broad_area_threshold or min(sx, sy) < broad_span_threshold:
            continue
        suppressed.append(
            {
                "object_id": str(obj_id),
                "overlap_area_xy_m2": round(float(overlap_area), 6),
                "obstacle_xy_size_m": [round(float(sx), 6), round(float(sy), 6)],
                "obstacle_xy_area_m2": round(float(area), 6),
                "reason": (
                    "zero PhysX contacts and broad non-structural AABB; treated as a coarse "
                    "USD occupancy false positive"
                ),
            }
        )
    return suppressed


def _adjust_verdict_for_broad_aabb_false_positives(
    *,
    verdict,
    obstacles: Sequence[Any],
    target_obj,
    record: Mapping[str, Any] | None,
):
    suppressed = _broad_aabb_false_positive_clip_ids(
        verdict=verdict,
        obstacles=obstacles,
        target_obj=target_obj,
        record=record,
    )
    if not suppressed:
        return verdict, []
    suppressed_ids = {item["object_id"] for item in suppressed}
    remaining_clips = [
        (obj_id, area)
        for obj_id, area in (getattr(verdict, "clipping", []) or [])
        if str(obj_id) not in suppressed_ids
    ]
    failures = [
        failure
        for failure in (getattr(verdict, "failures", []) or [])
        if not str(failure).startswith("clips:")
    ]
    if remaining_clips:
        failures.append("clips:" + ",".join(str(obj_id) for obj_id, _ in remaining_clips))
    ok = not failures
    notes = (
        "placement valid; suppressed broad AABB clip false positives"
        if ok
        else "INVALID: " + "; ".join(str(failure) for failure in failures)
    )
    return replace(
        verdict,
        ok=ok,
        failures=failures,
        clipping=remaining_clips,
        notes=notes,
    ), suppressed


def _placement_verdict_to_dict(verdict) -> dict[str, Any]:
    return {
        "ok": bool(getattr(verdict, "ok", False)),
        "failures": list(getattr(verdict, "failures", []) or []),
        "clipping": [
            {"object_id": str(obj_id), "overlap_area_xy_m2": float(area)}
            for obj_id, area in (getattr(verdict, "clipping", []) or [])
        ],
        "near_clearance": [
            {"object_id": str(obj_id), "gap_m": float(gap)}
            for obj_id, gap in (getattr(verdict, "near_clearance", []) or [])
        ],
        "outside_boundary": list(getattr(verdict, "outside_boundary", []) or []),
        "min_obstacle_clearance_m": getattr(verdict, "min_obstacle_clearance_m", None),
        "facing_error_deg": getattr(verdict, "facing_error_deg", None),
        "standoff_m": getattr(verdict, "standoff_m", None),
        "on_floor": bool(getattr(verdict, "on_floor", False)),
        "notes": str(getattr(verdict, "notes", "") or ""),
    }


def _footprint_box_for_pose(pose, half_extent=None) -> dict[str, list[float]]:
    # Resolve at call time (not def time) so apply_robot_profile() overrides land here too.
    hx, hy, hz = (abs(float(v)) for v in (half_extent or ROBOT_FOOTPRINT_HALF_EXTENT))
    return {
        "bbox_min_xyz": _rounded_xyz((float(pose[0]) - hx, float(pose[1]) - hy, float(pose[2]) - hz)),
        "bbox_max_xyz": _rounded_xyz((float(pose[0]) + hx, float(pose[1]) + hy, float(pose[2]) + hz)),
    }


def _find_standoff_fixtures(scene_objects: Sequence[Any], target_obj) -> list[Any]:
    target_text = (
        f"{getattr(target_obj, 'label', '')} {getattr(target_obj, 'id', '')}"
    ).lower()
    # If the resolver selected the actual fixture (sink/counter/cabinet/etc.), standoff must be
    # measured against that target's own AABB. Using a broad nearby cabinet run can otherwise make a
    # robot far from the sink look reachable because it is merely in front of generic drawers.
    if any(
        word in target_text
        for word in ("sink", "counter", "cabinet", "dishwasher", "island", "stove", "oven")
    ):
        return []
    target_xy = getattr(target_obj, "footprint_center")()
    out = []
    fixture_words = ("counter", "cabinet", "sink", "basin", "dishwasher", "island")
    for obj in scene_objects:
        label = str(getattr(obj, "label", "") or "").lower()
        obj_id = str(getattr(obj, "id", "") or "").lower()
        if not any(word in f"{label} {obj_id}" for word in fixture_words):
            continue
        try:
            cx, cy = getattr(obj, "footprint_center")()
        except Exception:  # noqa: BLE001
            continue
        if math.hypot(float(cx) - float(target_xy[0]), float(cy) - float(target_xy[1])) <= 1.75:
            out.append(obj)
    return out


def _build_placement_validation_manifest(
    *,
    stage,
    robot_prim_path: str,
    stance_plan: Mapping[str, Any] | None,
    accepted_pose,
    accepted_yaw: float,
    root_diagnostics: Mapping[str, Any] | None,
    scene_objects: Sequence[Any],
    scenario_id: str,
    visual_qc: Mapping[str, Any] | None = None,
    topdown_frame: str | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    accepted_pose_xyz = _rounded_xyz(accepted_pose)
    floor_z = (
        float((stance_plan or {}).get("floor_z_hint"))
        if stance_plan and (stance_plan or {}).get("floor_z_hint") is not None
        else float(accepted_pose[2]) - ROBOT_PELVIS_HEIGHT_M
    )
    target_obj = _target_object_from_stance_plan(stance_plan or {})
    if target_obj is None:
        blockers.append("placement_validation_missing_target_bounds")
    intended_geometry: dict[str, Any] | None = None
    if target_obj is not None:
        try:
            from blueprint_pipeline.scene_placement import validate_stand_pose  # type: ignore
        except Exception:
            from scene_placement import validate_stand_pose  # type: ignore
        selected_record = None
        try:
            candidates = list((stance_plan or {}).get("candidates") or [])
            idx = int((stance_plan or {}).get("selected_candidate_index") or 0)
            if 0 <= idx < len(candidates):
                selected_record = candidates[idx]
        except Exception:  # noqa: BLE001
            selected_record = None
        validation_standoff_range = TASK_STANCE_DEFAULT_VALIDATION_STANDOFF_RANGE_M
        try:
            recorded_range = (
                ((stance_plan or {}).get("placement_validation") or {}).get(
                    "validation_standoff_range_m"
                )
            )
            if (
                isinstance(recorded_range, Sequence)
                and not isinstance(recorded_range, (str, bytes))
                and len(recorded_range) == 2
            ):
                validation_standoff_range = (float(recorded_range[0]), float(recorded_range[1]))
        except Exception:  # noqa: BLE001
            validation_standoff_range = TASK_STANCE_DEFAULT_VALIDATION_STANDOFF_RANGE_M
        obstacles = list(scene_objects)
        # If the target was resolved from the stance plan but not present in the current catalog,
        # include it so clipping into the target itself is still detected.
        if not any(str(getattr(o, "id", "")) == str(getattr(target_obj, "id", "")) for o in obstacles):
            obstacles.append(target_obj)
        verdict = validate_stand_pose(
            tuple(float(v) for v in accepted_pose),
            float(accepted_yaw),
            target_obj,
            obstacles,
            floor_z=floor_z,
            footprint_half_extent=ROBOT_FOOTPRINT_HALF_EXTENT,
            pelvis_height=ROBOT_PELVIS_HEIGHT_M,
            max_facing_error_deg=30.0,
            standoff_range=validation_standoff_range,
            standoff_obstacles=_find_standoff_fixtures(obstacles, target_obj),
        )
        adjusted_verdict, suppressed_clips = _adjust_verdict_for_broad_aabb_false_positives(
            verdict=verdict,
            obstacles=obstacles,
            target_obj=target_obj,
            record=selected_record,
        )
        if suppressed_clips:
            intended_geometry = {
                **_placement_verdict_to_dict(adjusted_verdict),
                "raw_geometry": _placement_verdict_to_dict(verdict),
                "adjustments": {
                    "suppressed_broad_aabb_clips": suppressed_clips,
                    "claim_boundary": (
                        "Suppression is allowed only after PhysX reports zero contacts; it corrects "
                        "coarse USD AABB occupancy, not physical collision or task success."
                    ),
                },
            }
            verdict = adjusted_verdict
        else:
            intended_geometry = _placement_verdict_to_dict(verdict)
        if not verdict.ok:
            blockers.append("placement_geometry_invalid")
    actual_bbox = _world_bbox_for_prim(stage, robot_prim_path)
    upright_report = _robot_upright_report(stage, robot_prim_path)
    if upright_report.get("status") == "blocked":
        blockers.append("placement_robot_not_upright")
    ground_truth: dict[str, Any] = {
        "robot_prim_path": robot_prim_path,
        "accepted_pose_xyz": _rounded_xyz(accepted_pose),
        "accepted_pose_xy": [round(float(accepted_pose[0]), 6), round(float(accepted_pose[1]), 6)],
        "max_xy_error_m": PLACEMENT_GROUND_TRUTH_MAX_FOOTPRINT_CENTER_DELTA_M,
        "place_root_diagnostics": dict(root_diagnostics or {}),
        "upright_report": upright_report,
    }
    if actual_bbox is None:
        ground_truth["status"] = "blocked"
        ground_truth["blockers"] = ["placed_robot_bbox_unavailable"]
        ground_truth["xform_diagnostics"] = _root_transform_diagnostics(stage, robot_prim_path)
        blockers.append("placement_ground_truth_bbox_unavailable")
    else:
        actual_center = _footprint_center_xy_from_bbox(actual_bbox)
        actual_center_xyz = actual_bbox["center_xyz"]
        xy_error = _xy_distance(actual_center, ground_truth["accepted_pose_xy"])
        computed_offset_xyz = [
            round(float(actual_center_xyz[0]) - float(accepted_pose[0]), 6),
            round(float(actual_center_xyz[1]) - float(accepted_pose[1]), 6),
            round(float(actual_center_xyz[2]) - float(accepted_pose[2]), 6),
        ]
        ground_truth.update({
            "status": (
                "passed"
                if xy_error <= PLACEMENT_GROUND_TRUTH_MAX_FOOTPRINT_CENTER_DELTA_M
                else "blocked"
            ),
            "actual_world_aabb": actual_bbox,
            "actual_footprint_center_xyz": actual_center_xyz,
            "actual_footprint_center_xy": actual_center,
            "xy_error_m": round(float(xy_error), 6),
            "computed_xyz_offset_m": computed_offset_xyz,
        })
        if xy_error > PLACEMENT_GROUND_TRUTH_MAX_FOOTPRINT_CENTER_DELTA_M:
            ground_truth["blockers"] = ["placed_robot_footprint_center_mismatch"]
            ground_truth["xform_diagnostics"] = _root_transform_diagnostics(stage, robot_prim_path)
            blockers.append("placement_ground_truth_center_mismatch")
    if visual_qc is not None and _visual_qc_contains_parsed_failure(visual_qc):
        blockers.append("placement_visual_qc_failed")
    manifest = {
        "schema_version": "placement_validation.v1",
        "status": "PASS" if not blockers else "FAIL",
        "scenario_id": scenario_id,
        "blockers": sorted(set(blockers)),
        "accepted_pose": accepted_pose_xyz,
        "accepted_yaw": round(float(accepted_yaw), 6),
        "floor_z": round(float(floor_z), 6),
        "robot_footprint_half_extent": _rounded_xyz(ROBOT_FOOTPRINT_HALF_EXTENT),
        "robot_footprint_box_at_accepted_pose": _footprint_box_for_pose(accepted_pose),
        "target": _scene_object_to_dict(target_obj) if target_obj is not None else None,
        "scene_object_count": len(scene_objects),
        "scene_objects_sample": [_scene_object_to_dict(o) for o in list(scene_objects)[:40]],
        "intended_geometry": intended_geometry,
        "ground_truth_placement": ground_truth,
        "visual_qc": dict(visual_qc) if visual_qc is not None else None,
        "topdown_debug_frame": topdown_frame,
        "claim_boundary": (
            "Placement validation checks USD/world-frame geometry and rendered visual placement only. "
            "It is not manipulation success, learned policy success, physical robot readiness, safety "
            "validation, or deployment approval."
        ),
    }
    return manifest


def _placement_validation_passed_manifest(manifest: Mapping[str, Any] | None) -> bool:
    return bool(manifest and manifest.get("status") == "PASS" and not manifest.get("blockers"))


def _visual_qc_contains_parsed_failure(report: Mapping[str, Any] | None) -> bool:
    """Return true only when a reviewed frame was parsed and explicitly failed.

    Model/API exhaustion and malformed responses are recorded as visual-QC evidence, but they are
    not the same as a parsed visual finding that the robot is misplaced or the POV is unusable.
    """
    if not isinstance(report, Mapping):
        return False
    per_frame = report.get("per_frame")
    if isinstance(per_frame, Sequence) and not isinstance(per_frame, (str, bytes)):
        for verdict in per_frame:
            if not isinstance(verdict, Mapping):
                continue
            if bool(verdict.get("parsed")) and not bool(verdict.get("passed")):
                return True
    for key in ("placement", "manipulation_pov"):
        nested = report.get(key)
        if isinstance(nested, Mapping) and _visual_qc_contains_parsed_failure(nested):
            return True
    return False


def _placement_visual_qc_target_label(stance_plan: Mapping[str, Any] | None) -> str:
    target = _target_object_from_stance_plan(stance_plan or {})
    if target is None:
        return "target"
    label = str(getattr(target, "label", "") or "").strip()
    return label or str(getattr(target, "id", "") or "target")


def _run_placement_visual_qc(
    frame_paths: Sequence[Path],
    *,
    target_label: str,
    task_description: str,
) -> dict[str, Any]:
    try:
        try:
            from render_visual_qc import qc_robot_placement_frames  # type: ignore
        except Exception:  # noqa: BLE001
            from blueprint_pipeline.render_visual_qc import qc_robot_placement_frames
    except Exception as exc:  # noqa: BLE001
        return {
            "schema_version": "robot_placement_visual_qc.v1",
            "status": "blocked",
            "target": target_label,
            "task_description": task_description,
            "frames_reviewed": 0,
            "blockers": ["placement_visual_qc_import_failed"],
            "error": repr(exc),
            "per_frame": [],
        }
    return qc_robot_placement_frames(
        list(frame_paths),
        target_label,
        task_description=task_description,
        sample_n=min(4, len(frame_paths)),
    )


def _run_task_visual_qc(
    placement_frame_paths: Sequence[Path],
    pov_frame_paths: Sequence[Path],
    *,
    target_label: str,
    task_description: str,
) -> dict[str, Any]:
    try:
        try:
            from render_visual_qc import (  # type: ignore
                qc_manipulation_pov_frames,
                qc_robot_placement_frames,
            )
        except Exception:  # noqa: BLE001
            from blueprint_pipeline.render_visual_qc import (
                qc_manipulation_pov_frames,
                qc_robot_placement_frames,
            )
    except Exception as exc:  # noqa: BLE001
        return {
            "schema_version": "robot_task_visual_qc.v1",
            "status": "blocked",
            "target": target_label,
            "task_description": task_description,
            "frames_reviewed": 0,
            "blockers": ["visual_qc_import_failed"],
            "error": repr(exc),
            "placement": None,
            "manipulation_pov": None,
        }
    placement_report = qc_robot_placement_frames(
        list(placement_frame_paths),
        target_label,
        task_description=task_description,
        sample_n=min(4, len(placement_frame_paths)),
    )
    pov_report = qc_manipulation_pov_frames(
        list(pov_frame_paths),
        target_label,
        task_description=task_description,
        sample_n=min(4, len(pov_frame_paths)),
    )
    blockers: list[str] = []
    if placement_report.get("status") != "passed":
        blockers.extend(placement_report.get("blockers") or ["placement_visual_qc_failed"])
    if pov_report.get("status") != "passed":
        blockers.extend(pov_report.get("blockers") or ["manipulation_pov_visual_qc_failed"])
    frames_reviewed = int(placement_report.get("frames_reviewed") or 0)
    frames_reviewed += int(pov_report.get("frames_reviewed") or 0)
    return {
        "schema_version": "robot_task_visual_qc.v1",
        "status": "passed" if not blockers else "blocked",
        "target": target_label,
        "task_description": task_description,
        "frames_reviewed": frames_reviewed,
        "blockers": sorted(set(str(b) for b in blockers)),
        "placement": placement_report,
        "manipulation_pov": pov_report,
    }


def _place_camera(stage, cam_path: str, eye, target) -> None:
    from pxr import UsdGeom, Gf  # type: ignore
    w, x, y, z = look_at_quat(eye, target)
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])))
    xform.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))


def _prim_world_translation(stage, prim) -> tuple[float, float, float] | None:
    try:
        from pxr import UsdGeom  # type: ignore
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        pos = matrix.ExtractTranslation()
        return (float(pos[0]), float(pos[1]), float(pos[2]))
    except Exception:  # noqa: BLE001
        return None


def _robot_authored_camera_mount(stage, robot_prim_path: str) -> dict[str, Any] | None:
    try:
        from pxr import Usd, UsdGeom  # type: ignore
        root = stage.GetPrimAtPath(robot_prim_path)
        if not root or not root.IsValid():
            return None
        for prim in Usd.PrimRange(root):
            try:
                is_camera = prim.IsA(UsdGeom.Camera)
            except Exception:  # noqa: BLE001
                is_camera = False
            if not is_camera:
                continue
            pos = _prim_world_translation(stage, prim)
            if pos is not None:
                return {
                    "source": "authored_robot_camera",
                    "prim_path": str(prim.GetPath()),
                    "eye_xyz": pos,
                }
    except Exception:  # noqa: BLE001
        return None
    return None


def _robot_link_mount(stage, robot_prim_path: str) -> dict[str, Any] | None:
    try:
        from pxr import Usd, UsdGeom  # type: ignore
        root = stage.GetPrimAtPath(robot_prim_path)
        if not root or not root.IsValid():
            return None
        candidates: list[tuple[int, Any]] = []
        preferences = (
            (0, ("camera",)),
            (1, ("head", "link")),
            (2, ("neck", "link")),
        )
        for prim in Usd.PrimRange(root):
            try:
                if not prim.IsA(UsdGeom.Xformable):
                    continue
            except Exception:  # noqa: BLE001
                continue
            name = prim.GetName().lower()
            for rank, tokens in preferences:
                if all(token in name for token in tokens):
                    candidates.append((rank, prim))
                    break
        for rank, prim in sorted(candidates, key=lambda item: item[0]):
            pos = _prim_world_translation(stage, prim)
            if pos is not None:
                return {
                    "source": "robot_link_mount",
                    "prim_path": str(prim.GetPath()),
                    "link_rank": rank,
                    "mount_role": (
                        "camera_link" if rank == 0 else "head_link" if rank == 1 else "neck_link"
                    ),
                    "eye_xyz": pos,
                }
    except Exception:  # noqa: BLE001
        return None
    return None


def _robot_arm_link_points(stage, robot_prim_path: str, *, arm: str = "right") -> dict[str, tuple[float, float, float]]:
    try:
        from pxr import Usd, UsdGeom  # type: ignore
        root = stage.GetPrimAtPath(robot_prim_path)
        if not root or not root.IsValid():
            return {}
        side = str(arm or "right").strip().lower()
        if side not in {"left", "right"}:
            side = "right"
        wanted = {
            "shoulder": ("shoulder",),
            "elbow": ("elbow",),
            "wrist": ("wrist",),
            "hand": ("hand", "palm"),
        }
        out: dict[str, tuple[float, float, float]] = {}
        for prim in Usd.PrimRange(root):
            try:
                if not prim.IsA(UsdGeom.Xformable):
                    continue
            except Exception:  # noqa: BLE001
                continue
            name = prim.GetName().lower()
            if side not in name or "link" not in name:
                continue
            for key, tokens in wanted.items():
                if key in out:
                    continue
                if any(token in name for token in tokens):
                    pos = _prim_world_translation(stage, prim)
                    if pos is not None:
                        out[key] = pos
                    break
        return out
    except Exception:  # noqa: BLE001
        return {}


def _robot_arm_link_points_by_arm(
    stage,
    robot_prim_path: str,
    *,
    arm: str = "right",
) -> dict[str, dict[str, tuple[float, float, float]]]:
    return {
        side: _robot_arm_link_points(stage, robot_prim_path, arm=side)
        for side in _required_manipulation_arms(arm)
    }


def _average_arm_link_points(
    arm_points_by_arm: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, tuple[float, float, float]]:
    averaged: dict[str, tuple[float, float, float]] = {}
    for role in ("shoulder", "elbow", "wrist", "hand"):
        pts = [
            tuple(float(v) for v in points[role])
            for points in arm_points_by_arm.values()
            if points.get(role) is not None
        ]
        if not pts:
            continue
        averaged[role] = (
            sum(pt[0] for pt in pts) / len(pts),
            sum(pt[1] for pt in pts) / len(pts),
            sum(pt[2] for pt in pts) / len(pts),
        )
    return averaged


def _robot_head_bounds_for_mount(stage, robot_prim_path: str, mount: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return world bounds for the robot's head/camera geometry near the selected mount.

    Some G1 USD link prims expose a link origin at the body root while their mesh descendants carry the
    visible head geometry. A lens derived from the link origin alone can land below the shoulders. This
    bounded lookup only inspects the robot subtree and prefers the selected head/camera mount, falling
    back to named head/neck/camera prims. It is robot-geometry based, not scene/task specific.
    """
    mount_path = str(mount.get("prim_path") or "")
    candidate_paths: list[str] = []
    if mount_path:
        candidate_paths.append(mount_path)
    try:
        from pxr import Usd, UsdGeom  # type: ignore

        root = stage.GetPrimAtPath(robot_prim_path)
        if root and root.IsValid():
            for prim in Usd.PrimRange(root):
                name = prim.GetName().lower()
                if not any(token in name for token in ("camera", "head", "neck", "face")):
                    continue
                try:
                    if not prim.IsA(UsdGeom.Xformable):
                        continue
                except Exception:  # noqa: BLE001
                    continue
                path = str(prim.GetPath())
                if path not in candidate_paths:
                    candidate_paths.append(path)
    except Exception:  # noqa: BLE001
        pass

    best: dict[str, Any] | None = None
    best_score = -1.0
    for path in candidate_paths[:64]:
        try:
            bbox = _world_bbox_for_prim(stage, path)
        except Exception:  # noqa: BLE001
            bbox = None
        if not bbox:
            continue
        bmin = bbox.get("bbox_min_xyz") or ()
        bmax = bbox.get("bbox_max_xyz") or ()
        center = bbox.get("center_xyz") or ()
        size = bbox.get("size_xyz") or ()
        if len(bmin) < 3 or len(bmax) < 3 or len(center) < 3 or len(size) < 3:
            continue
        size_z = abs(float(size[2]))
        size_xy = max(abs(float(size[0])), abs(float(size[1])))
        if size_z < 0.04 or size_xy < 0.04:
            continue
        name = path.lower()
        score = float(center[2])
        if path == mount_path:
            score += 2.0
        if "camera" in name or "face" in name:
            score += 1.0
        elif "head" in name:
            score += 0.7
        elif "neck" in name:
            score += 0.2
        if score > best_score:
            best_score = score
            best = {
                "source_prim_path": path,
                "bbox_min_xyz": [round(float(v), 6) for v in bmin],
                "bbox_max_xyz": [round(float(v), 6) for v in bmax],
                "center_xyz": [round(float(v), 6) for v in center],
                "size_xyz": [round(float(v), 6) for v in size],
            }
    return best


def _robot_head_lens_eye_from_mount(
    mount_eye: Sequence[float],
    yaw: float,
    *,
    authored_camera: bool = False,
    root_pose: Sequence[float] | None = None,
    arm_points: Mapping[str, Sequence[float]] | None = None,
    head_bounds: Mapping[str, Sequence[float]] | None = None,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Return the render eye for the robot-mounted POV.

    Authored USD cameras are used exactly. For a bare head/neck link, use a small robot-relative
    forward/up lens offset so the camera sits at the face of the head instead of inside the link mesh.
    """
    raw_eye = (float(mount_eye[0]), float(mount_eye[1]), float(mount_eye[2]))
    if authored_camera:
        return raw_eye, {
            "lens_offset_xyz_robot_frame": [0.0, 0.0, 0.0],
            "raw_mount_eye_xyz": [round(v, 6) for v in raw_eye],
            "lens_height_correction_applied": False,
            "head_lens_z_source": "authored_camera",
        }
    fx, fy = math.cos(float(yaw)), math.sin(float(yaw))
    # Link-origin fallback: keep the lens near the face/neck plane, not at the front of the whole
    # robot footprint. Pushing the camera too far forward can put the forearms behind the lens, which
    # crops them out of an otherwise valid head POV.
    forward_m = max(0.04, min(0.12, float(ROBOT_FOOTPRINT_HALF_EXTENT[0]) * 0.35))
    head_bounds_json: dict[str, Any] | None = None
    if head_bounds:
        bmin = head_bounds.get("bbox_min_xyz")
        bmax = head_bounds.get("bbox_max_xyz")
        center = head_bounds.get("center_xyz")
        size = head_bounds.get("size_xyz")
        if bmin is not None and bmax is not None:
            try:
                # Place the fallback lens on the visible front of the head/face along the robot's
                # current forward axis. This replaces the earlier fixed footprint-depth offset when
                # the USD exposes usable head bounds.
                corners = (
                    (float(bmin[0]), float(bmin[1])),
                    (float(bmin[0]), float(bmax[1])),
                    (float(bmax[0]), float(bmin[1])),
                    (float(bmax[0]), float(bmax[1])),
                )
                front_projection = max(x * fx + y * fy for x, y in corners)
                raw_projection = raw_eye[0] * fx + raw_eye[1] * fy
                derived_forward = front_projection - raw_projection
                if derived_forward > 0.02:
                    forward_m = max(0.03, derived_forward + 0.015)
            except Exception:  # noqa: BLE001
                pass
        if all(value is not None for value in (bmin, bmax, center, size)):
            try:
                head_bounds_json = {
                    "source_prim_path": head_bounds.get("source_prim_path"),
                    "bbox_min_xyz": [round(float(v), 6) for v in bmin],  # type: ignore[arg-type]
                    "bbox_max_xyz": [round(float(v), 6) for v in bmax],  # type: ignore[arg-type]
                    "center_xyz": [round(float(v), 6) for v in center],  # type: ignore[arg-type]
                    "size_xyz": [round(float(v), 6) for v in size],  # type: ignore[arg-type]
                }
            except Exception:  # noqa: BLE001
                head_bounds_json = None
    if arm_points:
        try:
            raw_projection = raw_eye[0] * fx + raw_eye[1] * fy
            ahead = []
            for role in ("elbow", "wrist", "hand"):
                pt = arm_points.get(role)
                if pt is None:
                    continue
                proj = float(pt[0]) * fx + float(pt[1]) * fy
                if proj > raw_projection + 0.03:
                    ahead.append(proj)
            if ahead:
                # Keep the eye behind the nearest visible forearm/hand link so those links project in
                # front of the head camera instead of being clipped behind it.
                max_forward_before_arm = min(ahead) - raw_projection - 0.035
                forward_m = max(0.035, min(forward_m, max_forward_before_arm))
        except Exception:  # noqa: BLE001
            pass
    up_m = max(0.015, float(ROBOT_FOOTPRINT_HALF_EXTENT[2]) * 0.03)
    shoulder_z_values = [
        float(arm_points[role][2])
        for role in ("shoulder",)
        if arm_points and arm_points.get(role) is not None
    ]
    shoulder_z = max(shoulder_z_values) if shoulder_z_values else None
    lens_z_floor = None
    lens_z_source = "raw_mount_eye"
    if head_bounds_json is not None:
        try:
            head_center_z = float(head_bounds_json["center_xyz"][2])
            head_min_z = float(head_bounds_json["bbox_min_xyz"][2])
            head_max_z = float(head_bounds_json["bbox_max_xyz"][2])
            if (
                head_max_z > head_min_z
                and (shoulder_z is None or head_center_z > shoulder_z - 0.03)
            ):
                lens_z_floor = head_center_z
                lens_z_source = "head_bounds_center_above_shoulders"
        except Exception:  # noqa: BLE001
            lens_z_floor = None
    if lens_z_floor is None and shoulder_z is not None:
        lens_z_floor = shoulder_z + max(
            0.08,
            min(0.16, float(ROBOT_FOOTPRINT_HALF_EXTENT[2]) * 0.20),
        )
        lens_z_source = "shoulder_relative_fallback"
    if root_pose is not None:
        root_floor = float(root_pose[2]) + max(
            0.38,
            min(0.48, float(ROBOT_FOOTPRINT_HALF_EXTENT[2]) * 0.68),
        )
        if lens_z_floor is None or root_floor > lens_z_floor:
            lens_z_floor = root_floor
            lens_z_source = "root_height_fallback"
    corrected_z = raw_eye[2]
    height_corrected = False
    if lens_z_floor is not None and corrected_z < lens_z_floor:
        corrected_z = lens_z_floor
        height_corrected = True
    eye = (
        raw_eye[0] + fx * forward_m,
        raw_eye[1] + fy * forward_m,
        corrected_z + up_m,
    )
    return eye, {
        "lens_offset_xyz_robot_frame": [round(forward_m, 6), 0.0, round(up_m, 6)],
        "raw_mount_eye_xyz": [round(v, 6) for v in raw_eye],
        "lens_height_correction_applied": bool(height_corrected),
        "min_head_lens_z": round(float(lens_z_floor), 6) if lens_z_floor is not None else None,
        "head_lens_z_source": lens_z_source,
        "shoulder_to_lens_z_m": (
            round(float((lens_z_floor or corrected_z) - shoulder_z), 6)
            if shoulder_z is not None else None
        ),
        "head_geometry_bounds": head_bounds_json,
    }


def _robot_mounted_manipulation_cam_pose(
    stage,
    robot_prim_path: str,
    root_pose,
    yaw,
    *,
    look_at=None,
    reach_arm: str = "right",
    vfov_deg: float | None = None,
    width: int | None = None,
    height: int | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float], dict[str, Any]]:
    fallback_eye, fallback_target = manipulation_cam_pose(
        root_pose,
        yaw,
        look_at=look_at,
        reach_arm=reach_arm,
    )
    if look_at is None:
        return fallback_eye, fallback_target, {"source": "root_yaw_fallback_no_look_at"}
    mount = _robot_authored_camera_mount(stage, robot_prim_path) or _robot_link_mount(stage, robot_prim_path)
    if mount is None:
        return fallback_eye, fallback_target, {"source": "root_yaw_fallback_no_robot_mount"}
    reach_selection = _normalize_reach_arm_selection(reach_arm)
    arm_points_by_arm = _robot_arm_link_points_by_arm(
        stage,
        robot_prim_path,
        arm=reach_selection,
    )
    if reach_selection == "both":
        arm_points = _average_arm_link_points(arm_points_by_arm)
    else:
        arm_points = dict(arm_points_by_arm.get(reach_selection) or {})
    target = _manipulation_camera_target_with_arm_context(look_at, arm_points)
    head_bounds = None
    if mount.get("source") != "authored_robot_camera":
        head_bounds = _robot_head_bounds_for_mount(stage, robot_prim_path, mount)
    eye, lens_meta = _robot_head_lens_eye_from_mount(
        mount["eye_xyz"],
        yaw,
        authored_camera=mount.get("source") == "authored_robot_camera",
        root_pose=root_pose,
        arm_points=arm_points,
        head_bounds=head_bounds,
    )
    target_meta: dict[str, Any] = {}
    if vfov_deg is not None and width is not None and height is not None:
        target, target_meta = _select_manipulation_camera_target_for_visible_arm(
            look_at,
            arm_points,
            eye,
            target,
            vfov_deg=float(vfov_deg),
            width=int(width),
            height=int(height),
            arm=reach_selection,
            arm_points_by_arm=arm_points_by_arm,
        )
    arm_points_by_arm_json = {
        side: {
            key: [round(float(v), 6) for v in value]
            for key, value in sorted(points.items())
        }
        for side, points in sorted(arm_points_by_arm.items())
    }
    return eye, target, {
        "source": mount.get("source"),
        "mount_prim_path": mount.get("prim_path"),
        "mount_role": mount.get("mount_role"),
        "camera_eye_xyz": [round(float(v), 6) for v in eye],
        "camera_target_xyz": [round(float(v), 6) for v in target],
        "camera_vfov_deg": round(float(vfov_deg), 6) if vfov_deg is not None else None,
        "viewport_size_px": [int(width), int(height)] if width is not None and height is not None else None,
        "required_arms": list(_required_manipulation_arms(reach_selection)),
        "arm_link_points_used": sorted(arm_points),
        "arm_link_points_xyz": {
            key: [round(float(v), 6) for v in value]
            for key, value in sorted(arm_points.items())
        },
        "arm_link_points_by_arm_xyz": arm_points_by_arm_json,
        **target_meta,
        **lens_meta,
        "claim_boundary": (
            "POV camera is mounted from robot USD geometry when an authored camera/head/neck link "
            "is available; orientation is aimed at the task affordance plus visible arm context."
        ),
    }


def _safe_prim_segment(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value or "debug"))
    cleaned = cleaned.strip("_") or "debug"
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def _set_flat_debug_box(stage, path: str, bbox_min, bbox_max, *, color, z_lift: float) -> None:
    from pxr import UsdGeom, Gf  # type: ignore

    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    center = (
        0.5 * (float(bbox_min[0]) + float(bbox_max[0])),
        0.5 * (float(bbox_min[1]) + float(bbox_max[1])),
        float(z_lift),
    )
    size = (
        max(0.03, float(bbox_max[0]) - float(bbox_min[0])),
        max(0.03, float(bbox_max[1]) - float(bbox_min[1])),
        0.08,
    )
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddScaleOp().Set(Gf.Vec3d(size[0], size[1], size[2]))
    gprim = UsdGeom.Gprim(cube.GetPrim())
    gprim.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    try:
        gprim.CreateDisplayOpacityAttr([0.9])
    except Exception:  # noqa: BLE001
        pass


def _set_yaw_debug_arrow(stage, path: str, pose, yaw: float, *, color, z_lift: float) -> None:
    from pxr import UsdGeom, Gf  # type: ignore

    fx, fy = math.cos(float(yaw)), math.sin(float(yaw))
    length = 0.75
    width = 0.055
    center = (
        float(pose[0]) + fx * length * 0.5,
        float(pose[1]) + fy * length * 0.5,
        float(z_lift) + 0.02,
    )
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddRotateZOp().Set(math.degrees(float(yaw)))
    xform.AddScaleOp().Set(Gf.Vec3d(length, width, 0.1))
    gprim = UsdGeom.Gprim(cube.GetPrim())
    gprim.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])


def _debug_overlay_objects(scene_objects: Sequence[Any], target_obj, pose) -> list[Any]:
    if target_obj is None:
        return []
    target_xy = getattr(target_obj, "footprint_center")()
    wanted_words = ("target", "sink", "counter", "cabinet", "dishwasher", "basin", "faucet", "island")
    out = []
    for obj in scene_objects:
        label = str(getattr(obj, "label", "") or "").lower()
        obj_id = str(getattr(obj, "id", "") or "").lower()
        try:
            cx, cy = getattr(obj, "footprint_center")()
        except Exception:  # noqa: BLE001
            continue
        near_target = math.hypot(float(cx) - float(target_xy[0]), float(cy) - float(target_xy[1])) <= 2.2
        near_robot = math.hypot(float(cx) - float(pose[0]), float(cy) - float(pose[1])) <= 2.2
        named = any(word in f"{label} {obj_id}" for word in wanted_words)
        if named and (near_target or near_robot):
            out.append(obj)
    return out[:24]


def _update_topdown_debug_scene(
    stage,
    *,
    root_path: str,
    robot_pose,
    robot_yaw: float,
    stance_plan: Mapping[str, Any] | None,
    scene_objects: Sequence[Any],
    floor_z: float,
) -> dict[str, Any]:
    from pxr import UsdGeom  # type: ignore

    try:
        stage.RemovePrim(root_path)
    except Exception:  # noqa: BLE001
        pass
    UsdGeom.Scope.Define(stage, root_path)
    target_obj = _target_object_from_stance_plan(stance_plan or {})
    # Keep debug footprints above the noisy reconstruction and cabinet tops so the orthographic
    # render shows unambiguous xy occupancy instead of hiding the overlays under floor/wall geometry.
    z_lift = float(floor_z) + 2.75
    robot_box = _footprint_box_for_pose(robot_pose)
    _set_flat_debug_box(
        stage,
        f"{root_path}/robot_footprint",
        robot_box["bbox_min_xyz"],
        robot_box["bbox_max_xyz"],
        color=(0.05, 0.9, 0.15),
        z_lift=z_lift,
    )
    _set_yaw_debug_arrow(
        stage,
        f"{root_path}/robot_yaw_arrow",
        robot_pose,
        robot_yaw,
        color=(0.0, 1.0, 1.0),
        z_lift=z_lift,
    )
    overlay_objects = _debug_overlay_objects(scene_objects, target_obj, robot_pose)
    if target_obj is not None:
        _set_flat_debug_box(
            stage,
            f"{root_path}/target_footprint",
            getattr(target_obj, "bbox_min"),
            getattr(target_obj, "bbox_max"),
            color=(1.0, 0.05, 0.05),
            z_lift=z_lift + 0.015,
        )
    for idx, obj in enumerate(overlay_objects):
        _set_flat_debug_box(
            stage,
            f"{root_path}/object_{idx:02d}_{_safe_prim_segment(getattr(obj, 'id', 'object'))}",
            getattr(obj, "bbox_min"),
            getattr(obj, "bbox_max"),
            color=(1.0, 0.65, 0.05),
            z_lift=z_lift + 0.03,
        )
    xs = [float(robot_pose[0])]
    ys = [float(robot_pose[1])]
    if target_obj is not None:
        bmin = getattr(target_obj, "bbox_min")
        bmax = getattr(target_obj, "bbox_max")
        xs.extend([float(bmin[0]), float(bmax[0])])
        ys.extend([float(bmin[1]), float(bmax[1])])
    for obj in overlay_objects:
        bmin = getattr(obj, "bbox_min")
        bmax = getattr(obj, "bbox_max")
        xs.extend([float(bmin[0]), float(bmax[0])])
        ys.extend([float(bmin[1]), float(bmax[1])])
    center_xy = (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)))
    radius = max(max(xs) - min(xs), max(ys) - min(ys), 2.0) * 0.65 + 0.6
    return {
        "root_path": root_path,
        "robot_footprint_box": robot_box,
        "overlay_object_count": len(overlay_objects),
        "center_xy": [round(center_xy[0], 6), round(center_xy[1], 6)],
        "radius_m": round(float(radius), 6),
        "overlay_z_lift": round(float(z_lift), 6),
    }


def _place_topdown_debug_camera(stage, cam_path: str, *, center_xy, radius_m: float,
                                width: int, height: int, floor_z: float) -> None:
    from pxr import UsdGeom, Gf  # type: ignore

    cam = UsdGeom.Camera(stage.GetPrimAtPath(cam_path))
    try:
        cam.GetProjectionAttr().Set(UsdGeom.Tokens.orthographic)
        view_width = max(1.0, float(radius_m) * 2.0)
        cam.GetHorizontalApertureAttr().Set(view_width)
        cam.GetVerticalApertureAttr().Set(view_width * (float(height) / float(width)))
        cam.GetClippingRangeAttr().Set((0.01, 2000.0))
    except Exception:  # noqa: BLE001
        pass
    eye_z = float(floor_z) + max(4.0, float(radius_m) * 2.5)
    # A USD camera with identity orientation already looks down local -Z. For an overhead camera,
    # translating it above the scene is more stable than using the generic look-at path's singular
    # straight-down orientation case.
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(center_xy[0]), float(center_xy[1]), eye_z))


def _write_topdown_debug_layout_image(
    path: Path,
    *,
    robot_pose,
    robot_yaw: float,
    stance_plan: Mapping[str, Any] | None,
    scene_objects: Sequence[Any],
    width: int,
    height: int,
) -> dict[str, Any] | None:
    """Write a deterministic xy layout beside the RTX top-down frame.

    The RTX overhead render can be visually noisy in the reconstructed kitchen. This pure 2D artifact
    uses the exact same world-frame AABBs as placement validation, so the robot footprint, target, and
    counter/cabinet footprints are readable without camera foreshortening.
    """
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    target_obj = _target_object_from_stance_plan(stance_plan or {})
    robot_box = _footprint_box_for_pose(robot_pose)
    overlay_objects = _debug_overlay_objects(scene_objects, target_obj, robot_pose)
    boxes: list[tuple[str, Any, tuple[int, int, int]]] = [
        ("robot", robot_box, (0, 170, 60)),
    ]
    if target_obj is not None:
        boxes.append(("target", {
            "bbox_min_xyz": getattr(target_obj, "bbox_min"),
            "bbox_max_xyz": getattr(target_obj, "bbox_max"),
        }, (220, 30, 30)))
    for obj in overlay_objects:
        boxes.append(("object", {
            "bbox_min_xyz": getattr(obj, "bbox_min"),
            "bbox_max_xyz": getattr(obj, "bbox_max"),
        }, (230, 150, 20)))
    xs: list[float] = []
    ys: list[float] = []
    for _kind, box, _color in boxes:
        bmin = box["bbox_min_xyz"]
        bmax = box["bbox_max_xyz"]
        xs.extend([float(bmin[0]), float(bmax[0])])
        ys.extend([float(bmin[1]), float(bmax[1])])
    if not xs or not ys:
        return None
    pad = 0.45
    min_x, max_x = min(xs) - pad, max(xs) + pad
    min_y, max_y = min(ys) - pad, max(ys) + pad
    span_x = max(0.1, max_x - min_x)
    span_y = max(0.1, max_y - min_y)
    scale = min((width - 80) / span_x, (height - 80) / span_y)
    offset_x = 40 + 0.5 * ((width - 80) - span_x * scale)
    offset_y = 40 + 0.5 * ((height - 80) - span_y * scale)

    def px(x: float, y: float) -> tuple[float, float]:
        return (
            offset_x + (float(x) - min_x) * scale,
            height - (offset_y + (float(y) - min_y) * scale),
        )

    img = Image.new("RGB", (int(width), int(height)), (248, 248, 244))
    draw = ImageDraw.Draw(img, "RGBA")
    for kind, box, color in boxes:
        bmin = box["bbox_min_xyz"]
        bmax = box["bbox_max_xyz"]
        p0 = px(float(bmin[0]), float(bmin[1]))
        p1 = px(float(bmax[0]), float(bmax[1]))
        rect = [min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])]
        alpha = 160 if kind == "robot" else 110
        draw.rectangle(rect, fill=(*color, alpha), outline=(*color, 255), width=3 if kind == "robot" else 2)
    cx, cy = px(float(robot_pose[0]), float(robot_pose[1]))
    fx, fy = math.cos(float(robot_yaw)), math.sin(float(robot_yaw))
    tip = px(float(robot_pose[0]) + fx * 0.7, float(robot_pose[1]) + fy * 0.7)
    draw.line([(cx, cy), tip], fill=(0, 140, 190, 255), width=5)
    draw.ellipse([tip[0] - 7, tip[1] - 7, tip[0] + 7, tip[1] + 7], fill=(0, 140, 190, 255))
    draw.text((18, 14), "top-down placement layout: green=robot footprint, cyan=facing, red=target, orange=fixtures",
              fill=(30, 30, 30))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return {
        "path": str(path),
        "bounds_xy": [round(min_x, 6), round(min_y, 6), round(max_x, 6), round(max_y, 6)],
        "overlay_object_count": len(overlay_objects),
    }


def _force_cheap_collision(stage, approximation: str = "boundingCube") -> int:
    """Override every mesh collision approximation. The 47-object kitchen's default SDF cooking on
    non-watertight meshes takes >4 min and blocks the RTX render; ``boundingCube`` cooks ~instantly
    but is coarse (collision volumes far bigger than the visual shape, which shoves the robot off a
    head-on approach). ``convexHull`` is shape-accurate enough for the robot to stand centered + close
    and still cooks far faster than SDF (a watertight convex per mesh). Visual geometry is untouched.
    Returns the number of meshes overridden."""
    from pxr import UsdPhysics  # type: ignore
    tokens = {
        "boundingCube": UsdPhysics.Tokens.boundingCube,
        "convexHull": UsdPhysics.Tokens.convexHull,
        "convexDecomposition": UsdPhysics.Tokens.convexDecomposition,
    }
    approx = tokens.get(approximation, UsdPhysics.Tokens.boundingCube)
    n = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr().Set(approx)
            n += 1
    return n


def _author_target_contact_material(
    stage,
    target_prim_path,
    *,
    friction: float = 0.85,
    restitution: float = 0.02,
    mass: float | None = 2.0,
    density: float | None = None,
) -> dict[str, Any]:
    """Best-effort contact material authoring scoped to the resolved task target prim only."""
    diag: dict[str, Any] = {
        "schema_version": "isaac_target_contact_material_authoring.v1",
        "status": "blocked",
        "target_prim_path": str(target_prim_path or ""),
        "mass_kg": mass,
        "density_kg_per_m3": density,
        "static_friction": float(friction),
        "dynamic_friction": float(friction),
        "restitution": float(restitution),
        "approximation": "convexDecomposition",
        "bind_purpose": "physics",
        "mutated_prim_paths": [],
        "blockers": [],
    }
    try:
        from pxr import UsdPhysics, UsdShade  # type: ignore

        prim = stage.GetPrimAtPath(str(target_prim_path))
        if not prim or (hasattr(prim, "IsValid") and not prim.IsValid()):
            diag["blockers"].append("target_prim_unavailable")
            return diag

        mutated_paths = {str(target_prim_path)}
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        if mass is not None and hasattr(mass_api, "CreateMassAttr"):
            mass_api.CreateMassAttr(float(mass)).Set(float(mass))
        if density is not None and hasattr(mass_api, "CreateDensityAttr"):
            mass_api.CreateDensityAttr(float(density)).Set(float(density))

        material_path = (
            f"/World/BlueprintPhysicsMaterials/"
            f"{_safe_prim_segment(str(target_prim_path).strip('/').replace('/', '_'))}_contact"
        )
        material = UsdShade.Material.Define(stage, material_path)
        material_prim = material.GetPrim() if hasattr(material, "GetPrim") else material
        material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
        if hasattr(material_api, "CreateStaticFrictionAttr"):
            material_api.CreateStaticFrictionAttr(float(friction)).Set(float(friction))
        if hasattr(material_api, "CreateDynamicFrictionAttr"):
            material_api.CreateDynamicFrictionAttr(float(friction)).Set(float(friction))
        if hasattr(material_api, "CreateRestitutionAttr"):
            material_api.CreateRestitutionAttr(float(restitution)).Set(float(restitution))

        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_api.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)

        bind_api = UsdShade.MaterialBindingAPI.Apply(prim)
        physics_purpose = getattr(getattr(UsdShade, "Tokens", object()), "physics", "physics")
        try:
            bind_api.Bind(material, materialPurpose=physics_purpose)
        except TypeError:
            bind_api.Bind(material, purpose=physics_purpose)

        diag.update({
            "status": "authored",
            "mutated_prim_paths": sorted(mutated_paths),
            "material_prim_path": material_path,
        })
    except Exception as exc:  # noqa: BLE001
        diag["blockers"].append(f"target_contact_material_authoring_failed:{exc!r}")
    return diag


def _prune_to_focus(stage, route_points, focus_radius: float, keep_substrings) -> dict:
    """Task-aware scene subset: deactivate kitchen object prims whose placement is farther than
    ``focus_radius`` (m, xy) from the robot's route, keeping the task region + structural shell
    (walls/floor/lights). Deactivated prims are excluded from BOTH PhysX cooking and the render,
    so this is the lever for 'the scene is too large'. Returns {kept, pruned, kept_names}."""
    from pxr import UsdGeom  # type: ignore
    xc = UsdGeom.XformCache()
    keep_subs = [s.strip().lower() for s in keep_substrings if s and s.strip()]
    pts = [(float(p[0]), float(p[1])) for p in route_points] or [(0.0, 0.0)]
    root = stage.GetPrimAtPath("/root")
    if not (root and root.IsValid()):
        root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    kept, pruned, kept_names = 0, 0, []
    for child in root.GetChildren():
        name = child.GetName()
        low = name.lower()
        if any(s in low for s in keep_subs):
            kept += 1
            kept_names.append(name)
            continue
        try:
            t = xc.GetLocalToWorldTransform(child).ExtractTranslation()
            pos = (float(t[0]), float(t[1]))
            dmin = min(math.hypot(pos[0] - x, pos[1] - y) for x, y in pts)
        except Exception:  # noqa: BLE001
            kept += 1
            kept_names.append(name)
            continue
        if dmin <= focus_radius:
            kept += 1
            kept_names.append(name)
        else:
            child.SetActive(False)
            pruned += 1
    return {"kept": kept, "pruned": pruned, "kept_names": kept_names[:40]}


def _make_render_product(
    camera_path: str,
    width: int,
    height: int,
    *,
    with_depth: bool = False,
    with_segmentation: bool = False,
):
    import omni.replicator.core as rep  # type: ignore
    rp = rep.create.render_product(camera_path, (width, height))
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach([rp])
    if not with_depth and not with_segmentation:
        return annot
    annots: dict[str, Any] = {"rgb": annot}
    if with_depth:
        depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        depth_annot.attach([rp])
        annots["depth"] = depth_annot
    if with_segmentation:
        inst_annot = rep.AnnotatorRegistry.get_annotator(
            "instance_segmentation",
            init_params={"colorize": True},
        )
        sem_annot = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation",
            init_params={"colorize": True},
        )
        inst_annot.attach([rp])
        sem_annot.attach([rp])
        annots["instance"] = inst_annot
        annots["semantic"] = sem_annot
    if with_depth and not with_segmentation:
        depth_annot = annots["depth"]
        return annot, depth_annot
    if with_segmentation:
        return annots
    return annot


def _isaac_camera_contract(stage, cam_path: str, width: int, height: int) -> dict[str, Any]:
    try:
        from pxr import UsdGeom  # type: ignore

        prim = stage.GetPrimAtPath(cam_path)
        cam = UsdGeom.Camera(prim)
        focal = cam.GetFocalLengthAttr().Get()
        h_ap = cam.GetHorizontalApertureAttr().Get()
        v_ap = cam.GetVerticalApertureAttr().Get()
        proj_token = str(cam.GetProjectionAttr().Get() or "perspective")
        xform_cache = UsdGeom.XformCache()
        matrix = xform_cache.GetLocalToWorldTransform(cam.GetPrim())
        translation = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotationMatrix()
        if proj_token == "orthographic":
            intrinsics = {
                "available": False,
                "projection_token": "orthographic",
            }
        else:
            intrinsics = _camera_intrinsics_from_usd_aperture(
                focal,
                h_ap,
                v_ap,
                width,
                height,
            )
        return {
            "available": True,
            "camera_id": str(cam_path).rsplit("/", 1)[-1],
            "camera_path": cam_path,
            "intrinsics": intrinsics,
            "camera_world_xyz_m": [round(float(translation[i]), 6) for i in range(3)],
            "camera_xmat_row_major": [
                [round(float(rotation[i][j]), 8) for j in range(3)]
                for i in range(3)
            ],
            "resolution": [int(width), int(height)],
            "projection_token": proj_token,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "camera_path": cam_path,
            "blockers": ["isaac_camera_contract_unavailable"],
            "error": repr(exc),
        }


def _render_step_watchdog_seconds() -> float:
    raw = os.getenv("PARITY_RENDER_STEP_WATCHDOG_SECONDS", "").strip()
    if not raw:
        return DEFAULT_RENDER_STEP_WATCHDOG_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_RENDER_STEP_WATCHDOG_SECONDS


def _audit_render_step_watchdog_seconds() -> float:
    raw = os.getenv("PARITY_AUDIT_RENDER_STEP_WATCHDOG_SECONDS", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    # Never tighter than the generic step watchdog an operator may have raised.
    return max(DEFAULT_AUDIT_RENDER_STEP_WATCHDOG_SECONDS, _render_step_watchdog_seconds())


def _auto_render_settle_seconds(
    *,
    configured_settle_seconds: float,
    no_collision_probe: bool,
    manipulation_cam: bool,
    verify_cam: bool,
    manipulation_stand: bool,
    warmup_frames: int,
    render_subframes: int,
) -> float:
    """Return a post-placement settle time before repeated RTX stepping.

    Isaac/PhysX collision queries can start async cooking during task-stance probing. The old
    global settle happened before those probes, so repeated RTX steps could still render while
    cooking was active. Keep this task-agnostic and opt-out via env/config instead of baking in
    any scene or object coordinates.
    """
    if configured_settle_seconds > 0:
        return float(configured_settle_seconds)
    if no_collision_probe:
        return 0.0
    repeated_or_review_render = (
        bool(manipulation_cam)
        or bool(verify_cam)
        or bool(manipulation_stand)
        or int(warmup_frames) > 1
        or int(render_subframes) > 1
    )
    if not repeated_or_review_render:
        return 0.0
    raw = os.getenv("PARITY_AUTO_RENDER_SETTLE_SECONDS", "60").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 60.0


def _render_quality_config(
    *,
    render_subframes: int,
    manipulation_cam: bool,
    verify_cam: bool,
    mode: str | None = None,
    samples_per_pixel: int | None = None,
) -> dict[str, Any]:
    """Return scene-agnostic RTX quality settings for review manipulation frames.

    Review seed frames default to real-time RTX lighting because it is deterministic/rasterized
    enough for policy/WAM source observations and avoids Monte-Carlo speckle on non-white robot
    materials. Path tracing remains available as an explicit audit/rendering mode when physically
    richer lighting is more important than seed-frame cleanliness.
    """
    subframes = max(1, int(render_subframes))
    raw_mode = str(
        mode if mode is not None else os.getenv("PARITY_RENDER_QUALITY_MODE", "auto")
    ).strip().lower()
    normalized_mode = raw_mode or "auto"
    disable_modes = {"off", "disabled", "none", "raytracedlighting", "ray_traced_lighting", "realtime"}
    pathtraced_modes = {"pathtraced", "path_traced", "path-traced", "pt"}
    if normalized_mode in disable_modes:
        use_pathtraced = False
    elif normalized_mode in pathtraced_modes:
        use_pathtraced = True
    else:
        use_pathtraced = False

    if samples_per_pixel is None:
        raw_spp = os.getenv("PARITY_PATH_TRACING_SAMPLES_PER_PIXEL", "").strip()
        try:
            samples = int(raw_spp) if raw_spp else max(
                DEFAULT_PATH_TRACING_MIN_SAMPLES_PER_PIXEL,
                min(DEFAULT_PATH_TRACING_MAX_SAMPLES_PER_PIXEL, subframes * 2),
            )
        except ValueError:
            samples = max(
                DEFAULT_PATH_TRACING_MIN_SAMPLES_PER_PIXEL,
                min(DEFAULT_PATH_TRACING_MAX_SAMPLES_PER_PIXEL, subframes * 2),
            )
    else:
        samples = int(samples_per_pixel)
    samples = max(1, min(512, samples))
    return {
        "schema_version": "isaac_render_quality_config.v1",
        "mode": normalized_mode,
        "render_subframes": subframes,
        "use_pathtraced": bool(use_pathtraced),
        "samples_per_pixel": samples if use_pathtraced else 0,
        "optix_denoiser_requested": bool(use_pathtraced),
        "firefly_filter_requested": bool(use_pathtraced),
        "claim_boundary": (
            "Render-quality settings affect review PNG cleanliness only. They do not validate task "
            "success, policy quality, physical reach, safety, or deployment readiness."
        ),
    }


def _apply_render_quality_settings(
    rep,
    *,
    render_subframes: int,
    manipulation_cam: bool,
    verify_cam: bool,
    out_dir: Path,
) -> dict[str, Any]:
    config = _render_quality_config(
        render_subframes=render_subframes,
        manipulation_cam=manipulation_cam,
        verify_cam=verify_cam,
    )
    diagnostics = dict(config)
    diagnostics["settings_applied"] = []
    diagnostics["setting_errors"] = []
    if config["use_pathtraced"]:
        try:
            rep.settings.set_render_pathtraced(
                samples_per_pixel=int(config["samples_per_pixel"])
            )
            diagnostics["settings_applied"].append("rep.settings.set_render_pathtraced")
        except Exception as exc:  # noqa: BLE001
            diagnostics["setting_errors"].append({
                "setting": "rep.settings.set_render_pathtraced",
                "error": repr(exc),
            })
        try:
            import carb  # type: ignore

            settings = carb.settings.get_settings()
            for path, value in (
                # Explicit per-frame sample budget, matching the render-noise audit path.
                # rep.settings.set_render_pathtraced alone left /rtx/pathtracing/spp at the
                # engine default, so every captured frame after a robot pose change rendered
                # starved (metallic hf_noise 1.77 vs 0.58 with these keys set, 2026-07-02).
                ("/rtx/pathtracing/spp", int(config["samples_per_pixel"])),
                ("/rtx/pathtracing/totalSpp", int(config["samples_per_pixel"])),
                ("/rtx/pathtracing/optixDenoiser/enabled", True),
                ("/rtx/pathtracing/optixDenoiser/blendFactor", 0.0),
                ("/rtx/pathtracing/fireflyFilter/enabled", True),
                ("/rtx/pathtracing/fireflyFilter/maxIntensityPerSample", 350.0),
                ("/rtx/pathtracing/fireflyFilter/maxIntensityPerSampleDiffuse", 350.0),
                ("/rtx-transient/resourcemanager/enableTextureStreaming", False),
            ):
                try:
                    settings.set(path, value)
                    diagnostics["settings_applied"].append(path)
                except Exception as exc:  # noqa: BLE001
                    diagnostics["setting_errors"].append({
                        "setting": path,
                        "error": repr(exc),
                    })
        except Exception as exc:  # noqa: BLE001
            diagnostics["setting_errors"].append({
                "setting": "carb.settings",
                "error": repr(exc),
            })
    diagnostics["status"] = "PASS" if not diagnostics["setting_errors"] else "WARN"
    try:
        (out_dir / "render_quality_settings.json").write_text(
            json.dumps(diagnostics, indent=2),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        pass
    return diagnostics


def _effective_render_rt_subframes(
    render_subframes: int,
    render_quality: Mapping[str, Any] | None,
) -> int:
    """Replicator step subframes after renderer-mode selection.

    In realtime RTX lighting, ``render_subframes`` is the main accumulation lever. In path-traced
    mode ``samples_per_pixel`` is the quality lever ONLY for a static scene: path-tracing
    accumulation RESETS on scene changes, and this runner moves the robot pose between captured
    frames, so a single step per capture renders a starved, grainy frame (metallic robot measured
    hf_noise 1.77 vs 0.58 for the same materials rendered static, 2026-07-02). The REAL sample
    lever is the explicit ``/rtx/pathtracing/spp`` per-frame budget set by
    ``_apply_render_quality_settings`` (subframes measurably did NOT reduce noise when spp was
    starved, and multiply cost once spp is correct), so path-traced steps default to 1 subframe.
    ``PARITY_PATH_TRACED_RT_SUBFRAMES`` still overrides for experiments.
    """
    requested = max(1, int(render_subframes))
    if not (render_quality or {}).get("use_pathtraced"):
        return requested
    raw = os.getenv("PARITY_PATH_TRACED_RT_SUBFRAMES", "").strip()
    try:
        return max(1, min(8, int(raw))) if raw else 1
    except ValueError:
        return 1


def _capture_settle_steps(render_quality: Mapping[str, Any] | None) -> int:
    """Extra replicator steps rendered after a pose change before each capture.

    Ports the render-noise audit's empirically clean recipe (3 settle steps + 1
    capture step per frame, ``per_variant_settle_frames`` in its manifest) into
    the scenario loop. Path-tracing accumulates across repeated ``step()`` calls
    on a still scene; a single step after robot motion rendered hf_noise
    1.77-2.70 at the sink vs the audit's 0.58 with settling (2026-07-02).
    Realtime RTX needs no settling. ``PARITY_CAPTURE_SETTLE_FRAMES`` overrides.
    """
    if not (render_quality or {}).get("use_pathtraced"):
        return 0
    raw = os.getenv("PARITY_CAPTURE_SETTLE_FRAMES", "").strip()
    try:
        return max(0, min(8, int(raw))) if raw else 3
    except ValueError:
        return 3


def _effective_software_denoise(
    software_denoise: bool,
    render_quality: Mapping[str, Any] | None,
) -> bool:
    """Return whether saved RGB review frames should get deterministic CPU denoise.

    The renderer can still emit path-tracing speckle after a non-white review-material pose change
    even when RTX/OptiX denoise is requested. Keep the saved-frame cleanup deterministic and recorded
    in the render-quality manifest; the downstream source QA gate remains the authority on whether
    the frame is clean enough to use as policy/WAM evidence.
    """
    if (render_quality or {}).get("use_pathtraced"):
        raw = os.getenv("PARITY_SOFTWARE_DENOISE_PATH_TRACED", "auto").strip().lower()
        if raw in {"0", "false", "no", "off", "none", "disabled", "raw"}:
            return False
        return bool(software_denoise)
    return bool(software_denoise)


def _write_render_step_timeout_result(path: Path, *, label: str, seconds: float, scenario_id: str) -> None:
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "policy_id": None,
        "rendered_by_isaac_rtx": True,
        "scenarios_executed": 0,
        "scenarios_passed": 0,
        "blockers": ["render_step_timeout"],
        "render_step_timeout": {
            "label": label,
            "seconds": round(float(seconds), 3),
            "scenario_id": scenario_id,
        },
        "claim_boundary": (
            "The renderer watchdog only bounds a stuck RTX render step. It does not validate task "
            "success, manipulation success, physical readiness, safety, or deployment approval."
        ),
    }
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _replicator_step_with_watchdog(
    rep,
    *,
    label: str,
    result_path: Path,
    scenario_id: str,
    timeout_seconds: float | None = None,
    rt_subframes: int = 1,
) -> None:
    """Run one Replicator step with a process-level timeout for C++ RTX hangs.

    A Python signal cannot reliably interrupt a blocked native render call. If the watchdog fires,
    write a blocked result and exit this runner process; the parent bootstrap records ``runner_done``
    and uploads the final artifact zip so the paid pod can be stopped.
    """
    timeout = _render_step_watchdog_seconds() if timeout_seconds is None else float(timeout_seconds)
    subframes = max(1, int(rt_subframes))

    def _step() -> None:
        try:
            rep.orchestrator.step(rt_subframes=subframes)
        except TypeError:
            rep.orchestrator.step()
        try:
            rep.orchestrator.wait_until_complete()
        except Exception:  # noqa: BLE001
            pass

    if timeout <= 0:
        _step()
        return
    done = threading.Event()

    def watchdog() -> None:
        if done.wait(timeout):
            return
        _log(f"render step watchdog timeout after {timeout:.1f}s at {label}; exiting runner")
        _write_render_step_timeout_result(
            result_path,
            label=label,
            seconds=timeout,
            scenario_id=scenario_id,
        )
        os._exit(124)

    thread = threading.Thread(target=watchdog, name="render-step-watchdog", daemon=True)
    thread.start()
    try:
        _step()
    finally:
        done.set()


def _software_denoise_image(img):
    """Best-effort CPU denoise for review PNGs when RTX/NGX denoising is unavailable on a pod.

    The noisy Isaac failure mode seen on manipulation POVs is mostly salt-and-pepper/path-tracing
    firefly speckle. OpenCV NLM alone leaves that pattern as high-frequency edges, so the default is
    a deterministic median firefly cleanup. This is intentionally not image generation or semantic
    enhancement; it only suppresses isolated renderer noise in saved review frames.
    """
    mode = str(os.getenv("PARITY_SOFTWARE_DENOISE_MODE", "median_firefly")).strip().lower()
    if mode in {"off", "none", "disabled"}:
        return img
    if mode in {"median", "median_firefly", "firefly"}:
        try:
            from PIL import ImageFilter  # type: ignore
            return (
                img
                .filter(ImageFilter.MedianFilter(size=3))
                .filter(ImageFilter.MedianFilter(size=3))
                .filter(ImageFilter.SMOOTH_MORE)
            )
        except Exception:  # noqa: BLE001
            return img
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
        arr = np.asarray(img)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        den = cv2.fastNlMeansDenoisingColored(bgr, None, 12, 12, 7, 21)
        out = Image.fromarray(cv2.cvtColor(den, cv2.COLOR_BGR2RGB))
        if mode in {"nlm_median", "opencv_median"}:
            from PIL import ImageFilter  # type: ignore
            out = out.filter(ImageFilter.MedianFilter(size=3)).filter(ImageFilter.SMOOTH_MORE)
        return out
    except Exception:  # noqa: BLE001
        pass
    try:
        from PIL import ImageFilter  # type: ignore
        return (
            img
            .filter(ImageFilter.MedianFilter(size=3))
            .filter(ImageFilter.MedianFilter(size=3))
            .filter(ImageFilter.SMOOTH_MORE)
        )
    except Exception:  # noqa: BLE001
        return img


def _save_rgb(annot, out_path: Path, *, software_denoise: bool = False) -> bool:
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore
    data = annot.get_data()
    if data is None or getattr(data, "size", 0) == 0:
        return False
    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    img = Image.fromarray(arr.astype("uint8"))
    if software_denoise:
        img = _software_denoise_image(img)
    img.save(out_path)
    return True


def _save_depth(depth_annot, out_path: Path, *, npy_path: Path | None = None) -> bool:
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    data = depth_annot.get_data()
    if data is None or getattr(data, "size", 0) == 0:
        return False
    arr = np.asarray(data).astype("float32")
    raw_path = npy_path or out_path.with_suffix(".npy")
    np.save(raw_path, arr)

    finite = np.isfinite(arr) & (arr > 0)
    preview = np.zeros(arr.shape, dtype="uint8")
    if np.any(finite):
        valid = arr[finite]
        dmin = float(valid.min())
        dmax = float(valid.max())
        if dmax > dmin:
            norm = (arr - dmin) / (dmax - dmin)
            preview = np.where(finite, np.clip(norm * 255.0, 0, 255), 0).astype("uint8")
        else:
            preview = np.where(finite, 255, 0).astype("uint8")
    Image.fromarray(preview).save(out_path)
    return True


def _segmentation_payload(data: Any) -> Any:
    if isinstance(data, Mapping):
        for key in ("data", "image", "rgba"):
            if data.get(key) is not None:
                return data.get(key)
        return None
    return data


def _save_segmentation(
    seg_annots: Mapping[str, Any],
    *,
    instance_png: Path,
    semantic_png: Path,
    id_label_json: Path,
) -> dict[str, Any]:
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    blockers: list[str] = []

    def _save_one(annot: Any, out_path: Path) -> tuple[bool, Any]:
        if annot is None:
            return False, None
        data = annot.get_data()
        payload = _segmentation_payload(data)
        if payload is None or getattr(payload, "size", 0) == 0:
            return False, data
        arr = np.asarray(payload)
        if arr.size == 0:
            return False, data
        Image.fromarray(arr.astype("uint8")).save(out_path)
        return True, data

    instance_png.parent.mkdir(parents=True, exist_ok=True)
    instance_saved, instance_data = _save_one(seg_annots.get("instance"), instance_png)
    semantic_saved, semantic_data = _save_one(seg_annots.get("semantic"), semantic_png)
    if not instance_saved:
        blockers.append("instance_segmentation_mask_not_saved")
    if not semantic_saved:
        blockers.append("semantic_segmentation_mask_not_saved")
    instance_info = instance_data.get("info") if isinstance(instance_data, Mapping) else {}
    if not isinstance(instance_info, Mapping):
        instance_info = {}
    id_to_labels = instance_info.get("idToLabels") or {}
    id_label_json.parent.mkdir(parents=True, exist_ok=True)
    id_label_json.write_text(json.dumps(id_to_labels, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "schema_version": "isaac_segmentation_frame_save.v1",
        "instance_saved": bool(instance_saved),
        "semantic_saved": bool(semantic_saved),
        "instance_png": str(instance_png) if instance_saved else None,
        "semantic_png": str(semantic_png) if semantic_saved else None,
        "id_label_json": str(id_label_json),
        "id_to_labels": id_to_labels,
        "blockers": blockers,
        "semantic_data_present": semantic_data is not None,
    }


def camera_aperture_for_fov(vfov_deg: float, width: int, height: int, focal_mm: float = 20.0):
    """Focal length + (horizontal, vertical) aperture that give a camera a vertical FOV of
    ``vfov_deg`` at the render aspect ratio. USD's default 50mm/20.955mm camera is a ~24deg
    telephoto — far too zoomed for the manipulation POV and it does NOT match the FOV
    the skeleton projection assumes, so the projected
    landmarks misalign with the render. Pure trig (no USD) so it is unit-testable."""
    vap = 2.0 * float(focal_mm) * math.tan(math.radians(float(vfov_deg)) / 2.0)
    hap = vap * (float(width) / float(height))
    return float(focal_mm), hap, vap


def _camera_intrinsics_from_usd_aperture(
    focal_mm,
    h_aperture_mm,
    v_aperture_mm,
    width: int,
    height: int,
) -> dict[str, Any]:
    if focal_mm is None or not h_aperture_mm or not v_aperture_mm:
        return {
            "available": False,
            "blockers": ["camera_aperture_unavailable"],
        }
    focal = float(focal_mm)
    h_aperture = float(h_aperture_mm)
    v_aperture = float(v_aperture_mm)
    return {
        "available": True,
        "fx": float(width) * focal / h_aperture,
        "fy": float(height) * focal / v_aperture,
        "cx": float(width) / 2.0,
        "cy": float(height) / 2.0,
        "image_width": int(width),
        "image_height": int(height),
        "focal_length_mm": focal,
        "horizontal_aperture_mm": h_aperture,
        "vertical_aperture_mm": v_aperture,
        "projection_method": "isaac_usd_camera_pinhole_from_focal_aperture",
    }


def _set_camera_fov(stage, cam_path: str, vfov_deg: float, width: int, height: int) -> None:
    """Set a USD camera's focal length + apertures so its vertical FOV == ``vfov_deg`` (matching the
    skeleton-projection FOV) instead of the narrow ~17deg default. GPU/USD only."""
    from pxr import UsdGeom  # type: ignore
    focal, hap, vap = camera_aperture_for_fov(vfov_deg, width, height)
    cam = UsdGeom.Camera(stage.GetPrimAtPath(cam_path))
    cam.GetFocalLengthAttr().Set(focal)
    cam.GetHorizontalApertureAttr().Set(hap)
    cam.GetVerticalApertureAttr().Set(vap)
    try:
        # A small near plane keeps the head-mounted POV from embedding in its own head mesh, but a
        # 0.01-0.02 near with a 1000-2000 far wrecks the depth-buffer precision the RTX denoiser relies
        # on -> heavy salt-and-pepper grain. A 0.05/50 range (1000:1) avoids embedding and remains
        # broad enough for room-scale manipulation scenes without wasting depth precision.
        cam.GetClippingRangeAttr().Set((0.05, 50.0))
    except Exception:  # noqa: BLE001
        pass


ROBOT_REVIEW_MATERIAL_SPECS = {
    "neutral_matte": {
        "label": "neutral_matte_light_gray",
        "diffuse_color": (0.82, 0.84, 0.86),
        "roughness": 0.72,
    },
    "non_white_matte": {
        "label": "non_white_matte_blue_gray",
        "diffuse_color": (0.32, 0.42, 0.50),
        "roughness": 0.78,
    },
}


def _robot_review_material_spec(mode: str | None) -> dict[str, Any]:
    normalized = str(mode or "neutral_matte").strip().lower().replace("-", "_")
    return dict(
        ROBOT_REVIEW_MATERIAL_SPECS.get(
            normalized,
            ROBOT_REVIEW_MATERIAL_SPECS["neutral_matte"],
        )
    )


def _scenario_instruction(sc: Mapping[str, Any]) -> str:
    for key in (
        "task_instruction",
        "task_prompt",
        "task_description",
        "instruction",
        "task",
        "description",
    ):
        value = sc.get(key)
        if value:
            return str(value)
    return "Execute the simulated kitchen manipulation task."


def _build_groot_policy_command_payload(
    *,
    scenario: Mapping[str, Any],
    frame_path: str | Path,
    step: int,
) -> dict[str, Any]:
    frame = Path(frame_path).expanduser()
    instruction = _scenario_instruction(scenario)
    observation = {
        "schema_version": "isaac_g1_kitchen_groot_policy_observation.v1",
        "task_id": scenario.get("task_id") or scenario.get("scenario_id") or scenario.get("id"),
        "scenario_id": scenario.get("scenario_id") or scenario.get("id"),
        "step": int(step),
        "camera_frame_path": str(frame),
        "visual_observation": {"camera_frame_path": str(frame)},
        "task_prompt": instruction,
        "task_description": instruction,
        "task_instruction": instruction,
        "unitree_g1_sonic_state": dict(UNITREE_G1_SONIC_NEUTRAL_STATE),
        "unitree_g1_sonic_state_source": "isaac_parity_neutral_unitree_g1_sonic_contract_state",
        "unitree_g1_sonic_state_metadata": {
            "schema_version": "unitree_g1_sonic_state_metadata.v1",
            "complete": True,
            "source": "isaac_parity_neutral_contract_probe",
            "physical_proprioception": False,
            "simulator_state_capture": False,
            "claim_boundary": (
                "Neutral contract state is only an interface probe for simulator policy action "
                "generation. It is not measured physical robot proprioception."
            ),
        },
    }
    return {"observation": observation}


def _policy_command_result_action(result: Mapping[str, Any]) -> dict[str, Any]:
    action = result.get("action") or result.get("normalized_action")
    if isinstance(action, Mapping):
        return dict(action)
    return dict(result)


def _run_groot_policy_command(
    *,
    command: str,
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    parts = shlex.split(str(command))
    if not parts:
        raise RuntimeError("groot_policy_command_empty")
    completed = subprocess.run(
        parts,
        input=json.dumps(dict(payload)),
        capture_output=True,
        text=True,
        check=False,
        timeout=float(timeout_seconds),
        env={**os.environ},
    )
    output = (completed.stdout or "").strip()
    if completed.returncode != 0:
        raise RuntimeError(
            "groot_policy_command_nonzero_exit:"
            f"returncode={completed.returncode}:stderr_bytes={len(completed.stderr or '')}"
        )
    if not output:
        raise RuntimeError("groot_policy_command_stdout_empty")
    try:
        value = json.loads(output.splitlines()[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("groot_policy_command_stdout_not_json") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("groot_policy_command_stdout_not_json_object")
    result = dict(value)
    if result.get("status") == "blocked" or result.get("blockers"):
        blockers = ",".join(str(item) for item in result.get("blockers") or [])
        raise RuntimeError(f"groot_policy_command_blocked:{blockers}")
    return result


def _make_groot_policy_command_infer(
    *,
    command: str,
    scenario: Mapping[str, Any],
    call_dir: Path,
    timeout_seconds: float,
):
    call_dir.mkdir(parents=True, exist_ok=True)

    def _infer(obs: Mapping[str, Any]) -> dict[str, Any]:
        step = int(obs.get("step") or 0)
        frame = str(obs.get("camera_rgb") or "").strip()
        if not frame or not Path(frame).expanduser().is_file():
            raise RuntimeError("groot_policy_observation_frame_missing")
        payload = _build_groot_policy_command_payload(
            scenario=scenario,
            frame_path=frame,
            step=step,
        )
        call_path = call_dir / f"groot_policy_call_{step:04d}.json"
        result = _run_groot_policy_command(
            command=command,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        call_path.write_text(
            json.dumps(
                {
                    "schema_version": "isaac_g1_groot_policy_command_call.v1",
                    "status": result.get("status"),
                    "step": step,
                    "command_configured": True,
                    "command_value_redacted": "<configured>",
                    "payload": payload,
                    "result": result,
                    "raw_credentials_written_to_artifacts": False,
                    "secret_hashes_written_to_artifacts": False,
                    "claim_boundary": (
                        "This records the simulator policy-command request/response shape. It is "
                        "not task-success proof until the returned action drives an episode that "
                        "passes semantic grading."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return _policy_command_result_action(result)

    return _infer


def _apply_robot_review_material(
    stage,
    robot_prim_path: str,
    *,
    override_authored_materials: bool = True,
    material_mode: str = "neutral_matte",
) -> int:
    """Bind a neutral matte material to robot geometry so review renders can see G1 against dark targets.

    This is intentionally robot-scoped and scene/task agnostic. Visibility/purpose normalization is
    always applied, but authored materials/textures are preserved unless ``override_authored_materials``
    is true. The override is for missing-material diagnostics and review proxies, not final seed media.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # type: ignore

    robot = stage.GetPrimAtPath(robot_prim_path)
    if not (robot and robot.IsValid()):
        return 0
    material = None
    bound = 0
    if override_authored_materials:
        material_spec = _robot_review_material_spec(material_mode)
        color = tuple(float(v) for v in material_spec["diffuse_color"])
        mat_path = "/World/Materials/RobotReviewVisible"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            float(material_spec["roughness"])
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        try:
            UsdShade.MaterialBindingAPI(robot).Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
            bound = 1
        except Exception:  # noqa: BLE001
            bound = 0
    for prim in Usd.PrimRange(robot):
        try:
            if prim.IsA(UsdGeom.Imageable):
                imageable = UsdGeom.Imageable(prim)
                imageable.MakeVisible()
                purpose_attr = imageable.GetPurposeAttr()
                purpose = purpose_attr.Get()
                if str(purpose or "") in {"guide", "proxy"}:
                    purpose_attr.Set("default")
            if not prim.IsA(UsdGeom.Gprim):
                continue
            if material is not None:
                UsdShade.MaterialBindingAPI(prim).Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                )
                UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(0.82, 0.84, 0.86)])
                bound += 1
        except Exception:  # noqa: BLE001
            continue
    return bound


def _robot_render_visibility_diagnostics(stage, robot_prim_path: str) -> dict[str, Any]:
    """Summarize robot visual/renderability state for debugging RTX frames.

    The placement validator can pass from collision/BBox geometry even when the render products do not
    show a readable robot. This artifact is robot-subtree scoped and dynamic; it records whether G1 has
    visible imageable/gprim descendants after material binding without assuming scene/task coordinates.
    """
    from pxr import Usd, UsdGeom, UsdShade  # type: ignore

    robot = stage.GetPrimAtPath(robot_prim_path)
    if not (robot and robot.IsValid()):
        return {
            "schema_version": "robot_render_visibility_diagnostics.v1",
            "status": "FAIL",
            "blockers": ["robot_prim_missing"],
            "robot_prim_path": robot_prim_path,
        }

    purpose_counts: dict[str, int] = {}
    visibility_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    total_prims = imageable_count = gprim_count = mesh_count = material_bound_count = 0
    instanceable_count = 0
    sample_gprims: list[dict[str, Any]] = []
    arm_gprims: list[dict[str, Any]] = []
    try:
        prim_iter = Usd.PrimRange(robot, Usd.TraverseInstanceProxies())
        traversed_instance_proxies = True
    except Exception:  # noqa: BLE001
        prim_iter = Usd.PrimRange(robot)
        traversed_instance_proxies = False
    for prim in prim_iter:
        total_prims += 1
        type_name = str(prim.GetTypeName() or "typeless")
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
        try:
            if prim.IsInstanceable():
                instanceable_count += 1
        except Exception:  # noqa: BLE001
            pass
        try:
            if prim.IsA(UsdGeom.Imageable):
                imageable_count += 1
                imageable = UsdGeom.Imageable(prim)
                purpose = str(imageable.GetPurposeAttr().Get() or "default")
                visibility = str(imageable.ComputeVisibility() or "unknown")
                purpose_counts[purpose] = purpose_counts.get(purpose, 0) + 1
                visibility_counts[visibility] = visibility_counts.get(visibility, 0) + 1
        except Exception:  # noqa: BLE001
            pass
        try:
            if not prim.IsA(UsdGeom.Gprim):
                continue
            gprim_count += 1
            if prim.IsA(UsdGeom.Mesh):
                mesh_count += 1
            material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            material_path = str(material.GetPath()) if material else None
            if material_path:
                material_bound_count += 1
            entry = {
                "path": str(prim.GetPath()),
                "type_name": type_name,
                "material_path": material_path,
            }
            try:
                bbox = _world_bbox_for_prim(stage, str(prim.GetPath()))
                if bbox:
                    entry["bbox_min_xyz"] = bbox.get("bbox_min_xyz")
                    entry["bbox_max_xyz"] = bbox.get("bbox_max_xyz")
                    entry["size_xyz"] = bbox.get("size_xyz")
            except Exception:  # noqa: BLE001
                pass
            if len(sample_gprims) < 24:
                sample_gprims.append(entry)
            lower_path = str(prim.GetPath()).lower()
            if (
                len(arm_gprims) < 32
                and any(side in lower_path for side in ("left", "right"))
                and any(token in lower_path for token in MANIPULATION_ARM_LINK_NAME_TOKENS)
            ):
                arm_gprims.append(entry)
        except Exception:  # noqa: BLE001
            continue

    blockers: list[str] = []
    if imageable_count == 0:
        blockers.append("robot_imageable_prims_missing")
    if gprim_count == 0:
        blockers.append(ROBOT_VISUAL_MESH_MISSING_BLOCKER)
    if gprim_count > 0 and material_bound_count == 0:
        blockers.append("robot_gprims_unmaterialized")
    if visibility_counts and visibility_counts.get("invisible", 0) >= max(1, imageable_count):
        blockers.append("robot_imageables_all_invisible")

    return {
        "schema_version": "robot_render_visibility_diagnostics.v1",
        "status": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "robot_prim_path": robot_prim_path,
        "root_bbox": _world_bbox_for_prim(stage, robot_prim_path),
        "total_descendant_prims": total_prims,
        "imageable_prim_count": imageable_count,
        "gprim_count": gprim_count,
        "mesh_count": mesh_count,
        "renderable_robot_geometry_present": gprim_count > 0,
        "traversed_instance_proxies": traversed_instance_proxies,
        "material_bound_gprim_count": material_bound_count,
        "instanceable_prim_count": instanceable_count,
        "purpose_counts": dict(sorted(purpose_counts.items())),
        "visibility_counts": dict(sorted(visibility_counts.items())),
        "type_counts": dict(sorted(type_counts.items())),
        "sample_gprims": sample_gprims,
        "arm_gprim_samples": arm_gprims,
        "claim_boundary": (
            "Diagnostics report USD robot visual/renderability state only. They do not prove "
            "manipulation success, policy quality, or physical readiness."
        ),
    }


def _add_robot_review_proxy_box(
    stage,
    path: str,
    *,
    center: Sequence[float],
    size: Sequence[float],
    color: Sequence[float],
) -> bool:
    """Add a non-physics review box in world coordinates."""
    from pxr import Gf, UsdGeom, UsdShade  # type: ignore

    try:
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        prim = cube.GetPrim()
        imageable = UsdGeom.Imageable(prim)
        imageable.MakeVisible()
        imageable.GetPurposeAttr().Set("default")
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])))
        xf.AddScaleOp().Set(Gf.Vec3f(
            max(0.01, float(size[0])),
            max(0.01, float(size[1])),
            max(0.01, float(size[2])),
        ))
        gprim = UsdGeom.Gprim(prim)
        gprim.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
        material_prim = stage.GetPrimAtPath("/World/Materials/RobotReviewVisible")
        if material_prim and material_prim.IsValid():
            UsdShade.MaterialBindingAPI(prim).Bind(
                UsdShade.Material(material_prim),
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
        return True
    except Exception:  # noqa: BLE001
        return False


def _create_robot_review_visual_proxies(
    stage,
    robot_prim_path: str,
    *,
    proxy_root_path: str,
    arm: str = "both",
) -> dict[str, Any]:
    """Create render-only robot geometry from the live G1 link/bbox state.

    Some Isaac worker images expose the official G1 as articulation/collision Xforms with no renderable
    Gprims. The placement and link projection math can still be valid, but RTX PNGs show no robot.
    These proxies are a visual review layer only: they are derived from the robot subtree's current
    world-space bbox and arm link transforms, live outside ``/World/G1`` so placement/collision bounds
    stay tied to the actual robot, and add no physics/collision APIs.
    """
    from pxr import UsdGeom  # type: ignore

    try:
        stage.RemovePrim(proxy_root_path)
    except Exception:  # noqa: BLE001
        pass
    try:
        UsdGeom.Scope.Define(stage, proxy_root_path)
    except Exception:  # noqa: BLE001
        pass

    created = 0
    blockers: list[str] = []
    bbox = _world_bbox_for_prim(stage, robot_prim_path)
    side_points = _robot_arm_link_points_by_arm(stage, robot_prim_path, arm=arm)
    color_body = (0.74, 0.77, 0.80)
    color_arm = (0.86, 0.88, 0.90)
    body_boxes: list[str] = []
    arm_boxes: list[str] = []

    if bbox:
        try:
            bmin = tuple(float(v) for v in bbox["bbox_min_xyz"])
            center = tuple(float(v) for v in bbox["center_xyz"])
            size = tuple(max(0.01, float(v)) for v in bbox["size_xyz"])
            torso_center = (
                center[0],
                center[1],
                bmin[2] + size[2] * 0.58,
            )
            torso_size = (
                max(0.14, size[0] * 0.42),
                max(0.12, size[1] * 0.52),
                max(0.26, size[2] * 0.28),
            )
            if _add_robot_review_proxy_box(
                stage,
                f"{proxy_root_path}/torso",
                center=torso_center,
                size=torso_size,
                color=color_body,
            ):
                created += 1
                body_boxes.append(f"{proxy_root_path}/torso")
            pelvis_center = (
                center[0],
                center[1],
                bmin[2] + size[2] * 0.35,
            )
            pelvis_size = (
                max(0.16, size[0] * 0.46),
                max(0.13, size[1] * 0.58),
                max(0.12, size[2] * 0.10),
            )
            if _add_robot_review_proxy_box(
                stage,
                f"{proxy_root_path}/pelvis",
                center=pelvis_center,
                size=pelvis_size,
                color=color_body,
            ):
                created += 1
                body_boxes.append(f"{proxy_root_path}/pelvis")
            leg_z = bmin[2] + size[2] * 0.18
            leg_h = max(0.18, size[2] * 0.28)
            leg_dx = max(0.035, size[0] * 0.11)
            leg_w = max(0.045, size[0] * 0.12)
            leg_d = max(0.045, size[1] * 0.18)
            for idx, offset in enumerate((-leg_dx, leg_dx)):
                if _add_robot_review_proxy_box(
                    stage,
                    f"{proxy_root_path}/leg_{idx}",
                    center=(center[0] + offset, center[1], leg_z),
                    size=(leg_w, leg_d, leg_h),
                    color=color_body,
                ):
                    created += 1
                    body_boxes.append(f"{proxy_root_path}/leg_{idx}")
        except Exception:  # noqa: BLE001
            blockers.append("robot_review_body_proxy_failed")
    else:
        blockers.append("robot_bbox_unavailable_for_review_proxy")

    arm_radius = 0.035
    if bbox:
        try:
            arm_radius = max(0.025, min(0.05, float(bbox["size_xyz"][2]) * 0.026))
        except Exception:  # noqa: BLE001
            arm_radius = 0.035
    for side, points in sorted(side_points.items()):
        if not points:
            blockers.append(f"{side}_arm_link_points_unavailable_for_review_proxy")
            continue
        for a_role, b_role in (("shoulder", "elbow"), ("elbow", "wrist"), ("wrist", "hand")):
            a = points.get(a_role)
            b = points.get(b_role)
            if a is None or b is None:
                continue
            try:
                a3 = tuple(float(v) for v in a)
                b3 = tuple(float(v) for v in b)
                center = tuple((a3[i] + b3[i]) * 0.5 for i in range(3))
                size = tuple(abs(b3[i] - a3[i]) + arm_radius * 2.0 for i in range(3))
                path = f"{proxy_root_path}/{side}_{a_role}_to_{b_role}"
                if _add_robot_review_proxy_box(stage, path, center=center, size=size, color=color_arm):
                    created += 1
                    arm_boxes.append(path)
            except Exception:  # noqa: BLE001
                blockers.append(f"{side}_{a_role}_to_{b_role}_review_proxy_failed")
        hand = points.get("hand") or points.get("wrist")
        if hand is not None:
            try:
                hand3 = tuple(float(v) for v in hand)
                path = f"{proxy_root_path}/{side}_hand_block"
                if _add_robot_review_proxy_box(
                    stage,
                    path,
                    center=hand3,
                    size=(arm_radius * 2.4, arm_radius * 2.4, arm_radius * 2.0),
                    color=color_arm,
                ):
                    created += 1
                    arm_boxes.append(path)
            except Exception:  # noqa: BLE001
                blockers.append(f"{side}_hand_review_proxy_failed")

    if created == 0:
        blockers.append("robot_review_visual_proxy_not_created")
    return {
        "schema_version": "robot_review_visual_proxy.v1",
        "status": "PASS" if not blockers else "FAIL",
        "blockers": sorted(set(blockers)),
        "proxy_root_path": proxy_root_path,
        "created_gprim_count": created,
        "body_proxy_paths": body_boxes,
        "arm_proxy_paths": arm_boxes,
        "arm_link_roles_by_arm": {
            side: sorted(points)
            for side, points in sorted(side_points.items())
        },
        "source_robot_prim_path": robot_prim_path,
        "source_robot_bbox": bbox,
        "claim_boundary": (
            "Review visual proxies are generated from the current G1 robot bbox/link transforms only "
            "when the worker lacks renderable G1 meshes. They are render aids, not collision, contact, "
            "policy, physical-reach, or deployment proof."
        ),
    }


# ---------------- render-noise audit GPU/USD helpers ----------------

def _remove_prim_quiet(stage, path: str) -> None:
    try:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            stage.RemovePrim(path)
    except Exception:  # noqa: BLE001
        pass


_AUDIT_BASE_COLOR_INPUT_NAMES = (
    "diffuseColor",
    "diffuse_color_constant",
    "baseColor",
    "base_color",
    "diffuse_tint",
    "albedo",
    "albedo_desaturation",
)


def _collect_robot_material_resolution(
    stage,
    robot_prim_path: str,
    *,
    robot_asset_uri: str | None = None,
    resolved_visual_asset: str | None = None,
) -> dict[str, Any]:
    """Raw robot-subtree material/texture resolution evidence for the render-noise audit.

    Walks every Gprim under the robot, resolves its bound material, and records each shader's
    asset-valued inputs with their authored + resolved paths and on-disk existence. This is the
    evidence that separates 'textures are noisy' from 'textures never resolved on this worker'.
    Robot-scoped and site/task agnostic.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdShade  # type: ignore

    raw: dict[str, Any] = {
        "robot_prim_path": robot_prim_path,
        "robot_asset_uri": robot_asset_uri,
        "resolved_visual_asset": resolved_visual_asset,
        "gprim_count": 0,
        "mesh_count": 0,
        "gprims_without_material": 0,
        "materials": [],
    }
    robot = stage.GetPrimAtPath(robot_prim_path)
    if not (robot and robot.IsValid()):
        raw["error"] = "robot_prim_missing"
        return raw
    materials_by_path: dict[str, dict[str, Any]] = {}
    try:
        prim_iter = Usd.PrimRange(robot, Usd.TraverseInstanceProxies())
    except Exception:  # noqa: BLE001
        prim_iter = Usd.PrimRange(robot)
    for prim in prim_iter:
        try:
            if not prim.IsA(UsdGeom.Gprim):
                continue
            raw["gprim_count"] += 1
            if prim.IsA(UsdGeom.Mesh):
                raw["mesh_count"] += 1
            material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            material_prim = material.GetPrim() if material else None
            if not (material_prim and material_prim.IsValid()):
                raw["gprims_without_material"] += 1
                continue
            mat_path = str(material_prim.GetPath())
            if mat_path in materials_by_path:
                materials_by_path[mat_path]["bound_gprim_count"] += 1
                continue
            entry: dict[str, Any] = {
                "path": mat_path,
                "bound_gprim_count": 1,
                "shader_ids": [],
                "texture_refs": [],
            }
            for child in Usd.PrimRange(material_prim):
                shader = UsdShade.Shader(child)
                if not shader:
                    continue
                try:
                    impl_id = shader.GetIdAttr().Get()
                except Exception:  # noqa: BLE001
                    impl_id = None
                source_asset = None
                try:
                    source_asset = shader.GetSourceAsset("mdl")
                except Exception:  # noqa: BLE001
                    pass
                shader_id = str(impl_id or (source_asset.path if source_asset else "") or "")
                if shader_id and shader_id not in entry["shader_ids"]:
                    entry["shader_ids"].append(shader_id)
                for shader_input in shader.GetInputs():
                    try:
                        if shader_input.GetTypeName() != Sdf.ValueTypeNames.Asset:
                            continue
                        value = shader_input.Get()
                    except Exception:  # noqa: BLE001
                        continue
                    if value is None:
                        continue
                    authored = str(getattr(value, "path", "") or "")
                    resolved = str(getattr(value, "resolvedPath", "") or "")
                    if not authored and not resolved:
                        continue
                    if resolved:
                        exists = True if "://" in resolved else Path(resolved).is_file()
                    else:
                        exists = False
                    entry["texture_refs"].append({
                        "input": shader_input.GetBaseName(),
                        "shader_path": str(child.GetPath()),
                        "authored_path": authored,
                        "resolved_path": resolved or None,
                        "exists": bool(exists),
                    })
            materials_by_path[mat_path] = entry
        except Exception:  # noqa: BLE001
            continue
    raw["materials"] = list(materials_by_path.values())
    return raw


def _apply_robot_simplified_diffuse_material(stage, robot_prim_path: str) -> dict[str, Any]:
    """Bind flat UsdPreviewSurface materials sampled from each gprim's authored base color.

    Keeps the robot's per-part color identity while removing image textures and
    metallic/specular response, so a clean simplified-diffuse render vs a noisy full-PBR
    render points at the PBR/texture response rather than geometry or lighting.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # type: ignore

    robot = stage.GetPrimAtPath(robot_prim_path)
    result: dict[str, Any] = {
        "gprims_bound": 0,
        "distinct_colors": 0,
        "color_sources": {"shader_input": 0, "display_color": 0, "default": 0},
        "sampled_colors": [],
    }
    if not (robot and robot.IsValid()):
        result["error"] = "robot_prim_missing"
        return result

    def _sampled_base_color(prim) -> tuple[tuple[float, float, float], str]:
        try:
            material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            material_prim = material.GetPrim() if material else None
            if material_prim and material_prim.IsValid():
                for child in Usd.PrimRange(material_prim):
                    shader = UsdShade.Shader(child)
                    if not shader:
                        continue
                    for name in _AUDIT_BASE_COLOR_INPUT_NAMES:
                        shader_input = shader.GetInput(name)
                        if not shader_input:
                            continue
                        try:
                            value = shader_input.Get()
                        except Exception:  # noqa: BLE001
                            continue
                        if value is None:
                            continue
                        try:
                            rgb = (float(value[0]), float(value[1]), float(value[2]))
                        except Exception:  # noqa: BLE001
                            continue
                        return rgb, "shader_input"
        except Exception:  # noqa: BLE001
            pass
        try:
            display = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
            if display:
                rgb = (float(display[0][0]), float(display[0][1]), float(display[0][2]))
                return rgb, "display_color"
        except Exception:  # noqa: BLE001
            pass
        return (0.62, 0.63, 0.65), "default"

    materials_by_color: dict[tuple[float, float, float], Any] = {}
    try:
        prim_iter = Usd.PrimRange(robot, Usd.TraverseInstanceProxies())
    except Exception:  # noqa: BLE001
        prim_iter = Usd.PrimRange(robot)
    for prim in prim_iter:
        try:
            if not prim.IsA(UsdGeom.Gprim):
                continue
            rgb, source = _sampled_base_color(prim)
            key = tuple(round(max(0.0, min(1.0, c)), 3) for c in rgb)
            material = materials_by_color.get(key)
            if material is None:
                index = len(materials_by_color)
                mat_path = f"/World/Materials/RenderNoiseAuditSimplified_{index}"
                material = UsdShade.Material.Define(stage, mat_path)
                shader = UsdShade.Shader.Define(stage, f"{mat_path}/PreviewSurface")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*key))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.70)
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                materials_by_color[key] = material
                result["sampled_colors"].append({"rgb": list(key), "source": source})
            UsdShade.MaterialBindingAPI(prim).Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
            UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*key)])
            result["gprims_bound"] += 1
            result["color_sources"][source] += 1
        except Exception:  # noqa: BLE001
            continue
    result["distinct_colors"] = len(materials_by_color)
    return result


def _apply_audit_render_settings(
    rep,
    *,
    samples_per_pixel: int,
    denoiser_enabled: bool,
) -> dict[str, Any]:
    """Path-traced render settings for one audit variant; only spp + denoiser vary per variant.

    The firefly filter and texture-streaming settings are held constant across the whole audit
    so the denoiser and sample budget are the only changed render variables.
    """
    diagnostics: dict[str, Any] = {
        "renderer_mode_requested": "PathTracing",
        "samples_per_pixel": int(samples_per_pixel),
        "denoiser_enabled": bool(denoiser_enabled),
        "firefly_filter_enabled": True,
        "settings_applied": [],
        "setting_errors": [],
    }
    try:
        rep.settings.set_render_pathtraced(samples_per_pixel=int(samples_per_pixel))
        diagnostics["settings_applied"].append("rep.settings.set_render_pathtraced")
    except Exception as exc:  # noqa: BLE001
        diagnostics["setting_errors"].append({
            "setting": "rep.settings.set_render_pathtraced",
            "error": repr(exc),
        })
    try:
        import carb  # type: ignore

        settings = carb.settings.get_settings()
        for path, value in (
            ("/rtx/pathtracing/spp", int(samples_per_pixel)),
            ("/rtx/pathtracing/totalSpp", int(samples_per_pixel)),
            ("/rtx/pathtracing/optixDenoiser/enabled", bool(denoiser_enabled)),
            ("/rtx/pathtracing/optixDenoiser/blendFactor", 0.0),
            ("/rtx/pathtracing/fireflyFilter/enabled", True),
            ("/rtx/pathtracing/fireflyFilter/maxIntensityPerSample", 350.0),
            ("/rtx/pathtracing/fireflyFilter/maxIntensityPerSampleDiffuse", 350.0),
            ("/rtx-transient/resourcemanager/enableTextureStreaming", False),
        ):
            try:
                settings.set(path, value)
                diagnostics["settings_applied"].append(path)
            except Exception as exc:  # noqa: BLE001
                diagnostics["setting_errors"].append({"setting": path, "error": repr(exc)})
        effective: dict[str, Any] = {}
        for path in (
            "/rtx/rendermode",
            "/rtx/pathtracing/spp",
            "/rtx/pathtracing/totalSpp",
            "/rtx/pathtracing/optixDenoiser/enabled",
            "/rtx/pathtracing/optixDenoiser/blendFactor",
            "/rtx/pathtracing/fireflyFilter/enabled",
            "/rtx/post/tonemap/op",
            "/rtx/post/tonemap/filmIso",
            "/rtx/post/tonemap/cameraShutter",
            "/rtx/post/aa/op",
        ):
            try:
                effective[path] = settings.get(path)
            except Exception:  # noqa: BLE001
                effective[path] = None
        diagnostics["effective_settings"] = effective
    except Exception as exc:  # noqa: BLE001
        diagnostics["setting_errors"].append({"setting": "carb.settings", "error": repr(exc)})
    diagnostics["status"] = "PASS" if not diagnostics["setting_errors"] else "WARN"
    return diagnostics


def _scene_lighting_summary(stage, *, max_lights: int = 64) -> dict[str, Any]:
    """Dome/key/fill light inventory (type, intensity, color, temperature, HDRI texture)."""
    lights: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}
    for prim in stage.Traverse():
        type_name = str(prim.GetTypeName() or "")
        if not type_name.endswith("Light"):
            continue
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
        if len(lights) >= max_lights:
            continue
        entry: dict[str, Any] = {"path": str(prim.GetPath()), "type": type_name}
        for label, names in (
            ("intensity", ("inputs:intensity", "intensity")),
            ("exposure", ("inputs:exposure", "exposure")),
            ("color", ("inputs:color", "color")),
            ("color_temperature", ("inputs:colorTemperature", "colorTemperature")),
            ("enable_color_temperature", ("inputs:enableColorTemperature", "enableColorTemperature")),
            ("texture_file", ("inputs:texture:file", "texture:file")),
        ):
            for name in names:
                try:
                    attr = prim.GetAttribute(name)
                    if attr and attr.IsValid() and attr.HasAuthoredValue():
                        value = attr.Get()
                        if label == "color" and value is not None:
                            entry[label] = [round(float(v), 4) for v in value]
                        elif label == "texture_file":
                            entry[label] = str(getattr(value, "path", value) or "") or None
                        elif value is not None:
                            entry[label] = round(float(value), 4) if isinstance(value, (int, float)) else value
                        break
                except Exception:  # noqa: BLE001
                    continue
        lights.append(entry)
    return {
        "schema_version": "isaac_scene_lighting_summary.v1",
        "light_count": sum(type_counts.values()),
        "type_counts": dict(sorted(type_counts.items())),
        "lights": lights,
    }


def _audit_runtime_metadata() -> dict[str, Any]:
    """Best-effort GPU/driver/Isaac/image identity so noisy output can be tied to a
    specific worker image + GPU + driver combination."""
    meta: dict[str, Any] = {}
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        line = (query.stdout or "").strip().splitlines()
        if query.returncode == 0 and line:
            parts = [p.strip() for p in line[0].split(",")]
            meta["gpu_name"] = parts[0] if parts else None
            meta["driver_version"] = parts[1] if len(parts) > 1 else None
            meta["gpu_memory_total"] = parts[2] if len(parts) > 2 else None
    except Exception as exc:  # noqa: BLE001
        meta["nvidia_smi_error"] = repr(exc)
    for candidate in ("/isaac-sim/VERSION", "/isaac-sim/version.txt"):
        try:
            version_path = Path(candidate)
            if version_path.is_file():
                meta["isaac_version"] = version_path.read_text(encoding="utf-8").strip()[:120]
                break
        except Exception:  # noqa: BLE001
            continue
    for env_key in (
        "ISAACSIM_VERSION", "ISAAC_SIM_VERSION",
        "BLUEPRINT_WORKER_IMAGE_REF", "BLUEPRINT_WORKER_IMAGE_DIGEST",
        "RUNPOD_POD_ID", "CUDA_VERSION", "NVIDIA_DRIVER_CAPABILITIES",
    ):
        value = os.getenv(env_key, "").strip()
        if value:
            meta[env_key.lower()] = value
    meta["python_version"] = sys.version.split()[0]
    return meta


def _robot_pixel_ratio_from_instance(instance_annot, robot_prim_path: str) -> dict[str, Any]:
    """Fraction of frame pixels belonging to the robot subtree via the instance annotator.

    Colorized instance output maps '(r, g, b, a)' color keys to prim paths in
    info['idToLabels']; plain output maps integer ids. Both are handled best-effort.
    """
    import numpy as np  # type: ignore

    try:
        data = instance_annot.get_data()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": repr(exc)}
    payload = _segmentation_payload(data)
    info = data.get("info") if isinstance(data, Mapping) else {}
    id_to_labels = (info or {}).get("idToLabels") or {}
    if payload is None or getattr(payload, "size", 0) == 0 or not id_to_labels:
        return {"available": False, "error": "instance_annotator_payload_or_labels_missing"}
    arr = np.asarray(payload)
    total = float(arr.shape[0] * arr.shape[1]) if arr.ndim >= 2 else float(arr.size)
    if total <= 0:
        return {"available": False, "error": "instance_annotator_empty_frame"}
    robot_root = str(robot_prim_path).rstrip("/")

    def _is_robot(label: Any) -> bool:
        text = str(label if not isinstance(label, Mapping) else (label.get("class") or label))
        return text.startswith(robot_root)

    robot_pixels = 0
    matched_ids = 0
    try:
        for key, label in id_to_labels.items():
            if not _is_robot(label):
                continue
            matched_ids += 1
            key_text = str(key).strip()
            if key_text.startswith("("):
                rgba = [int(v) for v in key_text.strip("()").split(",")[:4]]
                if arr.ndim == 3 and arr.shape[2] >= len(rgba):
                    mask = np.all(arr[:, :, : len(rgba)] == np.asarray(rgba, dtype=arr.dtype), axis=2)
                    robot_pixels += int(mask.sum())
            else:
                robot_pixels += int((arr == int(key_text)).sum())
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": repr(exc)}
    return {
        "available": True,
        "robot_pixel_ratio": round(robot_pixels / total, 6),
        "matched_instance_ids": matched_ids,
        "claim_boundary": (
            "Instance-mask pixel coverage of the robot subtree in this frame. It is render "
            "visibility evidence only, not manipulation or task evidence."
        ),
    }


def _add_workspace_fill_light(stage, target, *, intensity: float, height: float = 2.0,
                              path: str = "/World/WorkspaceFill") -> None:
    """Add a local sphere fill light above the manipulation workspace so the task surface and seeded
    arms are lit. Intensity is configurable (blind-tunable via re-render). GPU/USD."""
    from pxr import UsdLux, UsdGeom, Gf  # type: ignore
    light = UsdLux.SphereLight.Define(stage, path)
    light.CreateIntensityAttr(float(intensity))
    light.CreateRadiusAttr(0.4)
    # Idempotent on a reused (warm --serve) stage: reuse the existing translate op instead of adding a
    # second one (AddTranslateOp raises "xformOp:translate already exists" on the 2nd+ job otherwise).
    xf = UsdGeom.Xformable(light.GetPrim())
    translate_op = next(
        (op for op in xf.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
        None,
    )
    if translate_op is None:
        translate_op = xf.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(target[0]), float(target[1]) - 0.25, float(height)))


def _add_pov_headlamp(stage, eye, look_at, *, intensity: float = 20000.0,
                      path: str = "/World/PovHeadlamp") -> None:
    """Camera-side fill light for the manipulation POV so the REACHING ARM + gripper are front-lit.

    The workspace fill light sits at the door/affordance, BEYOND the arm from the head-mounted camera,
    so the camera otherwise sees only the arm's shadow side (it renders black). This places a SOFT fill
    in front of the camera eye, aimed toward the workspace, lighting the arms+grippers the camera sees.

    It MUST be soft: a small, very bright sphere this close to the arm is a path-tracing firefly source
    (salt-and-pepper grain that the denoiser can't recover). So use a LARGE radius + a CAPPED intensity
    + a little more standoff -> even fill, no fireflies. Idempotent on a reused (warm --serve) stage."""
    from pxr import UsdLux, UsdGeom, Gf  # type: ignore
    fx, fy, fz = (float(look_at[0]) - float(eye[0]),
                  float(look_at[1]) - float(eye[1]),
                  float(look_at[2]) - float(eye[2]))
    length = math.sqrt(fx * fx + fy * fy + fz * fz) or 1e-6
    pos = (float(eye[0]) + fx / length * 0.30,
           float(eye[1]) + fy / length * 0.30,
           float(eye[2]) + fz / length * 0.30 + 0.10)  # 30cm toward the workspace, a touch above
    light = UsdLux.SphereLight.Define(stage, path)
    # Cap intensity (callers pass the bright 30000 workspace value) and use a large soft radius so the
    # close camera-side fill does not create fireflies on the nearby arm.
    light.CreateIntensityAttr().Set(min(float(intensity), 6000.0))
    light.CreateRadiusAttr().Set(0.5)
    xf = UsdGeom.Xformable(light.GetPrim())
    translate_op = next(
        (op for op in xf.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
        None,
    )
    if translate_op is None:
        translate_op = xf.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*pos))


def _neutralize_environment(stage, *, intensity: float = 1500.0) -> int:
    """Replace any outdoor-HDRI DomeLight in the loaded scene with a NEUTRAL uniform environment.

    The Lightwheel kitchen ships a DomeLight (e.g. ``DomeLight_01`` with ``texture/chive 7000x3500.hdr``)
    that projects an outdoor cityscape, visible through the kitchen windows — an incongruous background
    for an enclosed manipulation scene. Clearing the HDRI texture + setting a neutral bright color turns
    it into even ambient: windows read neutral (no city) AND the dark cabinet/basin surfaces get lifted
    by global fill. Returns the number of dome lights neutralized. GPU/USD only."""
    from pxr import UsdLux, Sdf, Gf  # type: ignore
    n = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "DomeLight":
            continue
        dome = UsdLux.DomeLight(prim)
        for attr_name in ("inputs:texture:file", "texture:file"):
            attr = prim.GetAttribute(attr_name)
            if attr and attr.IsValid():
                try:
                    attr.Set(Sdf.AssetPath(""))  # drop the HDRI -> uniform dome (no cityscape)
                except Exception:  # noqa: BLE001
                    pass
        try:
            dome.CreateColorAttr().Set(Gf.Vec3f(0.92, 0.92, 0.95))  # neutral cool-white
            dome.CreateIntensityAttr(float(intensity))
        except Exception:  # noqa: BLE001
            pass
        n += 1
    return n


def run_scenarios(*, kitchen_usd: str, g1_usd: str, scenarios: Sequence[dict], out_dir: Path,
                  policy_id: str, steps: int, width: int, height: int, fps: int,
                  warmup_frames: int, capture_every: int, no_collision_probe: bool = False,
                  per_scenario_seconds: int = 480, focus_radius: float = 0.0,
                  keep_substrings: Sequence[str] = ("room", "floor", "wall", "ground", "ceiling", "light"),
                  disable_physx: bool = False, settle_seconds: int = 0,
                  cheap_collision: bool = False, articulated: bool = False,
                  camera_vfov_deg: float = 50.0, manipulation_cam: bool = False,
                  manipulation_look_at=None, render_subframes: int = 1,
                  manipulation_reach: bool = False, manipulation_reach_arm: str = "both",
                  fill_light_intensity: float = 0.0,
                  physics_articulation_drive: bool = False,
                  dynamic_standing_contact_steps: int = 0,
                  neutral_environment: bool = False,
                  robot_review_material_override: bool = False,
                  robot_review_material_mode: str = "neutral_matte",
                  kinematic_arm_pose: bool = False,
                  collision_approximation: str = "boundingCube",
                  verify_cam: bool = False,
                  manipulation_stand: bool = False,
                  software_denoise: bool = True,
                  depth_pass: bool = False,
                  segmentation: bool = False,
                  effort_drive: bool = False,
                  torque_drive: bool = False,
                  author_target_contact_material: bool = False,
                  groot_policy_command: str = "",
                  groot_policy_command_timeout_seconds: float = 120.0,
                  groot_policy_initial_frame: str = "",
                  serve: bool = False, serve_dir: "Path | None" = None,
                  serve_idle_timeout_s: float = 600.0,
                  serve_max_jobs: "int | None" = None) -> dict:
    """GPU orchestration: boot Isaac, load scene + G1, run the controller per scenario with RTX
    render + (optional) PhysX collision probe, emit traces + MP4s + outcomes. Instrumented with
    flushed progress + a per-scenario wall-clock cap so it cannot hang silently.

    Warm mode (``serve=True``): after the one-time setup (Isaac boot, scene + G1 load, cameras,
    settle), keep the process alive and render a STREAM of task scenarios pulled from ``serve_dir``
    via :func:`blueprint_pipeline.warm_render_server.serve_render_loop`, so each rerun skips image
    pull + Isaac boot + stage load + most settle. Single-shot behavior (``serve=False``) is unchanged."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _log("booting Isaac (headless RTX) ...")
    sim = _boot_sim(headless=True)
    _log("Isaac booted; enabling Replicator")
    rep = _enable_and_import_replicator()  # after boot: omni.* now importable + extension enabled
    _log("Replicator ready")
    render_quality_diag = _apply_render_quality_settings(
        rep,
        render_subframes=int(render_subframes),
        manipulation_cam=bool(manipulation_cam),
        verify_cam=bool(verify_cam),
        out_dir=out_dir,
    )
    if render_quality_diag.get("use_pathtraced"):
        _log(
            "path-traced render quality enabled: "
            f"spp={render_quality_diag.get('samples_per_pixel')} "
            f"rt_subframes={render_quality_diag.get('render_subframes')}"
        )
    capture_rt_subframes = _effective_render_rt_subframes(
        int(render_subframes),
        render_quality_diag,
    )
    capture_settle_steps = _capture_settle_steps(render_quality_diag)
    software_denoise = _effective_software_denoise(software_denoise, render_quality_diag)
    render_quality_diag["requested_replicator_rt_subframes"] = max(1, int(render_subframes))
    render_quality_diag["effective_replicator_rt_subframes"] = int(capture_rt_subframes)
    render_quality_diag["capture_settle_steps"] = int(capture_settle_steps)
    render_quality_diag["software_denoise_applied_to_saves"] = bool(software_denoise)
    try:
        (out_dir / "render_quality_settings.json").write_text(
            json.dumps(render_quality_diag, indent=2),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        pass
    if capture_rt_subframes != max(1, int(render_subframes)):
        _log(
            "effective Replicator rt_subframes reduced for path-traced capture: "
            f"{capture_rt_subframes}"
        )
    if disable_physx:
        # NOTE: confirmed on GPU to break the RTX renderer (hangs at render-product creation) —
        # kept only for experiments. Keep PhysX on and use settle_seconds instead.
        _disable_physics_cooking()
        _log("PhysX cooking disabled (WARNING: breaks the renderer on this image)")
    blockers: list[str] = []
    outcomes: list[dict] = []
    physics_contact_reports: list[dict[str, Any]] = []
    effective_effort_drive = bool(effort_drive or torque_drive)
    authored_target_contact_material_records: list[dict[str, Any]] = []
    authored_target_contact_material_paths: set[str] = set()
    segmentation_summary: dict[str, Any] = {
        "schema_version": "isaac_g1_kitchen_parity_segmentation_summary.v1",
        "enabled": bool(segmentation),
        "labeled_prim_count": 0,
        "instance_mask_frames": 0,
        "semantic_mask_frames": 0,
        "id_label_path": None,
        "sample_labels": [],
        "blockers": [],
    }
    result = None
    if dynamic_standing_contact_steps > 0:
        articulated = True
        physics_articulation_drive = True
        if int(steps) != 1:
            blockers.append("physics_articulation_dynamic_standing_contact_requires_single_step")
        if len(scenarios) != 1:
            blockers.append("physics_articulation_dynamic_standing_contact_requires_single_scenario")
    if effective_effort_drive and not physics_articulation_drive:
        blockers.append("effort_drive_requires_physics_articulation_drive")
    if author_target_contact_material and not physics_articulation_drive:
        blockers.append("target_contact_material_requires_physics_articulation_drive")
    try:
        kitchen_usd, kitchen_resolution = _resolve_existing_kitchen_usd(kitchen_usd)
        try:
            (out_dir / "kitchen_usd_resolution.json").write_text(
                json.dumps(kitchen_resolution, indent=2),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass
        _log(f"opening kitchen USD: {kitchen_usd}")
        stage = _open_stage(_resolve_asset_uri(kitchen_usd))
        if cheap_collision:
            nc = _force_cheap_collision(stage, approximation=collision_approximation)
            _log(f"forced {collision_approximation} collision on {nc} mesh-collision prims")
        _log("kitchen stage open; binding G1 articulation")
        binding = _bind_g1_with_visual_fallback(stage, g1_usd)
        _log(f"G1 binding: articulation={binding['controllable_articulation_detected']} "
             f"collision={binding['collision_enabled_verified']} "
             f"visual={binding.get('visual_binding_status')}")
        (out_dir / "g1_binding.json").write_text(json.dumps(binding, indent=2))
        if not binding["controllable_articulation_detected"]:
            blockers.append("official_isaac_unitree_g1_articulation_api_unverified")
        robot_render_diag: dict[str, Any] | None = dict(
            binding.get("robot_render_diagnostics") or {}
        )
        robot_visual_missing = _robot_visual_geometry_missing(robot_render_diag)
        if robot_visual_missing and (manipulation_cam or verify_cam):
            if ROBOT_VISUAL_MESH_MISSING_BLOCKER not in blockers:
                blockers.append(ROBOT_VISUAL_MESH_MISSING_BLOCKER)
            _log(
                "G1 visual mesh/Gprim readiness FAILED after candidate load attempts; "
                "link projections will be treated as geometry-only evidence"
            )
        if segmentation:
            label_summary = _author_scene_semantic_labels(
                stage,
                robot_prim_path=binding.get("prim_path"),
                keep_substrings=keep_substrings,
            )
            segmentation_summary["labeled_prim_count"] = int(
                label_summary.get("labeled_prim_count") or 0
            )
            segmentation_summary["sample_labels"] = list(label_summary.get("sample_labels") or [])
            segmentation_summary["blockers"] = list(label_summary.get("blockers") or [])
            (out_dir / "segmentation_semantic_label_authoring.json").write_text(
                json.dumps(label_summary, indent=2)
            )
        robot_neutral_xforms: dict[str, Any] = {}
        if kinematic_arm_pose or manipulation_reach:
            robot_neutral_xforms = _capture_robot_neutral_descendant_xforms(
                stage,
                binding["prim_path"],
            )
            _log(f"G1 neutral descendant xforms captured: {len(robot_neutral_xforms)} prim(s)")
        has_deferred_task_targets = any(sc.get("task_target_deferred") for sc in scenarios)
        if focus_radius > 0 and has_deferred_task_targets:
            _log(
                "focus prune skipped: one or more task-only scenarios need scene-placement "
                "target resolution before route points exist"
            )
            (out_dir / "focus_prune.json").write_text(json.dumps({
                "status": "skipped",
                "reason": "deferred_task_target_requires_scene_resolution",
            }, indent=2))
        elif focus_radius > 0:
            route_pts = [p for sc in scenarios for p in sc.get("route_points", [])]
            pr = _prune_to_focus(stage, route_pts, focus_radius, keep_substrings)
            _log(f"focus prune (r={focus_radius}m): kept {pr['kept']} objects, deactivated {pr['pruned']}")
            (out_dir / "focus_prune.json").write_text(json.dumps(pr, indent=2))
        placement_scene_objects = _scene_objects_for_stage(stage)
        (out_dir / "placement_scene_objects.json").write_text(
            json.dumps([_scene_object_to_dict(obj) for obj in placement_scene_objects], indent=2)
        )
        _log(f"placement scene object catalog: {len(placement_scene_objects)} object(s)")
        rest_offsets = None
        art_ctx = None
        dynamic_seed_decisions: dict[str, Any] = {}
        preplanned_task_stance_plans: dict[str, dict[str, Any]] = {}
        if articulated:
            rest_offsets = _g1_link_rest_offsets(stage, binding["prim_path"])
            if dynamic_standing_contact_steps > 0 and scenarios:
                if no_collision_probe:
                    def seed_probe(pose, yaw):  # noqa: ANN001
                        return 0
                else:
                    seed_probe = _overlap_probe(binding["prim_path"])
                if manipulation_stand:
                    seed_sid = scenarios[0]["scenario_id"]
                    seed_stance_plan = _plan_task_stance_for_stage(
                        stage=stage,
                        scenario=scenarios[0],
                        manipulation_look_at=manipulation_look_at,
                        probe=seed_probe,
                        no_collision_probe=no_collision_probe,
                        robot_prim_path=binding["prim_path"],
                    )
                    preplanned_task_stance_plans[seed_sid] = seed_stance_plan
                    (out_dir / "dynamic_task_stance_seed_plan.json").write_text(
                        json.dumps(seed_stance_plan, indent=2)
                    )
                    if seed_stance_plan.get("status") == "accepted":
                        seed_root = tuple(float(v) for v in seed_stance_plan["accepted_pose"])
                        seed_yaw = float(seed_stance_plan["accepted_yaw"])
                        dynamic_seed_decisions[seed_sid] = {
                            "source": "task_stance_plan",
                            "root_pose": seed_root,
                            "yaw": seed_yaw,
                        }
                        _place_root(stage, binding["prim_path"], seed_root, seed_yaw)
                        _log(
                            "dynamic standing/contact root seeded from task stance plan before "
                            f"articulation tensor view: pose={seed_root}, yaw={seed_yaw:.4f}"
                        )
                    else:
                        blockers.append("task_stance_plan_failed")
                        _log(
                            "dynamic standing/contact task stance seed blocked "
                            f"{seed_stance_plan.get('blockers') or seed_stance_plan.get('status')}"
                        )
                else:
                    seed_policy = policy_mod.make_policy(policy_id)
                    seed_policy.reset(scenarios[0])
                    seed_decision = seed_policy.step(
                        policy_mod.StepContext(
                            step=0,
                            num_steps=steps,
                            probe_collision=seed_probe,
                        )
                    )
                    dynamic_seed_decisions[scenarios[0]["scenario_id"]] = seed_decision
                    _place_root(stage, binding["prim_path"], seed_decision.root_pose, seed_decision.yaw)
                    _log(
                        "dynamic standing/contact root seeded before articulation tensor view: "
                        f"pose={seed_decision.root_pose}, yaw={seed_decision.yaw:.4f}"
                    )
            # The physics articulation drive is OPT-IN and default-OFF. Dynamic standing/contact
            # proof intentionally uses a plain SimulationContext with no SingleArticulation tensor
            # view because the official G1 USD invalidates that tensor view during joint drive/read.
            if physics_articulation_drive:
                try:
                    gravity_z = -9.81 if dynamic_standing_contact_steps > 0 else 0.0
                    if dynamic_standing_contact_steps > 0:
                        art_ctx = _setup_physics_context_only(gravity_z=gravity_z)
                        _log(
                            "G1 USD PhysX articulation settle ready: "
                            f"tensor_view_used={art_ctx['tensor_view_used']}, gravity_z={gravity_z}"
                        )
                    else:
                        art_ctx = _setup_articulated_g1(binding["prim_path"], gravity_z=gravity_z)
                        _log(
                            "G1 articulation drive ready: "
                            f"{art_ctx['dof_count']} joints, {len(art_ctx['link_names'])} links, "
                            f"gravity_z={gravity_z}"
                        )
                except Exception as exc:  # noqa: BLE001
                    blockers.append("official_isaac_unitree_g1_joint_drive_unavailable")
                    _log(f"G1 articulation drive unavailable ({exc!r}); using USD skeleton fallback")
                if art_ctx is not None and not art_ctx.get("link_names"):
                    _log("G1 articulation body/link names unavailable; using USD skeleton fallback for landmarks")
            _log(f"G1 skeleton (USD rest offsets): {len(rest_offsets)} link landmarks")
        from pxr import UsdGeom, UsdLux  # type: ignore
        UsdGeom.Scope.Define(stage, "/World")
        over_cam = "/World/Cameras/overview"
        pov_cam = "/World/Cameras/robot_pov"
        verify_cam_path = "/World/Cameras/verify"
        topdown_cam_path = "/World/Cameras/placement_topdown"
        UsdGeom.Camera.Define(stage, over_cam)
        UsdGeom.Camera.Define(stage, pov_cam)
        if verify_cam:
            UsdGeom.Camera.Define(stage, verify_cam_path)
        if manipulation_stand:
            UsdGeom.Camera.Define(stage, topdown_cam_path)
        key = UsdLux.DistantLight.Define(stage, "/World/Key")
        try:
            key.CreateIntensityAttr(3000.0)  # lift the global key so the workspace is not crushed dark
        except Exception:  # noqa: BLE001
            pass
        # POV camera: widen from USD's ~17deg telephoto default to the projection FOV so the frame
        # shows the forward task workspace and visible arms instead of a tight near-field crop, while
        # keeping the rendered view aligned with skeleton projection. Overview frames the whole scene.
        pov_vfov_deg = (
            max(float(camera_vfov_deg), float(MANIPULATION_POV_MIN_VFOV_DEG))
            if manipulation_cam
            else float(camera_vfov_deg)
        )
        _set_camera_fov(stage, pov_cam, pov_vfov_deg, width, height)
        _set_camera_fov(stage, over_cam, 60.0, width, height)
        if verify_cam:
            _set_camera_fov(stage, verify_cam_path, 70.0, width, height)
        if manipulation_cam and fill_light_intensity > 0 and manipulation_look_at is not None:
            _add_workspace_fill_light(stage, manipulation_look_at, intensity=fill_light_intensity)
            _log(f"workspace fill light @ {tuple(round(float(c),2) for c in manipulation_look_at)} "
                 f"intensity={fill_light_intensity}")
        if neutral_environment:
            try:
                n_dome = _neutralize_environment(stage)
                _log(f"neutralized {n_dome} outdoor-HDRI dome light(s) -> enclosed neutral environment")
            except Exception as exc:  # noqa: BLE001
                _log(f"environment neutralize skipped ({exc!r})")
        if manipulation_cam or verify_cam:
            try:
                force_review_material = (
                    bool(robot_review_material_override)
                    or os.getenv("PARITY_ROBOT_REVIEW_MATERIAL_OVERRIDE", "") == "1"
                )
                material_mode = (
                    os.getenv("PARITY_ROBOT_REVIEW_MATERIAL_MODE", "").strip()
                    or robot_review_material_mode
                    or "neutral_matte"
                )
                material_spec = _robot_review_material_spec(material_mode)
                override_robot_material = bool(robot_visual_missing or force_review_material)
                n_robot_mat = _apply_robot_review_material(
                    stage,
                    binding["prim_path"],
                    override_authored_materials=override_robot_material,
                    material_mode=material_mode,
                )
                if override_robot_material:
                    _log(f"robot review material bound to {n_robot_mat} G1 geometry prim(s)")
                else:
                    _log("robot authored materials preserved; visibility/purpose normalized")
                robot_render_diag = _robot_render_visibility_diagnostics(stage, binding["prim_path"])
                robot_render_diag["review_material_bound_count"] = int(n_robot_mat)
                robot_render_diag["review_material_override_applied"] = bool(override_robot_material)
                robot_render_diag["authored_robot_materials_preserved"] = not bool(override_robot_material)
                robot_render_diag["review_material_mode"] = str(material_mode)
                robot_render_diag["review_material_label"] = str(material_spec["label"])
                robot_render_diag["review_material_diffuse_color"] = [
                    round(float(v), 6) for v in material_spec["diffuse_color"]
                ]
                robot_render_diag["review_material_non_white"] = bool(
                    override_robot_material
                    and max(float(v) for v in material_spec["diffuse_color"]) < 0.75
                )
                robot_render_diag["visual_binding_status"] = binding.get("visual_binding_status")
                robot_render_diag["visual_candidate_attempts"] = binding.get(
                    "visual_candidate_attempts", []
                )
                robot_visual_missing = _robot_visual_geometry_missing(robot_render_diag)
                (out_dir / "robot_render_diagnostics.json").write_text(
                    json.dumps(robot_render_diag, indent=2)
                )
            except Exception as exc:  # noqa: BLE001
                _log(f"robot review material skipped ({exc!r})")
        _log(f"creating render products ({width}x{height})")
        over_depth_annot = None
        pov_depth_annot = None
        pov_seg_annots = None
        if depth_pass:
            over_annot, over_depth_annot = _make_render_product(
                over_cam,
                width,
                height,
                with_depth=True,
            )
            if segmentation:
                pov_seg_annots = _make_render_product(
                    pov_cam,
                    width,
                    height,
                    with_depth=True,
                    with_segmentation=True,
                )
                pov_annot = pov_seg_annots["rgb"]
                pov_depth_annot = pov_seg_annots["depth"]
            else:
                pov_annot, pov_depth_annot = _make_render_product(
                    pov_cam,
                    width,
                    height,
                    with_depth=True,
                )
        else:
            over_annot = _make_render_product(over_cam, width, height)
            if segmentation:
                pov_seg_annots = _make_render_product(
                    pov_cam,
                    width,
                    height,
                    with_segmentation=True,
                )
                pov_annot = pov_seg_annots["rgb"]
            else:
                pov_annot = _make_render_product(pov_cam, width, height)
        verify_annot = _make_render_product(verify_cam_path, width, height) if verify_cam else None
        topdown_annot = None
        topdown_enabled = bool(manipulation_stand)
        if software_denoise:
            _log("software PNG denoise enabled for saved render frames")
        center, radius = scene_framing(scenarios)
        _place_camera(stage, over_cam,
                      (center[0] + radius * 1.4, center[1] - radius * 1.4, center[2] + radius * 1.1),
                      center)
        _log("render products + overview camera ready")
        if no_collision_probe:
            _log("collision probe DISABLED (policy goes direct every step)")
            def probe(pose, yaw):  # noqa: ANN001
                return 0
        else:
            probe = _overlap_probe(binding["prim_path"])
        render_settle_seconds = _auto_render_settle_seconds(
            configured_settle_seconds=float(settle_seconds),
            no_collision_probe=bool(no_collision_probe),
            manipulation_cam=bool(manipulation_cam),
            verify_cam=bool(verify_cam),
            manipulation_stand=bool(manipulation_stand),
            warmup_frames=int(warmup_frames),
            render_subframes=int(render_subframes),
        )
        render_settle_done = False
        if render_settle_seconds > 0:
            _log(
                f"post-placement render settle configured for {render_settle_seconds:g}s "
                "(after task collision probes, before repeated RTX steps)"
            )

        def _settle_after_task_probe_if_needed(scenario_id: str) -> None:
            nonlocal render_settle_done
            if render_settle_done or render_settle_seconds <= 0:
                return
            # Let PhysX finish async collision-cooking AFTER task stance/probe queries. Rendering
            # during that window can hang second-and-later RTX steps on some worker nodes.
            _log(
                f"scenario {scenario_id}: settling {render_settle_seconds:g}s after task "
                "placement probes before RTX stepping"
            )
            t_settle = time.time()
            interval = 15.0 if render_settle_seconds >= 15 else max(1.0, render_settle_seconds)
            while time.time() - t_settle < render_settle_seconds:
                time.sleep(min(interval, max(0.0, render_settle_seconds - (time.time() - t_settle))))
                _log(
                    f"  scenario {scenario_id}: settle "
                    f"{int(min(time.time() - t_settle, render_settle_seconds))}/"
                    f"{int(render_settle_seconds)}s"
                )
            render_settle_done = True
            _log(f"scenario {scenario_id}: render settle complete; starting RTX capture")

        def _render_scenario(sc):
            nonlocal over_annot, pov_annot, verify_annot, topdown_annot
            sid = sc["scenario_id"]
            sdir = out_dir / sid
            (sdir / "frames").mkdir(parents=True, exist_ok=True)
            camera_contract_rows: list[dict[str, Any]] = []
            depth_frames_written = 0
            segmentation_instance_frames = 0
            segmentation_semantic_frames = 0
            segmentation_blockers: list[str] = list(segmentation_summary.get("blockers") or [])
            segmentation_id_label_path = sdir / "frames" / "segmentation_id_labels.json"
            stand_root = stand_yaw = None
            stance_plan = None
            scene_objects_for_validation: list[Any] = []
            placement_validation_manifest: dict[str, Any] | None = None
            placement_validation_path = sdir / "placement_validation.json"
            placement_validation_blocker_recorded = False
            placement_topdown_frame_path: Path | None = None
            placement_topdown_layout_frame_path: Path | None = None
            pov_reach_arm = _normalize_reach_arm_selection(manipulation_reach_arm)
            rendered_reach_arm = pov_reach_arm
            pov_geometry_path = sdir / "manipulation_pov_geometry.json"
            pov_geometry_records: list[dict[str, Any]] = []
            pov_geometry_report: dict[str, Any] | None = None
            pov_geometry_blocker_recorded = False
            last_root_diagnostics: dict[str, Any] | None = None
            stance_plan_path = sdir / "task_stance_plan.json"
            scenario_robot_render_diag = dict(robot_render_diag or {})
            pov_camera_mode = "robot_mounted_manipulation" if manipulation_cam else "root_follow"
            pov_camera_label = (
                "robot-mounted manipulation POV"
                if manipulation_cam
                else "root-follow camera saved under the legacy robot_pov filename"
            )
            def _write_pov_geometry_report() -> dict[str, Any] | None:
                if not pov_geometry_records:
                    return None
                geom_blockers = sorted({
                    str(blocker)
                    for record in pov_geometry_records
                    for blocker in (record.get("blockers") or [])
                })
                report = {
                    "schema_version": "manipulation_pov_geometry_index.v1",
                    "status": "PASS" if not geom_blockers else "FAIL",
                    "blockers": geom_blockers,
                    "scenario_id": sid,
                    "frames_checked": len(pov_geometry_records),
                    "frames": pov_geometry_records,
                    "claim_boundary": (
                        "Frame geometry proves task-affordance and USD arm-link visibility in the "
                        "render camera. It is not manipulation success or physical robot validation."
                    ),
                }
                pov_geometry_path.write_text(json.dumps(report, indent=2))
                return report

            def _append_camera_contract_row(
                contract: Mapping[str, Any],
                frame_path: Path,
                frame_index: int,
                *,
                camera_role: str,
                camera_mode: str,
                extra: Mapping[str, Any] | None = None,
            ) -> None:
                row = dict(contract)
                row.update({
                    "frame_index": int(frame_index),
                    "rgb_frame_path": str(frame_path),
                    "scenario_id": sid,
                    "camera_role": camera_role,
                    "camera_mode": camera_mode,
                })
                if extra:
                    row.update(dict(extra))
                camera_contract_rows.append(row)

            if manipulation_stand:
                stance_plan = preplanned_task_stance_plans.get(sid)
                if stance_plan is None:
                    stance_plan = _plan_task_stance_for_stage(
                        stage=stage,
                        scenario=sc,
                        manipulation_look_at=manipulation_look_at,
                        probe=probe,
                        no_collision_probe=no_collision_probe,
                        robot_prim_path=binding["prim_path"],
                    )
                stance_plan_path.write_text(json.dumps(stance_plan, indent=2))
                if stance_plan.get("status") == "accepted":
                    stand_root = tuple(float(v) for v in stance_plan["accepted_pose"])
                    stand_yaw = float(stance_plan["accepted_yaw"])
                    validation_target_obj = _target_object_from_stance_plan(stance_plan)
                    if validation_target_obj is not None:
                        t_obs = time.time()
                        scene_objects_for_validation = _placement_obstacles_for_stage(
                            stage,
                            focus_bounds=(validation_target_obj.bbox_min, validation_target_obj.bbox_max),
                        )
                        _log(
                            f"scenario {sid}: placement validation obstacles "
                            f"{len(scene_objects_for_validation)} object(s) in "
                            f"{time.time() - t_obs:.1f}s"
                        )
                    _log(
                        f"scenario {sid}: task stance accepted -> "
                        f"{tuple(round(float(c), 2) for c in stand_root)} yaw={stand_yaw:.2f} "
                        f"after {len(stance_plan.get('candidates', []))} candidate probe(s)"
                    )
                    if physics_articulation_drive and author_target_contact_material:
                        selected = (
                            ((stance_plan.get("target_resolution") or {}).get("selected") or {})
                            if isinstance(stance_plan.get("target_resolution"), Mapping)
                            else {}
                        )
                        target_prim_path = str(selected.get("prim_path") or "")
                        if target_prim_path and target_prim_path not in authored_target_contact_material_paths:
                            material_diag = _author_target_contact_material(
                                stage,
                                target_prim_path,
                                friction=0.85,
                                restitution=0.02,
                                mass=2.0,
                                density=None,
                            )
                            material_diag["scenario_id"] = sid
                            authored_target_contact_material_records.append(material_diag)
                            authored_target_contact_material_paths.add(target_prim_path)
                            (sdir / "target_contact_material_authoring.json").write_text(
                                json.dumps(material_diag, indent=2)
                            )
                            _log(
                                f"scenario {sid}: target contact material authoring "
                                f"{material_diag.get('status')} target={target_prim_path}"
                            )
                        elif not target_prim_path:
                            blockers.append("target_contact_material_target_prim_unresolved")
                            _log(f"scenario {sid}: target contact material requested but target prim missing")
                else:
                    blockers.append("task_stance_plan_failed")
                    _log(
                        f"scenario {sid}: task stance blocked "
                        f"{stance_plan.get('blockers') or stance_plan.get('status')}"
                    )
            # When no explicit look-at is given, the manipulation camera + arm reach face the SAME
            # target the stance was planned around (resolved from the scene+task via scene_placement),
            # so a fully dynamic render needs NO hardcoded coordinates at all. Falls back to the
            # explicit manipulation_look_at when one is provided (identical to prior behavior).
            effective_look_at = manipulation_look_at
            if (effective_look_at is None and stance_plan is not None
                    and stance_plan.get("status") == "accepted"):
                effective_look_at = (
                    _surface_affordance_point_for_stance(stance_plan, stand_root)
                    or stance_plan.get("task_target_xyz")
                )
                if effective_look_at is not None:
                    _log(f"scenario {sid}: camera/reach look-at resolved from scene+task surface -> "
                         f"{tuple(round(float(c), 2) for c in effective_look_at)}")
                    # The pre-loop fill light only fires for an explicit look-at; light the
                    # dynamically-resolved workspace too so the dynamic (no-coords) path isn't dark.
                    if manipulation_cam and fill_light_intensity > 0:
                        try:
                            _add_workspace_fill_light(stage, effective_look_at,
                                                      intensity=fill_light_intensity)
                            _log(f"scenario {sid}: workspace fill light @ "
                                 f"{tuple(round(float(c),2) for c in effective_look_at)} "
                                 f"intensity={fill_light_intensity} (resolved)")
                        except Exception as exc:  # noqa: BLE001 - lighting is best-effort
                            _log(f"dynamic workspace fill light skipped ({exc!r})")
            if sc.get("task_target_deferred"):
                if (
                    manipulation_stand
                    and stance_plan is not None
                    and stance_plan.get("status") == "accepted"
                    and stand_root is not None
                ):
                    _materialize_deferred_task_route(
                        sc,
                        stance_plan=stance_plan,
                        root_pose=stand_root,
                        look_at=effective_look_at,
                    )
                    _log(
                        f"scenario {sid}: task-only route materialized from accepted stance "
                        f"start={tuple(round(float(c), 2) for c in sc['start'])} "
                        f"target={tuple(round(float(c), 2) for c in sc['target'])}"
                    )
                else:
                    blockers.append("task_only_target_resolution_failed")
                    _log(f"scenario {sid}: task-only target could not be resolved into an accepted stance")
                    return None
            if not sc.get("route_points") or sc.get("start") is None or sc.get("target") is None:
                blockers.append("scenario_route_missing_after_task_resolution")
                _log(f"scenario {sid}: missing start/target/route after task resolution")
                return None
            _settle_after_task_probe_if_needed(sid)
            effective_groot_policy_command = (
                str(groot_policy_command or "").strip()
                or os.getenv(GROOT_POLICY_COMMAND_ENV, "").strip()
            )
            groot_policy_enabled = (
                str(policy_id).strip()
                in {"groot_sonic", "groot", "groot_n17_sonic", "unitree_groot_n17_sonic_policy"}
                and bool(effective_groot_policy_command)
            )
            seed_pol = None
            last_groot_policy_frame_path = (
                Path(groot_policy_initial_frame).expanduser()
                if str(groot_policy_initial_frame or "").strip()
                else None
            )
            if groot_policy_enabled:
                seed_pol = policy_mod.make_policy(
                    "blueprint_default_walk_to_target_smoke_policy"
                )
                seed_pol.reset(sc)
                pol = policy_mod.make_policy(
                    policy_id,
                    infer=_make_groot_policy_command_infer(
                        command=effective_groot_policy_command,
                        scenario=sc,
                        call_dir=sdir / "groot_policy_calls",
                        timeout_seconds=float(groot_policy_command_timeout_seconds),
                    ),
                )
            else:
                pol = policy_mod.make_policy(policy_id)
            pol.reset(sc)
            t_sc = time.time()
            actions: list[dict] = []
            skel_rows: list[dict] = []
            manipulation_reach_fractions: list[float] = []
            trace = (sdir / "trace.jsonl").open("w")
            rejected_total = response_total = 0
            cap = 0
            truncated = False
            _log(f"scenario {sid}: stepping {steps}")
            for step in range(steps):
                if manipulation_stand and stand_root is None:
                    truncated = True
                    break
                if time.time() - t_sc > per_scenario_seconds:
                    _log(f"scenario {sid}: per-scenario cap {per_scenario_seconds}s hit at step {step}; truncating")
                    truncated = True
                    break
                ctx = policy_mod.StepContext(
                    step=step,
                    num_steps=steps,
                    probe_collision=probe,
                    camera_rgb=(
                        str(last_groot_policy_frame_path)
                        if last_groot_policy_frame_path is not None
                        and last_groot_policy_frame_path.is_file()
                        else None
                    ),
                    joint_state=dict(UNITREE_G1_SONIC_NEUTRAL_STATE),
                    instruction=_scenario_instruction(sc),
                )
                if (
                    groot_policy_enabled
                    and seed_pol is not None
                    and (
                        last_groot_policy_frame_path is None
                        or not last_groot_policy_frame_path.is_file()
                    )
                ):
                    decision = seed_pol.step(ctx)
                    decision.policy_action = "seed_frame_collection_before_groot_policy_requery"
                else:
                    decision = pol.step(ctx)
                if manipulation_stand and stand_root is not None:
                    # Manipulation task: place the robot at the probed clear-floor task stance,
                    # facing the target. The target is what the robot faces, not the pelvis position.
                    decision.root_pose = stand_root
                    decision.yaw = stand_yaw
                    decision.desired_root_position = stand_root
                    decision.policy_action = "accepted_task_stance_collision_checked_placement"
                    if stance_plan is not None:
                        candidates = stance_plan.get("candidates", [])
                        decision.collision_probe_candidate_count = len(candidates)
                        decision.rejected_collision_probe_count = int(
                            stance_plan.get("selected_candidate_index") or 0
                        )
                        decision.rejected_probes = [
                            c for c in candidates
                            if int(c.get("scene_collision_contact_count") or 0) > 0
                        ]
                rejected_total += decision.rejected_collision_probe_count
                if decision.policy_action != "accepted_direct_collision_checked_motion":
                    response_total += 1
                route_distance_m = policy_mod.route_distance(sc["route_points"])
                alpha = 0.0 if steps <= 1 else step / float(steps - 1)
                manipulation_reach_frac = None
                if manipulation_reach and effective_look_at is not None:
                    manipulation_reach_frac = manipulation_reach_fraction_for_frame(
                        alpha,
                        manipulation_cam=bool(manipulation_cam),
                        frame_count=int(steps),
                    )
                phase = policy_mod.gait_phase(alpha, route_distance_m)
                moving = route_distance_m > 0.05 and step < max(1, steps - 1)
                if art_ctx is not None:
                    if dynamic_standing_contact_steps > 0:
                        report = _settle_dynamic_standing_contacts(
                            stage=stage,
                            art_ctx=art_ctx,
                            robot_prim_path=binding["prim_path"],
                            root_pose=decision.root_pose,
                            yaw=decision.yaw,
                            phase=phase,
                            moving=moving,
                            settle_steps=dynamic_standing_contact_steps,
                            scenario_id=sid,
                            manipulation_ready=bool(manipulation_reach),
                            manipulation_reach_arm=manipulation_reach_arm,
                            root_pose_seeded_before_tensor_view=sid in dynamic_seed_decisions,
                            effort_drive=effective_effort_drive,
                        )
                        report["step"] = step
                        physics_contact_reports.append(report)
                        if report["status"] != "completed":
                            blockers.append("physics_articulation_standing_contact_settle_failed")
                            _log(f"dynamic standing/contact settle failed at step {step}: {report['errors']}")
                    else:
                        _drive_g1_walk(
                            art_ctx["art"],
                            art_ctx["dof_index"],
                            art_ctx["default"],
                            root_pose=decision.root_pose,
                            yaw=decision.yaw,
                            phase=phase,
                            moving=moving,
                            manipulation_ready=bool(manipulation_reach),
                            manipulation_reach_arm=manipulation_reach_arm,
                            policy_joint_targets=decision.joint_targets,
                        )
                else:
                    if robot_neutral_xforms:
                        restored = _restore_robot_neutral_descendant_xforms(stage, robot_neutral_xforms)
                        if step == 0:
                            _log(
                                f"scenario {sid}: neutral G1 descendant xforms restored "
                                f"({restored} prims) before root placement/reach pose"
                            )
                    last_root_diagnostics = _place_root(
                        stage,
                        binding["prim_path"],
                        decision.root_pose,
                        decision.yaw,
                    )
                    if (
                        manipulation_stand
                        and stance_plan is not None
                        and stance_plan.get("status") == "accepted"
                        and placement_validation_manifest is None
                    ):
                        placement_validation_manifest = _build_placement_validation_manifest(
                            stage=stage,
                            robot_prim_path=binding["prim_path"],
                            stance_plan=stance_plan,
                            accepted_pose=decision.root_pose,
                            accepted_yaw=decision.yaw,
                            root_diagnostics=last_root_diagnostics,
                            scene_objects=scene_objects_for_validation,
                            scenario_id=sid,
                        )
                        placement_validation_path.write_text(
                            json.dumps(placement_validation_manifest, indent=2)
                        )
                        if not _placement_validation_passed_manifest(placement_validation_manifest):
                            if "placement_validation_failed" not in blockers:
                                blockers.append("placement_validation_failed")
                            placement_validation_blocker_recorded = True
                            _log(
                                f"scenario {sid}: placement validation FAILED "
                                f"{placement_validation_manifest.get('blockers')}"
                            )
                        else:
                            _log(
                                f"scenario {sid}: placement validation PASS "
                                f"xy_error={placement_validation_manifest['ground_truth_placement'].get('xy_error_m')}"
                            )
                    # Show manipulation-ready arms in the RENDERED frame (pure USD, no physics tensor
                    # -> no crash). The same requested arm set is used for render, camera metadata, and
                    # geometry validation so a both-arm seed cannot be validated as a one-arm frame.
                    if (kinematic_arm_pose and manipulation_reach
                            and effective_look_at is not None
                            and manipulation_reach_frac is not None):
                        try:
                            posed_count = _pose_arm_kinematic_usd(
                                stage,
                                binding["prim_path"],
                                effective_look_at,
                                arm=rendered_reach_arm,
                                reach_frac=manipulation_reach_frac,
                                forward_yaw=decision.yaw,
                            )
                            if step == 0:
                                _log(
                                    f"scenario {sid}: kinematic arm pose requested "
                                    f"arm={rendered_reach_arm} "
                                    f"requested_arm={manipulation_reach_arm} "
                                    f"posed_count={posed_count} "
                                    f"reach_fraction={manipulation_reach_frac:.3f} "
                                    f"affordance={tuple(round(float(c), 3) for c in effective_look_at)}"
                                )
                                review_proxy_diag = None
                                if robot_visual_missing:
                                    review_proxy_diag = _create_robot_review_visual_proxies(
                                        stage,
                                        binding["prim_path"],
                                        proxy_root_path=(
                                            f"/World/RobotReviewVisualProxies/{_safe_prim_segment(sid)}"
                                        ),
                                        arm=rendered_reach_arm,
                                    )
                                    _log(
                                        f"scenario {sid}: using review-only robot visual proxies "
                                        f"status={review_proxy_diag.get('status')} "
                                        f"gprims={review_proxy_diag.get('created_gprim_count')}"
                                    )
                                scenario_robot_render_diag = _robot_render_visibility_diagnostics(
                                    stage,
                                    binding["prim_path"],
                                )
                                scenario_robot_render_diag["posed_arm_link_count"] = int(posed_count)
                                scenario_robot_render_diag["visual_binding_status"] = binding.get(
                                    "visual_binding_status"
                                )
                                scenario_robot_render_diag["visual_candidate_attempts"] = binding.get(
                                    "visual_candidate_attempts", []
                                )
                                if review_proxy_diag is not None:
                                    scenario_robot_render_diag[
                                        "review_visual_proxy"
                                    ] = review_proxy_diag
                                (sdir / "robot_render_diagnostics.json").write_text(
                                    json.dumps(scenario_robot_render_diag, indent=2)
                                )
                        except Exception as exc:  # noqa: BLE001 - pose is best-effort, never blocks frames
                            if step == 0:
                                _log(f"kinematic arm pose skipped ({exc!r})")
                rec = policy_mod.action_record(
                    decision=decision, step=step, sim_time_s=step / float(fps), target=sc["target"],
                    scenario_eval_run_id=sc.get("scenario_eval_run_id"))
                if manipulation_reach_frac is not None:
                    rec["manipulation_reach_fraction"] = round(
                        float(manipulation_reach_frac), 6
                    )
                    rec["manipulation_reach_fraction_source"] = (
                        "temporal_robot_pov_reach_ramp"
                        if manipulation_cam and steps > 1
                        else "linear_reach_ramp"
                    )
                    rec["manipulation_stand_root_static_by_design"] = bool(
                        manipulation_stand
                    )
                    rec["not_a_learned_robot_policy_action"] = True
                    manipulation_reach_fractions.append(float(manipulation_reach_frac))
                actions.append(rec)
                trace.write(json.dumps(rec) + "\n")
                if step % max(1, capture_every) == 0:
                    cam_meta: dict[str, Any] | None = None
                    if manipulation_cam:
                        eye, tgt, cam_meta = _robot_mounted_manipulation_cam_pose(
                            stage,
                            binding["prim_path"],
                            decision.root_pose,
                            decision.yaw,
                            look_at=effective_look_at,
                            reach_arm=pov_reach_arm,
                            vfov_deg=pov_vfov_deg,
                            width=width,
                            height=height,
                        )
                        if cap == 0:
                            _log(
                                "scenario "
                                f"{sid}: robot_pov camera source={cam_meta.get('source')} "
                                f"mount={cam_meta.get('mount_prim_path')} "
                                f"arm_links={cam_meta.get('arm_link_points_used')} "
                                f"target_select={cam_meta.get('selected_camera_target')}"
                            )
                    else:
                        eye, tgt = follow_cam_pose(decision.root_pose, decision.yaw)
                    if (
                        manipulation_cam
                        and manipulation_reach
                        and effective_look_at is not None
                        and cam_meta is not None
                    ):
                        pov_geom = _manipulation_pov_geometry(
                            arm_points=cam_meta.get("arm_link_points_xyz") or {},
                            arm_points_by_arm=cam_meta.get("arm_link_points_by_arm_xyz") or {},
                            affordance=effective_look_at,
                            eye=eye,
                            target=tgt,
                            vfov_deg=pov_vfov_deg,
                            width=width,
                            height=height,
                            arm=pov_reach_arm,
                        )
                        pov_geom = {
                            **pov_geom,
                            "step": step,
                            "frame_index": cap,
                            "manipulation_reach_fraction": (
                                round(float(manipulation_reach_frac), 6)
                                if manipulation_reach_frac is not None
                                else None
                            ),
                            "camera_meta": cam_meta,
                            "robot_visual_geometry": {
                                "status": (
                                    "FAIL" if robot_visual_missing
                                    else str(
                                        (scenario_robot_render_diag or {}).get("status") or "PASS"
                                    )
                                ),
                                "blockers": (
                                    [ROBOT_VISUAL_MESH_MISSING_BLOCKER]
                                    if robot_visual_missing
                                    else list(
                                        (scenario_robot_render_diag or {}).get("blockers") or []
                                    )
                                ),
                                "gprim_count": (scenario_robot_render_diag or {}).get(
                                    "gprim_count"
                                ),
                                "mesh_count": (scenario_robot_render_diag or {}).get("mesh_count"),
                                "visual_binding_status": binding.get("visual_binding_status"),
                                "claim_boundary": (
                                    "USD arm-link projections are not visual proof unless the robot "
                                    "subtree also has renderable Gprim/Mesh surfaces."
                                ),
                            },
                        }
                        if robot_visual_missing:
                            geom_blockers = list(pov_geom.get("blockers") or [])
                            geom_blockers.append(ROBOT_VISUAL_MESH_MISSING_BLOCKER)
                            if (
                                isinstance(scenario_robot_render_diag, dict)
                                and (
                                    scenario_robot_render_diag.get("review_visual_proxy") or {}
                                ).get("status") == "PASS"
                            ):
                                geom_blockers.append(ROBOT_REVIEW_VISUAL_PROXY_USED_BLOCKER)
                                pov_geom["review_visual_proxy"] = scenario_robot_render_diag[
                                    "review_visual_proxy"
                                ]
                            pov_geom["blockers"] = sorted(set(str(b) for b in geom_blockers))
                            pov_geom["status"] = "FAIL"
                        pov_geometry_records.append(pov_geom)
                        pov_geometry_report = _write_pov_geometry_report()
                        if cap == 0:
                            _log(
                                f"scenario {sid}: manipulation POV geometry "
                                f"{pov_geom.get('status')} roles={pov_geom.get('arm_roles_in_frame')} "
                                f"target_in_frame={pov_geom.get('target_in_frame')}"
                            )
                        if pov_geom.get("status") != "PASS" and not pov_geometry_blocker_recorded:
                            blockers.append("manipulation_pov_geometry_failed")
                            pov_geometry_blocker_recorded = True
                    _place_camera(stage, pov_cam, eye, tgt)  # POV camera (manipulation egocentric or follow)
                    if manipulation_cam and manipulation_reach and effective_look_at is not None:
                        # Front-light the reaching arm+gripper from the camera side (the workspace fill
                        # light is BEYOND the arm, leaving its camera-facing side in shadow/black).
                        _add_pov_headlamp(stage, eye, effective_look_at,
                                          intensity=(fill_light_intensity if fill_light_intensity > 0 else 30000.0))
                    if verify_annot is not None:
                        v_eye, v_tgt = verify_cam_pose(
                            decision.root_pose,
                            decision.yaw,
                            look_at=effective_look_at,
                        )
                        _place_camera(stage, verify_cam_path, v_eye, v_tgt)  # 3rd-person: SHOW the robot
                    debug_root_path = (
                        f"/World/PlacementDebug/{_safe_prim_segment(sid)}"
                        if topdown_enabled and stance_plan is not None
                        else None
                    )
                    if debug_root_path is not None:
                        try:
                            stage.RemovePrim(debug_root_path)
                        except Exception:  # noqa: BLE001 - absence is fine
                            pass
                    if articulated and (art_ctx is not None or rest_offsets is not None):
                        skel = skeleton_world_for_frame(
                            art_ctx=art_ctx,
                            rest_offsets=rest_offsets,
                            root_pose=decision.root_pose,
                            yaw=decision.yaw,
                        )
                        if manipulation_reach and effective_look_at is not None:
                            reach_frac = (
                                manipulation_reach_frac
                                if manipulation_reach_frac is not None
                                else manipulation_reach_fraction_for_frame(
                                    alpha,
                                    manipulation_cam=bool(manipulation_cam),
                                    frame_count=int(steps),
                                )
                            )
                            skel = compute_arm_reach_skeleton(
                                skel,
                                effective_look_at,
                                reach_frac,
                                arm=rendered_reach_arm,
                                forward_yaw=decision.yaw,
                            )
                        lms = _project_skeleton(skel, eye=eye, target=tgt, up=(0.0, 0.0, 1.0),
                                                vfov_deg=pov_vfov_deg, width=width, height=height)
                        if cap == 0:
                            _log(f"step {step}: skeleton {len(skel)} links -> {len(lms)} landmarks in POV frame")
                        skel_rows.append({
                            "schema_version": "blueprint.isaac_g1.projected_upper_body_skeleton.v1",
                            "episode_id": sid,
                            "scenario_eval_run_id": sc.get("scenario_eval_run_id") or sid,
                            "step": step, "sim_time_s": round(step / float(fps), 6),
                            "camera": "robot_pov", "landmarks": lms,  # OSCAR reads row["landmarks"]
                            "projected_landmark_count": len(lms),
                            "manipulation_reach_fraction": round(float(reach_frac), 6),
                            "claim_boundary": {
                                "projected_skeleton_trace_derived_from_seed_render_geometry": bool(
                                    manipulation_reach
                                ),
                                "temporal_rows_are_target_conditioning_from_resolved_affordance_projection": bool(
                                    manipulation_reach and effective_look_at is not None
                                ),
                                "not_a_learned_robot_policy_action": True,
                                "policy_derived_action_conditioning": False,
                                "simulated_state_not_physical_robot_sensor_evidence": True,
                                "isaac_render_skeleton_conditioning_support_only": True,
                                "not_task_success_proof": True,
                            }})
                    ts = time.time()
                    if cap == 0 and warmup_frames > 0:
                        _log(
                            f"scenario {sid}: first-frame warmup {warmup_frames} render frames "
                            f"(capped {per_scenario_seconds}s)"
                        )
                        for wi in range(warmup_frames):
                            if time.time() - t_sc > per_scenario_seconds:
                                _log(f"first-frame warmup hit time cap at frame {wi}")
                                break
                            wts = time.time()
                            _replicator_step_with_watchdog(
                                rep,
                                label=f"{sid}:frame:{cap}:warmup:{wi}",
                                result_path=out_dir / "isaac_g1_kitchen_parity_result.json",
                                scenario_id=sid,
                            )
                            _log(f"first-frame warmup frame {wi} render took {time.time() - wts:.1f}s")
                    # Settle-then-capture, the render-noise audit's proven recipe: path tracing
                    # accumulates across repeated step() calls on a still scene, so after the
                    # pose change render a few settle steps before the capture step. (The
                    # rt_subframes step parameter alone measurably did NOT accumulate on this
                    # Kit build — settle steps did.)
                    for _si in range(capture_settle_steps):
                        if time.time() - t_sc > per_scenario_seconds:
                            _log(f"scenario {sid}: settle hit scenario time cap at step {_si}")
                            break
                        _replicator_step_with_watchdog(
                            rep,
                            label=f"{sid}:frame:{cap}:settle:{_si}",
                            result_path=out_dir / "isaac_g1_kitchen_parity_result.json",
                            scenario_id=sid,
                            rt_subframes=1,
                        )
                    _replicator_step_with_watchdog(
                        rep,
                        label=f"{sid}:frame:{cap}:rt_subframes:{capture_rt_subframes}",
                        result_path=out_dir / "isaac_g1_kitchen_parity_result.json",
                        scenario_id=sid,
                        rt_subframes=capture_rt_subframes,
                    )
                    rdt = time.time() - ts
                    over_frame_path = sdir / "frames" / f"overview_{cap:04d}.png"
                    over_ok = _save_rgb(
                        over_annot,
                        over_frame_path,
                        software_denoise=software_denoise,
                    )
                    if not over_ok:
                        try:
                            _log(
                                f"scenario {sid}: overview annotator empty at frame {cap}; "
                                "recreating render product"
                            )
                            over_annot = _make_render_product(over_cam, width, height)
                            _replicator_step_with_watchdog(
                                rep,
                                label=f"{sid}:overview:{cap}:recreate",
                                result_path=out_dir / "isaac_g1_kitchen_parity_result.json",
                                scenario_id=sid,
                                rt_subframes=capture_rt_subframes,
                            )
                            over_ok = _save_rgb(
                                over_annot,
                                over_frame_path,
                                software_denoise=software_denoise,
                            )
                        except Exception as exc:  # noqa: BLE001 - frame-save fallback is diagnostic
                            _log(f"scenario {sid}: overview render-product recreate failed ({exc!r})")
                    if over_ok:
                        _append_camera_contract_row(
                            _isaac_camera_contract(stage, over_cam, width, height),
                            over_frame_path,
                            cap,
                            camera_role="overview",
                            camera_mode="static_overview",
                        )
                    if over_ok and over_depth_annot is not None:
                        if _save_depth(
                            over_depth_annot,
                            sdir / "frames" / f"overview_depth_{cap:04d}.png",
                            npy_path=sdir / "frames" / f"overview_depth_{cap:04d}.npy",
                        ):
                            depth_frames_written += 1
                    pov_frame_path = sdir / "frames" / f"robot_pov_{cap:04d}.png"
                    pov_ok = _save_rgb(pov_annot, pov_frame_path,
                                       software_denoise=software_denoise)
                    if not pov_ok:
                        try:
                            _log(
                                f"scenario {sid}: robot_pov annotator empty at frame {cap}; "
                                "recreating render product"
                            )
                            pov_annot = _make_render_product(pov_cam, width, height)
                            _replicator_step_with_watchdog(
                                rep,
                                label=f"{sid}:robot_pov:{cap}:recreate",
                                result_path=out_dir / "isaac_g1_kitchen_parity_result.json",
                                scenario_id=sid,
                                rt_subframes=capture_rt_subframes,
                            )
                            pov_ok = _save_rgb(
                                pov_annot,
                                pov_frame_path,
                                software_denoise=software_denoise,
                            )
                        except Exception as exc:  # noqa: BLE001 - frame-save fallback is diagnostic
                            _log(f"scenario {sid}: robot_pov render-product recreate failed ({exc!r})")
                    if pov_ok:
                        _append_camera_contract_row(
                            _isaac_camera_contract(stage, pov_cam, width, height),
                            pov_frame_path,
                            cap,
                            camera_role="robot_pov_compat",
                            camera_mode=pov_camera_mode,
                            extra={
                                "robot_pov_filename_compatibility": True,
                                "true_robot_head_pov": bool(manipulation_cam),
                                "camera_mode_label": pov_camera_label,
                            },
                        )
                        if groot_policy_enabled:
                            last_groot_policy_frame_path = pov_frame_path
                    if pov_ok and segmentation and pov_seg_annots is not None:
                        seg_save = _save_segmentation(
                            pov_seg_annots,
                            instance_png=sdir / "frames" / f"robot_pov_instance_{cap:04d}.png",
                            semantic_png=sdir / "frames" / f"robot_pov_semantic_{cap:04d}.png",
                            id_label_json=segmentation_id_label_path,
                        )
                        if seg_save.get("instance_saved"):
                            segmentation_instance_frames += 1
                        if seg_save.get("semantic_saved"):
                            segmentation_semantic_frames += 1
                        segmentation_blockers.extend(seg_save.get("blockers") or [])
                    if pov_ok and pov_depth_annot is not None:
                        if _save_depth(
                            pov_depth_annot,
                            sdir / "frames" / f"robot_pov_depth_{cap:04d}.png",
                            npy_path=sdir / "frames" / f"robot_pov_depth_{cap:04d}.npy",
                        ):
                            depth_frames_written += 1
                    if manipulation_cam and manipulation_reach and pov_geometry_records:
                        frame_quality = (
                            _pov_seed_frame_quality(pov_frame_path)
                            if pov_ok
                            else {
                                "schema_version": "manipulation_pov_seed_frame_quality.v1",
                                "status": "FAIL",
                                "blockers": ["manipulation_pov_frame_not_saved"],
                                "frame_path": str(pov_frame_path),
                            }
                        )
                        last_pov_geom = dict(pov_geometry_records[-1])
                        frame_blockers = [str(b) for b in (frame_quality.get("blockers") or [])]
                        merged_blockers = sorted(set((last_pov_geom.get("blockers") or []) + frame_blockers))
                        last_pov_geom["seed_frame_quality"] = frame_quality
                        last_pov_geom["blockers"] = merged_blockers
                        last_pov_geom["status"] = "PASS" if not merged_blockers else "FAIL"
                        pov_geometry_records[-1] = last_pov_geom
                        pov_geometry_report = _write_pov_geometry_report()
                        if frame_quality.get("status") != "PASS" and not pov_geometry_blocker_recorded:
                            blockers.append("manipulation_pov_geometry_failed")
                            pov_geometry_blocker_recorded = True
                    if verify_annot is not None:
                        verify_frame_path = sdir / "frames" / f"verify_{cap:04d}.png"
                        verify_ok = _save_rgb(
                            verify_annot,
                            verify_frame_path,
                            software_denoise=software_denoise,
                        )
                        if not verify_ok:
                            try:
                                _log(
                                    f"scenario {sid}: verify annotator empty at frame {cap}; "
                                    "recreating render product"
                                )
                                verify_annot = _make_render_product(verify_cam_path, width, height)
                                _replicator_step_with_watchdog(
                                    rep,
                                    label=f"{sid}:verify:{cap}:recreate",
                                    result_path=out_dir / "isaac_g1_kitchen_parity_result.json",
                                    scenario_id=sid,
                                    rt_subframes=capture_rt_subframes,
                                )
                                verify_ok = _save_rgb(
                                    verify_annot,
                                    verify_frame_path,
                                    software_denoise=software_denoise,
                                )
                            except Exception as exc:  # noqa: BLE001 - frame-save fallback is diagnostic
                                _log(f"scenario {sid}: verify render-product recreate failed ({exc!r})")
                        if verify_ok:
                            _append_camera_contract_row(
                                _isaac_camera_contract(stage, verify_cam_path, width, height),
                                verify_frame_path,
                                cap,
                                camera_role="third_person_verify",
                                camera_mode="third_person_verify",
                            )
                    if topdown_enabled and stance_plan is not None and debug_root_path is not None:
                        floor_z = float(decision.root_pose[2]) - ROBOT_PELVIS_HEIGHT_M
                        if stance_plan.get("floor_z_hint") is not None:
                            floor_z = float(stance_plan.get("floor_z_hint"))
                        topdown_debug = _update_topdown_debug_scene(
                            stage,
                            root_path=debug_root_path,
                            robot_pose=decision.root_pose,
                            robot_yaw=decision.yaw,
                            stance_plan=stance_plan,
                            scene_objects=scene_objects_for_validation,
                            floor_z=floor_z,
                        )
                        _place_topdown_debug_camera(
                            stage,
                            topdown_cam_path,
                            center_xy=topdown_debug["center_xy"],
                            radius_m=float(topdown_debug["radius_m"]),
                            width=width,
                            height=height,
                            floor_z=floor_z,
                        )
                        if topdown_annot is None:
                            _log("creating lazy topdown render product")
                            topdown_annot = _make_render_product(topdown_cam_path, width, height)
                        _replicator_step_with_watchdog(
                            rep,
                            label=f"{sid}:topdown:{cap}:rt_subframes:{capture_rt_subframes}",
                            result_path=out_dir / "isaac_g1_kitchen_parity_result.json",
                            scenario_id=sid,
                            rt_subframes=capture_rt_subframes,
                        )
                        placement_topdown_frame_path = sdir / "frames" / f"placement_topdown_{cap:04d}.png"
                        topdown_ok = _save_rgb(
                            topdown_annot,
                            placement_topdown_frame_path,
                            software_denoise=software_denoise,
                        )
                        if topdown_ok:
                            _append_camera_contract_row(
                                _isaac_camera_contract(stage, topdown_cam_path, width, height),
                                placement_topdown_frame_path,
                                cap,
                                camera_role="placement_topdown",
                                camera_mode="orthographic_or_topdown_debug",
                            )
                        placement_topdown_layout_frame_path = (
                            sdir / "frames" / f"placement_topdown_layout_{cap:04d}.png"
                        )
                        _write_topdown_debug_layout_image(
                            placement_topdown_layout_frame_path,
                            robot_pose=decision.root_pose,
                            robot_yaw=decision.yaw,
                            stance_plan=stance_plan,
                            scene_objects=scene_objects_for_validation,
                            width=width,
                            height=height,
                        )
                        try:
                            stage.RemovePrim(debug_root_path)
                        except Exception:  # noqa: BLE001
                            pass
                    if cap == 0 or rdt > 5:
                        _log(f"scenario {sid}: frame {cap} captured (render {rdt:.1f}s, overview_ok={over_ok})")
                    cap += 1
            trace.close()
            if articulated and skel_rows:
                with (sdir / "g1_projected_skeleton_trace.jsonl").open("w") as sf:
                    for r in skel_rows:
                        sf.write(json.dumps(r) + "\n")
                total_lm = sum(r["projected_landmark_count"] for r in skel_rows)
                _log(f"scenario {sid}: skeleton trace {len(skel_rows)} frames, {total_lm} total landmarks")
            if camera_contract_rows:
                camera_contract_path = sdir / "frames" / "camera_contract.jsonl"
                with camera_contract_path.open("w", encoding="utf-8") as cf:
                    for row in camera_contract_rows:
                        cf.write(json.dumps(row) + "\n")
                _log(
                    f"scenario {sid}: camera contract rows {len(camera_contract_rows)} "
                    f"written to {camera_contract_path}"
                )
            scenario_contact_reports = [
                r for r in physics_contact_reports if r.get("scenario_id") == sid
            ]
            if scenario_contact_reports:
                (sdir / "physics_articulation_standing_contact_reports.json").write_text(
                    json.dumps(scenario_contact_reports, indent=2)
                )
            if manipulation_stand:
                if placement_validation_manifest is None and stance_plan is not None and stand_root is not None:
                    placement_validation_manifest = _build_placement_validation_manifest(
                        stage=stage,
                        robot_prim_path=binding["prim_path"],
                        stance_plan=stance_plan,
                        accepted_pose=stand_root,
                        accepted_yaw=float(stand_yaw),
                        root_diagnostics=last_root_diagnostics,
                        scene_objects=scene_objects_for_validation,
                        scenario_id=sid,
                        topdown_frame=(
                            str(placement_topdown_layout_frame_path or placement_topdown_frame_path)
                            if (placement_topdown_layout_frame_path or placement_topdown_frame_path)
                            else None
                        ),
                    )
                if placement_validation_manifest is not None:
                    placement_visual_frames = sorted((sdir / "frames").glob("verify_*.png"))
                    pov_visual_frames = sorted((sdir / "frames").glob("robot_pov_*.png"))
                    visual_qc = _run_task_visual_qc(
                        placement_visual_frames,
                        pov_visual_frames if manipulation_cam else [],
                        target_label=_placement_visual_qc_target_label(stance_plan),
                        task_description=_task_description_for_scenario(sc),
                    )
                    placement_validation_manifest["visual_qc"] = visual_qc
                    placement_validation_manifest["topdown_debug_frame"] = (
                        str(placement_topdown_layout_frame_path or placement_topdown_frame_path)
                        if (placement_topdown_layout_frame_path or placement_topdown_frame_path)
                        else None
                    )
                    placement_validation_manifest["orthographic_topdown_frame"] = (
                        str(placement_topdown_frame_path) if placement_topdown_frame_path else None
                    )
                    placement_validation_manifest["topdown_layout_frame"] = (
                        str(placement_topdown_layout_frame_path)
                        if placement_topdown_layout_frame_path
                        else None
                    )
                    blockers_now = set(placement_validation_manifest.get("blockers") or [])
                    if _visual_qc_contains_parsed_failure(visual_qc):
                        blockers_now.add("placement_visual_qc_failed")
                    placement_validation_manifest["robot_visual_geometry"] = {
                        "status": (
                            "FAIL" if robot_visual_missing
                            else str((scenario_robot_render_diag or {}).get("status") or "PASS")
                        ),
                        "blockers": (
                            [ROBOT_VISUAL_MESH_MISSING_BLOCKER]
                            if robot_visual_missing
                            else list((scenario_robot_render_diag or {}).get("blockers") or [])
                        ),
                        "gprim_count": (scenario_robot_render_diag or {}).get("gprim_count"),
                        "mesh_count": (scenario_robot_render_diag or {}).get("mesh_count"),
                        "visual_binding_status": binding.get("visual_binding_status"),
                        "diagnostics_path": str(sdir / "robot_render_diagnostics.json"),
                        "claim_boundary": (
                            "Placement validation requires robot visual surfaces for rendered "
                            "review frames; USD link projections alone do not prove visible arms/body."
                        ),
                    }
                    if robot_visual_missing:
                        blockers_now.add(ROBOT_VISUAL_MESH_MISSING_BLOCKER)
                    if pov_geometry_report is not None:
                        placement_validation_manifest["manipulation_pov_geometry"] = {
                            "status": pov_geometry_report.get("status"),
                            "path": str(pov_geometry_path),
                            "blockers": pov_geometry_report.get("blockers", []),
                            "frames_checked": pov_geometry_report.get("frames_checked"),
                        }
                        if pov_geometry_report.get("status") != "PASS":
                            blockers_now.add("manipulation_pov_geometry_failed")
                            blockers_now.update(pov_geometry_report.get("blockers") or [])
                    placement_validation_manifest["blockers"] = sorted(blockers_now)
                    placement_validation_manifest["status"] = (
                        "PASS" if not placement_validation_manifest["blockers"] else "FAIL"
                    )
                    placement_validation_path.write_text(
                        json.dumps(placement_validation_manifest, indent=2)
                    )
                    if not _placement_validation_passed_manifest(placement_validation_manifest):
                        if "placement_validation_failed" not in blockers:
                            blockers.append("placement_validation_failed")
                        if not placement_validation_blocker_recorded:
                            _log(
                                f"scenario {sid}: placement validation FAILED "
                                f"{placement_validation_manifest.get('blockers')}"
                            )
                    else:
                        _log(f"scenario {sid}: placement validation final PASS")
            _log(f"scenario {sid}: {cap} frames captured, truncated={truncated}; assembling MP4 + outcome")
            summary = assemble_collision_summary(actions=actions, rejected_probe_total=rejected_total,
                                                 response_event_total=response_total)
            outcome = policy_mod.compute_task_outcome(
                actions=actions, start=sc["start"], target=sc["target"],
                route_distance_m=policy_mod.route_distance(sc["route_points"]),
                collision_summary=summary, bounded_steps=len(actions), model_timestep_s=1.0 / float(fps))
            outcome["frames_captured"] = cap
            outcome["truncated"] = truncated
            outcome["per_frame_camera_contract_emitted"] = bool(camera_contract_rows)
            outcome["per_frame_camera_contract_frames"] = len(camera_contract_rows)
            outcome["per_frame_camera_contract_available_intrinsics_frames"] = sum(
                1
                for row in camera_contract_rows
                if isinstance(row.get("intrinsics"), Mapping)
                and row["intrinsics"].get("available") is True
            )
            if manipulation_reach_fractions:
                unique_reach_fractions = sorted(
                    {round(float(value), 6) for value in manipulation_reach_fractions}
                )
                temporal_status = (
                    "PASS"
                    if len(unique_reach_fractions) > 1
                    and unique_reach_fractions[-1] > unique_reach_fractions[0]
                    else "FAIL"
                )
                outcome["manipulation_temporal_conditioning"] = {
                    "schema_version": "manipulation_temporal_conditioning.v1",
                    "status": temporal_status,
                    "frame_count": len(manipulation_reach_fractions),
                    "reach_fraction_start": unique_reach_fractions[0],
                    "reach_fraction_end": unique_reach_fractions[-1],
                    "unique_reach_fraction_count": len(unique_reach_fractions),
                    "root_static_by_design": bool(manipulation_stand),
                    "rendered_usd_arm_pose_requested": bool(kinematic_arm_pose),
                    "projected_skeleton_trace_requires_articulated": True,
                    "claim_boundary": (
                        "Temporal reach fractions are deterministic Isaac/WAM conditioning for "
                        "review media. They are not learned manipulation, contact proof, physical "
                        "reach proof, or task-success evidence."
                    ),
                }
                if temporal_status != "PASS" and steps > 1:
                    outcome["manipulation_temporal_conditioning"]["blockers"] = [
                        "manipulation_reach_conditioning_static_across_frames"
                    ]
            if depth_pass:
                outcome["depth_render_pass"] = {
                    "schema_version": "isaac_g1_kitchen_parity_depth_pass.v1",
                    "simulator_backend": "isaac",
                    "annotator": "distance_to_image_plane",
                    "depth_frames_written": int(depth_frames_written),
                    "co_registered_with_rgb": True,
                    "cameras": ["robot_pov", "overview"],
                    "units": "meters",
                    "depth_proven": depth_frames_written > 0,
                    "claim_boundary": (
                        "Isaac RTX depth render pass only; NOT MuJoCo evidence and not a "
                        "policy/physics/success proof."
                    ),
                }
            if segmentation:
                segmentation_summary["instance_mask_frames"] = int(
                    segmentation_summary.get("instance_mask_frames") or 0
                ) + int(segmentation_instance_frames)
                segmentation_summary["semantic_mask_frames"] = int(
                    segmentation_summary.get("semantic_mask_frames") or 0
                ) + int(segmentation_semantic_frames)
                if segmentation_id_label_path.is_file() and not segmentation_summary.get(
                    "id_label_path"
                ):
                    segmentation_summary["id_label_path"] = str(segmentation_id_label_path)
                segmentation_summary["blockers"] = sorted(set(segmentation_blockers))
                outcome["segmentation"] = {
                    "schema_version": "isaac_g1_kitchen_parity_segmentation_scenario.v1",
                    "status": "PASS"
                    if int(segmentation_summary.get("labeled_prim_count") or 0) > 0
                    and segmentation_instance_frames > 0
                    else "FAIL",
                    "labeled_prim_count": int(
                        segmentation_summary.get("labeled_prim_count") or 0
                    ),
                    "instance_mask_frames": int(segmentation_instance_frames),
                    "semantic_mask_frames": int(segmentation_semantic_frames),
                    "id_label_path": str(segmentation_id_label_path)
                    if segmentation_id_label_path.is_file()
                    else None,
                    "blockers": sorted(set(segmentation_blockers)),
                    "claim_boundary": (
                        "Isaac Replicator native instance/semantic segmentation diagnostic only; "
                        "not MuJoCo evidence, physical sensor proof, or task-success proof."
                    ),
                }
            if stance_plan is not None:
                outcome["task_stance_plan"] = {
                    "status": stance_plan.get("status"),
                    "path": str(stance_plan_path),
                    "accepted_pose": stance_plan.get("accepted_pose"),
                    "accepted_yaw": stance_plan.get("accepted_yaw"),
                    "candidate_count": len(stance_plan.get("candidates", [])),
                    "blockers": stance_plan.get("blockers", []),
                }
            if placement_validation_manifest is not None:
                outcome["placement_validation"] = {
                    "status": placement_validation_manifest.get("status"),
                    "path": str(placement_validation_path),
                    "blockers": placement_validation_manifest.get("blockers", []),
                    "ground_truth_xy_error_m": (
                        placement_validation_manifest.get("ground_truth_placement") or {}
                    ).get("xy_error_m"),
                    "visual_qc_status": (
                        placement_validation_manifest.get("visual_qc") or {}
                    ).get("status"),
                    "topdown_debug_frame": placement_validation_manifest.get("topdown_debug_frame"),
                }
            if manipulation_cam or verify_cam:
                outcome["robot_visual_geometry"] = {
                    "status": (
                        "FAIL" if robot_visual_missing
                        else str((scenario_robot_render_diag or {}).get("status") or "PASS")
                    ),
                    "blockers": (
                        [ROBOT_VISUAL_MESH_MISSING_BLOCKER]
                        if robot_visual_missing
                        else list((scenario_robot_render_diag or {}).get("blockers") or [])
                    ),
                    "gprim_count": (scenario_robot_render_diag or {}).get("gprim_count"),
                    "mesh_count": (scenario_robot_render_diag or {}).get("mesh_count"),
                    "visual_binding_status": binding.get("visual_binding_status"),
                    "diagnostics_path": str(sdir / "robot_render_diagnostics.json"),
                }
            if pov_geometry_report is not None:
                outcome["manipulation_pov_geometry"] = {
                    "status": pov_geometry_report.get("status"),
                    "path": str(pov_geometry_path),
                    "blockers": pov_geometry_report.get("blockers", []),
                    "frames_checked": pov_geometry_report.get("frames_checked"),
                }
            robot_visual_ready = bool(
                not robot_visual_missing
                and (
                    not (manipulation_cam or verify_cam)
                    or _status_passed(outcome.get("robot_visual_geometry"))
                )
            )
            pov_geometry_ready = _status_passed(pov_geometry_report)
            task_success_contract = _scenario_task_success_contract(sc)
            contract_result: dict[str, Any] | None = None
            if task_success_contract == "visible_reach_to_affordance":
                contract_result = _apply_visible_reach_to_affordance_success_contract(
                    outcome,
                    placement_validation=placement_validation_manifest,
                    pov_geometry=pov_geometry_report,
                    robot_visual_ready=robot_visual_ready,
                    temporal_conditioning=outcome.get("manipulation_temporal_conditioning"),
                )
            if task_success_contract == "visible_reach_to_affordance":
                visible_embodied_robot_action = bool(
                    contract_result is not None and contract_result.get("status") == "PASS"
                )
            else:
                visible_embodied_robot_action = bool(
                    manipulation_cam
                    and manipulation_reach
                    and robot_visual_ready
                    and pov_geometry_ready
                )
            review_camera_blockers: list[str] = []
            if pov_camera_mode == "root_follow":
                review_camera_blockers.append("robot_pov_is_root_follow_camera_not_head_pov")
            if not visible_embodied_robot_action:
                review_camera_blockers.append("visible_embodied_robot_action_not_proven")
            if manipulation_cam and manipulation_reach and not pov_geometry_ready:
                review_camera_blockers.append("manipulation_pov_geometry_not_review_ready")
            if not robot_visual_ready:
                review_camera_blockers.append("robot_visual_geometry_not_review_ready")
            outcome["review_camera_evidence"] = {
                "schema_version": REVIEW_CAMERA_EVIDENCE_SCHEMA_VERSION,
                "robot_pov_camera_mode": pov_camera_mode,
                "robot_pov_camera_mode_label": pov_camera_label,
                "robot_pov_filename_compatibility": True,
                "robot_pov_is_true_robot_head_pov": bool(manipulation_cam),
                "verify_camera_requested": bool(verify_cam),
                "manipulation_camera_requested": bool(manipulation_cam),
                "manipulation_reach_requested": bool(manipulation_reach),
                "task_success_contract": task_success_contract,
                "visible_embodied_robot_action_evidence": visible_embodied_robot_action,
                "robot_visual_geometry_review_ready": robot_visual_ready,
                "manipulation_pov_geometry_review_ready": pov_geometry_ready,
                "blockers": sorted(set(review_camera_blockers)),
                "claim_boundary": (
                    "The legacy robot_pov file can be a root-follow camera unless manipulation_cam is "
                    "enabled. Visible embodied robot-action evidence requires a true robot-mounted "
                    "manipulation camera plus review-ready robot/action geometry."
                ),
            }
            outcomes.append(outcome)  # record BEFORE MP4 — MP4 is optional, frames already uploaded
            for name in ("overview", "robot_pov", "placement_topdown"):
                glob = str(sdir / "frames" / f"{name}_*.png")
                try:
                    subprocess.call(mp4_command(glob, fps, str(sdir / f"{name}.mp4")))
                except Exception as e:  # noqa: BLE001
                    _log(f"mp4 assembly for {name} failed ({e!r}); frames preserved for local assembly")
            _log(f"scenario {sid}: done")
            return outcome

        if serve:
            try:  # worker: flat module in the bundle dir (sys.path has it); repo/tests: package
                from warm_render_server import (  # type: ignore
                    FileJobSource, SignedUrlJobSource, serve_render_loop)
            except ImportError:
                from blueprint_pipeline.warm_render_server import (  # type: ignore
                    FileJobSource, SignedUrlJobSource, serve_render_loop)
            inbox_get_url = os.environ.get("BLUEPRINT_WARM_INBOX_GET_URL", "").strip()
            if inbox_get_url:
                warm_source = SignedUrlJobSource(inbox_get_url, out_dir)
                source_label = "signed_url_inbox"
            else:
                serve_root = Path(serve_dir) if serve_dir is not None else (out_dir / "warm_jobs")
                warm_source = FileJobSource(serve_root)
                source_label = f"file:{serve_root}"

            def _serve_render_one(job_scenario):
                normalized = parse_scenarios({"scenarios": [dict(job_scenario)]})
                if not normalized:
                    return {"status": "blocked", "blockers": ["scenario_unparseable"]}
                produced = _render_scenario(normalized[0])
                return {
                    "status": "completed" if produced is not None else "skipped",
                    "scenario_id": normalized[0].get("scenario_id"),
                    "outcome": produced,
                }

            # Readiness marker: the control plane polls the output zip for this so it knows Isaac is
            # booted + the scene is loaded + the loop is accepting jobs (so it can start submitting).
            (out_dir / "warm_serve_ready.json").write_text(json.dumps(
                {"status": "serving", "source": source_label,
                 "launch_session_id": os.environ.get("BLUEPRINT_LAUNCH_SESSION_ID", ""),
                 "idle_timeout_s": serve_idle_timeout_s, "max_jobs": serve_max_jobs}))
            _log(f"warm serve: setup complete; serving jobs from {source_label} "
                 f"(idle_timeout={serve_idle_timeout_s}s, max_jobs={serve_max_jobs})")
            serve_summary = serve_render_loop(
                render_one=_serve_render_one, job_source=warm_source,
                idle_timeout_s=serve_idle_timeout_s, max_jobs=serve_max_jobs, log=_log)
            (out_dir / "warm_serve_summary.json").write_text(json.dumps(serve_summary, indent=2))
            _log(f"warm serve: exit {serve_summary}")
        else:
            for sc in scenarios:
                _render_scenario(sc)
    except Exception as exc:  # noqa: BLE001
        blocker = (
            "isaac_runner_exception_before_scenario_outcome"
            if not outcomes
            else "isaac_runner_exception_after_partial_scenario_outcomes"
        )
        if blocker not in blockers:
            blockers.append(blocker)
        exception_payload = {
            "schema_version": "isaac_g1_kitchen_parity_runner_exception.v1",
            "status": "blocked",
            "blocker": blocker,
            "exception_type": type(exc).__name__,
            "exception_repr": repr(exc),
            "scenarios_total": len(scenarios),
            "scenarios_executed_before_exception": len(outcomes),
            "traceback_tail": traceback.format_exc()[-4000:],
            "claim_boundary": (
                "Runner exception diagnostics explain why the Isaac render did not complete. "
                "They do not validate task success, manipulation success, physical readiness, "
                "safety, or deployment approval."
            ),
        }
        try:
            (out_dir / "isaac_runner_exception.json").write_text(
                json.dumps(exception_payload, indent=2),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass
        _log(
            "runner exception recorded before final result: "
            f"{blocker} {type(exc).__name__}: {exc!r}"
        )
    finally:
        try:
            # SimulationApp.close() can terminate or stall the worker process on some
            # remote runtimes, so persist the collector-visible result before closing Isaac.
            if dynamic_standing_contact_steps > 0 and not outcomes and not blockers:
                blockers.append("physics_articulation_dynamic_standing_contact_stopped_before_outcome")
            result = build_result(scenarios=scenarios, outcomes=outcomes, policy_id=policy_id,
                                  kitchen_usd=kitchen_usd, g1_usd=g1_usd, blockers=blockers,
                                  physics_articulation_contact_reports=physics_contact_reports,
                                  segmentation_summary=segmentation_summary if segmentation else None,
                                  authored_target_contact_material=({
                                      "schema_version": (
                                          "isaac_target_contact_material_authoring_summary.v1"
                                      ),
                                      "enabled": True,
                                      "records": authored_target_contact_material_records,
                                  } if author_target_contact_material else None))
            (out_dir / "isaac_g1_kitchen_parity_result.json").write_text(json.dumps(result, indent=2))
        finally:
            try:
                sim.close()
            except Exception:  # noqa: BLE001
                pass
    assert result is not None
    return result


# --------------------------------------------------------------------------------------------------
# Local dry-render preview (NO GPU, NO Isaac) — catch placement/camera/POV-framing bugs in <1s
# before spending a cloud render. It reproduces the SAME stance + camera + arm-skeleton math the GPU
# path runs (plan_task_stance / manipulation_cam_pose / compute_arm_reach_skeleton / project_point_to_
# pixel), so wrong-side stance, a camera that crops the reaching arm, or a camera aimed into an
# appliance are all visible locally. The arm skeleton uses a NOMINAL (Isaac-free) G1, so the arm is
# approximate; the stance, camera pose, and projection are exact.
# --------------------------------------------------------------------------------------------------

# Provenance for every dry-render artifact (the preview PNG metadata AND the summary JSON) is stamped
# from the module-level DRY_RENDER_SOURCE_MARKER / DRY_RENDER_NOT_RENDERED_NOTE constants (defined near
# the top of this module). The dry-render preview is a CPU geometry sketch — stance/camera/projection
# are exact, but the arm is a nominal Isaac-free skeleton and NOTHING is path-traced — so it must never
# be filed or screenshotted as if it were a real Isaac/RTX render. This mirrors the native_runtime
# placeholder path's "X-Blueprint-Render-Source: placeholder_cosmos_pending" claim-boundary marker.

# Nominal G1 link offsets from the pelvis root (robot frame: +x forward, +y left, +z up). Approximate
# but dimensionally faithful enough that camera framing of the reaching arm transfers to the GPU run.
# Link names mirror the real USD ("...link") so compute_arm_reach_skeleton recognizes the arm chain.
_NOMINAL_G1_REST_OFFSETS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("pelvis_link", (0.0, 0.0, 0.0)),
    ("torso_link", (0.0, 0.0, 0.20)),
    ("head_link", (0.06, 0.0, 0.45)),
    ("right_shoulder_link", (0.0, -0.16, 0.34)),
    ("right_elbow_link", (0.06, -0.20, 0.10)),
    ("right_wrist_link", (0.12, -0.22, -0.06)),
    ("right_hand_link", (0.16, -0.23, -0.14)),
    ("left_shoulder_link", (0.0, 0.16, 0.34)),
    ("left_elbow_link", (0.06, 0.20, 0.10)),
    ("left_wrist_link", (0.12, 0.22, -0.06)),
    ("left_hand_link", (0.16, 0.23, -0.14)),
)


def _open_stage_local(usd_path: str):
    """Open a USD stage with plain pxr (NO omni/Isaac) for the local dry-render's geometry reads."""
    from pxr import Usd  # type: ignore
    return Usd.Stage.Open(str(usd_path))


def _bind_proxy_robot(
    stage,
    prim_path: str = "/World/G1",
    *,
    footprint_xy: tuple[float, float] = (0.50, 0.36),
    height: float = 1.55,
) -> str:
    """Bind a footprint-sized proxy robot box at ``prim_path`` so the geometric placement validator can
    place + bbox a 'robot' locally (the real G1 USD is an Isaac asset not present off-GPU). The proxy
    reproduces the standing footprint the validator checks for clip/standoff, no Isaac needed."""
    from pxr import UsdGeom, Gf  # type: ignore
    UsdGeom.Xform.Define(stage, prim_path)
    cube = UsdGeom.Cube.Define(stage, prim_path + "/proxy_body")
    cube.GetSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddScaleOp().Set(Gf.Vec3d(float(footprint_xy[0]), float(footprint_xy[1]), float(height)))
    return prim_path


def nominal_g1_rest_offsets() -> list[tuple[str, tuple[float, float, float]]]:
    """Isaac-free nominal G1 rest-pose link offsets (pelvis frame), for the local dry-render skeleton.

    Same shape as :func:`_g1_link_rest_offsets` (which reads the real USD), so it feeds straight into
    :func:`_rest_skeleton_world` + :func:`compute_arm_reach_skeleton`. Approximate dimensions — used to
    preview whether the reaching arm lands in the camera frame, not to claim exact joint geometry.
    """
    return [(name, (float(o[0]), float(o[1]), float(o[2]))) for name, o in _NOMINAL_G1_REST_OFFSETS]


def _arm_link_points_by_arm_from_skeleton(
    skeleton_world: Sequence[tuple[str, Sequence[float]]],
    *,
    arm: str = "both",
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Extract shoulder/elbow/wrist/hand role points from a skeleton-world list.

    Used by the local dry-render path with the nominal Isaac-free G1 skeleton. The GPU path keeps
    using USD link geometry from the actual bound robot.
    """
    sides = _required_manipulation_arms(arm)
    out: dict[str, dict[str, tuple[float, float, float]]] = {side: {} for side in sides}
    role_tokens = {
        "shoulder": ("shoulder",),
        "elbow": ("elbow",),
        "wrist": ("wrist",),
        "hand": ("hand", "palm", "gripper"),
    }
    for name, point in skeleton_world:
        low = str(name).lower()
        for side in sides:
            if not low.startswith(f"{side}_"):
                continue
            for role, tokens in role_tokens.items():
                if role in out[side]:
                    continue
                if any(token in low for token in tokens):
                    out[side][role] = (
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                    )
                    break
    return out


def _manipulation_pov_camera_meta_for_sidecar(
    *,
    base_meta: Mapping[str, Any] | None,
    eye: Sequence[float],
    target: Sequence[float],
    vfov_deg: float,
    width: int,
    height: int,
    arm: str,
    arm_points: Mapping[str, Sequence[float]] | None,
    arm_points_by_arm: Mapping[str, Mapping[str, Sequence[float]]] | None,
) -> dict[str, Any]:
    arm_points_json = {
        key: [round(float(v), 6) for v in value]
        for key, value in sorted((arm_points or {}).items())
    }
    arm_points_by_arm_json = {
        side: {
            key: [round(float(v), 6) for v in value]
            for key, value in sorted(points.items())
        }
        for side, points in sorted((arm_points_by_arm or {}).items())
    }
    return {
        **dict(base_meta or {}),
        "camera_eye_xyz": [round(float(v), 6) for v in eye],
        "camera_target_xyz": [round(float(v), 6) for v in target],
        "camera_vfov_deg": round(float(vfov_deg), 6),
        "viewport_size_px": [int(width), int(height)],
        "required_arms": list(_required_manipulation_arms(arm)),
        "arm_link_points_used": sorted(arm_points_json),
        "arm_link_points_xyz": arm_points_json,
        "arm_link_points_by_arm_xyz": arm_points_by_arm_json,
        "claim_boundary": (
            "Camera metadata is emitted so action-conditioning bridges can project policy "
            "actions against the same seed geometry used for local manipulation POV screening. "
            "It is seed-conditioning evidence, not contact, IK, task-success, or physical "
            "robot proof."
        ),
    }


def _facing_error_deg(root_pose, yaw: float, target) -> float | None:
    """Angle (deg) between where the robot faces and the XY direction from the root to the target."""
    if target is None:
        return None
    dx = float(target[0]) - float(root_pose[0])
    dy = float(target[1]) - float(root_pose[1])
    if math.hypot(dx, dy) < 1e-6:
        return 0.0
    want = math.atan2(dy, dx)
    err = math.degrees(abs((want - float(yaw) + math.pi) % (2 * math.pi) - math.pi))
    return round(err, 3)


def _dry_render_checks(summary) -> dict[str, bool]:
    """Boolean pass/fail for the dry-render checklist. Centralized so the drawn labels and any callers
    agree — and so a perfect ``0.0`` facing error is not swallowed by a ``0.0 or default`` falsy trap."""
    chk = summary.get("checks", {}) or {}
    pf = summary.get("pov_framing", {}) or {}
    st = summary.get("stance", {}) or {}
    rv = summary.get("robot_visual_geometry", {}) or {}
    pov_geom = summary.get("manipulation_pov_geometry", {}) or {}
    fe = chk.get("facing_error_deg")
    pitch_down = chk.get("camera_pitch_down_deg")
    return {
        "faces_target": fe is not None and float(fe) < 8.0,
        "target_in_frame": bool(pf.get("target_in_frame")),
        "arm_in_frame": int(pf.get("arm_landmarks_in_frame") or 0) >= 1,
        "no_blockers": not st.get("blockers"),
        "robot_visual_mesh_present": bool(rv.get("renderable_robot_geometry_present", True)),
        "camera_pitch_within_cap": (
            pitch_down is None
            or float(pitch_down) <= float(MANIPULATION_POV_MAX_CAMERA_PITCH_DOWN_DEG)
        ),
        "pov_geometry_pass": not pov_geom or pov_geom.get("status") == "PASS",
    }


def _dry_render_provenance() -> dict[str, str]:
    return {
        DRY_RENDER_SOURCE_HEADER: DRY_RENDER_SOURCE_MARKER,
        DRY_RENDER_NOTE_HEADER: DRY_RENDER_NOT_RENDERED_NOTE,
        "render_source": DRY_RENDER_SOURCE_MARKER,
        "render_source_note": DRY_RENDER_NOT_RENDERED_NOTE,
        "claim_boundary": (
            "CPU dry-render previews are local stance/camera/projection support artifacts. They are "
            "not Isaac RTX frames, simulator execution proof, task success, physical reach proof, "
            "safety validation, deployment approval, or raw capture truth."
        ),
    }


def _draw_dry_render_preview(
    path,
    *,
    scenario,
    stance_plan,
    root_pose,
    yaw: float,
    look_at,
    eye,
    target,
    pov_vfov_deg: float,
    width: int,
    height: int,
    skeleton_world,
    scene_objects,
    arm: str,
    summary,
) -> None:
    """Draw the 3-panel dry-render preview PNG: top-down placement, egocentric POV framing, summary."""
    from PIL import Image, ImageDraw, PngImagePlugin  # local: only the dry-render path needs PIL

    PW, PH, GUT, TOP = 460, 420, 18, 34
    W = GUT + 3 * (PW + GUT)
    H = TOP + PH + GUT
    img = Image.new("RGB", (W, H), (22, 24, 28))
    d = ImageDraw.Draw(img)
    panels = [GUT + i * (PW + GUT) for i in range(3)]
    d.text((GUT, 8), f"DRY RENDER (no GPU) - {scenario.get('description') or scenario.get('instruction') or ''}"[:120],
           fill=(235, 235, 235))
    for x0 in panels:
        d.rectangle([x0, TOP, x0 + PW, TOP + PH], outline=(70, 74, 82))

    # ---- Panel A: top-down placement ----
    ax0 = panels[0]
    d.text((ax0 + 6, TOP + 4), "top-down placement", fill=(180, 200, 220))
    xs, ys = [float(root_pose[0]), float(eye[0])], [float(root_pose[1]), float(eye[1])]
    tb = stance_plan.get("task_target_bounds") if isinstance(stance_plan, dict) else None
    if tb:
        bmin, bmax = tb["bbox_min_xyz"], tb["bbox_max_xyz"]
        xs += [bmin[0], bmax[0]]
        ys += [bmin[1], bmax[1]]
    for obj in scene_objects[:60]:
        try:
            xs += [obj.bbox_min[0], obj.bbox_max[0]]
            ys += [obj.bbox_min[1], obj.bbox_max[1]]
        except Exception:  # noqa: BLE001
            continue
    if look_at is not None:
        xs.append(float(look_at[0]))
        ys.append(float(look_at[1]))
    pad = 0.6
    minx, maxx, miny, maxy = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    span = max(maxx - minx, maxy - miny, 1e-3)
    inner = PH - 36

    def w2s(wx, wy):
        sx = ax0 + 14 + (float(wx) - minx) / span * inner
        sy = TOP + 24 + (maxy - float(wy)) / span * inner  # +y world is up on screen
        return (sx, sy)

    for obj in scene_objects[:60]:
        try:
            p0 = w2s(obj.bbox_min[0], obj.bbox_min[1])
            p1 = w2s(obj.bbox_max[0], obj.bbox_max[1])
            d.rectangle([min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])],
                        outline=(95, 100, 110))
        except Exception:  # noqa: BLE001
            continue
    if tb:
        p0 = w2s(bmin[0], bmin[1])
        p1 = w2s(bmax[0], bmax[1])
        d.rectangle([min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])],
                    outline=(240, 150, 40), width=2)
    # robot footprint (rotated rectangle) + facing arrow
    hx, hy = 0.25, 0.18
    cyaw, syaw = math.cos(yaw), math.sin(yaw)
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    poly = [w2s(root_pose[0] + cx * cyaw - cy * syaw, root_pose[1] + cx * syaw + cy * cyaw)
            for cx, cy in corners]
    d.polygon(poly, outline=(90, 170, 250))
    rc = w2s(root_pose[0], root_pose[1])
    fa = w2s(root_pose[0] + cyaw * 0.5, root_pose[1] + syaw * 0.5)
    d.line([rc, fa], fill=(230, 80, 80), width=2)
    d.ellipse([rc[0] - 3, rc[1] - 3, rc[0] + 3, rc[1] + 3], fill=(90, 170, 250))
    # camera eye + look-at + frustum edges
    ec = w2s(eye[0], eye[1])
    d.ellipse([ec[0] - 3, ec[1] - 3, ec[0] + 3, ec[1] + 3], fill=(80, 220, 130))
    if look_at is not None:
        lc = w2s(look_at[0], look_at[1])
        d.line([ec[0] - 5, ec[1], ec[0] + 5, ec[1]], fill=(240, 150, 40))
        d.line([lc[0] - 5, lc[1], lc[0] + 5, lc[1]], fill=(240, 150, 40))
        d.line([lc[0], lc[1] - 5, lc[0], lc[1] + 5], fill=(240, 150, 40))
        fdx, fdy = float(look_at[0]) - float(eye[0]), float(look_at[1]) - float(eye[1])
        ang = math.atan2(fdy, fdx)
        half = math.radians(pov_vfov_deg * (width / max(height, 1)) / 2.0)
        for s in (-1.0, 1.0):
            a = ang + s * half
            fp = w2s(eye[0] + math.cos(a) * span * 0.5, eye[1] + math.sin(a) * span * 0.5)
            d.line([ec, fp], fill=(60, 130, 90))

    # ---- Panel B: egocentric POV framing ----
    bx0 = panels[1]
    d.text((bx0 + 6, TOP + 4), f"egocentric POV  vfov={pov_vfov_deg:.0f}deg  (arm=APPROX)", fill=(180, 200, 220))
    fx0, fy0 = bx0 + 18, TOP + 28
    fw = PW - 36
    fh = int(fw * height / max(width, 1))
    if fh > PH - 46:
        fh = PH - 46
        fw = int(fh * width / max(height, 1))
    d.rectangle([fx0, fy0, fx0 + fw, fy0 + fh], outline=(120, 125, 135))

    def proj(wp):
        px = project_point_to_pixel(wp, eye, target, (0.0, 0.0, 1.0), pov_vfov_deg, width, height)
        if px is None:
            return None
        return (fx0 + px[0] / width * fw, fy0 + px[1] / height * fh)

    for obj in scene_objects[:60]:
        try:
            sp = proj(obj.footprint_center() + (float(0.5 * (obj.bbox_min[2] + obj.bbox_max[2])),))
        except Exception:  # noqa: BLE001
            sp = None
        if sp:
            d.ellipse([sp[0] - 1.5, sp[1] - 1.5, sp[0] + 1.5, sp[1] + 1.5], fill=(110, 115, 125))
    sk = {n: p for n, p in skeleton_world}
    sides = ("right", "left") if str(arm) == "both" else (str(arm),)
    for side in sides:
        chain = [f"{side}_shoulder_link", f"{side}_elbow_link", f"{side}_wrist_link", f"{side}_hand_link"]
        prev = None
        for i, name in enumerate(chain):
            wp = sk.get(name)
            sp = proj(wp) if wp is not None else None
            if sp is None:
                prev = None
                continue
            if prev is not None:
                d.line([prev, sp], fill=(90, 170, 250), width=2)
            r = 5 if "hand" in name else 3
            d.ellipse([sp[0] - r, sp[1] - r, sp[0] + r, sp[1] + r], fill=(120, 200, 255))
            prev = sp
    if look_at is not None:
        lp = proj((float(look_at[0]), float(look_at[1]), float(look_at[2])))
        if lp:
            d.line([lp[0] - 7, lp[1], lp[0] + 7, lp[1]], fill=(240, 150, 40), width=2)
            d.line([lp[0], lp[1] - 7, lp[0], lp[1] + 7], fill=(240, 150, 40), width=2)

    # ---- Panel C: summary + checklist ----
    cx0 = panels[2] + 8
    d.text((cx0, TOP + 4), "summary", fill=(180, 200, 220))
    chk = summary.get("checks", {})
    pf = summary.get("pov_framing", {})
    st = summary.get("stance", {})
    ok = _dry_render_checks(summary)

    def mark(flag):
        return "OK" if flag else "XX"

    lines = [
        f"stance: {st.get('status')}",
        f"pose:  {['%.2f' % v for v in (st.get('accepted_pose') or [])]}",
        f"yaw:   {st.get('accepted_yaw')}",
        f"look_at: {['%.2f' % v for v in (summary.get('look_at') or [])]}",
        f"standoff_gap_m: {chk.get('standoff_gap_m')}",
        "",
        f"[{mark(ok['faces_target'])}] faces target  ({chk.get('facing_error_deg')} deg)",
        f"[{mark(ok['target_in_frame'])}] target in POV frame",
        f"[{mark(ok['arm_in_frame'])}] arm in POV frame  ({pf.get('arm_landmarks_in_frame')})",
        f"[{mark(ok['no_blockers'])}] no stance blockers",
        f"[{mark(ok['robot_visual_mesh_present'])}] robot visual mesh present",
        f"[{mark(ok['camera_pitch_within_cap'])}] camera pitch within cap",
        f"[{mark(ok['pov_geometry_pass'])}] POV geometry gate",
    ]
    for blk in (st.get("blockers") or [])[:4]:
        lines.append(f"   blocker: {blk}")
    yy = TOP + 26
    for ln in lines:
        d.text((cx0, yy), ln[:60], fill=(225, 225, 225))
        yy += 18

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    png_info = PngImagePlugin.PngInfo()
    provenance = summary.get("render_provenance", {}) or _dry_render_provenance()
    png_info.add_text(DRY_RENDER_SOURCE_HEADER, str(provenance.get(DRY_RENDER_SOURCE_HEADER)))
    png_info.add_text(DRY_RENDER_NOTE_HEADER, str(provenance.get(DRY_RENDER_NOTE_HEADER)))
    img.save(str(path), pnginfo=png_info)


def render_local_preview(
    *,
    stage,
    scenario,
    out_dir,
    manipulation_reach_arm: str = "right",
    camera_vfov_deg: float = 50.0,
    width: int = 1280,
    height: int = 960,
    manipulation_look_at=None,
    robot_prim_path: str | None = None,
    robot_visual_prim_path: str | None = None,
    robot_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce a no-GPU dry-render preview (PNG + summary JSON) for a task on an open USD stage.

    Reproduces the GPU runner's stance plan, manipulation camera, and arms-forward seed skeleton, then draws
    a top-down + egocentric-POV + checklist preview so wrong-side stance / cropped-arm / mis-aimed
    camera show up locally in <1s. ``robot_prim_path`` (when a placeable robot proxy is bound) enables
    the geometric placement validator; otherwise stance is planned without obstacle rejection.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = str(scenario.get("scenario_id") or scenario.get("episode_id") or "scenario")

    stance_plan = _plan_task_stance_for_stage(
        stage=stage,
        scenario=scenario,
        manipulation_look_at=manipulation_look_at,
        probe=lambda pose, yaw: 0,
        no_collision_probe=False,
        robot_prim_path=robot_prim_path,
    )
    (out_dir / "task_stance_plan.json").write_text(json.dumps(stance_plan, indent=2))

    summary: dict[str, Any] = {
        "scenario_id": sid,
        "task": scenario.get("description") or scenario.get("instruction") or scenario.get("task"),
        "manipulation_reach_arm": manipulation_reach_arm,
        "render_source": DRY_RENDER_SOURCE_MARKER,
        "render_source_headers": {
            DRY_RENDER_SOURCE_HEADER: DRY_RENDER_SOURCE_MARKER,
            DRY_RENDER_NOTE_HEADER: DRY_RENDER_NOT_RENDERED_NOTE,
        },
        "render_provenance": _dry_render_provenance(),
        "stance": {
            "status": stance_plan.get("status"),
            "blockers": stance_plan.get("blockers"),
            "accepted_pose": stance_plan.get("accepted_pose"),
            "accepted_yaw": stance_plan.get("accepted_yaw"),
        },
        "claim_boundary": (
            "Local dry-render preview: stance/camera/projection are exact; the reaching arm uses a "
            "nominal (Isaac-free) G1, so arm framing is approximate, not a manipulation-success claim."
        ),
    }

    try:
        scene_objects = _scene_objects_for_stage(stage)
    except Exception:  # noqa: BLE001 - the preview must not hard-fail on scene enumeration
        scene_objects = []
    summary["scene"] = {
        "object_count": len(scene_objects),
        "full_scene_check_source": "scene_placement_usd_objects",
        "claim_boundary": (
            "Object count is a local USD scene-index sanity check. It proves the dry-render opened "
            "a non-empty task scene, not that all meshes/textures are complete or physically valid."
        ),
    }

    placement_validation = stance_plan.get("placement_validation")
    if not isinstance(placement_validation, Mapping):
        placement_validation = {
            "schema_version": "placement_validation.v1",
            "status": "blocked",
            "blockers": ["placement_validation_unavailable_in_task_stance_plan"],
            "claim_boundary": (
                "Placement validation must be emitted by the task stance planner before this "
                "scenario can pass local task-scaling preflight."
            ),
        }
    elif placement_validation.get("status") == "accepted":
        placement_validation = {
            **dict(placement_validation),
            "status": "PASS",
            "raw_status": "accepted",
            "normalized_status_source": "dry_render_task_stance_candidate_validation",
        }
    (out_dir / "placement_validation.json").write_text(json.dumps(placement_validation, indent=2))

    if stance_plan.get("status") != "accepted":
        summary["pov_framing"] = {"target_in_frame": False, "arm_landmarks_in_frame": 0,
                                  "projected_landmark_count": 0}
        summary["checks"] = {"facing_error_deg": None, "standoff_gap_m": None}
        manipulation_pov_geometry = {
            "schema_version": "manipulation_pov_geometry.v1",
            "status": "FAIL",
            "blockers": ["task_stance_not_accepted"],
            "target_in_frame": False,
            "claim_boundary": (
                "Manipulation POV geometry is not evaluated unless a task stance is accepted."
            ),
        }
        (out_dir / "manipulation_pov_geometry.json").write_text(
            json.dumps(manipulation_pov_geometry, indent=2)
        )
        (out_dir / "dry_render_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    root = tuple(float(v) for v in stance_plan["accepted_pose"])
    yaw = float(stance_plan["accepted_yaw"])
    look_at = manipulation_look_at
    if look_at is None:
        look_at = _surface_affordance_point_for_stance(stance_plan, root) or stance_plan.get("task_target_xyz")
    look_at = tuple(float(v) for v in look_at) if look_at is not None else None

    pov_vfov_deg = max(
        float(camera_vfov_deg),
        float(MANIPULATION_POV_MIN_VFOV_DEG),
    )  # manipulation widen — mirrors the GPU runner
    visual_prim_path = robot_visual_prim_path or robot_prim_path
    if visual_prim_path and stage.GetPrimAtPath(visual_prim_path).IsValid():
        _set_root_xform(stage, visual_prim_path, root, yaw)
    robot_camera_meta: dict[str, Any] = {"source": "nominal_root_yaw_fallback"}
    if visual_prim_path:
        robot_diag = _robot_render_visibility_diagnostics(stage, visual_prim_path)
        summary["robot_visual_geometry"] = {
            "status": robot_diag.get("status"),
            "blockers": robot_diag.get("blockers", []),
            "gprim_count": robot_diag.get("gprim_count"),
            "mesh_count": robot_diag.get("mesh_count"),
            "renderable_robot_geometry_present": bool(
                robot_diag.get("renderable_robot_geometry_present")
            ),
            "visual_binding_status": (
                (robot_binding or {}).get("visual_binding_status")
                if robot_binding is not None else "dry_render_proxy_robot"
            ),
            "diagnostics": robot_diag,
            "claim_boundary": (
                "Robot visual diagnostics prove only whether the local USD subtree exposes renderable "
                "Gprim/Mesh surfaces for review media. They do not prove physical geometry, contact, "
                "policy success, or live robot readiness."
            ),
        }
    if robot_binding is not None and visual_prim_path:
        eye, tgt, robot_camera_meta = _robot_mounted_manipulation_cam_pose(
            stage,
            visual_prim_path,
            root,
            yaw,
            look_at=look_at,
            reach_arm=manipulation_reach_arm,
            vfov_deg=pov_vfov_deg,
            width=width,
            height=height,
        )
    else:
        eye, tgt = manipulation_cam_pose(
            root,
            yaw,
            look_at=look_at,
            reach_arm=manipulation_reach_arm,
        )
    if look_at is not None:
        capped_tgt = _target_raised_to_max_pitch_down(
            eye,
            tgt,
            MANIPULATION_POV_MAX_CAMERA_PITCH_DOWN_DEG,
        )
        if capped_tgt != tgt:
            robot_camera_meta = {
                **robot_camera_meta,
                "pitch_cap_applied": True,
                "uncapped_pov_target": [round(float(v), 6) for v in tgt],
                "max_pitch_down_deg": float(MANIPULATION_POV_MAX_CAMERA_PITCH_DOWN_DEG),
            }
            tgt = capped_tgt

    skeleton_world = _rest_skeleton_world(nominal_g1_rest_offsets(), root, yaw)
    if look_at is not None:
        skeleton_world = compute_arm_reach_skeleton(
            skeleton_world,
            look_at,
            1.0,
            arm=manipulation_reach_arm,
            forward_yaw=yaw,
        )
    pov_lms = _project_skeleton(skeleton_world, eye=eye, target=tgt, up=(0.0, 0.0, 1.0),
                                vfov_deg=pov_vfov_deg, width=width, height=height)

    active = ("right", "left") if str(manipulation_reach_arm) == "both" else (str(manipulation_reach_arm),)
    arm_in_frame = sum(
        1 for lm in pov_lms
        if any(lm["landmark_id"].startswith(f"{s}_") and any(k in lm["landmark_id"] for k in ("hand", "wrist", "elbow"))
               for s in active)
    )
    target_px = (project_point_to_pixel(look_at, eye, tgt, (0.0, 0.0, 1.0), pov_vfov_deg, width, height)
                 if look_at is not None else None)

    summary["look_at"] = list(look_at) if look_at is not None else None
    _target_center = stance_plan.get("task_target_xyz")
    _target_bounds = stance_plan.get("task_target_bounds")
    summary["target"] = {
        "center_xyz": list(_target_center) if _target_center is not None else None,
        "bbox_min_xyz": (_target_bounds or {}).get("bbox_min_xyz"),
        "bbox_max_xyz": (_target_bounds or {}).get("bbox_max_xyz"),
        "resolution_source": (stance_plan.get("target_resolution") or {}).get("source"),
    }
    summary["camera"] = {
        "pov_eye": [round(float(v), 4) for v in eye],
        "pov_target": [round(float(v), 4) for v in tgt],
        "pov_vfov_deg": round(float(pov_vfov_deg), 3),
        "pov_pitch_down_deg": round(float(_camera_pitch_down_deg(eye, tgt)), 3),
        "source": robot_camera_meta.get("source"),
        "metadata": robot_camera_meta,
    }
    summary["pov_framing"] = {
        "target_in_frame": target_px is not None,
        "arm_landmarks_in_frame": int(arm_in_frame),
        "projected_landmark_count": len(pov_lms),
    }
    summary["checks"] = {
        "facing_error_deg": _facing_error_deg(root, yaw, look_at),
        "camera_pitch_down_deg": round(float(_camera_pitch_down_deg(eye, tgt)), 3),
        "standoff_gap_m": (stance_plan.get("candidates") or [{}])[stance_plan.get("selected_candidate_index", 0)]
        .get("standoff_from_target_surface_m"),
    }
    if robot_binding is not None and visual_prim_path and look_at is not None:
        reach_selection = _normalize_reach_arm_selection(manipulation_reach_arm)
        arm_points_by_arm = _robot_arm_link_points_by_arm(
            stage,
            visual_prim_path,
            arm=reach_selection,
        )
        if reach_selection == "both":
            arm_points = _average_arm_link_points(arm_points_by_arm)
        else:
            arm_points = dict(arm_points_by_arm.get(reach_selection) or {})
        summary["manipulation_pov_geometry"] = _manipulation_pov_geometry(
            arm_points=arm_points,
            arm_points_by_arm=arm_points_by_arm,
            affordance=look_at,
            eye=eye,
            target=tgt,
            vfov_deg=pov_vfov_deg,
            width=width,
            height=height,
            arm=reach_selection,
        )
        summary["manipulation_pov_geometry"]["camera_meta"] = (
            _manipulation_pov_camera_meta_for_sidecar(
                base_meta=robot_camera_meta,
                eye=eye,
                target=tgt,
                vfov_deg=pov_vfov_deg,
                width=width,
                height=height,
                arm=reach_selection,
                arm_points=arm_points,
                arm_points_by_arm=arm_points_by_arm,
            )
        )
    if not isinstance(summary.get("manipulation_pov_geometry"), Mapping):
        reach_selection = _normalize_reach_arm_selection(manipulation_reach_arm)
        nominal_arm_points_by_arm = _arm_link_points_by_arm_from_skeleton(
            skeleton_world,
            arm=reach_selection,
        )
        if reach_selection == "both":
            nominal_arm_points = _average_arm_link_points(nominal_arm_points_by_arm)
        else:
            nominal_arm_points = dict(nominal_arm_points_by_arm.get(reach_selection) or {})
        summary["manipulation_pov_geometry"] = _manipulation_pov_geometry(
            arm_points=nominal_arm_points,
            arm_points_by_arm=nominal_arm_points_by_arm,
            affordance=look_at,
            eye=eye,
            target=tgt,
            vfov_deg=pov_vfov_deg,
            width=width,
            height=height,
            arm=reach_selection,
        )
        summary["manipulation_pov_geometry"]["camera_meta"] = (
            _manipulation_pov_camera_meta_for_sidecar(
                base_meta=robot_camera_meta,
                eye=eye,
                target=tgt,
                vfov_deg=pov_vfov_deg,
                width=width,
                height=height,
                arm=reach_selection,
                arm_points=nominal_arm_points,
                arm_points_by_arm=nominal_arm_points_by_arm,
            )
        )
        summary["manipulation_pov_geometry"]["geometry_source"] = "nominal_g1_dry_render_skeleton"
        summary["manipulation_pov_geometry"]["claim_boundary"] = (
            "Local preview uses a nominal Isaac-free G1 skeleton for hand/wrist/extension "
            "screening. It is not real USD link geometry, physical reach proof, contact proof, "
            "or deployment readiness; paid Isaac/GPU remains the strict geometry authority."
        )
    (out_dir / "manipulation_pov_geometry.json").write_text(
        json.dumps(summary["manipulation_pov_geometry"], indent=2)
    )

    _draw_dry_render_preview(
        out_dir / "dry_render_preview.png",
        scenario=scenario, stance_plan=stance_plan, root_pose=root, yaw=yaw, look_at=look_at,
        eye=eye, target=tgt, pov_vfov_deg=pov_vfov_deg, width=width, height=height,
        skeleton_world=skeleton_world, scene_objects=scene_objects, arm=manipulation_reach_arm,
        summary=summary,
    )
    (out_dir / "dry_render_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _import_render_noise_audit_module():
    """The pure audit module (variant plan + stats + manifests): bundle flat copy on the
    worker, package import in the repo/tests, None when neither resolves."""
    try:
        import g1_render_noise_audit as audit_mod  # type: ignore  # bundle (worker)
        return audit_mod
    except Exception:  # noqa: BLE001
        try:
            from blueprint_pipeline import g1_render_noise_audit as audit_mod  # repo (tests)
            return audit_mod
        except Exception:  # noqa: BLE001
            return None


def run_render_noise_audit(*, kitchen_usd: str, g1_usd: str, scenario: Mapping[str, Any],
                           out_dir: Path, width: int, height: int,
                           camera_vfov_deg: float = 50.0, reach_arm: str = "both",
                           warmup_frames: int = DEFAULT_AUDIT_WARMUP_FRAMES,
                           render_subframes: int = 16,
                           variant_plan: Mapping[str, Any] | None = None,
                           high_samples_per_pixel: int | None = None,
                           boost_light_intensity: float | None = None,
                           fill_light_intensity: float = 0.0,
                           neutral_environment: bool = False,
                           no_collision_probe: bool = True,
                           per_variant_seconds: float = 420.0) -> dict[str, Any]:
    """GPU orchestration for the textured-robot render-noise audit (spec matrix A-G).

    One dynamic scene/stance/arm-pose/camera setup (the SAME task-resolution and placement
    code as the normal Isaac seed render — no hardcoded scene coordinates), then one raw PNG
    per variant while changing only the declared material/render levers. Everything needed to
    interpret the frames is recorded: material/texture resolution, applied + effective render
    settings, camera contract, lighting inventory, GPU/driver/Isaac identity, and timings.
    Honesty boundary: render-quality evidence only — not task success, policy, or readiness.
    """
    out_dir = Path(out_dir)
    audit_dir = out_dir / RENDER_NOISE_AUDIT_DIR_NAME
    audit_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / RENDER_NOISE_AUDIT_RESULT_NAME
    sid = str(scenario.get("scenario_id") or "render_noise_audit")
    task = str(
        scenario.get("task")
        or scenario.get("instruction")
        or scenario.get("description")
        or ""
    )
    plan = dict(variant_plan) if variant_plan else render_noise_audit_plan_from_request({})
    variants_by_id = {str(v.get("variant_id")): dict(v) for v in plan.get("variants") or []}
    execution_order = [vid for vid in (plan.get("execution_order") or []) if vid in variants_by_id]
    result: dict[str, Any] = {
        "schema_version": RENDER_NOISE_AUDIT_RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "scenario_id": sid,
        "task": task or None,
        "blockers": [],
        "variants_planned": sorted(variants_by_id),
        "variants_rendered": [],
        "audit_dir": str(audit_dir),
        "claim_boundary": (
            "Render-noise audit evidence only: frames + manifests describe renderer/material/"
            "lighting behavior. Not task success, policy quality, physical readiness, or WAM "
            "rank fidelity."
        ),
    }

    def _write_result() -> None:
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def _write_audit_json(name: str, payload: Mapping[str, Any]) -> None:
        (audit_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not execution_order:
        result["blockers"].append("render_noise_audit_variant_plan_empty")
        _write_result()
        return result

    _log("render-noise audit: booting Isaac (headless RTX) ...")
    sim = _boot_sim(headless=True)
    try:
        rep = _enable_and_import_replicator()
        _log(f"render-noise audit: opening kitchen USD: {kitchen_usd}")
        stage = _open_stage(_resolve_asset_uri(kitchen_usd))
        binding = _bind_g1_with_visual_fallback(stage, g1_usd)
        (audit_dir / "g1_binding.json").write_text(json.dumps(binding, indent=2))
        robot_prim_path = binding["prim_path"]
        robot_asset = {
            "requested_g1_usd": g1_usd,
            "resolved_visual_asset": binding.get("candidate_g1_usd") or binding.get("resolved_g1_usd"),
            "visual_binding_status": binding.get("visual_binding_status"),
            "visual_candidate_attempts": binding.get("visual_candidate_attempts", []),
        }

        if no_collision_probe:
            def probe(pose, yaw):  # noqa: ANN001
                return 0
            probe_source = "no_probe_geometry_validation_only"
        else:
            probe = _overlap_probe(robot_prim_path)
            probe_source = "physx_overlap_probe"

        stance_plan = _plan_task_stance_for_stage(
            stage=stage,
            scenario=scenario,
            manipulation_look_at=None,
            probe=probe,
            no_collision_probe=no_collision_probe,
            robot_prim_path=robot_prim_path,
        )
        stance_plan["collision_probe_source"] = probe_source
        _write_audit_json("task_stance_plan.json", stance_plan)
        placement_validation = stance_plan.get("placement_validation")
        if not isinstance(placement_validation, Mapping):
            placement_validation = {
                "schema_version": "placement_validation.v1",
                "status": "blocked",
                "blockers": ["placement_validation_unavailable_in_task_stance_plan"],
            }
        _write_audit_json("placement_validation.json", dict(placement_validation))
        if stance_plan.get("status") != "accepted":
            result["blockers"].append("task_stance_plan_not_accepted")
            result["stance_blockers"] = stance_plan.get("blockers")
            _write_result()
            return result

        root = tuple(float(v) for v in stance_plan["accepted_pose"])
        yaw = float(stance_plan["accepted_yaw"])
        look_at = _surface_affordance_point_for_stance(stance_plan, root) or stance_plan.get("task_target_xyz")
        look_at = tuple(float(v) for v in look_at) if look_at is not None else None
        if look_at is None:
            result["blockers"].append("task_affordance_or_target_unresolved")
            _write_result()
            return result

        robot_neutral_xforms = _capture_robot_neutral_descendant_xforms(stage, robot_prim_path)
        if robot_neutral_xforms:
            _restore_robot_neutral_descendant_xforms(stage, robot_neutral_xforms)
        root_diagnostics = _place_root(stage, robot_prim_path, root, yaw)
        _write_audit_json("place_root_diagnostics.json", root_diagnostics)
        reach_selection = _normalize_reach_arm_selection(reach_arm)
        posed_count = 0
        try:
            posed_count = _pose_arm_kinematic_usd(
                stage, robot_prim_path, look_at,
                arm=reach_selection, reach_frac=1.0, forward_yaw=yaw,
            )
        except Exception as exc:  # noqa: BLE001
            result["blockers"].append("kinematic_arm_pose_failed")
            result["kinematic_arm_pose_error"] = repr(exc)
        _log(f"render-noise audit: stance accepted, arm pose links={posed_count}")

        robot_render_diag = _robot_render_visibility_diagnostics(stage, robot_prim_path)
        _write_audit_json("robot_render_diagnostics.json", robot_render_diag)

        material_raw = _collect_robot_material_resolution(
            stage, robot_prim_path,
            robot_asset_uri=g1_usd,
            resolved_visual_asset=robot_asset.get("resolved_visual_asset"),
        )
        audit_mod = _import_render_noise_audit_module()
        if audit_mod is not None:
            material_resolution = audit_mod.summarize_material_resolution(material_raw)
        else:
            material_resolution = {
                "schema_version": "robot_material_resolution_manifest.v1",
                "status": "raw_only_normalizer_unavailable",
                **material_raw,
            }
        _write_audit_json("robot_material_resolution_manifest.json", material_resolution)

        from pxr import UsdGeom  # type: ignore

        UsdGeom.Camera.Define(stage, RENDER_NOISE_AUDIT_CAM_PATH)
        pov_vfov_deg = max(float(camera_vfov_deg), float(MANIPULATION_POV_MIN_VFOV_DEG))
        _set_camera_fov(stage, RENDER_NOISE_AUDIT_CAM_PATH, pov_vfov_deg, width, height)
        eye, tgt, cam_meta = _robot_mounted_manipulation_cam_pose(
            stage, robot_prim_path, root, yaw,
            look_at=look_at, reach_arm=reach_selection,
            vfov_deg=pov_vfov_deg, width=width, height=height,
        )
        capped_tgt = _target_raised_to_max_pitch_down(
            eye, tgt, MANIPULATION_POV_MAX_CAMERA_PITCH_DOWN_DEG,
        )
        if capped_tgt != tgt:
            cam_meta = {**cam_meta, "pitch_cap_applied": True}
            tgt = capped_tgt
        _place_camera(stage, RENDER_NOISE_AUDIT_CAM_PATH, eye, tgt)

        # Constant scene lighting matches the normal manipulation seed path: camera-side
        # headlamp + optional workspace fill + optional neutral dome. Variant G adds ONE
        # extra recorded boost light on top; nothing else changes per variant.
        _add_pov_headlamp(
            stage, eye, look_at,
            intensity=(fill_light_intensity if fill_light_intensity > 0 else 30000.0),
        )
        if fill_light_intensity > 0:
            _add_workspace_fill_light(stage, look_at, intensity=fill_light_intensity)
        if neutral_environment:
            try:
                _neutralize_environment(stage)
            except Exception as exc:  # noqa: BLE001
                _log(f"render-noise audit: environment neutralize skipped ({exc!r})")

        camera_contract = _isaac_camera_contract(stage, RENDER_NOISE_AUDIT_CAM_PATH, width, height)
        try:
            clip = UsdGeom.Camera(stage.GetPrimAtPath(RENDER_NOISE_AUDIT_CAM_PATH)).GetClippingRangeAttr().Get()
            camera_contract["clipping_range_m"] = [float(clip[0]), float(clip[1])] if clip else None
        except Exception:  # noqa: BLE001
            camera_contract["clipping_range_m"] = None
        camera_contract["camera_source"] = cam_meta.get("source")
        camera_contract["camera_mount_metadata"] = cam_meta
        camera_contract["pitch_down_deg"] = round(float(_camera_pitch_down_deg(eye, tgt)), 3)
        camera_contract["vfov_deg"] = round(float(pov_vfov_deg), 3)
        camera_contract["eye_xyz"] = [round(float(v), 6) for v in eye]
        camera_contract["target_xyz"] = [round(float(v), 6) for v in tgt]
        camera_contract["look_at_xyz"] = [round(float(v), 6) for v in look_at]
        _write_audit_json("camera_contract.json", camera_contract)

        arm_points_by_arm = _robot_arm_link_points_by_arm(stage, robot_prim_path, arm=reach_selection)
        if reach_selection == "both":
            arm_points = _average_arm_link_points(arm_points_by_arm)
        else:
            arm_points = dict(arm_points_by_arm.get(reach_selection) or {})
        pov_geometry = _manipulation_pov_geometry(
            arm_points=arm_points, arm_points_by_arm=arm_points_by_arm,
            affordance=look_at, eye=eye, target=tgt,
            vfov_deg=pov_vfov_deg, width=width, height=height, arm=reach_selection,
        )
        _write_audit_json("manipulation_pov_geometry.json", pov_geometry)
        arm_visibility = audit_arm_visibility_from_pov_geometry(pov_geometry)

        try:
            _author_scene_semantic_labels(
                stage, robot_prim_path=robot_prim_path,
                keep_substrings=("room", "floor", "wall", "ground", "ceiling", "light"),
            )
        except Exception as exc:  # noqa: BLE001
            _log(f"render-noise audit: semantic label authoring skipped ({exc!r})")
        annots = _make_render_product(
            RENDER_NOISE_AUDIT_CAM_PATH, width, height,
            with_depth=False, with_segmentation=True,
        )
        rgb_annot = annots["rgb"] if isinstance(annots, Mapping) else annots
        instance_annot = annots.get("instance") if isinstance(annots, Mapping) else None

        default_spp = int(_render_quality_config(
            render_subframes=int(render_subframes),
            manipulation_cam=True, verify_cam=False, mode="pathtraced",
        )["samples_per_pixel"])
        env_high = os.getenv("PARITY_AUDIT_HIGH_SAMPLES_PER_PIXEL", "").strip()
        try:
            high_spp = int(high_samples_per_pixel or (int(env_high) if env_high else DEFAULT_AUDIT_HIGH_SAMPLES_PER_PIXEL))
        except ValueError:
            high_spp = DEFAULT_AUDIT_HIGH_SAMPLES_PER_PIXEL
        high_spp = max(default_spp, min(512, high_spp))
        boost_intensity = float(boost_light_intensity or DEFAULT_AUDIT_BOOST_LIGHT_INTENSITY)

        lighting_summary = _scene_lighting_summary(stage)
        runtime_metadata = _audit_runtime_metadata()
        render_settings_manifest = {
            "schema_version": "render_settings_manifest.v1",
            "renderer": "rtx_pathtracing",
            "resolution": [int(width), int(height)],
            "render_subframes_requested": int(render_subframes),
            "replicator_rt_subframes_per_capture": 1,
            "default_budget_samples_per_pixel": default_spp,
            "high_budget_samples_per_pixel": high_spp,
            "firefly_filter_constant_enabled": True,
            "software_denoise_applied": False,
            "boost_light_intensity_variant_g": boost_intensity,
            "shader_warmup_frames": int(warmup_frames),
            "per_variant_settle_frames": DEFAULT_AUDIT_PER_VARIANT_SETTLE_FRAMES,
            "lighting_summary": lighting_summary,
            "runtime_metadata": runtime_metadata,
        }
        _write_audit_json("render_settings_manifest.json", render_settings_manifest)

        # Shader/RTX-cache warmup BEFORE the first variant so cold-start compile/denoiser
        # variance (hypothesis 6) is spent here and measured, not inside variant A/B.
        warmup_settings = _apply_audit_render_settings(
            rep, samples_per_pixel=default_spp, denoiser_enabled=True,
        )
        warmup_seconds: list[float] = []
        for wi in range(max(0, int(warmup_frames))):
            t_warm = time.time()
            _replicator_step_with_watchdog(
                rep, label=f"{sid}:audit:warmup:{wi}",
                result_path=result_path, scenario_id=sid, rt_subframes=1,
                timeout_seconds=_audit_render_step_watchdog_seconds(),
            )
            warmup_seconds.append(round(time.time() - t_warm, 3))
        _log(f"render-noise audit: warmup done ({[f'{s:.1f}' for s in warmup_seconds]})")

        variant_results: list[dict[str, Any]] = []
        current_material = "textured_original"
        material_apply_records: dict[str, Any] = {}
        last_rank = -1
        for vid in execution_order:
            variant = variants_by_id[vid]
            material = str(variant.get("robot_material") or "textured_original")
            rank = _AUDIT_MATERIAL_MONOTONIC_RANK.get(material, 3)
            record: dict[str, Any] = {
                "variant_id": vid,
                "variant": variant,
                "execution_index": len(variant_results),
            }
            if rank < last_rank:
                # A custom plan asked to go back to a less-overridden material; the
                # monotonic override strategy cannot un-author that, so skip honestly.
                record["status"] = "skipped"
                record["blockers"] = ["variant_execution_order_material_not_monotonic"]
                variant_results.append(record)
                result["blockers"].append(f"variant_skipped_non_monotonic_material:{vid}")
                continue
            t_variant = time.time()
            if material != current_material:
                if material == "simplified_diffuse":
                    material_apply_records["simplified_diffuse"] = (
                        _apply_robot_simplified_diffuse_material(stage, robot_prim_path)
                    )
                elif material == "white_proxy":
                    material_apply_records["white_proxy"] = {
                        "gprims_bound": _apply_robot_review_material(
                            stage, robot_prim_path, override_authored_materials=True,
                        ),
                    }
                current_material = material
                last_rank = rank
            record["material_apply"] = material_apply_records.get(material)
            spp = audit_samples_per_pixel(
                str(variant.get("render_budget") or "current_default"),
                default_spp=default_spp, high_spp=high_spp,
            )
            record["render_settings"] = _apply_audit_render_settings(
                rep, samples_per_pixel=spp,
                denoiser_enabled=bool(variant.get("denoiser_enabled")),
            )
            if variant.get("lighting_boost"):
                _add_workspace_fill_light(
                    stage, look_at, intensity=boost_intensity,
                    path=RENDER_NOISE_AUDIT_BOOST_LIGHT_PATH,
                )
                record["boost_light"] = {
                    "path": RENDER_NOISE_AUDIT_BOOST_LIGHT_PATH,
                    "intensity": boost_intensity,
                }
            try:
                for si in range(DEFAULT_AUDIT_PER_VARIANT_SETTLE_FRAMES):
                    if time.time() - t_variant > per_variant_seconds:
                        record.setdefault("blockers", []).append("variant_time_cap_hit_in_settle")
                        break
                    _replicator_step_with_watchdog(
                        rep, label=f"{sid}:audit:{vid}:settle:{si}",
                        result_path=result_path, scenario_id=sid, rt_subframes=1,
                        timeout_seconds=_audit_render_step_watchdog_seconds(),
                    )
                t_capture = time.time()
                _replicator_step_with_watchdog(
                    rep, label=f"{sid}:audit:{vid}:capture",
                    result_path=result_path, scenario_id=sid, rt_subframes=1,
                    timeout_seconds=_audit_render_step_watchdog_seconds(),
                )
                record["capture_seconds"] = round(time.time() - t_capture, 3)
                variant_dir = audit_dir / "variants" / vid
                variant_dir.mkdir(parents=True, exist_ok=True)
                png_path = variant_dir / "frame_raw.png"
                saved = _save_rgb(rgb_annot, png_path, software_denoise=False)
                record["frame_png"] = f"variants/{vid}/frame_raw.png" if saved else None
                if not saved:
                    record.setdefault("blockers", []).append("variant_frame_capture_empty")
                else:
                    try:
                        record["seed_frame_quality"] = _pov_seed_frame_quality(png_path)
                    except Exception:  # noqa: BLE001
                        pass
                if instance_annot is not None and saved:
                    record["robot_pixel_mask"] = _robot_pixel_ratio_from_instance(
                        instance_annot, robot_prim_path,
                    )
                    try:
                        record["segmentation_save"] = _save_segmentation(
                            annots,
                            instance_png=variant_dir / "robot_instance_mask.png",
                            semantic_png=variant_dir / "semantic_mask.png",
                            id_label_json=variant_dir / "instance_id_labels.json",
                        )
                        record["segmentation_save"].pop("id_to_labels", None)
                    except Exception as exc:  # noqa: BLE001
                        record["segmentation_error"] = repr(exc)
            finally:
                if variant.get("lighting_boost"):
                    _remove_prim_quiet(stage, RENDER_NOISE_AUDIT_BOOST_LIGHT_PATH)
            record["variant_seconds"] = round(time.time() - t_variant, 3)
            record["status"] = "completed" if record.get("frame_png") else "blocked"
            (audit_dir / "variants" / vid / "variant_manifest.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8",
            )
            variant_results.append(record)
            if record["status"] == "completed":
                result["variants_rendered"].append(vid)
            _log(
                f"render-noise audit: variant {vid} {record['status']} "
                f"material={material} spp={spp} denoiser={bool(variant.get('denoiser_enabled'))} "
                f"({record['variant_seconds']}s)"
            )

        target_resolution = stance_plan.get("target_resolution")
        run_manifest = {
            "schema_version": RENDER_NOISE_AUDIT_RUN_SCHEMA_VERSION,
            "scenario_id": sid,
            "task": task or scenario.get("instruction"),
            "scenario": {k: v for k, v in scenario.items() if k != "route_points"},
            "target_resolution": target_resolution,
            "stance_plan_summary": {
                "status": stance_plan.get("status"),
                "accepted_pose": stance_plan.get("accepted_pose"),
                "accepted_yaw": stance_plan.get("accepted_yaw"),
                "task_target_xyz": stance_plan.get("task_target_xyz"),
                "task_target_bounds": stance_plan.get("task_target_bounds"),
                "affordance_focus_source": stance_plan.get("affordance_focus_source"),
                "collision_probe_source": probe_source,
            },
            "placement_validation": dict(placement_validation),
            "robot_asset": robot_asset,
            "robot_render_diagnostics_status": robot_render_diag.get("status"),
            "posed_arm_link_count": int(posed_count),
            "reach_arm": reach_selection,
            "camera_contract": camera_contract,
            "arm_visibility": arm_visibility,
            "variant_plan": plan,
            "variant_results": variant_results,
            "warmup": {
                "frames": int(warmup_frames),
                "frame_seconds": warmup_seconds,
                "settings": warmup_settings,
            },
            "lighting_summary": lighting_summary,
            "runtime_metadata": runtime_metadata,
            "render_settings": render_settings_manifest,
            "claim_boundary": result["claim_boundary"],
        }
        _write_audit_json("audit_run_manifest.json", run_manifest)

        if audit_mod is not None:
            try:
                analysis = audit_mod.analyze_render_noise_audit_run(out_dir)
                result["worker_analysis_status"] = analysis.get("status")
                result["worker_primary_diagnosis"] = (
                    (analysis.get("interpretation") or {}).get("primary_diagnosis")
                )
            except Exception as exc:  # noqa: BLE001
                result["worker_analysis_status"] = "failed"
                result["worker_analysis_error"] = repr(exc)
        else:
            result["worker_analysis_status"] = "analyzer_module_unavailable"

        if not result["variants_rendered"]:
            result["blockers"].append("no_audit_variants_rendered")
        result["status"] = "completed" if not result["blockers"] else "blocked"
        _write_result()
        return result
    except Exception as exc:  # noqa: BLE001
        result["blockers"].append("render_noise_audit_exception")
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()[-4000:]
        _write_result()
        return result
    finally:
        try:
            sim.close()
        except Exception:  # noqa: BLE001
            pass


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Isaac G1 kitchen parity eval (GPU)")
    ap.add_argument("--request", help="execution request JSON (scenarios + asset hints)")
    ap.add_argument("--kitchen-usd", help="path/URI to Collected_KitchenRoom/KitchenRoom.usd")
    ap.add_argument("--g1-usd", help="path/URI to the official Isaac G1 USD")
    ap.add_argument(
        "--robot-id",
        help="registered robot profile id (default: unitree_g1); drives footprint/"
             "pelvis/reach scaling for placement and seed gating",
    )
    ap.add_argument(
        "--robot-profile-json",
        help="path to a RobotProfile JSON (wins over --robot-id); lets a new robot "
             "be pure data with no code change",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--policy", default="blueprint_default_walk_to_target_smoke_policy")
    ap.add_argument(
        "--groot-policy-command",
        default=os.getenv(GROOT_POLICY_COMMAND_ENV, ""),
        help=(
            "command that reads a Blueprint policy observation JSON on stdin and returns "
            "GR00T/SONIC action JSON; required for --policy groot_sonic to drive Isaac"
        ),
    )
    ap.add_argument(
        "--groot-policy-command-timeout-seconds",
        type=float,
        default=float(os.getenv(GROOT_POLICY_COMMAND_TIMEOUT_ENV, "120") or 120),
        help="timeout for each GR00T/SONIC policy command call",
    )
    ap.add_argument(
        "--groot-policy-initial-frame",
        default=os.getenv(GROOT_POLICY_INITIAL_FRAME_ENV, ""),
        help="optional seed robot-POV frame path for the first GR00T/SONIC policy call",
    )
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--warmup-frames", type=int, default=6)
    ap.add_argument("--capture-every", type=int, default=1)
    ap.add_argument("--no-collision-probe", action="store_true",
                    help="skip the PhysX overlap probe (policy goes direct) — decouples render from physics")
    ap.add_argument("--per-scenario-seconds", type=int, default=480,
                    help="wall-clock cap per scenario so a slow/hung render cannot run forever")
    ap.add_argument("--focus-radius", type=float, default=0.0,
                    help="task-aware scene subset: keep only objects within N m of the route (0=full scene)")
    ap.add_argument("--keep-objects", default="room,floor,wall,ground,ceiling,light",
                    help="comma substrings of object names to always keep (structural shell)")
    ap.add_argument("--settle-seconds", type=int, default=0,
                    help="wait N s after scene load (PhysX on) for cooking to drain before rendering")
    ap.add_argument("--disable-physx", action="store_true",
                    help="(experiment only) disable physx cooking — known to break the renderer")
    ap.add_argument("--cheap-collision", action="store_true",
                    help="force bounding-box collision on all meshes (fast cooking; keeps full scene)")
    ap.add_argument("--articulated", action="store_true",
                    help="drive the G1 joints with the walk gait + emit g1_projected_skeleton_trace.jsonl (for OSCAR)")
    ap.add_argument("--camera-vfov", type=float, default=50.0, help="POV camera vertical FOV (deg) for skeleton projection")
    ap.add_argument("--manipulation-cam", action="store_true",
                    help="egocentric manipulation POV from the robot head/face looking forward into "
                         "the task workspace instead of the behind-and-above follow cam")
    ap.add_argument("--manipulation-look-at", default=None,
                    help="fixed world 'x,y,z' the manipulation cam aims at — pins the "
                         "framing to the known workspace instead of the policy's noisy final yaw")
    ap.add_argument("--render-subframes", type=int, default=1,
                    help="RTX orchestrator steps accumulated per captured frame to denoise grain (e.g. 16)")
    ap.add_argument("--no-software-denoise", action="store_true",
                    help="disable CPU-side PNG denoise fallback for saved frames")
    ap.add_argument("--manipulation-reach", action="store_true",
                    help="pose the visible G1 arms into the workspace for manipulation POV review; "
                         "this is posed simulator media, not manipulation-success proof")
    ap.add_argument("--manipulation-reach-arm", default="both", choices=["right", "left", "both"],
                    help="which arm is posed for the task")
    ap.add_argument("--fill-light-intensity", type=float, default=0.0,
                    help="add a sphere fill light over the manipulation workspace at this intensity; "
                         "0 disables")
    ap.add_argument("--physics-articulation-drive", action="store_true",
                    help="(opt-in, default off) drive the G1 via the physics articulation tensor view. "
                         "All root seeds stay on the articulation API; the pure-USD root xform fallback "
                         "is used only when this is off.")
    ap.add_argument("--effort-drive", action="store_true",
                    help="(opt-in, requires --physics-articulation-drive) drive settle joints with "
                         "ported PD joint_efforts instead of position targets")
    ap.add_argument("--author-target-contact-material", action="store_true",
                    help="(opt-in, requires --physics-articulation-drive) author MassAPI, physics "
                         "material, and convexDecomposition on the resolved task target prim only")
    ap.add_argument("--dynamic-standing-contact-steps", type=int, default=0,
                    help="opt-in PhysX standing/contact settle steps per sampled placement. This "
                         "forces --articulated, enables gravity, avoids the SingleArticulation "
                         "tensor view, and records physics_articulation_standing_contact_reports.json. "
                         "It is standing/contact evidence, not full dynamic walking.")
    ap.add_argument("--neutral-environment", action="store_true",
                    help="replace the kitchen asset's outdoor-HDRI dome light with a neutral bright "
                         "environment (no cityscape through the windows + lifts shadowed surfaces)")
    ap.add_argument("--robot-review-material-override", action="store_true",
                    help="bind a neutral matte material over authored G1 materials/textures for a "
                         "clearer untextured manipulation seed image")
    ap.add_argument(
        "--robot-review-material-mode",
        default="neutral_matte",
        choices=["neutral_matte", "non_white_matte"],
        help="review material to bind when --robot-review-material-override is active",
    )
    ap.add_argument("--kinematic-arm-pose", action="store_true",
                    help="pose the RENDERED arm(s) into a forward manipulation-ready seed via pure-USD "
                         "shoulder rotation (no physics tensor -> crash-safe); needs --manipulation-reach")
    ap.add_argument("--collision-approximation", default="boundingCube",
                    choices=["boundingCube", "convexHull", "convexDecomposition"],
                    help="mesh collision shape: boundingCube (fast, coarse) vs convexHull (shape-"
                         "accurate enough for close task stances, still fast)")
    ap.add_argument("--verify-cam", action="store_true",
                    help="render a 3rd-person verify_*.png that frames the whole robot at the workspace "
                         "(proves where it stands vs the egocentric POV)")
    ap.add_argument("--depth-pass", action="store_true",
                    help="attach a co-registered distance_to_image_plane depth annotator alongside "
                         "RGB and save per-frame depth (.npy + preview PNG)")
    ap.add_argument("--segmentation", action="store_true",
                    help="attach native Replicator instance/semantic segmentation annotators and "
                         "save deterministic masks (Isaac-only diagnostic)")
    ap.add_argument("--manipulation-stand", action="store_true",
                    help="place the robot AT the scenario target facing --manipulation-look-at every "
                         "step (task start pose; no navigation/redirect) — for manipulation, not locomotion")
    ap.add_argument("--render-noise-audit", action="store_true",
                    help="RENDER-QUALITY AUDIT: one dynamic scene/stance/camera setup from the request's "
                         "first scenario, then one RAW PNG per material/render variant (white proxy, "
                         "textured raw/denoised, high-sample, simplified diffuse, bright lighting) with "
                         "material-resolution + render-settings + camera-contract manifests. Runs instead "
                         "of the scenario eval; render-quality evidence only.")
    ap.add_argument("--audit-high-spp", type=int, default=0,
                    help="samples per pixel for the audit's high-budget variants "
                         f"(default {DEFAULT_AUDIT_HIGH_SAMPLES_PER_PIXEL}, env PARITY_AUDIT_HIGH_SAMPLES_PER_PIXEL)")
    ap.add_argument("--audit-warmup-frames", type=int, default=DEFAULT_AUDIT_WARMUP_FRAMES,
                    help="shader/RTX-cache warmup render steps before the first audit variant so "
                         "cold-start variance is measured, not baked into a variant frame")
    ap.add_argument("--audit-boost-light-intensity", type=float, default=0.0,
                    help="workspace boost light intensity for the audit's bright-lighting variant "
                         f"(default {DEFAULT_AUDIT_BOOST_LIGHT_INTENSITY})")
    ap.add_argument("--dry-render", action="store_true",
                    help="NO-GPU local preview: reproduce stance + camera + arms-forward framing from the "
                         "kitchen USD + task string and write a preview PNG, so placement/POV bugs are caught "
                         "locally before a cloud render. Needs --kitchen-usd + --request (no --g1-usd, no GPU).")
    ap.add_argument("--dry-render-out", default=None,
                    help="output dir for --dry-render artifacts (default: <out-dir>/dry_render)")
    ap.add_argument("--serve", action="store_true",
                    help="PERSISTENT WARM MODE: boot Isaac + load the scene ONCE, then serve a stream of "
                         "task scenarios dropped into --serve-dir (each rerun skips image pull + Isaac boot "
                         "+ stage load + most settle). Exits on a stop sentinel, idle timeout, or max jobs.")
    ap.add_argument("--serve-dir", default=None,
                    help="job/result directory the warm worker polls (default: <out-dir>/warm_jobs)")
    ap.add_argument("--serve-idle-timeout", type=float, default=600.0,
                    help="seconds with no new job before the warm worker exits (default 600)")
    ap.add_argument("--serve-max-jobs", type=int, default=None,
                    help="optional cap on jobs served before the warm worker exits (default: unlimited)")
    return ap


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Robot embodiment first: every downstream placement/reach/seed constant must
    # already be profile-scaled before scenarios are planned or rendered.
    apply_robot_profile(resolve_robot_profile_from_args(args))

    manip_look_at = None
    if args.manipulation_look_at:
        parts = [float(v) for v in str(args.manipulation_look_at).replace(" ", "").split(",") if v]
        if len(parts) == 3:
            manip_look_at = (parts[0], parts[1], parts[2])

    request = load_request(args.request) if args.request else {}
    scenarios = parse_scenarios(request)
    kitchen_usd = args.kitchen_usd or request.get("kitchen_usd") or request.get("scene_usd")
    g1_usd = args.g1_usd or request.get("g1_usd")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_render:
        dry_out = Path(args.dry_render_out) if args.dry_render_out else (out_dir / "dry_render")
        if not scenarios or not kitchen_usd:
            res = {"status": "blocked", "blockers": ["missing_scenarios_or_kitchen_usd"],
                   "have_scenarios": bool(scenarios), "have_kitchen_usd": bool(kitchen_usd)}
            print(json.dumps(res))
            return 1
        stage = _open_stage_local(kitchen_usd)
        robot_prim_path = "/World/G1"
        robot_binding: dict[str, Any] | None = None
        if g1_usd:
            robot_binding = _bind_g1_with_visual_fallback(stage, g1_usd, prim_path=robot_prim_path)
        else:
            _bind_proxy_robot(stage, robot_prim_path)
        reach_arm = args.manipulation_reach_arm
        summaries = []
        for sc in scenarios:
            sid = str(sc.get("scenario_id") or sc.get("episode_id") or f"scenario_{len(summaries)}")
            summ = render_local_preview(
                stage=stage, scenario=sc, out_dir=dry_out / sid,
                manipulation_reach_arm=reach_arm, camera_vfov_deg=args.camera_vfov,
                width=args.width, height=args.height, manipulation_look_at=manip_look_at,
                robot_prim_path=robot_prim_path,
                robot_visual_prim_path=robot_prim_path,
                robot_binding=robot_binding,
            )
            summaries.append(summ)
        (dry_out / "dry_render_index.json").write_text(json.dumps(summaries, indent=2))
        accepted = sum(1 for s in summaries if s.get("stance", {}).get("status") == "accepted")
        print(json.dumps({"status": "dry_render_complete", "scenarios": len(summaries),
                          "accepted": accepted, "out_dir": str(dry_out)}))
        return 0

    put_url = os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL", "")

    if args.render_noise_audit:
        if not kitchen_usd or not g1_usd or not scenarios:
            res = {"schema_version": RENDER_NOISE_AUDIT_RESULT_SCHEMA_VERSION, "status": "blocked",
                   "blockers": ["missing_scenarios_or_kitchen_usd_or_g1_usd"],
                   "have_scenarios": bool(scenarios), "have_kitchen_usd": bool(kitchen_usd),
                   "have_g1_usd": bool(g1_usd)}
            (out_dir / RENDER_NOISE_AUDIT_RESULT_NAME).write_text(json.dumps(res, indent=2))
            upload_zip(out_dir, put_url)
            print(json.dumps(res))
            return 1
        audit_result = run_render_noise_audit(
            kitchen_usd=kitchen_usd, g1_usd=g1_usd, scenario=scenarios[0], out_dir=out_dir,
            width=args.width, height=args.height, camera_vfov_deg=args.camera_vfov,
            reach_arm=args.manipulation_reach_arm,
            warmup_frames=args.audit_warmup_frames,
            render_subframes=max(1, int(args.render_subframes or 16)),
            variant_plan=render_noise_audit_plan_from_request(request),
            high_samples_per_pixel=(args.audit_high_spp or None),
            boost_light_intensity=(args.audit_boost_light_intensity or None),
            fill_light_intensity=args.fill_light_intensity,
            neutral_environment=args.neutral_environment,
            no_collision_probe=args.no_collision_probe,
            per_variant_seconds=float(args.per_scenario_seconds),
        )
        upload_zip(out_dir, put_url)
        print(json.dumps({"status": audit_result["status"],
                          "variants_rendered": audit_result.get("variants_rendered"),
                          "blockers": audit_result.get("blockers")}))
        return 0 if audit_result["status"] == "completed" else 1

    # Warm serve mode boots Isaac with NO initial scenarios — jobs arrive at runtime via --serve-dir —
    # so it requires the assets but not a pre-supplied scenario list.
    if not kitchen_usd or not g1_usd or (not scenarios and not args.serve):
        res = {"schema_version": RESULT_SCHEMA_VERSION, "status": "blocked",
               "blockers": ["missing_scenarios_or_kitchen_usd_or_g1_usd"],
               "have_scenarios": bool(scenarios), "have_kitchen_usd": bool(kitchen_usd),
               "have_g1_usd": bool(g1_usd)}
        (out_dir / "isaac_g1_kitchen_parity_result.json").write_text(json.dumps(res, indent=2))
        upload_zip(out_dir, put_url)
        print(json.dumps(res))
        return 1
    result = run_scenarios(
        kitchen_usd=kitchen_usd, g1_usd=g1_usd, scenarios=scenarios, out_dir=out_dir,
        policy_id=args.policy, steps=args.steps, width=args.width, height=args.height,
        fps=args.fps, warmup_frames=args.warmup_frames, capture_every=args.capture_every,
        no_collision_probe=args.no_collision_probe, per_scenario_seconds=args.per_scenario_seconds,
        focus_radius=args.focus_radius,
        keep_substrings=tuple(s for s in args.keep_objects.split(",") if s.strip()),
        disable_physx=args.disable_physx, settle_seconds=args.settle_seconds,
        cheap_collision=args.cheap_collision, articulated=args.articulated,
        camera_vfov_deg=args.camera_vfov, manipulation_cam=args.manipulation_cam,
        manipulation_look_at=manip_look_at, render_subframes=args.render_subframes,
        manipulation_reach=args.manipulation_reach, manipulation_reach_arm=args.manipulation_reach_arm,
        fill_light_intensity=args.fill_light_intensity,
        physics_articulation_drive=args.physics_articulation_drive,
        effort_drive=args.effort_drive,
        author_target_contact_material=args.author_target_contact_material,
        groot_policy_command=args.groot_policy_command,
        groot_policy_command_timeout_seconds=args.groot_policy_command_timeout_seconds,
        groot_policy_initial_frame=args.groot_policy_initial_frame,
        dynamic_standing_contact_steps=args.dynamic_standing_contact_steps,
        neutral_environment=args.neutral_environment,
        robot_review_material_override=args.robot_review_material_override,
        robot_review_material_mode=args.robot_review_material_mode,
        kinematic_arm_pose=args.kinematic_arm_pose,
        collision_approximation=args.collision_approximation, verify_cam=args.verify_cam,
        manipulation_stand=args.manipulation_stand,
        software_denoise=not args.no_software_denoise,
        depth_pass=args.depth_pass,
        segmentation=args.segmentation,
        serve=args.serve,
        serve_dir=(Path(args.serve_dir) if args.serve_dir else None),
        serve_idle_timeout_s=args.serve_idle_timeout, serve_max_jobs=args.serve_max_jobs)
    upload_zip(out_dir, put_url)
    print(json.dumps({"status": result["status"], "passed": result["scenarios_passed"],
                      "executed": result["scenarios_executed"]}))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
