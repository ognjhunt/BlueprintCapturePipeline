"""Launcher-facing contract for the sealed ``blueprint-groot-oscar-eval`` image.

This module is the seam between the closed-loop eval lane
(``oscar_isaac_closed_loop_eval`` + ``groot_sonic_policy_endpoint``) and a
prebaked worker image that freezes tonight's dependency archaeology (torch /
cuDNN / transformer-engine + OSCAR + Isaac-GR00T venv + checkpoints + our
package). When the sealed image ref is configured and confirmed, a paid pod is
``docker pull`` + start the baked GR00T server + run the closed-loop CLI — with
no clone / venv / download churn.

Everything here is pure and hermetic (no Docker, no pod, no network): it
resolves configuration and emits *plans* (arg-lists, path allow-lists, env
markers). Side-effecting build/launch belongs in the shell scripts and the
runbook that consume these plans.

Claim boundary: a configured/confirmed sealed image proves build + runtime
readiness only. It does not prove provider startup, GR00T policy inference, WAM
generation quality, or semantic task success.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .lane_hardware_requirements import LANE_HARDWARE_REQUIREMENTS
from .oscar_cosmos_wam_evaluator import (
    WAM_CONSISTENCY_COMMAND_ENV,
    WAM_CONSISTENCY_GATE_ENV,
    WAM_SUCCESS_LABEL_COMMAND_ENV,
    WAM_SUCCESS_LABEL_GATE_ENV,
)
from .oscar_official_release import OFFICIAL_OSCAR_HF_REVISION

# --------------------------------------------------------------------------- #
# configuration keys
# --------------------------------------------------------------------------- #

IMAGE_REF_ENV = "BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_REF"
IMAGE_REF_FILE_ENV = "BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_REF_FILE"
ROBOT_EVAL_WORKER_IMAGE_REF_ENV = "BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF"
DEFAULT_IMAGE_REF_FILE = "~/.blueprint-secrets/groot_oscar_closed_loop_image_ref"

SEALED_CONFIRMED_ENV = "BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_SEALED_IMAGE_CONFIRMED"
DEFAULT_MIN_TASK_ADAPTIVE_STEPS = 3
DEFAULT_VISUAL_MOTION_SMOKE_COMMAND = (
    "python -m blueprint_pipeline.wam_episode_consistency_label_local"
)
DEFAULT_WAM_CONSISTENCY_COMMAND: str | None = None
DEFAULT_WAM_CONSISTENCY_TIMEOUT_SECONDS = 300.0
DEFAULT_WAM_SUCCESS_LABEL_TIMEOUT_SECONDS = 900.0
DEFAULT_ACTION_SKELETON_COMMAND = (
    "/opt/oscar-venv/bin/python -m blueprint_pipeline.gear_sonic_controller_fk_adapter"
)
DEFAULT_TASK_COMPLETION_COMMAND = (
    "/opt/oscar-venv/bin/python -m blueprint_pipeline.isaac_persistent_task_completion_client"
)
DEFAULT_TASK_SUCCESS_CONTRACT_PATH = "/workspace/task_success_contract.json"
DEFAULT_INITIAL_G1_SONIC_STATE_PATH = "/workspace/initial_g1_sonic_state.json"
DEFAULT_KITCHEN_STAGE_PATH = "/workspace/kitchen/KitchenRoom.usd"
DEFAULT_ATTEMPT_INPUT_MANIFEST_PATH = "/workspace/attempt_input_manifest.json"

# Baked-path env vars (defaults match the image layout; overridable so the same
# wiring works whether the image was produced by the Dockerfile or by snapshot).
OSCAR_REPO_ENV = "BLUEPRINT_GROOT_OSCAR_OSCAR_REPO"
OSCAR_CHECKPOINT_ENV = "BLUEPRINT_GROOT_OSCAR_OSCAR_CHECKPOINT"
GROOT_ROOT_ENV = "BLUEPRINT_GROOT_OSCAR_GROOT_ROOT"
GROOT_VENV_PYTHON_ENV = "BLUEPRINT_GROOT_OSCAR_GROOT_VENV_PYTHON"
SONIC_CHECKPOINT_ENV = "BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT"
UNITREE_G1_USD_ENV = "BLUEPRINT_ISAAC_UNITREE_G1_USD"
GEAR_SONIC_CHECKPOINT_REPO_ENV = "GEAR_SONIC_CHECKPOINT_REPO"
GEAR_SONIC_CHECKPOINT_REVISION_ENV = "GEAR_SONIC_CHECKPOINT_REVISION"

DEFAULT_OSCAR_REPO = "/opt/OSCAR"
DEFAULT_OSCAR_CHECKPOINT = "/opt/blueprint/ckpts/oscar"
DEFAULT_GROOT_ROOT = "/opt/gr00t"
DEFAULT_SONIC_CHECKPOINT = "/opt/blueprint/ckpts/sonic"
DEFAULT_GROOT_VENV = "/opt/gr00t-venv"
DEFAULT_WBC_ROOT = "/opt/wbc"
DEFAULT_UNITREE_G1_USD = "/isaac-sim/Isaac/Robots/Unitree/G1/g1.usd"
OFFICIAL_GEAR_SONIC_CHECKPOINT_REPO = "nvidia/GEAR-SONIC"
OFFICIAL_GEAR_SONIC_CHECKPOINT_REVISION = (
    "5e22ddc69abcea2a9aafc40536b14c232d3f9d7f"
)
DEFAULT_OSCAR_VENV_PYTHON = "/opt/oscar-venv/bin/python"

POLICY_SERVER_PORT = 5550
POLICY_SERVER_URL = f"tcp://127.0.0.1:{POLICY_SERVER_PORT}"
SONIC_EMBODIMENT_TAG = "UNITREE_G1_SONIC"

_UNSTABLE_TAGS = {"latest", "local", "dev", "test"}

_CLAIM_BOUNDARY = (
    "A configured/confirmed sealed image proves build + runtime readiness only. "
    "It does not prove provider startup, GR00T policy inference, WAM generation "
    "quality, or semantic task success."
)


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


# --------------------------------------------------------------------------- #
# image-ref resolution (mirrors _configured_isaac_worker_image_ref order)
# --------------------------------------------------------------------------- #

def configured_image_ref(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Resolve the sealed image ref: explicit env > secret file > generic env."""
    env = _env(env)
    explicit = _string(env.get(IMAGE_REF_ENV))
    if explicit:
        return {
            "image_ref": explicit,
            "source": IMAGE_REF_ENV,
            "configured": True,
            "image_ref_file": None,
            "image_ref_file_present": False,
            "raw_secret_values_recorded": False,
        }
    file_value = _string(env.get(IMAGE_REF_FILE_ENV))
    image_ref_file = Path(file_value or DEFAULT_IMAGE_REF_FILE).expanduser()
    if image_ref_file.is_file():
        image_ref = image_ref_file.read_text(encoding="utf-8").strip()
        return {
            "image_ref": image_ref,
            "source": IMAGE_REF_FILE_ENV if file_value else "default_blueprint_secret_file_path",
            "configured": bool(image_ref),
            "image_ref_file": str(image_ref_file),
            "image_ref_file_present": True,
            "raw_secret_values_recorded": False,
        }
    generic = _string(env.get(ROBOT_EVAL_WORKER_IMAGE_REF_ENV))
    if generic:
        return {
            "image_ref": generic,
            "source": ROBOT_EVAL_WORKER_IMAGE_REF_ENV,
            "configured": True,
            "image_ref_file": str(image_ref_file),
            "image_ref_file_present": False,
            "raw_secret_values_recorded": False,
        }
    return {
        "image_ref": "",
        "source": None,
        "configured": False,
        "image_ref_file": str(image_ref_file),
        "image_ref_file_present": False,
        "raw_secret_values_recorded": False,
    }


def image_ref_launch_blockers(image_ref: str) -> list[str]:
    """Refuse missing/unversioned/unstable refs (same rule as the build script)."""
    ref = _string(image_ref)
    if not ref:
        return ["missing_image_ref"]
    if ":" not in ref and "@sha256:" not in ref:
        return ["image_ref_must_be_versioned"]
    tag = ref.rsplit("@", maxsplit=1)[0].rsplit(":", maxsplit=1)[-1] if "@sha256:" not in ref else ""
    if tag in _UNSTABLE_TAGS:
        return ["image_ref_refuses_unstable_tag"]
    return []


# --------------------------------------------------------------------------- #
# sealed-mode gate
# --------------------------------------------------------------------------- #

def sealed_image_contract(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Fail-closed sealed-mode contract for the closed-loop launcher.

    ``sealed_active`` is True only when the image ref is configured, launchable
    (versioned, not an unstable tag), and explicitly confirmed. Otherwise the
    legacy runtime-bootstrap path must be used and ``blockers`` names why.
    """
    env = _env(env)
    resolved = configured_image_ref(env=env)
    image_ref = _string(resolved.get("image_ref"))
    configured = bool(resolved.get("configured"))

    blockers = list(image_ref_launch_blockers(image_ref))
    sealed_confirmed = _string(env.get(SEALED_CONFIRMED_ENV)).lower() == "true"
    if not sealed_confirmed:
        blockers.append("sealed_image_not_confirmed")

    sealed_active = configured and not blockers

    return {
        "schema_version": "groot_oscar_closed_loop_sealed_image_contract.v1",
        "image_ref": image_ref,
        "image_ref_source": resolved.get("source"),
        "image_ref_configured": configured,
        "sealed_confirmed": sealed_confirmed,
        "sealed_active": sealed_active,
        "oscar_repo": _string(env.get(OSCAR_REPO_ENV)) or DEFAULT_OSCAR_REPO,
        "oscar_checkpoint": _string(env.get(OSCAR_CHECKPOINT_ENV)) or DEFAULT_OSCAR_CHECKPOINT,
        "groot_root": _string(env.get(GROOT_ROOT_ENV)) or DEFAULT_GROOT_ROOT,
        "groot_venv": DEFAULT_GROOT_VENV,
        "groot_venv_python": _string(env.get(GROOT_VENV_PYTHON_ENV))
        or f"{DEFAULT_GROOT_VENV}/bin/python",
        "sonic_checkpoint": _string(env.get(SONIC_CHECKPOINT_ENV)) or DEFAULT_SONIC_CHECKPOINT,
        "unitree_g1_usd": _string(env.get(UNITREE_G1_USD_ENV)) or DEFAULT_UNITREE_G1_USD,
        "gear_sonic_checkpoint_repo": (
            _string(env.get(GEAR_SONIC_CHECKPOINT_REPO_ENV))
            or OFFICIAL_GEAR_SONIC_CHECKPOINT_REPO
        ),
        "gear_sonic_checkpoint_revision": (
            _string(env.get(GEAR_SONIC_CHECKPOINT_REVISION_ENV))
            or OFFICIAL_GEAR_SONIC_CHECKPOINT_REVISION
        ),
        "policy_server_url": POLICY_SERVER_URL,
        "policy_server_port": POLICY_SERVER_PORT,
        "oscar_hf_revision": OFFICIAL_OSCAR_HF_REVISION,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
        "claim_boundary": _CLAIM_BOUNDARY,
    }


# --------------------------------------------------------------------------- #
# launch plan (what the sealed image actually runs)
# --------------------------------------------------------------------------- #

def build_sealed_launch_plan(
    *,
    start_frame: str,
    route_file: str,
    steps: int,
    task_prompt: str,
    output_dir: str,
    # OSCAR's native training resolution. 240x320 was a 2026-07-06 OOM-era
    # mitigation that bought no memory (weights dominate) while degrading
    # generation quality — never bake sub-native resolution into a launch plan.
    oscar_height: int = 480,
    oscar_width: int = 640,
    min_coherent_horizon_frames: int = 2,
    min_task_adaptive_steps: int = DEFAULT_MIN_TASK_ADAPTIVE_STEPS,
    require_forward_inverse_consistency: bool = True,
    wam_consistency_command: str | None = DEFAULT_WAM_CONSISTENCY_COMMAND,
    allow_wam_consistency_scoring: bool = True,
    wam_consistency_timeout_seconds: float | None = DEFAULT_WAM_CONSISTENCY_TIMEOUT_SECONDS,
    require_generated_video_success_label: bool = False,
    wam_success_label_command: str | None = None,
    allow_wam_success_labeling: bool = False,
    wam_success_label_timeout_seconds: float | None = DEFAULT_WAM_SUCCESS_LABEL_TIMEOUT_SECONDS,
    action_skeleton_command: str = DEFAULT_ACTION_SKELETON_COMMAND,
    task_completion_command: str = DEFAULT_TASK_COMPLETION_COMMAND,
    task_success_contract_path: str = DEFAULT_TASK_SUCCESS_CONTRACT_PATH,
    initial_g1_sonic_state_path: str = DEFAULT_INITIAL_G1_SONIC_STATE_PATH,
    kitchen_stage_path: str = DEFAULT_KITCHEN_STAGE_PATH,
    attempt_input_manifest_path: str = DEFAULT_ATTEMPT_INPUT_MANIFEST_PATH,
    device: str = "cuda:0",
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Ordered command plan: start the baked GR00T server, then the closed loop.

    Commands are arg-lists (not shell strings) so they are safe to render and
    exact to assert. When the sealed contract is not active the plan is blocked
    with empty commands and the caller must fall back to runtime bootstrap.
    """
    contract = sealed_image_contract(env=env)
    plan: dict[str, Any] = {
        "schema_version": "groot_oscar_closed_loop_sealed_launch_plan.v1",
        "sealed_active": contract["sealed_active"],
        "blockers": list(contract["blockers"]),
        # Pod sizing is part of the launch contract: provisioning against this
        # plan must satisfy the lane floor (see lane_hardware_requirements).
        "lane": "kitchen_g1_groot_sonic_eval",
        "lane_hardware_requirements": dict(
            LANE_HARDWARE_REQUIREMENTS["kitchen_g1_groot_sonic_eval"]
        ),
        "image_ref": contract["image_ref"],
        "policy_server_url": contract["policy_server_url"],
        "policy_server_port": contract["policy_server_port"],
        "groot_server_command": [],
        "isaac_task_executor_command": [],
        "gear_sonic_controller_command": [],
        "closed_loop_command": [],
        "episode_length_contract": {
            "episode_length_unit": "closed_loop_control_steps",
            "stop_condition": "task_completion_or_step_cap",
            "steps_cap": int(steps),
            "min_steps_before_task_completion": int(min_task_adaptive_steps),
            "steps_is_safety_cap": True,
            "oscar_num_frames_scope": "per_generation_clip_not_episode_limit",
            "episode_not_bound_to_oscar_clip_frames": True,
        },
        "quality_gate_contract": {
            "min_coherent_horizon_frames": int(min_coherent_horizon_frames),
            "forward_inverse_consistency_required": bool(
                require_forward_inverse_consistency
            ),
            "forward_inverse_consistency_command": _string(wam_consistency_command) or None,
            "forward_inverse_consistency_allow_scoring": bool(
                allow_wam_consistency_scoring
            ),
            "generated_video_success_label_required": bool(
                require_generated_video_success_label
            ),
            "generated_video_success_label_command": _string(wam_success_label_command)
            or None,
            "generated_video_success_label_allow_labeling": bool(
                allow_wam_success_labeling
            ),
            "claim_boundary": {
                "forward_inverse_consistency_is_required_for_eval_run_quality": bool(
                    require_forward_inverse_consistency
                ),
                "generated_video_success_label_is_separate_semantic_review": True,
                "generated_video_success_label_is_not_real_world_task_success": True,
            },
        },
        "env": {},
        "raw_secret_values_recorded": False,
        "claim_boundary": _CLAIM_BOUNDARY,
    }
    if not contract["sealed_active"]:
        return plan
    if require_forward_inverse_consistency and not _string(wam_consistency_command):
        plan["blockers"].append("wam_consistency_command_required")
        plan["sealed_active"] = False
        return plan
    if (
        require_forward_inverse_consistency
        and _string(wam_consistency_command) == DEFAULT_VISUAL_MOTION_SMOKE_COMMAND
    ):
        plan["blockers"].append(
            "visual_motion_smoke_cannot_satisfy_forward_inverse_consistency"
        )
        plan["sealed_active"] = False
        return plan
    if require_generated_video_success_label and not _string(wam_success_label_command):
        plan["blockers"].append("wam_success_label_command_required")
        plan["sealed_active"] = False
        return plan

    groot_root = contract["groot_root"]
    oscar_repo = contract["oscar_repo"]
    plan["groot_server_command"] = [
        contract["groot_venv_python"],
        "-m", "gr00t.eval.run_gr00t_server",
        "--model-path", contract["sonic_checkpoint"],
        "--embodiment-tag", SONIC_EMBODIMENT_TAG,
        "--device", str(device),
        "--port", str(contract["policy_server_port"]),
    ]
    plan["isaac_task_executor_command"] = [
        "/isaac-sim/python.sh",
        "-m",
        "blueprint_pipeline.isaac_persistent_task_executor_service",
        "--stage",
        kitchen_stage_path,
        "--g1-usd",
        contract["unitree_g1_usd"],
        "--route-file",
        route_file,
        "--attempt-input-manifest",
        attempt_input_manifest_path,
        "--initial-state-output",
        initial_g1_sonic_state_path,
    ]
    plan["gear_sonic_controller_command"] = [
        "bash",
        "-lc",
        "cd /opt/wbc/gear_sonic_deploy && source scripts/setup_env.sh && exec "
        "./target/release/g1_deploy_onnx_ref lo "
        "policy/release/model_decoder.onnx reference/example "
        "--obs-config policy/release/observation_config.yaml "
        "--encoder-file policy/release/model_encoder.onnx "
        "--planner-file planner/target_vel/V2/planner_sonic.onnx "
        "--input-type zmq_manager --output-type zmq --zmq-host localhost "
        "--disable-crc-check",
    ]
    plan["closed_loop_command"] = [
        DEFAULT_OSCAR_VENV_PYTHON, "-m", "blueprint_pipeline.oscar_isaac_closed_loop_eval",
        "--start-frame", start_frame,
        "--route-file", route_file,
        "--steps", str(int(steps)),
        "--task-prompt", task_prompt,
        "--oscar-repo", oscar_repo,
        "--checkpoint", contract["oscar_checkpoint"],
        "--output-dir", output_dir,
        "--groot-sonic-policy-server-url", contract["policy_server_url"],
        "--groot-root", groot_root,
        "--groot-policy-initial-state", initial_g1_sonic_state_path,
        "--require-fresh-learned-policy-requery",
        "--action-skeleton-command", action_skeleton_command,
        "--task-success-contract", task_success_contract_path,
        "--task-completion-command", task_completion_command,
        "--attempt-input-manifest", attempt_input_manifest_path,
        # Episode length is task-adaptive: --steps is the hard cap and the
        # episode ends when the target-reached criterion fires.
        "--stop-on-task-completion",
        "--min-steps", str(int(min_task_adaptive_steps)),
        "--harness-backend-kind", "real_provider_probe",
        "--require-real-perception-backend",
        "--require-sam3-completed",
        "--require-da3-completed",
        "--oscar-height", str(int(oscar_height)),
        "--oscar-width", str(int(oscar_width)),
        "--min-coherent-horizon-frames", str(int(min_coherent_horizon_frames)),
    ]
    consistency_command = _string(wam_consistency_command)
    if (
        require_forward_inverse_consistency
        or allow_wam_consistency_scoring
        or consistency_command
    ):
        if consistency_command:
            plan["closed_loop_command"].extend(
                ["--wam-consistency-command", consistency_command]
            )
        if allow_wam_consistency_scoring:
            plan["closed_loop_command"].append("--allow-wam-consistency-scoring")
        if wam_consistency_timeout_seconds is not None:
            plan["closed_loop_command"].extend(
                [
                    "--wam-consistency-timeout-seconds",
                    str(float(wam_consistency_timeout_seconds)),
                ]
            )
        if require_forward_inverse_consistency:
            plan["closed_loop_command"].append("--require-forward-inverse-consistency")
    success_label_command = _string(wam_success_label_command)
    if (
        require_generated_video_success_label
        or allow_wam_success_labeling
        or success_label_command
    ):
        if success_label_command:
            plan["closed_loop_command"].extend(
                ["--wam-success-label-command", success_label_command]
            )
        if allow_wam_success_labeling:
            plan["closed_loop_command"].append("--allow-wam-success-labeling")
        if wam_success_label_timeout_seconds is not None:
            plan["closed_loop_command"].extend(
                [
                    "--wam-success-label-timeout-seconds",
                    str(float(wam_success_label_timeout_seconds)),
                ]
            )
        if require_generated_video_success_label:
            plan["closed_loop_command"].append("--require-generated-video-success-label")
    plan["env"] = {
        "MUJOCO_GL": "osmesa",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTHONPATH": oscar_repo,
        "BLUEPRINT_OSCAR_WAM_HF_REVISION": contract["oscar_hf_revision"],
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT": groot_root,
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL": contract["policy_server_url"],
        "BLUEPRINT_GEAR_SONIC_ROOT": DEFAULT_WBC_ROOT,
        GEAR_SONIC_CHECKPOINT_REPO_ENV: contract["gear_sonic_checkpoint_repo"],
        GEAR_SONIC_CHECKPOINT_REVISION_ENV: contract["gear_sonic_checkpoint_revision"],
        "BLUEPRINT_GEAR_SONIC_ROBOT_MODEL": (
            "/opt/wbc/gear_sonic_deploy/g1/g1_29dof_with_hand.xml"
        ),
        "BLUEPRINT_GEAR_SONIC_EXECUTOR_COMMAND": (
            f"{DEFAULT_OSCAR_VENV_PYTHON} -m "
            "blueprint_pipeline.gear_sonic_official_zmq_executor"
        ),
    }
    if consistency_command or allow_wam_consistency_scoring or require_forward_inverse_consistency:
        plan["env"][WAM_CONSISTENCY_GATE_ENV] = "true"
        if consistency_command:
            plan["env"][WAM_CONSISTENCY_COMMAND_ENV] = consistency_command
    if success_label_command or allow_wam_success_labeling or require_generated_video_success_label:
        plan["env"][WAM_SUCCESS_LABEL_GATE_ENV] = "true"
        if success_label_command:
            plan["env"][WAM_SUCCESS_LABEL_COMMAND_ENV] = success_label_command
    return plan


# --------------------------------------------------------------------------- #
# snapshot layer plan (source of truth for the crane-snapshot build)
# --------------------------------------------------------------------------- #

def build_snapshot_layer_plan(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Allow-list of pod paths + image env markers for the crane-snapshot script.

    The snapshot build (primary path) tars exactly these trees onto the pod's
    base image so no ephemeral/customer cruft is captured, and stamps the sealed
    markers so the resulting image is self-describing.
    """
    contract = sealed_image_contract(env=env)
    snapshot_paths = [
        contract["oscar_repo"],
        contract["groot_root"],
        DEFAULT_GROOT_VENV,
        DEFAULT_WBC_ROOT,
        contract["oscar_checkpoint"],
        contract["sonic_checkpoint"],
        "/opt/blueprint",
    ]
    # de-dup while preserving order
    seen: set[str] = set()
    ordered_paths = [p for p in snapshot_paths if not (p in seen or seen.add(p))]
    return {
        "schema_version": "groot_oscar_closed_loop_snapshot_layer_plan.v1",
        "snapshot_paths": ordered_paths,
        "image_env": {
            SEALED_CONFIRMED_ENV: "true",
            OSCAR_REPO_ENV: contract["oscar_repo"],
            OSCAR_CHECKPOINT_ENV: contract["oscar_checkpoint"],
            GROOT_ROOT_ENV: contract["groot_root"],
            SONIC_CHECKPOINT_ENV: contract["sonic_checkpoint"],
            "MUJOCO_GL": "osmesa",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTHONPATH": contract["oscar_repo"],
            "BLUEPRINT_OSCAR_WAM_HF_REVISION": contract["oscar_hf_revision"],
            GEAR_SONIC_CHECKPOINT_REPO_ENV: contract["gear_sonic_checkpoint_repo"],
            GEAR_SONIC_CHECKPOINT_REVISION_ENV: contract[
                "gear_sonic_checkpoint_revision"
            ],
            "BLUEPRINT_WORKER_IMAGE_FAMILY": "groot-oscar-closed-loop-eval",
        },
        "raw_secret_values_recorded": False,
        "claim_boundary": _CLAIM_BOUNDARY,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _emit(payload: Mapping[str, Any]) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not payload.get("blockers") and payload.get("sealed_active", True) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the sealed blueprint-groot-oscar-eval image contract."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--print-sealed-contract", action="store_true")
    mode.add_argument("--print-launch-plan", action="store_true")
    mode.add_argument("--print-snapshot-plan", action="store_true")
    parser.add_argument("--start-frame", default="/workspace/initial_policy_frame.png")
    parser.add_argument("--route-file", default="/workspace/route.json")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--task-prompt", default="open the fridge")
    parser.add_argument("--output-dir", default="/workspace/t4_out")
    parser.add_argument("--oscar-height", type=int, default=480)
    parser.add_argument("--oscar-width", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    if args.print_snapshot_plan:
        return _emit(build_snapshot_layer_plan())
    if args.print_launch_plan:
        return _emit(
            build_sealed_launch_plan(
                start_frame=args.start_frame,
                route_file=args.route_file,
                steps=args.steps,
                task_prompt=args.task_prompt,
                output_dir=args.output_dir,
                oscar_height=args.oscar_height,
                oscar_width=args.oscar_width,
                device=args.device,
            )
        )
    return _emit(sealed_image_contract())


if __name__ == "__main__":
    raise SystemExit(main())
