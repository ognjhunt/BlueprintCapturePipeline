"""Strict per-cell no-policy controls using the existing native IK and runner."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest

CONTROL_IDS = ("zero_action_negative", "deterministic_scripted_positive")


def controls_required(contract: Mapping[str, Any]) -> bool:
    return (contract.get("criteria") or {}).get("controls") == {
        "mode": "required_per_cell", "control_ids": list(CONTROL_IDS)}


def _file(path: Path, root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return {"relative_path": str(path.relative_to(root)), "size_bytes": path.stat().st_size,
            "sha256": "sha256:"+digest.hexdigest()}


def camera_binding(plan: Mapping[str, Any], gate: Mapping[str, Any]) -> str:
    return canonical_digest({"cameras": plan.get("cameras"),
                             "selected_mount": gate.get("wrist_camera_mount_selection_digest")})


def verify_files(records: list[Mapping[str, Any]], root: Path) -> bool:
    if not records:
        return False
    for row in records:
        path = (root / str(row.get("relative_path") or "")).resolve()
        if (root.resolve() not in path.parents or path.is_symlink() or not path.is_file()
                or _file(path, root.resolve()) != dict(row)):
            return False
    return True



def validate_strict_camera_gate(gate: Mapping[str, Any]) -> None:
    cameras = {row.get("role"): row for row in (gate.get("snapshot") or {}).get("cameras", [])}
    if gate.get("policy_observation_integrity_passed") is not True or set(cameras) != {"external", "wrist", "overview"}:
        raise RuntimeError("strict_controls_native_camera_gate_failed")
    for role, camera in cameras.items():
        minimum = ((camera.get("observability") or {}).get("thresholds") or {}).get("effective_minimum_pixels")
        pixels = camera.get("semantic_label_pixels") or {}
        if (type(minimum) is not int or minimum <= 0
                or any(type(pixels.get(label)) is not int or pixels[label] < minimum
                       for label in ("task_object", "task_support"))):
            raise RuntimeError("strict_controls_subject_destination_visibility_failed:" + str(role))

def _control_candidate(scene_plan: Mapping[str, Any], phase_plan: Mapping[str, Any]) -> dict[str, Any]:
    spec = scene_plan["task_spec"]
    # Construction qualification retains its recovery/reset rehearsal. A scored
    # positive control must instead preserve the released destination state and
    # withdrawn gripper through its terminal scoring window.
    phases = [row for row in phase_plan["phases"] if row["phase_id"] != "recovery"]
    if not phases or phases[-1]["phase_id"] != "retreat":
        raise RuntimeError("strict_controls_terminal_retreat_required")
    execution = phase_plan["execution_parameters"]
    settle = int(spec["settle_window_samples"])
    maximum = min(int(execution["maximum_steps_per_phase"]),
                  (int(spec["maximum_action_steps"]) - settle) // len(phases))
    stable = int(execution["stable_samples"])
    if maximum < stable:
        raise RuntimeError("strict_controls_authored_episode_budget_insufficient")
    plan = {
        "schema_version": "adp_task_control_plan.v1", "task_kind": "rigid_pick_place",
        "cell_id": scene_plan["scenario"]["cell_id"],
        "task_spec_digest": canonical_digest(spec),
        "trajectory_source": "native_ik_preflight",
        "planner_receipt_digest": phase_plan["plan_digest"],
        "zero_action_steps": settle, "maximum_pose_phase_attempts": 1,
        "maximum_scripted_and_settle_steps": len(phases)*maximum+settle,
        "scripted_positive_actions": [{
            "phase_id": row["phase_id"], "mode": "ik_pose",
            "target_position_world_m": row["position_world_m"],
            "target_quaternion_world_xyzw": row["orientation_world_xyzw"],
            "gripper_state": row["gripper_state"], "minimum_steps": stable,
            "maximum_steps": maximum, "arrival_stability_steps": stable,
            "arrival_tolerance_m": execution["arrival_tolerance_m"],
            "arrival_orientation_tolerance_rad": execution["arrival_orientation_tolerance_rad"],
            "max_joint_delta_rad": execution["max_joint_delta_rad"],
            "max_joint_setpoint_lead_rad": execution["max_joint_setpoint_lead_rad"],
        } for row in phases], "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def execute_native_controls(*, cell_runtime: Any, built: Any, scene_plan: Mapping[str, Any],
                            gate: Mapping[str, Any], output_root: Path,
                            execution_binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from .adp009d_control_episode import run_task_neutral_controls, validate_task_control_plan
    from .native_task_construction_plan import materialize_rigid_construction_phase_plan
    from .native_task_arena_controls_worker import _control_plan_global_ik_joint_targets
    from .native_franka_pose_servo import PINK_GLOBAL_REFERENCE_SEEDS

    root = output_root / "strict_controls"
    root.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {
        "schema_version": "policy_canary_cell_controls.v1", "status": "blocked",
        "cell_id": scene_plan["scenario"]["cell_id"], "seed": scene_plan["scenario"]["seed"],
        "scene_plan_digest": scene_plan["plan_digest"],
        "task_spec_digest": canonical_digest(scene_plan["task_spec"]),
        "camera_binding_digest": camera_binding(scene_plan, gate),
        "candidate_policy_queried": False, "controls": [], "blockers": [], "files": [],
        "execution_binding": dict(execution_binding or {}),
    }
    try:
        validate_strict_camera_gate(gate)
        env = built.env
        seed = int(scene_plan["scenario"]["seed"])
        env.reset(seed=seed)
        robot = env.unwrapped.scene["robot"]
        gripper = cell_runtime.gripper_probe(env=env, robot=robot, seed=seed)
        if gripper.get("status") != "measured":
            raise RuntimeError("strict_controls_gripper_unresolved")
        env.reset(seed=seed)
        servo = cell_runtime.make_servo(env=env, robot=robot, gripper_convention=gripper)
        phases = materialize_rigid_construction_phase_plan(scene_plan)
        plan = _control_candidate(scene_plan, phases)
        targets, ik = _control_plan_global_ik_joint_targets(
            servo=servo, control_plan=plan, bound_targets=[], reference_seeds=PINK_GLOBAL_REFERENCE_SEEDS)
        (root / "native_ik_preflight.json").write_text(json.dumps(ik, indent=2))
        (root / "native_phase_plan.json").write_text(json.dumps(phases, indent=2))
        if ik.get("status") != "all_unique_poses_solved_or_bound":
            raise RuntimeError("strict_controls_every_waypoint_ik_required")
        plan["planner_receipt_digest"] = canonical_digest(ik)
        plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
        plan = validate_task_control_plan(plan, task_spec=scene_plan["task_spec"])
        plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
        readback = cell_runtime.make_rigid_task_readback(built)
        environment, _ = cell_runtime.build_episode_environment(
            built=built, gripper_convention=gripper, servo=servo, task_readback=readback,
            to_tensor=cell_runtime.to_tensor, scripted_pose_joint_targets=targets,
            scripted_pose_phase_targets=plan["scripted_positive_actions"])
        environment = cell_runtime.wrap_rigid_scoring_environment(
            environment=environment, task_readback=readback, task_spec=scene_plan["task_spec"])
        pair = run_task_neutral_controls(
            environment=environment, task_spec=scene_plan["task_spec"], control_plan=plan,
            gripper_open_command=float(gripper["open_command"]),
            gripper_closed_command=float(gripper["closed_command"]), output_dir=root)
        receipt["pair"] = pair
        receipt["controls"] = [json.loads((root / f"adp_task_control_episode.{control}.json").read_text())
                               for control in CONTROL_IDS]
        if (pair.get("cell_admitted_for_policy_execution") is not True
                or any(row.get("control_passed") is not True or row.get("candidate_policy_queried") is not False
                       or row.get("visual_evidence", {}).get("status") != "complete"
                       or row.get("score", {}).get("status") != "scored" for row in receipt["controls"])):
            raise RuntimeError("strict_controls_pair_failed")
        receipt["status"] = "passed"
    except Exception as exc:
        receipt["blockers"] = [str(exc)]
        # Preserve whichever real control episodes completed before the failure.
        receipt["controls"] = [json.loads(path.read_text()) for path in sorted(root.glob("adp_task_control_episode.*.json"))]
    receipt["files"] = [_file(path, output_root) for path in sorted(root.rglob("*")) if path.is_file()]
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    (output_root / "policy_canary_cell_controls.v1.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def validate_controls_receipt(receipt: Mapping[str, Any], *, scene_plan: Mapping[str, Any],
                              gate: Mapping[str, Any], root: Path,
                              execution_binding: Mapping[str, Any] | None = None) -> None:
    validate_strict_camera_gate(gate)
    if (receipt.get("status") != "passed" or receipt.get("candidate_policy_queried") is not False
            or receipt.get("scene_plan_digest") != scene_plan["plan_digest"]
            or (execution_binding is not None and receipt.get("execution_binding") != dict(execution_binding))
            or receipt.get("camera_binding_digest") != camera_binding(scene_plan, gate)
            or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
            or {row.get("control_id") for row in receipt.get("controls", [])} != set(CONTROL_IDS)
            or not verify_files(list(receipt.get("files") or []), root)):
        raise RuntimeError("strict_controls_receipt_binding_or_retention_failed")


def strict_result_controls_valid(payload: Mapping[str, Any]) -> bool:
    if not controls_required(payload.get("task_success_contract") or {}):
        return True
    if payload.get("status") == "blocked" and payload.get("strict_gate_blockers"):
        return True  # Retained failed admission; never a completed scientific result.
    controls = payload.get("controls") or []
    episodes = payload.get("episodes") or []
    cells = {(row.get("cell_id"), row.get("seed")) for row in episodes}
    identities = {(row.get("cell_id"), row.get("seed"), row.get("control_id")) for row in controls}
    if len(cells) != 10 or len(controls) != 20 or identities != {
        (cell, seed, control) for cell, seed in cells for control in CONTROL_IDS
    }:
        return False
    for row in controls:
        receipt = row.get("receipt") or {}
        score = receipt.get("score") or {}
        if (row.get("control_passed") is not True or receipt.get("control_passed") is not True
                or receipt.get("control_id") != row.get("control_id")
                or receipt.get("candidate_policy_queried") is not False
                or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
                or score.get("status") != "scored" or score.get("task_succeeded") is not (row["control_id"] == CONTROL_IDS[1])
                or (receipt.get("visual_evidence") or {}).get("status") != "complete"):
            return False
    return ((payload.get("controls_gate") or {}).get("status") == "passed"
            and (payload.get("strict_paired_gate") or {}).get("status") == "passed"
            and (payload.get("paired_delivery") or {}).get("status") == "uploaded_and_readback_verified")
