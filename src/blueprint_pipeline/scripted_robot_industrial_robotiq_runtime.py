"""Bind five industrial Robotiq simulation variants without changing existing script identities.

There is no checkpoint, server, provider credential or model download. Legacy
execution fields named checkpoint_digest bind the shipped controller source;
the explicit evidence kind prevents interpreting them as learned weights.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
from pathlib import Path
import math
import numpy as np

from .scripted_robot_grasp_frames import _FRAMES, SCHEMA, CONVENTION
from .team_arm_kinematics import pose_matrix, quaternion_xyzw
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .scripted_robot_baseline import ACTION_SOURCE, EVIDENCE_KIND, INTERFACE_ID, PLAN_SCHEMA, ScriptedRobotBaseline, validate_plan
from .team_embodiment import camera_names, validate_embodiment_manifest

PROFILE_ID = "blueprint_scripted_pick_place_industrial_robotiq.v1"
BODIES = frozenset({'fanuc_crx_10ia_robotiq_2f85_sim', 'abb_gofa_crb15000_5_95_robotiq_2f85_sim', 'fanuc_m710ic_45m_robotiq_2f85_sim', 'abb_gofa_crb15000_12_127_robotiq_2f85_sim', 'fanuc_crx_10ia_l_robotiq_2f85_sim'})
MODEL_SOURCE = "blueprint-owned:scripted_pick_place"
MODEL_REVISION = "1"
MINIMUM_GPU_MEMORY_GB = 46  # Existing simulator allocation floor, not measured peak.
CODE_FILES = ("scripted_robot_baseline.py", "scripted_robot_grasp_frames.py", "scripted_robot_industrial_robotiq_runtime.py")


def source_identity() -> dict[str, Any]:
    package = Path(__file__).parent
    return {"schema_version": "blueprint_scripted_controller_source.v1", "files": [
        {"path": name, "sha256": "sha256:" + hashlib.sha256((package / name).read_bytes()).hexdigest()}
        for name in CODE_FILES]}


def build_policy_spec(manifest: Mapping[str, Any]) -> dict[str, Any]:
    declared = validate_embodiment_manifest(manifest)
    bindings = grasp_bindings(declared)
    return {"policy_interface_id": INTERFACE_ID, "policy_id": PROFILE_ID,
            "model_identity_digest": canonical_digest(source_identity()), "source_revision": MODEL_REVISION,
            "embodiment_id": declared["embodiment_id"], "configuration_digest": declared["embodiment_digest"],
            "action_chunk_rows": 1, "open_loop_horizon": 1, "action_hz": declared["control_hz"],
            "camera_names": camera_names(declared), "grasp_bindings": bindings,
            "execution_evidence_kind": EVIDENCE_KIND, "action_source": ACTION_SOURCE}


def validate_policy_spec(policy: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if dict(policy) != build_policy_spec(manifest):
        raise ValueError("scripted_baseline_robot_or_source_binding_changed")


def candidate_id(embodiment_id: str) -> str:
    return "blueprint_scripted_" + embodiment_id


def build_task_plan(scene_plan: Mapping[str, Any], policy: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    from .native_task_construction_plan import materialize_native_task_construction_phase_plan

    validate_policy_spec(policy, manifest)
    if scene_plan.get("task_kind") != "rigid_pick_place":
        raise ValueError("scripted_baseline_task_kind_unsupported")
    authored = materialize_native_task_construction_phase_plan(scene_plan)
    if authored["manipulation_strategy"] != "pick_and_place":
        raise ValueError("scripted_baseline_manipulation_strategy_unsupported")
    # Copy authored geometry only. Construction qualification, scored gate
    # verdicts and any achieved reference-arm trajectory are never inputs.
    phases = []
    for original in authored["phases"]:
        if original["phase_id"] == "recovery":
            continue  # Episode reset is owned by the simulator, not this script.
        phase = {key: deepcopy(original[key]) for key in (
            "phase_id", "position_world_m", "orientation_world_xyzw", "gripper_state")}
        if phase["phase_id"] == "grasp_contact":
            phases.append({**phase, "phase_id": "approach_contact_open", "gripper_state": "open"})
        phases.append(phase)
    params = authored["execution_parameters"]
    plan = {"schema_version": PLAN_SCHEMA, "execution_evidence_kind": EVIDENCE_KIND,
            "action_source": ACTION_SOURCE, "task_kind": "rigid_pick_place",
            "scene_plan_digest": scene_plan["plan_digest"], "task_spec_digest": canonical_digest(scene_plan["task_spec"]),
            "authored_construction_plan_digest": authored["plan_digest"],
            "configuration_digest": manifest["embodiment_digest"],
            "arm_names": [arm["name"] for arm in manifest["arms"]], "active_arm": "nearest_initial_tcp",
            "grasp_bindings": deepcopy(policy["grasp_bindings"]), "phases": phases,
            "action_hz": policy["action_hz"], "phase_timeout_s": 5., "gripper_dwell_s": .35,
            "arrival_tolerance_m": params["arrival_tolerance_m"],
            "arrival_orientation_tolerance_rad": params["arrival_orientation_tolerance_rad"],
            "simulation_assumptions": ["Arm with nearest initial measured TCP to authored pregrasp acts; other arms hold their measured reset TCP and gripper closure. Distance is not a reachability guarantee.",
                "Authored grasp targets are privileged scripted inputs; no learned perception or policy was evaluated.",
                "Pose tolerance, 0.35 s grip dwell and 5 s phase timeout are initial simulation control settings.",
                "Phase completion is pose arrival or timeout, not contact/task success; the native scorer is independent."],
            "plan_digest": ""}
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return validate_plan(plan)


def interface_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    manifest = spec["team_embodiment"]["manifest"]
    policy = spec["policy_spec"]
    validate_policy_spec(policy, manifest)
    dimensions = {(camera["height"], camera["width"]) for camera in manifest["cameras"]}
    if len(dimensions) != 1:
        raise ValueError("scripted_baseline_camera_review_dimensions_mixed")
    height, width = next(iter(dimensions))
    if spec["checkpoint_digest"] != policy["model_identity_digest"] or spec["open_loop_horizon"] != 1:
        raise ValueError("scripted_baseline_execution_identity_changed")
    return {"policy_interface_id": INTERFACE_ID, "robot_profile_id": manifest["embodiment_id"],
            "observation_profile_id": "blueprint_scripted_authored_targets_live_tcp.v1",
            "action_schema_id": "blueprint.named_arm_pose_chunk.v1", "control_hz": manifest["control_hz"],
            "action_horizon": 1, "execution_horizon": 1, "camera_roles": camera_names(manifest),
            "image_height": height, "image_width": width, "image_encoding": "rgb_uint8",
            "image_preprocessing": "review_only_not_used_by_scripted_controller", "history_frames": 1,
            "normalization_digest": None}


def validate_execution_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if (spec.get("policy_interface_id") != INTERFACE_ID or spec.get("evaluation_backend") != "robolab.v1"
            or spec.get("policy_endpoint") != {"transport": "in_process"}
            or spec.get("execution_evidence_kind") != EVIDENCE_KIND or spec.get("action_source") != ACTION_SOURCE
            or spec.get("execution_spec_digest") != canonical_digest(spec, digest_field="execution_spec_digest")):
        raise ValueError("scripted_baseline_execution_spec_invalid")
    interface_contract(spec)
    plan = validate_plan(spec["scripted_task_plan"])
    if (plan["configuration_digest"] != spec["policy_spec"]["configuration_digest"]
            or plan["grasp_bindings"] != spec["policy_spec"]["grasp_bindings"]
            or plan["action_hz"] != spec["policy_spec"]["action_hz"]):
        raise ValueError("scripted_baseline_task_binding_changed")
    return dict(spec)


def grasp_bindings(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return TCP-from-task-grasp transforms for the exact declared source tools.

    Named links, actual jaw mappings and arm count must match the audited
    source; a changed tool declaration cannot silently reuse this approximation.
    The manifest's sealed TCP may be calibrated differently: composition uses
    its full transform instead of assuming its position/orientation is zero.
    """
    declared = validate_embodiment_manifest(manifest)
    identity = declared["embodiment_id"]
    if identity not in BODIES:
        raise ValueError("scripted_grasp_source_profile_unsupported:" + identity)
    profile = deepcopy(_FRAMES["robotiq_ur"])
    profile["dof"] = 6
    arms, channels = declared.get("arms") or [], declared["grippers"]
    arm_names = ["arm"]
    if [arm["name"] for arm in arms] != arm_names or len(channels) != len(arm_names):
        raise ValueError("scripted_grasp_source_arm_binding_mismatch")
    channel_map = {channel["name"]: channel for channel in channels}
    result = {}
    for arm in arms:
        prefix = "" if arm["name"] == "arm" else (arm["name"] + "_")
        channel = channel_map[arm["gripper_name"]]
        expected_jaws = {prefix + name: values for name, values in profile["jaws"].items()}
        if (arm["tool_link"] != prefix + profile["tool"] or len(arm["joint_names"]) != profile["dof"]
                or set(channel["contact_links"]) != {prefix + name for name in profile["contacts"]}
                or {joint["name"] for joint in channel["joints"]} != set(expected_jaws)):
            raise ValueError("scripted_grasp_source_tool_binding_mismatch:" + arm["name"])
        if any(not math.isclose(joint[field], expected_jaws[joint["name"]][index], abs_tol=1e-8)
               for joint in channel["joints"] for index, field in enumerate(("open", "closed"))):
            raise ValueError("scripted_grasp_source_closure_mismatch:" + arm["name"])
        z, y = np.array(profile["approach"]), np.array(profile["jaw"])
        tool_from_grasp = np.eye(4)
        tool_from_grasp[:3, :3] = np.column_stack((np.cross(y, z), y, z))
        tool_from_grasp[:3, 3] = profile["position"]
        tcp = arm["tool_from_tcp"]
        tcp_from_grasp = np.linalg.inv(pose_matrix(tcp["position_m"], tcp["orientation_xyzw"])) @ tool_from_grasp
        result[arm["name"]] = {
            "schema_version": SCHEMA, "tcp_from_grasp": {"position_m": tcp_from_grasp[:3, 3].tolist(),
                "orientation_xyzw": quaternion_xyzw(tcp_from_grasp[:3, :3])},
            "basis_reference": profile["url"], "source_geometry_basis": profile["basis"],
            "task_grasp_convention": CONVENTION,
            "additional_source_references": [profile["additional_source"]] if "additional_source" in profile else [],
            "simulation_assumptions": deepcopy(profile["assumptions"]),
            "publisher_calibration_claimed": False, "physical_contact_point_measured": False,
            "native_task_success_claimed": False,
        }
    return result


class ScriptedRobotiqBaseline(ScriptedRobotBaseline):
    """The unchanged pose servo, with this profile's exact per-cell rebinding."""

    def bind_scene_plan(self, scene_plan: Mapping[str, Any]) -> None:
        self.plan = build_task_plan(scene_plan, self.policy_spec, self.manifest)
