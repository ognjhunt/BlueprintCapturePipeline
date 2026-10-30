"""Blueprint scripted simulation baseline; no learned model or task self-grading.

The sealed task supplies privileged authored grasp targets. Live arm-base/TCP
readback closes a bounded pose servo; independent native contacts/object state
remain exclusively with the scorer. This is not deployment-policy evidence.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

import numpy as np

from .decision_evidence_contracts import canonical_digest
from .team_arm_kinematics import POSE_CHUNK_SCHEMA, pose_matrix, quaternion_xyzw

INTERFACE_ID = "blueprint_scripted_pick_place.v1"
EVIDENCE_KIND = "scripted_simulator_state_baseline"
ACTION_SOURCE = "scripted_simulator_state"
PLAN_SCHEMA = "blueprint_scripted_pick_place_plan.v1"


def validate_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    plan = deepcopy(dict(value))
    if (plan.get("schema_version") != PLAN_SCHEMA or plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest")
            or plan.get("execution_evidence_kind") != EVIDENCE_KIND
            or plan.get("action_source") != ACTION_SOURCE or plan.get("task_kind") != "rigid_pick_place"
            or not isinstance(plan.get("arm_names"), list) or not plan["arm_names"]
            or len(set(plan["arm_names"])) != len(plan["arm_names"])
            or plan.get("active_arm") not in [*plan["arm_names"], "nearest_initial_tcp"]
            or set(plan.get("grasp_bindings", {})) != set(plan["arm_names"])
            or not isinstance(plan.get("phases"), list) or not plan["phases"]):
        raise ValueError("scripted_baseline_plan_invalid")
    for binding in plan["grasp_bindings"].values():
        transform = binding["tcp_from_grasp"]
        pose_matrix(transform["position_m"], transform["orientation_xyzw"])
        if not binding.get("basis_reference"):
            raise ValueError("scripted_baseline_tool_reference_missing")
    for phase in plan["phases"]:
        pose_matrix(phase["position_world_m"], phase["orientation_world_xyzw"])
        if phase["gripper_state"] not in {"open", "closed"}:
            raise ValueError("scripted_baseline_gripper_state_invalid")
    for field in ("action_hz", "phase_timeout_s", "gripper_dwell_s", "arrival_tolerance_m", "arrival_orientation_tolerance_rad"):
        if type(plan.get(field)) not in (int, float) or not math.isfinite(plan[field]) or plan[field] <= 0:
            raise ValueError("scripted_baseline_parameters_invalid")
    return plan


class ScriptedRobotBaseline:
    execution_evidence_kind = EVIDENCE_KIND
    action_source = ACTION_SOURCE
    require_named_arm_state = True
    require_scripted_arm_world_state = True
    candidate_policy_queried = False

    def __init__(self, *, plan: Mapping[str, Any], policy_id: str, policy_spec: Mapping[str, Any] | None = None,
                 manifest: Mapping[str, Any] | None = None):
        self.plan = validate_plan(plan)
        self.policy_id = policy_id
        self.action_hz = float(self.plan["action_hz"])
        self.policy_spec, self.manifest = deepcopy(policy_spec), deepcopy(manifest)
        self.reset()

    def bind_scene_plan(self, scene_plan: Mapping[str, Any]) -> None:
        """Re-derive authored targets for the current sealed scenario, not a scorer output."""
        if self.policy_spec is not None and self.manifest is not None:
            from .scripted_robot_baseline_runtime import build_task_plan
            self.plan = build_task_plan(scene_plan, self.policy_spec, self.manifest)
        elif scene_plan.get("plan_digest") != self.plan["scene_plan_digest"]:
            raise ValueError("scripted_baseline_scene_binding_changed")

    def reset(self, **_: Any) -> None:
        self.scripted_controller_invoked = False
        self.phase_index, self.phase_calls, self.stable_calls = 0, 0, 0
        self.held = None
        self.active_arm = None
        self.phase_receipts: list[dict[str, Any]] = []
        self.last_inference_evidence: dict[str, Any] | None = None
        self.finished = False
        self._request_sink = None

    def bind_request_evidence_sink(self, sink: Any) -> None:
        self._request_sink = sink

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        self.scripted_controller_invoked = True
        state = observation["state"]
        arms = self.plan["arm_names"]
        if (state.get("arm_names") != arms or set(state.get("arm_eef_pose", {})) != set(arms)
                or set(state.get("scripted_arm_base_pose_world", {})) != set(arms)):
            raise ValueError("scripted_baseline_named_arm_observation_invalid")
        if self._request_sink is not None:
            from .policy_request_evidence import capture_request
            self._request_sink(capture_request({"authored_task_plan_digest": self.plan["plan_digest"],
                "state": {key: state[key] for key in ("arm_names", "arm_eef_pose", "scripted_arm_base_pose_world",
                    "gripper_names", "gripper_position", "arm_gripper_names")}}, transport="in_process_scripted_controller"))
        if self.held is None:
            self.held = deepcopy(state["arm_eef_pose"])
            gripper_names = state["gripper_names"]
            self.held_closures = {arm: state["gripper_position"][gripper_names.index(state["arm_gripper_names"][arm])]
                                  for arm in arms}
            self.active_arm = self.plan["active_arm"]
            if self.active_arm == "nearest_initial_tcp":
                grasp = self.plan["phases"][0]["position_world_m"]
                distances = {}
                for arm in arms:
                    base, tcp = state["scripted_arm_base_pose_world"][arm], self.held[arm]
                    world_tcp = pose_matrix(base["position_m"], base["orientation_xyzw"]) @ pose_matrix(tcp["position_m"], tcp["orientation_xyzw"])
                    distances[arm] = float(np.linalg.norm(world_tcp[:3, 3] - grasp))
                self.active_arm = min(arms, key=lambda arm: distances[arm])
        active = self.active_arm
        phase = self.plan["phases"][self.phase_index]
        base = state["scripted_arm_base_pose_world"][active]
        transform = self.plan["grasp_bindings"][active]["tcp_from_grasp"]
        world_tcp = (pose_matrix(phase["position_world_m"], phase["orientation_world_xyzw"])
                     @ np.linalg.inv(pose_matrix(transform["position_m"], transform["orientation_xyzw"])))
        target = np.linalg.inv(pose_matrix(base["position_m"], base["orientation_xyzw"])) @ world_tcp
        observed = state["arm_eef_pose"][active]
        actual = pose_matrix(observed["position_m"], observed["orientation_xyzw"])
        distance = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        cosine = float(np.clip((np.trace(target[:3, :3] @ actual[:3, :3].T) - 1.) / 2., -1., 1.))
        arrived = distance <= self.plan["arrival_tolerance_m"] and math.acos(cosine) <= self.plan["arrival_orientation_tolerance_rad"]
        self.phase_calls += 1
        self.stable_calls = self.stable_calls + 1 if arrived else 0
        dwell = max(2, math.ceil(self.plan["gripper_dwell_s"] * self.action_hz))
        # Grip dwell is time for physical drives; it is never a contact/success
        # assertion. A blocked finger may correctly remain short of full closure.
        complete = self.stable_calls >= dwell
        timeout = self.phase_calls / self.action_hz >= self.plan["phase_timeout_s"]
        commands = deepcopy(self.held)
        commands[active] = {"position_m": target[:3, 3].tolist(), "orientation_xyzw": quaternion_xyzw(target[:3, :3])}
        closures = dict(self.held_closures)
        closures[active] = float(phase["gripper_state"] == "closed")
        result = {"schema_version": POSE_CHUNK_SCHEMA, "frame": "arm_base", "arm_names": arms,
                  "positions_m": [[commands[arm]["position_m"] for arm in arms]],
                  "orientations_xyzw": [[commands[arm]["orientation_xyzw"] for arm in arms]],
                  "gripper_closures": [[closures[arm] for arm in arms]]}
        self.last_inference_evidence = {"execution_evidence_kind": EVIDENCE_KIND,
            "scripted_controller_invoked": True, "candidate_policy_queried": False,
            "active_arm": active,
            "action_source": ACTION_SOURCE, "phase_id": phase["phase_id"], "position_error_m": distance,
            "orientation_error_rad": math.acos(cosine), "authored_task_plan_digest": self.plan["plan_digest"],
            "returned_action": deepcopy(result), "task_success_claimed": False}
        if complete or timeout:
            self.phase_receipts.append({"phase_id": phase["phase_id"], "pose_arrival_observed": complete,
                                       "timeout": timeout and not complete, "controller_calls": self.phase_calls})
            if self.phase_index < len(self.plan["phases"]) - 1:
                self.phase_index += 1
                self.phase_calls = self.stable_calls = 0
            else:
                self.finished = True
        return result

    def evidence_summary(self) -> dict[str, Any]:
        return {"execution_evidence_kind": EVIDENCE_KIND, "action_source": ACTION_SOURCE,
                "candidate_policy_queried": False, "scripted_controller_invoked": self.scripted_controller_invoked,
                "phase_receipts": deepcopy(self.phase_receipts), "authored_task_plan_digest": self.plan["plan_digest"],
                "active_arm": self.active_arm,
                "policy_outcome_interpretable": False, "physical_success_proven": False,
                "claim_boundary": "Scripted simulation control using authored task targets; independently scored, not learned/vendor-policy evidence."}
