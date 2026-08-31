from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_robot_placement_agent import (
    ROBOT_PLACEMENT_AGENT_MAX_OUTPUT_TOKENS,
    ROBOT_PLACEMENT_AGENT_MODEL,
    ROBOT_PLACEMENT_AGENT_REASONING_EFFORT,
    RobotPlacementAgentError,
    RobotPlacementProposalOutput,
    RobotPlacementVisualReviewOutput,
    _exact_inventory_member,
    _validated_gate,
    robot_placement_agents_sdk_config,
    run_task_evaluation_robot_placement_agent,
    validate_robot_placement_receipt,
)
from blueprint_pipeline.task_evaluation_robot_placement_trajectory import (
    placement_trajectory_from_native_plan,
)
from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import (
    AgentsSDKInvocationResult,
)


_DIGEST = "sha256:" + "a" * 64


def _proposal(
    candidate_id: str,
    x: float,
    *,
    orientation_xyzw: list[float] | None = None,
) -> RobotPlacementProposalOutput:
    return RobotPlacementProposalOutput.model_validate(
        {
            "candidate_id": candidate_id,
            "pose": {
                "position_world_m": [x, -6.1, 0.7545],
                "orientation_xyzw": orientation_xyzw or [0.0, 0.0, 0.0, 1.0],
            },
            "support_surface_id": "/Scene/CounterTop",
            "rationale": "Mount on the observed counter beside the task object.",
            "addressed_blockers": ["base_embedded_in_support"],
            "uncertainty": "Native reset-state collision remains to be checked.",
        }
    )


def _visual(status: str) -> RobotPlacementVisualReviewOutput:
    passed = status == "passed"
    return RobotPlacementVisualReviewOutput.model_validate(
        {
            "status": status,
            "robot_supported_by_declared_surface": passed,
            "robot_not_visibly_clipping_site_geometry": passed,
            "robot_faces_task_workspace": passed,
            "task_workspace_visually_reachable": passed,
            "camera_views_are_sufficient": True,
            "reason": "Placement is visually coherent." if passed else "Robot clips the counter.",
            "revision_guidance": [] if passed else ["Move fully onto the support surface."],
        }
    )


@dataclass
class _Invoker:
    outputs: list[object]

    def __post_init__(self) -> None:
        self.specs = []

    def invoke(self, spec, input_value):
        self.specs.append((spec, input_value))
        output = self.outputs.pop(0)
        return AgentsSDKInvocationResult(
            output=output,
            provider="openai",
            model=spec.model,
            sdk_version="0.19.1",
            latency_seconds=0.01,
            usage={"total_tokens": 20},
            cost_usd=None,
            cost_status="test",
            trace_id="trace_test",
        )


def _gate(candidate_id: str, status: str) -> dict:
    value = {
        "schema_version": "task_evaluation_robot_placement_geometry_gate.v1",
        "candidate_id": candidate_id,
        "status": status,
        "blockers": [] if status == "passed" else ["base_collision_detected"],
        "support_passed": status == "passed",
        "collision_passed": status == "passed",
        "reachability_passed": status == "passed",
        "facing_passed": status == "passed",
        "geometry_gate_digest": "",
    }
    value["geometry_gate_digest"] = canonical_digest(
        value, digest_field="geometry_gate_digest"
    )
    return value


def test_geometry_gate_validation_reports_exact_failed_contract() -> None:
    gate = _gate("candidate-17", "passed")
    gate["geometry_gate_digest"] = "sha256:" + "0" * 64

    with pytest.raises(
        RobotPlacementAgentError,
        match="robot_placement_geometry_gate_digest_mismatch:candidate-17",
    ):
        _validated_gate(gate)


def _images():
    return [
        {
            "label": "overview",
            "digest": _DIGEST,
            "image_url": "data:image/png;base64,AA==",
            "detail": "high",
        }
    ]


def _native_attempt(
    status: str,
    *,
    blocker: str | None = None,
    robot_root_pose_world: list[float] | None = None,
) -> dict:
    value = {
        "schema_version": "task_evaluation_robot_placement_native_attempt.v1",
        "status": status,
        "phase_reached": "controls" if status == "passed" else "construction",
        "blockers": [blocker] if blocker else [],
        "native_result_digest": "sha256:" + "b" * 64,
        "provider_instance_id": 123,
        "provider_allocations_performed": 0,
        "native_feedback": (
            {"initial_robot_root_pose_world": robot_root_pose_world}
            if robot_root_pose_world is not None
            else {}
        ),
        "native_attempt_digest": "",
    }
    value["native_attempt_digest"] = canonical_digest(
        value, digest_field="native_attempt_digest"
    )
    value["feedback_images"] = _images()
    return value


def _trajectory() -> dict:
    plan = {
        "schema_version": "native_rigid_construction_phase_plan.v1",
        "task_kind": "rigid_pick_place",
        "manipulation_strategy": "planar_push",
        "phase_count": 2,
        "execution_parameters": {
            "arrival_tolerance_m": 0.02,
            "arrival_orientation_tolerance_rad": 0.08,
        },
        "phases": [
            {
                "phase_id": "precontact",
                "position_world_m": [2.79, -6.76, 0.818],
                "orientation_world_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
                "gripper_state": "open",
                "gate_ids": ["precontact_reachability"],
            },
            {
                "phase_id": "push_contact",
                "position_world_m": [2.91, -6.76, 0.818],
                "orientation_world_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
                "gripper_state": "closed",
                "gate_ids": ["push_contact"],
            },
        ],
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return placement_trajectory_from_native_plan(plan)


def _inventory_context(proposal: RobotPlacementProposalOutput) -> dict:
    value = proposal.model_dump(mode="json")
    member = {
        "candidate_id": value["candidate_id"],
        "pose": value["pose"],
        "support_surface_id": value["support_surface_id"],
        "geometry_gate_digest": _DIGEST,
        "trajectory_position_ik_gate_digest": "sha256:" + "b" * 64,
    }
    inventory = [member]
    return {
        "deterministic_geometry_passing_candidate_inventory": inventory,
        "deterministic_geometry_passing_candidate_inventory_digest": canonical_digest(
            {"trajectory_digest": None, "candidates": inventory}
        ),
        "deterministic_geometry_passing_candidate_inventory_trajectory_digest": None,
    }


def test_exact_inventory_member_accepts_identical_selection() -> None:
    proposal = _proposal("geometry_0000", 3.4).model_dump(mode="json")
    member = _exact_inventory_member(
        proposal, scene_context=_inventory_context(_proposal("geometry_0000", 3.4))
    )
    assert member is not None
    assert member["candidate_id"] == "geometry_0000"


def test_exact_inventory_member_rejects_unknown_candidate() -> None:
    with pytest.raises(
        RobotPlacementAgentError, match="robot_placement_candidate_not_in_inventory"
    ):
        _exact_inventory_member(
            _proposal("unknown", 3.4).model_dump(mode="json"),
            scene_context=_inventory_context(_proposal("geometry_0000", 3.4)),
        )


@pytest.mark.parametrize("mutation", ["position", "orientation", "support"])
def test_exact_inventory_member_rejects_pose_or_support_mutation(mutation) -> None:
    selected = _proposal("geometry_0000", 3.4).model_dump(mode="json")
    if mutation == "position":
        selected["pose"]["position_world_m"][0] += 0.001
    elif mutation == "orientation":
        selected["pose"]["orientation_xyzw"] = [0.0, 0.0, 1.0, 0.0]
    else:
        selected["support_surface_id"] = "/Scene/OtherSurface"
    with pytest.raises(
        RobotPlacementAgentError,
        match="robot_placement_candidate_inventory_member_mutated",
    ):
        _exact_inventory_member(
            selected,
            scene_context=_inventory_context(_proposal("geometry_0000", 3.4)),
        )


def test_production_config_pins_sol_high_agent_contract() -> None:
    config = robot_placement_agents_sdk_config(
        max_inference_cost_usd=0.5,
        allow_live_invocation=True,
        tracing_disabled=True,
    )

    assert config.model == "gpt-5.6-sol"
    assert config.max_inference_cost_usd == 0.5
    assert config.input_cost_per_million_tokens_usd == 4.0
    assert config.output_cost_per_million_tokens_usd == 20.0
    assert config.max_output_tokens == ROBOT_PLACEMENT_AGENT_MAX_OUTPUT_TOKENS


def test_loop_revises_geometry_failure_and_freezes_only_dual_pass() -> None:
    invoker = _Invoker(
        outputs=[
            _proposal("embedded", 2.7),
            _proposal("supported", 3.4),
            _visual("passed"),
        ]
    )

    receipt = run_task_evaluation_robot_placement_agent(
        invoker=invoker,
        run_id="placement-test",
        scene_binding={"scene": "839873"},
        task_binding={"task": "move mug"},
        overview_images=_images(),
        validate_candidate=lambda proposal: _gate(
            proposal["candidate_id"],
            "rejected" if proposal["candidate_id"] == "embedded" else "passed",
        ),
        render_candidate=lambda _proposal, _round: _images(),
        max_rounds=3,
    )

    assert receipt["status"] == "accepted"
    assert receipt["accepted_candidate_id"] == "supported"
    assert receipt["round_count"] == 2
    assert receipt["rounds"][0]["geometry_gate"]["status"] == "rejected"
    assert receipt["rounds"][0]["visual_review"] is None
    assert receipt["rounds"][1]["visual_review"]["status"] == "passed"
    assert all(spec.model == ROBOT_PLACEMENT_AGENT_MODEL for spec, _ in invoker.specs)
    assert all(
        spec.reasoning_effort == ROBOT_PLACEMENT_AGENT_REASONING_EFFORT
        for spec, _ in invoker.specs
    )
    assert all(
        spec.max_output_tokens == ROBOT_PLACEMENT_AGENT_MAX_OUTPUT_TOKENS
        for spec, _ in invoker.specs
    )
    assert all(spec.stable_prefix_tokens >= 1_024 for spec, _ in invoker.specs)
    assert all(spec.stable_developer_prefix for spec, _ in invoker.specs)
    assert all(spec.scene_static_prefix for spec, _ in invoker.specs)
    assert all("placement-test" not in spec.stable_developer_prefix for spec, _ in invoker.specs)
    assert all("placement-test" not in spec.scene_static_prefix for spec, _ in invoker.specs)
    assert validate_robot_placement_receipt(receipt) == receipt


def test_visual_review_can_veto_but_cannot_override_geometry() -> None:
    invoker = _Invoker(outputs=[_proposal("candidate", 3.4), _visual("rejected")])

    receipt = run_task_evaluation_robot_placement_agent(
        invoker=invoker,
        run_id="placement-visual-veto",
        scene_binding={"scene": "839873"},
        task_binding={"task": "move mug"},
        overview_images=_images(),
        validate_candidate=lambda proposal: _gate(proposal["candidate_id"], "passed"),
        render_candidate=lambda _proposal, _round: _images(),
        max_rounds=1,
    )

    assert receipt["status"] == "blocked"
    assert receipt["accepted_pose"] is None
    with pytest.raises(RobotPlacementAgentError, match="receipt_invalid"):
        validate_robot_placement_receipt(receipt)


def test_agent_creates_next_pose_from_native_failure_until_controls_pass() -> None:
    invoker = _Invoker(
        outputs=[
            _proposal("native-collision", 3.4),
            _visual("passed"),
            _proposal("native-clear", 3.7),
            _visual("passed"),
        ]
    )
    native_results = [
        _native_attempt("rejected", blocker="native_reset_collision"),
        _native_attempt("passed"),
    ]
    provisional_receipts: list[dict] = []

    def execute_candidate(_proposal_value, receipt, _round_index):
        provisional_receipts.append(receipt)
        return native_results.pop(0)

    receipt = run_task_evaluation_robot_placement_agent(
        invoker=invoker,
        run_id="placement-native-loop",
        scene_binding={"scene": "839873"},
        task_binding={"task": "move mug"},
        overview_images=_images(),
        validate_candidate=lambda proposal: _gate(
            proposal["candidate_id"], "passed"
        ),
        render_candidate=lambda _proposal, _round: _images(),
        execute_candidate=execute_candidate,
        task_trajectory=_trajectory(),
        max_rounds=3,
    )

    assert receipt["status"] == "accepted"
    assert receipt["accepted_candidate_id"] == "native-clear"
    assert receipt["native_agent_loop_enabled"] is True
    assert receipt["native_attempt_count"] == 2
    assert receipt["native_construction_required"] is False
    assert receipt["task_trajectory_digest"] == _trajectory()["trajectory_digest"]
    assert receipt["rounds"][0]["native_attempt"]["status"] == "rejected"
    assert receipt["rounds"][1]["native_attempt"]["status"] == "passed"
    assert provisional_receipts[0]["native_construction_required"] is True
    second_proposal_input = invoker.specs[2][1][0]["content"]
    assert sum(item["type"] == "input_image" for item in second_proposal_input) == 2
    second_prompt = next(
        item["text"] for item in second_proposal_input if item["type"] == "input_text"
    )
    assert '"phase_id": "precontact"' not in second_prompt
    assert (
        '"phase_id":"precontact"'
        in str(invoker.specs[2][0].scene_static_prefix)
    )
    assert "native_reset_collision" in second_prompt
    assert validate_robot_placement_receipt(receipt) == receipt


def test_agent_cannot_reuse_a_pose_rejected_by_native_execution() -> None:
    invoker = _Invoker(
        outputs=[
            _proposal("native-collision", 3.4),
            _visual("passed"),
            _proposal("same-pose-again", 3.4),
            _proposal("native-clear", 3.7),
            _visual("passed"),
        ]
    )
    native_results = [
        _native_attempt("rejected", blocker="native_reset_collision"),
        _native_attempt("passed"),
    ]

    receipt = run_task_evaluation_robot_placement_agent(
        invoker=invoker,
        run_id="placement-native-repeat-guard",
        scene_binding={"scene": "839873"},
        task_binding={"task": "move mug"},
        overview_images=_images(),
        validate_candidate=lambda proposal: _gate(
            proposal["candidate_id"], "passed"
        ),
        render_candidate=lambda _proposal, _round: _images(),
        execute_candidate=lambda _proposal, _receipt, _round: native_results.pop(0),
        task_trajectory=_trajectory(),
        max_rounds=3,
    )

    assert receipt["status"] == "accepted"
    assert receipt["accepted_candidate_id"] == "native-clear"
    assert receipt["native_attempt_count"] == 2
    assert receipt["rounds"][1]["geometry_gate"]["status"] == "rejected"
    assert receipt["rounds"][1]["geometry_gate"]["blockers"] == [
        "prior_native_pose_reused"
    ]
    assert receipt["rounds"][1]["visual_review"] is None


def test_agent_can_retry_same_position_with_materially_different_yaw() -> None:
    invoker = _Invoker(
        outputs=[
            _proposal("native-yaw-zero", 3.4),
            _visual("passed"),
            _proposal(
                "native-yaw-twenty-degrees",
                3.4,
                orientation_xyzw=[0.0, 0.0, 0.1736481777, 0.9848077530],
            ),
            _visual("passed"),
        ]
    )
    native_results = [
        _native_attempt("rejected", blocker="native_task_phase_ik_unreached:precontact"),
        _native_attempt("passed"),
    ]

    receipt = run_task_evaluation_robot_placement_agent(
        invoker=invoker,
        run_id="placement-native-yaw-retry",
        scene_binding={"scene": "839873"},
        task_binding={"task": "move mug"},
        overview_images=_images(),
        validate_candidate=lambda proposal: _gate(
            proposal["candidate_id"], "passed"
        ),
        render_candidate=lambda _proposal, _round: _images(),
        execute_candidate=lambda _proposal, _receipt, _round: native_results.pop(0),
        task_trajectory=_trajectory(),
        max_rounds=2,
    )

    assert receipt["status"] == "accepted"
    assert receipt["accepted_candidate_id"] == "native-yaw-twenty-degrees"
    assert receipt["native_attempt_count"] == 2


def test_native_loop_refuses_point_only_context_without_exact_trajectory() -> None:
    invoker = _Invoker(outputs=[])

    with pytest.raises(
        RobotPlacementAgentError, match="robot_placement_native_trajectory_missing"
    ):
        run_task_evaluation_robot_placement_agent(
            invoker=invoker,
            run_id="placement-native-no-trajectory",
            scene_binding={"scene": "839873"},
            task_binding={"task": "move mug"},
            overview_images=_images(),
            validate_candidate=lambda proposal: _gate(
                proposal["candidate_id"], "passed"
            ),
            render_candidate=lambda _proposal, _round: _images(),
            execute_candidate=lambda _proposal, _receipt, _round: _native_attempt(
                "passed"
            ),
        )
def test_agent_honors_rejected_native_pose_history_from_prior_run() -> None:
    invoker = _Invoker(
        outputs=[
            _proposal("prior-native-pose", 3.4),
            _proposal("new-pose", 3.7),
            _visual("passed"),
        ]
    )

    receipt = run_task_evaluation_robot_placement_agent(
        invoker=invoker,
        run_id="placement-prior-run-repeat-guard",
        scene_binding={"scene": "839873"},
        task_binding={"task": "move mug"},
        scene_context={
            "rejected_native_base_poses": [
                {"position_world_m": [3.4, -6.1, 0.7545]}
            ]
        },
        overview_images=_images(),
        validate_candidate=lambda proposal: _gate(
            proposal["candidate_id"], "passed"
        ),
        render_candidate=lambda _proposal, _round: _images(),
        max_rounds=2,
    )

    assert receipt["status"] == "accepted"
    assert receipt["accepted_candidate_id"] == "new-pose"
    assert receipt["rounds"][0]["geometry_gate"]["blockers"] == [
        "prior_native_pose_reused"
    ]


def test_agent_resumes_prior_native_metrics_frames_and_rejected_pose() -> None:
    invoker = _Invoker(
        outputs=[
            _proposal("prior-native-pose", 3.4),
            _proposal("new-pose", 3.7),
            _visual("passed"),
        ]
    )
    prior = _native_attempt(
        "rejected",
        blocker="native_task_phase_ik_unreached:precontact",
        robot_root_pose_world=[3.4, -6.1, 0.7545, 0.0, 0.0, 0.0, 1.0],
    )

    receipt = run_task_evaluation_robot_placement_agent(
        invoker=invoker,
        run_id="placement-prior-native-resume",
        scene_binding={"scene": "839873"},
        task_binding={"task": "move mug"},
        overview_images=_images(),
        validate_candidate=lambda proposal: _gate(
            proposal["candidate_id"], "passed"
        ),
        render_candidate=lambda _proposal, _round: _images(),
        task_trajectory=_trajectory(),
        prior_native_attempts=[prior],
        max_rounds=2,
    )

    assert receipt["status"] == "accepted"
    assert receipt["accepted_candidate_id"] == "new-pose"
    assert receipt["prior_native_attempt_count"] == 1
    assert receipt["prior_native_attempts"][0]["blockers"] == [
        "native_task_phase_ik_unreached:precontact"
    ]
    assert receipt["rounds"][0]["geometry_gate"]["blockers"] == [
        "prior_native_pose_reused"
    ]
    first_proposal_content = invoker.specs[0][1][0]["content"]
    assert sum(
        item["type"] == "input_image" for item in first_proposal_content
    ) == 2
    first_prompt = next(
        item["text"]
        for item in first_proposal_content
        if item["type"] == "input_text"
    )
    assert "native_task_phase_ik_unreached:precontact" in first_prompt
    assert '"initial_robot_root_pose_world": [3.4, -6.1, 0.7545' in first_prompt


def test_receipt_tampering_is_rejected() -> None:
    invoker = _Invoker(outputs=[_proposal("candidate", 3.4), _visual("passed")])
    receipt = run_task_evaluation_robot_placement_agent(
        invoker=invoker,
        run_id="placement-tamper",
        scene_binding={"scene": "839873"},
        task_binding={"task": "move mug"},
        overview_images=_images(),
        validate_candidate=lambda proposal: _gate(proposal["candidate_id"], "passed"),
        render_candidate=lambda _proposal, _round: _images(),
        max_rounds=1,
    )
    receipt = json.loads(json.dumps(receipt))
    receipt["accepted_pose"]["position_world_m"][0] += 1.0
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )

    with pytest.raises(RobotPlacementAgentError, match="receipt_acceptance_invalid"):
        validate_robot_placement_receipt(receipt)


def test_receipt_exposes_the_exact_candidate_inventory_binding() -> None:
    proposal = _proposal("inventory-member", 3.4)
    context = _inventory_context(proposal)
    invoker = _Invoker(outputs=[proposal, _visual("passed")])

    receipt = run_task_evaluation_robot_placement_agent(
        invoker=invoker,
        run_id="placement-inventory-binding",
        scene_binding={"scene": "839873"},
        task_binding={"task": "move mug"},
        scene_context=context,
        overview_images=_images(),
        validate_candidate=lambda selected: _gate(
            selected["candidate_id"], "passed"
        ),
        render_candidate=lambda _proposal, _round: _images(),
        max_rounds=1,
    )

    assert receipt["candidate_inventory_digest"] == context[
        "deterministic_geometry_passing_candidate_inventory_digest"
    ]
    assert receipt["candidate_inventory_trajectory_digest"] is None
    assert validate_robot_placement_receipt(receipt) == receipt
