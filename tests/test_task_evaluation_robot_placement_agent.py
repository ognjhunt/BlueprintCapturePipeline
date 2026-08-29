from __future__ import annotations

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
    robot_placement_agents_sdk_config,
    run_task_evaluation_robot_placement_agent,
    validate_robot_placement_receipt,
)
from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import (
    AgentsSDKInvocationResult,
)


_DIGEST = "sha256:" + "a" * 64


def _proposal(candidate_id: str, x: float) -> RobotPlacementProposalOutput:
    return RobotPlacementProposalOutput.model_validate(
        {
            "candidate_id": candidate_id,
            "pose": {
                "position_world_m": [x, -6.1, 0.7545],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
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


def _images():
    return [
        {
            "label": "overview",
            "digest": _DIGEST,
            "image_url": "data:image/png;base64,AA==",
            "detail": "high",
        }
    ]


def _native_attempt(status: str, *, blocker: str | None = None) -> dict:
    value = {
        "schema_version": "task_evaluation_robot_placement_native_attempt.v1",
        "status": status,
        "phase_reached": "controls" if status == "passed" else "construction",
        "blockers": [blocker] if blocker else [],
        "native_result_digest": "sha256:" + "b" * 64,
        "provider_instance_id": 123,
        "provider_allocations_performed": 0,
        "native_attempt_digest": "",
    }
    value["native_attempt_digest"] = canonical_digest(
        value, digest_field="native_attempt_digest"
    )
    value["feedback_images"] = _images()
    return value


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
        max_rounds=3,
    )

    assert receipt["status"] == "accepted"
    assert receipt["accepted_candidate_id"] == "native-clear"
    assert receipt["native_agent_loop_enabled"] is True
    assert receipt["native_attempt_count"] == 2
    assert receipt["native_construction_required"] is False
    assert receipt["rounds"][0]["native_attempt"]["status"] == "rejected"
    assert receipt["rounds"][1]["native_attempt"]["status"] == "passed"
    assert provisional_receipts[0]["native_construction_required"] is True
    second_proposal_input = invoker.specs[2][1][0]["content"]
    assert sum(item["type"] == "input_image" for item in second_proposal_input) == 2
    assert validate_robot_placement_receipt(receipt) == receipt


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
    receipt["accepted_pose"]["position_world_m"][0] += 1.0

    with pytest.raises(RobotPlacementAgentError, match="receipt_invalid"):
        validate_robot_placement_receipt(receipt)
