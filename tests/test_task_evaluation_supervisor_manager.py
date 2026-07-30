from __future__ import annotations

from typing import Any

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import (
    AgentsSDKInvocationResult,
    OpenAIAgentsSDKConfig,
)
from blueprint_pipeline.task_evaluation_supervisor.capabilities import SupervisorContext
from blueprint_pipeline.task_evaluation_supervisor.contracts import (
    CapabilityKind,
    CapabilityResult,
)
from blueprint_pipeline.task_evaluation_supervisor.manager import (
    AgentsSDKSupervisorManagerOutput,
    OpenAIAgentsSDKSupervisorManager,
    SUPERVISOR_MANAGER_CAPABILITY_ID,
    SupervisorManagerError,
    validate_manager_decision,
)


class _ManagerInvoker:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def invoke(self, spec, input_text: str) -> AgentsSDKInvocationResult:
        import json

        payload = json.loads(input_text)
        self.calls.append({"spec": spec, "payload": payload})
        output = self.outputs.pop(0)
        return AgentsSDKInvocationResult(
            output=spec.output_type.model_validate(output),
            provider="openai_agents_sdk_fixture",
            model=spec.model,
            sdk_version="0.18.3",
            latency_seconds=0.001,
            usage={"requests": 1},
            cost_usd=0.0,
            cost_status="fixture",
        )


def _result(kind: CapabilityKind, *, status: str = "proposed") -> CapabilityResult:
    return CapabilityResult.from_mapping(
        {
            "schema_version": "task_evaluation_supervisor_capability_result.v1",
            "result_id": f"manager-test-{kind.value}",
            "run_id": "manager-test",
            "capability": kind.value,
            "status": status,
            "artifact": {"schema_version": f"{kind.value}.fixture.v1"},
            "proposals": [],
            "proposal_dispositions": [],
            "blockers": ["fixture_blocker"] if status == "blocked" else [],
            "evidence_refs": [],
            "authoritative": False,
            "proof_booleans_mutable": False,
            "proof_effect": "none",
        }
    )


def _manager(outputs: list[dict[str, Any]]):
    invoker = _ManagerInvoker(outputs)
    manager = OpenAIAgentsSDKSupervisorManager(
        invoker=invoker,
        config=OpenAIAgentsSDKConfig(),
        tool_registry_digest="sha256:" + "a" * 64,
    )
    return manager, invoker


def test_manager_agent_selects_first_specialist_then_replans_from_result() -> None:
    claim = _result(CapabilityKind.CLAIM_TASK_INTERPRETER)
    manager, invoker = _manager(
        [
            {
                "status": "continue",
                "step_index": 0,
                "next_capability": "claim_task_interpreter",
                "terminal_reason": None,
                "rationale": "Interpret the request before routing later work.",
                "observed_capability_result_digests": [],
                "uncertainty": "not_a_proof_signal",
            },
            {
                "status": "continue",
                "step_index": 1,
                "next_capability": "capture_testbed_supervisor",
                "terminal_reason": None,
                "rationale": "Inspect capture sufficiency after interpretation.",
                "observed_capability_result_digests": [claim.digest],
                "uncertainty": "not_a_proof_signal",
            },
        ]
    )
    context = SupervisorContext(
        run_id="manager-test",
        customer_question="Can this robot complete the task?",
        capture_build={"capture_build_digest": "sha256:" + "b" * 64},
    )

    first = manager.choose_next(context=context, completed_results=[], step_index=0)
    second = manager.choose_next(
        context=context,
        completed_results=[claim],
        step_index=1,
    )

    assert first.value["next_capability"] == "claim_task_interpreter"
    assert second.value["next_capability"] == "capture_testbed_supervisor"
    assert second.value["observed_capability_result_digests"] == [claim.digest]
    assert all(
        call["spec"].capability == SUPERVISOR_MANAGER_CAPABILITY_ID for call in invoker.calls
    )
    assert all(
        call["spec"].output_type is AgentsSDKSupervisorManagerOutput for call in invoker.calls
    )

    forged_menu = dict(first.value)
    forged_menu["eligible_next_capabilities"] = ["post_run_diagnostician"]
    forged_menu["supervisor_manager_decision_digest"] = canonical_digest(
        forged_menu,
        digest_field="supervisor_manager_decision_digest",
    )
    with pytest.raises(SupervisorManagerError, match="eligible_capabilities_mismatch"):
        validate_manager_decision(
            forged_menu,
            context=context,
            completed_results=[],
            step_index=0,
        )


def test_manager_rejects_unavailable_or_repeated_specialist() -> None:
    manager, _ = _manager(
        [
            {
                "status": "continue",
                "step_index": 0,
                "next_capability": "post_run_diagnostician",
                "terminal_reason": None,
                "rationale": "Attempt to skip required interpretation.",
                "observed_capability_result_digests": [],
                "uncertainty": "not_a_proof_signal",
            }
        ]
    )
    with pytest.raises(SupervisorManagerError, match="next_capability_not_eligible"):
        manager.choose_next(
            context=SupervisorContext(
                run_id="manager-test",
                customer_question="What can be claimed?",
            ),
            completed_results=[],
            step_index=0,
        )


def test_manager_rejects_false_observation_and_unearned_terminal_decision() -> None:
    claim = _result(CapabilityKind.CLAIM_TASK_INTERPRETER)
    missing_observation, _ = _manager(
        [
            {
                "status": "continue",
                "step_index": 1,
                "next_capability": "capture_testbed_supervisor",
                "terminal_reason": None,
                "rationale": "Pretend the prior result was observed.",
                "observed_capability_result_digests": [],
                "uncertainty": "not_a_proof_signal",
            }
        ]
    )
    context = SupervisorContext(
        run_id="manager-test",
        customer_question="What can be claimed?",
    )
    with pytest.raises(SupervisorManagerError, match="observed_result_set_mismatch"):
        missing_observation.choose_next(
            context=context,
            completed_results=[claim],
            step_index=1,
        )

    false_terminal, _ = _manager(
        [
            {
                "status": "terminal",
                "step_index": 1,
                "next_capability": None,
                "terminal_reason": "decision_ready",
                "rationale": "Invent a decision without a Decision Envelope.",
                "observed_capability_result_digests": [claim.digest],
                "uncertainty": "not_a_proof_signal",
            }
        ]
    )
    with pytest.raises(SupervisorManagerError, match="terminal_reason_not_eligible"):
        false_terminal.choose_next(
            context=context,
            completed_results=[claim],
            step_index=1,
        )

    premature_abstention, _ = _manager(
        [
            {
                "status": "terminal",
                "step_index": 1,
                "next_capability": None,
                "terminal_reason": "abstention",
                "rationale": "Stop even though capture inspection is still eligible.",
                "observed_capability_result_digests": [claim.digest],
                "uncertainty": "not_a_proof_signal",
            }
        ]
    )
    with pytest.raises(SupervisorManagerError, match="terminal_reason_not_eligible"):
        premature_abstention.choose_next(
            context=SupervisorContext(
                run_id="manager-test",
                customer_question="What can be claimed?",
                capture_build={"capture_build_digest": "sha256:" + "b" * 64},
            ),
            completed_results=[claim],
            step_index=1,
        )


def test_manager_requires_post_run_diagnosis_before_terminal_decision() -> None:
    claim = _result(CapabilityKind.CLAIM_TASK_INTERPRETER)
    capture = _result(CapabilityKind.CAPTURE_TESTBED_SUPERVISOR)
    manager, _ = _manager(
        [
            {
                "status": "terminal",
                "step_index": 2,
                "next_capability": None,
                "terminal_reason": "decision_ready",
                "rationale": "A deterministic decision exists.",
                "observed_capability_result_digests": sorted([claim.digest, capture.digest]),
                "uncertainty": "not_a_proof_signal",
            }
        ]
    )
    with pytest.raises(SupervisorManagerError, match="terminal_reason_not_eligible"):
        manager.choose_next(
            context=SupervisorContext(
                run_id="manager-test",
                customer_question="What is the decision?",
                decision_envelope={"overall_outcome": "decision"},
            ),
            completed_results=[claim, capture],
            step_index=2,
        )
