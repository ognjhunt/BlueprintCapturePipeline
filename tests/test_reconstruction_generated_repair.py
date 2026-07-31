from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.reconstruction_enhancement_audit import enhancement_method_audit
from blueprint_pipeline.reconstruction_generated_repair import (
    GENERATED_REPAIR_REQUEST_SCHEMA_VERSION,
    GeneratedRepairContractError,
    build_generated_repair_candidate_request,
    run_generated_repair_candidate,
)
from blueprint_pipeline.task_evaluation_supervisor import (
    AutonomyMode,
    SupervisorContext,
    ToolRegistry,
)
from blueprint_pipeline.task_evaluation_supervisor.supervisor import default_authority_envelope
from blueprint_pipeline.task_evaluation_supervisor.tools import non_spend_tool_bindings


def _request(method_id: str = "artifixer") -> dict:
    return build_generated_repair_candidate_request(
        {
            "schema_version": GENERATED_REPAIR_REQUEST_SCHEMA_VERSION,
            "stable_run_identity": "generated-repair-1",
            "source_capture_digest": "sha256:" + "1" * 64,
            "method_id": method_id,
            "enhancement_method_audit": enhancement_method_audit(method_id),
            "baseline_reconstruction_digest": "sha256:" + "2" * 64,
            "frozen_split_digest": "sha256:" + "3" * 64,
            "heldout_evaluation_contract_digest": "sha256:" + "4" * 64,
            "baseline_preserved": True,
            "candidate_may_read_hidden_heldout": False,
            "generated_pixels_may_be_promoted_to_capture": False,
            "authority_used": {"local_non_spend": True},
            "timestamp": "2026-07-30T22:00:00Z",
        }
    )


@pytest.mark.parametrize("method_id", ["artifixer", "difix3d", "harmonizer"])
def test_current_generated_repair_candidates_emit_deterministic_rejection(
    method_id: str,
) -> None:
    request = _request(method_id)
    first = run_generated_repair_candidate(request)
    second = run_generated_repair_candidate(request)

    assert first == second
    assert first["status"] == "blocked_not_qualified"
    assert first["execution_started"] is False
    assert first["generated_artifact_references"] == []
    assert first["baseline_preserved"] is True
    assert first["hidden_heldout_available_to_candidate"] is False
    assert first["metric_or_collision_proof_effect"] is False
    assert first["physical_or_deployment_proof_effect"] is False
    assert first["blockers"] == enhancement_method_audit(method_id)["blockers"]


def test_generated_repair_request_rejects_fabricated_audit_and_hidden_access() -> None:
    fabricated = dict(_request())
    fabricated.pop("generated_repair_candidate_request_digest")
    fabricated["enhancement_method_audit"] = dict(fabricated["enhancement_method_audit"])
    fabricated["enhancement_method_audit"]["status"] = "qualified"
    with pytest.raises(GeneratedRepairContractError, match="audit_binding_invalid"):
        build_generated_repair_candidate_request(fabricated)

    hidden = dict(_request())
    hidden.pop("generated_repair_candidate_request_digest")
    hidden["candidate_may_read_hidden_heldout"] = True
    with pytest.raises(GeneratedRepairContractError, match="hidden_access_forbidden"):
        build_generated_repair_candidate_request(hidden)


def test_registered_generated_repair_tool_cannot_start_rejected_candidate(
    tmp_path: Path,
) -> None:
    request = _request()
    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="generated-repair-tool",
        customer_question="Try the qualified repair candidate if allowed.",
        supervisor_output_dir=str(tmp_path / "run"),
        generated_repair_candidate_request=request,
    )
    authority = default_authority_envelope(
        run_id=context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[request["generated_repair_candidate_request_digest"]],
    ).to_mapping()
    binding = next(
        binding
        for binding in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=authority,
        )
        if binding.tool_id == "run_generated_repair_candidate"
    )
    observation = binding.invoke(
        {
            "generated_repair_candidate_request_digest": request[
                "generated_repair_candidate_request_digest"
            ]
        }
    )

    assert observation["status"] == "completed"
    assert observation["typed_result"]["status"] == "blocked_not_qualified"
    assert observation["typed_result"]["execution_started"] is False
    assert observation["typed_result"]["baseline_preserved"] is True
    assert observation["proof_effect"] == "none"
