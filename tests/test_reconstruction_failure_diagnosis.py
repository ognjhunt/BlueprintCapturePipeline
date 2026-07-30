from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.reconstruction_failure_diagnosis import (
    FAILURE_DIAGNOSIS_REQUEST_SCHEMA_VERSION,
    ReconstructionFailureDiagnosisError,
    build_reconstruction_failure_diagnosis_request,
    diagnose_reconstruction_failure,
)
from blueprint_pipeline.reconstruction_worker_contracts import FAILURE_CODES
from blueprint_pipeline.task_evaluation_supervisor import (
    AutonomyMode,
    SupervisorContext,
    ToolRegistry,
)
from blueprint_pipeline.task_evaluation_supervisor.supervisor import default_authority_envelope
from blueprint_pipeline.task_evaluation_supervisor.tools import non_spend_tool_bindings


def _request(*, repeat: bool = False, changed_configuration: bool = False) -> dict:
    attempts = [
        {
            "attempt_id": "attempt-1",
            "failure_code": "provider_capacity",
            "input_digest": "sha256:" + "3" * 64,
            "configuration_digest": "sha256:" + "4" * 64,
            "event_digest": "sha256:" + "5" * 64,
            "failed_evidence_preserved": True,
        }
    ]
    if repeat or changed_configuration:
        attempts.append(
            {
                **attempts[0],
                "attempt_id": "attempt-2",
                "configuration_digest": (
                    "sha256:" + "6" * 64
                    if changed_configuration
                    else attempts[0]["configuration_digest"]
                ),
                "event_digest": "sha256:" + "7" * 64,
            }
        )
    return build_reconstruction_failure_diagnosis_request(
        {
            "schema_version": FAILURE_DIAGNOSIS_REQUEST_SCHEMA_VERSION,
            "stable_run_identity": "reconstruction-run-1",
            "source_capture_identity": "capture-1",
            "source_capture_digest": "sha256:" + "1" * 64,
            "failed_event_digest": attempts[-1]["event_digest"],
            "stage_id": "train_gaussian_reconstruction",
            "failure_code": "provider_capacity",
            "attempt_ledger": attempts,
            "authority_state": {
                "paid_execution_authorized": True,
                "provider_execution_authorized": True,
            },
            "execution_requires_paid_compute": True,
            "execution_requires_provider": True,
            "timestamp": "2026-07-30T18:30:00Z",
        }
    )


def test_first_transient_failure_allows_one_bounded_retry() -> None:
    request = _request()
    diagnosis = diagnose_reconstruction_failure(request)
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/reconstruction_failure_diagnosis.v1.schema.json"
        ).read_text()
    )
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(request)
    validator.validate(diagnosis)

    assert diagnosis["diagnosed_failure_code"] == "provider_capacity"
    assert diagnosis["identical_attempt_count"] == 1
    assert diagnosis["unchanged_deterministic_retry_allowed"] is True
    assert diagnosis["terminal_for_current_configuration"] is False
    assert "retry_once_same_worker" in diagnosis["legal_next_actions"]
    assert diagnosis["recovery_executed"] is False
    assert diagnosis["proof_effect"] == "none"


def test_second_identical_failure_stops_loop_but_changed_configuration_does_not() -> None:
    repeated = diagnose_reconstruction_failure(_request(repeat=True))
    changed = diagnose_reconstruction_failure(_request(changed_configuration=True))

    assert repeated["diagnosed_failure_code"] == "repeated_identical_blocker"
    assert repeated["identical_attempt_count"] == 2
    assert repeated["unchanged_deterministic_retry_allowed"] is False
    assert repeated["terminal_for_current_configuration"] is True
    assert repeated["legal_next_actions"] == ["preserve_evidence_and_stop", "abstain"]
    assert changed["diagnosed_failure_code"] == "provider_capacity"
    assert changed["identical_attempt_count"] == 1


def test_failure_request_rejects_unknown_code_and_suppressed_failed_evidence() -> None:
    unknown = dict(_request())
    unknown.pop("reconstruction_failure_diagnosis_request_digest")
    unknown["failure_code"] = "invented_success"
    with pytest.raises(ReconstructionFailureDiagnosisError, match="failure_code_invalid"):
        build_reconstruction_failure_diagnosis_request(unknown)

    suppressed = dict(_request())
    suppressed.pop("reconstruction_failure_diagnosis_request_digest")
    suppressed["attempt_ledger"] = [dict(suppressed["attempt_ledger"][0])]
    suppressed["attempt_ledger"][0]["failed_evidence_preserved"] = False
    with pytest.raises(ReconstructionFailureDiagnosisError, match="evidence_not_preserved"):
        build_reconstruction_failure_diagnosis_request(suppressed)


def test_missing_execution_authority_never_becomes_agent_granted_retry() -> None:
    request = dict(_request())
    request.pop("reconstruction_failure_diagnosis_request_digest")
    request["authority_state"] = {
        "paid_execution_authorized": False,
        "provider_execution_authorized": False,
    }

    diagnosis = diagnose_reconstruction_failure(request)

    assert diagnosis["authority_requested_not_granted"] is True
    assert diagnosis["unchanged_deterministic_retry_allowed"] is False
    assert diagnosis["legal_next_actions"][0] == "request_additional_authority"
    assert "retry_once_same_worker" not in diagnosis["legal_next_actions"]
    assert diagnosis["agent_granted_authority"] is False


def test_every_registered_worker_failure_has_a_deterministic_recovery_policy() -> None:
    for index, failure_code in enumerate(sorted(FAILURE_CODES)):
        request = dict(_request())
        request.pop("reconstruction_failure_diagnosis_request_digest")
        request["failure_code"] = failure_code
        request["attempt_ledger"] = [dict(request["attempt_ledger"][0])]
        request["attempt_ledger"][0]["failure_code"] = failure_code
        request["attempt_ledger"][0]["event_digest"] = (
            "sha256:" + f"{index + 10:064x}"
        )
        request["failed_event_digest"] = request["attempt_ledger"][0]["event_digest"]

        diagnosis = diagnose_reconstruction_failure(request)

        assert diagnosis["diagnosed_failure_code"] == failure_code
        assert diagnosis["legal_next_actions"]


def test_registered_failure_diagnosis_uses_only_frozen_request_digest(
    tmp_path: Path,
) -> None:
    request = _request(repeat=True)
    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="reconstruction-failure-tool",
        customer_question="Why did reconstruction fail and what is legal next?",
        supervisor_output_dir=str(tmp_path / "run"),
        reconstruction_failure_diagnosis_request=request,
    )
    authority = default_authority_envelope(
        run_id=context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[
            request["reconstruction_failure_diagnosis_request_digest"]
        ],
    ).to_mapping()
    binding = next(
        binding
        for binding in non_spend_tool_bindings(
            capability="runtime_failure_recovery",
            context=context,
            registry=registry,
            authority=authority,
        )
        if binding.tool_id == "diagnose_reconstruction_failure"
    )

    assert set(binding.input_schema["properties"]) == {
        "reconstruction_failure_diagnosis_request_digest"
    }
    observation = binding.invoke(
        {
            "reconstruction_failure_diagnosis_request_digest": request[
                "reconstruction_failure_diagnosis_request_digest"
            ]
        }
    )

    assert observation["status"] == "completed"
    assert observation["typed_result"]["diagnosed_failure_code"] == (
        "repeated_identical_blocker"
    )
    assert observation["typed_result"]["unchanged_deterministic_retry_allowed"] is False
    assert observation["typed_result"]["failed_evidence_preserved"] is True
    assert observation["proof_effect"] == "none"
