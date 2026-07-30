import json
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.openai_candidate_paid_admission import (
    OPENAI_API_CANDIDATE_ADMISSION_SCHEMA_VERSION,
    build_openai_api_candidate_admission,
)
from blueprint_pipeline.paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    require_paid_resource_admission_grant,
)
from blueprint_pipeline.task_evaluation_supervisor.candidate_policy import (
    CANDIDATE_EVALUATION_SUITE_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_supervisor.phase2_artifacts import (
    AUTHORIZATION_RECEIPT_SCHEMA_VERSION,
)


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
COMMIT = "d" * 40


def _inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    suite: dict[str, Any] = {
        "schema_version": CANDIDATE_EVALUATION_SUITE_SCHEMA_VERSION,
        "hidden_label_manifest_digest": SHA_C,
        "candidate_evaluation_run_specs": [
            {
                "policy_adapter": {
                    "policy_id": "pigey-verify-recover",
                    "candidate_policy_manifest_digest": SHA_A,
                    "runtime_configuration_digest": SHA_B,
                    "max_cost_usd": 1.25,
                    "retry_limit": 1,
                    "hidden_labels_included": False,
                },
                "metadata": {"candidate_policy_manifest_digest": SHA_A},
            }
        ],
        "hidden_labels_sent_to_candidates": False,
        "candidate_agents_control_evaluator": False,
        "candidate_agents_grade_themselves": False,
        "proof_effect": "none",
    }
    suite["candidate_evaluation_suite_digest"] = canonical_digest(
        suite,
        digest_field="candidate_evaluation_suite_digest",
    )
    authorization: dict[str, Any] = {
        "schema_version": AUTHORIZATION_RECEIPT_SCHEMA_VERSION,
        "approved": True,
        "issued_by_agent": False,
        "operator_id": "candidate-operations-owner",
        "granted_tool_id": "execute_candidate_policy_suite",
        "granted_action_ids": ["pigey-verify-recover"],
        "granted_provider_ids": ["pigey_external_candidate"],
        "immutable_input_digests": [
            suite["candidate_evaluation_suite_digest"],
            SHA_C,
        ],
        "granted_max_cost_usd": 1.25,
        "granted_retry_count": 1,
        "granted_ttl_seconds": 600,
        "issued_at": "2026-07-30T15:00:00Z",
        "expires_at": "2026-07-30T15:10:00Z",
        "authorization_request_digest": SHA_A,
        "proof_effect": "none",
    }
    authorization["authorization_receipt_digest"] = canonical_digest(
        authorization,
        digest_field="authorization_receipt_digest",
    )
    return suite, authorization


def _build(**overrides: Any) -> dict[str, Any]:
    suite, authorization = _inputs()
    values: dict[str, Any] = {
        "suite": suite,
        "execution_authorization": authorization,
        "candidate_id": "pigey-verify-recover",
        "provider_id": "pigey_external_candidate",
        "runtime_configuration_digest": SHA_B,
        "cost_authority_binding_digest": SHA_C,
        "license_attestation_digest": SHA_A,
        "expected_source_commit": COMMIT,
        "checkout_source_commit": COMMIT,
        "checkout_clean": True,
        "maximum_execution_seconds": 590,
        "runtime_watchdog_enforced": True,
        "teardown_enforced": True,
        "execute_requested": True,
        "admitted_at": "2026-07-30T15:00:01Z",
    }
    values.update(overrides)
    return build_openai_api_candidate_admission(**values)


def test_openai_candidate_admission_binds_every_paid_execution_boundary() -> None:
    admission = _build()

    assert admission["schema_version"] == OPENAI_API_CANDIDATE_ADMISSION_SCHEMA_VERSION
    assert admission["status"] == "admitted"
    assert admission["blockers"] == []
    assert admission["expected_source_commit"] == COMMIT
    assert admission["maximum_execution_seconds"] == 590
    assert admission["candidate_max_cost_usd"] == 1.25
    assert admission["granted_ttl_seconds"] == 600
    assert admission["provider_mutations_performed"] == 0
    assert admission["persistent_provider_resource_created"] is False
    assert admission["candidate_reported_cost_is_authoritative"] is False
    assert admission["evaluator_authority_granted"] is False
    assert admission["proof_effect"] == "none"
    assert admission["allocation_binding_digest"] == canonical_digest(
        admission,
        digest_field="allocation_binding_digest",
    )


def test_openai_candidate_dry_run_never_becomes_execution_authority() -> None:
    admission = _build(execute_requested=False)
    assert admission["status"] == "dry_run_ready"
    assert admission["execute_requested"] is False
    assert admission["provider_mutations_performed"] == 0


@pytest.mark.parametrize(
    ("overrides", "blocker"),
    [
        ({"checkout_clean": False}, "openai_candidate_checkout_not_clean"),
        ({"maximum_execution_seconds": 601}, "openai_candidate_ttl_envelope_insufficient"),
        ({"runtime_watchdog_enforced": False}, "openai_candidate_runtime_watchdog_missing"),
        ({"teardown_enforced": False}, "openai_candidate_teardown_missing"),
        ({"license_attestation_digest": "missing"}, "openai_candidate_license_attestation_invalid"),
        (
            {"source_authority_blockers": ("gpu_canary_checkout_not_remote_main",)},
            "openai_candidate_checkout_not_remote_main",
        ),
    ],
)
def test_openai_candidate_admission_fails_closed(overrides: dict[str, Any], blocker: str) -> None:
    admission = _build(**overrides)
    assert admission["status"] == "blocked"
    assert blocker in admission["blockers"]
    assert admission["provider_mutations_performed"] == 0


def test_canonical_allocator_issues_bound_in_process_grant_only_for_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite, authorization = _inputs()
    monkeypatch.setattr(
        allocator, "_source_checkout_blockers", lambda *_args, **_kwargs: ([], COMMIT)
    )
    monkeypatch.setattr(allocator, "_current_checkout_source_state", lambda: (COMMIT, True))
    output = tmp_path / "openai_candidate_admission.json"

    admission, grant = allocator.admit_openai_api_candidate(
        suite=suite,
        execution_authorization=authorization,
        candidate_id="pigey-verify-recover",
        provider_id="pigey_external_candidate",
        runtime_configuration_digest=SHA_B,
        cost_authority_binding_digest=SHA_C,
        license_attestation_digest=SHA_A,
        expected_source_commit=COMMIT,
        maximum_execution_seconds=590,
        runtime_watchdog_enforced=True,
        teardown_enforced=True,
        admission_out=output,
        execute=True,
        admitted_at="2026-07-30T15:00:01Z",
    )

    assert grant is not None
    assert grant.allocation_binding_digest == admission["allocation_binding_digest"]
    require_paid_resource_admission_grant(
        grant,
        resource_class="openai_api_candidate",
        allocation_binding_digest=admission["allocation_binding_digest"],
    )
    assert json.loads(output.read_text(encoding="utf-8")) == admission

    dry_run, dry_grant = allocator.admit_openai_api_candidate(
        suite=suite,
        execution_authorization=authorization,
        candidate_id="pigey-verify-recover",
        provider_id="pigey_external_candidate",
        runtime_configuration_digest=SHA_B,
        cost_authority_binding_digest=SHA_C,
        license_attestation_digest=SHA_A,
        expected_source_commit=COMMIT,
        maximum_execution_seconds=590,
        runtime_watchdog_enforced=True,
        teardown_enforced=True,
        admission_out=tmp_path / "dry_run.json",
        execute=False,
        admitted_at="2026-07-30T15:00:01Z",
    )
    assert dry_run["status"] == "dry_run_ready"
    assert dry_grant is None


def test_pigey_allocator_wrapper_derives_runtime_envelope_and_injects_grant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite, authorization = _inputs()
    monkeypatch.setattr(
        allocator, "_source_checkout_blockers", lambda *_args, **_kwargs: ([], COMMIT)
    )
    monkeypatch.setattr(allocator, "_current_checkout_source_state", lambda: (COMMIT, True))

    class Runtime:
        candidate_id = "pigey-verify-recover"
        provider_id = "pigey_external_candidate"
        runtime_configuration_digest = SHA_B
        cost_authority_binding_digest = SHA_C
        license_attestation = {"license_attestation_digest": SHA_A}
        maximum_execution_seconds = 590
        runtime_watchdog_enforced = True
        teardown_enforced = True
        paid_resource_admission_grant = None

    runtime = Runtime()
    admission, grant = allocator.admit_pigey_candidate_runtime(
        suite=suite,
        execution_authorization=authorization,
        runtime=runtime,
        expected_source_commit=COMMIT,
        admission_out=tmp_path / "pigey_admission.json",
        execute=True,
        admitted_at="2026-07-30T15:00:01Z",
    )

    assert grant is not None
    assert runtime.paid_resource_admission_grant is grant
    assert admission["maximum_execution_seconds"] == 590
    assert admission["runtime_watchdog_enforced"] is True
    assert admission["teardown_enforced"] is True


def test_canonical_allocator_persists_refusal_before_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite, authorization = _inputs()
    monkeypatch.setattr(
        allocator,
        "_source_checkout_blockers",
        lambda *_args, **_kwargs: (["gpu_canary_checkout_not_clean"], COMMIT),
    )
    monkeypatch.setattr(allocator, "_current_checkout_source_state", lambda: (COMMIT, False))
    output = tmp_path / "blocked.json"

    with pytest.raises(PaidResourceAdmissionBlocked):
        allocator.admit_openai_api_candidate(
            suite=suite,
            execution_authorization=authorization,
            candidate_id="pigey-verify-recover",
            provider_id="pigey_external_candidate",
            runtime_configuration_digest=SHA_B,
            cost_authority_binding_digest=SHA_C,
            license_attestation_digest=SHA_A,
            expected_source_commit=COMMIT,
            maximum_execution_seconds=590,
            runtime_watchdog_enforced=True,
            teardown_enforced=True,
            admission_out=output,
            execute=True,
            admitted_at="2026-07-30T15:00:01Z",
        )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["status"] == "blocked"
    assert persisted["provider_mutations_performed"] == 0
    assert "openai_candidate_checkout_not_clean" in persisted["blockers"]
