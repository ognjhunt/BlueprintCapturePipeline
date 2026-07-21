from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.evaluator_evidence_profiles import (
    COMMON_DIGEST_FIELDS,
    canonical_evaluator_backend_manifest_sha256,
)
from blueprint_pipeline.evaluator_runtime_evidence import (
    EVALUATOR_RUNTIME_NORMALIZATION_REQUEST_SCHEMA_VERSION,
    EVALUATOR_RUNTIME_RECEIPT_SCHEMA_VERSION,
    build_wam_runtime_receipt,
    canonical_json_sha256,
    main,
    normalize_evaluator_runtime_evidence,
    validate_evaluator_runtime_receipt,
)


def _digest(index: int) -> str:
    return f"sha256:{index:064x}"


def _backend(*, family: str = "cosmos3", version: str = "nano-v1") -> dict[str, object]:
    return {
        "schema_version": "evaluator_backend_manifest.v1",
        "backend_id": f"{family}-runtime",
        "backend_kind": "world_model",
        "model_family": family,
        "model_version": version,
        "adapter_version": "1.0.0",
        "execution_interface": "provider_worker",
        "model_artifact_sha256": _digest(1),
        "adapter_code_sha256": _digest(2),
        "runtime_manifest_sha256": _digest(3),
        "license_manifest_sha256": _digest(4),
        "backend_is_compute_provider": False,
    }


def _runtime_output(schema_version: str) -> dict[str, object]:
    oscar = schema_version == "oscar_wam_command_adapter.v1"
    subprocess_field = "oscar_subprocess" if oscar else "cosmos3_subprocess"
    adapter_id = (
        "blueprint_oscar_wam_command_adapter"
        if oscar
        else "blueprint_cosmos3_nano_wam_command_adapter"
    )
    return {
        "schema_version": schema_version,
        "status": "completed",
        "adapter_id": adapter_id,
        "blockers": [],
        "learned_wam_model_ran": True,
        "fresh_model_command_executed_this_invocation": True,
        "fresh_model_run_claimed": True,
        "fresh_model_run_steps": 1,
        "configured_inference_steps_per_model_run": 35,
        subprocess_field: {"status": "completed", "returncode": 0},
        "truth_boundary": {"generated_video_is_model_output": True},
        "model_provenance": {"model": "cosmos3/nano/action-cond"},
        "official_oscar_release": {"model_name": "OSCAR-2B"},
        "rollouts": [
            {
                "rollout_id": "rollout-1",
                "generated_video_sha256": _digest(10),
                "generated_video_review_validation": {"status": "completed"},
            }
        ],
    }


def _wam_request(schema_version: str = "cosmos3_wam_command_adapter.v1") -> dict[str, object]:
    runtime_output = _runtime_output(schema_version)
    runtime_output_sha256 = canonical_json_sha256(runtime_output)
    backend = _backend(
        family="oscar" if schema_version.startswith("oscar") else "cosmos3",
        version=("OSCAR-2B" if schema_version.startswith("oscar") else "cosmos3/nano/action-cond"),
    )
    runtime_identity = {
        "runtime_id": "runtime-1",
        "provider_id": "runpod",
        "runtime_adapter_version": "1.0.0",
    }
    provider_execution = {
        "schema_version": "evaluator_provider_execution.v1",
        "status": "succeeded",
        "execution_id": "provider-execution-1",
        "runtime_id": runtime_identity["runtime_id"],
        "provider_id": runtime_identity["provider_id"],
        "runtime_output_sha256": runtime_output_sha256,
        "model_artifact_sha256": backend["model_artifact_sha256"],
        "adapter_code_sha256": _digest(2),
        "runtime_manifest_sha256": _digest(3),
        "provider_is_evaluator_identity": False,
    }
    return {
        "runtime_output": runtime_output,
        "runtime_output_sha256": runtime_output_sha256,
        "evaluator_backend": backend,
        "runtime_identity": runtime_identity,
        "adapter_code_sha256": _digest(2),
        "runtime_manifest_sha256": _digest(3),
        "license_manifest_sha256": _digest(4),
        "provider_execution": provider_execution,
        "provider_execution_sha256": canonical_json_sha256(provider_execution),
        "infrastructure_status": "succeeded",
        "fixture_or_proxy_model_output_used": False,
        "fallback_model_output_used": False,
        "stale_model_output_used": False,
    }


def _row(
    backend: dict[str, object],
    *,
    provider_execution_sha256: str = _digest(5),
    evaluator_runtime_output_sha256: str = _digest(20),
) -> dict[str, object]:
    row: dict[str, object] = {
        field: _digest(index + 100) for index, field in enumerate(COMMON_DIGEST_FIELDS)
    }
    row.update(
        {
            "evaluator_profile_id": "generic_evaluator_bounded_v1",
            "evaluator_backend": backend,
            "evaluator_backend_manifest_sha256": canonical_evaluator_backend_manifest_sha256(
                backend
            ),
            "evaluator_checkpoint_sha256": backend["model_artifact_sha256"],
            "evaluator_runtime_output_sha256": evaluator_runtime_output_sha256,
            "model_output_sha256": _digest(10),
            "provider_execution_sha256": provider_execution_sha256,
            "fresh_evaluator_model_execution_proven": True,
            "fresh_evaluator_model_run_steps": 1,
            "action_control_suite_status": "passed",
            "authoritative_manifest_status": "completed",
            "infrastructure_status": "succeeded",
            "evaluator_identity_is_compute_provider": False,
            "evaluator_outcome_status": "valid",
            "criterion_result_status": "valid",
            "generic_evaluator_contract_status": "validated",
        }
    )
    return row


@pytest.mark.parametrize(
    "schema_version",
    ["cosmos3_wam_command_adapter.v1", "oscar_wam_command_adapter.v1"],
)
def test_actual_wam_outputs_build_provider_neutral_runtime_receipts(
    schema_version: str,
) -> None:
    receipt = build_wam_runtime_receipt(_wam_request(schema_version))

    assert receipt["status"] == "validated"
    assert receipt["runtime_status"] == "completed"
    assert receipt["fresh_model_run_steps"] == 1
    assert receipt["configured_inference_steps_per_model_run"] == 35
    assert receipt["model_outputs"] == [
        {
            "output_id": "rollout-1",
            "model_output_sha256": _digest(10),
            "model_output_status": "completed",
        }
    ]
    assert receipt["provider_id"] == "runpod"
    assert receipt["backend_is_compute_provider"] is False


def test_runtime_receipt_normalizes_to_generic_evaluator_row() -> None:
    receipt = build_wam_runtime_receipt(_wam_request())
    backend = _backend(version="cosmos3/nano/action-cond")
    request = {
        "schema_version": EVALUATOR_RUNTIME_NORMALIZATION_REQUEST_SCHEMA_VERSION,
        "runtime_receipt": receipt,
        "runtime_receipt_sha256": canonical_json_sha256(receipt),
        "model_output_id": "rollout-1",
        "evaluator_row": _row(
            backend,
            provider_execution_sha256=receipt["provider_execution_sha256"],
            evaluator_runtime_output_sha256=receipt["runtime_output_sha256"],
        ),
    }

    result = normalize_evaluator_runtime_evidence(request)

    assert result["status"] == "normalized"
    assert result["decision_grade_row_admitted"] is True
    assert result["evaluator_model_family"] == "cosmos3"
    assert result["provider_id"] == "runpod"
    assert (
        result["evaluator_row"]["policy_runtime_output_sha256"] != receipt["runtime_output_sha256"]
    )
    assert result["claim_boundary"]["provider_identity_is_separate_from_evaluator_identity"]


def test_future_world_model_uses_standard_receipt_without_new_profile_logic() -> None:
    backend = _backend(family="future_world_model", version="v9")
    provider_execution = {
        "schema_version": "evaluator_provider_execution.v1",
        "status": "succeeded",
        "execution_id": "future-provider-execution-1",
        "runtime_id": "future-runtime-1",
        "provider_id": "vast",
        "runtime_output_sha256": _digest(20),
        "model_artifact_sha256": backend["model_artifact_sha256"],
        "adapter_code_sha256": backend["adapter_code_sha256"],
        "runtime_manifest_sha256": backend["runtime_manifest_sha256"],
        "provider_is_evaluator_identity": False,
    }
    receipt = {
        "schema_version": EVALUATOR_RUNTIME_RECEIPT_SCHEMA_VERSION,
        "runtime_id": "future-runtime-1",
        "runtime_adapter_id": "customer_future_adapter",
        "runtime_adapter_version": "9.0.0",
        "backend_id": backend["backend_id"],
        "model_family": backend["model_family"],
        "model_version": backend["model_version"],
        "runtime_output_sha256": _digest(20),
        "model_artifact_sha256": backend["model_artifact_sha256"],
        "adapter_code_sha256": backend["adapter_code_sha256"],
        "runtime_manifest_sha256": backend["runtime_manifest_sha256"],
        "license_manifest_sha256": backend["license_manifest_sha256"],
        "provider_id": "vast",
        "provider_execution": provider_execution,
        "provider_execution_sha256": canonical_json_sha256(provider_execution),
        "runtime_status": "completed",
        "infrastructure_status": "succeeded",
        "fresh_model_execution_proven": True,
        "fresh_model_run_steps": 1,
        "backend_is_compute_provider": False,
        "model_outputs": [
            {
                "output_id": "future-output-1",
                "model_output_sha256": _digest(10),
                "model_output_status": "completed",
            }
        ],
        "fixture_or_proxy_model_output_used": False,
        "fallback_model_output_used": False,
        "stale_model_output_used": False,
    }
    row = _row(
        backend,
        provider_execution_sha256=receipt["provider_execution_sha256"],
        evaluator_runtime_output_sha256=receipt["runtime_output_sha256"],
    )
    row["fresh_evaluator_model_run_steps"] = 1
    request = {
        "schema_version": EVALUATOR_RUNTIME_NORMALIZATION_REQUEST_SCHEMA_VERSION,
        "runtime_receipt": receipt,
        "runtime_receipt_sha256": canonical_json_sha256(receipt),
        "model_output_id": "future-output-1",
        "evaluator_row": row,
    }

    result = normalize_evaluator_runtime_evidence(request)

    assert validate_evaluator_runtime_receipt(receipt)["status"] == "validated"
    assert result["status"] == "normalized"
    assert result["evaluator_model_family"] == "future_world_model"
    assert result["provider_id"] == "vast"


def test_completed_runtime_cannot_override_blocked_authoritative_manifest() -> None:
    receipt = build_wam_runtime_receipt(_wam_request())
    row = _row(
        _backend(version="cosmos3/nano/action-cond"),
        provider_execution_sha256=receipt["provider_execution_sha256"],
        evaluator_runtime_output_sha256=receipt["runtime_output_sha256"],
    )
    row["authoritative_manifest_status"] = "blocked"
    request = {
        "schema_version": EVALUATOR_RUNTIME_NORMALIZATION_REQUEST_SCHEMA_VERSION,
        "runtime_receipt": receipt,
        "runtime_receipt_sha256": canonical_json_sha256(receipt),
        "model_output_id": "rollout-1",
        "evaluator_row": row,
    }

    result = normalize_evaluator_runtime_evidence(request)

    assert result["status"] == "blocked"
    assert result["evaluator_row"] is None
    assert "evaluator_evidence:authoritative_manifest_not_completed" in result["blockers"]


@pytest.mark.parametrize(
    ("mutation", "expected_blocker"),
    [
        (
            lambda request: request["runtime_output"].update({"fresh_model_run_steps": 0}),
            "wam_runtime_fresh_model_run_steps_missing_or_invalid",
        ),
        (
            lambda request: request["runtime_output"].update({"fresh_model_run_steps": 35}),
            "wam_runtime_fresh_model_run_steps_do_not_match_outputs",
        ),
        (
            lambda request: request.update({"fixture_or_proxy_model_output_used": True}),
            "runtime_receipt_forbidden_or_unproven:fixture_or_proxy_model_output_used",
        ),
        (
            lambda request: request["runtime_output"]["rollouts"][0].update(
                {"generated_video_sha256": "not-a-digest"}
            ),
            "wam_runtime_model_output_digest_missing_or_invalid:0",
        ),
        (
            lambda request: request["evaluator_backend"].update(
                {"model_family": "unrelated_model"}
            ),
            "wam_runtime_backend_model_family_mismatch",
        ),
        (
            lambda request: request["evaluator_backend"].update(
                {"model_version": "stale-model-version"}
            ),
            "wam_runtime_backend_model_version_mismatch",
        ),
        (
            lambda request: request["evaluator_backend"].update(
                {"backend_is_compute_provider": True}
            ),
            "wam_runtime_backend_must_not_be_compute_provider",
        ),
        (
            lambda request: request["provider_execution"].update({"status": "failed"}),
            "wam_provider_execution_not_succeeded",
        ),
        (
            lambda request: request.update({"infrastructure_status": "failed"}),
            "wam_runtime_infrastructure_not_succeeded",
        ),
        (
            lambda request: request["provider_execution"].update(
                {"provider_is_evaluator_identity": True}
            ),
            "wam_provider_execution_must_not_be_evaluator_identity",
        ),
    ],
)
def test_wam_runtime_receipt_fails_closed_on_non_decision_grade_execution(
    mutation,
    expected_blocker: str,
) -> None:
    request = _wam_request()
    mutation(request)
    request["runtime_output_sha256"] = canonical_json_sha256(request["runtime_output"])
    request["provider_execution"]["runtime_output_sha256"] = request["runtime_output_sha256"]
    request["provider_execution"]["model_artifact_sha256"] = request["evaluator_backend"][
        "model_artifact_sha256"
    ]
    request["provider_execution_sha256"] = canonical_json_sha256(request["provider_execution"])

    receipt = build_wam_runtime_receipt(request)

    assert receipt["status"] == "blocked"
    assert expected_blocker in receipt["blockers"]


def test_runtime_normalization_rejects_stale_receipt_or_row_binding() -> None:
    receipt = build_wam_runtime_receipt(_wam_request())
    row = _row(
        _backend(version="cosmos3/nano/action-cond"),
        provider_execution_sha256=receipt["provider_execution_sha256"],
        evaluator_runtime_output_sha256=receipt["runtime_output_sha256"],
    )
    row["model_output_sha256"] = _digest(999)
    row["evaluator_runtime_output_sha256"] = _digest(997)
    request = {
        "schema_version": EVALUATOR_RUNTIME_NORMALIZATION_REQUEST_SCHEMA_VERSION,
        "runtime_receipt": receipt,
        "runtime_receipt_sha256": _digest(998),
        "model_output_id": "rollout-1",
        "evaluator_row": row,
    }

    result = normalize_evaluator_runtime_evidence(copy.deepcopy(request))

    assert result["status"] == "blocked"
    assert "runtime_receipt_digest_mismatch" in result["blockers"]
    assert "runtime_evaluator_row_digest_binding_mismatch:model_output_sha256" in result["blockers"]
    assert (
        "runtime_evaluator_row_digest_binding_mismatch:evaluator_runtime_output_sha256"
        in result["blockers"]
    )


def test_runtime_normalization_rejects_evaluator_digest_as_policy_runtime() -> None:
    receipt = build_wam_runtime_receipt(_wam_request())
    row = _row(
        _backend(version="cosmos3/nano/action-cond"),
        provider_execution_sha256=receipt["provider_execution_sha256"],
        evaluator_runtime_output_sha256=receipt["runtime_output_sha256"],
    )
    row["policy_runtime_output_sha256"] = receipt["runtime_output_sha256"]

    result = normalize_evaluator_runtime_evidence(
        {
            "schema_version": EVALUATOR_RUNTIME_NORMALIZATION_REQUEST_SCHEMA_VERSION,
            "runtime_receipt": receipt,
            "runtime_receipt_sha256": canonical_json_sha256(receipt),
            "model_output_id": "rollout-1",
            "evaluator_row": row,
        }
    )

    assert result["status"] == "blocked"
    assert result["evaluator_row"] is None
    assert "runtime_policy_and_evaluator_output_digests_must_be_distinct" in result["blockers"]


def test_runtime_normalization_rejects_receipt_that_declares_blockers() -> None:
    receipt = build_wam_runtime_receipt(_wam_request())
    receipt["status"] = "blocked"
    receipt["blockers"] = ["late_provider_audit_failed"]
    row = _row(
        _backend(version="cosmos3/nano/action-cond"),
        provider_execution_sha256=receipt["provider_execution_sha256"],
        evaluator_runtime_output_sha256=receipt["runtime_output_sha256"],
    )
    result = normalize_evaluator_runtime_evidence(
        {
            "schema_version": EVALUATOR_RUNTIME_NORMALIZATION_REQUEST_SCHEMA_VERSION,
            "runtime_receipt": receipt,
            "runtime_receipt_sha256": canonical_json_sha256(receipt),
            "model_output_id": "rollout-1",
            "evaluator_row": row,
        }
    )

    assert result["status"] == "blocked"
    assert "runtime_receipt:runtime_receipt_declared_status_not_validated" in result["blockers"]
    assert "runtime_receipt:runtime_receipt_declares_blockers" in result["blockers"]


def test_runtime_evidence_does_not_retain_unexpected_provider_secrets() -> None:
    request = _wam_request()
    request["provider_execution"]["api_token"] = "must-not-be-retained"
    request["provider_execution_sha256"] = canonical_json_sha256(request["provider_execution"])

    receipt = build_wam_runtime_receipt(request)

    assert receipt["status"] == "blocked"
    assert "wam_provider_execution_fields_invalid" in receipt["blockers"]
    assert "api_token" not in receipt["provider_execution"]
    assert "must-not-be-retained" not in json.dumps(receipt)


def test_runtime_normalization_rejects_and_omits_sensitive_evaluator_row() -> None:
    receipt = build_wam_runtime_receipt(_wam_request())
    row = _row(
        _backend(version="cosmos3/nano/action-cond"),
        provider_execution_sha256=receipt["provider_execution_sha256"],
        evaluator_runtime_output_sha256=receipt["runtime_output_sha256"],
    )
    row["api_token"] = "must-not-be-retained"
    result = normalize_evaluator_runtime_evidence(
        {
            "schema_version": EVALUATOR_RUNTIME_NORMALIZATION_REQUEST_SCHEMA_VERSION,
            "runtime_receipt": receipt,
            "runtime_receipt_sha256": canonical_json_sha256(receipt),
            "model_output_id": "rollout-1",
            "evaluator_row": row,
        }
    )

    assert result["status"] == "blocked"
    assert "runtime_evaluator_row_contains_sensitive_fields" in result["blockers"]
    assert result["evaluator_row"] is None
    assert "must-not-be-retained" not in json.dumps(result)


def test_runtime_evidence_cli_builds_receipt_and_fails_closed(
    tmp_path: Path,
) -> None:
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "receipt.json"
    request_path.write_text(json.dumps(_wam_request()), encoding="utf-8")

    assert (
        main(
            [
                "build-wam-receipt",
                "--request",
                str(request_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "validated"

    blocked_request = _wam_request()
    blocked_request["fallback_model_output_used"] = True
    request_path.write_text(json.dumps(blocked_request), encoding="utf-8")
    assert (
        main(
            [
                "build-wam-receipt",
                "--request",
                str(request_path),
                "--output",
                str(output_path),
            ]
        )
        == 2
    )
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "blocked"
