from __future__ import annotations

import copy
import hashlib
import json

import pytest

from blueprint_pipeline import operator_policy_canary_preprovider_closeout as closeout
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest


def inputs():
    registration = {
        "schema_version": "task_evaluation_operator_policy_canary_registration.v1",
        "run_id": "operator-canary-null", "capture_session_id": "capture-1", "intake_id": "intake-1",
        "team_namespace": "team-1", "firebase_tenant_id": None,
        "run_kind": "internal_policy_canary", "claim_ceiling": "diagnostic_policy_execution",
        "request_digest": "sha256:" + "a" * 64, "configuration_digest": "sha256:" + "b" * 64,
    }
    registration["registration_digest"] = cross_runtime_canonical_digest(registration, digest_field="registration_digest")
    receipt = {
        "schema_version": "operator_policy_no_allocation_closeout.v1",
        "run_id": registration["run_id"], "status": "blocked_without_provider_allocation",
        "provider_allocation_performed": False, "provider_instance_charge_usd": 0.0,
        "provider_instance_charge_not_applicable": True,
        "inner_adapter_sha256": "sha256:" + "c" * 64,
        "provider_create_attempted": False, "vast_side_effects_may_have_occurred": False,
        "vast_instance_ids": [], "all_staged_objects_absent": True,
        "continuing_spend_from_this_run": False, "watchdog_status": "cancelled_no_allocation",
        "provider_zero_verified": True, "legacy_no_allocation_predicate_passed": False,
        "legacy_predicate_gap": "Mutation counter absent; retained direct no-create fields.",
        "provider_attempt_classification": {
            "schema_version": "provider_attempt_classification.v1",
            "classification": "pre_execution_provider_null", "provider_bundle_started": False,
            "provider_entrypoint_started": False, "provider_output_returned": False,
            "scientific_attempt_consumed": False, "automatic_requeue_authorized": False,
            "automatic_requeue_executed": False, "maximum_automatic_requeues": 0,
            "pre_execution_requeue_eligible_in_principle": True,
            "authority_required_for_next_provider_mutation": True,
            "blockers": ["no_compatible_offer"],
        },
    }
    return registration, receipt


def sealed_bytes(receipt):
    value = copy.deepcopy(receipt)
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    return (json.dumps(value, indent=2) + "\n").encode()


def test_native_float_spelling_and_original_bytes_are_preserved():
    registration, receipt = inputs()
    raw = sealed_bytes(receipt)
    publication = closeout.build_operator_preprovider_publication(registration=registration, closeout_bytes=raw)
    source = publication["no_allocation_closeout"]
    assert source["raw_json"].encode() == raw
    assert '"provider_instance_charge_usd": 0.0' in source["raw_json"]
    assert source["size_bytes"] == len(raw)
    assert source["sha256"] == "sha256:" + hashlib.sha256(raw).hexdigest()
    assert publication["payload_digest"] == cross_runtime_canonical_digest(publication, digest_field="payload_digest")
    assert "activation_id" not in publication and "result_delivery" not in publication


@pytest.mark.parametrize("namespace", ["user:blueprint-production-runner", "team:example"])
def test_owner_metadata_namespace_matches_website_grammar(namespace):
    registration, receipt = inputs()
    registration["team_namespace"] = namespace
    registration["registration_digest"] = cross_runtime_canonical_digest(registration, digest_field="registration_digest")
    result = closeout.build_operator_preprovider_publication(registration=registration, closeout_bytes=sealed_bytes(receipt))
    assert result["team_namespace"] == namespace


@pytest.mark.parametrize("namespace", ["../other", "/absolute", "namespace with spaces", "user:bad/child"])
def test_metadata_namespace_refuses_path_and_whitespace(namespace):
    registration, receipt = inputs()
    registration["team_namespace"] = namespace
    registration["registration_digest"] = cross_runtime_canonical_digest(registration, digest_field="registration_digest")
    with pytest.raises(ValueError, match="metadata_identifier_invalid"):
        closeout.build_operator_preprovider_publication(registration=registration, closeout_bytes=sealed_bytes(receipt))


@pytest.mark.parametrize(("field", "value"), [
    ("run_id", "wrong-run"), ("provider_create_attempted", True),
    ("provider_allocation_performed", True), ("vast_side_effects_may_have_occurred", True),
    ("continuing_spend_from_this_run", True), ("all_staged_objects_absent", False),
    ("provider_zero_verified", False), ("watchdog_status", "armed"),
    ("vast_instance_ids", ["42"]), ("provider_instance_charge_usd", 0.01),
    ("provider_instance_charge_usd", False), ("provider_create_attempted", 0),
])
def test_resealed_allocated_or_uncertain_closeout_refuses(field, value):
    registration, receipt = inputs()
    receipt[field] = value
    with pytest.raises(ValueError, match="no_allocation_proof_invalid"):
        closeout.build_operator_preprovider_publication(registration=registration, closeout_bytes=sealed_bytes(receipt))


@pytest.mark.parametrize(("field", "value"), [
    ("scientific_attempt_consumed", True), ("provider_bundle_started", True),
    ("provider_entrypoint_started", True), ("provider_output_returned", True),
    ("automatic_requeue_executed", True), ("automatic_requeue_authorized", True),
    ("maximum_automatic_requeues", False), ("maximum_automatic_requeues", 1),
    ("authority_required_for_next_provider_mutation", False),
])
def test_attempt_evidence_refuses_execution_or_retry(field, value):
    registration, receipt = inputs()
    receipt["provider_attempt_classification"][field] = value
    with pytest.raises(ValueError, match="attempt_proof_invalid"):
        closeout.build_operator_preprovider_publication(registration=registration, closeout_bytes=sealed_bytes(receipt))


def test_original_native_digest_and_registration_digest_cannot_be_rewritten():
    registration, receipt = inputs()
    raw = sealed_bytes(receipt).replace(b'"provider_instance_charge_usd": 0.0', b'"provider_instance_charge_usd": 0')
    with pytest.raises(ValueError, match="native_receipt_digest_mismatch"):
        closeout.build_operator_preprovider_publication(registration=registration, closeout_bytes=raw)
    registration["run_id"] = "changed-run"
    with pytest.raises(ValueError, match="registration_invalid"):
        closeout.build_operator_preprovider_publication(registration=registration, closeout_bytes=sealed_bytes(receipt))


@pytest.mark.parametrize("changed_field", [None, "run_id", "capture_session_id", "intake_id",
    "operator_registration_digest", "request_digest", "configuration_digest", "payload_digest",
    "provider_allocation_performed", "notification_delivery"])
def test_signed_send_requires_exact_ack_and_retains_disabled_email(monkeypatch, changed_field):
    registration, receipt = inputs()
    captured = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self, _limit):
            value = json.loads(captured[0].data)
            reply = {key: value[key] for key in ("run_id", "capture_session_id", "intake_id",
                "operator_registration_digest", "request_digest", "configuration_digest", "payload_digest")}
            reply.update(schema_version="capture_task_evaluation_operator_policy_canary_blocked_receipt.v1",
                         status="blocked", provider_allocation_performed=False,
                         notification_delivery={"status": "failed", "failure_reason": "email_disabled",
                            "terminal_state": "blocked", "run_result_digest": value["payload_digest"]})
            if changed_field:
                reply[changed_field] = "wrong-binding"
            return json.dumps(reply).encode()

    def send(request, **_):
        captured.append(request)
        return Response()

    monkeypatch.setattr(closeout.urllib_request, "urlopen", send)
    result = closeout.sync_operator_preprovider_closeout(
        registration=registration, closeout_bytes=sealed_bytes(receipt), token="test-secret",
        endpoint_url="https://webapp.example/api/internal/pipeline/capture-task-evaluation-runs",
    )
    assert len(captured) == 1
    assert any("signature" in key.lower() for key in captured[0].headers)
    assert result["status"] == ("failed" if changed_field else "succeeded")
    if changed_field is None:
        assert result["notification_delivery"]["failure_reason"] == "email_disabled"
