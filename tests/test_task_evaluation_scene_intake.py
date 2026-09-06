"""ADP-009D: persistent owner authority does not reset on deploy or retry."""

import copy
from concurrent.futures import ThreadPoolExecutor

import pytest

from blueprint_pipeline.task_evaluation_scene_intake import (
    REQUEST_SCHEMA, SceneIntakeError, reserve_scene_attempt, stage_scene_intent, scene_intent_status,
    revoke_scene_intent,
)


def request():
    return {"schema_version": REQUEST_SCHEMA, "submission_id": "upload-1",
        "owner": {"user_id": "u1", "organization_id": "org1"},
        "source": {"kind": "mesh", "binding_id": "mesh-1", "content_digest": "sha256:" + "a" * 64},
        "task": {"task_id": "pick-book", "strategy": "pick_and_place", "subject": {"id": "book"},
                 "support": {"id": "table"}, "destination": {"id": "tray"}, "success": {"inside": True}},
        "execution": {"max_total_spend_usd": 4, "max_paid_attempts": 2, "max_retries": 0,
            "expires_at_epoch": 1000, "allowed_providers": ["vast"], "claim_scope": "development_only",
            "policy_candidates": [{"id": "pi05", "artifact_digest": "sha256:" + "b" * 64},
                                  {"id": "groot", "artifact_digest": "sha256:" + "c" * 64}]},
        "consent": {"accepted_by": "u1", "accepted_at_epoch": 99, "rights_reference": "rights-v1",
            "provider_terms_reference": "terms-v1", "private_processing_authorized": True,
            "provider_training_authorized": False, "task_confirmed": True, "spend_authorized": True}}


def stage(root, value=None):
    return stage_scene_intent(value=value or request(), queue_root=root,
        authenticated_client="webapp", trusted_clients={"webapp"}, now=100)


def attempt(root, intent, attempt_id="a1", commit="d", cost=2, now=101):
    return reserve_scene_attempt(queue_root=root, intent_id=intent["intent_id"], attempt_id=attempt_id,
        source_commit=commit * 40, runtime_digest="sha256:" + "e" * 64,
        input_digest="sha256:" + "f" * 64, provider="vast", maximum_spend_usd=cost, now=now)


def test_intake_is_idempotent_and_commit_independent(tmp_path):
    first = stage(tmp_path)
    assert stage(tmp_path) == first
    a = attempt(tmp_path, first)
    b = attempt(tmp_path, first, "a2", "c")
    assert a["intent_digest"] == b["intent_digest"] == first["intent_digest"]
    assert a["source_commit"] != b["source_commit"]
    assert stage(tmp_path) == first
    assert first["provider_mutation_performed_inside_http_request"] is False


def test_untrusted_database_writer_cannot_issue_intent(tmp_path):
    with pytest.raises(SceneIntakeError, match="issuer_not_authorized"):
        stage_scene_intent(value=request(), queue_root=tmp_path, authenticated_client="db-writer",
                           trusted_clients={"webapp"}, now=100)
    assert list(tmp_path.iterdir()) == []


def test_consent_actor_cannot_be_substituted(tmp_path):
    value = request()
    value["consent"]["accepted_by"] = "admin"
    with pytest.raises(SceneIntakeError, match="consent_actor"):
        stage(tmp_path, value)


def test_idempotency_key_cannot_replace_task(tmp_path):
    stage(tmp_path)
    value = request()
    value["task"]["subject"]["id"] = "another-book"
    with pytest.raises(SceneIntakeError, match="idempotency_conflict"):
        stage(tmp_path, value)


def test_attempt_identity_cannot_be_rebound_and_duplicate_does_not_debit(tmp_path):
    intent = stage(tmp_path)
    first = attempt(tmp_path, intent)
    assert attempt(tmp_path, intent) == first
    with pytest.raises(SceneIntakeError, match="attempt_immutable_conflict"):
        attempt(tmp_path, intent, commit="c")
    assert attempt(tmp_path, intent, "a2")["status"] == "reserved"
    with pytest.raises(SceneIntakeError, match="attempt_cap_exhausted"):
        attempt(tmp_path, intent, "a3")


def test_concurrent_reservations_cannot_overspend(tmp_path):
    intent = stage(tmp_path)
    def reserve(n):
        try:
            attempt(tmp_path, intent, "a" + str(n), cost=3)
            return True
        except SceneIntakeError as exc:
            assert "spend_cap_exhausted" in str(exc)
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(reserve, [1, 2])) == [False, True]


def test_exact_decimal_cap_and_javascript_numeric_identity(tmp_path):
    payload = request()
    payload["execution"].update(max_total_spend_usd=6.56, max_paid_attempts=3)
    intent = stage(tmp_path, payload)
    attempt(tmp_path, intent, "a1", cost=2.0)
    attempt(tmp_path, intent, "a2", cost=2.0)
    assert attempt(tmp_path, intent, "a3", cost=2.56)["status"] == "reserved"
    from blueprint_pipeline.task_evaluation_scene_intake import canonical_digest
    assert canonical_digest({"a": 20.0, "z": 1e-7}) == canonical_digest({"z": 0.0000001, "a": 20})


def test_expiry_and_revocation_survive_restarts(tmp_path):
    intent = stage(tmp_path)
    with pytest.raises(SceneIntakeError, match="authority_expired"):
        attempt(tmp_path, intent, now=1000)
    (tmp_path / intent["intent_id"] / "revoked.json").write_text("{}")
    with pytest.raises(SceneIntakeError, match="authority_revoked"):
        attempt(tmp_path, intent)


def test_status_is_read_only_and_does_not_expose_paths(tmp_path):
    intent = stage(tmp_path)
    attempt(tmp_path, intent)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = scene_intent_status(queue_root=tmp_path, intent_id=intent["intent_id"], now=102)
    assert result["status"] == "accepted" and len(result["attempts"]) == 1
    assert result["owner"] == request()["owner"]
    assert result["provider_mutation_performed_by_status_read"] is False
    assert str(tmp_path) not in str(result)
    after = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
    assert scene_intent_status(queue_root=tmp_path, intent_id=intent["intent_id"], now=1001)["status"] == "expired"


def test_owner_revocation_is_idempotent_and_stops_future_reservation(tmp_path):
    intent = stage(tmp_path)
    arguments = dict(queue_root=tmp_path, intent_id=intent["intent_id"], intent_digest=intent["intent_digest"],
                     owner=request()["owner"], now=102)
    with pytest.raises(SceneIntakeError, match="owner_or_intent_mismatch"):
        revoke_scene_intent(**{**arguments, "owner": {"user_id": "someone-else", "organization_id": "org1"}})
    receipt = revoke_scene_intent(**arguments)
    assert receipt["scope"] == "future_execution" and receipt["provider_mutation_performed"] is False
    assert revoke_scene_intent(**{**arguments, "now": 103}) == receipt
    assert scene_intent_status(queue_root=tmp_path, intent_id=intent["intent_id"], now=104)["status"] == "revoked"
    with pytest.raises(SceneIntakeError, match="authority_revoked"):
        attempt(tmp_path, intent)


@pytest.mark.parametrize("field,value", [("max_total_spend_usd", float("nan")),
    ("max_paid_attempts", True), ("allowed_providers", [{}]), ("policy_candidates", []),
    ("claim_scope", "qualified_evaluation")])
def test_invalid_or_expanded_authority_rejected(tmp_path, field, value):
    payload = copy.deepcopy(request())
    payload["execution"][field] = value
    with pytest.raises(SceneIntakeError):
        stage(tmp_path, payload)
