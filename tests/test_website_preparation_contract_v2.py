"""Synthetic offline contract vectors; no current grant or production evidence."""

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline import website_task_context as context
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

VECTOR = json.loads(
    (Path(__file__).parent / "fixtures/website-preparation-contract-v2.json").read_text()
)
NOW = VECTOR["sponsorship_response"]["consent"]["accepted_at_epoch"]


@pytest.fixture(autouse=True)
def synthetic_clock(monkeypatch):
    from blueprint_pipeline import website_preparation_authority

    monkeypatch.setattr(website_preparation_authority, "current_time", lambda: NOW)


def validate(value, **kwargs):
    return context.validate_website_task_context(
        value,
        request_id=value["request_id"],
        scene_id=value["scene_id"],
        capture_id=value["capture_id"],
        **kwargs,
    )


@pytest.mark.parametrize(
    "name",
    [
        "baseline_existing_confirmed_context",
        "baseline_unconfirmed_context",
        "task_context_response_preparation",
    ],
)
def test_frozen_digest_vector_unchanged(name):
    value = VECTOR[name]
    assert canonical_digest(value, digest_field="context_digest") == value["context_digest"]


def test_default_confirmation_stays_strict_even_when_context_declares_preparation():
    assert validate(VECTOR["baseline_existing_confirmed_context"])["confirmed"] is True
    for name in ["baseline_unconfirmed_context", "task_context_response_preparation"]:
        with pytest.raises(ValueError, match="not_confirmed"):
            validate(VECTOR[name])


def test_explicit_preparation_read_preserves_false_null_and_absent_targets():
    value = validate(VECTOR["task_context_response_preparation"], purpose="scene_preparation")
    assert value == VECTOR["task_context_response_preparation"]
    assert value["confirmed"] is False and value["confirmed_at"] is None
    assert "operator_task_details" not in value and "success_criteria" not in value


def test_explicit_current_read_sends_purpose_and_checks_full_projection(monkeypatch):
    calls = []

    def read(**kwargs):
        calls.append(kwargs)
        return copy.deepcopy(VECTOR["task_context_response_preparation"])

    monkeypatch.setattr(context, "website_webapp_request", read)
    value = VECTOR["task_context_response_preparation"]
    assert (
        context.load_current_website_task_context(
            request_id=value["request_id"],
            scene_id=value["scene_id"],
            capture_id=value["capture_id"],
            purpose="scene_preparation",
        )
        == value
    )
    assert calls[0]["payload"] == VECTOR["task_context_request_preparation"]


def test_sponsorship_request_binds_explicit_purpose_full_context_digest(monkeypatch):
    calls = []

    def read(**kwargs):
        calls.append(kwargs)
        return copy.deepcopy(VECTOR["sponsorship_response"])

    monkeypatch.setattr(context, "website_webapp_request", read)
    assert (
        context.load_website_scene_sponsorship(
            task_context=VECTOR["task_context_response_preparation"], now=NOW
        )
        == VECTOR["sponsorship_response"]
    )
    assert calls[0]["payload"] == VECTOR["sponsorship_request"]


def pending_geometry(tmp_path):
    """Separate existing measured-in-fixture geometry; frozen vector untouched."""
    from tests.test_website_task_preparation import _arguments

    args = _arguments(tmp_path)
    args["base_scene"]["splat_binding_id"] = (
        "website-splat-" + args["base_scene"]["splat_digest"][7:39]
    )
    task = args["task_context"]
    task.update(
        purpose="scene_preparation",
        confirmed=False,
        confirmed_at=None,
        capture_rights=copy.deepcopy(VECTOR["task_context_response_preparation"]["capture_rights"]),
        unresolved=["required_final_state", "cycle_time"],
    )
    task["context_digest"] = canonical_digest(task, digest_field="context_digest")
    grant = copy.deepcopy(VECTOR["sponsorship_response"])
    for k in ["request_id", "scene_id", "capture_id"]:
        grant[k] = task[k]
    grant["task_context_digest"] = task["context_digest"]
    grant["consent"]["accepted_at_epoch"] = args["now"] - 100
    grant["expires_at_epoch"] = args["now"] + 3600
    for k in ["request_id", "capture_id"]:
        grant["assessment_preparation_proposal"][k] = task[k]
    grant["authority_digest"] = canonical_digest(grant, digest_field="authority_digest")
    args["spend"] = grant
    return args


def test_pending_preparation_compiles_and_construction_preserves_unknowns(tmp_path):
    from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
    from blueprint_pipeline.website_native_background import construction_rights_admission

    args = pending_geometry(tmp_path)
    prepared = compile_website_scene_preparation(**args)
    assert prepared["status"] == "intake_ready", prepared["blockers"]
    assert prepared["intake_request"]["consent"]["task_confirmed"] is False
    assert prepared["owner_success_criteria"]["status"] == "not_supplied"
    assert prepared["website_preparation_authority"] == args["spend"]
    rights = construction_rights_admission(
        preparation=prepared, task_context=args["task_context"], now=args["now"]
    )
    assert rights["consent"]["task_confirmed"] is False
    assert rights["provider_disclosure"]["raw_capture_video_allowed"] is False
    assert rights["physical_measurement_proven"] is False


def test_false_preparation_consent_is_structural_only_no_issuance_without_source(tmp_path):
    from blueprint_pipeline import task_evaluation_scene_intake as intake

    value = copy.deepcopy(VECTOR["prepared_scene_body"]["request"])
    # The frozen subject/support are shape examples; valid geometry is separate above.
    assert intake.validate_request(value, now=NOW) == value
    with pytest.raises(ValueError, match="scene_intake_preparation_authority_invalid"):
        intake.stage_scene_intent(
            value=value,
            queue_root=tmp_path / "intents",
            authenticated_client="blueprint-webapp",
            trusted_clients={"blueprint-webapp"},
            now=NOW,
        )
    assert not (tmp_path / "intents").exists()


@pytest.mark.parametrize(
    "change",
    [
        "stale_context",
        "proposal_capture",
        "proposal_scope",
        "physical",
        "withdrawn",
        "wrong_owner",
        "expired",
        "consent_coerced",
        "consent_zero",
        "missing_purpose",
    ],
)
def test_pending_grant_refuses_uncurrent_or_false_authority(change):
    from blueprint_pipeline.website_preparation_authority import validate_preparation_authority

    task = copy.deepcopy(VECTOR["task_context_response_preparation"])
    grant = copy.deepcopy(VECTOR["sponsorship_response"])
    if change == "stale_context":
        grant["task_context_digest"] = "sha256:" + "0" * 64
    elif change == "proposal_capture":
        grant["assessment_preparation_proposal"]["capture_id"] = "walkthrough-other"
    elif change == "proposal_scope":
        grant["assessment_preparation_proposal"]["scope"] = "evaluation"
    elif change == "physical":
        grant["assessment_preparation_proposal"]["physical_trial_authorized"] = True
    elif change == "wrong_owner":
        grant["owner"]["user_id"] = "different-owner"
    elif change == "expired":
        grant["expires_at_epoch"] = NOW
    elif change == "consent_coerced":
        grant["consent"]["task_confirmed"] = True
    elif change == "consent_zero":
        grant["consent"]["task_confirmed"] = 0
    elif change == "missing_purpose":
        grant.pop("purpose")
    else:
        task["capture_rights"]["consent_revoked"] = True
        task["context_digest"] = canonical_digest(task, digest_field="context_digest")
        grant["task_context_digest"] = task["context_digest"]
    grant["authority_digest"] = canonical_digest(grant, digest_field="authority_digest")
    with pytest.raises(ValueError):
        validate_preparation_authority(task_context=task, authority=grant, now=NOW)


def test_child_does_not_inherit_original_preparation_proposal():
    from blueprint_pipeline.website_preparation_authority import validate_preparation_authority

    task = VECTOR["continuation_mechanical_vector"]["child_full_context"]
    grant = copy.deepcopy(VECTOR["sponsorship_response"])
    grant.update(capture_id=task["capture_id"], task_context_digest=task["context_digest"])
    grant["authority_digest"] = canonical_digest(grant, digest_field="authority_digest")
    with pytest.raises(ValueError):
        validate_preparation_authority(task_context=task, authority=grant, now=NOW)


@pytest.mark.parametrize(
    "change", ["evaluation", "policies", "robot", "evaluation_source", "zero", "null"]
)
def test_structural_pending_consent_never_crosses_evaluation_scope(change):
    from blueprint_pipeline import task_evaluation_scene_intake as intake

    value = copy.deepcopy(VECTOR["prepared_scene_body"]["request"])
    if change == "evaluation":
        value["execution"].pop("purpose")
    elif change == "policies":
        value["execution"]["policy_candidates"] = [
            {"id": "robot", "artifact_digest": "sha256:" + "a" * 64}
        ]
    elif change == "robot":
        value["task"]["robot_binding_id"] = "robot-binding"
    elif change == "evaluation_source":
        value["task"]["evaluation_source"] = {}
    else:
        value["consent"]["task_confirmed"] = 0 if change == "zero" else None
    with pytest.raises(ValueError):
        intake.validate_request(value, now=NOW)


def retained_source(tmp_path):
    """Server-owned local reference rehearsal, never production provenance."""
    from blueprint_pipeline import website_preparation_authority as authority
    from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
    import hashlib

    context_value = copy.deepcopy(VECTOR["task_context_response_preparation"])
    grant = copy.deepcopy(VECTOR["sponsorship_response"])
    request = copy.deepcopy(VECTOR["prepared_scene_body"]["request"])
    prepared = {
        "schema_version": "SYNTHETIC.retained_preparation",
        "intake_request": request,
        "website_preparation_authority": grant,
        "binding": {"task_context_digest": context_value["context_digest"]},
    }
    prepared["digest"] = canonical_digest(prepared, digest_field="digest")
    root = tmp_path / "website-source-bindings"
    root.mkdir()

    def save(name, value):
        path = root / name
        raw = json.dumps(value, sort_keys=True).encode()
        path.write_bytes(raw)
        return {
            "path": str(path),
            "size_bytes": len(raw),
            "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        }

    registration = {
        "schema_version": "website_scene_source_registration.v1",
        "request_digest": cross_runtime_canonical_digest(request),
        "execution_authority_granted": False,
        "references": {
            "preparation": save("preparation.json", prepared),
            "task_context": save("context.json", context_value),
            "runtime_inputs": save("runtime.json", {"synthetic": True, "production_proof": False}),
        },
    }
    registration["registration_digest"] = canonical_digest(
        registration, digest_field="registration_digest"
    )
    save(cross_runtime_canonical_digest(request)[7:] + ".json", registration)
    return authority, context_value, grant, request


def test_authenticated_intake_and_paid_reservation_reopen_without_grant_creation(
    tmp_path, monkeypatch
):
    from blueprint_pipeline import task_evaluation_scene_intake as intake

    _, task, grant, value = retained_source(tmp_path)
    calls = []

    def signed_read(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return copy.deepcopy(task if kwargs["operation"] == "task-context" else grant)

    monkeypatch.setattr(context, "website_webapp_request", signed_read)
    result = intake.stage_scene_intent(
        value=value,
        queue_root=tmp_path / "intents",
        authenticated_client="blueprint-webapp",
        trusted_clients={"blueprint-webapp"},
        now=NOW,
    )
    assert result["status"] == "accepted"
    assert calls[1]["payload"] == {**VECTOR["sponsorship_request"], "create": False}
    reserved = intake.reserve_scene_attempt(
        queue_root=tmp_path / "intents",
        intent_id=result["intent_id"],
        attempt_id="scene-configuration-synthetic",
        source_commit="a" * 40,
        runtime_digest="sha256:" + "b" * 64,
        input_digest="sha256:" + "c" * 64,
        provider="vast",
        maximum_spend_usd=1,
        now=NOW,
    )
    assert reserved["status"] == "reserved" and len(calls) == 4
    assert reserved["maximum_spend_usd"] == grant["max_total_spend_usd"]
    assert all(
        c["payload"].get("create") is False for c in calls if c["operation"] == "scene-sponsorship"
    )
    assert grant == VECTOR["sponsorship_response"]


@pytest.mark.parametrize(
    "change", ["current_context", "current_proposal", "changed_reference", "wrong_request_owner"]
)
def test_retained_preparation_refuses_changed_current_or_stored_source(
    tmp_path, monkeypatch, change
):
    from blueprint_pipeline import task_evaluation_scene_intake as intake

    _, task, grant, value = retained_source(tmp_path)
    if change == "current_context":
        task["unresolved"].append("new-question")
    elif change == "current_proposal":
        grant["assessment_preparation_proposal"] = copy.deepcopy(
            VECTOR["changed_assessment_preparation_proposal"]
        )
        grant["authority_digest"] = canonical_digest(grant, digest_field="authority_digest")
    elif change == "changed_reference":
        (tmp_path / "website-source-bindings/context.json").write_text("{}")
    else:
        value["owner"]["user_id"] = "wrong-owner"
    monkeypatch.setattr(
        context,
        "website_webapp_request",
        lambda **kwargs: copy.deepcopy(task if kwargs["operation"] == "task-context" else grant),
    )
    with pytest.raises(ValueError):
        intake.stage_scene_intent(
            value=value,
            queue_root=tmp_path / "intents",
            authenticated_client="blueprint-webapp",
            trusted_clients={"blueprint-webapp"},
            now=NOW,
        )
    assert not (tmp_path / "intents").exists()


def test_authority_expiring_during_signed_reads_refuses_before_intake_mutation(
    tmp_path, monkeypatch
):
    from blueprint_pipeline import task_evaluation_scene_intake as intake

    helper, task, grant, value = retained_source(tmp_path)
    clock = [NOW]
    monkeypatch.setattr(helper, "current_time", lambda: clock[0])

    def signed_read(**kwargs):
        if kwargs["operation"] == "scene-sponsorship":
            clock[0] = grant["expires_at_epoch"] + 1
        return copy.deepcopy(task if kwargs["operation"] == "task-context" else grant)

    monkeypatch.setattr(context, "website_webapp_request", signed_read)
    with pytest.raises(ValueError, match="scene_intake_preparation_authority_invalid"):
        intake.stage_scene_intent(
            value=value,
            queue_root=tmp_path / "intents",
            authenticated_client="blueprint-webapp",
            trusted_clients={"blueprint-webapp"},
            now=NOW,
        )
    assert not (tmp_path / "intents").exists()


def test_paid_configuration_admission_is_preparation_only_and_reopens_grant(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_intake as intake
    from blueprint_pipeline import task_evaluation_scene_execution_authority as execution

    helper, task, grant, value = retained_source(tmp_path)
    clock = [NOW]
    monkeypatch.setattr(helper, "current_time", lambda: clock[0])
    monkeypatch.setattr(
        context,
        "website_webapp_request",
        lambda **kwargs: copy.deepcopy(task if kwargs["operation"] == "task-context" else grant),
    )
    root = tmp_path / "intents"
    staged = intake.stage_scene_intent(
        value=value,
        queue_root=root,
        authenticated_client="blueprint-webapp",
        trusted_clients={"blueprint-webapp"},
        now=NOW,
    )
    attempt = intake.reserve_scene_attempt(
        queue_root=root,
        intent_id=staged["intent_id"],
        attempt_id="scene-configuration-" + "c" * 24,
        source_commit="a" * 40,
        runtime_digest="sha256:" + "b" * 64,
        input_digest="sha256:" + "c" * 64,
        provider="vast",
        maximum_spend_usd=1,
        now=NOW,
    )
    config = {
        **execution.bind_scene_attempt(attempt),
        "source_commit": "a" * 40,
        "allocator": {"max_spend_usd": 1, "argv": ["--provider", "vast"]},
    }
    assert execution.scene_execution_authority_blockers(config, queue_root=root, now=NOW) == []
    robot = {**config, "task_evaluation_run": {"run_mode": "robot_evaluation"}}
    assert execution.scene_execution_authority_blockers(robot, queue_root=root, now=NOW) == [
        "scene_execution_owner_scope_excludes_robot_controls"
    ]
    grant["assessment_preparation_proposal"]["capture_id"] = "stale-capture"
    grant["authority_digest"] = canonical_digest(grant, digest_field="authority_digest")
    assert execution.scene_execution_authority_blockers(config, queue_root=root, now=NOW) == [
        "scene_execution_owner_preparation_authority_invalid"
    ]


@pytest.mark.parametrize("action", ["reservation", "configuration", "owner_review"])
def test_expiry_during_reopen_cannot_create_new_consequential_authority(
    tmp_path, monkeypatch, action
):
    import hashlib
    from blueprint_pipeline import task_evaluation_scene_intake as intake
    from blueprint_pipeline import task_evaluation_scene_execution_authority as execution
    from blueprint_pipeline import task_evaluation_scene_owner_authority as owner_review

    helper, task, grant, value = retained_source(tmp_path)
    clock = [NOW]
    expire_read = [False]
    monkeypatch.setattr(helper, "current_time", lambda: clock[0])

    def signed_read(**kwargs):
        if expire_read[0] and kwargs["operation"] == "scene-sponsorship":
            clock[0] = grant["expires_at_epoch"] + 1
        return copy.deepcopy(task if kwargs["operation"] == "task-context" else grant)

    monkeypatch.setattr(context, "website_webapp_request", signed_read)
    root = tmp_path / "intents"
    staged = intake.stage_scene_intent(
        value=value,
        queue_root=root,
        authenticated_client="blueprint-webapp",
        trusted_clients={"blueprint-webapp"},
        now=NOW,
    )
    common = dict(
        queue_root=root,
        intent_id=staged["intent_id"],
        attempt_id="scene-configuration-" + "c" * 24,
        source_commit="a" * 40,
        runtime_digest="sha256:" + "b" * 64,
        input_digest="sha256:" + "c" * 64,
        provider="vast",
        maximum_spend_usd=1,
        now=NOW,
    )
    if action == "configuration":
        reserved = intake.reserve_scene_attempt(**common)
        config = {
            **execution.bind_scene_attempt(reserved),
            "source_commit": "a" * 40,
            "allocator": {"max_spend_usd": 1, "argv": ["--provider", "vast"]},
        }
    expire_read[0] = True
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    if action == "configuration":
        assert execution.scene_execution_authority_blockers(config, queue_root=root, now=NOW) == [
            "scene_execution_owner_preparation_authority_invalid"
        ]
    elif action == "reservation":
        with pytest.raises(ValueError, match="scene_intake_preparation_authority_invalid"):
            intake.reserve_scene_attempt(**common)
        assert not (root / staged["intent_id"] / "attempts").exists()
    else:
        monkeypatch.setenv(intake.ROOT_ENV, str(root))
        path = root / staged["intent_id"] / "intent.json"
        raw = path.read_bytes()
        reference = {
            "path": str(path),
            "size_bytes": len(raw),
            "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        }
        with pytest.raises(ValueError, match="website_preparation_authority_invalid"):
            owner_review.reopen_scene_intent(reference, now=NOW)
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_signed_http_pending_preparation_refusal_is_safe_and_structured(tmp_path, monkeypatch):
    import hmac
    import time
    from datetime import datetime, timezone
    from fastapi.testclient import TestClient
    from blueprint_pipeline import live_pipeline_intake_service as service
    from blueprint_pipeline import task_evaluation_scene_intake as intake

    monkeypatch.setenv(service.INTAKE_TOKEN_ENV, "synthetic-test-token")
    monkeypatch.delenv(service.INTAKE_CLIENT_SECRETS_ENV, raising=False)
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"))
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(tmp_path / "admission"))
    monkeypatch.setenv(intake.ROOT_ENV, str(tmp_path / "intents"))
    monkeypatch.setenv(intake.CLIENTS_ENV, "blueprint-webapp")
    service._INTAKE_NONCE_CACHE.clear()
    monkeypatch.setattr(service, "deployment_identity_payload", lambda: {})
    value = copy.deepcopy(VECTOR["prepared_scene_body"]["request"])
    value["execution"]["expires_at_epoch"] = time.time() + 300
    value["consent"]["accepted_at_epoch"] = time.time() - 1
    body = json.dumps(value)
    timestamp = datetime.now(timezone.utc).isoformat()
    nonce = "synthetic-pending-refusal"
    signature = hmac.new(
        b"synthetic-test-token", f"{timestamp}.blueprint-webapp.{nonce}.{body}".encode(), "sha256"
    ).hexdigest()
    response = TestClient(service.create_app()).post(
        "/api/live-pipeline/task-evaluation-scene-intents",
        content=body,
        headers={
            "Content-Type": "application/json",
            "x-blueprint-pipeline-client-id": "blueprint-webapp",
            "x-blueprint-pipeline-timestamp": timestamp,
            "x-blueprint-pipeline-nonce": nonce,
            "x-blueprint-pipeline-signature": "sha256=" + signature,
        },
    )
    assert response.status_code == 422
    assert response.json() == {
        "status": "rejected",
        "blockers": ["scene_intake_preparation_authority_invalid"],
        "provider_mutation_performed_inside_http_request": False,
    }
    assert str(tmp_path) not in response.text and "synthetic-test-token" not in response.text
    assert not (tmp_path / "intents").exists()
