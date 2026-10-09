"""Nullable capture account is data, never ownership/retirement permission."""

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline import capture_original_owner_observer as observer
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest

VECTOR = json.loads(
    (Path(__file__).parent / "fixtures/website-prep-owner-observation-v3.json").read_text()
)


def check(value, **kwargs):
    return observer.validate_observation(
        value,
        bucket=value["bucket"],
        scene_id=value["scene_id"],
        capture_id=value["capture_id"],
        marker_generation=value["completion_marker"]["generation"],
        now_epoch=value["observed_at_epoch"],
        **kwargs,
    )


def test_frozen_default_owner_projection_unchanged():
    value = VECTOR["default_response"]
    assert check(value) == value
    assert (
        cross_runtime_canonical_digest(value, digest_field="observation_digest")
        == value["observation_digest"]
    )


def test_explicit_preparation_observation_preserves_null_owner():
    value = VECTOR["preparation_response"]
    assert check(value, expected_purpose="scene_preparation") == value
    assert value["capture_owner"] is None


def test_default_cannot_accept_preparation_null_owner():
    with pytest.raises(ValueError):
        check(VECTOR["preparation_response"])


@pytest.mark.parametrize("purpose", [None, "evaluation", "restore", "delete"])
def test_nullable_owner_cannot_cross_purpose_boundary(purpose):
    value = copy.deepcopy(VECTOR["preparation_response"])
    if purpose is None:
        value.pop("purpose")
    else:
        value["purpose"] = purpose
    with pytest.raises(ValueError):
        check(value, expected_purpose="scene_preparation")


def test_preparation_observation_keeps_present_real_owner():
    value = copy.deepcopy(VECTOR["preparation_response"])
    value["capture_owner"] = copy.deepcopy(VECTOR["default_response"]["capture_owner"])
    fields = (*observer._SOURCE_KEYS, "purpose")
    value["source_projection_digest"] = cross_runtime_canonical_digest(
        {k: value[k] for k in fields}
    )
    value["observation_digest"] = cross_runtime_canonical_digest(
        value, digest_field="observation_digest"
    )
    assert (
        check(value, expected_purpose="scene_preparation")["capture_owner"]
        == VECTOR["default_response"]["capture_owner"]
    )


def prep_birth_fixture(tmp_path, monkeypatch):
    from tests.test_capture_generation_birth import _fixture

    access, policy, target, value, selector, raw = _fixture(tmp_path, monkeypatch)
    value.update(purpose="scene_preparation", capture_owner=None)
    value["source_projection_digest"] = cross_runtime_canonical_digest(
        {k: value[k] for k in (*observer._SOURCE_KEYS, "purpose")}
    )
    value["observation_digest"] = cross_runtime_canonical_digest(
        value, digest_field="observation_digest"
    )
    return access, policy, target, value, selector, raw


def test_data_only_birth_requires_explicit_purpose_and_projection_preserves_null(
    tmp_path, monkeypatch
):
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import (
        birth_capture_member,
        capture_birth_source_projection,
    )

    _, _, target, value, selector, raw = prep_birth_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        birth_capture_member(
            target, observation=value, membership_selector=selector, membership_raw=raw
        )
    assert not target.exists()
    born = birth_capture_member(
        target,
        observation=value,
        membership_selector=selector,
        membership_raw=raw,
        expected_purpose="scene_preparation",
    )
    assert born["capture_owner_user_id"] is None
    assert born["capture_observation_purpose"] == "scene_preparation"
    with pytest.raises(ValueError):
        capture_birth_source_projection(target)
    projected = capture_birth_source_projection(target, expected_purpose="scene_preparation")
    assert projected["capture_owner_user_id"] is None
    assert projected["capture_observation_purpose"] == "scene_preparation"


def test_null_owner_cannot_authorize_retired_rebirth(tmp_path, monkeypatch):
    import hashlib
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import birth_capture_member

    _, policy, target, value, selector, raw = prep_birth_fixture(tmp_path, monkeypatch)
    born = birth_capture_member(
        target,
        observation=value,
        membership_selector=selector,
        membership_raw=raw,
        expected_purpose="scene_preparation",
    )
    # Synthetic state injection only: no deletion/restore operation is executed.
    state_path = Path(policy["generation_store"]) / (
        hashlib.sha256(str(target).encode()).hexdigest() + ".json"
    )
    state = dict(born, state="retired")
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    state["state_digest"] = canonical_digest(state, digest_field="state_digest")
    state_path.write_text(json.dumps(state))
    # Otherwise-valid new delivery: varying only absent account ownership
    # isolates the nonnull re-birth restriction rather than stale media.
    from tests.test_capture_generation_birth import _next_delivery

    value, selector, raw = _next_delivery(value, raw, new_video=True)
    value["source_projection_digest"] = cross_runtime_canonical_digest(
        {k: value[k] for k in (*observer._SOURCE_KEYS, "purpose")}
    )
    value["observation_digest"] = cross_runtime_canonical_digest(
        value, digest_field="observation_digest"
    )
    target.rmdir()  # This disposable test-created empty directory only.
    with pytest.raises(ValueError, match="scene_capture_generation_unavailable"):
        birth_capture_member(
            target,
            observation=value,
            membership_selector=selector,
            membership_raw=raw,
            expected_purpose="scene_preparation",
        )
    assert not target.exists() and state_path.read_text() == json.dumps(state)


@pytest.mark.parametrize("current_grant", ["valid", "stale_proposal", "expired", "withdrawn"])
def test_pending_status_requires_current_grant_and_keeps_null_account_data(
    tmp_path, monkeypatch, current_grant
):
    import time
    from blueprint_pipeline import pubsub_handoff_listener as listener
    from blueprint_pipeline import website_task_context as context_reader
    from blueprint_pipeline import website_preparation_status as status_reader
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import birth_capture_member
    from blueprint_pipeline.common import write_json
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from tests.test_website_preparation_contract_v2 import VECTOR as CONTRACT

    _, _, root, owner, selector, raw = prep_birth_fixture(tmp_path, monkeypatch)
    birth_capture_member(
        root,
        observation=owner,
        membership_selector=selector,
        membership_raw=raw,
        expected_purpose="scene_preparation",
    )
    task = copy.deepcopy(CONTRACT["task_context_response_preparation"])
    task.update({key: owner[key] for key in ("request_id", "scene_id", "capture_id")})
    task["task_items"] = []  # This existing capture fixture has no supplementary item photos.
    task["capture_rights"] = copy.deepcopy(owner["capture_rights"])
    task["context_digest"] = canonical_digest(task, digest_field="context_digest")
    write_json(root / "pipeline/website_task_context.json", task)
    grant = copy.deepcopy(CONTRACT["sponsorship_response"])
    grant.update({key: task[key] for key in ("request_id", "scene_id", "capture_id")})
    grant["task_context_digest"] = task["context_digest"]
    grant["consent"]["rights_reference"] = cross_runtime_canonical_digest(task["capture_rights"])
    grant["consent"]["accepted_at_epoch"] = time.time() - 1
    grant["expires_at_epoch"] = time.time() + 300
    grant["assessment_preparation_proposal"].update(
        {key: task[key] for key in ("request_id", "capture_id")}
    )
    if current_grant == "stale_proposal":
        grant["assessment_preparation_proposal"]["capture_id"] = "stale-capture"
    elif current_grant == "expired":
        grant["expires_at_epoch"] = time.time() - 0.5
    elif current_grant == "withdrawn":
        task["capture_rights"]["consent_revoked"] = True
        task["context_digest"] = canonical_digest(task, digest_field="context_digest")
    grant["authority_digest"] = canonical_digest(grant, digest_field="authority_digest")
    monkeypatch.setattr(
        observer,
        "load_original_owner_observation",
        lambda **kwargs: observer.validate_observation(owner, **kwargs),
    )
    calls = []

    def signed_read(**kwargs):
        calls.append(kwargs)
        return copy.deepcopy(task if kwargs["operation"] == "task-context" else grant)

    monkeypatch.setattr(context_reader, "website_webapp_request", signed_read)
    payload_hash = listener.payload_sha256({"source_finalize": json.loads(raw)["source_finalize"]})
    selected = {
        **{key: owner[key] for key in ("request_id", "scene_id", "capture_id")},
        "completion_marker_generation": owner["completion_marker"]["generation"],
        "producer_delivery_key": owner["producer_delivery"]["delivery_key"],
        "source_payload_sha256": payload_hash,
        "task_context_digest": json.loads(
            (root / "pipeline/website_task_context.json").read_text()
        )["context_digest"],
    }
    # Retain the source's original context digest, even for the withdrawn response.
    claim, _ = listener._claim_job_lease(
        root,
        scene_id=owner["scene_id"],
        capture_id=owner["capture_id"],
        owner="synthetic-worker",
        lease_seconds=900,
        producer_delivery_key=selected["producer_delivery_key"],
        payload_sha256=payload_hash,
    )
    assert claim == "claimed"
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    if current_grant == "valid":
        result = status_reader.read_preparation_status(capture_root=root, selectors=selected)
        assert result["state"] == "preparing" and result["attempt_count"] == 1
        assert calls[1]["payload"]["create"] is False
        assert "capture_owner" not in result and "completed" not in result["state"]
    else:
        with pytest.raises(ValueError):
            status_reader.read_preparation_status(capture_root=root, selectors=selected)
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
