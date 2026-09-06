"""Authenticated owners traverse the real standing and candidate review consumers."""
import json
import time
from datetime import datetime, timezone

import pytest

from blueprint_pipeline import public_scene_sam31_track_selection_review as review
from blueprint_pipeline import task_evaluation_sam31_preparation_review_authority as authority
from blueprint_pipeline.task_evaluation_scene_intake import stage_scene_intent
from blueprint_pipeline.task_evaluation_scene_owner_authority import task_contract_projection
from blueprint_pipeline.task_evaluation_sam31_prefix_adoption import record
from tests.test_task_evaluation_scene_intake import request
from tests import test_task_evaluation_sam31_preparation_review_authority as authority_fixtures
from tests.test_task_evaluation_sam31_preparation_review_authority import (
    _candidate, _write,
)


@pytest.fixture
def authority_inputs(tmp_path):
    return authority_fixtures.authority_inputs.__wrapped__(tmp_path)


def bind_owner(inputs, tmp_path, monkeypatch):
    _, task_path, terms = inputs
    task = json.loads(task_path.read_text())
    value = request()
    now = time.time()
    value["task"] = task_contract_projection(task)
    value["execution"].update(expires_at_epoch=now + 3600, allowed_providers=["vast", "openai"])
    value["consent"].update(accepted_at_epoch=now - 1, provider_terms_reference=record(terms)["sha256"])
    root = tmp_path / "intents"
    staged = stage_scene_intent(value=value, queue_root=root, authenticated_client="blueprint-webapp",
        trusted_clients={"blueprint-webapp"}, now=now)
    path = root / staged["intent_id"] / "intent.json"
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", str(root))
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS", "blueprint-webapp")
    task["scene_intent_authority"] = {"intent": record(path), "intent_digest": staged["intent_digest"]}
    task["human_authority"].update(accepted_by="u1",
        accepted_on=datetime.fromtimestamp(now - 1, timezone.utc).isoformat(),
        authority_reference="scene-intent:" + staged["intent_digest"])
    _write(task_path, task, "request_digest")
    return path


def test_real_standing_and_candidate_review_accept_authenticated_nonlegacy_owner(authority_inputs, tmp_path, monkeypatch):
    intent_path = bind_owner(authority_inputs, tmp_path, monkeypatch)
    standing = tmp_path / "standing.json"
    authority.materialize_sam31_review_authority(task_request_path=authority_inputs[1],
        provider_terms_evidence_path=authority_inputs[2], output_path=standing)
    candidate = _candidate(authority_inputs, tmp_path)
    rights = tmp_path / "rights.json"
    authority.resolve_sam31_review_rights(authority_path=standing, task_request_path=authority_inputs[1],
        candidate_path=candidate, output_path=rights)
    _, value = review.validate_sam31_ai_visual_review_rights(candidate_path=candidate, rights_attestation_path=rights)
    assert value["accepted_by"] == "u1"
    assert value["scene_owner_authority"]["standing_authority"] == record(standing)
    (intent_path.parent / "revoked.json").write_text('{}')
    with pytest.raises(ValueError, match="authority_revoked"):
        review.validate_sam31_ai_visual_review_rights(candidate_path=candidate, rights_attestation_path=rights)


@pytest.mark.parametrize("fault", ["owner", "task", "terms", "path"])
def test_authenticated_owner_cannot_be_rebound(authority_inputs, tmp_path, monkeypatch, fault):
    bind_owner(authority_inputs, tmp_path, monkeypatch)
    task_path = authority_inputs[1]
    task = json.loads(task_path.read_text())
    if fault == "owner":
        task["human_authority"]["accepted_by"] = "forged-owner"
    elif fault == "task":
        task["subject"]["source_instance_id"] = "another-object"
    elif fault == "path":
        monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", str(tmp_path / "wrong-root"))
    else:
        authority_inputs[2].write_text(authority_inputs[2].read_text() + " ")
    _write(task_path, task, "request_digest")
    with pytest.raises(ValueError):
        authority.materialize_sam31_review_authority(task_request_path=task_path,
            provider_terms_evidence_path=authority_inputs[2], output_path=tmp_path / "refused.json")
    assert not (tmp_path / "refused.json").exists()
