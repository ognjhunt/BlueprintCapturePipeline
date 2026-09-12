"""SDK public entry reopens exact rights before any cost, tool or model call."""
import json

import pytest

from blueprint_pipeline import public_scene_sam31_ai_visual_reviewer as sdk
from blueprint_pipeline import public_scene_sam31_track_selection_review as review
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_public_scene_calibrated_object_masks import _fixture


@pytest.mark.parametrize('fault', ['raw_bytes', 'resealed_disclosure', 'resealed_candidate', 'resealed_overlay'])
def test_actual_sdk_rejects_tampered_rights_before_any_paid_boundary(tmp_path, monkeypatch, fault):
    fixture = _fixture(tmp_path, camera_count=8)
    candidate_root = tmp_path / 'candidate'
    review.materialize_sam31_track_selection_review_candidate(
        task_freeze_paths=fixture['tasks'], task_inputs=fixture['task_inputs'],
        selected_track_ids_by_task={'task_a':['washer-track'], 'task_b':['laptop-track']},
        output_root=candidate_root)
    candidate = candidate_root / 'public_scene_sam31_track_selection_review_candidate.v1.json'
    rights = tmp_path / 'rights.json'
    checked = []
    original = review.validate_sam31_ai_visual_review_rights
    def validate(**kwargs):
        checked.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(review, 'validate_sam31_ai_visual_review_rights', validate)
    review.materialize_sam31_ai_visual_review_rights(candidate_path=candidate,
        accepted_by=review.AI_REVIEW_ACCEPTED_BY, accepted_on='2026-09-12',
        human_authority_reference='fixture-owner-permission', output_path=rights)
    assert checked == [{'candidate_path':candidate, 'rights_attestation_path':rights}]
    # Creation still validates its readback. Mutate only afterward, so the SDK
    # must independently refuse the changed bytes rather than trust the creator.
    value = json.loads(rights.read_bytes())
    if fault in {'raw_bytes', 'resealed_disclosure'}:
        value['private_derived_frame_disclosure_authorized'] = False
    elif fault == 'resealed_candidate':
        value['source_candidate_digest'] = 'sha256:' + '0' * 64
    else:
        value['overlay_sha256'][0] = 'sha256:' + '0' * 64
    if fault != 'raw_bytes':
        value['attestation_digest'] = canonical_digest(value, digest_field='attestation_digest')
    rights.write_text(json.dumps(value))
    before = rights.read_bytes()
    checked.clear()
    monkeypatch.setattr(sdk, 'validate_sam31_ai_visual_review_rights', validate)
    def forbidden(*args, **kwargs):
        pytest.fail('invalid rights reached a cost, inference, or provider boundary')
    from blueprint_pipeline.agent_execution import visual_producer
    monkeypatch.setattr(visual_producer, 'best_effort_visual_investigation', forbidden)
    monkeypatch.setattr(sdk, 'build_openai_official_cost_run_gate', forbidden)
    monkeypatch.setattr(sdk, 'OpenAIAgentsSDKInvoker', forbidden)
    monkeypatch.setattr(sdk, '_scoped_review_model_provider', forbidden)
    destination = tmp_path / 'sdk-output'
    with pytest.raises(review.Sam31TrackSelectionReviewError, match='rights_attestation_invalid'):
        sdk.run_sam31_ai_visual_review(candidate_path=candidate, rights_attestation_path=rights,
            output_root=destination, openai_cost_scope_attestation_path=tmp_path / 'scope-not-read.json',
            openai_admin_api_key_file=tmp_path / 'admin-not-read', openai_project_id='fixture',
            openai_api_key_id='fixture', openai_api_key_file=tmp_path / 'inference-key-not-read')
    assert checked == [{'candidate_path':candidate, 'rights_attestation_path':rights}]
    assert not destination.exists()
    assert rights.read_bytes() == before
