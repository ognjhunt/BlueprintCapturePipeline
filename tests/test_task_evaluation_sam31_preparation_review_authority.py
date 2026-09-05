"""Standing task permission never extends historical candidate authorization."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import public_scene_sam31_track_selection_review as review
from blueprint_pipeline import task_evaluation_sam31_preparation_review_authority as module
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_removal_selection import materialize_public_scene_removal_selections
from tests.test_public_scene_calibrated_object_masks import _fixture as media_fixture
from tests.test_public_scene_removal_selection import _source_fixture
from tests.test_task_evaluation_scene_configuration_submission import SHA
from tests import test_task_evaluation_sam31_preparation_profile as profile_tests


@pytest.fixture
def profile_inputs(tmp_path, monkeypatch):
    return profile_tests.inputs.__wrapped__(tmp_path, monkeypatch)


def _write(p: Path, v: dict, field: str | None = None) -> Path:
    if field:
        v[field] = canonical_digest(v, digest_field=field)
    p.write_text(json.dumps(v))
    return p


@pytest.fixture
def authority_inputs(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    fixture = _source_fixture(root)
    task_path = fixture['task_request']
    task = json.loads(task_path.read_text())
    task['human_authority'] = {
        'accepted_by': review.AI_REVIEW_ACCEPTED_BY, 'accepted_on': '2026-09-05',
        'authority_reference': 'current-user-directive-scene841757',
        'sam31_visual_review_authorized': True, 'sam31_visual_review_maximum_cost_usd': 1.,
        'private_derived_frame_disclosure_authorized': True,
        'provider_retention_terms_accepted': True, 'provider_training_terms_accepted': True,
        'provider_training_authorized': False,
    }
    _write(task_path, task, 'request_digest')
    terms = _write(tmp_path / 'historical-candidate-rights.json', {
        'schema_version': review.AI_RIGHTS_SCHEMA_VERSION,
        'status': 'accepted_for_private_derived_visual_review', **module.TERMS,
        'source_candidate_digest': 'sha256:' + 'a' * 64,
        'review_media_digest': 'sha256:' + 'b' * 64,
        'accepted_by': review.AI_REVIEW_ACCEPTED_BY, 'accepted_on': '2026-08-01',
        'human_authority_reference': 'historical-terms-acceptance-for-another-scene',
    }, 'attestation_digest')
    return fixture, task_path, terms


def _standing(inputs, tmp_path):
    _, task, terms = inputs
    path = tmp_path / 'standing.json'
    module.materialize_sam31_review_authority(task_request_path=task,
        provider_terms_evidence_path=terms, output_path=path)
    return path


def _candidate(inputs, tmp_path):
    fixture, task, _ = inputs
    result = materialize_public_scene_removal_selections(task_request_path=task,
        installation_receipt_path=fixture['installation_receipt'],
        publisher_intake_path=fixture['publisher_intake'],
        source_preparation_receipt_path=fixture['source_preparation'],
        expected_production_commit=SHA, output_root=tmp_path / 'selection')
    freeze_path = Path(result['task_selection']['path'])
    freeze = json.loads(freeze_path.read_text())
    media_root = tmp_path / 'media'
    media_root.mkdir()
    media = media_fixture(media_root, camera_count=16)
    task_id = freeze['task_id']
    output = tmp_path / 'candidate'
    review.materialize_sam31_track_selection_review_candidate(task_freeze_paths=[freeze_path],
        task_inputs={task_id: media['task_inputs']['task_b']},
        selected_track_ids_by_task={task_id: ['laptop-track']}, output_root=output)
    return output / 'public_scene_sam31_track_selection_review_candidate.v1.json'


def test_current_task_scope_materializes_before_candidate_exists(authority_inputs, tmp_path):
    p = _standing(authority_inputs, tmp_path)
    v = module.validate_sam31_review_authority(p, task_request_path=authority_inputs[1])
    assert v['human_authority_reference'] == 'current-user-directive-scene841757'
    assert v['historical_candidate_authority_reused'] is False
    assert v['provider_terms_evidence_use'] == 'retained_provider_terms_only'
    assert v['human_review_required'] is False
    assert v['track_selection_accepted'] is False
    assert v['review_frame_count'] == 16
    assert v['max_inference_spend_usd'] == 1.


def test_real_candidate_gets_fresh_exact_rights_and_derivation(authority_inputs, tmp_path):
    standing = _standing(authority_inputs, tmp_path)
    candidate = _candidate(authority_inputs, tmp_path)
    out = tmp_path / 'rights.json'
    assert module.resolve_sam31_review_rights(authority_path=standing,
        task_request_path=authority_inputs[1], candidate_path=candidate, output_path=out) == out
    _, rights = review.validate_sam31_ai_visual_review_rights(candidate_path=candidate,
                                                            rights_attestation_path=out)
    assert rights['source_candidate_digest'] != 'sha256:' + 'a' * 64
    assert rights['review_media_digest'] != 'sha256:' + 'b' * 64
    assert rights['accepted_on'] == '2026-09-05'
    assert rights['issued_by_agent'] is False
    assert len(rights['overlay_sha256']) == 16
    derivation = json.loads(out.with_suffix('.derivation.json').read_text())
    assert derivation['new_terms_acceptance'] is False
    assert derivation['receipt_digest'] == canonical_digest(derivation, digest_field='receipt_digest')
    # Compatibility preserves exact receipt bytes and does not create a new grant.
    assert module.resolve_sam31_review_rights(authority_path=out,
        task_request_path=authority_inputs[1], candidate_path=candidate,
        output_path=tmp_path / 'unused.json') == out
    assert not (tmp_path / 'unused.json').exists()


@pytest.mark.parametrize('field,value', [('sam31_visual_review_authorized', False),
    ('sam31_visual_review_maximum_cost_usd', 2.), ('sam31_visual_review_maximum_cost_usd', True),
    ('private_derived_frame_disclosure_authorized', False), ('provider_retention_terms_accepted', False),
    ('provider_training_terms_accepted', False), ('provider_training_authorized', True),
    ('accepted_by', 'another-person')])
def test_historical_terms_never_supply_missing_current_task_scope(authority_inputs, tmp_path, field, value):
    task_path = authority_inputs[1]
    v = json.loads(task_path.read_text())
    v['human_authority'][field] = value
    _write(task_path, v, 'request_digest')
    with pytest.raises(module.Sam31ReviewAuthorityError, match='task_'):
        _standing(authority_inputs, tmp_path)


@pytest.mark.parametrize('target', ['task', 'terms', 'receipt'])
def test_reopening_rejects_changed_authority_bytes(authority_inputs, tmp_path, target):
    p = _standing(authority_inputs, tmp_path)
    changed = {'task': authority_inputs[1], 'terms': authority_inputs[2], 'receipt': p}[target]
    v = json.loads(changed.read_text())
    v['changed'] = True
    _write(changed, v)
    with pytest.raises(ValueError):
        module.validate_sam31_review_authority(p)


def test_standing_receipt_cannot_be_used_for_another_task(authority_inputs, tmp_path):
    p = _standing(authority_inputs, tmp_path)
    copied = tmp_path / 'other-task.json'
    copied.write_bytes(authority_inputs[1].read_bytes())
    with pytest.raises(module.Sam31ReviewAuthorityError, match='task_mismatch'):
        module.validate_sam31_review_authority(p, task_request_path=copied)


def test_terms_actor_and_policy_are_rechecked_even_after_resigning(authority_inputs, tmp_path):
    p = authority_inputs[2]
    v = json.loads(p.read_text())
    v['openai_image_safety_review_terms_accepted'] = False
    _write(p, v, 'attestation_digest')
    with pytest.raises(module.Sam31ReviewAuthorityError, match='provider_terms_invalid'):
        _standing(authority_inputs, tmp_path)


def test_cli_exclusive_write_preserves_first_receipt(authority_inputs, tmp_path, capsys):
    out = tmp_path / 'authority.json'
    argv = ['--task-request', str(authority_inputs[1]), '--provider-terms-evidence',
            str(authority_inputs[2]), '--output', str(out)]
    assert module.main(argv) == 0
    original = out.read_bytes()
    capsys.readouterr()
    assert module.main(argv) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'blocked'
    assert out.read_bytes() == original


def test_profile_accepts_and_reopens_standing_task_authority(authority_inputs, profile_inputs, tmp_path):
    from blueprint_pipeline.task_evaluation_sam31_preparation_profile import (
        materialize_sam31_preparation_profile, Sam31PreparationProfileError,
    )
    task = json.loads(authority_inputs[1].read_text())
    task['expected_production_commit'] = profile_inputs['source_commit']
    _write(authority_inputs[1], task, 'request_digest')
    path = _standing(authority_inputs, tmp_path)
    profile_inputs['sam31_review_rights_attestation_path'] = path
    profile = materialize_sam31_preparation_profile(**profile_inputs)
    assert profile['sam31_visual_review']['rights_attestation_digest'] == (
        module.validate_sam31_review_authority(path)['authority_digest'])
    authority_inputs[1].write_text('{}')
    with pytest.raises(Sam31PreparationProfileError, match='review_rights_attestation_invalid'):
        materialize_sam31_preparation_profile(**profile_inputs)


def test_historical_candidate_grant_cannot_be_rebound_implicitly(authority_inputs, tmp_path):
    candidate = _candidate(authority_inputs, tmp_path)
    with pytest.raises(review.Sam31TrackSelectionReviewError, match='rights_attestation_invalid'):
        module.resolve_sam31_review_rights(authority_path=authority_inputs[2],
            task_request_path=authority_inputs[1], candidate_path=candidate,
            output_path=tmp_path / 'must-not-exist.json')
    assert not (tmp_path / 'must-not-exist.json').exists()


def test_profile_rejects_standing_authority_from_another_commit(authority_inputs, profile_inputs, tmp_path):
    from blueprint_pipeline.task_evaluation_sam31_preparation_profile import (
        materialize_sam31_preparation_profile, Sam31PreparationProfileError,
    )
    profile_inputs['sam31_review_rights_attestation_path'] = _standing(authority_inputs, tmp_path)
    with pytest.raises(Sam31PreparationProfileError, match='review_authority_commit_mismatch'):
        materialize_sam31_preparation_profile(**profile_inputs)
