"""Capture action proof must select the original native generation and delivery."""

import json
import hashlib
import os
import time
from pathlib import Path

import pytest

from tests.test_capture_generation_birth import _fixture
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest


def test_action_generation_selects_original_capture_proofs(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement import _generation
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import birth_capture_member

    _, policy, target, owner, selector, raw = _fixture(tmp_path, monkeypatch)
    born = birth_capture_member(target, observation=owner, membership_selector=selector,
                                membership_raw=raw)
    info = target.stat()
    member = dict(canonical_path=str(target), **{'class': 'site_capture'},
                  generation_id=born['generation_id'], dev=info.st_dev, ino=info.st_ino,
                  mode=info.st_mode, inventory_sha256='sha256:' + 'b' * 64,
                  capture_owner_user_id=owner['capture_owner']['user_id'],
                  request_id=owner['request_id'], sponsoring_intent_id='scene-1',
                  owner_observation_raw_ref=born['owner_observation_raw_ref'],
                  birth_delivery_raw_ref=born['birth_delivery_raw_ref'],
                  source_membership_raw_ref=json.loads(
                      Path(born['birth_delivery_raw_ref']['path']).read_bytes())[
                          'source_membership_raw_ref'],
                  association_raw_ref=born['birth_delivery_raw_ref'],
                  scene_intent_raw_ref=born['owner_observation_raw_ref'])
    assert _generation(policy, member, expected_states={'active'})[0] == born
    for changed in (dict(capture_owner_user_id='sponsor-2'),
                    dict(owner_observation_raw_ref=member['birth_delivery_raw_ref']),
                    dict(source_membership_raw_ref=member['owner_observation_raw_ref'])):
        with pytest.raises(ValueError, match='scene_retirement_generation_unavailable'):
            _generation(policy, dict(member, **changed), expected_states={'active'})


def test_capture_action_requires_current_owner_and_distinct_sponsor_association(
        tmp_path, monkeypatch):
    from blueprint_pipeline import capture_original_owner_observer as observer
    from blueprint_pipeline.task_evaluation_scene_retirement import _capture_action_current
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import (
        birth_capture_member, capture_birth_source_projection,
    )
    from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance

    _, policy, target, owner, selector, raw = _fixture(tmp_path, monkeypatch)
    born = birth_capture_member(target, observation=owner, membership_selector=selector,
                                membership_raw=raw)
    proof = capture_birth_source_projection(target)
    request = {'owner': {'user_id': 'sponsor-2'}, 'consent': {'accepted_by': 'sponsor-2'}}
    intent = {'intent_id': 'scene-1', 'request': request}
    intent['intent_digest'] = canonical_digest(intent, digest_field='intent_digest')
    selected = {key: value for key, value in proof.items() if key != 'capture_rights'}
    selected.update(capture_rights_digest=cross_runtime_canonical_digest(owner['capture_rights']),
                    sponsoring_owner=request['owner'],
                    request_digest=cross_runtime_canonical_digest(request))
    intent_root = tmp_path / 'intents'
    factory_root = tmp_path / 'factory'
    registration_root = tmp_path / 'bindings'
    policy['reference_context'] = {'roots': {
        'intent_root': str(intent_root), 'factory_output_root': str(factory_root),
        'website_source_binding_root': str(registration_root)}}

    def retained(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        data = path.read_bytes()
        return {'path': str(path), 'sha256': 'sha256:' + hashlib.sha256(data).hexdigest(),
                'size_bytes': len(data)}

    registration = {'schema_version': 'website_scene_source_registration.v1',
                    'request_digest': selected['request_digest'], 'capture_source': selected}
    registration['registration_digest'] = canonical_digest(
        registration,digest_field='registration_digest')
    registration_ref = retained(registration_root /
                                (selected['request_digest'][7:] + '.json'), registration)
    association = {'schema_version': 'website_scene_source_binding.v1',
                   'intent_digest': intent['intent_digest'], 'owner': request['owner'],
                   'capture_source': selected, 'registration': registration_ref}
    association['binding_digest'] = canonical_digest(association,digest_field='binding_digest')
    intent_ref = retained(intent_root / 'scene-1' / 'intent.json', intent)
    association_ref = retained(factory_root / 'scene-1' / 'website-source' /
                               (association['binding_digest'][7:] + '.json'), association)
    member = dict(canonical_path=str(target), **{'class': 'site_capture'},
                  generation_id=born['generation_id'], dev=born['dev'], ino=born['ino'],
                  mode=born['mode'], inventory_sha256='sha256:' + 'b' * 64,
                  capture_owner_user_id=owner['capture_owner']['user_id'],
                  request_id=owner['request_id'], sponsoring_intent_id='scene-1',
                  owner_observation_raw_ref=born['owner_observation_raw_ref'],
                  birth_delivery_raw_ref=born['birth_delivery_raw_ref'],
                  source_membership_raw_ref=proof['source_membership_raw_ref'],
                  association_raw_ref=association_ref, scene_intent_raw_ref=intent_ref)
    consent = {'intent_id': 'scene-1', 'intent_raw_ref': intent_ref}
    allowance = ActionAllowance(expires_at=int(time.time()) + 60, elapsed_seconds=30)
    calls = []

    def current(**kwargs):
        calls.append(kwargs)
        return owner, 100

    monkeypatch.setattr(observer, 'load_original_owner_observation', current)
    assert _capture_action_current(policy, consent, member, allowance,
                                   expected_states={'active'}) == born
    assert calls[0]['marker_generation'] == owner['completion_marker']['generation']
    assert calls[0]['include_response_bytes'] is True
    assert allowance.counts['remote_bytes'] == 100
    swapped = dict(association, owner={'user_id': 'other-sponsor'})
    swapped['binding_digest'] = canonical_digest(swapped, digest_field='binding_digest')
    swapped_ref = retained(factory_root / 'scene-1' / 'website-source' /
                           (swapped['binding_digest'][7:] + '.json'), swapped)
    with pytest.raises(ValueError, match='scene_retirement_capture_association_unproven'):
        _capture_action_current(policy, consent, dict(member, association_raw_ref=swapped_ref),
                                ActionAllowance(expires_at=int(time.time()) + 60,
                                                elapsed_seconds=30), expected_states={'active'})
    assert len(calls) == 1
    changed = json.loads(json.dumps(owner))
    changed['capture_rights']['consent_revoked'] = True
    monkeypatch.setattr(observer, 'load_original_owner_observation', lambda **_: (changed, 100))
    with pytest.raises(ValueError, match='scene_retirement_capture_current_owner_changed'):
        _capture_action_current(policy, consent, member,
                                ActionAllowance(expires_at=int(time.time()) + 60,
                                                elapsed_seconds=30), expected_states={'active'})

    # The public action itself must return KEEP before a journal, archive or
    # removal when the independently signed current source has changed.
    from blueprint_pipeline.task_evaluation_scene_retirement import retire_scene
    from blueprint_pipeline.task_evaluation_scene_retirement_authority import cohort_digest, load_authority

    plan_ref = retained(tmp_path / 'plan.json', {
        'selected_intent_provenance': dict(intent_ref, role='intent')})
    policy['principals'] = [dict(principal_id='operator', actions=['retire'],
                                 owner_intent_ids=['scene-1'], private_archive_classes=[],
                                 capture_owner_scopes=[dict(user_id='owner-1', request_id='req-1')])]
    policy['limits'] = {'elapsed_seconds': 30}
    policy['policy_digest'] = canonical_digest(policy, digest_field='policy_digest')
    policy_path = Path(os.environ['BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE'])
    policy_path.write_text(json.dumps(policy))
    protected = {'schema_version': 'scene_retirement_consent.v1', 'consent_id': '1' * 32,
                 'principal_id': 'operator', 'intent_id': 'scene-1', 'intent_raw_ref': intent_ref,
                 'plan_raw_ref': plan_ref, 'retired_journal_raw_ref': None,
                 'policy_sha256': 'sha256:' + hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                 'cohort_sha256': cohort_digest(policy['consumer_cohort']), 'action': 'retire',
                 'created_at': int(time.time()) - 1, 'expires_at': int(time.time()) + 60,
                 'members': [member], 'private_archive_classes': []}
    protected['consent_digest'] = canonical_digest(protected, digest_field='consent_digest')
    consent_path = tmp_path / 'consent.json'
    consent_path.write_text(json.dumps(protected))
    consent_path.chmod(0o600)
    assert load_authority(consent_path, action='retire', now=time.time)['consent'] == protected
    result = retire_scene(Path(plan_ref['path']), consent_path, transport=object())
    assert result['status'] == 'kept' and result['reason'] == 'scene_retirement_capture_current_owner_changed'
    assert result['mutations'] == 0 and target.exists()
    assert not Path(policy['journal_store']).exists()
    monkeypatch.setattr(observer, 'load_original_owner_observation', current)
    result = retire_scene(Path(plan_ref['path']), consent_path, transport=object())
    assert result['reason'] == 'scene_retirement_installed_context_unproven'
    assert result['mutations'] == 0 and target.exists()
