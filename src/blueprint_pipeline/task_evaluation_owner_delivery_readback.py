"""Finish an owned policy dispatch only after inbox and full-download readback.

This is terminal delivery: owner authority is checked at its original acceptance
time, so later execution expiry or revocation cannot strand already paid evidence.
No allocator, reservation, or scientific producer is invoked here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import task_evaluation_scene_intake as intake
from .task_evaluation_scene_policy_binding import scene_store
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_submission_inputs import read as read_preparation_record
from .task_evaluation_delivery_readback import (
    _atomic, delivery_readback_matches, verify_website_delivery,
)


def _retained_json(reference):
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256', 'size_bytes'}:
        raise ValueError('owner_delivery_reference_invalid')
    path = Path(reference['path'])
    if (not path.is_absolute() or not path.is_file()
            or any(p.is_symlink() for p in (path, *path.parents)) or _record(path) != reference):
        raise ValueError('owner_delivery_reference_changed')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('owner_delivery_reference_invalid')
    return value


def _website_team(directory, intent, setup):
    """Reopen the factory's original request through the immutable owner event chain.

    Owner organizations and Website scene namespaces are distinct. The latter
    comes from the retained prepared request, never a reconstruction of the
    namespace formula or the mutable latest preparation-link pointer. Scanning
    retained events also keeps an older paid run deliverable after a successor.
    """
    paths = sorted((directory / 'progression-events').glob('*.json'))
    if not 1 <= len(paths) <= 10000:
        raise ValueError('owner_delivery_preparation_lineage_missing')
    previous, candidates, seen = None, set(), set()
    for sequence, path in enumerate(paths, 1):
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('owner_delivery_preparation_lineage_unsafe')
        event = intake._read(path, 'event_digest')
        if (event.get('schema_version') != 'task_evaluation_scene_progression_event.v1'
                or event.get('intent_id') != intent['intent_id']
                or event.get('intent_digest') != intent['intent_digest']
                or event.get('sequence') != sequence or path.name != f'{sequence:06d}.json'
                or event.get('previous_event_digest') != previous):
            raise ValueError('owner_delivery_preparation_lineage_invalid')
        previous = event['event_digest']
        state = event.get('state') or {}
        reference = state.get('factory')
        if reference is None or reference.get('sha256') in seen:
            continue
        seen.add(reference.get('sha256'))
        factory = _retained_json(reference)
        if (factory.get('factory_digest') != canonical_digest(factory, digest_field='factory_digest')
                or factory.get('intent_digest') != intent['intent_digest']
                or factory.get('status') != 'publication_ready'):
            raise ValueError('owner_delivery_factory_invalid')
        preparation = _retained_json(factory['submission_request'])
        if preparation.get('run_id') != setup['capture_session_id']:
            continue
        from .task_evaluation_launch_preparation_contract import validate_launch_preparation_request
        validate_launch_preparation_request(preparation)
        task = _retained_json(factory['task_request'])
        attempt = _retained_json(state['attempt'])
        binding = read_preparation_record(Path(reference['path']).parent / 'source_binding.json', digest_field='binding_digest')
        source = intent['request']['source']
        team = preparation.get('team_namespace')
        if (preparation.get('scene_intent_digest') != intent['intent_digest']
                or not intake._identifier(team) or task.get('team_namespace') != team
                or preparation.get('task', {}).get('identity', {}).get('id') != intent['request']['task']['task_id']
                or attempt.get('attempt_digest') != intake.canonical_digest(attempt, digest_field='attempt_digest')
                or attempt.get('attempt_digest') != factory.get('attempt_digest')
                or attempt.get('intent_digest') != intent['intent_digest']
                or attempt.get('source_commit') != factory.get('source_commit')
                or preparation.get('expected_production_commit') != factory.get('source_commit')
                or task.get('expected_production_commit') != factory.get('source_commit')
                or binding.get('binding_digest') != state.get('binding_digest')
                or binding.get('binding_digest') != attempt.get('input_digest')
                or binding.get('binding_id') != source['binding_id']
                or binding.get('source_content_digest') != source['content_digest']):
            raise ValueError('owner_delivery_preparation_binding_invalid')
        candidates.add(team)
    if len(candidates) != 1:
        raise ValueError('owner_delivery_preparation_not_unique')
    return candidates.pop()


def _owner_identity(setup, runtime_inputs, run_id):
    binding = setup['scene_attempt_binding']
    intent_id, attempt_id = binding.get('intent_id'), binding.get('attempt_id')
    if not intake._identifier(intent_id) or not intake._identifier(attempt_id):
        raise ValueError('owner_delivery_binding_invalid')
    root, trusted_clients = scene_store()
    paths = (root / intent_id / 'intent.json', root / intent_id / 'attempts' / (attempt_id + '.json'))
    if any(p.is_symlink() for path in paths for p in (path, *path.parents)):
        raise ValueError('owner_delivery_store_unsafe')
    intent = intake._read(paths[0], 'intent_digest')
    attempt = intake._read(paths[1], 'attempt_digest')
    if (intent.get('schema_version') != intake.INTENT_SCHEMA
            or intent.get('intent_id') != intent_id
            or intent.get('authenticated_issuer') not in trusted_clients
            or intent['intent_digest'] != binding.get('intent_digest')
            or intent['intent_digest'] != setup.get('scene_intent_digest')
            or any(attempt.get(key) != binding.get(key) for key in (
                'intent_digest', 'attempt_id', 'source_commit', 'runtime_digest', 'input_digest'))):
        raise ValueError('owner_delivery_intent_mismatch')
    request = intake.validate_request(intent['request'], now=intent['accepted_at_epoch'])
    return {'run_id': run_id, 'capture_session_id': setup['capture_session_id'],
            'request_digest': setup['request_digest'],
            'configuration_digest': runtime_inputs['configuration_digest'],
            'owner_user_id': request['owner']['user_id'],
            'team_namespace': _website_team(root / intent_id, intent, setup)}


def _record(path):
    with path.open('rb') as stream:
        digest = 'sha256:' + hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'size_bytes': path.stat().st_size}


def verify_owner_policy_delivery(*, root, setup, runtime_inputs, delivery, projection,
                                 publication, readback_runner=None):
    """Resume downloaded-byte journals; no dispatch receipt while any proof is missing."""
    if 'scene_attempt_binding' not in setup:
        return {'status': 'not_applicable_historical_unowned_dispatch'}
    try:
        authority = _owner_identity(setup, runtime_inputs, projection['run_id'])
        identity = {key: value for key, value in authority.items() if key != 'capture_session_id'}
        identity.update(result_delivery_digest=delivery['delivery_digest'],
                        policy_canary_projection_digest=projection['projection_digest'])
        path = Path(root) / 'artifacts/result_delivery/owner_delivery_readback.json'
        if path.is_symlink():
            raise ValueError('owner_delivery_receipt_symlink')
        if path.is_file():
            readback = json.loads(path.read_text())
        else:
            runner = readback_runner or verify_website_delivery
            readback = runner(run_root=root, owner_execution=authority, result_delivery=delivery,
                              policy_canary_result=projection, publication=publication)
        if not delivery_readback_matches(readback, identity=identity, artifacts=delivery['artifacts'],
                team_namespace=authority['team_namespace'], owner_user_id=authority['owner_user_id']):
            return {'status': 'pending', 'blockers': ['policy_canary_owner_delivery_readback_pending']}
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic(path, readback)
        return {'status': 'verified', 'receipt': _record(path)}
    except Exception as exc:
        return {'status': 'pending', 'blockers': ['policy_canary_owner_delivery_readback_pending'],
                'failure_type': type(exc).__name__, 'raw_exception_recorded': False}
