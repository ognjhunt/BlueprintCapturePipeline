"""Resolve an independently funded team request against immutable scene provenance.

This admission performs no reservation or provider call. Private source access
must belong to the request owner; sharing requires its own admitted grant.
"""
from collections.abc import Mapping

from . import task_evaluation_scene_intake as intake
from . import task_evaluation_scene_policy_binding as policy
from .configured_scene_run_identity import evaluation_scope
from .task_evaluation_scene_execution_scope import scene_preparation_only


def _require(condition, code):
    if not condition:
        raise ValueError('team_evaluation_' + code)


def authority_scope(evaluation_run_id, authority):
    if authority is None:
        return {}
    evaluation_scope(evaluation_run_id)
    _require(evaluation_run_id is not None and isinstance(authority, Mapping)
        and set(authority) == {'evaluation_run_id', 'source_launch_id', 'source_profile_digest',
                              'configured_scene_revision_digest', 'scene_intent_digest'}
        and authority.get('evaluation_run_id') == evaluation_run_id
        and intake._identifier(authority.get('source_launch_id'))
        and all(isinstance(authority.get(k), str) and intake._DIGEST.fullmatch(authority[k])
            for k in ('source_profile_digest', 'configured_scene_revision_digest', 'scene_intent_digest')),
        'authority_invalid')
    return {'evaluation_authority': dict(authority)}


def evaluation_owner(*, source_profile, authority, source_launch_id,
                     configured_scene_revision_digest, evaluation_run_id, now=None):
    """Reopen the selected run's consent without inheriting preparation spend."""
    _require(evaluation_run_id is not None, 'run_identity_missing')
    evaluation_scope(evaluation_run_id)
    source = {
        'evaluation_run_id': evaluation_run_id,
        'source_launch_id': source_launch_id,
        'source_profile_digest': source_profile.get('profile_digest'),
        'configured_scene_revision_digest': configured_scene_revision_digest,
    }
    _require(isinstance(authority, Mapping) and set(authority) == {*source, 'scene_intent_digest'}
             and all(authority[k] == v for k,v in source.items()), 'source_binding_mismatch')
    _require(source_profile.get('profile_digest') == policy.canonical_digest(source_profile, digest_field='profile_digest')
             and intake._identifier(source_launch_id)
             and isinstance(configured_scene_revision_digest, str)
             and intake._DIGEST.fullmatch(configured_scene_revision_digest), 'source_identity_invalid')
    owner = policy.owner_for_profile({
        'scene_intent_digest': authority['scene_intent_digest'],
        'task_evaluation_run': source_profile.get('task_evaluation_run'),
    }, now=now)
    request = owner['request']
    _require(not scene_preparation_only(request)
             and request['submission_id'] == evaluation_run_id
             and request['task'].get('evaluation_source') == source
             and intake._identifier(request['task'].get('robot_binding_id')), 'request_scope_invalid')
    root, clients = policy.scene_store()
    original = None
    for path in root.glob('scene-*/intent.json'):
        _require(not path.is_symlink() and not path.parent.is_symlink(), 'source_store_unsafe')
        retained = intake._read(path, 'intent_digest')
        if retained['intent_digest'] == source_profile.get('scene_intent_digest'):
            _require(retained.get('schema_version') == intake.INTENT_SCHEMA
                     and retained.get('authenticated_issuer') in clients
                     and not (path.parent/'revoked.json').exists(), 'source_authority_invalid')
            original = intake.validate_request(retained['request'], now=retained['accepted_at_epoch'])
            break
    _require(original is not None, 'source_owner_missing')
    _require(request['owner'] == original['owner'], 'source_access_not_admitted')
    _require(request['source'] == original['source'] and all(
        request['task'].get(k) == original['task'].get(k)
        for k in ('task_id','strategy','subject','support','destination','success')
    ), 'source_task_mismatch')
    _require(owner['intent_digest'] != source_profile.get('scene_intent_digest'), 'separate_authority_required')
    return owner


def authorization_profile(source_profile, authority, *, source_launch_id, configured_scene_revision_digest):
    """Authorization-only projection; source artifact validation uses the original."""
    if authority is None:
        return source_profile
    owner = evaluation_owner(source_profile=source_profile, authority=authority,
        source_launch_id=source_launch_id, configured_scene_revision_digest=configured_scene_revision_digest,
        evaluation_run_id=authority.get('evaluation_run_id'))
    return {'scene_intent_digest': owner['intent_digest'],
            'profile_digest': source_profile['profile_digest'],
            'task_evaluation_run': source_profile['task_evaluation_run']}
