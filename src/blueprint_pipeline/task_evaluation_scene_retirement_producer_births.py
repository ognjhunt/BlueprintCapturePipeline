"""Connect real downstream creation to one exact authenticated preparation.

The caller first validates its native request and materialized references. An
unregistered preparation remains compatible, without granting retirement proof.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from . import task_evaluation_scene_retirement_access as access
from .task_evaluation_scene_retirement_access import _canonical, _identity, _opened, _require
from .task_evaluation_scene_retirement_authority import load_document, selected_document
from .task_evaluation_scene_retirement_generations import _birth_gate, _sealed, _write


def _inherit(target, *, policy, authority, intent, reference, now):
    born = access.birth_scene_member(target, owner_intent_id=intent['intent_id'],
        owner_raw_ref=authority['intent_raw_ref'],
        birth_request_raw_ref=authority['attempt_raw_ref'], now=now)
    if born is None:
        return None
    child_key = hashlib.sha256(str(_canonical(str(target))).encode()).hexdigest()
    store = Path(policy['generation_store'])
    with _opened(store, directory=True) as (fd, info), \
            _birth_gate(fd, child_key, parent_identity=_identity(info)):
        current, _ = load_document(store / (child_key + '.json'), maximum=65536)
        _require(current == born
                 and current.get('source_storage_authority_raw_ref') in (None, reference),
                 'scene_retirement_storage_authority_unproven')
        if current.get('source_storage_authority_raw_ref') is None:
            current = _sealed(dict(current, source_storage_authority_raw_ref=reference,
                                   state_sequence=current['state_sequence'] + 1))
            _write(fd, child_key + '.json', current, parent_identity=_identity(info), replace=True)
        return current


def enroll_preparation_child(target, *, preparation_root, request, verified_paths=(), now=None):
    policy = access._policy()
    if policy is None:
        return None
    parent = _canonical(str(preparation_root))
    _require(parent.name == request.get('preparation_id'),
             'scene_retirement_storage_authority_unproven')
    key = hashlib.sha256(str(parent).encode()).hexdigest() + '.json'
    with access.scene_access(parent):
        try:
            generation, _ = load_document(Path(policy['generation_store']) / key, maximum=65536)
        except FileNotFoundError:
            return None
        _require(generation.get('canonical_path') == str(parent)
                 and generation.get('schema_version') == 'scene_member_generation.v1',
                 'scene_retirement_storage_authority_unproven')
        reference = generation.get('source_storage_authority_raw_ref')
        if reference is None:
            return None
        authority = selected_document(reference, maximum=65536)
        from .task_evaluation_scene_retirement_cache import _validate
        intent, _ = _validate(authority, request, now=time.time() if now is None else now)
        _require(generation.get('owner_intent_id') == intent['intent_id']
                 and generation.get('owner_raw_ref') == authority['intent_raw_ref']
                 and generation.get('birth_request_raw_ref') == authority['attempt_raw_ref'],
                 'scene_retirement_storage_authority_unproven')
        # No source-parent/name search: these are the exact native verifier's
        # positively selected projection paths, never global CAS ownership.
        _require(len(verified_paths) <= 10000
                 and all(_canonical(str(path)).is_relative_to(parent) for path in verified_paths),
                 'scene_retirement_storage_authority_unproven')
        return _inherit(target, policy=policy, authority=authority, intent=intent,
                        reference=reference, now=now)


def enroll_activation_child(target, *, activation_result, now=None):
    """Bind a native canary writer to its exact current activation, never a name search."""
    policy = access._policy()
    if policy is None or activation_result is None:
        return None
    raw_path = activation_result.get('policy_canary_runtime_inputs_path')
    if raw_path is None:
        return None  # Unsupported legacy activation supplies no birth authority.
    runtime_path = _canonical(raw_path)
    parent = runtime_path.parent
    if not any(parent.is_relative_to(Path(row['root'])) for row in policy['roots']):
        return None
    key = hashlib.sha256(str(parent).encode()).hexdigest() + '.json'
    code = 'scene_retirement_storage_authority_unproven'
    with access.scene_access(parent):
        try:
            generation, _ = load_document(Path(policy['generation_store']) / key, maximum=65536)
        except FileNotFoundError:
            return None
        _require(generation.get('canonical_path') == str(parent)
                 and generation.get('schema_version') == 'scene_member_generation.v1', code)
        reference = generation.get('source_storage_authority_raw_ref')
        if reference is None:
            return None
        authority = selected_document(reference, maximum=65536)
        from .task_evaluation_scene_retirement_cache import _sidecar, _validate
        _sidecar(authority)
        request = selected_document(authority['submission_request_raw_ref'], maximum=65536)
        intent, _ = _validate(authority, request, now=time.time() if now is None else now)
        _require(generation.get('owner_intent_id') == intent['intent_id']
                 and generation.get('owner_raw_ref') == authority['intent_raw_ref']
                 and generation.get('birth_request_raw_ref') == authority['attempt_raw_ref'], code)
        # These are the real activation worker's retained selectors. No future
        # receipt, caller-chosen ancestor, or mutable projection grants ownership.
        _require(activation_result.get('schema_version') == 'task_evaluation_launch_activation_result.v1'
            and activation_result.get('status') == 'policy_campaign_queue_materialized_no_execution'
            and activation_result.get('run_kind') == 'internal_policy_canary'
            and activation_result.get('claim_ceiling') == 'diagnostic_policy_execution'
            and activation_result.get('provider_mutation_performed') is False
            and activation_result.get('paid_execution_requested') is False
            and activation_result.get('full_byte_activation_reference_readback_passed') is True
            and activation_result.get('result_digest') == canonical_digest(activation_result, digest_field='result_digest')
            and activation_result.get('activation_id') == parent.name == _canonical(str(target)).name
            and activation_result.get('preparation_id') == request['preparation_id']
            and activation_result.get('request_digest') == authority['request_digest']
            and activation_result.get('source_commit') == authority['source_commit']
            and activation_result.get('scene_intent_digest') == intent['intent_digest']
            and runtime_path.name == 'task_evaluation_policy_canary_runtime_inputs.v1.json', code)
        runtime, runtime_ref = load_document(runtime_path, maximum=65536)
        manifest, manifest_ref = load_document(
            parent / 'task_evaluation_policy_campaign_activation.v1.json', maximum=65536)
        _require(runtime_ref['sha256'] == activation_result.get('policy_canary_runtime_inputs_sha256')
            and manifest_ref['sha256'] == activation_result.get('policy_campaign_activation_sha256')
            and runtime.get('schema_version') == 'task_evaluation_policy_canary_runtime_inputs.v1'
            and manifest.get('schema_version') == 'task_evaluation_policy_campaign_activation.v1'
            and runtime.get('runtime_inputs_digest') == activation_result.get('policy_canary_runtime_inputs_digest')
                == canonical_digest(runtime, digest_field='runtime_inputs_digest')
            and runtime.get('activation_digest') == manifest.get('activation_digest')
                == activation_result.get('policy_campaign_activation_digest')
                == canonical_digest(manifest, digest_field='activation_digest'), code)
        return _inherit(target, policy=policy, authority=authority, intent=intent,
                        reference=reference, now=now)


def existing_scene_directory(target):
    """Existing bytes retain their original admission; this never adopts a birth."""
    try:
        with _opened(_canonical(str(target)), directory=True):
            return True
    except FileNotFoundError:
        return False


def current_directory_generation(target):
    policy = access._policy()
    if policy is None:
        return None
    target = _canonical(str(target))
    key = hashlib.sha256(str(target).encode()).hexdigest() + '.json'
    try:
        value, _ = load_document(Path(policy['generation_store']) / key, maximum=65536)
    except FileNotFoundError:
        return None
    _require(value.get('canonical_path') == str(target)
             and value.get('schema_version') == 'scene_member_generation.v1'
             and value.get('state') in {'active', 'restored-active'}
             and value.get('state_digest') == canonical_digest(value, digest_field='state_digest'),
             'scene_retirement_generation_unavailable')
    return value


def _existing_canary_identity(target, value):
    generation = current_directory_generation(target)
    if generation is None:
        return  # No original birth is adopted by a legacy retained read.
    code = 'scene_retirement_storage_authority_unproven'
    _require(type(value) is dict and generation.get('source_storage_authority_raw_ref') is not None, code)
    authority = selected_document(generation['source_storage_authority_raw_ref'], maximum=65536)
    from .task_evaluation_scene_retirement_cache import _sidecar
    _sidecar(authority)
    request = selected_document(authority['submission_request_raw_ref'], maximum=65536)
    _require(value.get('result_digest') == canonical_digest(value, digest_field='result_digest')
        and value.get('activation_id') == Path(target).name
        and value.get('request_digest') == authority['request_digest']
        and value.get('preparation_id') == request['preparation_id']
        and value.get('source_commit') == authority['source_commit']
        and value.get('scene_intent_digest') == request['scene_intent_digest']
        and generation.get('owner_raw_ref') == authority['intent_raw_ref']
        and generation.get('birth_request_raw_ref') == authority['attempt_raw_ref'], code)


def create_canary_output(target, *, activation_result):
    if access._policy() is None:
        Path(target).mkdir(parents=True, exist_ok=True)
        return
    with access.scene_access(target):
        if existing_scene_directory(target):
            _existing_canary_identity(target, activation_result)
            return  # Retained delivery/teardown never needs a new paid reservation.
        if enroll_activation_child(target, activation_result=activation_result) is None:
            Path(target).mkdir(parents=True, exist_ok=True)


def create_preparation_output(target, *, preparation_root, request):
    """First-write enrollment; retained bytes keep their exact original request."""
    if access._policy() is None:
        Path(target).mkdir(parents=True, exist_ok=True)
        return
    with access.scene_access(target):
        if existing_scene_directory(target):
            generation = current_directory_generation(target)
            if generation is not None:
                reference = generation.get('source_storage_authority_raw_ref')
                _require(reference is not None, 'scene_retirement_storage_authority_unproven')
                authority = selected_document(reference, maximum=65536)
                from .task_evaluation_scene_retirement_cache import _sidecar
                _sidecar(authority)
                original = selected_document(authority['submission_request_raw_ref'], maximum=65536)
                _require(original == request
                    and _canonical(str(preparation_root)).name == request.get('preparation_id')
                    and generation.get('owner_raw_ref') == authority['intent_raw_ref']
                    and generation.get('birth_request_raw_ref') == authority['attempt_raw_ref'],
                    'scene_retirement_storage_authority_unproven')
            return  # Legacy existing output is not adopted into a new generation.
        if enroll_preparation_child(target, preparation_root=preparation_root, request=request) is None:
            Path(target).mkdir(parents=True, exist_ok=True)
