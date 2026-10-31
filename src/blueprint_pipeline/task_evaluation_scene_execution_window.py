"""Explicit owner-approved time extensions with immutable spending bounds."""
from __future__ import annotations

import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .task_evaluation_scene_execution_window_evidence import (
    ACK,
    DIRECTORY,
    MAXIMUM_EXTENSION_SECONDS,
    OPERATOR_DOOR,
    SCHEMA,
    _DOOR_CALLER,
    _DOOR_REQUEST,
    _bounds,
    effective_execution_expiry,
    operator_bounds,
    operator_extension_valid,
    valid_authorization_reference,
    verified_extensions,
)


def extend_scene_execution_window(*, queue_root: str | Path, intent_id: str,
        intent_digest: str, owner: Mapping[str, Any], authenticated_client: str,
        trusted_clients: set[str], expires_at_epoch: float, authorization_reference: str,
        ack: str, now: float | None = None) -> dict[str, Any]:
    """Record an explicitly approved time-only extension; never renew implicitly."""
    from . import task_evaluation_scene_intake as intake
    moment = time.time() if now is None else now
    if (ack != ACK or authenticated_client not in trusted_clients
            or not intake._identifier(intent_id) or not isinstance(authorization_reference, str)
            or not authorization_reference.strip()
            or isinstance(expires_at_epoch, bool) or not isinstance(expires_at_epoch, (int, float))
            or not math.isfinite(expires_at_epoch) or not moment < expires_at_epoch <= moment+7*86400):
        raise ValueError('scene_execution_window_extension_not_authorized')
    root = intake._root(Path(queue_root))
    with intake._lock(root):
        directory = root / intent_id
        intent = intake._read(directory/'intent.json', 'intent_digest')
        if (intent['intent_digest'] != intent_digest or intent['request']['owner'] != dict(owner)
                or intent['authenticated_issuer'] != authenticated_client
                or (directory/'revoked.json').exists()):
            raise ValueError('scene_execution_window_owner_or_revocation_mismatch')
        current = effective_execution_expiry(directory, intent)
        if expires_at_epoch <= current:
            return {'status': 'execution_window_already_covers_request', 'expires_at_epoch': current,
                    'intent_digest': intent_digest, 'provider_mutation_performed': False}
        value = intake._seal({'schema_version': SCHEMA, 'scope': 'execution_time_only',
            'intent_id': intent_id, 'intent_digest': intent_digest, 'owner': dict(owner),
            'authenticated_issuer': authenticated_client, 'authorization_reference': authorization_reference,
            'original_expires_at_epoch': intent['request']['execution']['expires_at_epoch'],
            'expires_at_epoch': expires_at_epoch, 'issued_at_epoch': moment,
            'unchanged_execution_bounds': _bounds(intent), 'provider_mutation_performed': False}, 'extension_digest')
        path = directory/DIRECTORY/(value['extension_digest'].removeprefix('sha256:')+'.json')
        path.parent.mkdir(mode=0o750, exist_ok=True)
        intake.write_exclusive(path, value)
        assert effective_execution_expiry(directory, intent) == expires_at_epoch
    return {'status': 'execution_window_extended', 'record_path': str(path), **value}


def _refuse(code: str) -> None:
    raise ValueError('scene_execution_window_' + code)


def extend_scene_execution_window_by_operator(*, queue_root: str | Path, intent_id: str, intent_digest: str,
        expires_at_epoch: float, authorization_reference: str, ack: str, door_caller: str,
        door_request_id: str, trusted_clients: set[str], capture_rights: Mapping[str, Any],
        apply: bool, now: float | None = None) -> dict[str, Any]:
    """The operator door's only write: one sealed, time-only extension of a sealed intent.

    The owner's approval is named in ``authorization_reference``; the door caller
    is recorded as such and never as the intent's WebApp issuer. Spend caps,
    attempts, retries, providers, task, source and consent are copied from the
    intent and must stay identical for every reader. ``capture_rights`` is the
    caller's current read of the capture's rights (WebApp task context): a
    revoked consent or a capture that may no longer be processed is refused.
    An equal or earlier expiry is already covered and writes nothing; ``apply``
    False plans the same checks and writes nothing.
    """
    from . import task_evaluation_scene_intake as intake
    moment = time.time() if now is None else now
    if ack != ACK:
        _refuse('extension_unacknowledged')
    if (not intake._identifier(intent_id) or not isinstance(intent_digest, str)
            or not isinstance(door_caller, str) or _DOOR_CALLER.fullmatch(door_caller) is None
            or not isinstance(door_request_id, str) or _DOOR_REQUEST.fullmatch(door_request_id) is None
            or type(apply) is not bool):
        _refuse('extension_request_invalid')
    if not valid_authorization_reference(authorization_reference):
        _refuse('authorization_reference_invalid')
    if (type(expires_at_epoch) not in (int, float) or not math.isfinite(expires_at_epoch)
            or not moment < expires_at_epoch <= moment + MAXIMUM_EXTENSION_SECONDS):
        _refuse('expiry_out_of_bounds')
    if (not isinstance(capture_rights, Mapping) or capture_rights.get('consent_revoked') is not False
            or capture_rights.get('derived_scene_generation_allowed') is not True):
        _refuse('capture_rights_revoked')
    root = intake._root(Path(queue_root))
    with intake._lock(root):
        directory = root / intent_id
        if directory.is_symlink():
            _refuse('store_unsafe')
        intent = intake._read(directory/'intent.json', 'intent_digest')
        if intent['intent_digest'] != intent_digest:
            _refuse('intent_digest_mismatch')
        if intent.get('authenticated_issuer') not in trusted_clients:
            _refuse('intent_issuer_not_trusted')
        if (directory/'revoked.json').exists():
            _refuse('intent_revoked')
        original = intent['request']['execution']['expires_at_epoch']
        if not expires_at_epoch <= original + MAXIMUM_EXTENSION_SECONDS:
            _refuse('expiry_beyond_original_window')
        current = effective_execution_expiry(directory, intent)
        if expires_at_epoch <= current:
            covering = [value for value in verified_extensions(directory, intent)
                        if value['expires_at_epoch'] == current and value.get('authorized_via') == OPERATOR_DOOR]
            return {'status': 'execution_window_already_covers_request', 'expires_at_epoch': current,
                    'intent_id': intent_id, 'intent_digest': intent_digest,
                    'covering_extension': covering[-1] if covering else None, 'provider_mutation_performed': False}
        value = intake._seal({'schema_version': SCHEMA, 'scope': 'execution_time_only',
            'authorized_via': OPERATOR_DOOR, 'door_caller': door_caller, 'door_request_id': door_request_id,
            'intent_id': intent_id, 'intent_digest': intent_digest, 'owner': intent['request']['owner'],
            'intent_authenticated_issuer': intent['authenticated_issuer'],
            'authorization_reference': authorization_reference,
            'original_expires_at_epoch': original, 'expires_at_epoch': expires_at_epoch,
            'issued_at_epoch': moment, 'unchanged_execution_bounds': operator_bounds(intent),
            'provider_mutation_performed': False}, 'extension_digest')
        name = value['extension_digest'].removeprefix('sha256:') + '.json'
        if not operator_extension_valid(value, intent=intent, file_name=name):
            _refuse('extension_invalid')
        if not apply:
            return {'status': 'execution_window_extension_planned', 'previous_expires_at_epoch': current,
                    'extension': value, 'provider_mutation_performed': False}
        path = directory/DIRECTORY/name
        path.parent.mkdir(mode=0o750, exist_ok=True)
        intake.write_exclusive(path, value)
        if effective_execution_expiry(directory, intent) != expires_at_epoch:
            _refuse('extension_not_effective')
    return {'status': 'execution_window_extended', 'previous_expires_at_epoch': current,
            'record_path': str(path), 'extension': value, 'provider_mutation_performed': False}


# Preserve the original import surface while validation lives in pure readers.
__all__ = [
    'ACK',
    'DIRECTORY',
    'SCHEMA',
    '_bounds',
    'effective_execution_expiry',
    'extend_scene_execution_window',
    'extend_scene_execution_window_by_operator',
]
