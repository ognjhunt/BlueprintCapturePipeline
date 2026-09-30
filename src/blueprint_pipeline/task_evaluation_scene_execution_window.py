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
    SCHEMA,
    _bounds,
    effective_execution_expiry,
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


# Preserve the original import surface while validation lives in pure readers.
__all__ = [
    'ACK',
    'DIRECTORY',
    'SCHEMA',
    '_bounds',
    'effective_execution_expiry',
    'extend_scene_execution_window',
]
