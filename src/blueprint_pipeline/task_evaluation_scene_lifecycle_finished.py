"""Pure retained authority-end metadata; no runtime admission or cleanup right."""
from __future__ import annotations

import math
from pathlib import PurePosixPath

from .task_evaluation_scene_lifecycle_acquisition import require
from .task_evaluation_scene_lineage_budget import _work_items
from .task_evaluation_scene_preparation_lineage import _seal

GRACE = 7 * 86400


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def finished(history, decoded, roots, intent_id, now, budget, *, extensions_observed):
    owner = roots['intent_root'] + '/' + intent_id
    indexed = {}
    for row in _work_items(decoded, budget):
        budget.charge('facts')
        indexed.setdefault(row['role'], []).append(row)
    intents = [row for row in _work_items(indexed.get('intent', []), budget) if row['path'] == owner+'/intent.json']
    require(len(intents) == 1, 'intent_unavailable')
    intent = intents[0]['value']
    execution = intent['request'].get('execution')
    expiry = execution.get('expires_at_epoch') if isinstance(execution, dict) else None
    bounds = {key: value for key, value in _work_items(execution.items(), budget) if key != 'expires_at_epoch'} if isinstance(execution, dict) else None
    unknown, revoked = False, None
    for row in _work_items(indexed.get('revocations', []), budget):
        value = row['value']
        if value.get('schema_version') != 'task_evaluation_scene_intent_revocation.v1':
            unknown = True
            continue
        _seal(value, {}, 'receipt_digest', work_budget=budget)
        require(row['path'] == owner+'/revoked.json' and value.get('intent_id') == intent_id
                and value.get('intent_digest') == intent['intent_digest'] and value.get('owner') == intent['request']['owner']
                and value.get('status') == 'revoked' and value.get('scope') == 'future_execution'
                and value.get('provider_mutation_performed') is False and finite(value.get('revoked_at_epoch')),
                'revocation_invalid')
        revoked = value['revoked_at_epoch']
    for row in _work_items(indexed.get('extensions', []), budget):
        value = row['value']
        if value.get('schema_version') != 'task_evaluation_scene_execution_window_extension.v1':
            unknown = True
            continue
        _seal(value, {}, 'extension_digest', work_budget=budget)
        issued, expires = value.get('issued_at_epoch'), value.get('expires_at_epoch')
        require(finite(expiry) and finite(issued) and finite(expires)
                and str(PurePosixPath(row['path']).parent) == owner+'/execution-window-extensions'
                and PurePosixPath(row['path']).name == value['extension_digest'][7:]+'.json'
                and value.get('scope') == 'execution_time_only' and value.get('intent_id') == intent_id
                and value.get('intent_digest') == intent['intent_digest'] and value.get('owner') == intent['request']['owner']
                and value.get('authenticated_issuer') == intent.get('authenticated_issuer')
                and type(value.get('original_expires_at_epoch')) in (int, float)
                and value['original_expires_at_epoch'] == execution['expires_at_epoch']
                and value.get('unchanged_execution_bounds') == bounds
                and isinstance(value.get('authorization_reference'), str) and bool(value['authorization_reference'].strip())
                and issued < expires <= issued+GRACE and expires > value['original_expires_at_epoch']
                and value.get('provider_mutation_performed') is False, 'extension_invalid')
        expiry = max(expiry, expires)
    current = history['chain_validated'] and history['projection_state'] == 'current'
    projection = next((row['value'] for row in _work_items(indexed.get('projection', []), budget)
                       if row['path'] == owner+'/progression.json'), None)
    status, reason = 'unknown', 'history_or_authority_end_unproven'
    if current and projection is not None and projection.get('status') == 'completed':
        status, reason = 'completed', 'validated_current_completed_history'
    elif current and not unknown and revoked is not None and now >= revoked+GRACE:
        status, reason = 'revoked_grace_elapsed', 'sealed_revocation_grace_elapsed'
    elif current and not unknown and revoked is None and extensions_observed and finite(expiry) and now >= expiry+GRACE:
        status, reason = 'expired_grace_elapsed', 'all_observed_extensions_and_grace_elapsed'
    budget.charge('facts')
    return {'status': status, 'reason': reason, 'observed_at_epoch': now,
            'effective_expires_at_epoch': expiry if finite(expiry) else None,
            'authority_metadata_unknown': unknown or not extensions_observed,
            'finished_for_cleanup_authority': False, 'intent_digest': intent['intent_digest']}
