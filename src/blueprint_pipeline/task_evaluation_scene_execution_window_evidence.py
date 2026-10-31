"""Read-only validation of retained scene execution-window extensions.

Two record variants share one directory and one schema. The original is issued
by the intent's own authenticated client (the WebApp). The operator variant is
written through the operator door after the owner approves a time-only
renewal: it names the door caller and the owner's approval and never claims
to be the WebApp. Both bind the exact sealed intent and every execution bound
except the expiry, so neither can change spend, attempts, providers, task or
rights.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = 'task_evaluation_scene_execution_window_extension.v1'
DIRECTORY = 'execution-window-extensions'
ACK = 'extend-scene-execution-window'
OPERATOR_DOOR = 'operator_door'
MAXIMUM_EXTENSION_SECONDS = 7 * 86400
MAXIMUM_REFERENCE_CHARACTERS = 500
_DOOR_CALLER = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z')
_DOOR_REQUEST = re.compile(r'[0-9]{8}T[0-9]{6}Z-extend-scene-execution-window-[0-9a-f]{8}\Z')
_DIGEST = re.compile(r'sha256:[0-9a-f]{64}\Z')
OPERATOR_FIELDS = frozenset({
    'schema_version', 'scope', 'authorized_via', 'door_caller', 'door_request_id', 'intent_id',
    'intent_digest', 'owner', 'intent_authenticated_issuer', 'authorization_reference',
    'original_expires_at_epoch', 'expires_at_epoch', 'issued_at_epoch', 'unchanged_execution_bounds',
    'provider_mutation_performed', 'extension_digest',
})


def _bounds(intent: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in intent['request']['execution'].items() if k != 'expires_at_epoch'}


def operator_bounds(intent: Mapping[str, Any]) -> dict[str, Any]:
    """Everything an operator extension must leave exactly as the owner sealed it."""
    return {'execution': _bounds(intent), 'task_content_digest': intent.get('task_content_digest'),
            'source_content_digest': intent.get('source_content_digest'),
            'consent': intent['request']['consent']}


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def valid_authorization_reference(value: Any) -> bool:
    return (isinstance(value, str) and value == value.strip() and 0 < len(value) <= MAXIMUM_REFERENCE_CHARACTERS
            and value.isprintable())


def operator_extension_valid(value: Mapping[str, Any], *, intent: Mapping[str, Any], file_name: str) -> bool:
    """As strict as the WebApp-issued variant, plus a closed field set and the raw+7d cap."""
    execution = intent['request']['execution']
    issued, expires = value.get('issued_at_epoch'), value.get('expires_at_epoch')
    original = value.get('original_expires_at_epoch')
    caller, request_id = value.get('door_caller'), value.get('door_request_id')
    return bool(
        set(value) == OPERATOR_FIELDS
        and value['schema_version'] == SCHEMA and value['scope'] == 'execution_time_only'
        and value['authorized_via'] == OPERATOR_DOOR
        and isinstance(caller, str) and _DOOR_CALLER.fullmatch(caller) is not None
        and isinstance(request_id, str) and _DOOR_REQUEST.fullmatch(request_id) is not None
        and value['intent_id'] == intent['intent_id'] and value['intent_digest'] == intent['intent_digest']
        and value['owner'] == intent['request']['owner']
        and isinstance(intent.get('authenticated_issuer'), str) and bool(intent['authenticated_issuer'].strip())
        and value['intent_authenticated_issuer'] == intent['authenticated_issuer']
        and value['unchanged_execution_bounds'] == operator_bounds(intent)
        and _finite(original) and _finite(execution.get('expires_at_epoch'))
        and original == execution['expires_at_epoch']
        and valid_authorization_reference(value['authorization_reference'])
        and _finite(issued) and _finite(expires)
        and issued < expires <= issued + MAXIMUM_EXTENSION_SECONDS
        and original < expires <= original + MAXIMUM_EXTENSION_SECONDS
        and value['provider_mutation_performed'] is False
        and isinstance(value['extension_digest'], str) and _DIGEST.fullmatch(value['extension_digest']) is not None
        and file_name == value['extension_digest'].removeprefix('sha256:') + '.json'
    )


def _issuer_extension_valid(value: Mapping[str, Any], path: Path, intent: Mapping[str, Any]) -> bool:
    from . import task_evaluation_scene_intent_contracts as intake
    issued, expires = value.get('issued_at_epoch'), value.get('expires_at_epoch')
    return not (
        value.get('schema_version') != SCHEMA or value.get('scope') != 'execution_time_only'
        or value.get('intent_id') != intent['intent_id']
        or value.get('intent_digest') != intent['intent_digest']
        or value.get('owner') != intent['request']['owner']
        or value.get('authenticated_issuer') != intent['authenticated_issuer']
        or value.get('unchanged_execution_bounds') != _bounds(intent)
        or value.get('original_expires_at_epoch') != intent['request']['execution']['expires_at_epoch']
        or not isinstance(value.get('authorization_reference'), str)
        or not value['authorization_reference'].strip()
        or not intake._number(issued) or not intake._number(expires)
        or not issued < expires <= issued + 7*86400
        or expires <= value['original_expires_at_epoch']
        or value.get('provider_mutation_performed') is not False
        or path.name != value['extension_digest'].removeprefix('sha256:')+'.json'
    )


def verified_extensions(directory: Path, intent: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every retained extension, each validated; one invalid record fails the whole window closed."""
    from . import task_evaluation_scene_intent_contracts as intake
    root = directory / DIRECTORY
    if root.is_symlink():
        raise ValueError('scene_execution_window_store_unsafe')
    values = []
    for path in sorted(root.glob('*.json')):
        value = intake._read(path, 'extension_digest')
        valid = (operator_extension_valid(value, intent=intent, file_name=path.name)
                 if 'authorized_via' in value else _issuer_extension_valid(value, path, intent))
        if not valid:
            raise ValueError('scene_execution_window_extension_invalid')
        values.append(value)
    return values


def effective_execution_expiry(directory: Path, intent: Mapping[str, Any]) -> float:
    expiry = float(intent['request']['execution']['expires_at_epoch'])
    for value in verified_extensions(directory, intent):
        expiry = max(expiry, float(value['expires_at_epoch']))
    return expiry
