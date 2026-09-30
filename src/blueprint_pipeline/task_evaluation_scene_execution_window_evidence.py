"""Read-only validation of retained scene execution-window extensions."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = 'task_evaluation_scene_execution_window_extension.v1'
DIRECTORY = 'execution-window-extensions'
ACK = 'extend-scene-execution-window'


def _bounds(intent: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in intent['request']['execution'].items() if k != 'expires_at_epoch'}


def effective_execution_expiry(directory: Path, intent: Mapping[str, Any]) -> float:
    from . import task_evaluation_scene_intent_contracts as intake
    expiry = float(intent['request']['execution']['expires_at_epoch'])
    root = directory / DIRECTORY
    if root.is_symlink():
        raise ValueError('scene_execution_window_store_unsafe')
    for path in sorted(root.glob('*.json')):
        value = intake._read(path, 'extension_digest')
        issued, expires = value.get('issued_at_epoch'), value.get('expires_at_epoch')
        if (
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
        ):
            raise ValueError('scene_execution_window_extension_invalid')
        expiry = max(expiry, float(expires))
    return expiry
