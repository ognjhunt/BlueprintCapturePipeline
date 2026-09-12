"""Write-once operational robot choice for an owner task that omitted one.

The original task and its completed source evidence stay immutable. Only the
controls producer consumes this choice; existing explicit choices always win.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest

SCHEMA = 'task_evaluation_scene_robot_assignment.v1'
AUTHORIZATION_SCHEMA = 'task_evaluation_scene_robot_assignment_authorization.v1'
FILENAME = 'robot-assignment.json'
ACK = 'assign-scene-robot'


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError('scene_robot_assignment_' + code)


def _safe(path: Path) -> None:
    _require(path.is_absolute() and '..' not in path.parts
             and not any(p.is_symlink() for p in (path, *path.parents)), 'path_unsafe')


def catalog_binding_digest(binding: Mapping[str, Any]) -> str:
    return canonical_digest({k: v for k, v in binding.items() if k != 'expected_production_commit'})


def _private_authorization(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _safe(path)
    _require(path.is_file() and not path.stat().st_mode & 0o027, 'authorization_file_unsafe')
    raw = path.read_bytes()
    value = json.loads(raw)
    _require(isinstance(value, dict), 'authorization_invalid')
    return value, {'path': str(path), 'sha256': 'sha256:' + hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def _authorization_matches(value, *, intent, binding_id, binding_digest):
    _require(value.get('schema_version') == AUTHORIZATION_SCHEMA and value.get('authorized') is True
        and value.get('intent_id') == intent['intent_id'] and value.get('intent_digest') == intent['intent_digest']
        and value.get('owner') == intent['request']['owner']
        and value.get('authenticated_issuer') == intent['authenticated_issuer']
        and value.get('robot_binding_id') == binding_id and value.get('catalog_binding_digest') == binding_digest,
        'authorization_scope_mismatch')


def _binding(catalog, binding_id):
    from .task_evaluation_controls_autoprovision import CATALOG_SCHEMA, CONTENT_CATALOG_SCHEMA
    from .task_evaluation_scene_intake import _identifier
    _require(catalog.get('schema_version') in {CATALOG_SCHEMA, CONTENT_CATALOG_SCHEMA}
        and catalog.get('catalog_digest') == canonical_digest(catalog, digest_field='catalog_digest'), 'catalog_invalid')
    binding = catalog.get('bindings', {}).get(binding_id)
    _require(_identifier(binding_id) and isinstance(binding, dict), 'binding_missing')
    return binding


def _owner(directory, intent, now):
    from . import task_evaluation_scene_intake as intake
    _safe(directory)
    _safe(directory / 'revoked.json')
    _require(directory.name == intent['intent_id'] and not (directory / 'revoked.json').exists(), 'owner_revoked')
    _require(intake._number(now) and intent['accepted_at_epoch'] <= now
             < intake.effective_execution_expiry(directory, intent), 'owner_expired')


def read_scene_robot_assignment(*, directory: Path, intent: Mapping[str, Any],
                                catalog: Mapping[str, Any], now: float | None = None) -> dict[str, Any]:
    from . import task_evaluation_scene_intake as intake
    moment = time.time() if now is None else now
    _owner(directory, intent, moment)
    _require('robot_binding_id' not in intent['request']['task'], 'explicit_choice_present')
    path = directory / FILENAME
    _safe(path)
    _require(path.is_file(), 'missing')
    row = intake._read(path, 'assignment_digest')
    _require(set(row) == {'schema_version', 'scope', 'intent_id', 'intent_digest', 'owner',
        'authenticated_issuer', 'robot_binding_id', 'catalog_binding_digest', 'authorization_reference',
        'issued_at_epoch', 'original_task_digest', 'provider_mutation_performed', 'original_task_modified',
        'assignment_digest'}, 'record_invalid')
    binding = _binding(catalog, row.get('robot_binding_id'))
    _require(row['schema_version'] == SCHEMA and row['scope'] == 'omitted_controls_robot_only'
        and row['intent_id'] == intent['intent_id'] and row['intent_digest'] == intent['intent_digest']
        and row['owner'] == intent['request']['owner'] and row['authenticated_issuer'] == intent['authenticated_issuer']
        and row['original_task_digest'] == intake.canonical_digest(intent['request']['task'])
        and row['catalog_binding_digest'] == catalog_binding_digest(binding)
        and intake._number(row['issued_at_epoch']) and intent['accepted_at_epoch'] <= row['issued_at_epoch'] <= moment
        and row['provider_mutation_performed'] is False and row['original_task_modified'] is False, 'record_invalid')
    ref = row['authorization_reference']
    _require(isinstance(ref, dict) and set(ref) == {'path', 'sha256', 'size_bytes'}, 'authorization_reference_invalid')
    authorization, observed = _private_authorization(Path(ref['path']))
    _require(observed == ref, 'authorization_changed')
    _authorization_matches(authorization, intent=intent, binding_id=row['robot_binding_id'],
                           binding_digest=row['catalog_binding_digest'])
    return row


def resolve_controls_robot_binding(*, directory: Path, intent: Mapping[str, Any],
                                   catalog: Mapping[str, Any], now: float | None = None):
    """An explicit owner choice never falls back to a default or assignment."""
    if 'robot_binding_id' in intent['request']['task']:
        return _binding(catalog, intent['request']['task']['robot_binding_id']), None
    row = read_scene_robot_assignment(directory=directory, intent=intent, catalog=catalog, now=now)
    return _binding(catalog, row['robot_binding_id']), row


def assign_scene_robot(*, queue_root: str | Path, intent_id: str, intent_digest: str,
        owner: Mapping[str, Any], authenticated_client: str, trusted_clients: set[str],
        robot_catalog_path: str | Path, robot_binding_id: str, authorization_reference: str | Path,
        ack: str, now: float | None = None) -> dict[str, Any]:
    from . import task_evaluation_scene_intake as intake
    from .task_evaluation_controls_autoprovision import _asset, _sealed, payload_digest
    moment = time.time() if now is None else now
    _require(ack == ACK and authenticated_client in trusted_clients and intake._identifier(intent_id)
             and intake._number(moment), 'not_authorized')
    root, catalog_path = Path(queue_root), Path(robot_catalog_path)
    _safe(root)
    _safe(catalog_path)
    catalog = _sealed(catalog_path, 'catalog_digest')
    binding = _binding(catalog, robot_binding_id)
    # Validate actual admitted robot/camera/runtime bytes, not merely the catalog hash.
    _asset(binding['robot_asset_usd'])
    _asset(binding['embodiment_camera_template'])
    _require(payload_digest(Path(binding['runtime_source_payload_dir'])) == binding['runtime_digest'], 'runtime_changed')
    cap = binding.get('phase_hard_cap_usd', 2.0)
    _require(intake._number(cap) and 0 < cap <= 50
        and all(isinstance(binding.get(k), str) and binding[k].strip() for k in ('openai_project_id', 'openai_api_key_id')),
        'binding_scope_invalid')
    authorization, reference = _private_authorization(Path(authorization_reference))
    with intake._lock(root):
        directory = root / intent_id
        _safe(directory)
        intent = intake._read(directory / 'intent.json', 'intent_digest')
        _require(intent['intent_digest'] == intent_digest and intent['request']['owner'] == dict(owner)
                 and intent['authenticated_issuer'] == authenticated_client, 'owner_mismatch')
        _owner(directory, intent, moment)
        _require('robot_binding_id' not in intent['request']['task'], 'explicit_choice_present')
        binding_digest = catalog_binding_digest(binding)
        _authorization_matches(authorization, intent=intent, binding_id=robot_binding_id, binding_digest=binding_digest)
        path = directory / FILENAME
        _safe(path)
        if path.exists():
            row = read_scene_robot_assignment(directory=directory, intent=intent, catalog=catalog, now=moment)
            _require(row['robot_binding_id'] == robot_binding_id and row['authorization_reference'] == reference, 'immutable_conflict')
            return {'status': 'robot_assignment_already_recorded', 'record_path': str(path), **row}
        _require(intent['accepted_at_epoch'] <= moment, 'issued_time_invalid')
        row = intake._seal({'schema_version': SCHEMA, 'scope': 'omitted_controls_robot_only',
            'intent_id': intent_id, 'intent_digest': intent_digest, 'owner': dict(owner),
            'authenticated_issuer': authenticated_client, 'robot_binding_id': robot_binding_id,
            'catalog_binding_digest': binding_digest, 'authorization_reference': reference,
            'issued_at_epoch': moment, 'original_task_digest': intake.canonical_digest(intent['request']['task']),
            'provider_mutation_performed': False, 'original_task_modified': False}, 'assignment_digest')
        intake.write_exclusive(path, row)
        read_scene_robot_assignment(directory=directory, intent=intent, catalog=catalog, now=moment)
    return {'status': 'robot_assignment_recorded', 'record_path': str(path), **row}
