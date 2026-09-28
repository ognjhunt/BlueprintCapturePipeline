"""Owner-directed scope reduction; never changes sealed inputs or releases spend."""
import time
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_execution_scope import scene_preparation_only

FILE = 'scene-preparation-only.json'
SCHEMA = 'task_evaluation_scene_scope_restriction.v1'
ACK = 'restrict-existing-intent-to-scene-preparation'


def preparation_only(*, directory, intent):
    from . import task_evaluation_scene_intake as intake
    from .task_evaluation_scene_execution_budget import _safe
    path = Path(directory) / FILE
    _safe(path)
    if path.exists():
        value = intake._read(path, 'restriction_digest')
        intake._require(value.get('schema_version') == SCHEMA
            and value.get('purpose') == 'scene_preparation'
            and value.get('intent_id') == intent['intent_id']
            and value.get('intent_digest') == intent['intent_digest']
            and value.get('owner') == intent['request']['owner']
            and value.get('authenticated_issuer') == intent['authenticated_issuer']
            and value.get('historical_reservations_released') is False,
            'scope_restriction_invalid')
        return True
    return scene_preparation_only(intent['request'])


def restrict_to_scene_preparation(*, queue_root, intent_id, intent_digest, owner,
        authenticated_client, trusted_clients, authorization_reference, ack, now=None):
    from . import task_evaluation_scene_intake as intake
    from .task_evaluation_scene_execution_budget import _safe
    intake._require(ack == ACK and authenticated_client in trusted_clients
        and intake._identifier(intent_id) and isinstance(authorization_reference, str)
        and bool(authorization_reference.strip()), 'scope_restriction_not_authorized')
    moment = time.time() if now is None else now
    intake._require(intake._number(moment), 'scope_restriction_time_invalid')
    root = intake._root(Path(queue_root))
    _safe(root)
    with intake._lock(root):
        directory = root / intent_id
        _safe(directory)
        intent = intake._read(directory / 'intent.json', 'intent_digest')
        intake._require(intent['intent_digest'] == intent_digest
            and intent['request']['owner'] == dict(owner)
            and intent['authenticated_issuer'] == authenticated_client,
            'scope_restriction_owner_mismatch')
        if preparation_only(directory=directory, intent=intent):
            return {'status': 'already_restricted', 'provider_mutation_performed': False}
        value = {'schema_version': SCHEMA, 'purpose': 'scene_preparation',
            'intent_id': intent_id, 'intent_digest': intent_digest, 'owner': dict(owner),
            'authenticated_issuer': authenticated_client,
            'authorization_reference': authorization_reference,
            'issued_at_epoch': moment,
            'historical_reservations_released': False, 'provider_mutation_performed': False}
        value['restriction_digest'] = canonical_digest(value, digest_field='restriction_digest')
        path = directory / FILE
        intake.write_exclusive(path, value)
        preparation_only(directory=directory, intent=intent)
    return {'status': 'restricted_to_scene_preparation', 'record_path': str(path), **value}
