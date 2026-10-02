# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_finished.py
"""Pure authority-end metadata observations never confer cleanup permission."""
import json

import pytest

from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
from tests.test_scene_inventory_history import fixture, seal


def data():
    args = fixture()
    intent = json.loads(args['records']['intent'][1])
    intent['request']['execution'] = {'expires_at_epoch': 100, 'maximum_spend_usd': 1}
    intent['authenticated_issuer'] = 'trusted-client'
    intent = seal(intent, 'intent_digest', cross=True)
    owner = args['roots']['intent_root']+'/'+args['intent_id']
    rows = [{'role': 'intent', 'path': owner+'/intent.json', 'value': intent}]
    return args, intent, owner, rows


def observe(rows, args, now):
    from blueprint_pipeline.task_evaluation_scene_lifecycle_finished import finished
    return finished({'chain_validated': True, 'projection_state': 'current'}, rows, args['roots'],
                    args['intent_id'], now, ReferenceCollectionBudget(monotonic=lambda: 0), extensions_observed=True)


def test_revocation_exact_time_grace_and_never_mtime_fallback():
    args, intent, owner, rows = data()
    value = seal({'schema_version': 'task_evaluation_scene_intent_revocation.v1', 'intent_id': args['intent_id'],
        'intent_digest': intent['intent_digest'], 'owner': intent['request']['owner'], 'status': 'revoked',
        'scope': 'future_execution', 'revoked_at_epoch': 20, 'provider_mutation_performed': False}, 'receipt_digest')
    rows.append({'role': 'revocations', 'path': owner+'/revoked.json', 'value': value})
    assert observe(rows, args, 20+7*86400-1)['status'] == 'unknown'
    assert observe(rows, args, 20+7*86400)['status'] == 'revoked_grace_elapsed'
    value.pop('revoked_at_epoch')
    rows[-1]['value'] = seal(value, 'receipt_digest')
    with pytest.raises(ValueError):
        observe(rows, args, 900000)


def test_all_sealed_extensions_max_expiry_and_unknown_scope():
    args, intent, owner, rows = data()
    for issued, expires in ((100, 200), (200, 300)):
        value = seal({'schema_version': 'task_evaluation_scene_execution_window_extension.v1',
            'scope': 'execution_time_only', 'intent_id': args['intent_id'], 'intent_digest': intent['intent_digest'],
            'owner': intent['request']['owner'], 'authenticated_issuer': intent['authenticated_issuer'],
            'authorization_reference': 'owner-reviewed', 'original_expires_at_epoch': 100,
            'issued_at_epoch': issued, 'expires_at_epoch': expires, 'unchanged_execution_bounds': {'maximum_spend_usd': 1},
            'provider_mutation_performed': False}, 'extension_digest')
        rows.append({'role': 'extensions', 'path': owner+'/execution-window-extensions/'+value['extension_digest'][7:]+'.json', 'value': value})
    assert observe(rows, args, 300+7*86400-1)['status'] == 'unknown'
    assert observe(rows, args, 300+7*86400)['status'] == 'expired_grace_elapsed'
    from blueprint_pipeline.task_evaluation_scene_lifecycle_finished import finished
    assert finished({'chain_validated': True, 'projection_state': 'current'}, rows, args['roots'], args['intent_id'],
                    900000, ReferenceCollectionBudget(monotonic=lambda: 0), extensions_observed=False)['status'] == 'unknown'


@pytest.mark.parametrize('change', [{'expires_at_epoch': True}, {'unchanged_execution_bounds': {'maximum_spend_usd': 2}},
                                     {'owner': {'user_id': 'foreign'}}, {'provider_mutation_performed': True}])
def test_resealed_available_extension_contradictions_refuse(change):
    args, intent, owner, rows = data()
    value = {'schema_version': 'task_evaluation_scene_execution_window_extension.v1', 'scope': 'execution_time_only',
        'intent_id': args['intent_id'], 'intent_digest': intent['intent_digest'], 'owner': intent['request']['owner'],
        'authenticated_issuer': intent['authenticated_issuer'], 'authorization_reference': 'owner-reviewed',
        'original_expires_at_epoch': 100, 'issued_at_epoch': 100, 'expires_at_epoch': 200,
        'unchanged_execution_bounds': {'maximum_spend_usd': 1}, 'provider_mutation_performed': False, **change}
    value = seal(value, 'extension_digest')
    rows.append({'role': 'extensions', 'path': owner+'/execution-window-extensions/'+value['extension_digest'][7:]+'.json', 'value': value})
    with pytest.raises(ValueError):
        observe(rows, args, 900000)


def test_extension_missing_authenticated_issuer_never_equals_missing_owner_issuer():
    args, intent, owner, rows = data()
    intent.pop('authenticated_issuer')
    intent = seal(intent, 'intent_digest', cross=True)
    rows[0]['value'] = intent
    value = seal({'schema_version': 'task_evaluation_scene_execution_window_extension.v1', 'scope': 'execution_time_only',
        'intent_id': args['intent_id'], 'intent_digest': intent['intent_digest'], 'owner': intent['request']['owner'],
        'authorization_reference': 'reviewed', 'original_expires_at_epoch': 100, 'issued_at_epoch': 100,
        'expires_at_epoch': 200, 'unchanged_execution_bounds': {'maximum_spend_usd': 1},
        'provider_mutation_performed': False}, 'extension_digest')
    rows.append({'role': 'extensions', 'path': owner+'/execution-window-extensions/'+value['extension_digest'][7:]+'.json',
                 'value': value})
    with pytest.raises(ValueError):
        observe(rows, args, 900000)
