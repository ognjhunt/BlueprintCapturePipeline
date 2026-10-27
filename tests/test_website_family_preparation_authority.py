"""Typed family preparation keeps the existing signed unconfirmed-task scope."""
from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_preparation_contracts import validate_preparation_request
from tests.test_website_preparation_contract_v2 import VECTOR, NOW


def family_authority_vector(declared='general_task'):
    task = copy.deepcopy(VECTOR['task_context_response_preparation'])
    task['task_family'] = declared
    task['context_digest'] = canonical_digest(task, digest_field='context_digest')
    authority = copy.deepcopy(VECTOR['sponsorship_response'])
    authority['task_context_digest'] = task['context_digest']
    authority['authority_digest'] = canonical_digest(authority, digest_field='authority_digest')
    request = copy.deepcopy(VECTOR['prepared_scene_body']['request'])
    request['source']['binding_id'] = 'website-splat-' + request['source']['content_digest'][7:39]
    request['task'] = {'task_id': 'website-' + task['context_digest'][7:27], 'strategy': 'button_press',
        'task_success_contract_digest': 'sha256:' + 'a' * 64, 'native_runtime_contract_digest': 'sha256:' + 'b' * 64,
        'task_family_runtime_selection': {'runtime_id': 'qualified-button', 'runtime_catalog_digest': 'sha256:' + 'c' * 64}}
    return {'request': request, 'task_context': task, 'authority': authority, 'now': NOW}


@pytest.mark.parametrize('declared', ['button_press', 'general_task'])
def test_unconfirmed_family_has_typed_preparation_authority_without_rigid_subject(declared):
    args = family_authority_vector(declared)
    original = copy.deepcopy(args)
    assert validate_preparation_request(**args) == args['authority']
    assert args == original
    assert args['request']['consent']['task_confirmed'] is False
    assert args['request']['execution']['policy_candidates'] == [] and 'subject' not in args['request']['task']


@pytest.mark.parametrize('fault', ['task_id', 'owner', 'rights', 'source', 'provider', 'budget', 'retry',
    'purpose', 'policy', 'runtime', 'extra_task', 'unknown_strategy', 'family_alias'])
def test_family_preparation_preserves_authority_scope_and_exact_task_shape(fault):
    args = family_authority_vector()
    request = args['request']
    if fault == 'task_id':
        request['task']['task_id'] = 'different-task'
    elif fault == 'owner':
        request['owner']['user_id'] = 'different-owner'
    elif fault == 'rights':
        request['consent']['rights_reference'] = 'different-rights'
    elif fault == 'source':
        request['source']['content_digest'] = 'sha256:' + 'f' * 64
    elif fault == 'provider':
        request['execution']['allowed_providers'] = ['vast']
    elif fault == 'budget':
        request['execution']['max_total_spend_usd'] += 1
    elif fault == 'retry':
        request['execution']['max_retries'] = 1
    elif fault == 'purpose':
        request['execution'].pop('purpose')
    elif fault == 'policy':
        request['execution']['policy_candidates'] = [{'id': 'candidate'}]
    elif fault == 'runtime':
        request['task']['task_family_runtime_selection']['runtime_catalog_digest'] = 'unsealed'
    elif fault == 'extra_task':
        request['task']['subject'] = {'geometry_origin': 'removed_before_reconstruction'}
    else:
        request['task']['strategy'] = 'mining' if fault == 'unknown_strategy' else 'buttons'
    with pytest.raises(ValueError, match='website_preparation_request_authority_invalid'):
        validate_preparation_request(**args)
