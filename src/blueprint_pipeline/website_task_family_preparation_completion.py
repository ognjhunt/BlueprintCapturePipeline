"""Complete capture-bound CPU family preparation without claiming native physics."""
from __future__ import annotations

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_attempt_evidence import record
from .task_evaluation_scene_configuration_submission_inputs import read
from .task_evaluation_scene_progression_contracts import require, safe_path
from .task_family_dispatch_contracts import validate_family_preparation_result_reference


def complete_family_preparation(*, factory, plan, intent, attempt, output_root):
    """Retain exact preparation evidence; policy admission remains independent."""
    task = intent['request']['task']
    reference = {'schema_version': 'website_task_family_preparation_result.v1',
        'digest': plan['plan_digest'], 'task_family': plan['task_family'],
        'task_success_contract_digest': task['task_success_contract_digest'],
        'native_runtime_contract_digest': task['native_runtime_contract_digest'],
        'task_family_runtime_selection': task['task_family_runtime_selection'],
        'native_application_claimed': False, 'policy_episode_claimed': False,
        'pre_policy_site_qualification_required': True}
    result = {**reference, 'scene_id': plan['scene_id'], 'task_id': plan['task_id'],
        'scene_revision_digest': plan['plan_digest'], 'intent_id': intent['intent_id'],
        'intent_digest': intent['intent_digest'], 'attempt_digest': attempt['attempt_digest'],
        'source_commit': factory['source_commit'], 'claim_ceiling': 'development_only',
        'references': {key: factory[key] for key in (
            'packet_receipt', 'scene_plan', 'native_runtime_contract', 'family_scene_inputs')},
        'execution_authority_granted': False, 'provider_mutation_performed': False,
        'robot_evaluation_performed': False, 'result_digest': ''}
    result['result_digest'] = canonical_digest(result, digest_field='result_digest')
    path = safe_path(output_root) / 'family_native_preparation_result.v1.json'
    if path.exists():
        require(read(path, digest_field='result_digest') == result, 'family_preparation_completion_changed')
    else:
        write_json(path, result)
    value = {**record(path), **reference}
    validate_family_preparation_result_reference(value, request=intent['request'],
        status='completed', phase='family_native_prepared')
    return value
