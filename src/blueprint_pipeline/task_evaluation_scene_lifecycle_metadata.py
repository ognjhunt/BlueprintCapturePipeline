"""Source-defined retained metadata observations; no effective service proof."""
from __future__ import annotations

from .task_evaluation_scene_lifecycle_acquisition import require
from .task_evaluation_scene_lineage_budget import _work_items
from .task_evaluation_scene_preparation_lineage import _seal

CONFIG_ROOTS = {
    'intent_root': 'intent_root', 'factory_output_root': 'factory_output_root',
    'capture_store_root': 'pubsub_root', 'preparation_queue_root': 'preparation_queue_root',
    'child_queue_root': 'sam_queue_root', 'child_execution_root': 'sam_execution_root',
    'launch_execution_root': 'launch_execution_root', 'terminal_result_root': 'terminal_result_root',
}


def configuration(decoded, context, budget):
    selected = []
    unknown = False
    for row in _work_items(decoded, budget):
        if row['role'] != 'progression_config':
            continue
        budget.charge('facts')
        value = row['value']
        proof = {'role': row['role'], 'path': row['path'], 'sha256': row['sha256'],
                 'size_bytes': len(row['raw']), 'seal_field': None, 'seal_digest': None}
        selected.append(proof)
        if value.get('schema_version') != 'task_evaluation_scene_progression_config.v1':
            unknown = True
            continue
        _seal(value, proof, 'config_digest', work_budget=budget)
        for field, root in _work_items(CONFIG_ROOTS.items(), budget):
            if field not in value:
                unknown = True
            else:
                require(value[field] == context['roots'][root], 'configuration_root_mismatch')
        worker = value.get('preparation_worker')
        if not isinstance(worker, dict) or 'input_root' not in worker:
            unknown = True
        else:
            require(worker['input_root'] == context['roots']['preparation_input_root'], 'configuration_root_mismatch')
    return {'status': 'retained_configuration_roots_matched' if len(selected) == 1 and not unknown
            else 'configuration_unproven', 'source_provenance': selected,
            'running_configuration_verified': False, 'service_fences_checked': False,
            'action': 'KEEP'}
