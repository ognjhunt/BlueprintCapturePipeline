"""Source-defined retained metadata observations; no effective service proof."""
from __future__ import annotations

from pathlib import PurePosixPath

from .task_evaluation_scene_lifecycle_acquisition import require
from .task_evaluation_scene_lineage_budget import _work_items, _work_call
from .task_evaluation_scene_preparation_lineage import _seal, _intent
from .task_evaluation_scene_inventory_seed import _raw_reference
from .decision_evidence_contracts import cross_runtime_canonical_digest

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



def _proof(row):
    return {'role': row['role'], 'path': row['path'], 'sha256': row['sha256'],
            'size_bytes': len(row['raw']), 'seal_field': None, 'seal_digest': None}


def other_capture_references(decoded, context, selected_id, budget, sink):
    """A retained other-owner declaration protects; it proves no live consumer."""
    registrations, intents = {}, []
    for row in _work_items(decoded, budget):
        value = row['value']
        if row['role'] == 'website_registrations' and value.get('schema_version') == 'website_scene_source_registration.v1':
            proof = _proof(row)
            _seal(value, proof, 'registration_digest', work_budget=budget)
            request_digest = value.get('request_digest')
            require(isinstance(request_digest, str) and len(request_digest) == 71
                    and request_digest.startswith('sha256:')
                    and all(c in '0123456789abcdef' for c in request_digest[7:]),
                    'registration_reference_invalid')
            require(row['path'] == context['roots']['website_source_binding_root']+'/'+request_digest[7:]+'.json',
                    'registration_reference_invalid')
            budget.charge('facts')
            registrations.setdefault(request_digest, []).append((value, proof))
        elif row['role'] == 'other_owner_intents' and value.get('schema_version') == 'task_evaluation_scene_intent.v1':
            budget.charge('facts')
            intents.append(row)
    result, seen = sink.rows(), set()
    for row in _work_items(intents, budget):
        value, proof = row['value'], _proof(row)
        owner_id = value.get('intent_id')
        if owner_id == selected_id:
            continue
        _intent(value, proof, owner_id, context['roots']['intent_root'], work_budget=budget)
        request_digest = _work_call(budget, cross_runtime_canonical_digest, value['request'])
        for registration, reg_proof in _work_items(registrations.get(request_digest, ()), budget):
            references = registration.get('references')
            require(isinstance(references, dict) and bool(references), 'registration_reference_invalid')
            for reference in _work_items(references.values(), budget):
                path, _, _ = _raw_reference(reference, work_budget=budget)
                p, root = PurePosixPath(path), PurePosixPath(context['roots']['pubsub_root'])
                if not p.is_relative_to(root):
                    continue
                parts = p.relative_to(root).parts
                allowed = {'preparation.json', 'task_context.json', 'native/runtime_inputs.json',
                           'development_test/preparation.json', 'development_test/task_context.json',
                           'development_test/runtime_inputs.json'}
                if (len(parts) < 8 or parts[1] != 'scenes' or parts[3] != 'captures'
                        or parts[5:7] != ('pipeline', 'website_scene_preparation')
                        or '/'.join(parts[7:]) not in allowed):
                    continue
                capture = str(root.joinpath(*parts[:5]))
                key = capture, proof['path'], reg_proof['path']
                if key in seen:
                    continue
                sink.available_occurrence()
                budget.charge('facts')
                seen.add(key)
                result.append({'kind': 'other_owner_capture_reference', 'path': capture,
                    'reason': 'retained_other_owner_capture_declaration', 'other_intent_id': owner_id,
                    'source_provenance': sink.reserve_provenance((proof, reg_proof)),
                    'current_owner_open_verified': False, 'consumer_fence_checked': False,
                    'payload_bytes_verified': False, 'action': 'KEEP'})
    return result
