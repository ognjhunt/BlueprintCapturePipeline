"""Current and retained SAM metadata only; no phase/runtime/science validators."""
from __future__ import annotations

import re

from . import task_evaluation_scene_source_family_contracts as c

PHASES = ('source_selections', 'standard_splat_conversion', 'calibrated_views', 'sam31_inputs', 'sam31_tracking',
          'sam31_review', 'calibrated_masks', 'removal_freezes', 'contribution_sweep', 'segment_cutout')
HOST_NAMES = {'task_request', 'installation_receipt', 'publisher_intake', 'source_preparation_receipt', 'interiorgs_terms'}
STATES = {'pending', 'processing', 'awaiting_source_preparation', 'awaiting_capacity', 'materialized', 'completed', 'blocked'}
CHILD = re.compile(r'sam31-[a-f0-9]{64}\Z')
NAME = re.compile(r'[A-Za-z][A-Za-z0-9_]{0,191}\Z')
SCHEMAS = {
    'sam_plans': ('task_evaluation_sam31_preparation_plan.v1', 'plan_digest'),
    'sam_profiles': ('task_evaluation_sam31_preparation_profile.v1', 'profile_digest'),
    'sam_recipes': ('task_evaluation_scene_construction_recipe.v1', 'recipe_digest'),
    'sam_stage_configurations': ('observed_appearance_object_removal_configuration.v1', None),
    'sam_host_tasks': ('task_evaluation_minimal_task_request.v1', None),
    'sam_parent_envelopes': ('task_evaluation_launch_preparation_envelope.v1', 'envelope_digest'),
    'sam_jobs': ('task_evaluation_sam31_preparation_execution_job.v1', 'job_digest'),
    'sam_results': ('task_evaluation_sam31_preparation_execution_result.v1', 'result_digest'),
    'sam_execution_progress': ('task_evaluation_sam31_preparation_execution_progress.v1', 'progress_digest'),
    'source_progress': ('task_evaluation_sam31_preparation_progress.v1', 'progress_digest'),
    'source_resume_signals': ('task_evaluation_sam31_preparation_resume.v1', 'signal_digest'),
}
EVIDENCE_ORDER = ('calibrated_mask_set', 'segment_cutout_set', 'track_selection_review', 'selection_inputs', 'standard_splat_conversion')
EVIDENCE = set(EVIDENCE_ORDER)


def _allowed(context, row):
    roots = [*context.metadata_roots, context.roots['host_input_root'], context.roots['sam_queue_root'],
             context.roots['sam_execution_root'], context.roots['factory_output_root'],
             *[r['input_root'] for r in context.routes]]
    c.require(any(c.under(row[1]['path'], root) for root in roots), 'sam_record_path_invalid')


def refs(context, value, proof, *, maximum=64, nonempty=False):
    c.require(isinstance(value, dict) and len(value) <= maximum and (value or not nonempty)
        and all(c.matches(k, NAME) for k in value), 'sam_artifacts_invalid')
    for reference in value.values():
        context.selected(reference, proof)
    return value


def _parent(context, row):
    value, proof = row
    request = value.get('request')
    c.require(isinstance(request, dict), 'sam_parent_invalid')
    if request.get('schema_version') != 'task_evaluation_launch_preparation_request.v1' or request.get('run_mode') != 'scene_configuration':
        context.missing('sam_parent', 'unsupported_retained_request', [proof])
        return None
    for field in ('preparation_id', 'run_id', 'team_namespace'):
        c.require(c.matches(request.get(field), c.ID), 'sam_parent_identity_invalid')
    c.require(c.matches(request.get('expected_production_commit'), c.COMMIT)
        and c.matches(value.get('request_digest')) and value['request_digest'] == c.canonical_digest(request), 'sam_parent_identity_invalid')
    stem = request['preparation_id'] + '-' + value['request_digest'][7:]
    route = next((r for r in context.routes if c.under(proof['path'], r['queue_root'])), None)
    c.require(route is not None and proof['path'][len(route['queue_root']) + 1:].split('/') in
        [[state, stem + '.json'] for state in STATES], 'sam_parent_path_invalid')
    c.require(all(isinstance(request.get(role), dict) and isinstance(request[role].get('identity'), dict)
        and bool(request[role]['identity']) for role in ('scene', 'task')), 'sam_parent_identity_invalid')
    return row


def _plan(context, row):
    value, proof = row
    c.require(c.matches(value.get('source_commit'), c.COMMIT) and value.get('phase_sequence') == list(PHASES)
        and isinstance(value.get('scene_identity'), dict) and bool(value['scene_identity'])
        and isinstance(value.get('task_identity'), dict) and bool(value['task_identity'])
        and c.matches(value.get('publisher_scene_id'), c.ID) and c.matches(value.get('server_profile_sha256')),
        'sam_plan_invalid')
    c.require(isinstance(value.get('host_inputs'), dict) and set(value['host_inputs']) == HOST_NAMES, 'sam_plan_inputs_invalid')
    refs(context, value['host_inputs'], proof)
    task = context.selected(value['host_inputs']['task_request'], proof, {'sam_host_tasks'})
    if task and task[0].get('schema_version') == SCHEMAS['sam_host_tasks'][0]:
        c.require(all(task[0].get(k) == value[k] for k in ('scene_identity', 'task_identity', 'publisher_scene_id')), 'sam_task_identity_invalid')
        if task[0].get('expected_production_commit') is not None:
            c.require(task[0]['expected_production_commit'] == value['source_commit'], 'sam_task_commit_invalid')
        else:
            context.missing('sam_task_commit', 'historical_task_commit_unavailable', [task[1]])
    installed = context.selected(value['host_inputs']['installation_receipt'], proof, {'sam_host_evidence', 'opaque_evidence'})
    prepared = context.selected(value['host_inputs']['source_preparation_receipt'], proof, {'sam_host_evidence', 'opaque_evidence'})
    if installed and installed[0] is not None and installed[0].get('schema_version') == 'public_scene_host_input_installation_receipt.v1':
        c.require(installed[0]['scene_id'] == value['publisher_scene_id'], 'sam_installation_scene_invalid')
        if prepared and prepared[0] is not None and prepared[0].get('schema_version') == 'public_scene_source_preparation.v1':
            c.require(prepared[0]['source_installation_digest'] == installed[0]['receipt_digest']
                and prepared[0]['scene_id'] == installed[0]['scene_id'], 'sam_source_installation_invalid')


def _job(context, row):
    value, proof = row
    c.require(set(value) == {'schema_version', 'child_id', 'parent_preparation_id', 'parent_request_digest',
        'plan_digest', 'phase', 'inputs_digest', 'expected_source_commit', 'plan_ref', 'inputs', 'job_digest'}, 'sam_job_invalid')
    c.require(value['phase'] in PHASES and c.matches(value['child_id'], CHILD)
        and c.matches(value['parent_preparation_id'], c.ID) and c.matches(value['expected_source_commit'], c.COMMIT)
        and all(c.matches(value[k]) for k in ('parent_request_digest', 'plan_digest', 'inputs_digest')), 'sam_job_invalid')
    refs(context, value['inputs'], proof)
    context.selected(value['plan_ref'], proof, {'sam_plans'})
    c.require(value['plan_ref']['sha256'] == value['plan_digest'], 'sam_job_plan_invalid')
    identities = {name: {k: ref[k] for k in ('sha256', 'size_bytes')} for name, ref in value['inputs'].items()}
    key = {k: value[k] for k in ('parent_request_digest', 'plan_digest', 'phase', 'inputs_digest')}
    c.require(value['inputs_digest'] == c.canonical_digest(identities)
        and value['child_id'] == 'sam31-' + c.canonical_digest(key)[7:], 'sam_job_identity_invalid')
    c.require(proof['path'] in [c.child(context.roots['sam_queue_root'], state, value['child_id'] + '.json')
        for state in ('pending', 'processing', 'waiting_external', 'completed', 'failed')], 'sam_job_path_invalid')


def _result(context, row):
    value, proof = row
    c.require(c.matches(value.get('child_id'), CHILD) and c.matches(value.get('source_commit'), c.COMMIT), 'sam_result_invalid')
    status = value.get('status')
    if status not in {'completed', 'failed'}:
        context.missing('sam_result', 'unsupported_retained_status', [proof])
        return False
    for field in ('job_digest', 'parent_request_digest', 'plan_digest'):
        c.require(value.get(field) is None and status == 'failed' or c.matches(value.get(field)), 'sam_result_invalid')
    c.require(value.get('phase') in PHASES or (status == 'failed' and value.get('phase') is None), 'sam_result_invalid')
    refs(context, value.get('artifacts'), proof, nonempty=status == 'completed')
    c.require(proof['path'] in [c.child(context.roots['sam_queue_root'], 'results', value['child_id'] + suffix + '.json')
        for suffix in ('', '.conflict-' + value['result_digest'][7:])], 'sam_result_path_invalid')
    if value.get('executor_result') is not None:
        outcome = value['executor_result']
        c.require(isinstance(outcome, dict) and outcome.get('status') == status
            and outcome.get('artifacts') == value['artifacts'], 'sam_result_outcome_invalid')
    return True


def _plan_parent(context, plan, parent):
    value, proof = plan
    request = parent[0]['request']
    c.require(value['source_commit'] == request['expected_production_commit']
        and value['scene_identity'] == request['scene']['identity'] and value['task_identity'] == request['task']['identity'],
        'sam_parent_plan_invalid')
    mounts = request.get('runtime', {}).get('mounts', [])
    c.require(isinstance(mounts, list) and len(mounts) <= 64, 'sam_parent_mounts_invalid')
    matching = [m.get('source') for m in mounts if isinstance(m, dict) and isinstance(m.get('source'), dict)
                and m['source'].get('digest') == proof['sha256']]
    c.require(matching and all(r.get('size_bytes') == proof['size_bytes'] for r in matching), 'sam_parent_plan_mount_invalid')
    recipe_ref = request.get('construction', {}).get('recipe')
    if recipe_ref is None:
        context.missing('sam_recipe', 'parent_recipe_selector_unavailable', [parent[1]])
        return
    selected = context.by_sha.get((recipe_ref.get('digest'), recipe_ref.get('size_bytes'), 'sam_recipes'), [])
    if len(selected) != 1:
        context.missing('sam_recipe', 'recipe_bytes_unavailable_or_ambiguous', [parent[1]], selector={
            k: recipe_ref.get(k) for k in ('uri', 'digest', 'size_bytes')})
        return
    stages = selected[0][0].get('stage_sequence')
    c.require(isinstance(stages, list) and 1 <= len(stages) <= 64 and isinstance(stages[0], dict), 'sam_recipe_invalid')
    stage_ref = stages[0].get('configuration', {})
    selected_stage = context.by_sha.get((stage_ref.get('digest'), stage_ref.get('size_bytes'), 'sam_stage_configurations'), [])
    for stage in selected_stage:
        if stage[0].get('schema_version') != SCHEMAS['sam_stage_configurations'][0]:
            continue
        declared = stage[0].get('sam31_preparation_plan', {})
        c.require(stage[0].get('sam31_review_kind') == 'ai' and any(declared == m for m in matching), 'sam_stage_plan_invalid')
    if not selected_stage:
        context.missing('sam_stage_configuration', 'stage_bytes_unavailable', [selected[0][1]], selector=stage_ref)


def inventory(context, old):
    tables = {}
    for role, (schema, field) in SCHEMAS.items():
        tables[role] = context.known(role, schema, field)
        if role not in {'sam_parent_envelopes', 'source_progress', 'source_resume_signals'}:
            for row in tables[role]:
                _allowed(context, row)
    context.sam_tables = tables
    context.by_sha = {}
    for role, rows in context.decoded.items():
        for row in rows:
            context.by_sha.setdefault((row[1]['sha256'], row[1]['size_bytes'], role), []).append(row)
    parents = []
    for row in tables['sam_parent_envelopes'] + context.decoded['preparation_envelopes']:
        parent = _parent(context, row)
        if parent:
            parents.append(parent)
    parent_index = {}
    for row in parents:
        parent_index.setdefault((row[0]['request']['preparation_id'], row[0]['request_digest']), []).append(row)
    context.sam_parents = parent_index
    plan_index = {r[1]['sha256']: r for r in tables['sam_plans']}
    context.sam_plans = plan_index
    for row in tables['sam_plans']:
        _plan(context, row)
        profiles = context.by_sha.get((row[0]['server_profile_sha256'], context.sizes.get(row[0]['server_profile_sha256']), 'sam_profiles'), [])
        for profile in profiles:
            c.require(profile[0]['source_commit'] == row[0]['source_commit'], 'sam_profile_commit_invalid')
            refs(context, profile[0].get('artifact_references'), profile[1])
    for row in tables['sam_profiles']:
        c.require(c.matches(row[0].get('source_commit'), c.COMMIT), 'sam_profile_invalid')
        refs(context, row[0].get('artifact_references'), row[1])
    for task in tables['sam_host_tasks']:
        if 'request_digest' in task[0]:
            c.c.seal(task, 'request_digest')
    observations, job_index = context.rows(), {}
    for parent in parents:
        observations.append(c.observation(parent, role='sam_parent', preparation_id=parent[0]['request']['preparation_id'],
                                          request_digest=parent[0]['request_digest']))
    for row in tables['sam_jobs']:
        _job(context, row)
        job_index.setdefault((row[0]['child_id'], row[0]['job_digest']), []).append(row)
    context.sam_jobs = job_index
    results_by_job = {}
    for row in tables['sam_results']:
        if _result(context, row):
            value, proof = row
            jobs = job_index.get((value['child_id'], value.get('job_digest')), [])
            for job in jobs:
                c.require(all(value.get(k) == job[0][k] for k in ('parent_request_digest', 'plan_digest', 'phase'))
                    and value['source_commit'] == job[0]['expected_source_commit'], 'sam_result_job_invalid')
            results_by_job.setdefault((value['child_id'], value.get('job_digest')), []).append(row)
            observations.append(c.observation(row, role='sam_result', child_id=value['child_id'], result_status=value['status']))
    context.sam_results = results_by_job
    receipts = _receipts(context, results_by_job, job_index)
    context.sam_receipts = receipts
    for row in tables['sam_jobs']:
        value, proof = row
        selected_plan = context.selected(value['plan_ref'], proof, {'sam_plans'})
        if selected_plan and selected_plan[0].get('schema_version') == SCHEMAS['sam_plans'][0]:
            c.require(selected_plan[0]['source_commit'] == value['expected_source_commit'], 'sam_job_plan_commit_invalid')
        parent_rows = parent_index.get((value['parent_preparation_id'], value['parent_request_digest']), [])
        for parent in parent_rows:
            if selected_plan:
                _plan_parent(context, selected_plan, parent)
        matched = results_by_job.get((value['child_id'], value['job_digest']), [])
        observations.append(c.observation(row, role='sam_job', child_id=value['child_id'], phase=value['phase'],
            result_binding_verified=len(matched) == 1, parent_binding_verified=len(parent_rows) == 1))
        owner = [m for m in old['seed']['members'] if m['kind'] == 'preparation_workspace'
                 and m.get('binding', {}).get('preparation_id') == value['parent_preparation_id']
                 and m.get('binding', {}).get('request_digest') == value['parent_request_digest']]
        if len(owner) == len(parent_rows) == 1:
            context.member(c.child(context.roots['sam_execution_root'], value['parent_request_digest'][7:], value['child_id']),
                'sam_execution_dependency', {'binding_strength': 'owner_parent_exact_job', 'intent_id': context.intent_id},
                [proof, parent_rows[0][1]])
        if not parent_rows:
            context.missing('sam_parent', 'parent_selector_unavailable', [proof], selector={
                'preparation_id': value['parent_preparation_id'], 'request_digest': value['parent_request_digest']})
    _execution_progress(context, job_index, observations)
    _source_progress(context, parent_index, observations)
    _host_evidence(context)
    context.sam_observations = observations
    return observations


def _receipts(context, results, jobs):
    index = {}
    for row in context.decoded['sam_execution_receipts']:
        value, proof = row
        if value.get('schema_version') not in {'task_evaluation_sam31_phase_execution_receipt.v1', 'task_evaluation_sam31_phase_replay_receipt.v1'}:
            context.missing('sam_execution_receipt', 'unsupported_retained_schema', [proof])
            continue
        c.c.seal(row, 'receipt_digest')
        c.require(c.matches(value.get('source_commit'), c.COMMIT) and c.matches(value.get('job_digest'))
            and value.get('phase') in PHASES and isinstance(value.get('outcome'), dict), 'sam_receipt_invalid')
        outcome = value['outcome']
        refs(context, outcome.get('artifacts'), proof)
        c.require(outcome.get('status') in {'completed', 'failed', 'waiting_for_external_result'}, 'sam_receipt_invalid')
        child_id = proof['path'].rsplit('/', 2)[1]
        if value['schema_version'] == 'task_evaluation_sam31_phase_execution_receipt.v1':
            relative = proof['path'][len(context.roots['sam_execution_root']) + 1:].split('/')
            c.require(c.under(proof['path'], context.roots['sam_execution_root']) and len(relative) == 3
                and c.matches('sha256:' + relative[0]) and c.matches(relative[1], CHILD)
                and relative[2] == 'phase_execution_receipt.v1.json', 'sam_receipt_path_invalid')
        candidates = jobs.get((child_id, value['job_digest']), [])
        for job in candidates:
            c.require(value['phase'] == job[0]['phase'] and value['source_commit'] == job[0]['expected_source_commit'], 'sam_receipt_job_invalid')
            if value['schema_version'] == 'task_evaluation_sam31_phase_execution_receipt.v1':
                c.require(proof['path'] == c.child(context.roots['sam_execution_root'], job[0]['parent_request_digest'][7:],
                    child_id, 'phase_execution_receipt.v1.json'), 'sam_receipt_path_invalid')
                for result in results.get((child_id, value['job_digest']), []):
                    c.require(outcome.get('status') == result[0]['status'] and outcome.get('artifacts') == result[0]['artifacts'],
                              'sam_receipt_result_invalid')
        index.setdefault((child_id, value['job_digest']), []).append(row)
    return index


def _execution_progress(context, jobs, observations):
    for row in context.sam_tables['sam_execution_progress']:
        value, proof = row
        c.require(c.matches(value.get('job_digest')) and type(value.get('sequence')) is int and value['sequence'] > 0
            and value.get('status') == 'waiting_for_external_result' and isinstance(value.get('executor_result'), dict), 'sam_execution_progress_invalid')
        c.require(value['executor_result'].get('status') == value['status'], 'sam_execution_progress_invalid')
        refs(context, value['executor_result'].get('artifacts'), proof)
        name = proof['path'].rsplit('/', 2)[1]
        c.require(c.matches(name, CHILD) and proof['path'] == c.child(context.roots['sam_queue_root'], 'progress', name,
            f"{value['sequence']:06d}.json"), 'sam_execution_progress_path_invalid')
        observations.append(c.observation(row, role='sam_external_wait', child_id=name, job_digest=value['job_digest']))


def _source_progress(context, parents, observations):
    index, branches = {}, {}
    for row in context.sam_tables['source_progress']:
        value, proof = row
        c.require(c.matches(value.get('preparation_id'), c.ID) and c.matches(value.get('run_id'), c.ID)
            and c.matches(value.get('request_digest')) and c.matches(value.get('source_commit'), c.COMMIT)
            and type(value.get('sequence')) is int and value['sequence'] > 0 and isinstance(value.get('advancement'), dict)
            and value.get('status') == value['advancement'].get('status')
            and value.get('provider_mutation_performed') is False and value.get('paid_execution_requested') is False,
            'sam_source_progress_invalid')
        stem = value['preparation_id'] + '-' + value['request_digest'][7:]
        c.require(any(proof['path'] == c.child(route['queue_root'], 'source-progress', stem,
            f"{value['sequence']:06d}-" + value['progress_digest'][7:] + '.json') for route in context.routes), 'sam_source_progress_path_invalid')
        c.require(value.get('previous_progress_digest') is None if value['sequence'] == 1
            else c.matches(value.get('previous_progress_digest')), 'sam_progress_predecessor_invalid')
        index.setdefault((value['request_digest'], value['progress_digest']), []).append(row)
        branches.setdefault((value['request_digest'], value['sequence']), []).append(row)
    for row in context.sam_tables['source_progress']:
        value, proof = row
        previous = index.get((value['request_digest'], value.get('previous_progress_digest')), [])
        for predecessor in previous:
            c.require(predecessor[0]['sequence'] + 1 == value['sequence'], 'sam_progress_predecessor_invalid')
        if value['sequence'] > 1 and not previous:
            context.missing('sam_previous_progress', 'canonical_predecessor_unavailable', [proof],
                            selector={'progress_digest': value['previous_progress_digest']})
        selected_parents = parents.get((value['preparation_id'], value['request_digest']), [])
        for parent in selected_parents:
            request = parent[0]['request']
            c.require(request['run_id'] == value['run_id'] and request['expected_production_commit'] == value['source_commit'],
                      'sam_progress_parent_invalid')
        final = value['advancement'].get('sam31_preparation_result')
        if final is not None:
            c.require(isinstance(final, dict), 'sam_final_invalid')
            if final.get('schema_version') != 'task_evaluation_sam31_preparation_result.v1':
                context.missing('sam_final', 'unsupported_retained_schema', [proof])
                continue
            nested = context.nested(final, proof, '/advancement/sam31_preparation_result', 'result_digest')
            c.require(final.get('status') == 'exact_mask_inputs_ready' and final.get('source_commit') == value['source_commit']
                and c.matches(final.get('plan_digest')) and isinstance(final.get('evidence'), dict) and set(final['evidence']) == EVIDENCE
                and final['evidence'] == value['advancement'].get('sam31_exact_mask_inputs')
                and value['advancement'].get('evidence_refs') == [final['evidence'][name] for name in EVIDENCE_ORDER], 'sam_final_invalid')
            refs(context, final['evidence'], nested[1])
            receipts = final.get('stage_result_receipts')
            c.require(isinstance(receipts, list) and len(receipts) <= 10, 'sam_final_receipts_invalid')
            declared_adoption = final.get('completed_prefix_adoption')
            start = 0
            if declared_adoption is not None:
                c.require(isinstance(declared_adoption, dict) and declared_adoption.get('through_phase') in PHASES[2:]
                    and c.matches(declared_adoption.get('original_execution_commit'), c.COMMIT)
                    and isinstance(declared_adoption.get('original_phase_result_receipts'), list)
                    and len(declared_adoption['original_phase_result_receipts']) <= 10, 'sam_final_adoption_invalid')
                context.selected(declared_adoption.get('receipt'), nested[1], {'sam_adoptions'})
                start = PHASES.index(declared_adoption['through_phase']) + 1
                for original_result in declared_adoption['original_phase_result_receipts']:
                    context.selected(original_result, nested[1], {'sam_results'})
            c.require(len(receipts) == len(PHASES[start:]), 'sam_final_receipts_invalid')
            for phase, reference in zip(PHASES[start:], receipts):
                result = context.selected(reference, nested[1], {'sam_results'})
                if result and result[0].get('schema_version') == SCHEMAS['sam_results'][0]:
                    c.require(result[0].get('status') == 'completed' and result[0].get('source_commit') == final['source_commit']
                        and result[0].get('plan_digest') == final['plan_digest']
                        and result[0].get('parent_request_digest') == value['request_digest']
                        and result[0].get('phase') == phase, 'sam_final_receipts_invalid')
            observations.append(c.observation(nested, role='sam_final', plan_digest=final['plan_digest'],
                parent_binding_verified=len(selected_parents) == 1 and len(branches[(value['request_digest'], value['sequence'])]) == 1))
        observations.append(c.observation(row, role='sam_source_progress', sequence=value['sequence'], progress_status=value['status']))
    for row in context.sam_tables['source_resume_signals']:
        value, proof = row
        c.require(c.matches(value.get('preparation_id'), c.ID) and c.matches(value.get('request_digest'))
            and c.matches(value.get('progress_digest')) and c.matches(value.get('source_commit'), c.COMMIT)
            and value.get('kind') in {'human_review', 'child_result'}, 'sam_resume_invalid')
        context.selected(value.get('evidence_ref'), proof)
        stem = value['preparation_id'] + '-' + value['request_digest'][7:]
        c.require(any(proof['path'] in (c.child(route['queue_root'], 'source-resume-pending', value['signal_digest'][7:] + '.json'),
            c.child(route['queue_root'], 'source-resume-completed', stem, value['signal_digest'][7:] + '.json')) for route in context.routes),
            'sam_resume_path_invalid')
        for prior in index.get((value['request_digest'], value['progress_digest']), []):
            c.require(prior[0]['source_commit'] == value['source_commit'] and prior[0]['preparation_id'] == value['preparation_id'],
                      'sam_resume_progress_invalid')
        observations.append(c.observation(row, role='sam_resume', wake_authorized=False))


def _host_evidence(context):
    supported = {'public_scene_host_input_installation_receipt.v1', 'public_scene_source_preparation.v1',
                 'standard_splat_conversion_receipt.v1'}
    for row in context.decoded['sam_host_evidence']:
        _allowed(context, row)
        if row[0].get('schema_version') in supported:
            c.c.seal(row, 'receipt_digest')
            value, proof = row
            schema = value['schema_version']
            if schema == 'public_scene_host_input_installation_receipt.v1':
                c.require(value.get('status') == 'installed' and c.matches(value.get('scene_id'), c.ID)
                    and c.matches(value.get('packet_id'), c.ID) and c.matches(value.get('source_commit_sha'), c.COMMIT)
                    and c.matches(value.get('packet_digest')) and value.get('authoritative_request_digest') == value['packet_digest']
                    and value.get('destination_root') == proof['path'].rsplit('/', 1)[0]
                    and value.get('service_readable') is True and value.get('provider_mutation_performed') is False
                    and value.get('paid_resource_used') is False, 'sam_installation_invalid')
            elif schema == 'public_scene_source_preparation.v1':
                c.require(value.get('status') in {'blocked', 'source_context_prepared_pending_calibrated_views'}
                    and c.matches(value.get('source_commit'), c.COMMIT) and c.matches(value.get('scene_id'), c.ID)
                    and c.matches(value.get('source_installation_digest')) and all(value.get(k) is False for k in
                        ('provider_mutation_performed', 'paid_resource_used', 'candidate_policy_queried')), 'sam_source_preparation_invalid')
            elif schema == 'standard_splat_conversion_receipt.v1':
                output = row[0].get('output')
                c.require(isinstance(output, dict) and c.matches(output.get('sha256'))
                    and type(output.get('size_bytes')) is int and output['size_bytes'] > 0, 'sam_conversion_invalid')
                c.relative(output.get('relative_path'))
        else:
            context.missing('sam_host_evidence', 'unsupported_retained_schema', [row[1]])
