"""Current and retained SAM metadata only; no phase/runtime/science validators."""
from __future__ import annotations

from .task_evaluation_scene_lineage_budget import _work_collect, _work, _work_call, _work_hash, _work_items, _work_kwargs

import re
import json
import hashlib

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
    'sam_parent_results': ('task_evaluation_launch_preparation_result.v1', 'result_digest'),
    'sam_jobs': ('task_evaluation_sam31_preparation_execution_job.v1', 'job_digest'),
    'sam_results': ('task_evaluation_sam31_preparation_execution_result.v1', 'result_digest'),
    'sam_execution_progress': ('task_evaluation_sam31_preparation_execution_progress.v1', 'progress_digest'),
    'source_progress': ('task_evaluation_sam31_preparation_progress.v1', 'progress_digest'),
    'source_resume_signals': ('task_evaluation_sam31_preparation_resume.v1', 'signal_digest'),
}
EVIDENCE_ORDER = ('calibrated_mask_set', 'segment_cutout_set', 'track_selection_review', 'selection_inputs', 'standard_splat_conversion')
EVIDENCE = set(EVIDENCE_ORDER)


def _allowed(context, row, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    roots = [*context.metadata_roots, context.roots['host_input_root'], context.roots['sam_queue_root'],
             context.roots['sam_execution_root'], context.roots['factory_output_root'],
             *[r['input_root'] for r in (_work_items(context.routes, work_budget) if work_budget is not None else context.routes)]]
    c.require(any(c.under(row[1]['path'], root, **_work_kwargs(work_budget)) for root in (_work_items(roots, work_budget) if work_budget is not None else roots)), 'sam_record_path_invalid', **_work_kwargs(work_budget))


def refs(context, value, proof, *, maximum=64, nonempty=False, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    c.require(isinstance(value, dict) and len(value) <= maximum and (value or not nonempty)
        and all(c.matches(k, NAME, **_work_kwargs(work_budget)) for k in (_work_items(value, work_budget) if work_budget is not None else value)), 'sam_artifacts_invalid', **_work_kwargs(work_budget))
    for reference in (_work_items(value.values(), work_budget) if work_budget is not None else value.values()):
        c.require(isinstance(reference, dict) and (_work_collect(work_budget, set, reference) if work_budget is not None else set(reference)) == {'path', 'sha256', 'size_bytes'}, 'sam_artifact_reference_invalid', **_work_kwargs(work_budget))
        context.selected(reference, proof)
    return value


def _parent(context, row, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    value, proof = row
    request = value.get('request')
    c.require(isinstance(request, dict), 'sam_parent_invalid', **_work_kwargs(work_budget))
    if request.get('schema_version') != 'task_evaluation_launch_preparation_request.v1' or request.get('run_mode') != 'scene_configuration':
        context.missing('sam_parent', 'unsupported_retained_request', [proof])
        return None
    for field in (_work_items(('preparation_id', 'run_id', 'team_namespace'), work_budget) if work_budget is not None else ('preparation_id', 'run_id', 'team_namespace')):
        c.require(c.matches(request.get(field), c.ID, **_work_kwargs(work_budget)), 'sam_parent_identity_invalid', **_work_kwargs(work_budget))
    c.require(c.matches(request.get('expected_production_commit'), c.COMMIT, **_work_kwargs(work_budget))
        and c.matches(value.get('request_digest'), **_work_kwargs(work_budget)) and value['request_digest'] == (_work_call(work_budget, c.canonical_digest, request) if work_budget is not None else c.canonical_digest(request)), 'sam_parent_identity_invalid', **_work_kwargs(work_budget))
    stem = request['preparation_id'] + '-' + value['request_digest'][7:]
    route = next((r for r in (_work_items(context.routes, work_budget) if work_budget is not None else context.routes) if c.under(proof['path'], r['queue_root'], **_work_kwargs(work_budget))), None)
    c.require(route is not None and proof['path'][len(route['queue_root']) + 1:].split('/') in
        [[state, stem + '.json'] for state in (_work_items(STATES, work_budget) if work_budget is not None else STATES)], 'sam_parent_path_invalid', **_work_kwargs(work_budget))
    c.require(all(isinstance(request.get(role), dict) and isinstance(request[role].get('identity'), dict)
        and bool(request[role]['identity']) for role in (_work_items(('scene', 'task'), work_budget) if work_budget is not None else ('scene', 'task'))), 'sam_parent_identity_invalid', **_work_kwargs(work_budget))
    return row


def _parent_result(context, row, parents, *, work_budget=None):
    """A retained SAM result needs one exact selected parent and worker status."""
    if work_budget is None:
        work_budget = getattr(context, 'work_budget', None)
    if work_budget is not None:
        _work(work_budget)
    value, proof = row
    matches = []
    for parent in (_work_items(parents, work_budget) if work_budget is not None else parents):
        selected = parent[1]['path']
        route = next((r for r in (_work_items(context.routes, work_budget) if work_budget is not None else context.routes)
                      if c.under(selected, r['queue_root'], **_work_kwargs(work_budget))), None)
        if route is not None and proof['path'] == c.child(route['queue_root'], 'results',
                                                          selected.rsplit('/', 1)[1], **_work_kwargs(work_budget)):
            matches.append(parent)
    c.require(len({(parent[1]['sha256'], parent[1]['size_bytes']) for parent in matches}) == 1,
              'sam_parent_result_parent_invalid', **_work_kwargs(work_budget))
    parent = matches[0]
    request = parent[0]['request']
    c.require(all(value.get(result_key) == request[request_key] for result_key, request_key in (
        ('preparation_id', 'preparation_id'), ('run_id', 'run_id'),
        ('team_namespace', 'team_namespace'), ('source_commit', 'expected_production_commit')))
        and value.get('provider_mutation_performed') is False
        and value.get('paid_execution_requested') is False,
        'sam_parent_result_identity_invalid', **_work_kwargs(work_budget))
    successful = value.get('status') == 'queued_for_production_scene_configuration'
    if not successful:
        context.missing('sam_parent_result', 'unsupported_retained_status', [proof, parent[1]])
    return (c.observation(row, role='sam_parent_result', parent_binding_verified=successful,
                          preparation_id=request['preparation_id'], request_digest=parent[0]['request_digest'],
                          **_work_kwargs(work_budget)),
            parent if successful and len(matches) == 1 else None)


def _mounted_plan_cache_member(context, result, parent, plans, *, work_budget=None):
    """Name one readback plan CAS candidate; action still proves its generation."""
    if parent is None:
        return
    if work_budget is None:
        work_budget = getattr(context, 'work_budget', None)
    if work_budget is not None:
        _work(work_budget)
    value, proof = result
    request = parent[0]['request']
    mounts = request.get('runtime', {}).get('mounts', [])
    references = value.get('references')
    if (type(references) is not list or len(references) > 256
            or value.get('full_byte_service_account_readback_passed') is not True):
        return
    candidates = []
    for index, mount in enumerate(_work_items(mounts, work_budget) if work_budget is not None else mounts):
        source = mount.get('source') if isinstance(mount, dict) else None
        if not isinstance(source, dict) or set(source) != {'uri', 'digest', 'size_bytes'}:
            continue
        selected = [plan for plan in (_work_items(plans, work_budget) if work_budget is not None else plans)
                    if plan[1]['sha256'] == source['digest'] and plan[1]['size_bytes'] == source['size_bytes']]
        if len(selected) == 1:
            candidates.append((index, source, selected[0]))
    if len(candidates) != 1:
        return
    index, source, plan = candidates[0]
    _plan_parent(context, plan, parent, **_work_kwargs(work_budget))
    task = context.selected(plan[0]['host_inputs']['task_request'], plan[1], {'sam_host_tasks'})
    if task is None or task[0].get('schema_version') != SCHEMAS['sam_host_tasks'][0]:
        return
    contract_path = f'runtime.mounts.{index}.source'
    matching = [row for row in (_work_items(references, work_budget) if work_budget is not None else references)
                if isinstance(row, dict) and row.get('contract_path') == contract_path]
    if len(matching) != 1:
        return
    row = matching[0]
    digest = source['digest']
    projected = c.child(context.roots['preparation_input_root'], request['preparation_id'], digest[7:],
                        **_work_kwargs(work_budget))
    if (any(row.get(key) != source[key] for key in ('uri', 'digest', 'size_bytes'))
            or row.get('materialized_path') != projected
            or row.get('full_byte_service_account_readback_passed') is not True):
        return
    cache = c.child(context.roots['content_store_root'], digest[7:], **_work_kwargs(work_budget))
    context.member(cache, 'prepared_cache_object',
                   {'binding_strength': 'sam_parent_plan_exact_readback',
                    'preparation_id': request['preparation_id'],
                    'request_digest': parent[0]['request_digest'], 'plan_digest': digest},
                   [task[1], plan[1], parent[1], proof])


def _plan(context, row, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    value, proof = row
    c.require(c.matches(value.get('source_commit'), c.COMMIT, **_work_kwargs(work_budget)) and value.get('phase_sequence') == (_work_collect(work_budget, list, PHASES) if work_budget is not None else list(PHASES))
        and isinstance(value.get('scene_identity'), dict) and bool(value['scene_identity'])
        and isinstance(value.get('task_identity'), dict) and bool(value['task_identity'])
        and c.matches(value.get('publisher_scene_id'), c.ID, **_work_kwargs(work_budget)) and c.matches(value.get('server_profile_sha256'), **_work_kwargs(work_budget)),
        'sam_plan_invalid', **_work_kwargs(work_budget))
    c.require(isinstance(value.get('host_inputs'), dict) and (_work_collect(work_budget, set, value['host_inputs']) if work_budget is not None else set(value['host_inputs'])) == HOST_NAMES, 'sam_plan_inputs_invalid', **_work_kwargs(work_budget))
    refs(context, value['host_inputs'], proof, **_work_kwargs(work_budget))
    task = context.selected(value['host_inputs']['task_request'], proof, {'sam_host_tasks'})
    if task and task[0].get('schema_version') == SCHEMAS['sam_host_tasks'][0]:
        c.require(all(task[0].get(k) == value[k] for k in (_work_items(('scene_identity', 'task_identity', 'publisher_scene_id'), work_budget) if work_budget is not None else ('scene_identity', 'task_identity', 'publisher_scene_id'))), 'sam_task_identity_invalid', **_work_kwargs(work_budget))
        if task[0].get('expected_production_commit') is not None:
            c.require(task[0]['expected_production_commit'] == value['source_commit'], 'sam_task_commit_invalid', **_work_kwargs(work_budget))
        else:
            context.missing('sam_task_commit', 'historical_task_commit_unavailable', [task[1]])
    installed = context.selected(value['host_inputs']['installation_receipt'], proof, {'sam_host_evidence', 'opaque_evidence'})
    prepared = context.selected(value['host_inputs']['source_preparation_receipt'], proof, {'sam_host_evidence', 'opaque_evidence'})
    if installed and installed[0] is not None and installed[0].get('schema_version') == 'public_scene_host_input_installation_receipt.v1':
        c.require(installed[0]['scene_id'] == value['publisher_scene_id'], 'sam_installation_scene_invalid', **_work_kwargs(work_budget))
        if prepared and prepared[0] is not None and prepared[0].get('schema_version') == 'public_scene_source_preparation.v1':
            c.require(prepared[0]['source_installation_digest'] == installed[0]['receipt_digest']
                and prepared[0]['scene_id'] == installed[0]['scene_id'], 'sam_source_installation_invalid', **_work_kwargs(work_budget))


def _job(context, row, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    value, proof = row
    c.require((_work_collect(work_budget, set, value) if work_budget is not None else set(value)) == {'schema_version', 'child_id', 'parent_preparation_id', 'parent_request_digest',
        'plan_digest', 'phase', 'inputs_digest', 'expected_source_commit', 'plan_ref', 'inputs', 'job_digest'}, 'sam_job_invalid', **_work_kwargs(work_budget))
    c.require(value['phase'] in PHASES and c.matches(value['child_id'], CHILD, **_work_kwargs(work_budget))
        and c.matches(value['parent_preparation_id'], c.ID, **_work_kwargs(work_budget)) and c.matches(value['expected_source_commit'], c.COMMIT, **_work_kwargs(work_budget))
        and all(c.matches(value[k], **_work_kwargs(work_budget)) for k in (_work_items(('parent_request_digest', 'plan_digest', 'inputs_digest'), work_budget) if work_budget is not None else ('parent_request_digest', 'plan_digest', 'inputs_digest'))), 'sam_job_invalid', **_work_kwargs(work_budget))
    refs(context, value['inputs'], proof, **_work_kwargs(work_budget))
    context.selected(value['plan_ref'], proof, {'sam_plans'})
    c.require(value['plan_ref']['sha256'] == value['plan_digest'], 'sam_job_plan_invalid', **_work_kwargs(work_budget))
    identities = {name: {k: ref[k] for k in (_work_items(('sha256', 'size_bytes'), work_budget) if work_budget is not None else ('sha256', 'size_bytes'))} for name, ref in (_work_items(value['inputs'].items(), work_budget) if work_budget is not None else value['inputs'].items())}
    key = {k: value[k] for k in (_work_items(('parent_request_digest', 'plan_digest', 'phase', 'inputs_digest'), work_budget) if work_budget is not None else ('parent_request_digest', 'plan_digest', 'phase', 'inputs_digest'))}
    c.require(value['inputs_digest'] == (_work_call(work_budget, c.canonical_digest, identities) if work_budget is not None else c.canonical_digest(identities))
        and value['child_id'] == 'sam31-' + (_work_call(work_budget, c.canonical_digest, key) if work_budget is not None else c.canonical_digest(key))[7:], 'sam_job_identity_invalid', **_work_kwargs(work_budget))
    c.require(proof['path'] in [c.child(context.roots['sam_queue_root'], state, value['child_id'] + '.json', **_work_kwargs(work_budget))
        for state in (_work_items(('pending', 'processing', 'waiting_external', 'completed', 'failed'), work_budget) if work_budget is not None else ('pending', 'processing', 'waiting_external', 'completed', 'failed'))], 'sam_job_path_invalid', **_work_kwargs(work_budget))


def _result(context, row, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    value, proof = row
    c.require(c.matches(value.get('child_id'), CHILD, **_work_kwargs(work_budget)) and c.matches(value.get('source_commit'), c.COMMIT, **_work_kwargs(work_budget)), 'sam_result_invalid', **_work_kwargs(work_budget))
    status = value.get('status')
    if status not in {'completed', 'failed'}:
        context.missing('sam_result', 'unsupported_retained_status', [proof])
        return False
    for field in (_work_items(('job_digest', 'parent_request_digest', 'plan_digest'), work_budget) if work_budget is not None else ('job_digest', 'parent_request_digest', 'plan_digest')):
        c.require(value.get(field) is None and status == 'failed' or c.matches(value.get(field), **_work_kwargs(work_budget)), 'sam_result_invalid', **_work_kwargs(work_budget))
    c.require(value.get('phase') in PHASES or (status == 'failed' and value.get('phase') is None), 'sam_result_invalid', **_work_kwargs(work_budget))
    refs(context, value.get('artifacts'), proof, nonempty=status == 'completed', **_work_kwargs(work_budget))
    c.require(proof['path'] in [c.child(context.roots['sam_queue_root'], 'results', value['child_id'] + suffix + '.json', **_work_kwargs(work_budget))
        for suffix in (_work_items(('', '.conflict-' + value['result_digest'][7:]), work_budget) if work_budget is not None else ('', '.conflict-' + value['result_digest'][7:]))], 'sam_result_path_invalid', **_work_kwargs(work_budget))
    if value.get('executor_result') is not None:
        outcome = value['executor_result']
        c.require(isinstance(outcome, dict) and outcome.get('status') == status
            and outcome.get('artifacts') == value['artifacts'], 'sam_result_outcome_invalid', **_work_kwargs(work_budget))
    return True


def _plan_parent(context, plan, parent, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    value, proof = plan
    request = parent[0]['request']
    key = (proof['sha256'], parent[0]['request_digest'])
    if key in context.sam_plan_parent_checked:
        return
    context.sam_plan_parent_checked.add(key)
    c.require(value['source_commit'] == request['expected_production_commit']
        and value['scene_identity'] == request['scene']['identity'] and value['task_identity'] == request['task']['identity'],
        'sam_parent_plan_invalid', **_work_kwargs(work_budget))
    mounts = request.get('runtime', {}).get('mounts', [])
    c.require(isinstance(mounts, list) and len(mounts) <= 64, 'sam_parent_mounts_invalid', **_work_kwargs(work_budget))
    matching = [m.get('source') for m in (_work_items(mounts, work_budget) if work_budget is not None else mounts) if isinstance(m, dict) and isinstance(m.get('source'), dict)
                and m['source'].get('digest') == proof['sha256']]
    c.require(matching and all(r.get('size_bytes') == proof['size_bytes'] for r in (_work_items(matching, work_budget) if work_budget is not None else matching)), 'sam_parent_plan_mount_invalid', **_work_kwargs(work_budget))
    recipe_ref = request.get('construction', {}).get('recipe')
    if recipe_ref is None:
        context.missing('sam_recipe', 'parent_recipe_selector_unavailable', [parent[1]])
        return
    selected = [r for r in (_work_items(context.by_sha.get((recipe_ref.get('digest'), recipe_ref.get('size_bytes'), 'sam_recipes'), []), work_budget) if work_budget is not None else context.by_sha.get((recipe_ref.get('digest'), recipe_ref.get('size_bytes'), 'sam_recipes'), []))
                if r[0].get('schema_version') == SCHEMAS['sam_recipes'][0]]
    if not selected:
        context.missing('sam_recipe', 'recipe_bytes_unavailable_or_ambiguous', [parent[1]], selector={
            k: recipe_ref.get(k) for k in (_work_items(('uri', 'digest', 'size_bytes'), work_budget) if work_budget is not None else ('uri', 'digest', 'size_bytes'))})
        return
    if len(selected) > 1:
        context.missing('sam_recipe', 'recipe_copy_provenance_unresolved', context.provenance(r[1] for r in (_work_items(selected, work_budget) if work_budget is not None else selected)), selector={
            k: recipe_ref.get(k) for k in (_work_items(('uri', 'digest', 'size_bytes'), work_budget) if work_budget is not None else ('uri', 'digest', 'size_bytes'))})
    stages = selected[0][0].get('stage_sequence')
    c.require(isinstance(stages, list) and 1 <= len(stages) <= 64 and isinstance(stages[0], dict), 'sam_recipe_invalid', **_work_kwargs(work_budget))
    stage_ref = stages[0].get('configuration', {})
    selected_stage = [r for r in (_work_items(context.by_sha.get((stage_ref.get('digest'), stage_ref.get('size_bytes'), 'sam_stage_configurations'), []), work_budget) if work_budget is not None else context.by_sha.get((stage_ref.get('digest'), stage_ref.get('size_bytes'), 'sam_stage_configurations'), []))
                      if r[0].get('schema_version') == SCHEMAS['sam_stage_configurations'][0]]
    for stage in (_work_items(selected_stage, work_budget) if work_budget is not None else selected_stage):
        if stage[0].get('schema_version') != SCHEMAS['sam_stage_configurations'][0]:
            continue
        declared = stage[0].get('sam31_preparation_plan', {})
        c.require(stage[0].get('sam31_review_kind') == 'ai' and any(declared == m for m in (_work_items(matching, work_budget) if work_budget is not None else matching)), 'sam_stage_plan_invalid', **_work_kwargs(work_budget))
    if not selected_stage:
        context.missing('sam_stage_configuration', 'stage_bytes_unavailable', [selected[0][1]], selector={
            k: stage_ref.get(k) for k in (_work_items(('uri', 'digest', 'size_bytes'), work_budget) if work_budget is not None else ('uri', 'digest', 'size_bytes'))})


def inventory(context, old, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    tables = {}
    for role, (schema, field) in (_work_items(SCHEMAS.items(), work_budget) if work_budget is not None else SCHEMAS.items()):
        tables[role] = context.known(role, schema, field)
        if role not in {'sam_parent_envelopes', 'sam_parent_results', 'source_progress', 'source_resume_signals'}:
            for row in (_work_items(tables[role], work_budget) if work_budget is not None else tables[role]):
                _allowed(context, row, **_work_kwargs(work_budget))
    context.sam_tables = tables
    context.by_sha = {}
    for role, rows in (_work_items(context.decoded.items(), work_budget) if work_budget is not None else context.decoded.items()):
        for row in (_work_items(rows, work_budget) if work_budget is not None else rows):
            context.by_sha.setdefault((row[1]['sha256'], row[1]['size_bytes'], role), []).append(row)
    parents = []
    for row in (_work_items(tables['sam_parent_envelopes'] + context.decoded['preparation_envelopes'], work_budget) if work_budget is not None else tables['sam_parent_envelopes'] + context.decoded['preparation_envelopes']):
        parent = _parent(context, row, **_work_kwargs(work_budget))
        if parent:
            parents.append(parent)
    parent_index, parent_proofs, parent_raw_keys = {}, {}, set()
    for row in (_work_items(parents, work_budget) if work_budget is not None else parents):
        key = (row[0]['request']['preparation_id'], row[0]['request_digest'])
        raw_key = (*key, row[1]['sha256'], row[1]['size_bytes'])
        if work_budget is not None:
            work_budget.charge('facts', 3)
        parent_proofs.setdefault(raw_key, []).append(row[1])
        if raw_key not in parent_raw_keys:
            parent_raw_keys.add(raw_key)
            parent_index.setdefault(key, []).append(row)
    context.sam_parents = parent_index
    context.sam_current_plans = {}
    context.sam_plan_parent_checked = set()
    for row in (_work_items(tables['sam_plans'], work_budget) if work_budget is not None else tables['sam_plans']):
        _plan(context, row, **_work_kwargs(work_budget))
        profiles = context.by_sha.get((row[0]['server_profile_sha256'], context.sizes.get(row[0]['server_profile_sha256']), 'sam_profiles'), [])
        for profile in (_work_items(profiles, work_budget) if work_budget is not None else profiles):
            if profile[0].get('schema_version') != SCHEMAS['sam_profiles'][0]:
                continue
            c.require(profile[0]['source_commit'] == row[0]['source_commit'], 'sam_profile_commit_invalid', **_work_kwargs(work_budget))
            refs(context, profile[0].get('artifact_references'), profile[1], **_work_kwargs(work_budget))
    for row in (_work_items(tables['sam_profiles'], work_budget) if work_budget is not None else tables['sam_profiles']):
        c.require(c.matches(row[0].get('source_commit'), c.COMMIT, **_work_kwargs(work_budget)), 'sam_profile_invalid', **_work_kwargs(work_budget))
        refs(context, row[0].get('artifact_references'), row[1], **_work_kwargs(work_budget))
    for task in (_work_items(tables['sam_host_tasks'], work_budget) if work_budget is not None else tables['sam_host_tasks']):
        if 'request_digest' in task[0]:
            c.c.seal(task, 'request_digest', **_work_kwargs(work_budget))
    owner_keys = {(m.get('binding', {}).get('preparation_id'), m.get('binding', {}).get('request_digest'))
                  for m in (_work_items(old['seed']['members'], work_budget) if work_budget is not None else old['seed']['members']) if m['kind'] == 'preparation_workspace'}
    semantic_plans = {}
    for plan in (_work_items(tables['sam_plans'], work_budget) if work_budget is not None else tables['sam_plans']):
        semantic_plans.setdefault((plan[1]['sha256'], plan[1]['size_bytes']), plan)
    # Raw-identical copies have the same metadata; retain every raw path in
    # context.raw/observations, but do not Cartesian-scan their content joins.
    semantic_parents = {}
    for parent in (_work_items(parents, work_budget) if work_budget is not None else parents):
        semantic_parents.setdefault((parent[0]['request_digest'], parent[1]['sha256'], parent[1]['size_bytes']), parent)
    for parent in (_work_items(semantic_parents.values(), work_budget) if work_budget is not None else semantic_parents.values()):
        request = parent[0]['request']
        mounts = request.get('runtime', {}).get('mounts', [])
        c.require(isinstance(mounts, list) and len(mounts) <= 64, 'sam_parent_mounts_invalid', **_work_kwargs(work_budget))
        for mount in (_work_items(mounts, work_budget) if work_budget is not None else mounts):
            source = mount.get('source') if isinstance(mount, dict) else None
            if not isinstance(source, dict):
                continue
            plan = semantic_plans.get((source.get('digest'), source.get('size_bytes')))
            if plan:
                _plan_parent(context, plan, parent, **_work_kwargs(work_budget))
                if (request['preparation_id'], parent[0]['request_digest']) in owner_keys:
                    task_ref = plan[0]['host_inputs']['task_request']
                    key = tuple(task_ref[k] for k in (_work_items(('path', 'sha256', 'size_bytes'), work_budget) if work_budget is not None else ('path', 'sha256', 'size_bytes')))
                    context.sam_current_plans.setdefault(key, []).append(plan)
    observations, job_index = context.rows(), {}
    for parent in (_work_items(parents, work_budget) if work_budget is not None else parents):
        observations.append(c.observation(parent, role='sam_parent', preparation_id=parent[0]['request']['preparation_id'],
                                          request_digest=parent[0]['request_digest'], **_work_kwargs(work_budget)))
    for row in (_work_items(tables['sam_parent_results'], work_budget) if work_budget is not None else tables['sam_parent_results']):
        observation, parent = _parent_result(context, row, tables['sam_parent_envelopes'], **_work_kwargs(work_budget))
        observations.append(observation)
        _mounted_plan_cache_member(context, row, parent, tables['sam_plans'], **_work_kwargs(work_budget))
    for row in (_work_items(tables['sam_jobs'], work_budget) if work_budget is not None else tables['sam_jobs']):
        _job(context, row, **_work_kwargs(work_budget))
        job_index.setdefault((row[0]['child_id'], row[0]['job_digest']), []).append(row)
    context.sam_jobs = job_index
    results_by_job = {}
    for row in (_work_items(tables['sam_results'], work_budget) if work_budget is not None else tables['sam_results']):
        if _result(context, row, **_work_kwargs(work_budget)):
            value, proof = row
            jobs = job_index.get((value['child_id'], value.get('job_digest')), [])
            for job in (_work_items(jobs, work_budget) if work_budget is not None else jobs):
                c.require(all(value.get(k) == job[0][k] for k in (_work_items(('parent_request_digest', 'plan_digest', 'phase'), work_budget) if work_budget is not None else ('parent_request_digest', 'plan_digest', 'phase')))
                    and value['source_commit'] == job[0]['expected_source_commit'], 'sam_result_job_invalid', **_work_kwargs(work_budget))
            results_by_job.setdefault((value['child_id'], value.get('job_digest')), []).append(row)
            observations.append(c.observation(row, role='sam_result', child_id=value['child_id'], result_status=value['status'], **_work_kwargs(work_budget)))
    context.sam_results = results_by_job
    receipts = _receipts(context, results_by_job, job_index, **_work_kwargs(work_budget))
    context.sam_receipts = receipts
    for row in (_work_items(tables['sam_jobs'], work_budget) if work_budget is not None else tables['sam_jobs']):
        value, proof = row
        selected_plan = context.selected(value['plan_ref'], proof, {'sam_plans'})
        if selected_plan and selected_plan[0].get('schema_version') == SCHEMAS['sam_plans'][0]:
            c.require(selected_plan[0]['source_commit'] == value['expected_source_commit'], 'sam_job_plan_commit_invalid', **_work_kwargs(work_budget))
        parent_rows = parent_index.get((value['parent_preparation_id'], value['parent_request_digest']), [])
        if selected_plan:
            for selected_parent in (_work_items(parent_rows, work_budget) if work_budget is not None else parent_rows):
                _plan_parent(context, selected_plan, selected_parent, **_work_kwargs(work_budget))
        matched = results_by_job.get((value['child_id'], value['job_digest']), [])
        observations.append(c.observation(row, role='sam_job', child_id=value['child_id'], phase=value['phase'],
            result_binding_verified=len(matched) == 1, parent_binding_verified=len(parent_rows) == 1, **_work_kwargs(work_budget)))
        if (value['parent_preparation_id'], value['parent_request_digest']) in owner_keys and len(parent_rows) == 1:
            context.member(c.child(context.roots['sam_execution_root'], value['parent_request_digest'][7:], value['child_id'], **_work_kwargs(work_budget)),
                'sam_execution_dependency', {'binding_strength': 'owner_parent_exact_job', 'intent_id': context.intent_id},
                [proof, *parent_proofs[(value['parent_preparation_id'], value['parent_request_digest'],
                    parent_rows[0][1]['sha256'], parent_rows[0][1]['size_bytes'])]])
        if not parent_rows:
            context.missing('sam_parent', 'parent_selector_unavailable', [proof], selector={
                'preparation_id': value['parent_preparation_id'], 'request_digest': value['parent_request_digest']})
    _execution_progress(context, job_index, observations, **_work_kwargs(work_budget))
    _source_progress(context, parent_index, observations, **_work_kwargs(work_budget))
    _host_evidence(context, **_work_kwargs(work_budget))
    _artifact_metadata(context, **_work_kwargs(work_budget))
    for rows in (_work_items((tables['sam_jobs'], tables['sam_results']), work_budget) if work_budget is not None else (tables['sam_jobs'], tables['sam_results'])):
        for row in (_work_items(rows, work_budget) if work_budget is not None else rows):
            if row[1]['role'] == 'sam_results' and row[0].get('status') not in {'completed', 'failed'}:
                continue
            artifact_edges(context, row[0].get('inputs', row[0].get('artifacts', {})), row[1], **_work_kwargs(work_budget))
    context.sam_observations = observations
    return observations


def _receipts(context, results, jobs, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    index = {}
    for row in (_work_items(context.decoded['sam_execution_receipts'], work_budget) if work_budget is not None else context.decoded['sam_execution_receipts']):
        value, proof = row
        if value.get('schema_version') not in {'task_evaluation_sam31_phase_execution_receipt.v1', 'task_evaluation_sam31_phase_replay_receipt.v1'}:
            context.missing('sam_execution_receipt', 'unsupported_retained_schema', [proof])
            continue
        c.c.seal(row, 'receipt_digest', **_work_kwargs(work_budget))
        c.require(c.matches(value.get('source_commit'), c.COMMIT, **_work_kwargs(work_budget)) and c.matches(value.get('job_digest'), **_work_kwargs(work_budget))
            and value.get('phase') in PHASES and isinstance(value.get('outcome'), dict), 'sam_receipt_invalid', **_work_kwargs(work_budget))
        outcome = value['outcome']
        refs(context, outcome.get('artifacts'), proof, **_work_kwargs(work_budget))
        c.require(outcome.get('status') in {'completed', 'failed'}, 'sam_receipt_invalid', **_work_kwargs(work_budget))
        if value['schema_version'] == 'task_evaluation_sam31_phase_replay_receipt.v1':
            c.require(value.get('production_execution_authorized') is False, 'sam_replay_authority_invalid', **_work_kwargs(work_budget))
            c.path(value.get('diagnostic_replay_code_root'), **_work_kwargs(work_budget))
        child_id = proof['path'].rsplit('/', 2)[1]
        if value['schema_version'] == 'task_evaluation_sam31_phase_execution_receipt.v1':
            relative = proof['path'][len(context.roots['sam_execution_root']) + 1:].split('/')
            c.require(c.under(proof['path'], context.roots['sam_execution_root'], **_work_kwargs(work_budget)) and len(relative) == 3
                and c.matches('sha256:' + relative[0], **_work_kwargs(work_budget)) and c.matches(relative[1], CHILD, **_work_kwargs(work_budget))
                and relative[2] == 'phase_execution_receipt.v1.json', 'sam_receipt_path_invalid', **_work_kwargs(work_budget))
        candidates = jobs.get((child_id, value['job_digest']), [])
        for job in (_work_items(candidates, work_budget) if work_budget is not None else candidates):
            c.require(value['phase'] == job[0]['phase'] and value['source_commit'] == job[0]['expected_source_commit'], 'sam_receipt_job_invalid', **_work_kwargs(work_budget))
            if value['schema_version'] == 'task_evaluation_sam31_phase_execution_receipt.v1':
                c.require(proof['path'] == c.child(context.roots['sam_execution_root'], job[0]['parent_request_digest'][7:],
                    child_id, 'phase_execution_receipt.v1.json', **_work_kwargs(work_budget)), 'sam_receipt_path_invalid', **_work_kwargs(work_budget))
        if value['schema_version'] == 'task_evaluation_sam31_phase_execution_receipt.v1':
            selected_results = results.get((child_id, value['job_digest']), [])
            if len(selected_results) == 1:
                result = selected_results[0]
                c.require(outcome.get('status') == result[0]['status'] and outcome.get('artifacts') == result[0]['artifacts'],
                          'sam_receipt_result_invalid', **_work_kwargs(work_budget))
            elif len(selected_results) > 1:
                context.missing('sam_receipt_result', 'historical_result_selector_ambiguous', [proof],
                                selector={'job_digest': value['job_digest']})
        index.setdefault((child_id, value['job_digest']), []).append(row)
    return index


def _execution_progress(context, jobs, observations, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    for row in (_work_items(context.sam_tables['sam_execution_progress'], work_budget) if work_budget is not None else context.sam_tables['sam_execution_progress']):
        value, proof = row
        c.require(c.matches(value.get('job_digest'), **_work_kwargs(work_budget)) and type(value.get('sequence')) is int and value['sequence'] > 0
            and value.get('status') == 'waiting_for_external_result' and isinstance(value.get('executor_result'), dict), 'sam_execution_progress_invalid', **_work_kwargs(work_budget))
        c.require(value['executor_result'].get('status') == value['status'], 'sam_execution_progress_invalid', **_work_kwargs(work_budget))
        refs(context, value['executor_result'].get('artifacts'), proof, **_work_kwargs(work_budget))
        name = proof['path'].rsplit('/', 2)[1]
        c.require(c.matches(name, CHILD, **_work_kwargs(work_budget)) and proof['path'] == c.child(context.roots['sam_queue_root'], 'progress', name,
            f"{value['sequence']:06d}.json", **_work_kwargs(work_budget)), 'sam_execution_progress_path_invalid', **_work_kwargs(work_budget))
        observations.append(c.observation(row, role='sam_external_wait', child_id=name, job_digest=value['job_digest'], **_work_kwargs(work_budget)))


def _source_progress(context, parents, observations, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    index, branches = {}, {}
    for row in (_work_items(context.sam_tables['source_progress'], work_budget) if work_budget is not None else context.sam_tables['source_progress']):
        value, proof = row
        c.require(c.matches(value.get('preparation_id'), c.ID, **_work_kwargs(work_budget)) and c.matches(value.get('run_id'), c.ID, **_work_kwargs(work_budget))
            and c.matches(value.get('request_digest'), **_work_kwargs(work_budget)) and c.matches(value.get('source_commit'), c.COMMIT, **_work_kwargs(work_budget))
            and type(value.get('sequence')) is int and value['sequence'] > 0 and isinstance(value.get('advancement'), dict)
            and value.get('status') == value['advancement'].get('status')
            and value.get('provider_mutation_performed') is False and value.get('paid_execution_requested') is False,
            'sam_source_progress_invalid', **_work_kwargs(work_budget))
        stem = value['preparation_id'] + '-' + value['request_digest'][7:]
        c.require(any(proof['path'] == c.child(route['queue_root'], 'source-progress', stem,
            f"{value['sequence']:06d}-" + value['progress_digest'][7:] + '.json', **_work_kwargs(work_budget)) for route in (_work_items(context.routes, work_budget) if work_budget is not None else context.routes)), 'sam_source_progress_path_invalid', **_work_kwargs(work_budget))
        c.require(value.get('previous_progress_digest') is None if value['sequence'] == 1
            else c.matches(value.get('previous_progress_digest'), **_work_kwargs(work_budget)), 'sam_progress_predecessor_invalid', **_work_kwargs(work_budget))
        index.setdefault((value['request_digest'], value['progress_digest']), []).append(row)
        branches.setdefault((value['request_digest'], value['sequence']), []).append(row)
    for row in (_work_items(context.sam_tables['source_progress'], work_budget) if work_budget is not None else context.sam_tables['source_progress']):
        value, proof = row
        previous = index.get((value['request_digest'], value.get('previous_progress_digest')), [])
        for predecessor in (_work_items(previous, work_budget) if work_budget is not None else previous):
            c.require(predecessor[0]['sequence'] + 1 == value['sequence'], 'sam_progress_predecessor_invalid', **_work_kwargs(work_budget))
        if value['sequence'] > 1 and not previous:
            context.missing('sam_previous_progress', 'canonical_predecessor_unavailable', [proof],
                            selector={'progress_digest': value['previous_progress_digest']})
        selected_parents = parents.get((value['preparation_id'], value['request_digest']), [])
        if selected_parents:
            request = selected_parents[0][0]['request']
            c.require(request['run_id'] == value['run_id'] and request['expected_production_commit'] == value['source_commit'],
                      'sam_progress_parent_invalid', **_work_kwargs(work_budget))
        final = value['advancement'].get('sam31_preparation_result')
        if final is not None:
            c.require(isinstance(final, dict), 'sam_final_invalid', **_work_kwargs(work_budget))
            if final.get('schema_version') != 'task_evaluation_sam31_preparation_result.v1':
                context.missing('sam_final', 'unsupported_retained_schema', [proof])
                continue
            nested = context.nested(final, proof, '/advancement/sam31_preparation_result', 'result_digest')
            if final.get('status') != 'exact_mask_inputs_ready':
                context.missing('sam_final', 'unsupported_retained_status', [nested[1]])
                observations.append(c.observation(row, role='sam_source_progress', sequence=value['sequence'], progress_status=value['status'], **_work_kwargs(work_budget)))
                continue
            c.require(final.get('status') == 'exact_mask_inputs_ready' and final.get('source_commit') == value['source_commit']
                and c.matches(final.get('plan_digest'), **_work_kwargs(work_budget)) and isinstance(final.get('evidence'), dict) and (_work_collect(work_budget, set, final['evidence']) if work_budget is not None else set(final['evidence'])) == EVIDENCE
                and final['evidence'] == value['advancement'].get('sam31_exact_mask_inputs')
                and value['advancement'].get('evidence_refs') == [final['evidence'][name] for name in (_work_items(EVIDENCE_ORDER, work_budget) if work_budget is not None else EVIDENCE_ORDER)], 'sam_final_invalid', **_work_kwargs(work_budget))
            refs(context, final['evidence'], nested[1], **_work_kwargs(work_budget))
            receipts = final.get('stage_result_receipts')
            c.require(isinstance(receipts, list) and len(receipts) <= 10, 'sam_final_receipts_invalid', **_work_kwargs(work_budget))
            declared_adoption = final.get('completed_prefix_adoption')
            start = 0
            if declared_adoption is not None:
                c.require(isinstance(declared_adoption, dict) and declared_adoption.get('through_phase') in PHASES[2:]
                    and c.matches(declared_adoption.get('original_execution_commit'), c.COMMIT, **_work_kwargs(work_budget))
                    and isinstance(declared_adoption.get('original_phase_result_receipts'), list)
                    and 1 <= len(declared_adoption['original_phase_result_receipts']) <= 10, 'sam_final_adoption_invalid', **_work_kwargs(work_budget))
                _final_adoption(context, declared_adoption, nested, final, **_work_kwargs(work_budget))
                start = PHASES.index(declared_adoption['through_phase']) + 1
            c.require(len(receipts) == len(PHASES[start:]), 'sam_final_receipts_invalid', **_work_kwargs(work_budget))
            for phase, reference in (_work_items(zip(PHASES[start:], receipts), work_budget) if work_budget is not None else zip(PHASES[start:], receipts)):
                result = context.selected(reference, nested[1], {'sam_results'})
                if result and result[0].get('schema_version') == SCHEMAS['sam_results'][0]:
                    if result[0].get('status') not in {'completed', 'failed'}:
                        context.missing('sam_final_phase_result', 'unsupported_retained_status', [result[1]])
                        continue
                    c.require(result[0].get('status') == 'completed' and result[0].get('source_commit') == final['source_commit']
                        and result[0].get('plan_digest') == final['plan_digest']
                        and result[0].get('parent_request_digest') == value['request_digest']
                        and result[0].get('phase') == phase, 'sam_final_receipts_invalid', **_work_kwargs(work_budget))
                    _final_aliases(final, result[0]['artifacts'], **_work_kwargs(work_budget))
                    if phase == 'standard_splat_conversion':
                        _final_aliases(final, {'standard_splat_conversion': result[0]['artifacts'].get('standard_splat_conversion_receipt')}, **_work_kwargs(work_budget))
            observations.append(c.observation(nested, role='sam_final', plan_digest=final['plan_digest'],
                parent_binding_verified=len(selected_parents) == 1 and len(branches[(value['request_digest'], value['sequence'])]) == 1, **_work_kwargs(work_budget)))
        observations.append(c.observation(row, role='sam_source_progress', sequence=value['sequence'], progress_status=value['status'], **_work_kwargs(work_budget)))
    for row in (_work_items(context.sam_tables['source_resume_signals'], work_budget) if work_budget is not None else context.sam_tables['source_resume_signals']):
        value, proof = row
        c.require(c.matches(value.get('preparation_id'), c.ID, **_work_kwargs(work_budget)) and c.matches(value.get('request_digest'), **_work_kwargs(work_budget))
            and c.matches(value.get('progress_digest'), **_work_kwargs(work_budget)) and c.matches(value.get('source_commit'), c.COMMIT, **_work_kwargs(work_budget))
            and value.get('kind') in {'human_review', 'child_result'}, 'sam_resume_invalid', **_work_kwargs(work_budget))
        context.selected(value.get('evidence_ref'), proof)
        stem = value['preparation_id'] + '-' + value['request_digest'][7:]
        c.require(any(proof['path'] in (c.child(route['queue_root'], 'source-resume-pending', value['signal_digest'][7:] + '.json', **_work_kwargs(work_budget)),
            c.child(route['queue_root'], 'source-resume-completed', stem, value['signal_digest'][7:] + '.json', **_work_kwargs(work_budget))) for route in (_work_items(context.routes, work_budget) if work_budget is not None else context.routes)),
            'sam_resume_path_invalid', **_work_kwargs(work_budget))
        for prior in (_work_items(index.get((value['request_digest'], value['progress_digest']), []), work_budget) if work_budget is not None else index.get((value['request_digest'], value['progress_digest']), [])):
            c.require(prior[0]['source_commit'] == value['source_commit'] and prior[0]['preparation_id'] == value['preparation_id'],
                      'sam_resume_progress_invalid', **_work_kwargs(work_budget))
        observations.append(c.observation(row, role='sam_resume', wake_authorized=False, **_work_kwargs(work_budget)))


def _final_aliases(final, artifacts, *, original=False, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
    for name in (_work_items(EVIDENCE.intersection(artifacts), work_budget) if work_budget is not None else EVIDENCE.intersection(artifacts)):
        if original and name == 'standard_splat_conversion':
            continue  # An unavailable adoption can administratively rebind this.
        c.require(final['evidence'][name] == artifacts[name], 'sam_final_evidence_invalid', **_work_kwargs(work_budget))


def _final_adoption(context, declared, nested, final, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    references = declared['original_phase_result_receipts']
    end = PHASES.index(declared['through_phase']) + 1
    c.require(len(references) <= end, 'sam_final_adoption_invalid', **_work_kwargs(work_budget))
    known_results = {}
    for phase, reference in (_work_items(zip(PHASES[end-len(references):end], references), work_budget) if work_budget is not None else zip(PHASES[end-len(references):end], references)):
        result = context.selected(reference, nested[1], {'sam_results'})
        if result:
            if result[0].get('status') not in {'completed', 'failed'}:
                context.missing('sam_final_original_result', 'unsupported_retained_status', [result[1]])
                continue
            c.require(result[0]['status'] == 'completed' and result[0]['source_commit'] == declared['original_execution_commit']
                and result[0]['phase'] == phase, 'sam_final_original_result_invalid', **_work_kwargs(work_budget))
            _final_aliases(final, result[0]['artifacts'], original=True, **_work_kwargs(work_budget))
            known_results[tuple(reference[k] for k in (_work_items(('path', 'sha256', 'size_bytes'), work_budget) if work_budget is not None else ('path', 'sha256', 'size_bytes')))] = result
    selected = context.selected(declared.get('receipt'), nested[1], {'sam_adoptions'})
    if selected is None:
        return
    adoption, proof = selected
    if adoption.get('status') != 'verified_completed_prefix':
        context.missing('sam_final_adoption', 'unsupported_retained_status', [proof])
        return
    c.c.seal(selected, 'adoption_digest', **_work_kwargs(work_budget))
    phase_rows = adoption.get('phase_records')
    c.require(isinstance(phase_rows, list) and len(phase_rows) <= 10 and all(isinstance(r, dict) for r in (_work_items(phase_rows, work_budget) if work_budget is not None else phase_rows)),
              'sam_final_adoption_invalid', **_work_kwargs(work_budget))
    c.require(adoption.get('source_commit') == final['source_commit']
        and all(adoption.get(k) == declared[k] for k in (_work_items(('original_execution_commit', 'through_phase'), work_budget) if work_budget is not None else ('original_execution_commit', 'through_phase')))
        and declared['original_phase_result_receipts'] == [r.get('result') for r in (_work_items(phase_rows, work_budget) if work_budget is not None else phase_rows)], 'sam_final_adoption_invalid', **_work_kwargs(work_budget))
    artifacts = {}
    for phase in (_work_items(phase_rows, work_budget) if work_budget is not None else phase_rows):
        reference = phase['result']
        result = known_results.get(tuple(reference[k] for k in (_work_items(('path', 'sha256', 'size_bytes'), work_budget) if work_budget is not None else ('path', 'sha256', 'size_bytes'))))
        if result and result[0].get('status') == 'completed':
            artifacts.update(result[0]['artifacts'])
            if phase.get('phase') == 'standard_splat_conversion':
                artifacts['standard_splat_conversion'] = result[0]['artifacts'].get('standard_splat_conversion_receipt')
    rebindings = adoption.get('administrative_rebindings')
    c.require(isinstance(rebindings, dict), 'sam_final_adoption_invalid', **_work_kwargs(work_budget))
    for name in (_work_items(EVIDENCE.intersection(rebindings), work_budget) if work_budget is not None else EVIDENCE.intersection(rebindings)):
        c.require(isinstance(rebindings[name], dict), 'sam_final_adoption_invalid', **_work_kwargs(work_budget))
        artifacts[name] = rebindings[name].get('successor')
    _final_aliases(final, artifacts, **_work_kwargs(work_budget))


def _host_evidence(context, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    supported = {'public_scene_host_input_installation_receipt.v1', 'public_scene_source_preparation.v1',
                 'standard_splat_conversion_receipt.v1'}
    for row in (_work_items(context.decoded['sam_host_evidence'], work_budget) if work_budget is not None else context.decoded['sam_host_evidence']):
        _allowed(context, row, **_work_kwargs(work_budget))
        if row[0].get('schema_version') in supported:
            c.c.seal(row, 'receipt_digest', **_work_kwargs(work_budget))
            value, proof = row
            schema = value['schema_version']
            if schema == 'public_scene_host_input_installation_receipt.v1':
                c.require(value.get('status') == 'installed' and c.matches(value.get('scene_id'), c.ID, **_work_kwargs(work_budget))
                    and c.matches(value.get('packet_id'), c.ID, **_work_kwargs(work_budget)) and c.matches(value.get('source_commit_sha'), c.COMMIT, **_work_kwargs(work_budget))
                    and c.matches(value.get('packet_digest'), **_work_kwargs(work_budget)) and value.get('authoritative_request_digest') == value['packet_digest']
                    and value.get('destination_root') == proof['path'].rsplit('/', 1)[0]
                    and value.get('service_readable') is True and value.get('provider_mutation_performed') is False
                    and value.get('paid_resource_used') is False, 'sam_installation_invalid', **_work_kwargs(work_budget))
            elif schema == 'public_scene_source_preparation.v1':
                c.require(value.get('status') in {'blocked', 'source_context_prepared_pending_calibrated_views'}
                    and c.matches(value.get('source_commit'), c.COMMIT, **_work_kwargs(work_budget)) and c.matches(value.get('scene_id'), c.ID, **_work_kwargs(work_budget))
                    and c.matches(value.get('source_installation_digest'), **_work_kwargs(work_budget)) and all(value.get(k) is False for k in
                        (_work_items(('provider_mutation_performed', 'paid_resource_used', 'candidate_policy_queried'), work_budget) if work_budget is not None else ('provider_mutation_performed', 'paid_resource_used', 'candidate_policy_queried'))), 'sam_source_preparation_invalid', **_work_kwargs(work_budget))
                context.missing('sam_source_installation', 'canonical_installation_selector_without_raw_identity', [proof],
                                selector={'receipt_digest': value['source_installation_digest']})
            elif schema == 'standard_splat_conversion_receipt.v1':
                output = row[0].get('output')
                c.require(isinstance(output, dict) and c.matches(output.get('sha256'), **_work_kwargs(work_budget))
                    and type(output.get('size_bytes')) is int and output['size_bytes'] > 0, 'sam_conversion_invalid', **_work_kwargs(work_budget))
                c.relative(output.get('relative_path'), **_work_kwargs(work_budget))
                rights = value.get('rights')
                if isinstance(rights, dict) and 'terms_digest' in rights:
                    c.require(c.matches(rights['terms_digest'], **_work_kwargs(work_budget)), 'sam_conversion_terms_invalid', **_work_kwargs(work_budget))
                    context.missing('sam_conversion_terms', 'raw_terms_selector_without_size', [proof],
                                    selector={'sha256': rights['terms_digest']})
        else:
            context.missing('sam_host_evidence', 'unsupported_retained_schema', [row[1]])


def _ascii_digest(value, *, work_budget=None):
    # This producer uses stdlib sort_keys/ensure_ascii=True, distinct from
    # decision canonical seals. Feed bounded characters instead of encoding a
    # potentially six-times-amplified Unicode document as one allocation.
    if work_budget is not None:
        _work(work_budget)
    def tokens(item):
        if work_budget is not None:
            _work(work_budget)
        if isinstance(item, dict):
            yield '{'
            for index, key in (_work_items(enumerate(sorted(item)), work_budget) if work_budget is not None else enumerate(sorted(item))):
                if index:
                    yield ','
                yield from tokens(key)
                yield ':'
                yield from tokens(item[key])
            yield '}'
        elif isinstance(item, list):
            yield '['
            for index, child in (_work_items(enumerate(item), work_budget) if work_budget is not None else enumerate(item)):
                if index:
                    yield ','
                yield from tokens(child)
            yield ']'
        elif isinstance(item, str):
            yield '"'
            for char in (_work_items(item, work_budget) if work_budget is not None else item):
                yield (_work_call(work_budget, json.dumps, char, ensure_ascii=True) if work_budget is not None else json.dumps(char, ensure_ascii=True))[1:-1]
            yield '"'
        else:
            yield (_work_call(work_budget, json.dumps, item, allow_nan=False, ensure_ascii=True, separators=(',', ':')) if work_budget is not None else json.dumps(item, allow_nan=False, ensure_ascii=True, separators=(',', ':')))
    digest = (_work_hash(work_budget, hashlib.sha256, ) if work_budget is not None else hashlib.sha256())
    for token in (_work_items(tokens(value), work_budget) if work_budget is not None else tokens(value)):
        digest.update(token.encode('ascii'))
    return 'sha256:' + digest.hexdigest()


def _artifact_metadata(context, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    for row in (_work_items(context.decoded['sam_artifact_metadata'], work_budget) if work_budget is not None else context.decoded['sam_artifact_metadata']):
        _allowed(context, row, **_work_kwargs(work_budget))
        value, proof = row
        schema = value.get('schema_version')
        if schema == 'public_scene_sam31_task_input_packet.v1':
            c.c.seal(row, 'receipt_digest', **_work_kwargs(work_budget))
            c.require(value.get('status') == 'prepared_no_upload_no_execution'
                and value.get('paid_execution_started') is False
                and type(value.get('provider_mutations_performed')) is int and value['provider_mutations_performed'] == 0,
                'sam_packet_invalid', **_work_kwargs(work_budget))
            for name, seal in (_work_items((('task_freeze', 'task_freeze_digest'), ('calibrated_view_receipt', 'receipt_digest'),
                               ('provider_profile', 'profile_digest')), work_budget) if work_budget is not None else (('task_freeze', 'task_freeze_digest'), ('calibrated_view_receipt', 'receipt_digest'),
                               ('provider_profile', 'profile_digest'))):
                declared = value.get(name)
                context.selected(declared, proof)
                c.require(c.matches(declared.get(seal), **_work_kwargs(work_budget)), 'sam_packet_selector_invalid', **_work_kwargs(work_budget))
            run = value.get('run_request')
            c.require(isinstance(run, dict) and c.matches(run.get('request_digest'), **_work_kwargs(work_budget)), 'sam_packet_request_invalid', **_work_kwargs(work_budget))
            reference = {'path': c.child(proof['path'].rsplit('/', 1)[0], c.relative(run.get('relative_path'), **_work_kwargs(work_budget)), **_work_kwargs(work_budget)),
                         'sha256': run.get('sha256'), 'size_bytes': run.get('size_bytes')}
            request = context.selected(reference, proof, {'sam_artifact_metadata'})
            if request and request[0].get('schema_version') == 'semantic_sam31_source_track_run_request.v1':
                c.require(_ascii_digest(request[0], **_work_kwargs(work_budget)) == run['request_digest'], 'sam_packet_request_digest_invalid', **_work_kwargs(work_budget))
            elif request:
                context.missing('sam_run_request', 'unsupported_retained_schema', [request[1]])
        elif schema in {'semantic_sam31_source_track_run_request.v1', 'public_scene_interiorgs_edit_input_request.v2'}:
            # No invented universal signature, checkpoint or science validator.
            if schema == 'public_scene_interiorgs_edit_input_request.v2':
                scene = value.get('scene')
                c.require(isinstance(scene, dict), 'sam_render_request_invalid', **_work_kwargs(work_budget))
                for name in (_work_items(('scene_freeze_path', 'task_freeze_path', 'standard_splat_conversion_receipt_path',
                             'standard_splat_path', 'labels_path', 'structure_path', 'registered_frame_receipt_path'), work_budget) if work_budget is not None else ('scene_freeze_path', 'task_freeze_path', 'standard_splat_conversion_receipt_path',
                             'standard_splat_path', 'labels_path', 'structure_path', 'registered_frame_receipt_path')):
                    if name in scene:
                        c.path(scene[name], **_work_kwargs(work_budget))
                        context.missing('sam_renderer_input', 'bare_input_path_without_raw_identity', [proof],
                                        scene[name], {'field': name})
        else:
            context.missing('sam_artifact_metadata', 'unsupported_retained_schema', [proof])


def artifact_edges(context, artifacts, proof, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    packet_ref = artifacts.get('sam31_task_input_packet')
    if packet_ref:
        packet = context.selected(packet_ref, proof, {'sam_artifact_metadata', 'opaque_evidence'})
        if packet and packet[0] is not None and packet[0].get('schema_version') == 'public_scene_sam31_task_input_packet.v1':
            for name, alias in (_work_items((('task_freeze', 'task_selection'), ('calibrated_view_receipt', 'calibrated_view_receipt')), work_budget) if work_budget is not None else (('task_freeze', 'task_selection'), ('calibrated_view_receipt', 'calibrated_view_receipt'))):
                declared = packet[0][name]
                if alias in artifacts:
                    c.require(all(declared[k] == artifacts[alias][k] for k in (_work_items(('path', 'sha256', 'size_bytes'), work_budget) if work_budget is not None else ('path', 'sha256', 'size_bytes'))), 'sam_packet_original_alias_invalid', **_work_kwargs(work_budget))
                else:
                    context.missing('sam_packet_original_alias', 'original_artifact_selector_unavailable', [packet[1]], selector={'alias': alias})
            if 'sam31_run_request' in artifacts:
                run = packet[0]['run_request']
                expected = {'path': c.child(packet[1]['path'].rsplit('/', 1)[0], c.relative(run['relative_path'], **_work_kwargs(work_budget)), **_work_kwargs(work_budget)),
                            'sha256': run['sha256'], 'size_bytes': run['size_bytes']}
                c.require(artifacts['sam31_run_request'] == expected, 'sam_packet_original_request_invalid', **_work_kwargs(work_budget))
