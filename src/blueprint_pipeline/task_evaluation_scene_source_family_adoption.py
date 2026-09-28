"""Bounded retained prefix graph; original evidence is protected, never owned anew."""
from __future__ import annotations

from . import task_evaluation_scene_source_family_contracts as c
from . import task_evaluation_scene_source_family_sam as sam

SCHEMA = 'task_evaluation_sam31_completed_prefix_adoption.v1'
REBIND_NAMES = {'standard_splat', 'standard_splat_conversion_receipt', 'standard_splat_conversion'}
CONTRACT_REVISIONS = {
    'scene_preparation_25200_single_pass.v1': '1df1785e48220633e90b507ef68b04929ef74103',
    'scene_preparation_25200_repair.v1': '6bae36660c460c3115ce17fe6352145395e50e8f',
    'scene_preparation_27000_repair.v1': 'ac689e03ab6a7c6fb598f855d4f5bc37b4f87d43',
    'scene_preparation_27000_astra.v1': '',
}
FROZEN_SCHEMAS = {
    (False, False): ('task_evaluation_retained_scene_preparation.v1.schema.json', '8d1f6826901b7e4fdbc486ec83ee4de7fa1dffcda4b3060c82b464577866da8c'),
    (False, True): ('task_evaluation_retained_astra_scene_preparation.v1.schema.json', 'f7943b8db04d71edfdc7d8d1a1fd46043f505e8b166c7efd040019d3a60e85b6'),
    (True, False): ('task_evaluation_retained_source_scene_preparation.v1.schema.json', '95900c65b3b7dd5058d4583bb0355535667cd5d999d502d1c0263949a03f74d4'),
    (True, True): ('task_evaluation_retained_source_astra_scene_preparation.v1.schema.json', 'e7dc3d7b95471db1e601227e2927ebb73d04e3350246ec48a53d41e09394c30b'),
}


def _raw_key(row):
    return (*[row[1][k] for k in ('path', 'sha256', 'size_bytes')], row[1].get('json_pointer'))


def _raw_tuple(reference):
    return {key: reference[key] for key in ('path', 'sha256', 'size_bytes')}


def _selected(context, reference, proof, role, schema):
    row = context.selected(reference, proof, {role})
    if row and role == 'sam_adoptions' and row[0].get('status') != 'verified_completed_prefix':
        context.missing(role, 'unsupported_retained_status', [row[1]])
        return None
    if row and role == 'sam_results' and row[0].get('status') not in {'completed', 'failed'}:
        context.missing(role, 'unsupported_retained_status', [row[1]])
        return None
    if row and row[0].get('schema_version') == schema:
        return row
    if row:
        context.missing(role, 'unsupported_retained_schema', [row[1]])
    return None


class Graph:
    """Memoized within this call only; active identities and incoming edges differ."""
    def __init__(self, context):
        self.context = context
        self.active, self.memo, self.visited = set(), {}, set()
        self.phases = context.rows()

    def visit(self, row, depth=1):
        context = self.context
        key = _raw_key(row)
        c.require(key not in self.active, 'adoption_cycle')
        c.require(depth <= context.limits['MAX_ADOPTION_DEPTH'], 'adoption_depth_limit')
        if key in self.memo:
            return self.memo[key]
        c.require(len(self.visited) < context.limits['MAX_ADOPTION_NODES'], 'adoption_nodes_limit')
        self.visited.add(key)
        self.active.add(key)
        try:
            result = self._visit(row, depth)
            self.memo[key] = result
            return result
        finally:
            self.active.remove(key)

    def _visit(self, row, depth):
        context, value, proof = self.context, row[0], row[1]
        c.require(value.get('status') == 'verified_completed_prefix' and value.get('through_phase') in sam.PHASES[2:]
            and c.matches(value.get('source_commit'), c.COMMIT) and c.matches(value.get('original_execution_commit'), c.COMMIT)
            and c.matches(value.get('original_parent_request_digest')) and all(value.get(k) is False for k in
                ('historical_receipts_modified', 'paid_execution_performed', 'candidate_policy_queried')), 'adoption_invalid')
        end = sam.PHASES.index(value['through_phase']) + 1
        plan = _selected(context, value.get('source_plan'), proof, 'sam_plans', sam.SCHEMAS['sam_plans'][0])
        profile = _selected(context, value.get('source_profile'), proof, 'sam_profiles', sam.SCHEMAS['sam_profiles'][0])
        parent = context.selected(value.get('original_parent_envelope'), proof, {'sam_parent_envelopes', 'preparation_envelopes'})
        if parent and (parent[0].get('schema_version') != sam.SCHEMAS['sam_parent_envelopes'][0] or sam._parent(context, parent) is None):
            parent = None
        current_host = value.get('current_host_inputs')
        c.require(isinstance(current_host, dict) and set(current_host) == sam.HOST_NAMES, 'adoption_current_host_invalid')
        sam.refs(context, current_host, proof)
        task_key = tuple(current_host['task_request'][k] for k in ('path', 'sha256', 'size_bytes'))
        current_plans = [plan for plan in context.sam_current_plans.get(task_key, [])
                         if plan[0]['source_commit'] == value['source_commit']]
        for current_plan in current_plans:
            c.require(current_plan[0]['source_commit'] == value['source_commit']
                and current_plan[0]['host_inputs'] == current_host, 'adoption_current_parent_plan_invalid')
        if not current_plans:
            context.missing('sam_current_parent_plan', 'owner_parent_plan_selector_unavailable', [proof])
        context.selected(value.get('current_sam31_provider_profile'), proof)
        context.selected(value.get('provider_zero_at_adoption'), proof)
        if 'current_release_root' in value:
            c.path(value['current_release_root'])
            context.missing('sam_current_release', 'current_release_validity_unverified', [proof],
                            value['current_release_root'], {'source_commit': value['source_commit']})
        _retained_selectors(context, row)
        if plan:
            c.require(plan[0]['source_commit'] == value['original_execution_commit'], 'adoption_original_commit_invalid')
        if profile:
            c.require(profile[0]['source_commit'] == value['original_execution_commit'], 'adoption_original_commit_invalid')
            if 'repo_root' in profile[0]:
                context.missing('sam_source_release', 'historical_release_validity_unverified', [profile[1]],
                    c.path(profile[0]['repo_root']), {'source_commit': profile[0]['source_commit']})
        if plan and profile:
            c.require(plan[0]['server_profile_sha256'] == profile[1]['sha256'], 'adoption_original_profile_invalid')
        if parent:
            request = parent[0]['request']
            c.require(parent[0]['request_digest'] == value['original_parent_request_digest']
                and request['expected_production_commit'] == value['original_execution_commit'], 'adoption_original_parent_invalid')
            if plan:
                sam._plan_parent(context, plan, parent)
        contract_complete = _historical_contract(context, value.get('historical_parent_contract'), parent, proof)
        inherited, predecessor_missing = None, False
        if profile and profile[0].get('completed_prefix_adoption') is not None:
            predecessor = _selected(context, profile[0]['completed_prefix_adoption'], profile[1], 'sam_adoptions', SCHEMA)
            if predecessor:
                inherited = self.visit(predecessor, depth + 1)
                c.require(predecessor[0]['source_commit'] == value['original_execution_commit'], 'adoption_inherited_commit_invalid')
            else:
                predecessor_missing = True
        inputs, artifacts = {}, {}
        if plan:
            inputs.update(plan[0]['host_inputs'])
        if profile:
            _extend(inputs, profile[0]['artifact_references'])
        if inherited:
            _extend(inputs, inherited['successor_artifacts'])
            artifacts.update(inherited['successor_artifacts'])
        complete = bool(plan and profile and parent and contract_complete and not predecessor_missing and (not inherited or inherited['complete']))
        inputs_available = bool(plan and profile and not predecessor_missing and
                                (not inherited or inherited['inputs_available']))
        start = inherited['phase_count'] if inherited else 0
        phase_rows = value.get('phase_records')
        c.require(isinstance(phase_rows, list) and 1 <= len(phase_rows) <= 10
            and all(isinstance(r, dict) and set(r) == {'phase', 'job', 'result', 'execution_receipt'} for r in phase_rows), 'adoption_phase_rows_invalid')
        phase_names = [r['phase'] for r in phase_rows]
        c.require(all(name in sam.PHASES for name in phase_names) and len(set(phase_names)) == len(phase_names), 'adoption_phase_rows_invalid')
        if profile and not predecessor_missing:
            c.require(start < end and phase_names == list(sam.PHASES[start:end]), 'adoption_phase_order_invalid')
        else:
            c.require(phase_names == list(sam.PHASES[end - len(phase_rows):end]), 'adoption_phase_order_invalid')
        for phase_row in phase_rows:
            complete, inputs_available = self._phase(row, phase_row, parent, inputs, artifacts, complete, inputs_available)
        rebindings, successors = _rebindings(context, row, artifacts, current_host)
        selection = inherited['selection_origin'] if inherited else {
            'task_request': plan[0]['host_inputs']['task_request'] if plan else None,
            'source_commit': value['original_execution_commit'], 'source_plan': _raw_tuple(value['source_plan']),
            'source_profile': _raw_tuple(value['source_profile']), 'parent_request_digest': value['original_parent_request_digest']}
        tracking = inherited['tracking_origin'] if inherited and inherited['phase_count'] >= 5 else {
            'source_commit': value['original_execution_commit'], 'source_profile': _raw_tuple(value['source_profile'])}
        if not complete:
            context.missing('sam_original_prefix', 'prefix_proof_unavailable_or_unresolved', [proof])
        return {'complete': complete, 'inputs_available': inputs_available, 'phase_count': end, 'original_artifacts': artifacts,
            'successor_artifacts': {**artifacts, **successors}, 'selection_origin': selection,
            'tracking_origin': tracking, 'rebindings': rebindings}

    def _phase(self, adoption, phase_row, parent, inputs, artifacts, complete, inputs_available):
        context, value, proof = self.context, adoption[0], adoption[1]
        phase = phase_row['phase']
        job = _selected(context, phase_row['job'], proof, 'sam_jobs', sam.SCHEMAS['sam_jobs'][0])
        result = _selected(context, phase_row['result'], proof, 'sam_results', sam.SCHEMAS['sam_results'][0])
        receipt = _selected(context, phase_row['execution_receipt'], proof, 'sam_execution_receipts', 'task_evaluation_sam31_phase_execution_receipt.v1')
        if job:
            c.require(job[0]['phase'] == phase and job[0]['parent_request_digest'] == value['original_parent_request_digest']
                and job[0]['expected_source_commit'] == value['original_execution_commit']
                and job[0]['plan_ref'] == value['source_plan'], 'adoption_job_invalid')
            c.require(job[1]['path'] == c.child(context.roots['sam_queue_root'], 'completed', job[0]['child_id'] + '.json'), 'adoption_job_path_invalid')
            if parent:
                c.require(job[0]['parent_preparation_id'] == parent[0]['request']['preparation_id'], 'adoption_job_parent_invalid')
            c.require(all(job[0]['inputs'].get(name) == reference for name, reference in inputs.items()),
                      'adoption_job_inputs_invalid')
            if inputs_available:
                c.require(job[0]['inputs'] == inputs, 'adoption_job_inputs_invalid')
        if result:
            c.require(result[0]['status'] == 'completed' and result[0]['phase'] == phase
                and result[0]['parent_request_digest'] == value['original_parent_request_digest']
                and result[0]['plan_digest'] == value['source_plan']['sha256']
                and result[0]['source_commit'] == value['original_execution_commit'], 'adoption_result_invalid')
            c.require(result[1]['path'] == c.child(context.roots['sam_queue_root'], 'results', result[0]['child_id'] + '.json'), 'adoption_result_path_invalid')
            if job:
                c.require(result[0]['child_id'] == job[0]['child_id'] and result[0]['job_digest'] == job[0]['job_digest'], 'adoption_result_job_invalid')
        if receipt:
            c.require(receipt[0]['phase'] == phase and receipt[0]['source_commit'] == value['original_execution_commit']
                and receipt[0]['outcome']['status'] == 'completed', 'adoption_receipt_invalid')
            for selected in (job, result):
                if selected:
                    c.require(receipt[0]['job_digest'] == selected[0]['job_digest']
                        and receipt[1]['path'] == c.child(context.roots['sam_execution_root'],
                            value['original_parent_request_digest'][7:], selected[0]['child_id'], 'phase_execution_receipt.v1.json'),
                        'adoption_receipt_job_invalid')
            if result:
                c.require(receipt[0]['outcome']['artifacts'] == result[0]['artifacts']
                    and receipt[0]['job_digest'] == result[0]['job_digest'], 'adoption_receipt_result_invalid')
        complete = complete and bool(job and result and receipt)
        if result:
            _extend(inputs, result[0]['artifacts'])
            _extend(artifacts, result[0]['artifacts'])
            if phase == 'standard_splat_conversion':
                c.require('standard_splat_conversion_receipt' in result[0]['artifacts'], 'adoption_conversion_alias_invalid')
                _extend(inputs, {'standard_splat_conversion': result[0]['artifacts']['standard_splat_conversion_receipt']})
                _extend(artifacts, {'standard_splat_conversion': result[0]['artifacts']['standard_splat_conversion_receipt']})
            sam.artifact_edges(context, artifacts, proof)
        self.phases.append(c.observation(adoption, role='sam_original_phase', phase=phase,
            phase_binding_verified=bool(job and result and receipt and inputs_available),
            selected_provenance=[r[1] for r in (job, result, receipt) if r], original_owner_transfer_authorized=False))
        # A missing parent/receipt does not erase a supplied input map. A missing
        # phase result does: its unknown additions cannot be inferred as empty.
        return complete, inputs_available and result is not None


def _extend(target, additions):
    for name, reference in additions.items():
        c.require(name not in target or target[name] == reference, 'adoption_artifact_conflict')
        target[name] = reference


def _retained_selectors(context, row):
    value, proof = row
    release = value.get('retained_release_pin')
    if release is not None:
        c.require(isinstance(release, dict) and c.matches(release.get('source_commit'), c.COMMIT)
            and c.matches(release.get('tree'), c.COMMIT), 'adoption_retained_release_invalid')
        context.missing('sam_retained_release', 'historical_release_validity_unverified', [proof], c.path(release.get('path')),
                        {k: release[k] for k in ('source_commit', 'tree')})
    tracking = value.get('tracking_identity')
    if tracking is not None:
        c.require(isinstance(tracking, dict), 'adoption_tracking_identity_invalid')
        instance = tracking.get('provider_instance_id')
        c.require(type(instance) is int and instance >= 0 or isinstance(instance, str) and 0 < len(instance) <= 192,
                  'adoption_tracking_instance_invalid')
        c.require(c.matches(tracking.get('checkpoint_digest')), 'adoption_tracking_checkpoint_invalid')
        context.selected(tracking.get('raw_runtime_result'), proof)
        charge = tracking.get('official_charge')
        c.require(isinstance(charge, dict), 'adoption_tracking_charge_invalid')
        context.selected(charge.get('provider_billing_source_receipt'), proof)
        context.missing('sam_tracking_identity', 'provider_checkpoint_billing_validity_unverified', [proof], selector={
            'provider_instance_id': instance, 'checkpoint_digest': tracking['checkpoint_digest']})


def _rebindings(context, row, artifacts, current_host):
    value, proof = row
    rebindings = value.get('administrative_rebindings')
    c.require(isinstance(rebindings, dict) and set(rebindings) == REBIND_NAMES, 'adoption_rebinding_invalid')
    successors = {}
    for name, rebinding in rebindings.items():
        c.require(isinstance(rebinding, dict) and set(rebinding) == {'original', 'successor'}, 'adoption_rebinding_invalid')
        context.selected(rebinding['original'], proof)
        context.selected(rebinding['successor'], proof)
        if name in artifacts:
            c.require(rebinding['original'] == artifacts[name], 'adoption_rebinding_original_invalid')
        successors[name] = _raw_tuple(rebinding['successor'])
    c.require(successors['standard_splat_conversion_receipt'] == successors['standard_splat_conversion'], 'adoption_conversion_alias_invalid')
    task = _selected(context, current_host['task_request'], proof, 'sam_host_tasks', sam.SCHEMAS['sam_host_tasks'][0])
    if task:
        if 'expected_production_commit' in task[0]:
            c.require(task[0]['expected_production_commit'] == value['source_commit'], 'adoption_current_task_commit_invalid')
        declared = task[0].get('source_input_references', {}).get('standard_splat_conversion_receipt')
        if declared is not None:
            c.require(declared == successors['standard_splat_conversion_receipt'], 'adoption_current_conversion_invalid')
    conversion = _selected(context, successors['standard_splat_conversion_receipt'], proof, 'sam_host_evidence', 'standard_splat_conversion_receipt.v1')
    if conversion:
        output = conversion[0]['output']
        expected = {'path': c.child(conversion[1]['path'].rsplit('/', 1)[0], c.relative(output['relative_path'])),
                    'sha256': output['sha256'], 'size_bytes': output['size_bytes']}
        c.require(successors['standard_splat'] == expected, 'adoption_current_standard_invalid')
    original_renderer = artifacts.get('calibrated_view_request')
    if original_renderer:
        renderer = context.selected(original_renderer, proof)
        if renderer and renderer[1]['role'] == 'sam_artifact_metadata':
            path = renderer[0].get('scene', {}).get('standard_splat_path')
            if path is not None and path != artifacts.get('standard_splat', {}).get('path'):
                c.path(path)
                context.missing('sam_original_render_source', 'original_render_source_selector_unresolved', [renderer[1]], path)
    return {name: {role: _raw_tuple(reference) for role, reference in rebinding.items()}
            for name, rebinding in rebindings.items()}, successors


def _historical_contract(context, value, parent, proof):
    if value is None:
        return True
    c.require(isinstance(value, dict), 'historical_contract_invalid')
    schema = value.get('schema_version')
    if schema not in {'task_evaluation_retained_preparation_contract_identity.v1', 'task_evaluation_retained_preparation_contract_identity.v2'}:
        context.missing('sam_historical_parent_contract', 'unsupported_retained_schema', [proof])
        return False
    context.nested(value, proof, '/historical_parent_contract', 'contract_digest')
    contract = value.get('contract_id')
    if contract not in CONTRACT_REVISIONS:
        context.missing('sam_historical_parent_contract', 'unsupported_retained_contract', [proof])
        return False
    c.require(c.matches(value.get('request_source_commit'), c.COMMIT) and c.matches(value.get('schema_sha256'))
        and value.get('request_schema_version') == 'task_evaluation_launch_preparation_request.v1', 'historical_contract_invalid')
    if parent:
        c.require(value['request_source_commit'] == parent[0]['request']['expected_production_commit'], 'historical_contract_parent_invalid')
    if schema.endswith('.v1'):
        c.require(contract != 'scene_preparation_27000_astra.v1'
            and value.get('schema_sha256') == 'sha256:' + FROZEN_SCHEMAS[(False, False)][1]
            and value.get('policy_source_revision') == CONTRACT_REVISIONS[contract]
            and value.get('policy_source_path') == 'src/blueprint_pipeline/task_evaluation_scene_configuration_runtime_budget.py',
            'historical_contract_identity_invalid')
    else:
        c.require(value.get('new_execution_authorized') is False and value.get('new_spend_authorized') is False, 'historical_contract_authority_invalid')
        candidates = {('docs/schemas/' + name, 'sha256:' + digest) for key, (name, digest) in FROZEN_SCHEMAS.items()
                      if key != (False, False)}
        c.require((value.get('policy_source_path'), value['schema_sha256']) in candidates, 'historical_contract_identity_invalid')
    if parent:
        request = parent[0]['request']
        geometry = request.get('scene', {}).get('geometry')
        source = (isinstance(geometry, dict) and geometry.get('source_derivation') is not None
                  or request.get('task', {}).get('surface_target') is not None)
        astra = request.get('replacement_authoring_backend') == 'astra_cad_blender_v1'
        name, digest = FROZEN_SCHEMAS[(bool(source), astra)]
        c.require(value['schema_sha256'] == 'sha256:' + digest and schema.endswith('.v2') == (source or astra),
                  'historical_contract_family_invalid')
        spend = request.get('spend')
        if isinstance(spend, dict):
            ttl = spend.get('hard_ttl_seconds')
            if type(ttl) is int and ttl in {25200, 27000}:
                expected = 'scene_preparation_27000_astra.v1' if ttl == 27000 and astra else 'scene_preparation_27000_repair.v1'
                if ttl == 25200:
                    cap = spend.get('external_service_caps', {}).get('openai', {}).get('maximum_cost_usd')
                    c.require(type(cap) in {int, float}, 'historical_contract_budget_identity_invalid')
                    expected = 'scene_preparation_25200_single_pass.v1' if cap <= 3 else 'scene_preparation_25200_repair.v1'
                c.require(contract == expected, 'historical_contract_budget_identity_invalid')
            else:
                context.missing('sam_historical_parent_contract', 'historical_budget_identity_unavailable', [proof])
                return False
        else:
            context.missing('sam_historical_parent_contract', 'historical_budget_identity_unavailable', [proof])
            return False
    return parent is not None


def inventory(context):
    supplied = context.known('sam_adoptions', SCHEMA, 'adoption_digest')
    known = []
    for row in supplied:
        if row[0].get('status') == 'verified_completed_prefix':
            known.append(row)
        else:
            context.missing('sam_adoption', 'unsupported_retained_status', [row[1]])
    graph, observations = Graph(context), context.rows()
    for row in known:
        sam._allowed(context, row)
        result = graph.visit(row)
        observations.append(c.observation(row, role='sam_adoption', prefix_binding_verified=result['complete'],
            phase_count=result['phase_count'], original_artifacts=result['original_artifacts'],
            successor_artifacts=result['successor_artifacts'], selection_origin=result['selection_origin'],
            tracking_origin=result['tracking_origin'], rebindings=result['rebindings']))
    _prefix_selections(context, graph, known)
    context.adoption_graph = graph
    return observations, graph.phases


def _prefix_selections(context, graph, adoptions):
    schema = 'task_evaluation_sam31_prefix_selection.v1'
    selected_factories = {}
    for factory in context.decoded['factories']:
        reference = factory[0].get('prefix_selection')
        if reference is not None:
            row = context.selected(reference, factory[1], {'sam_prefix_selections'})
            if row:
                c.require(row[1]['path'] == c.child(factory[1]['path'].rsplit('/', 1)[0], 'materialized', 'prefix_selection.json'),
                          'prefix_selection_factory_path_invalid')
                if factory[1]['path'].rsplit('/', 1)[0] in context.source_owner_workspaces:
                    selected_factories.setdefault(_raw_key(row), []).append(factory[1])
                else:
                    context.missing('sam_prefix_factory_owner', 'factory_owner_join_unavailable', [factory[1], row[1]])
    by_seal = {}
    for adoption in adoptions:
        by_seal.setdefault(adoption[0]['adoption_digest'], []).append(adoption)
    for row in context.known('sam_prefix_selections', schema, 'selection_digest'):
        sam._allowed(context, row)
        value, proof = row
        c.require(value.get('paid_execution_performed') is False, 'prefix_selection_authority_invalid')
        if value.get('status') not in {'reusable_prefix_selected', 'no_reusable_prefix'}:
            context.missing('sam_prefix_selection', 'unsupported_retained_status', [proof])
            continue
        for field in ('rejected_candidates', 'candidate_selections'):
            candidates = value.get(field, [])
            c.require(isinstance(candidates, list) and len(candidates) <= 1024
                and all(isinstance(candidate, dict) for candidate in candidates), 'prefix_selection_candidates_invalid')
            for index, candidate in enumerate(candidates):
                if 'through_phase' in candidate:
                    c.require(candidate['through_phase'] in sam.PHASES[2:], 'prefix_selection_candidates_invalid')
                # Human/provider blockers remain private; only fixed phase and
                # enclosing JSON-pointer provenance leave this boundary.
                context.sam_observations.append(c.observation(row, role='sam_prefix_candidate',
                    through_phase=candidate.get('through_phase'), candidate_pointer=f'/{field}/{index}'))
        nested = None
        if value['status'] == 'reusable_prefix_selected':
            adoption = value.get('adoption')
            c.require(isinstance(adoption, dict) and value.get('through_phase') in sam.PHASES[2:], 'prefix_selection_adoption_invalid')
            if adoption.get('schema_version') == SCHEMA and adoption.get('status') == 'verified_completed_prefix':
                nested = context.nested(adoption, proof, '/adoption', 'adoption_digest')
                c.require(adoption.get('through_phase') == value['through_phase'], 'prefix_selection_adoption_invalid')
                copies = by_seal.get(adoption['adoption_digest'], [])
                for copy in copies:
                    c.require(copy[0] == adoption, 'prefix_selection_adoption_invalid')
                if copies:
                    # Same document proof, distinct enclosing/raw provenance.
                    # The already validated verdict is safe to reuse; retain
                    # this incoming pointer separately, never invent raw bytes.
                    graph.visit(copies[0])
                else:
                    graph.visit(nested)
            else:
                context.missing('sam_prefix_adoption', 'unsupported_retained_schema', [proof])
        context.sam_observations.append(c.observation(row, role='sam_prefix_selection',
            factory_binding_verified=len(selected_factories.get(_raw_key(row), [])) == 1,
            nested_adoption_provenance=nested[1] if nested else None, prefix_selection_status=value['status']))
