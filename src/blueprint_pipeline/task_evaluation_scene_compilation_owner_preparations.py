"""Native preparation/compilation handoff metadata; no payload reads."""
from __future__ import annotations

from .task_evaluation_scene_lineage_budget import _work_collect, _work, _work_call, _work_items, _work_kwargs

from pathlib import PurePosixPath

from . import task_evaluation_scene_compilation_owner_contracts as c

STATES = {'pending', 'processing', 'awaiting_source_preparation', 'awaiting_capacity', 'materialized', 'completed', 'blocked'}
BASE = c.PREPARATION_BASE_FIELDS
HANDOFF = c.PREPARATION_HANDOFF_FIELDS
PRE = 'inputs_materialized_awaiting_construction_adapter'
FINAL = 'queued_for_production_episode_compilation'


def _route(context, path, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    parent = str(PurePosixPath(path).parent.parent)
    matched = [r for r in (_work_items(context.routes, work_budget) if work_budget is not None else context.routes) if parent == r['queue_root']]
    c.require(len(matched) == 1, 'preparation_route_invalid', **_work_kwargs(work_budget))
    return matched[0]


def _references(context, rows, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    c.require(isinstance(rows, list) and 1 <= len(rows) <= context.limits['MAX_REFERENCES'], 'references_invalid', **_work_kwargs(work_budget))
    identities, contracts, reuse = set(), set(), 0
    for row in (_work_items(rows, work_budget) if work_budget is not None else rows):
        c.require(isinstance(row, dict) and {'contract_path', 'uri', 'digest', 'size_bytes', 'materialized_path',
            'content_addressed_reuse', 'full_byte_service_account_readback_passed'} <= (_work_collect(work_budget, set, row) if work_budget is not None else set(row)), 'reference_invalid', **_work_kwargs(work_budget))
        name = c.text(row['contract_path'], 512, **_work_kwargs(work_budget))
        c.require(not any(ch.isspace() for ch in (_work_items(name, work_budget) if work_budget is not None else name)) and name not in contracts, 'reference_contract_invalid', **_work_kwargs(work_budget))
        contracts.add(name)
        c.require(isinstance(row['uri'], str) and len(row['uri']) <= 4096 and '://' in row['uri']
            and not any(ch.isspace() for ch in (_work_items(row['uri'], work_budget) if work_budget is not None else row['uri'])) and c.matches(row['digest'], **_work_kwargs(work_budget))
            and type(row['size_bytes']) is int and row['size_bytes'] > 0
            and type(row['content_addressed_reuse']) is bool and row['full_byte_service_account_readback_passed'] is True,
            'reference_invalid', **_work_kwargs(work_budget))
        c.path(row['materialized_path'], **_work_kwargs(work_budget))
        c.require(any(c.under(row['materialized_path'], route['input_root'], **_work_kwargs(work_budget)) for route in (_work_items(context.routes, work_budget) if work_budget is not None else context.routes)), 'reference_path_invalid', **_work_kwargs(work_budget))
        context.size(row['digest'], row['size_bytes'])
        identities.add((row['digest'], row['size_bytes']))
        reuse += int(row['content_addressed_reuse'])
    return len(identities), reuse


def _envelopes(context, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    index = {}
    for role in (_work_items(('native_preparation_envelopes', 'sam_parent_envelopes', 'preparation_envelopes'), work_budget) if work_budget is not None else ('native_preparation_envelopes', 'sam_parent_envelopes', 'preparation_envelopes')):
        for row in (_work_items(context.known(role, 'task_evaluation_launch_preparation_envelope.v1', 'envelope_digest'), work_budget) if work_budget is not None else context.known(role, 'task_evaluation_launch_preparation_envelope.v1', 'envelope_digest')):
            value, proof = row
            request = value.get('request')
            c.require(isinstance(request, dict), 'preparation_request_invalid', **_work_kwargs(work_budget))
            if request.get('run_mode') not in {'episode_evaluation', 'destination_qualification'}:
                if role == 'native_preparation_envelopes':
                    context.missing(role, 'unsupported_retained_request', [proof])
                continue
            c.require(request.get('schema_version') == 'task_evaluation_launch_preparation_request.v1'
                and all(c.matches(request.get(k), c.ID, **_work_kwargs(work_budget)) for k in (_work_items(('preparation_id', 'run_id', 'team_namespace'), work_budget) if work_budget is not None else ('preparation_id', 'run_id', 'team_namespace')))
                and c.matches(request.get('expected_production_commit'), c.COMMIT, **_work_kwargs(work_budget))
                and c.matches(value.get('request_digest'), **_work_kwargs(work_budget)) and value['request_digest'] == (_work_call(work_budget, c.canonical_digest, request) if work_budget is not None else c.canonical_digest(request))
                and request.get('construction', {}).get('mode') == 'reuse_configured_scene'
                and request.get('task', {}).get('binding_mode') == 'reuse_configured_template', 'preparation_request_invalid', **_work_kwargs(work_budget))
            context.fields(row, c.FIELD_SETS['native_preparation_envelopes'])
            context.fields((request, dict(proof, json_pointer='/request')), c.NATIVE_PREPARATION_REQUEST_FIELDS)
            c.require(all(field not in value or value[field] is False for field in
                (_work_items(('provider_mutation_performed_inside_intake', 'catalog_mutation_performed_inside_intake'), work_budget) if work_budget is not None else ('provider_mutation_performed_inside_intake', 'catalog_mutation_performed_inside_intake'))), 'preparation_intake_scope_invalid', **_work_kwargs(work_budget))
            stem = request['preparation_id']+'-'+value['request_digest'][7:]+'.json'
            route = _route(context, proof['path'], **_work_kwargs(work_budget))
            c.require(proof['path'] in {c.child(route['queue_root'], state, stem, **_work_kwargs(work_budget)) for state in (_work_items(STATES, work_budget) if work_budget is not None else STATES)}, 'preparation_path_invalid', **_work_kwargs(work_budget))
            index.setdefault((request['preparation_id'], value['request_digest']), []).append(row)
    return index


def _compilations(context, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    index = {}
    for row in (_work_items(context.known('compilation_envelopes', 'task_evaluation_episode_compilation_envelope.v1', 'envelope_digest'), work_budget) if work_budget is not None else context.known('compilation_envelopes', 'task_evaluation_episode_compilation_envelope.v1', 'envelope_digest')):
        value, proof = row
        c.require(c.matches(value.get('compilation_id'), c.ID, **_work_kwargs(work_budget)) and value.get('preparation_id') == value['compilation_id']
            and all(c.matches(value.get(k), c.ID, **_work_kwargs(work_budget)) for k in (_work_items(('run_id', 'team_namespace'), work_budget) if work_budget is not None else ('run_id', 'team_namespace')))
            and c.matches(value.get('expected_production_commit'), c.COMMIT, **_work_kwargs(work_budget))
            and c.matches(value.get('configured_scene_revision_digest'), **_work_kwargs(work_budget)) and c.matches(value.get('preparation_result_digest'), **_work_kwargs(work_budget))
            and all(value.get(k) is True for k in (_work_items(('automatic_progression_required', 'robot_specific_episode_packet_compiled_in_production',
                'production_compiler_owns_episode_packet'), work_budget) if work_budget is not None else ('automatic_progression_required', 'robot_specific_episode_packet_compiled_in_production',
                'production_compiler_owns_episode_packet')))
            and all(value.get(k) is False for k in (_work_items(('customer_supplied_prebuilt_episode_packet', 'provider_mutation_performed', 'paid_execution_requested'), work_budget) if work_budget is not None else ('customer_supplied_prebuilt_episode_packet', 'provider_mutation_performed', 'paid_execution_requested'))),
            'compilation_envelope_invalid', **_work_kwargs(work_budget))
        request = value.get('request')
        c.require(isinstance(request, dict) and request.get('preparation_id') == value['compilation_id']
            and request.get('run_id') == value['run_id'] and request.get('team_namespace') == value['team_namespace']
            and request.get('expected_production_commit') == value['expected_production_commit'], 'compilation_request_invalid', **_work_kwargs(work_budget))
        _references(context, value.get('materialized_references'), **_work_kwargs(work_budget))
        bundle = value.get('configured_scene_bundle')
        c.require(isinstance(bundle, dict) and any(bundle == r for r in (_work_items(value['materialized_references'], work_budget) if work_budget is not None else value['materialized_references']))
            and bundle.get('contract_path') == 'scene.configured_revision.configured_scene_bundle', 'compilation_bundle_invalid', **_work_kwargs(work_budget))
        p = PurePosixPath(proof['path'])
        c.require(str(p.parent.parent) == context.roots['compilation_queue_root'] and p.parent.name in {'pending', 'processing', 'completed', 'blocked'}
            and p.name == value['compilation_id']+'-'+value['envelope_digest'][7:]+'.json', 'compilation_path_invalid', **_work_kwargs(work_budget))
        index.setdefault((value['compilation_id'], value['envelope_digest']), []).append(row)
    return index


def _receipts(context, compilations, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    index = {}
    for row in (_work_items(context.known('compilation_intake_receipts'), work_budget) if work_budget is not None else context.known('compilation_intake_receipts')):
        context.metadata(row)
        value, proof = row
        c.require(value.get('status') == FINAL and c.matches(value.get('compilation_id'), c.ID, **_work_kwargs(work_budget))
            and c.matches(value.get('run_id'), c.ID, **_work_kwargs(work_budget)) and c.matches(value.get('configured_scene_revision_digest'), **_work_kwargs(work_budget))
            and c.matches(value.get('envelope_digest'), **_work_kwargs(work_budget)) and type(value.get('created')) is bool
            and value.get('automatic_progression_required') is True
            and value.get('provider_mutation_performed') is False and value.get('paid_execution_requested') is False, 'intake_invalid', **_work_kwargs(work_budget))
        c.path(value.get('queue_path'), **_work_kwargs(work_budget))
        filename = value['compilation_id']+'-'+value['envelope_digest'][7:]+'.json'
        c.require(value['queue_path'] in {c.child(context.roots['compilation_queue_root'], state, filename, **_work_kwargs(work_budget))
            for state in (_work_items(('pending', 'processing', 'completed', 'blocked'), work_budget) if work_budget is not None else ('pending', 'processing', 'completed', 'blocked'))}, 'intake_path_invalid', **_work_kwargs(work_budget))
        # The receipt records its publisher state; later retained copies move
        # between queue states without changing the immutable envelope identity.
        for envelope in (_work_items(compilations.get((value['compilation_id'], value['envelope_digest']), []), work_budget) if work_budget is not None else compilations.get((value['compilation_id'], value['envelope_digest']), [])):
            c.require(all(value[k] == envelope[0][k] for k in (_work_items(('run_id', 'configured_scene_revision_digest'), work_budget) if work_budget is not None else ('run_id', 'configured_scene_revision_digest'))), 'intake_binding_invalid', **_work_kwargs(work_budget))
        index.setdefault((value['compilation_id'], value['receipt_digest']), []).append(row)
    return index


def _result(context, row, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    value, proof = row
    c.require(c.matches(value.get('preparation_id'), c.ID, **_work_kwargs(work_budget)), 'preparation_result_invalid', **_work_kwargs(work_budget))
    route = _route(context, proof['path'], **_work_kwargs(work_budget))
    p = PurePosixPath(proof['path'])
    suffix = p.name[len(value['preparation_id'])+1:-5]
    c.require(str(p.parent) == c.child(route['queue_root'], 'results', **_work_kwargs(work_budget)) and p.name.startswith(value['preparation_id']+'-')
        and p.name.endswith('.json') and len(suffix) == 64 and all(char in '0123456789abcdef' for char in (_work_items(suffix, work_budget) if work_budget is not None else suffix)),
        'preparation_result_path_invalid', **_work_kwargs(work_budget))
    if value.get('status') == 'blocked':
        c.require(all(value.get(k) is False for k in (_work_items(('provider_mutation_performed', 'catalog_mutation_performed', 'paid_execution_requested'), work_budget) if work_budget is not None else ('provider_mutation_performed', 'catalog_mutation_performed', 'paid_execution_requested')))
            and isinstance(value.get('blockers'), list) and ('source_commit' not in value or c.matches(value['source_commit'], c.COMMIT, **_work_kwargs(work_budget))), 'preparation_scope_invalid', **_work_kwargs(work_budget))
        context.missing('native_preparation_result', 'blocked_retained_preparation', [proof])
        return False
    if value.get('status') not in {PRE, FINAL}:
        context.missing('native_preparation_result', 'unsupported_retained_status', [proof])
        return False
    c.require(all(c.matches(value.get(k), c.ID, **_work_kwargs(work_budget)) for k in (_work_items(('run_id', 'team_namespace'), work_budget) if work_budget is not None else ('run_id', 'team_namespace')))
        and c.matches(value.get('source_commit'), c.COMMIT, **_work_kwargs(work_budget)), 'preparation_result_invalid', **_work_kwargs(work_budget))
    unique, reuse = _references(context, value.get('references'), **_work_kwargs(work_budget))
    c.require(type(value.get('reference_count')) is int and value['reference_count'] == len(value['references'])
        and type(value.get('unique_object_count')) is int and value['unique_object_count'] == unique
        and type(value.get('content_addressed_reuse_count')) is int and value['content_addressed_reuse_count'] == reuse
        and value.get('full_byte_service_account_readback_passed') is True
        and all(value.get(k) is False for k in (_work_items(('provider_mutation_performed', 'catalog_mutation_performed', 'paid_execution_requested'), work_budget) if work_budget is not None else ('provider_mutation_performed', 'catalog_mutation_performed', 'paid_execution_requested')))
        and type(value.get('service_account_uid')) is int and value['service_account_uid'] >= 0, 'preparation_result_invalid', **_work_kwargs(work_budget))
    c.text(value.get('service_account'), **_work_kwargs(work_budget))
    c.text(value.get('observed_at_iso'), 128, **_work_kwargs(work_budget))
    return True


def _selected_revision(context, request, references, proof, *, work_budget=None):
    """Native requests declare remote bytes; receipts bind their local copy."""
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    reference = request.get('scene', {}).get('configured_revision')
    if isinstance(reference, dict) and 'uri' in reference:
        c.require((_work_collect(work_budget, set, reference) if work_budget is not None else set(reference))
            == {'uri', 'digest', 'size_bytes'} and isinstance(reference['uri'], str)
            and reference['uri'].startswith(('s3://', 'gs://', 'https://'))
            and len(reference['uri']) <= 4096
            and not any(ch.isspace() for ch in (_work_items(reference['uri'], work_budget) if work_budget is not None else reference['uri']))
            and c.matches(reference['digest'], **_work_kwargs(work_budget))
            and type(reference['size_bytes']) is int and reference['size_bytes'] > 0,
            'revision_reference_invalid', **_work_kwargs(work_budget))
        selected = None
        for item in (_work_items(references, work_budget) if work_budget is not None else references):
            if item.get('contract_path') != 'scene.configured_revision':
                continue
            c.require(selected is None and all(item[k] == reference[k] for k in
                (_work_items(('uri', 'digest', 'size_bytes'), work_budget) if work_budget is not None else ('uri', 'digest', 'size_bytes'))),
                'revision_materialized_binding_invalid', **_work_kwargs(work_budget))
            selected = item
        if selected is None:
            context.missing('configured_revisions', 'materialized_revision_selector_unavailable', [proof],
                selector=dict(reference))
            return None
        reference = {'path': selected['materialized_path'], 'sha256': selected['digest'], 'size_bytes': selected['size_bytes']}
    return context.selected(reference, proof, {'configured_revisions'})


def _revision(context, revisions, request, compilation, final, proof, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    row = _selected_revision(context, request, final['references'], proof, **_work_kwargs(work_budget))
    if row is None:
        return None
    value = row[0]
    c.require(value.get('status') == 'configured' and c.matches(value.get('source_commit'), c.COMMIT, **_work_kwargs(work_budget))
        and value.get('revision_digest') == compilation['configured_scene_revision_digest'] == final['configured_scene_revision_digest']
        and value.get('scene_identity') == request['scene']['identity']
        and value.get('task_template', {}).get('identity') == request['task']['identity'], 'revision_binding_invalid', **_work_kwargs(work_budget))
    bundle = compilation['configured_scene_bundle']
    c.require(value.get('configured_scene_bundle') == {k: bundle[k] for k in (_work_items(('uri', 'digest', 'size_bytes'), work_budget) if work_budget is not None else ('uri', 'digest', 'size_bytes'))}
        and final['configured_scene_bundle_digest'] == bundle['digest'], 'revision_bundle_invalid', **_work_kwargs(work_budget))
    return row



def _available_parent(context, row, by_filename, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    value, proof = row
    for parent in (_work_items(by_filename.get(PurePosixPath(proof['path']).name, [])[:1], work_budget) if work_budget is not None else by_filename.get(PurePosixPath(proof['path']).name, [])[:1]):
        request = parent[0]['request']
        c.require(all(value[a] == request[b] for a, b in (_work_items((('preparation_id', 'preparation_id'), ('run_id', 'run_id'),
            ('team_namespace', 'team_namespace'), ('source_commit', 'expected_production_commit')), work_budget) if work_budget is not None else (('preparation_id', 'preparation_id'), ('run_id', 'run_id'),
            ('team_namespace', 'team_namespace'), ('source_commit', 'expected_production_commit')))), 'preparation_parent_invalid', **_work_kwargs(work_budget))
        revision = _selected_revision(context, request, value['references'], proof, **_work_kwargs(work_budget))
        if value['status'] == FINAL:
            c.require(value.get('run_mode') == request['run_mode'] and value.get('configured_scene_revision_digest')
                == request['task'].get('configured_scene_revision_digest'), 'preparation_revision_invalid', **_work_kwargs(work_budget))
        if revision is not None:
            metadata = revision[0]
            c.require(metadata.get('status') == 'configured' and c.matches(metadata.get('source_commit'), c.COMMIT, **_work_kwargs(work_budget))
                and metadata.get('scene_identity') == request['scene']['identity']
                and metadata.get('task_template', {}).get('identity') == request['task']['identity']
                and metadata['revision_digest'] == request['task'].get('configured_scene_revision_digest'), 'revision_binding_invalid', **_work_kwargs(work_budget))
            bundle = metadata.get('configured_scene_bundle')
            c.require(isinstance(bundle, dict), 'revision_bundle_invalid', **_work_kwargs(work_budget))
            declared = [r for r in (_work_items(value['references'], work_budget) if work_budget is not None else value['references']) if r['contract_path'] == 'scene.configured_revision.configured_scene_bundle']
            c.require(len(declared) == 1 and {k: declared[0][k] for k in (_work_items(('uri', 'digest', 'size_bytes'), work_budget) if work_budget is not None else ('uri', 'digest', 'size_bytes'))} == bundle,
                'revision_bundle_invalid', **_work_kwargs(work_budget))
            if value['status'] == FINAL:
                c.require(value.get('configured_scene_bundle_digest') == bundle['digest'], 'revision_bundle_invalid', **_work_kwargs(work_budget))

def inventory(context, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    envelopes, compilations = _envelopes(context, **_work_kwargs(work_budget)), _compilations(context, **_work_kwargs(work_budget))
    receipts = _receipts(context, compilations, **_work_kwargs(work_budget))
    revisions = context.known('configured_revisions')
    for revision in (_work_items(revisions, work_budget) if work_budget is not None else revisions):
        context.metadata(revision)
    by_filename = {}
    for parents in (_work_items(envelopes.values(), work_budget) if work_budget is not None else envelopes.values()):
        for parent in (_work_items(parents, work_budget) if work_budget is not None else parents):
            by_filename.setdefault(PurePosixPath(parent[1]['path']).name, []).append(parent)
    results, finals, observations = {}, [], context.rows()
    for role in (_work_items(('native_preparation_results', 'preparation_results'), work_budget) if work_budget is not None else ('native_preparation_results', 'preparation_results')):
        for row in (_work_items(context.known(role, 'task_evaluation_launch_preparation_result.v1', 'result_digest'), work_budget) if work_budget is not None else context.known(role, 'task_evaluation_launch_preparation_result.v1', 'result_digest')):
            if role == 'preparation_results' and row[0].get('status') != FINAL:
                continue  # Old scene result interpretation is kept verbatim.
            if _result(context, row, **_work_kwargs(work_budget)):
                _available_parent(context, row, by_filename, **_work_kwargs(work_budget))
                results.setdefault((row[0]['preparation_id'], row[0]['result_digest']), []).append(row)
                if row[0]['status'] == FINAL:
                    finals.append(row)
    context.native_preparations, context.native_results, context.native_compilations = envelopes, results, compilations
    final_versions = {}
    for row in (_work_items(finals, work_budget) if work_budget is not None else finals):
        final_versions.setdefault((row[0]['preparation_id'], row[0]['result_digest']), []).append(row)
    for versions in (_work_items(final_versions.values(), work_budget) if work_budget is not None else final_versions.values()):
        row = versions[0]
        final, proof = row
        for field in (_work_items(('configured_scene_revision_digest', 'configured_scene_bundle_digest', 'episode_compilation_queue_envelope_digest',
                      'episode_compilation_queue_receipt_digest'), work_budget) if work_budget is not None else ('configured_scene_revision_digest', 'configured_scene_bundle_digest', 'episode_compilation_queue_envelope_digest',
                      'episode_compilation_queue_receipt_digest')):
            c.require(c.matches(final.get(field), **_work_kwargs(work_budget)), 'handoff_invalid', **_work_kwargs(work_budget))
        c.require(final.get('episode_compilation_id') == final['preparation_id']
            and final.get('run_mode') in {'episode_evaluation', 'destination_qualification'}
            and final.get('customer_supplied_prebuilt_episode_packet') is False and final.get('construction_packet_materialized') is False
            and final.get('automatic_progression_required') is True, 'handoff_invalid', **_work_kwargs(work_budget))
        candidates = compilations.get((final['preparation_id'], final['episode_compilation_queue_envelope_digest']), [])
        verified, strength = False, None
        sources = context.provenance(r[1] for r in (_work_items(versions, work_budget) if work_budget is not None else versions))
        if candidates:
            # Same seal identifies identical canonical metadata; raw formatting
            # and queue-state copies remain separately discoverable provenance.
            compilation = candidates[0][0]
            c.require(all(final[a] == compilation[b] for a, b in (_work_items((('run_id', 'run_id'), ('team_namespace', 'team_namespace'),
                ('source_commit', 'expected_production_commit')), work_budget) if work_budget is not None else (('run_id', 'run_id'), ('team_namespace', 'team_namespace'),
                ('source_commit', 'expected_production_commit')))), 'handoff_binding_invalid', **_work_kwargs(work_budget))
            request = compilation['request']
            c.require(final['run_mode'] == request['run_mode'] and final['references'] == compilation['materialized_references'], 'handoff_binding_invalid', **_work_kwargs(work_budget))
            request_digest = (_work_call(work_budget, c.canonical_digest, request) if work_budget is not None else c.canonical_digest(request))
            stem = final['preparation_id']+'-'+request_digest[7:]+'.json'
            route = _route(context, proof['path'], **_work_kwargs(work_budget))
            c.require(proof['path'] == c.child(route['queue_root'], 'results', stem, **_work_kwargs(work_budget)), 'preparation_result_path_invalid', **_work_kwargs(work_budget))
            parent_rows = envelopes.get((final['preparation_id'], request_digest), [])
            for parent in (_work_items(parent_rows[:1], work_budget) if work_budget is not None else parent_rows[:1]):
                c.require(parent[0]['request'] == request, 'preparation_request_binding_invalid', **_work_kwargs(work_budget))
            sources += context.provenance(p[1] for p in (_work_items(parent_rows, work_budget) if work_budget is not None else parent_rows))
            sources += context.provenance(p[1] for p in (_work_items(candidates, work_budget) if work_budget is not None else candidates))
            revision = _revision(context, revisions, request, compilation, final, proof, **_work_kwargs(work_budget))
            if revision is not None:
                sources += context.provenance((revision[1],))
            intake_rows = receipts.get((final['preparation_id'], final['episode_compilation_queue_receipt_digest']), [])
            for intake in (_work_items(intake_rows[:1], work_budget) if work_budget is not None else intake_rows[:1]):
                c.require(intake[0]['envelope_digest'] == compilation['envelope_digest'], 'handoff_intake_invalid', **_work_kwargs(work_budget))
            sources += context.provenance(p[1] for p in (_work_items(intake_rows, work_budget) if work_budget is not None else intake_rows))
            if not intake_rows:
                context.canonical('compilation_intake', final['episode_compilation_queue_receipt_digest'], proof)
            if (_work_collect(work_budget, set, final) if work_budget is not None else set(final)) in (BASE | HANDOFF, BASE | HANDOFF | {'policy_run_plan'}):
                inverse = {key: value for key, value in (_work_items(final.items(), work_budget) if work_budget is not None else final.items()) if key not in HANDOFF | {'result_digest'}}
                inverse['status'] = PRE
                context.emission_budget.reserve_row(inverse)  # Charge bounded temporary before canonical encoding/hash.
                derived = (_work_call(work_budget, c.canonical_digest, inverse, digest_field='result_digest') if work_budget is not None else c.canonical_digest(inverse, digest_field='result_digest'))
                c.require(derived == compilation['preparation_result_digest'], 'pre_handoff_inverse_invalid', **_work_kwargs(work_budget))
                retained = results.get((final['preparation_id'], derived), [])
                for actual in (_work_items(retained[:1], work_budget) if work_budget is not None else retained[:1]):
                    c.require(actual[0]['status'] == PRE and {k: v for k, v in (_work_items(actual[0].items(), work_budget) if work_budget is not None else actual[0].items()) if k != 'result_digest'} == inverse,
                              'pre_handoff_raw_binding_invalid', **_work_kwargs(work_budget))
                sources += context.provenance(p[1] for p in (_work_items(retained, work_budget) if work_budget is not None else retained))
                strength = 'retained_raw_and_derived_metadata' if retained else 'derived_metadata_inverse'
                verified = revision is not None and context.supported(revision) and all(context.supported(p) for group in (_work_items((versions, candidates, parent_rows, intake_rows), work_budget) if work_budget is not None else (versions, candidates, parent_rows, intake_rows)) for p in (_work_items(group, work_budget) if work_budget is not None else group))
                if not verified:
                    strength = None
                if 'policy_run_plan' in inverse:
                    context.missing('policy_run_plan', 'policy_semantics_deferred', [proof])
            else:
                context.missing('pre_handoff_inverse', 'unknown_retained_field_set', [proof])
        else:
            context.canonical('compilation_envelope', final['episode_compilation_queue_envelope_digest'], proof)
        observations.append(c.observation(row, pre_handoff_binding_verified=verified, pre_handoff_proof_strength=strength,
            pre_handoff_canonical_digest=candidates[0][0]['preparation_result_digest'] if verified else None,
            source_provenance=sources, historical_raw_bytes_created=False, **_work_kwargs(work_budget)))
    return observations
