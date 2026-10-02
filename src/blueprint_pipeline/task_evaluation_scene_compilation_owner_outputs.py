"""Supplied compiler/adapter metadata edges; never reads packet bytes.

A remote compile (plan 14) lands only what later stages read, so its packet's bytes stay remote; the
remote-output pointer beside its output lists the packet's path, digest and size and stands for them.
"""
from __future__ import annotations

from .task_evaluation_scene_lineage_budget import _work_collect, _work, _work_items, _work_kwargs

from pathlib import PurePosixPath

from . import task_evaluation_scene_compilation_owner_contracts as c
from .remote_cpu_job_records import validate_pointer

ADAPTER_NAME = 'task_evaluation_native_arena_adapter_result.v1.json'
RESULT_SCHEMA = 'task_evaluation_episode_compilation_result.v1'
ADAPTER_FIELDS = {'schema_version', 'status', 'preparation_id', 'source_commit', 'adapter_kind', 'adapter_version',
    'construction_manifest_digest', 'configured_scene_revision_digest', 'runtime_source_manifest_digest',
    'packet_receipt_digest', 'runtime_source_receipt_digest', 'packet_root', 'runtime_source_receipt',
    'provider_mutation_performed', 'catalog_mutation_performed', 'paid_execution_requested', 'result_digest'}
OUTPUT_FIELDS = {'schema_version', 'status', 'run_id', 'configured_scene_revision_digest', 'configured_task_template_adapter',
    'compiled_episode_packet', 'adapter_result', 'native_scene_appearance', 'compiled_by_production',
    'customer_supplied_prebuilt_episode_packet', 'provider_mutation_performed', 'paid_execution_requested',
    'raw_secret_values_recorded', 'compiler_output_digest'}
# The remote-output pointer schemas this census accepts; ``cloud_run`` compiles only once the pointer's is here.
REMOTE_OUTPUT_POINTER_SCHEMAS = frozenset({'remote_cpu_output_pointer.v1'})
POINTER_ROLE, POINTER_SUFFIX = 'compilation_remote_output_pointers', '.remote-output.v1.json'
MAX_POINTER_REFERENCES = 64


def _raw(context, artifact, proof, root, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    c.require(isinstance(artifact, dict) and c.matches(artifact.get('digest'), **_work_kwargs(work_budget))
        and type(artifact.get('size_bytes')) is int and artifact['size_bytes'] > 0, 'artifact_invalid', **_work_kwargs(work_budget))
    c.path(artifact.get('path'), **_work_kwargs(work_budget))
    c.require(c.under(artifact['path'], root, **_work_kwargs(work_budget)), 'artifact_path_invalid', **_work_kwargs(work_budget))
    ref = {'path': artifact['path'], 'sha256': artifact['digest'], 'size_bytes': artifact['size_bytes']}
    pointer = getattr(context, 'remote_pointers', {}).get((artifact['path'], artifact['digest'], artifact['size_bytes']))
    context.pointed_raw_ref(ref, proof, pointer) if pointer else context.raw_ref(ref, proof)
    return {k: artifact[k] for k in (_work_items(('path', 'digest', 'size_bytes'), work_budget) if work_budget is not None else ('path', 'digest', 'size_bytes'))}


def _pointers(context, *, work_budget=None):
    """Each supplied remote-output pointer by the (path, digest, size) it lists.  A pointer sits beside the
    output it stands for, is bound to that output and its queue row, and lists only bytes under it; one that
    contradicts itself refuses.  The remote bytes themselves are never read."""
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    _work(work_budget)
    index, root = {}, context.roots['compilation_output_root']
    for row in (_work_items(context.known(POINTER_ROLE), work_budget) if work_budget is not None else context.known(POINTER_ROLE)):
        value, proof = row
        try:  # the canonical contract first (fields, seal, attempt, CAS names, totals), against this census's root
            validate_pointer(value, output_roots=(root+'/',))
        except ValueError:
            c.require(False, 'remote_pointer_invalid')
        comp_id, queue_row, refs = value.get('compilation_id'), value.get('queue_row'), value.get('raw_references')
        c.require(c.matches(comp_id, c.ID) and value.get('stage') == 'episode_compilation'
            and value['schema_version'] in REMOTE_OUTPUT_POINTER_SCHEMAS and proof['path'] == c.child(root, comp_id+POINTER_SUFFIX)
            and value.get('output_root') == c.child(root, comp_id) and value.get('state') in ('landed', 'restored_full')
            and isinstance(queue_row, dict) and c.matches(queue_row.get('envelope_digest'))
            and queue_row.get('name') == comp_id+'-'+queue_row['envelope_digest'][7:]+'.json'
            and isinstance(refs, list) and len(refs) <= MAX_POINTER_REFERENCES, 'remote_pointer_invalid')
        for ref in (_work_items(refs, work_budget) if work_budget is not None else refs):
            c.require(isinstance(ref, dict) and set(ref) == {'path', 'digest', 'size_bytes'} and c.matches(ref['digest'])
                and type(ref['size_bytes']) is int and ref['size_bytes'] > 0 and c.under(ref['path'], value['output_root']),
                'remote_pointer_invalid')
            index.setdefault((ref['path'], ref['digest'], ref['size_bytes']), row)
    return index


def _adapter_selector(context, value, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    c.require(isinstance(value, dict) and c.matches(value.get('digest'), **_work_kwargs(work_budget)), 'adapter_selector_invalid', **_work_kwargs(work_budget))
    c.path(value.get('path'), **_work_kwargs(work_budget))
    p = PurePosixPath(value['path'])
    c.require(p.name == ADAPTER_NAME and p.parent.name == 'native-arena-adapter'
        and c.matches(p.parent.parent.name, c.ID, **_work_kwargs(work_budget)) and str(p.parent.parent.parent) == context.roots['compilation_output_root'],
        'adapter_path_invalid', **_work_kwargs(work_budget))
    return value['path'], value['digest']


def _adapters(context, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    index = {}
    context.unproven_adapter_seals = set()
    for row in (_work_items(context.known('compilation_adapter_results'), work_budget) if work_budget is not None else context.known('compilation_adapter_results')):
        value, proof = row
        comp_id = PurePosixPath(proof['path']).parent.parent.name
        _adapter_selector(context, {'path': proof['path'], 'digest': value['result_digest']}, **_work_kwargs(work_budget))
        c.require(c.matches(value.get('preparation_id'), c.ID, **_work_kwargs(work_budget)) and value['preparation_id'] == comp_id
            and c.matches(value.get('source_commit'), c.COMMIT, **_work_kwargs(work_budget)), 'adapter_identity_invalid', **_work_kwargs(work_budget))
        if value.get('status') != 'native_arena_adapter_materialized':
            context.missing('adapter', 'unsupported_retained_status', [proof])
            continue
        for field in (_work_items(('configured_scene_revision_digest', 'construction_manifest_digest', 'runtime_source_manifest_digest',
            'packet_receipt_digest', 'runtime_source_receipt_digest'), work_budget) if work_budget is not None else ('configured_scene_revision_digest', 'construction_manifest_digest', 'runtime_source_manifest_digest',
            'packet_receipt_digest', 'runtime_source_receipt_digest')):
            c.require(c.matches(value.get(field), **_work_kwargs(work_budget)), 'adapter_invalid', **_work_kwargs(work_budget))
            context.canonical(field, value[field], proof)
        c.require(value.get('adapter_kind') == 'native_task_arena' and value.get('adapter_version') == 'v1', 'adapter_kind_invalid', **_work_kwargs(work_budget))
        c.require(ADAPTER_FIELDS <= (_work_collect(work_budget, set, value) if work_budget is not None else set(value)), 'adapter_invalid', **_work_kwargs(work_budget))
        if (_work_collect(work_budget, set, value) if work_budget is not None else set(value)) != ADAPTER_FIELDS:
            context.unproven_adapter_seals.add(value['result_digest'])
            context.missing('adapter', 'unsupported_retained_field_set', [proof])
        c.require(all(value.get(k) is False for k in (_work_items(('provider_mutation_performed', 'catalog_mutation_performed', 'paid_execution_requested'), work_budget) if work_budget is not None else ('provider_mutation_performed', 'catalog_mutation_performed', 'paid_execution_requested'))),
            'adapter_scope_invalid', **_work_kwargs(work_budget))
        for field in (_work_items(('packet_root', 'runtime_source_receipt'), work_budget) if work_budget is not None else ('packet_root', 'runtime_source_receipt')):
            c.path(value.get(field), **_work_kwargs(work_budget))
            c.require(c.under(value[field], str(PurePosixPath(proof['path']).parent), **_work_kwargs(work_budget)), 'adapter_member_path_invalid', **_work_kwargs(work_budget))
            context.member(value[field], 'adapter_'+field, {'preparation_id': comp_id, 'canonical_result_digest': value['result_digest']},
                context.provenance((dict(proof, json_pointer='/'+field),)))
        index.setdefault((proof['path'], value['result_digest']), []).append(row)
    return index


def _outputs(context, adapters, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    index = {}
    context.unproven_output_seals = set()
    for row in (_work_items(context.known('compiler_outputs'), work_budget) if work_budget is not None else context.known('compiler_outputs')):
        value, proof = row
        context.metadata(row)
        if value.get('status') != 'completed':
            context.missing('compiler_output', 'unsupported_retained_status', [proof])
            continue
        c.require(c.matches(value.get('run_id'), c.ID, **_work_kwargs(work_budget)) and c.matches(value.get('configured_scene_revision_digest'), **_work_kwargs(work_budget))
            and value.get('compiled_by_production') is True
            and all(value.get(k) is False for k in (_work_items(('customer_supplied_prebuilt_episode_packet', 'provider_mutation_performed',
                'paid_execution_requested', 'raw_secret_values_recorded'), work_budget) if work_budget is not None else ('customer_supplied_prebuilt_episode_packet', 'provider_mutation_performed',
                'paid_execution_requested', 'raw_secret_values_recorded'))), 'compiler_output_invalid', **_work_kwargs(work_budget))
        c.require(OUTPUT_FIELDS <= (_work_collect(work_budget, set, value) if work_budget is not None else set(value)) and isinstance(value.get('configured_task_template_adapter'), dict)
            and isinstance(value.get('native_scene_appearance'), dict), 'compiler_output_invalid', **_work_kwargs(work_budget))
        fields = {'compiled_episode_packet': {'format', 'path', 'digest', 'size_bytes'},
            'adapter_result': {'path', 'digest', 'packet_receipt_digest', 'runtime_source_receipt_digest'},
            'configured_task_template_adapter': {'schema_version', 'adapter_digest', 'source_documents_digest', 'manipulation_strategy'}}
        if 'destination_native_probe_request' in value:
            fields['destination_native_probe_request'] = {'path', 'digest', 'size_bytes', 'request_digest'}
        c.require(all(isinstance(value.get(k), dict) and required <= (_work_collect(work_budget, set, value[k]) if work_budget is not None else set(value[k])) for k, required in (_work_items(fields.items(), work_budget) if work_budget is not None else fields.items())), 'compiler_artifact_invalid', **_work_kwargs(work_budget))
        template = value['configured_task_template_adapter']
        c.require(c.matches(template.get('adapter_digest'), **_work_kwargs(work_budget)) and c.matches(template.get('source_documents_digest'), **_work_kwargs(work_budget)), 'compiler_template_invalid', **_work_kwargs(work_budget))
        c.text(template.get('schema_version'), **_work_kwargs(work_budget))
        c.text(template.get('manipulation_strategy'), **_work_kwargs(work_budget))
        if (_work_collect(work_budget, set, value) if work_budget is not None else set(value)) not in (OUTPUT_FIELDS, OUTPUT_FIELDS | {'destination_native_probe_request'}) or any((_work_collect(work_budget, set, value[k]) if work_budget is not None else set(value[k])) != required for k, required in (_work_items(fields.items(), work_budget) if work_budget is not None else fields.items())):
            context.unproven_output_seals.add(value['compiler_output_digest'])
            context.missing('compiler_output', 'unsupported_retained_field_set', [proof])
        packet = value.get('compiled_episode_packet')
        c.require(isinstance(packet, dict) and packet.get('format') == 'native_task_arena_bundle_zip', 'compiler_packet_invalid', **_work_kwargs(work_budget))
        _raw(context, packet, proof, context.roots['compilation_output_root'], **_work_kwargs(work_budget))
        selector = _adapter_selector(context, value.get('adapter_result'), **_work_kwargs(work_budget))
        c.require(PurePosixPath(packet['path']).parent == PurePosixPath(selector[0]).parent.parent, 'compiler_packet_path_invalid', **_work_kwargs(work_budget))
        for field in (_work_items(('packet_receipt_digest', 'runtime_source_receipt_digest'), work_budget) if work_budget is not None else ('packet_receipt_digest', 'runtime_source_receipt_digest')):
            c.require(c.matches(value['adapter_result'].get(field), **_work_kwargs(work_budget)), 'compiler_adapter_invalid', **_work_kwargs(work_budget))
        for adapter in (_work_items(adapters.get(selector, [])[:1], work_budget) if work_budget is not None else adapters.get(selector, [])[:1]):
            c.require(all(adapter[0][k] == value['adapter_result'][k] for k in (_work_items(('packet_receipt_digest', 'runtime_source_receipt_digest'), work_budget) if work_budget is not None else ('packet_receipt_digest', 'runtime_source_receipt_digest')))
                and adapter[0]['configured_scene_revision_digest'] == value['configured_scene_revision_digest'], 'compiler_adapter_binding_invalid', **_work_kwargs(work_budget))
        if 'destination_native_probe_request' in value:
            probe = value['destination_native_probe_request']
            _raw(context, probe, proof, str(PurePosixPath(selector[0]).parent.parent), **_work_kwargs(work_budget))
            c.require(c.matches(probe.get('request_digest'), **_work_kwargs(work_budget)), 'probe_invalid', **_work_kwargs(work_budget))
            context.canonical('destination_probe', probe['request_digest'], proof, probe['path'])
        index.setdefault(value['compiler_output_digest'], []).append(row)
    return index


def inventory(context, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    context.remote_pointers = _pointers(context, **_work_kwargs(work_budget))
    adapters = _adapters(context, **_work_kwargs(work_budget))
    outputs = _outputs(context, adapters, **_work_kwargs(work_budget))
    envelopes = {}
    for versions in (_work_items(context.native_compilations.values(), work_budget) if work_budget is not None else context.native_compilations.values()):
        for row in (_work_items(versions, work_budget) if work_budget is not None else versions):
            envelopes.setdefault((row[0]['compilation_id'], row[0]['envelope_digest']), []).append(row)
    observations, results = context.rows(), {}
    for row in (_work_items(context.known('compilation_results', RESULT_SCHEMA, 'result_digest'), work_budget) if work_budget is not None else context.known('compilation_results', RESULT_SCHEMA, 'result_digest')):
        value, proof = row
        p = PurePosixPath(proof['path'])
        c.require(c.matches(value.get('compilation_id'), c.ID, **_work_kwargs(work_budget)) and str(p.parent) == c.child(context.roots['compilation_queue_root'], 'results', **_work_kwargs(work_budget)),
            'compilation_result_path_invalid', **_work_kwargs(work_budget))
        prefix = value['compilation_id']+'-'
        suffix = p.name[len(prefix):-5]
        c.require(p.name.startswith(prefix) and p.name.endswith('.json') and len(suffix) == 64
            and all(ch in '0123456789abcdef' for ch in (_work_items(suffix, work_budget) if work_budget is not None else suffix)), 'compilation_result_path_invalid', **_work_kwargs(work_budget))
        envelope_rows = envelopes.get((value['compilation_id'], 'sha256:'+suffix), [])
        if value.get('status') != 'compiled_for_production_launch':
            context.missing('compilation_result', 'unsupported_or_blocked_retained_status', [proof])
            continue
        c.require(all(c.matches(value.get(k), c.ID, **_work_kwargs(work_budget)) for k in (_work_items(('run_id', 'team_namespace'), work_budget) if work_budget is not None else ('run_id', 'team_namespace')))
            and c.matches(value.get('source_commit'), c.COMMIT, **_work_kwargs(work_budget)) and c.matches(value.get('configured_scene_revision_digest'), **_work_kwargs(work_budget))
            and value.get('compiled_by_production') is True and value.get('automatic_progression_required') is True
            and all(value.get(k) is False for k in (_work_items(('customer_supplied_prebuilt_episode_packet', 'provider_mutation_performed', 'paid_execution_requested'), work_budget) if work_budget is not None else ('customer_supplied_prebuilt_episode_packet', 'provider_mutation_performed', 'paid_execution_requested')))
            and isinstance(value.get('blockers'), list), 'compilation_result_invalid', **_work_kwargs(work_budget))
        root = c.child(context.roots['compilation_output_root'], value['compilation_id'], **_work_kwargs(work_budget))
        packet = {'path': value.get('compiled_episode_packet_path'), 'digest': value.get('compiled_episode_packet_digest'),
            'size_bytes': value.get('compiled_episode_packet_size_bytes')}
        _raw(context, packet, proof, root, **_work_kwargs(work_budget))
        selector = _adapter_selector(context, {'path': value.get('adapter_result_path'), 'digest': value.get('adapter_result_digest')}, **_work_kwargs(work_budget))
        c.require(c.under(selector[0], root, **_work_kwargs(work_budget)) and c.matches(value.get('compiler_output_digest'), **_work_kwargs(work_budget)), 'compilation_result_invalid', **_work_kwargs(work_budget))
        sources = context.provenance((proof,))
        for envelope in (_work_items(envelope_rows[:1], work_budget) if work_budget is not None else envelope_rows[:1]):
            c.require(all(value[a] == envelope[0][b] for a, b in (_work_items((('run_id', 'run_id'), ('team_namespace', 'team_namespace'),
                ('source_commit', 'expected_production_commit'), ('configured_scene_revision_digest', 'configured_scene_revision_digest')), work_budget) if work_budget is not None else (('run_id', 'run_id'), ('team_namespace', 'team_namespace'),
                ('source_commit', 'expected_production_commit'), ('configured_scene_revision_digest', 'configured_scene_revision_digest')))),
                'compilation_envelope_binding_invalid', **_work_kwargs(work_budget))
        sources += context.provenance(r[1] for r in (_work_items(envelope_rows, work_budget) if work_budget is not None else envelope_rows))
        adapter_rows = adapters.get(selector, [])
        for adapter in (_work_items(adapter_rows[:1], work_budget) if work_budget is not None else adapter_rows[:1]):
            c.require(all(adapter[0][a] == value[b] for a, b in (_work_items((('preparation_id', 'compilation_id'), ('source_commit', 'source_commit'),
                ('configured_scene_revision_digest', 'configured_scene_revision_digest')), work_budget) if work_budget is not None else (('preparation_id', 'compilation_id'), ('source_commit', 'source_commit'),
                ('configured_scene_revision_digest', 'configured_scene_revision_digest')))), 'adapter_result_binding_invalid', **_work_kwargs(work_budget))
        sources += context.provenance(r[1] for r in (_work_items(adapter_rows, work_budget) if work_budget is not None else adapter_rows))
        if not adapter_rows:
            context.canonical('adapter_result', selector[1], proof, selector[0])
        output_rows = outputs.get(value['compiler_output_digest'], [])
        for output in (_work_items(output_rows[:1], work_budget) if work_budget is not None else output_rows[:1]):
            metadata = output[0]
            c.require(all(metadata[k] == value[k] for k in (_work_items(('run_id', 'configured_scene_revision_digest'), work_budget) if work_budget is not None else ('run_id', 'configured_scene_revision_digest')))
                and {k: metadata['compiled_episode_packet'][k] for k in (_work_items(('path', 'digest', 'size_bytes'), work_budget) if work_budget is not None else ('path', 'digest', 'size_bytes'))} == packet
                and _adapter_selector(context, metadata['adapter_result'], **_work_kwargs(work_budget)) == selector, 'compiler_result_binding_invalid', **_work_kwargs(work_budget))
            probe = metadata.get('destination_native_probe_request')
            for envelope in (_work_items(envelope_rows[:1], work_budget) if work_budget is not None else envelope_rows[:1]):
                c.require((envelope[0]['request']['run_mode'] == 'destination_qualification') == (probe is not None), 'probe_mode_invalid', **_work_kwargs(work_budget))
            if probe is not None:
                c.require(all(value.get(a) == probe[b] for a, b in (_work_items((('destination_native_probe_request_path', 'path'),
                    ('destination_native_probe_request_digest', 'digest'), ('destination_native_probe_request_document_digest', 'request_digest')), work_budget) if work_budget is not None else (('destination_native_probe_request_path', 'path'),
                    ('destination_native_probe_request_digest', 'digest'), ('destination_native_probe_request_document_digest', 'request_digest')))), 'probe_binding_invalid', **_work_kwargs(work_budget))
            else:
                c.require(not any(k in value for k in (_work_items(('destination_native_probe_request_path', 'destination_native_probe_request_digest',
                    'destination_native_probe_request_document_digest'), work_budget) if work_budget is not None else ('destination_native_probe_request_path', 'destination_native_probe_request_digest',
                    'destination_native_probe_request_document_digest'))), 'probe_binding_invalid', **_work_kwargs(work_budget))
        sources += context.provenance(r[1] for r in (_work_items(output_rows, work_budget) if work_budget is not None else output_rows))
        if not output_rows:
            context.canonical('compiler_output', value['compiler_output_digest'], proof)
            if any(k.startswith('destination_native_probe_request_') for k in (_work_items(value, work_budget) if work_budget is not None else value)):
                probe_path = c.path(value.get('destination_native_probe_request_path'), **_work_kwargs(work_budget))
                c.require(c.under(probe_path, root, **_work_kwargs(work_budget)) and c.matches(value.get('destination_native_probe_request_digest'), **_work_kwargs(work_budget))
                    and c.matches(value.get('destination_native_probe_request_document_digest'), **_work_kwargs(work_budget)), 'probe_invalid', **_work_kwargs(work_budget))
                context.missing('destination_probe', 'raw_size_unavailable', [proof], probe_path,
                    {'sha256': value['destination_native_probe_request_digest'], 'canonical_digest': value['destination_native_probe_request_document_digest']})
        context.member(packet['path'], 'compiled_episode_packet', packet, sources)
        observations.append(c.observation(row, kind='compilation_output', adapter_metadata_binding_verified=bool(adapter_rows) and context.supported(row) and selector[1] not in context.unproven_adapter_seals,
            compiler_output_metadata_binding_verified=bool(output_rows) and context.supported(row) and value['compiler_output_digest'] not in context.unproven_output_seals, source_provenance=sources, **_work_kwargs(work_budget)))
        results.setdefault((value['compilation_id'], value['result_digest']), []).append(row)
    context.compilation_results, context.adapter_results = results, adapters
    return observations
