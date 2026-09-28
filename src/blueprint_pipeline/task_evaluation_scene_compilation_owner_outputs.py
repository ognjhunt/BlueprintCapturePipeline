"""Supplied compiler/adapter metadata edges; never reads packet bytes."""
from __future__ import annotations

from pathlib import PurePosixPath

from . import task_evaluation_scene_compilation_owner_contracts as c

ADAPTER_NAME = 'task_evaluation_native_arena_adapter_result.v1.json'
RESULT_SCHEMA = 'task_evaluation_episode_compilation_result.v1'


def _raw(context, artifact, proof, root):
    c.require(isinstance(artifact, dict) and c.matches(artifact.get('digest'))
        and type(artifact.get('size_bytes')) is int and artifact['size_bytes'] > 0, 'artifact_invalid')
    c.path(artifact.get('path'))
    c.require(c.under(artifact['path'], root), 'artifact_path_invalid')
    context.raw_ref({'path': artifact['path'], 'sha256': artifact['digest'], 'size_bytes': artifact['size_bytes']}, proof)
    return {k: artifact[k] for k in ('path', 'digest', 'size_bytes')}


def _adapter_selector(context, value):
    c.require(isinstance(value, dict) and c.matches(value.get('digest')), 'adapter_selector_invalid')
    c.path(value.get('path'))
    p = PurePosixPath(value['path'])
    c.require(p.name == ADAPTER_NAME and p.parent.name == 'native-arena-adapter'
        and c.matches(p.parent.parent.name, c.ID) and str(p.parent.parent.parent) == context.roots['compilation_output_root'],
        'adapter_path_invalid')
    return value['path'], value['digest']


def _adapters(context):
    index = {}
    for row in context.known('compilation_adapter_results'):
        value, proof = row
        comp_id = PurePosixPath(proof['path']).parent.parent.name
        _adapter_selector(context, {'path': proof['path'], 'digest': value['result_digest']})
        c.require(c.matches(value.get('preparation_id'), c.ID) and value['preparation_id'] == comp_id
            and c.matches(value.get('source_commit'), c.COMMIT), 'adapter_identity_invalid')
        if value.get('status') != 'native_arena_adapter_materialized':
            context.missing('adapter', 'unsupported_retained_status', [proof])
            continue
        for field in ('configured_scene_revision_digest', 'construction_manifest_digest', 'runtime_source_manifest_digest',
            'packet_receipt_digest', 'runtime_source_receipt_digest'):
            c.require(c.matches(value.get(field)), 'adapter_invalid')
            context.canonical(field, value[field], proof)
        c.text(value.get('adapter_kind'))
        c.text(value.get('adapter_version'))
        c.require(all(value.get(k) is False for k in ('provider_mutation_performed', 'catalog_mutation_performed', 'paid_execution_requested')),
            'adapter_scope_invalid')
        for field in ('packet_root', 'runtime_source_receipt'):
            c.path(value.get(field))
            c.require(c.under(value[field], str(PurePosixPath(proof['path']).parent)), 'adapter_member_path_invalid')
            context.member(value[field], 'adapter_'+field, {'preparation_id': comp_id, 'canonical_result_digest': value['result_digest']},
                context.provenance((dict(proof, json_pointer='/'+field),)))
        index.setdefault((proof['path'], value['result_digest']), []).append(row)
    return index


def _outputs(context, adapters):
    index = {}
    for row in context.known('compiler_outputs'):
        value, proof = row
        context.metadata(row)
        if value.get('status') != 'completed':
            context.missing('compiler_output', 'unsupported_retained_status', [proof])
            continue
        c.require(c.matches(value.get('run_id'), c.ID) and c.matches(value.get('configured_scene_revision_digest'))
            and value.get('compiled_by_production') is True
            and all(value.get(k) is False for k in ('customer_supplied_prebuilt_episode_packet', 'provider_mutation_performed',
                'paid_execution_requested', 'raw_secret_values_recorded')), 'compiler_output_invalid')
        packet = value.get('compiled_episode_packet')
        c.require(isinstance(packet, dict) and packet.get('format') == 'native_task_arena_bundle_zip', 'compiler_packet_invalid')
        _raw(context, packet, proof, context.roots['compilation_output_root'])
        selector = _adapter_selector(context, value.get('adapter_result'))
        c.require(PurePosixPath(packet['path']).parent == PurePosixPath(selector[0]).parent.parent, 'compiler_packet_path_invalid')
        for field in ('packet_receipt_digest', 'runtime_source_receipt_digest'):
            c.require(c.matches(value['adapter_result'].get(field)), 'compiler_adapter_invalid')
        for adapter in adapters.get(selector, []):
            c.require(all(adapter[0][k] == value['adapter_result'][k] for k in ('packet_receipt_digest', 'runtime_source_receipt_digest'))
                and adapter[0]['configured_scene_revision_digest'] == value['configured_scene_revision_digest'], 'compiler_adapter_binding_invalid')
        if 'destination_native_probe_request' in value:
            probe = value['destination_native_probe_request']
            _raw(context, probe, proof, str(PurePosixPath(selector[0]).parent.parent))
            c.require(c.matches(probe.get('request_digest')), 'probe_invalid')
            context.canonical('destination_probe', probe['request_digest'], proof, probe['path'])
        index.setdefault(value['compiler_output_digest'], []).append(row)
    return index


def inventory(context):
    adapters = _adapters(context)
    outputs = _outputs(context, adapters)
    envelopes = {}
    for versions in context.native_compilations.values():
        for row in versions:
            envelopes.setdefault((row[0]['compilation_id'], row[0]['envelope_digest']), []).append(row)
    observations, results = context.rows(), {}
    for row in context.known('compilation_results', RESULT_SCHEMA, 'result_digest'):
        value, proof = row
        p = PurePosixPath(proof['path'])
        c.require(c.matches(value.get('compilation_id'), c.ID) and str(p.parent) == c.child(context.roots['compilation_queue_root'], 'results'),
            'compilation_result_path_invalid')
        prefix = value['compilation_id']+'-'
        suffix = p.name[len(prefix):-5]
        c.require(p.name.startswith(prefix) and p.name.endswith('.json') and len(suffix) == 64
            and all(ch in '0123456789abcdef' for ch in suffix), 'compilation_result_path_invalid')
        envelope_rows = envelopes.get((value['compilation_id'], 'sha256:'+suffix), [])
        if value.get('status') != 'compiled_for_production_launch':
            context.missing('compilation_result', 'unsupported_or_blocked_retained_status', [proof])
            continue
        c.require(all(c.matches(value.get(k), c.ID) for k in ('run_id', 'team_namespace'))
            and c.matches(value.get('source_commit'), c.COMMIT) and c.matches(value.get('configured_scene_revision_digest'))
            and value.get('compiled_by_production') is True and value.get('automatic_progression_required') is True
            and all(value.get(k) is False for k in ('customer_supplied_prebuilt_episode_packet', 'provider_mutation_performed', 'paid_execution_requested'))
            and isinstance(value.get('blockers'), list), 'compilation_result_invalid')
        root = c.child(context.roots['compilation_output_root'], value['compilation_id'])
        packet = {'path': value.get('compiled_episode_packet_path'), 'digest': value.get('compiled_episode_packet_digest'),
            'size_bytes': value.get('compiled_episode_packet_size_bytes')}
        _raw(context, packet, proof, root)
        selector = _adapter_selector(context, {'path': value.get('adapter_result_path'), 'digest': value.get('adapter_result_digest')})
        c.require(c.under(selector[0], root) and c.matches(value.get('compiler_output_digest')), 'compilation_result_invalid')
        sources = context.provenance((proof,))
        for envelope in envelope_rows:
            c.require(all(value[a] == envelope[0][b] for a, b in (('run_id', 'run_id'), ('team_namespace', 'team_namespace'),
                ('source_commit', 'expected_production_commit'), ('configured_scene_revision_digest', 'configured_scene_revision_digest'))),
                'compilation_envelope_binding_invalid')
        sources += context.provenance(r[1] for r in envelope_rows)
        adapter_rows = adapters.get(selector, [])
        for adapter in adapter_rows:
            c.require(all(adapter[0][a] == value[b] for a, b in (('preparation_id', 'compilation_id'), ('source_commit', 'source_commit'),
                ('configured_scene_revision_digest', 'configured_scene_revision_digest'))), 'adapter_result_binding_invalid')
        sources += context.provenance(r[1] for r in adapter_rows)
        if not adapter_rows:
            context.canonical('adapter_result', selector[1], proof, selector[0])
        output_rows = outputs.get(value['compiler_output_digest'], [])
        for output in output_rows:
            metadata = output[0]
            c.require(all(metadata[k] == value[k] for k in ('run_id', 'configured_scene_revision_digest'))
                and {k: metadata['compiled_episode_packet'][k] for k in ('path', 'digest', 'size_bytes')} == packet
                and _adapter_selector(context, metadata['adapter_result']) == selector, 'compiler_result_binding_invalid')
            probe = metadata.get('destination_native_probe_request')
            if probe is not None:
                c.require(all(value.get(a) == probe[b] for a, b in (('destination_native_probe_request_path', 'path'),
                    ('destination_native_probe_request_digest', 'digest'), ('destination_native_probe_request_document_digest', 'request_digest'))), 'probe_binding_invalid')
            else:
                c.require(not any(k in value for k in ('destination_native_probe_request_path', 'destination_native_probe_request_digest',
                    'destination_native_probe_request_document_digest')), 'probe_binding_invalid')
        sources += context.provenance(r[1] for r in output_rows)
        if not output_rows:
            context.canonical('compiler_output', value['compiler_output_digest'], proof)
            if any(k.startswith('destination_native_probe_request_') for k in value):
                probe_path = c.path(value.get('destination_native_probe_request_path'))
                c.require(c.under(probe_path, root) and c.matches(value.get('destination_native_probe_request_digest'))
                    and c.matches(value.get('destination_native_probe_request_document_digest')), 'probe_invalid')
                context.missing('destination_probe', 'raw_size_unavailable', [proof], probe_path,
                    {'sha256': value['destination_native_probe_request_digest'], 'canonical_digest': value['destination_native_probe_request_document_digest']})
        context.member(packet['path'], 'compiled_episode_packet', packet, sources)
        observations.append(c.observation(row, kind='compilation_output', adapter_metadata_binding_verified=bool(adapter_rows),
            compiler_output_metadata_binding_verified=bool(output_rows), source_provenance=sources))
        results.setdefault((value['compilation_id'], value['result_digest']), []).append(row)
    context.compilation_results, context.adapter_results = results, adapters
    return observations
