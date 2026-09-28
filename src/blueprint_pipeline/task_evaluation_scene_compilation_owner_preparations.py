"""Native preparation/compilation handoff metadata; no payload reads."""
from __future__ import annotations

from pathlib import PurePosixPath

from . import task_evaluation_scene_compilation_owner_contracts as c

STATES = {'pending', 'processing', 'awaiting_source_preparation', 'awaiting_capacity', 'materialized', 'completed', 'blocked'}
BASE = {'schema_version', 'status', 'preparation_id', 'run_id', 'team_namespace', 'source_commit', 'reference_count',
    'unique_object_count', 'content_addressed_reuse_count', 'references', 'full_byte_service_account_readback_passed',
    'service_account', 'service_account_uid', 'provider_mutation_performed', 'catalog_mutation_performed',
    'paid_execution_requested', 'observed_at_iso', 'result_digest'}
HANDOFF = {'run_mode', 'configured_scene_revision_digest', 'configured_scene_bundle_digest', 'episode_compilation_id',
    'episode_compilation_queue_envelope_digest', 'episode_compilation_queue_receipt_digest',
    'customer_supplied_prebuilt_episode_packet', 'construction_packet_materialized', 'automatic_progression_required'}
PRE = 'inputs_materialized_awaiting_construction_adapter'
FINAL = 'queued_for_production_episode_compilation'


def _route(context, path):
    parent = str(PurePosixPath(path).parent.parent)
    matched = [r for r in context.routes if parent == r['queue_root']]
    c.require(len(matched) == 1, 'preparation_route_invalid')
    return matched[0]


def _references(context, rows):
    c.require(isinstance(rows, list) and 1 <= len(rows) <= context.limits['MAX_REFERENCES'], 'references_invalid')
    identities, contracts, reuse = set(), set(), 0
    for row in rows:
        c.require(isinstance(row, dict) and {'contract_path', 'uri', 'digest', 'size_bytes', 'materialized_path',
            'content_addressed_reuse', 'full_byte_service_account_readback_passed'} <= set(row), 'reference_invalid')
        name = c.text(row['contract_path'], 512)
        c.require(not any(ch.isspace() for ch in name) and name not in contracts, 'reference_contract_invalid')
        contracts.add(name)
        c.require(isinstance(row['uri'], str) and len(row['uri']) <= 4096 and '://' in row['uri']
            and not any(ch.isspace() for ch in row['uri']) and c.matches(row['digest'])
            and type(row['size_bytes']) is int and row['size_bytes'] > 0
            and type(row['content_addressed_reuse']) is bool and row['full_byte_service_account_readback_passed'] is True,
            'reference_invalid')
        c.path(row['materialized_path'])
        c.require(any(c.under(row['materialized_path'], route['input_root']) for route in context.routes), 'reference_path_invalid')
        context.size(row['digest'], row['size_bytes'])
        identities.add((row['digest'], row['size_bytes']))
        reuse += int(row['content_addressed_reuse'])
    return len(identities), reuse


def _envelopes(context):
    index = {}
    for role in ('native_preparation_envelopes', 'sam_parent_envelopes', 'preparation_envelopes'):
        for row in context.known(role, 'task_evaluation_launch_preparation_envelope.v1', 'envelope_digest'):
            value, proof = row
            request = value.get('request')
            c.require(isinstance(request, dict), 'preparation_request_invalid')
            if request.get('run_mode') not in {'episode_evaluation', 'destination_qualification'}:
                if role == 'native_preparation_envelopes':
                    context.missing(role, 'unsupported_retained_request', [proof])
                continue
            c.require(request.get('schema_version') == 'task_evaluation_launch_preparation_request.v1'
                and all(c.matches(request.get(k), c.ID) for k in ('preparation_id', 'run_id', 'team_namespace'))
                and c.matches(request.get('expected_production_commit'), c.COMMIT)
                and c.matches(value.get('request_digest')) and value['request_digest'] == c.canonical_digest(request)
                and request.get('construction', {}).get('mode') == 'reuse_configured_scene'
                and request.get('task', {}).get('binding_mode') == 'reuse_configured_template', 'preparation_request_invalid')
            stem = request['preparation_id']+'-'+value['request_digest'][7:]+'.json'
            route = _route(context, proof['path'])
            c.require(proof['path'] in {c.child(route['queue_root'], state, stem) for state in STATES}, 'preparation_path_invalid')
            index.setdefault((request['preparation_id'], value['request_digest']), []).append(row)
    return index


def _compilations(context):
    index = {}
    for row in context.known('compilation_envelopes', 'task_evaluation_episode_compilation_envelope.v1', 'envelope_digest'):
        value, proof = row
        c.require(c.matches(value.get('compilation_id'), c.ID) and value.get('preparation_id') == value['compilation_id']
            and all(c.matches(value.get(k), c.ID) for k in ('run_id', 'team_namespace'))
            and c.matches(value.get('expected_production_commit'), c.COMMIT)
            and c.matches(value.get('configured_scene_revision_digest')) and c.matches(value.get('preparation_result_digest'))
            and all(value.get(k) is True for k in ('automatic_progression_required', 'robot_specific_episode_packet_compiled_in_production',
                'production_compiler_owns_episode_packet'))
            and all(value.get(k) is False for k in ('customer_supplied_prebuilt_episode_packet', 'provider_mutation_performed', 'paid_execution_requested')),
            'compilation_envelope_invalid')
        request = value.get('request')
        c.require(isinstance(request, dict) and request.get('preparation_id') == value['compilation_id']
            and request.get('run_id') == value['run_id'] and request.get('team_namespace') == value['team_namespace']
            and request.get('expected_production_commit') == value['expected_production_commit'], 'compilation_request_invalid')
        _references(context, value.get('materialized_references'))
        bundle = value.get('configured_scene_bundle')
        c.require(isinstance(bundle, dict) and any(bundle == r for r in value['materialized_references'])
            and bundle.get('contract_path') == 'scene.configured_revision.configured_scene_bundle', 'compilation_bundle_invalid')
        p = PurePosixPath(proof['path'])
        c.require(str(p.parent.parent) == context.roots['compilation_queue_root'] and p.parent.name in {'pending', 'processing', 'completed', 'blocked'}
            and p.name == value['compilation_id']+'-'+value['envelope_digest'][7:]+'.json', 'compilation_path_invalid')
        index.setdefault((value['compilation_id'], value['envelope_digest']), []).append(row)
    return index


def _receipts(context, compilations):
    index = {}
    for row in context.known('compilation_intake_receipts'):
        context.metadata(row)
        value, proof = row
        c.require(value.get('status') == FINAL and c.matches(value.get('compilation_id'), c.ID)
            and c.matches(value.get('run_id'), c.ID) and c.matches(value.get('configured_scene_revision_digest'))
            and c.matches(value.get('envelope_digest')) and type(value.get('created')) is bool
            and value.get('automatic_progression_required') is True
            and value.get('provider_mutation_performed') is False and value.get('paid_execution_requested') is False, 'intake_invalid')
        c.path(value.get('queue_path'))
        filename = value['compilation_id']+'-'+value['envelope_digest'][7:]+'.json'
        c.require(value['queue_path'] in {c.child(context.roots['compilation_queue_root'], state, filename)
            for state in ('pending', 'processing', 'completed', 'blocked')}, 'intake_path_invalid')
        # The receipt records its publisher state; later retained copies move
        # between queue states without changing the immutable envelope identity.
        for envelope in compilations.get((value['compilation_id'], value['envelope_digest']), []):
            c.require(all(value[k] == envelope[0][k] for k in ('run_id', 'configured_scene_revision_digest')), 'intake_binding_invalid')
        index.setdefault((value['compilation_id'], value['receipt_digest']), []).append(row)
    return index


def _result(context, row):
    value, proof = row
    c.require(all(c.matches(value.get(k), c.ID) for k in ('preparation_id', 'run_id', 'team_namespace'))
        and c.matches(value.get('source_commit'), c.COMMIT), 'preparation_result_invalid')
    route = _route(context, proof['path'])
    p = PurePosixPath(proof['path'])
    suffix = p.name[len(value['preparation_id'])+1:-5]
    c.require(str(p.parent) == c.child(route['queue_root'], 'results') and p.name.startswith(value['preparation_id']+'-')
        and p.name.endswith('.json') and len(suffix) == 64 and all(char in '0123456789abcdef' for char in suffix),
        'preparation_result_path_invalid')
    if value.get('status') not in {PRE, FINAL}:
        context.missing('native_preparation_result', 'unsupported_retained_status', [proof])
        return False
    unique, reuse = _references(context, value.get('references'))
    c.require(type(value.get('reference_count')) is int and value['reference_count'] == len(value['references'])
        and type(value.get('unique_object_count')) is int and value['unique_object_count'] == unique
        and type(value.get('content_addressed_reuse_count')) is int and value['content_addressed_reuse_count'] == reuse
        and value.get('full_byte_service_account_readback_passed') is True
        and all(value.get(k) is False for k in ('provider_mutation_performed', 'catalog_mutation_performed', 'paid_execution_requested'))
        and type(value.get('service_account_uid')) is int and value['service_account_uid'] >= 0, 'preparation_result_invalid')
    c.text(value.get('service_account'))
    c.text(value.get('observed_at_iso'), 128)
    return True


def _revision(context, revisions, request, compilation, final, proof):
    reference = request.get('scene', {}).get('configured_revision')
    row = context.selected(reference, proof, {'configured_revisions'})
    if row is None:
        return False
    value = row[0]
    c.require(value.get('status') == 'configured' and c.matches(value.get('source_commit'), c.COMMIT)
        and value.get('revision_digest') == compilation['configured_scene_revision_digest'] == final['configured_scene_revision_digest']
        and value.get('scene_identity') == request['scene']['identity']
        and value.get('task_template', {}).get('identity') == request['task']['identity'], 'revision_binding_invalid')
    bundle = compilation['configured_scene_bundle']
    c.require(value.get('configured_scene_bundle') == {k: bundle[k] for k in ('uri', 'digest', 'size_bytes')}
        and final['configured_scene_bundle_digest'] == bundle['digest'], 'revision_bundle_invalid')
    return True


def inventory(context):
    envelopes, compilations = _envelopes(context), _compilations(context)
    receipts = _receipts(context, compilations)
    revisions = context.known('configured_revisions')
    for revision in revisions:
        context.metadata(revision)
    results, finals, observations = {}, [], context.rows()
    for role in ('native_preparation_results', 'preparation_results'):
        for row in context.known(role, 'task_evaluation_launch_preparation_result.v1', 'result_digest'):
            if role == 'preparation_results' and row[0].get('status') != FINAL:
                continue  # Old scene result interpretation is kept verbatim.
            if _result(context, row):
                results.setdefault((row[0]['preparation_id'], row[0]['result_digest']), []).append(row)
                if row[0]['status'] == FINAL:
                    finals.append(row)
    context.native_preparations, context.native_results, context.native_compilations = envelopes, results, compilations
    for row in finals:
        final, proof = row
        for field in ('configured_scene_revision_digest', 'configured_scene_bundle_digest', 'episode_compilation_queue_envelope_digest',
                      'episode_compilation_queue_receipt_digest'):
            c.require(c.matches(final.get(field)), 'handoff_invalid')
        c.require(final.get('episode_compilation_id') == final['preparation_id']
            and final.get('run_mode') in {'episode_evaluation', 'destination_qualification'}
            and final.get('customer_supplied_prebuilt_episode_packet') is False and final.get('construction_packet_materialized') is False
            and final.get('automatic_progression_required') is True, 'handoff_invalid')
        candidates = compilations.get((final['preparation_id'], final['episode_compilation_queue_envelope_digest']), [])
        verified, strength = False, None
        sources = context.provenance((proof,))
        if candidates:
            # Same seal identifies identical canonical metadata; raw formatting
            # and queue-state copies remain separately discoverable provenance.
            compilation = candidates[0][0]
            c.require(all(final[a] == compilation[b] for a, b in (('run_id', 'run_id'), ('team_namespace', 'team_namespace'),
                ('source_commit', 'expected_production_commit'))), 'handoff_binding_invalid')
            request = compilation['request']
            c.require(final['run_mode'] == request['run_mode'] and final['references'] == compilation['materialized_references'], 'handoff_binding_invalid')
            request_digest = c.canonical_digest(request)
            stem = final['preparation_id']+'-'+request_digest[7:]+'.json'
            route = _route(context, proof['path'])
            c.require(proof['path'] == c.child(route['queue_root'], 'results', stem), 'preparation_result_path_invalid')
            parent_rows = envelopes.get((final['preparation_id'], request_digest), [])
            for parent in parent_rows:
                c.require(parent[0]['request'] == request, 'preparation_request_binding_invalid')
            sources += context.provenance(p[1] for p in parent_rows)
            sources += context.provenance(p[1] for p in candidates)
            _revision(context, revisions, request, compilation, final, proof)
            intake_rows = receipts.get((final['preparation_id'], final['episode_compilation_queue_receipt_digest']), [])
            for intake in intake_rows:
                c.require(intake[0]['envelope_digest'] == compilation['envelope_digest'], 'handoff_intake_invalid')
            if not intake_rows:
                context.canonical('compilation_intake', final['episode_compilation_queue_receipt_digest'], proof)
            if set(final) in (BASE | HANDOFF, BASE | HANDOFF | {'policy_run_plan'}):
                inverse = {key: value for key, value in final.items() if key not in HANDOFF | {'result_digest'}}
                inverse['status'] = PRE
                context.emission_budget.reserve_row(inverse)  # Charge bounded temporary before canonical encoding/hash.
                derived = c.canonical_digest(inverse, digest_field='result_digest')
                c.require(derived == compilation['preparation_result_digest'], 'pre_handoff_inverse_invalid')
                retained = results.get((final['preparation_id'], derived), [])
                for actual in retained:
                    c.require(actual[0]['status'] == PRE and {k: v for k, v in actual[0].items() if k != 'result_digest'} == inverse,
                              'pre_handoff_raw_binding_invalid')
                sources += context.provenance(p[1] for p in retained)
                strength = 'retained_raw_and_derived_metadata' if retained else 'derived_metadata_inverse'
                verified = True
                if 'policy_run_plan' in inverse:
                    context.missing('policy_run_plan', 'policy_semantics_deferred', [proof])
            else:
                context.missing('pre_handoff_inverse', 'unknown_retained_field_set', [proof])
        else:
            context.canonical('compilation_envelope', final['episode_compilation_queue_envelope_digest'], proof)
        observations.append(c.observation(row, pre_handoff_binding_verified=verified, pre_handoff_proof_strength=strength,
            pre_handoff_canonical_digest=candidates[0][0]['preparation_result_digest'] if verified else None,
            source_provenance=sources, historical_raw_bytes_created=False))
    return observations
