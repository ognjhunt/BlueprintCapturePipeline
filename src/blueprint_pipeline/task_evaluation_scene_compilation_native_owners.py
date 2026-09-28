"""Retained non-policy owner metadata; no execution-authority validation."""
from __future__ import annotations

from .task_evaluation_scene_lineage_budget import _work, _work_call, _work_hash, _work_items, _work_kwargs

import hashlib
from pathlib import PurePosixPath

from . import task_evaluation_scene_compilation_owner_contracts as c

OWNER_FIELDS = {'scene_intent_digest', 'scene_attempt_id', 'scene_attempt_binding'}
RECORD_FIELDS = OWNER_FIELDS | {'schema_version', 'phase', 'team_namespace', 'scene_id', 'task_id',
    'runtime_source_bundle_digest', 'owner_attempt_digest'}
BINDING_FIELDS = {'schema_version', 'intent_id', 'intent_digest', 'attempt_id', 'source_commit', 'runtime_digest', 'input_digest'}
PHASES = {'native_task_arena_destination_qualification': 'destination', 'native_task_arena_controls': 'controls',
    'native_task_arena_construction': 'construction', 'native_task_arena_construction_after_destination': 'construction'}


def filename(activation_id, request_digest, *, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
    readable = activation_id+'-'+request_digest[7:]+'.json'
    return readable if len(readable.encode('utf-8')) <= 255 else (
        'activation-'+(_work_hash(work_budget, hashlib.sha256, activation_id.encode('utf-8')) if work_budget is not None else hashlib.sha256(activation_id.encode('utf-8'))).hexdigest()+'-'+request_digest[7:]+'.json')


def _owner(context, row, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    value, proof = row
    c.require(RECORD_FIELDS <= set(value) and value.get('schema_version') == c.SCHEMAS['native_owner_records'][0], 'owner_record_invalid', **_work_kwargs(work_budget))
    c.seal(row, 'owner_attempt_digest', **_work_kwargs(work_budget))
    context.fields(row, RECORD_FIELDS)
    c.require(all(c.matches(value.get(k), c.OWNER_ID, **_work_kwargs(work_budget)) for k in (_work_items(('scene_attempt_id', 'scene_id', 'task_id', 'team_namespace'), work_budget) if work_budget is not None else ('scene_attempt_id', 'scene_id', 'task_id', 'team_namespace')))
        and value.get('phase') in {'construction', 'destination', 'controls'}
        and all(c.matches(value.get(k), **_work_kwargs(work_budget)) for k in (_work_items(('scene_intent_digest', 'runtime_source_bundle_digest'), work_budget) if work_budget is not None else ('scene_intent_digest', 'runtime_source_bundle_digest'))), 'owner_record_invalid', **_work_kwargs(work_budget))
    binding = value.get('scene_attempt_binding')
    c.require(isinstance(binding, dict) and BINDING_FIELDS <= set(binding)
        and binding.get('schema_version') == 'task_evaluation_scene_attempt_binding.v1'
        and all(c.matches(binding.get(k), c.OWNER_ID, **_work_kwargs(work_budget)) for k in (_work_items(('intent_id', 'attempt_id'), work_budget) if work_budget is not None else ('intent_id', 'attempt_id')))
        and c.matches(binding.get('source_commit'), c.COMMIT, **_work_kwargs(work_budget))
        and all(c.matches(binding.get(k), **_work_kwargs(work_budget)) for k in (_work_items(('intent_digest', 'runtime_digest', 'input_digest'), work_budget) if work_budget is not None else ('intent_digest', 'runtime_digest', 'input_digest')))
        and binding['attempt_id'] == value['scene_attempt_id'] and binding['intent_digest'] == value['scene_intent_digest'], 'owner_binding_invalid', **_work_kwargs(work_budget))
    if set(binding) != BINDING_FIELDS:
        context.fields(row, set(value) - {'scene_attempt_binding'})
    intent = context.decoded['intent'][0][0]
    c.require(binding['intent_id'] == context.intent_id and binding['intent_digest'] == intent['intent_digest']
        and value['task_id'] == intent['request']['task']['task_id'], 'owner_intent_invalid', **_work_kwargs(work_budget))
    attempt_path = c.child(context.roots['intent_root'], context.intent_id, 'attempts', binding['attempt_id']+'.json', **_work_kwargs(work_budget))
    for attempt in (_work_items(context.by_path['attempts'].get(attempt_path, []), work_budget) if work_budget is not None else context.by_path['attempts'].get(attempt_path, [])):
        if attempt[0].get('schema_version') == 'task_evaluation_scene_attempt.v1':
            c.seal(attempt, 'attempt_digest', cross=True, **_work_kwargs(work_budget))
            c.require(all(attempt[0].get(k) == binding[k] for k in (_work_items(('intent_id', 'intent_digest', 'attempt_id',
                'source_commit', 'runtime_digest', 'input_digest'), work_budget) if work_budget is not None else ('intent_id', 'intent_digest', 'attempt_id',
                'source_commit', 'runtime_digest', 'input_digest'))), 'owner_attempt_invalid', **_work_kwargs(work_budget))
    return binding


def _originals(context, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    links = {}
    for row in (_work_items(context.known('preparation_links', 'task_evaluation_scene_preparation_link.v1', 'link_digest'), work_budget) if work_budget is not None else context.known('preparation_links', 'task_evaluation_scene_preparation_link.v1', 'link_digest')):
        links.setdefault(row[0]['request_digest'], []).append(row)
    envelopes = {}
    for row in (_work_items(context.known('preparation_envelopes', 'task_evaluation_launch_preparation_envelope.v1', 'envelope_digest'), work_budget) if work_budget is not None else context.known('preparation_envelopes', 'task_evaluation_launch_preparation_envelope.v1', 'envelope_digest')):
        if row[0]['request'].get('run_mode') == 'scene_configuration':
            envelopes.setdefault(row[0]['request_digest'], []).append(row)
    return links, envelopes


def _available_owner(context, row, request, native_rows, originals, standalone, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    owner, proof = row
    binding = _owner(context, row, **_work_kwargs(work_budget))
    c.require(owner['phase'] == PHASES[request['lane']] and owner['team_namespace'] == request['team_namespace']
        and binding['source_commit'] == request['expected_production_commit'], 'owner_activation_invalid', **_work_kwargs(work_budget))
    sources = context.provenance((proof,))
    exact_path = c.child(context.roots['activation_output_root'], request['activation_id'], 'scene_owner_attempt.json', **_work_kwargs(work_budget))
    for stored in (_work_items(standalone.get(exact_path, []), work_budget) if work_budget is not None else standalone.get(exact_path, [])):
        c.require(all(stored[0][k] == owner[k] for k in (_work_items(RECORD_FIELDS - {'scene_attempt_binding', 'owner_attempt_digest'}, work_budget) if work_budget is not None else RECORD_FIELDS - {'scene_attempt_binding', 'owner_attempt_digest'}))
            and all(stored[0]['scene_attempt_binding'][k] == binding[k] for k in (_work_items(BINDING_FIELDS, work_budget) if work_budget is not None else BINDING_FIELDS)), 'owner_copy_invalid', **_work_kwargs(work_budget))
    sources += context.provenance(r[1] for r in (_work_items(standalone.get(exact_path, []), work_budget) if work_budget is not None else standalone.get(exact_path, [])))
    for native in (_work_items(native_rows, work_budget) if work_budget is not None else native_rows):
        preparation = native[0]['request']
        c.require(binding['input_digest'] != native[0]['request_digest'], 'owner_original_input_invalid', **_work_kwargs(work_budget))
        c.require(owner['scene_id'] == preparation['scene']['identity']['id'] and owner['task_id'] == preparation['task']['identity']['id']
            and owner['runtime_source_bundle_digest'] == preparation['execution_adapter']['runtime_source_bundle']['digest'], 'owner_preparation_invalid', **_work_kwargs(work_budget))
        if 'scene_intent_digest' in preparation:
            c.require(preparation['scene_intent_digest'] == owner['scene_intent_digest'], 'owner_preparation_invalid', **_work_kwargs(work_budget))
    attempt_path = c.child(context.roots['intent_root'], context.intent_id, 'attempts', binding['attempt_id']+'.json', **_work_kwargs(work_budget))
    attempts = context.by_path['attempts'].get(attempt_path, [])
    available_attempt = False
    for attempt in (_work_items(attempts, work_budget) if work_budget is not None else attempts):
        if attempt[0].get('schema_version') != 'task_evaluation_scene_attempt.v1':
            context.missing('owner_attempt', 'unsupported_retained_schema', [attempt[1]], attempt_path)
            continue
        c.seal(attempt, 'attempt_digest', cross=True, **_work_kwargs(work_budget))
        c.require(all(attempt[0].get(k) == binding[k] for k in (_work_items(('intent_id', 'intent_digest', 'attempt_id', 'source_commit', 'runtime_digest', 'input_digest'), work_budget) if work_budget is not None else ('intent_id', 'intent_digest', 'attempt_id', 'source_commit', 'runtime_digest', 'input_digest'))),
            'owner_attempt_invalid', **_work_kwargs(work_budget))
        available_attempt = True
        sources += context.provenance((attempt[1],))
    if not available_attempt:
        context.missing('owner_attempt', 'owner_attempt_bytes_unavailable', [proof], attempt_path)
    links, envelopes = originals
    original_links = links.get(binding['input_digest'], [])
    original_rows = envelopes.get(binding['input_digest'], [])
    for link in (_work_items(original_links, work_budget) if work_budget is not None else original_links):
        value = link[0]
        c.require(value['intent_id'] == context.intent_id and value['intent_digest'] == binding['intent_digest']
            and all(value[k] == owner[k] for k in (_work_items(('scene_id', 'task_id', 'team_namespace'), work_budget) if work_budget is not None else ('scene_id', 'task_id', 'team_namespace'))), 'owner_original_invalid', **_work_kwargs(work_budget))
    for original in (_work_items(original_rows, work_budget) if work_budget is not None else original_rows):
        value = original[0]['request']
        c.require(value['scene']['identity']['id'] == owner['scene_id'] and value['task']['identity']['id'] == owner['task_id']
            and value['team_namespace'] == owner['team_namespace'] and value.get('scene_intent_digest') == owner['scene_intent_digest'], 'owner_original_invalid', **_work_kwargs(work_budget))
    sources += context.provenance(r[1] for r in (_work_items(original_links, work_budget) if work_budget is not None else original_links))
    sources += context.provenance(r[1] for r in (_work_items(original_rows, work_budget) if work_budget is not None else original_rows))
    if not original_links or not original_rows:
        context.canonical('original_scene_configuration_request', binding['input_digest'], proof)
    known_wrappers = [row, *native_rows, *standalone.get(exact_path, [])]
    return bool(available_attempt and original_links and original_rows and native_rows
        and all(context.supported(r) for r in (_work_items(known_wrappers, work_budget) if work_budget is not None else known_wrappers))), sources


def _envelopes(context, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    by_filename = {}
    for row in (_work_items(context.known('native_activation_envelopes'), work_budget) if work_budget is not None else context.known('native_activation_envelopes')):
        envelope, proof = row
        request = envelope.get('request')
        c.require(isinstance(request, dict) and request.get('schema_version') == 'task_evaluation_launch_activation_request.v1'
            and all(c.matches(request.get(k), c.ID, **_work_kwargs(work_budget)) for k in (_work_items(('activation_id', 'team_namespace'), work_budget) if work_budget is not None else ('activation_id', 'team_namespace')))
            and c.matches(request.get('expected_production_commit'), c.COMMIT, **_work_kwargs(work_budget))
            and c.matches(envelope.get('request_digest'), **_work_kwargs(work_budget)) and envelope['request_digest'] == (_work_call(work_budget, c.canonical_digest, request) if work_budget is not None else c.canonical_digest(request))
            and all(envelope.get(k) is False for k in (_work_items(('provider_mutation_performed_inside_intake', 'catalog_mutation_performed_inside_intake',
                'standing_authorization_published_inside_intake', 'paid_execution_requested'), work_budget) if work_budget is not None else ('provider_mutation_performed_inside_intake', 'catalog_mutation_performed_inside_intake',
                'standing_authorization_published_inside_intake', 'paid_execution_requested'))), 'activation_envelope_invalid', **_work_kwargs(work_budget))
        context.fields((request, dict(proof, json_pointer='/request')), c.NATIVE_ACTIVATION_REQUEST_FIELDS)
        preparation = request.get('preparation')
        c.require(isinstance(preparation, dict) and c.matches(preparation.get('preparation_id'), c.ID, **_work_kwargs(work_budget))
            and all(c.matches(preparation.get(k), **_work_kwargs(work_budget)) for k in (_work_items(('request_digest', 'result_digest'), work_budget) if work_budget is not None else ('request_digest', 'result_digest'))), 'activation_preparation_invalid', **_work_kwargs(work_budget))
        name = filename(request['activation_id'], envelope['request_digest'], **_work_kwargs(work_budget))
        c.require(proof['path'] in {c.child(context.roots['activation_queue_root'], state, name, **_work_kwargs(work_budget)) for state in (_work_items(('pending', 'processing', 'prepared', 'blocked'), work_budget) if work_budget is not None else ('pending', 'processing', 'prepared', 'blocked'))},
            'activation_path_invalid', **_work_kwargs(work_budget))
        by_filename.setdefault(name, []).append(row)
    return by_filename


def _results(context, envelopes, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    index = {}
    for role in (_work_items(('native_activation_results', 'activation_results'), work_budget) if work_budget is not None else ('native_activation_results', 'activation_results')):
        for row in (_work_items(context.known(role, 'task_evaluation_launch_activation_result.v1', 'result_digest'), work_budget) if work_budget is not None else context.known(role, 'task_evaluation_launch_activation_result.v1', 'result_digest')):
            value, proof = row
            c.require(c.matches(value.get('activation_id'), c.ID, **_work_kwargs(work_budget)) and isinstance(value.get('status'), str)
                and str(PurePosixPath(proof['path']).parent) == c.child(context.roots['activation_queue_root'], 'results', **_work_kwargs(work_budget)), 'activation_result_invalid', **_work_kwargs(work_budget))
            name = PurePosixPath(proof['path']).name
            request_hex = name[-69:-5]
            c.require(len(request_hex) == 64 and all(ch in '0123456789abcdef' for ch in (_work_items(request_hex, work_budget) if work_budget is not None else request_hex))
                and name == filename(value['activation_id'], 'sha256:'+request_hex, **_work_kwargs(work_budget)), 'activation_result_path_invalid', **_work_kwargs(work_budget))
            if name not in envelopes:
                context.canonical('native_activation_request', 'sha256:'+request_hex, proof)
            if value['status'] == 'blocked':
                c.require(value.get('provider_mutation_performed') is False and value.get('paid_execution_requested') is False
                    and isinstance(value.get('blockers'), list), 'activation_scope_invalid', **_work_kwargs(work_budget))
            if value['status'] != 'profile_authority_materialized_no_execution':
                context.missing('native_activation_result', 'unsupported_or_blocked_retained_status', [proof])
                continue
            c.require(all(c.matches(value.get(k), c.ID, **_work_kwargs(work_budget)) for k in (_work_items(('preparation_id', 'team_namespace', 'lane', 'profile_id'), work_budget) if work_budget is not None else ('preparation_id', 'team_namespace', 'lane', 'profile_id')))
                and c.matches(value.get('source_commit'), c.COMMIT, **_work_kwargs(work_budget))
                and all(c.matches(value.get(k), **_work_kwargs(work_budget)) for k in (_work_items(('preparation_result_digest', 'release_window_digest', 'profile_digest',
                    'profile_publication_receipt_digest', 'standing_authorization_digest'), work_budget) if work_budget is not None else ('preparation_result_digest', 'release_window_digest', 'profile_digest',
                    'profile_publication_receipt_digest', 'standing_authorization_digest')))
                and all(value.get(k) is True for k in (_work_items(('full_byte_activation_reference_readback_passed', 'profile_publication_performed',
                    'catalog_mutation_performed', 'standing_authorization_published'), work_budget) if work_budget is not None else ('full_byte_activation_reference_readback_passed', 'profile_publication_performed',
                    'catalog_mutation_performed', 'standing_authorization_published')))
                and all(value.get(k) is False for k in (_work_items(('provider_mutation_performed', 'paid_execution_requested'), work_budget) if work_budget is not None else ('provider_mutation_performed', 'paid_execution_requested')))
                and isinstance(value.get('blockers'), list), 'activation_result_invalid', **_work_kwargs(work_budget))
            context.fields(row, c.FIELD_SETS['native_activation_results'])
            name = PurePosixPath(proof['path']).name
            for envelope in (_work_items(envelopes.get(name, []), work_budget) if work_budget is not None else envelopes.get(name, [])):
                request = envelope[0]['request']
                c.require(all(value[a] == request[b] for a, b in (_work_items((('activation_id', 'activation_id'), ('team_namespace', 'team_namespace'),
                    ('lane', 'lane'), ('source_commit', 'expected_production_commit')), work_budget) if work_budget is not None else (('activation_id', 'activation_id'), ('team_namespace', 'team_namespace'),
                    ('lane', 'lane'), ('source_commit', 'expected_production_commit'))))
                    and value['preparation_id'] == request['preparation']['preparation_id']
                    and value['preparation_result_digest'] == request['preparation']['result_digest'], 'activation_result_binding_invalid', **_work_kwargs(work_budget))
            for field in (_work_items(('profile_publication_receipt_digest', 'standing_authorization_digest'), work_budget) if work_budget is not None else ('profile_publication_receipt_digest', 'standing_authorization_digest')):
                context.missing(field, 'raw_selector_size_unavailable', [proof], selector={'sha256': value[field]})
            context.canonical('release_window', value['release_window_digest'], proof)
            index.setdefault(name, []).append(row)
    return index


def inventory(context, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    originals = _originals(context, **_work_kwargs(work_budget))
    envelopes = _envelopes(context, **_work_kwargs(work_budget))
    results = _results(context, envelopes, **_work_kwargs(work_budget))
    standalone = {}
    for row in (_work_items(context.known('native_owner_records'), work_budget) if work_budget is not None else context.known('native_owner_records')):
        _owner(context, row, **_work_kwargs(work_budget))
        p = PurePosixPath(row[1]['path'])
        c.require(p.name == 'scene_owner_attempt.json' and c.matches(p.parent.name, c.ID, **_work_kwargs(work_budget))
            and str(p.parent.parent) == context.roots['activation_output_root'], 'owner_path_invalid', **_work_kwargs(work_budget))
        standalone.setdefault(row[1]['path'], []).append(row)
    profiles = {}
    for row in (_work_items(context.known('launch_profiles', 'task_evaluation_launch_profile.v1', 'profile_digest'), work_budget) if work_budget is not None else context.known('launch_profiles', 'task_evaluation_launch_profile.v1', 'profile_digest')):
        profiles.setdefault((row[0]['profile_id'], row[0]['profile_digest']), []).append(row)
    observations = context.rows()
    for name, versions in (_work_items(envelopes.items(), work_budget) if work_budget is not None else envelopes.items()):
        for envelope in (_work_items(versions[:1], work_budget) if work_budget is not None else versions[:1]):
            request, proof = envelope[0]['request'], envelope[1]
            if request.get('lane') not in PHASES:
                context.missing('native_owner', 'unsupported_retained_lane', [proof])
                continue
            prep = request['preparation']
            native_rows = context.native_preparations.get((prep['preparation_id'], prep['request_digest']), [])
            final_rows = context.native_results.get((prep['preparation_id'], prep['result_digest']), [])
            for native in (_work_items(native_rows, work_budget) if work_budget is not None else native_rows):
                c.require(native[0]['request']['team_namespace'] == request['team_namespace']
                    and native[0]['request']['expected_production_commit'] == request['expected_production_commit'], 'activation_preparation_binding_invalid', **_work_kwargs(work_budget))
            for final in (_work_items(final_rows, work_budget) if work_budget is not None else final_rows):
                c.require(final[0]['status'] == 'queued_for_production_episode_compilation'
                    and final[0]['team_namespace'] == request['team_namespace'] and final[0]['source_commit'] == request['expected_production_commit'],
                    'activation_preparation_binding_invalid', **_work_kwargs(work_budget))
                expected_name = prep['preparation_id']+'-'+prep['request_digest'][7:]+'.json'
                c.require(PurePosixPath(final[1]['path']).name == expected_name, 'activation_preparation_binding_invalid', **_work_kwargs(work_budget))
            sources = context.provenance(r[1] for r in (_work_items(versions, work_budget) if work_budget is not None else versions))
            sources += context.provenance(r[1] for r in (_work_items(native_rows, work_budget) if work_budget is not None else native_rows))
            sources += context.provenance(r[1] for r in (_work_items(final_rows, work_budget) if work_budget is not None else final_rows))
            owner = request.get('authorization', {}).get('scene_owner_attempt')
            if not isinstance(owner, dict) or owner.get('schema_version') != c.SCHEMAS['native_owner_records'][0]:
                context.missing('native_owner', 'owner_record_bytes_unavailable_or_unsupported', [proof])
                observations.append(c.observation(envelope, kind='native_owner', owner_metadata_binding_verified=False,
                    profile_metadata_binding_verified=False, source_provenance=sources, **_work_kwargs(work_budget)))
                continue
            owner_row = context.nested(owner, proof, '/request/authorization/scene_owner_attempt', 'owner_attempt_digest')
            bound, owner_sources = _available_owner(context, owner_row, request, native_rows, originals, standalone, **_work_kwargs(work_budget))
            sources += owner_sources
            sources += context.provenance(dict(v[1], json_pointer='/request/authorization/scene_owner_attempt',
                seal_field='owner_attempt_digest', seal_digest=owner['owner_attempt_digest']) for v in (_work_items(versions[1:], work_budget) if work_budget is not None else versions[1:]))
            matched_profiles, supported_results, supported_profiles = False, True, True
            for result in (_work_items(results.get(name, []), work_budget) if work_budget is not None else results.get(name, [])):
                supported_results = supported_results and context.supported(result)
                profile_rows = profiles.get((result[0]['profile_id'], result[0]['profile_digest']), [])
                for profile in (_work_items(profile_rows, work_budget) if work_budget is not None else profile_rows):
                    supported_profiles = supported_profiles and context.supported(profile)
                    c.require(profile[0]['source_commit'] == request['expected_production_commit']
                        and all(profile[0].get(k) == owner[k] for k in (_work_items(OWNER_FIELDS - {'scene_attempt_binding'}, work_budget) if work_budget is not None else OWNER_FIELDS - {'scene_attempt_binding'}))
                        and isinstance(profile[0].get('scene_attempt_binding'), dict)
                        and all(profile[0]['scene_attempt_binding'].get(k) == owner['scene_attempt_binding'][k] for k in (_work_items(BINDING_FIELDS, work_budget) if work_budget is not None else BINDING_FIELDS)), 'owner_profile_invalid', **_work_kwargs(work_budget))
                matched_profiles = matched_profiles or bool(profile_rows)
                sources += context.provenance(r[1] for r in (_work_items(profile_rows, work_budget) if work_budget is not None else profile_rows))
                sources += context.provenance((result[1],))
                if not profile_rows:
                    context.canonical('launch_profile', result[0]['profile_digest'], result[1])
            bound = bound and bool(final_rows) and all(context.supported(r) for r in (_work_items((*versions, *final_rows), work_budget) if work_budget is not None else (*versions, *final_rows)))
            if bound:
                context.member(c.child(context.roots['activation_output_root'], request['activation_id'], **_work_kwargs(work_budget)), 'native_activation_workspace',
                    {'activation_id': request['activation_id'], 'intent_id': context.intent_id, 'attempt_id': owner['scene_attempt_id']}, sources)
            observations.append(c.observation(envelope, kind='native_owner', owner_metadata_binding_verified=bound,
                profile_metadata_binding_verified=bool(matched_profiles) and bound and supported_results and supported_profiles, source_provenance=sources,
                configured_payload_to_runtime_bundle_equivalence_verified=False, **_work_kwargs(work_budget)))
    return observations
