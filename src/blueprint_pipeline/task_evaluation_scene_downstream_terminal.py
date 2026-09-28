"""Pure historical terminal/pointer joins, including protected nonexecution."""
from __future__ import annotations

from pathlib import PurePosixPath

from . import task_evaluation_scene_downstream_contracts as c

NONEXECUTION = {'prepared_no_execution', 'blocked_before_paid_dispatch', 'blocked_awaiting_website_notification',
                'blocked_without_provider_allocation', 'blocked_without_provider_allocation_awaiting_notification'}
PROJECTION_PATH = 'artifacts/result_delivery/policy_canary_result_projection.json'
SYNC_PATH = 'artifacts/result_delivery/policy_canary_webapp_sync.json'
SCHEMAS = {'canary_projections': 'task_evaluation_policy_canary_result_projection.v1',
           'canary_syncs': 'task_evaluation_policy_canary_webapp_sync_result.v1',
           'provider_zero_receipts': 'task_evaluation_policy_canary_vast_provider_zero.v1',
           'canary_offload_pointers': 'control_plane_evidence_offload_pointer.v1',
           'terminal_publications': 'task_evaluation_scene_terminal_result_publication.v1'}


def _relative(value):
    c.require(isinstance(value, str) and 0 < len(value) <= c.retained.MAX_PATH_BYTES, 'pointer_member_invalid')
    c.path('/' + value)
    c.require(not value.startswith('/') and len(value.split('/')) <= c.retained.MAX_PATH_COMPONENTS, 'pointer_member_invalid')


def _members(context, value):
    members = value.get('members')
    c.require(isinstance(members, list) and type(value.get('member_count')) is int
              and value['member_count'] == len(members), 'pointer_members_invalid')
    seen = set()
    for member in members:
        context.consume()
        c.require(isinstance(member, dict) and set(member) == {'relative_path', 'size_bytes', 'sha256'}, 'pointer_member_invalid')
        _relative(member['relative_path'])
        c.require(type(member['size_bytes']) is int and member['size_bytes'] >= 0 and c.matches(member['sha256'])
                  and member['relative_path'] not in seen, 'pointer_member_invalid')
        context.size(member['sha256'], member['size_bytes'])
        seen.add(member['relative_path'])
    return {m['relative_path']: m for m in members}


def _nonexecution(row):
    value, _ = row
    field = 'blocked_result_digest' if value.get('schema_version') == 'task_evaluation_policy_canary_preprovider_blocked.v1' else 'receipt_digest'
    c.seal(row, field)
    c.require(value.get('run_kind') == 'internal_policy_canary' and c.matches(value.get('run_id'), c.ID), 'dispatch_invalid')
    if value.get('status') not in NONEXECUTION:
        return field, 'executed'
    c.require(value.get('claim_ceiling') == 'diagnostic_policy_execution', 'nonexecution_invalid')
    if value.get('terminal_result_kind') == 'definite_provider_create_refusal':
        c.require(all(value.get(k) is True for k in ('provider_call_reached', 'provider_zero_required', 'paid_execution_requested'))
                  and value.get('provider_zero_not_applicable') is False
                  and value.get('provider_mutation_performed') is False and value.get('provider_allocation_performed') is False,
                  'provider_refusal_invalid')
        return field, 'provider_refusal'
    c.require(all(value.get(k) is False for k in ('provider_mutation_performed', 'provider_allocation_performed',
              'provider_zero_required', 'paid_execution_requested', 'automatic_retry_authorized', 'automatic_retry_performed', 'provider_call_reached'))
              and value.get('provider_zero_not_applicable') is True and type(value.get('retry_cap')) is int and value['retry_cap'] == 0
              and not {'provider_zero', 'policy_canary_result_projection', 'terminal_result_publication'}.intersection(value), 'nonexecution_invalid')
    return field, 'nonexecution'


def _validate(context):
    paths, dispatch_types, pointers = {}, {}, {}
    roots = context.roots
    for role in ('canary_dispatches', *SCHEMAS, 'allocator_results'):
        paths[role] = {}
        for row in context.decoded[role]:
            value, proof = row
            p = PurePosixPath(proof['path'])
            if role == 'canary_offload_pointers':
                c.require(p.name.endswith('.offloaded.v1.json') and c.under(proof['path'], roots['policy_canary_root']), 'terminal_path_invalid')
            elif role == 'terminal_publications':
                c.require(p.name == 'terminal_result_publication.json' and c.under(proof['path'], roots['terminal_result_root']), 'terminal_path_invalid')
            else:
                c.require(c.under(proof['path'], roots['policy_canary_root']) or c.under(proof['path'], roots['terminal_result_root']), 'terminal_path_invalid')
            paths[role].setdefault(proof['path'], []).append(row)
            schema = value.get('schema_version')
            if role == 'canary_dispatches':
                c.require(p.name in {'dispatch_receipt.json', 'preprovider_blocked.json', 'no_provider_allocation_blocked.json',
                          'policy_canary_nonexecution.json'}, 'terminal_path_invalid')
                if schema not in {'task_evaluation_policy_canary_dispatch.v1', 'task_evaluation_policy_canary_preprovider_blocked.v1'}:
                    continue
                field, kind = _nonexecution(row)
                if p.name == 'preprovider_blocked.json':
                    c.require(field == 'blocked_result_digest', 'dispatch_role_invalid')
                if p.name in {'dispatch_receipt.json', 'no_provider_allocation_blocked.json'}:
                    c.require(field == 'receipt_digest', 'dispatch_role_invalid')
                dispatch_types[proof['sha256']] = (field, kind)
                if kind == 'nonexecution' and value.get('allocator_result') is not None and c.under(proof['path'], roots['policy_canary_root']):
                    allocator = value['allocator_result']
                    c.require(isinstance(allocator, dict) and c.under(allocator.get('path'), str(p.parent)), 'allocator_reference_invalid')
            elif schema == SCHEMAS.get(role):
                if role == 'canary_projections':
                    c.seal(row, 'projection_digest', cross=True)
                    c.require(c.matches(value.get('run_id'), c.ID) and isinstance(value.get('result_status'), str)
                              and all(c.matches(value.get(k)) for k in ('request_digest', 'configuration_digest', 'result_delivery_digest')), 'projection_invalid')
                elif role == 'provider_zero_receipts':
                    c.seal(row, 'receipt_digest')
                    c.require(value.get('status') == 'provider_zero_confirmed' and value.get('api_confirmed') is True
                              and value.get('provider_zero_verified') is True and type(value.get('live_instance_count')) is int
                              and value['live_instance_count'] == 0 and value.get('blockers') == [], 'provider_zero_invalid')
                elif role == 'canary_syncs':
                    c.require(c.matches(value.get('run_id'), c.ID) and isinstance(value.get('status'), str)
                              and all(c.matches(value.get(k)) for k in ('request_digest', 'configuration_digest', 'policy_canary_projection_digest'))
                              and isinstance(value.get('result_status'), str), 'sync_invalid')
                elif role == 'canary_offload_pointers':
                    c.seal(row, 'pointer_digest')
                    c.require(value.get('status') == 'offloaded' and c.matches(value.get('directory'), c.ID)
                              and p.name == value['directory'] + '.offloaded.v1.json'
                              and value.get('terminal_receipt') == 'dispatch_receipt.json'
                              and isinstance(value.get('uri'), str) and value['uri'].startswith('s3://') and '?' not in value['uri']
                              and c.matches(value.get('digest')) and type(value.get('size_bytes')) is int and value['size_bytes'] > 0, 'pointer_invalid')
                    pointers[proof['sha256']] = _members(context, value)
                elif role == 'terminal_publications':
                    c.seal(row, 'publication_digest')
                    c.require(c.matches(value.get('run_id'), c.ID) and all(c.matches(value.get(k)) for k in ('digest', 'archive_digest', 'pointer_digest'))
                              and type(value.get('size_bytes')) is int and value['size_bytes'] > 0
                              and type(value.get('archive_member_count')) is int and value['archive_member_count'] >= 0
                              and value.get('provider_allocated') is False, 'publication_invalid')
    # Validate every available exact projection edge, even when its dispatch
    # version is not the one selected by a terminal state.
    projection_raw = {(p['path'], p['sha256'], p['size_bytes']): v for v, p in context.decoded['canary_projections']}
    for value, proof in context.decoded['canary_dispatches']:
        if dispatch_types.get(proof['sha256'], (None, None))[1] != 'executed':
            continue
        ref = value.get('policy_canary_result_projection')
        if isinstance(ref, dict):
            v = projection_raw.get(tuple(ref.get(k) for k in ('path', 'sha256', 'size_bytes')))
            if v is not None:
                c.require(v.get('run_id') == value['run_id'] and v.get('projection_digest') == value.get('policy_canary_projection_digest')
                          and v.get('result_delivery_digest') == value.get('result_delivery_digest') and v.get('result_status') == value.get('status'), 'projection_binding_invalid')
    return paths, dispatch_types, pointers


def _raw_select(paths, role, ref, expected, copied):
    c.require(isinstance(ref, dict) and {'path', 'sha256', 'size_bytes'} <= set(ref)
              and ref['path'] == expected, 'dispatch_reference_path_invalid')
    # A byte-identical indexed copy proves retained bytes, never original presence.
    candidates = paths[role].get(expected, []) + paths[role].get(copied, [])
    matched = [row for row in candidates if all(row[1][k] == ref[k] for k in ('sha256', 'size_bytes'))]
    return matched


def _archive(context, paths, pointer_members, state, dispatch, projection, directory, sources):
    canary = state['canary_run_root']
    expected = canary + '.offloaded.v1.json'
    candidates = paths['canary_offload_pointers'].get(expected, [])
    candidates = [row for row in candidates if row[0].get('schema_version') == SCHEMAS['canary_offload_pointers']]
    publications = paths['terminal_publications'].get(c.child(directory, 'terminal_result_publication.json'), [])
    if len(candidates) != 1:
        context.missing('canary_offload_pointer', 'canary_offload_pointer_unavailable_or_ambiguous' if candidates else 'canary_offload_pointer_unavailable', sources, expected)
        return False
    pointer, proof = candidates[0]
    member_index = pointer_members[proof['sha256']]
    for name, row in ((PROJECTION_PATH, projection), ('dispatch_receipt.json', dispatch)):
        member = member_index.get(name)
        c.require(member is not None and all(member[k] == row[1][k] for k in ('sha256', 'size_bytes')), 'pointer_member_binding_invalid')
    if not publications:
        context.missing('terminal_publication', 'terminal_publication_unavailable', sources + [proof], c.child(directory, 'terminal_result_publication.json'))
        return True
    for publication, _ in publications:
        if publication.get('schema_version') != SCHEMAS['terminal_publications']:
            continue
        c.require(publication['run_id'] == state['run_id'] and publication['digest'] == projection[0]['projection_digest']
                  and publication['archive_digest'] == pointer['digest'] and publication['pointer_digest'] == pointer['pointer_digest']
                  and publication.get('uri') == pointer['uri'] and publication['size_bytes'] == pointer['size_bytes']
                  and publication['archive_member_count'] == pointer['member_count'], 'publication_binding_invalid')
    return True


def terminal(context, launches):
    paths, types, members = _validate(context)
    observations, indexed = [], set()
    for row in context.decoded['terminal_states']:
        state, proof = row
        p = PurePosixPath(proof['path'])
        c.require(c.under(proof['path'], context.roots['terminal_result_root']), 'terminal_path_invalid')
        schema = state.get('schema_version')
        if schema not in {'task_evaluation_scene_terminal_index_state.v1', 'task_evaluation_scene_nonexecution_terminal_state.v1'}:
            observations.append(c.observation(row, reason='terminal_state_schema_unproven'))
            continue
        c.seal(row, 'state_digest')
        nonexecution = schema == 'task_evaluation_scene_nonexecution_terminal_state.v1'
        c.require(p.name == ('nonexecution_terminal_state.json' if nonexecution else 'terminal_index_state.json')
                  and c.matches(state.get('run_id'), c.ID) and c.under(state.get('canary_run_root'), context.roots['policy_canary_root']), 'terminal_state_invalid')
        field = 'record_digest' if nonexecution else 'dispatch_receipt_digest'
        c.require(c.matches(state.get(field)), 'terminal_state_invalid')
        if not nonexecution:
            c.require(c.matches(state.get('projection_digest')), 'terminal_state_invalid')
        directory, canary = str(p.parent), state['canary_run_root']
        for role, original, copied in (('canary_projections', PROJECTION_PATH, 'policy_canary_result_projection.json'),
                                      ('canary_syncs', SYNC_PATH, 'policy_canary_webapp_sync.json')):
            for record in paths[role].get(c.child(canary, original), []) + paths[role].get(c.child(directory, copied), []):
                if record[0].get('schema_version') == SCHEMAS[role]:
                    c.require(record[0]['run_id'] == state['run_id'], 'terminal_identity_invalid')
        original_names = ('preprovider_blocked.json', 'no_provider_allocation_blocked.json', 'dispatch_receipt.json') if nonexecution else ('dispatch_receipt.json',)
        candidate_paths = [c.child(canary, name) for name in original_names] + [c.child(directory, 'policy_canary_nonexecution.json' if nonexecution else 'dispatch_receipt.json')]
        candidates = [r for path in candidate_paths for r in paths['canary_dispatches'].get(path, [])
                      if r[1]['sha256'] in types and r[0].get(types[r[1]['sha256']][0]) == state[field]]
        # Same raw bytes at original+indexed names retain both provenances.
        variants = {r[1]['sha256'] for r in candidates}
        sources, reason, archive = [proof] + [r[1] for r in candidates], None, False
        if len(variants) != 1:
            reason = 'terminal_dispatch_unavailable_or_ambiguous'
        dispatch = candidates[0] if len(variants) == 1 else None
        if dispatch:
            indexed.update(r[1]['sha256'] for r in candidates)
            value = dispatch[0]
            kind = types[dispatch[1]['sha256']][1]
            c.require(value['run_id'] == state['run_id'], 'terminal_identity_invalid')
            if nonexecution:
                c.require(value.get('status') == state.get('status'), 'nonexecution_identity_invalid')
                if kind != 'nonexecution':
                    reason = 'kept_unresolved_provider_refusal' if kind == 'provider_refusal' else 'nonexecution_scope_unproven'
                allocator = value.get('allocator_result')
                if allocator is not None:
                    c.require(isinstance(allocator, dict) and c.under(allocator.get('path'), canary), 'allocator_reference_invalid')
                    matches = _raw_select(paths, 'allocator_results', allocator, allocator['path'], allocator['path'])
                    if not matches:
                        reason = reason or 'allocator_result_bytes_unavailable'
            else:
                c.require(kind == 'executed', 'terminal_scope_invalid')
                projection_ref, sync_ref, zero_ref = (value.get(k) for k in ('policy_canary_result_projection', 'policy_canary_webapp_sync', 'provider_zero'))
                selected = [_raw_select(paths, role, ref, c.child(canary, original), c.child(directory, copied))
                            for role, ref, original, copied in (
                            ('canary_projections', projection_ref, PROJECTION_PATH, 'policy_canary_result_projection.json'),
                            ('canary_syncs', sync_ref, SYNC_PATH, 'policy_canary_webapp_sync.json'),
                            ('provider_zero_receipts', zero_ref, 'post_teardown_global_provider_zero.json', 'provider_zero_closure.json'))]
                if selected[0]:
                    projection = selected[0][0]
                    v = projection[0]
                    c.require(v.get('schema_version') == SCHEMAS['canary_projections'] and v['run_id'] == state['run_id']
                              and v['projection_digest'] == state['projection_digest'] == value.get('policy_canary_projection_digest')
                              and v['result_delivery_digest'] == value.get('result_delivery_digest') and v['result_status'] == value.get('status'), 'projection_binding_invalid')
                    if selected[1]:
                        sync = selected[1][0][0]
                        c.require(sync.get('schema_version') == SCHEMAS['canary_syncs'] and sync.get('status') == 'succeeded'
                                  and all(sync.get(k) == v[k] for k in ('run_id', 'request_digest', 'configuration_digest', 'result_status'))
                                  and sync.get('policy_canary_projection_digest') == v['projection_digest']
                                  and sync.get('notification_delivery') == value.get('notification_delivery'), 'sync_binding_invalid')
                    archive = _archive(context, paths, members, state, dispatch, projection, directory, sources)
                if selected[2]:
                    c.require(selected[2][0][0].get('schema_version') == SCHEMAS['provider_zero_receipts']
                              and zero_ref.get('provider_zero_verified') is True, 'provider_zero_binding_invalid')
                if not all(selected):
                    reason = 'terminal_reference_bytes_unavailable'
                for matches in selected:
                    sources += [r[1] for r in matches]
        bridges = [r for r in launches.get(state['run_id'], []) if str(PurePosixPath(r[0][1]['path']).parent) == directory]
        if not bridges:
            reason = reason or 'terminal_owner_bridge_unavailable'
        else:
            sources += bridges[0][1]
        if reason:
            context.missing('terminal_join', reason, sources, selector={'run_id': state['run_id'], 'canary_run_root': canary})
        else:
            context.member(canary, 'canary_evidence_workspace', {'intent_id': context.intent_id, 'run_id': state['run_id']}, sources)
        observations.append(c.observation(row, status='matched_retained_bytes' if reason is None else 'kept_unresolved', reason=reason,
                                          run_id=state['run_id'], canary_run_root=canary, archive_binding_verified=archive))
    for row in context.decoded['canary_dispatches']:
        if row[1]['sha256'] not in indexed:
            kind = types.get(row[1]['sha256'], (None, 'unknown'))[1]
            observations.append(c.observation(row, reason='kept_unresolved_provider_refusal' if kind == 'provider_refusal' else 'terminal_owner_state_unavailable'))
            context.missing('terminal_state', 'terminal_owner_state_unavailable', [row[1]], selector={'run_id': row[0].get('run_id')})
    return observations


def compilations(context):
    envelopes, observations = {}, []
    for row in context.decoded['compilation_envelopes']:
        value, proof = row
        c.require(c.under(proof['path'], context.roots['compilation_queue_root']), 'compilation_path_invalid')
        if value.get('schema_version') != 'task_evaluation_episode_compilation_envelope.v1':
            observations.append(c.observation(row, reason='compilation_schema_unproven'))
            continue
        c.seal(row, 'envelope_digest')
        request = value.get('request')
        c.require(c.matches(value.get('compilation_id'), c.ID) and value.get('preparation_id') == value['compilation_id']
                  and c.matches(value.get('expected_production_commit'), c.COMMIT) and c.matches(value.get('preparation_result_digest'))
                  and c.matches(value.get('configured_scene_revision_digest')) and isinstance(request, dict)
                  and request.get('preparation_id') == value['preparation_id']
                  and request.get('run_mode') in {'episode_evaluation', 'destination_qualification'}
                  and isinstance(request.get('construction'), dict) and request['construction'].get('mode') == 'reuse_configured_scene'
                  and isinstance(request.get('task'), dict) and request['task'].get('binding_mode') == 'reuse_configured_template'
                  and request['task'].get('configured_scene_revision_digest') == value['configured_scene_revision_digest'], 'compilation_envelope_invalid')
        filename = value['compilation_id'] + '-' + value['envelope_digest'][7:] + '.json'
        p = PurePosixPath(proof['path'])
        c.require(p.name == filename and p.parent.name in {'pending', 'processing', 'completed', 'blocked'}
                  and str(p.parent.parent) == context.roots['compilation_queue_root'], 'compilation_path_invalid')
        envelopes.setdefault(filename, []).append(row)
    for row in context.decoded['compilation_results']:
        value, proof = row
        p = PurePosixPath(proof['path'])
        c.require(str(p.parent) == c.child(context.roots['compilation_queue_root'], 'results'), 'compilation_path_invalid')
        if value.get('schema_version') != 'task_evaluation_episode_compilation_result.v1':
            observations.append(c.observation(row, reason='compilation_schema_unproven'))
            continue
        c.seal(row, 'result_digest')
        c.require(c.matches(value.get('compilation_id'), c.ID) and isinstance(value.get('status'), str), 'compilation_result_invalid')
        if value['status'] == 'compiled_for_production_launch':
            c.require(c.matches(value.get('source_commit'), c.COMMIT)
                      and all(c.matches(value.get(k)) for k in ('configured_scene_revision_digest', 'compiled_episode_packet_digest',
                              'adapter_result_digest', 'compiler_output_digest'))
                      and type(value.get('compiled_episode_packet_size_bytes')) is int and value['compiled_episode_packet_size_bytes'] > 0
                      and all(c.under(value.get(k), c.child(context.roots['compilation_output_root'], value['compilation_id']))
                              for k in ('compiled_episode_packet_path', 'adapter_result_path'))
                      and value.get('provider_mutation_performed') is False and value.get('paid_execution_requested') is False
                      and value.get('compiled_by_production') is True and value.get('customer_supplied_prebuilt_episode_packet') is False,
                      'compilation_result_invalid')
        for envelope, _ in envelopes.get(p.name, []):
            c.require(all(value.get(a) == envelope[b] for a, b in (('compilation_id', 'compilation_id'), ('run_id', 'run_id'),
                      ('team_namespace', 'team_namespace'))) and (value['status'] != 'compiled_for_production_launch'
                      or value['source_commit'] == envelope['expected_production_commit']
                      and value['configured_scene_revision_digest'] == envelope['configured_scene_revision_digest']), 'compilation_identity_invalid')
        observations.append(c.observation(row, reason='compilation_owner_join_unproven', compilation_id=value['compilation_id']))
    for role in ('compilation_envelopes', 'compilation_results'):
        for row in context.decoded[role]:
            context.missing('compilation_owner_bridge', 'compilation_owner_join_unproven', [row[1]],
                            selector={'compilation_id': row[0].get('compilation_id'), 'required_proofs':
                            ['non_scene_configuration_owner_preparation', 'exact_pre_handoff_result', 'final_handoff_result',
                             'compilation_adapter_result', 'owner_bound_activation_profile_selection']})
    return observations
