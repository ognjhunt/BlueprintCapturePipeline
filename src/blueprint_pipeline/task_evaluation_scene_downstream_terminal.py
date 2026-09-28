"""Pure historical terminal/pointer joins, including protected nonexecution."""
from __future__ import annotations

from .task_evaluation_scene_lineage_budget import _work, _work_hash, _work_items, _work_kwargs

import hashlib
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


def _relative(value, *, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
    c.require(isinstance(value, str) and 0 < len(value) <= c.retained.MAX_PATH_BYTES, 'pointer_member_invalid', **_work_kwargs(work_budget))
    c.path('/' + value, **_work_kwargs(work_budget))
    c.require(not value.startswith('/') and len(value.split('/')) <= c.retained.MAX_PATH_COMPONENTS, 'pointer_member_invalid', **_work_kwargs(work_budget))


def _members(context, value, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    members = value.get('members')
    c.require(isinstance(members, list) and type(value.get('member_count')) is int
              and value['member_count'] == len(members), 'pointer_members_invalid', **_work_kwargs(work_budget))
    seen = set()
    for member in (_work_items(members, work_budget) if work_budget is not None else members):
        context.consume()
        c.require(isinstance(member, dict) and set(member) == {'relative_path', 'size_bytes', 'sha256'}, 'pointer_member_invalid', **_work_kwargs(work_budget))
        _relative(member['relative_path'], **_work_kwargs(work_budget))
        c.require(type(member['size_bytes']) is int and member['size_bytes'] >= 0 and c.matches(member['sha256'], **_work_kwargs(work_budget))
                  and member['relative_path'] not in seen, 'pointer_member_invalid', **_work_kwargs(work_budget))
        context.size(member['sha256'], member['size_bytes'])
        seen.add(member['relative_path'])
    return {m['relative_path']: m for m in (_work_items(members, work_budget) if work_budget is not None else members)}


def _nonexecution(row, *, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
    value, _ = row
    field = 'blocked_result_digest' if value.get('schema_version') == 'task_evaluation_policy_canary_preprovider_blocked.v1' else 'receipt_digest'
    c.seal(row, field, **_work_kwargs(work_budget))
    c.require(value.get('run_kind') == 'internal_policy_canary' and c.matches(value.get('run_id'), c.LAUNCH_ID, **_work_kwargs(work_budget)), 'dispatch_invalid', **_work_kwargs(work_budget))
    if value.get('status') not in NONEXECUTION:
        known = value.get('schema_version') == 'task_evaluation_policy_canary_dispatch.v1' and value.get('status') in {'completed_unqualified', 'blocked', 'cancelled'}
        return field, 'executed' if known else 'unknown_status'
    c.require(value.get('claim_ceiling') == 'diagnostic_policy_execution', 'nonexecution_invalid', **_work_kwargs(work_budget))
    if value.get('terminal_result_kind') == 'definite_provider_create_refusal':
        c.require(all(value.get(k) is True for k in (_work_items(('provider_call_reached', 'provider_zero_required', 'paid_execution_requested'), work_budget) if work_budget is not None else ('provider_call_reached', 'provider_zero_required', 'paid_execution_requested')))
                  and value.get('provider_zero_not_applicable') is False
                  and value.get('provider_mutation_performed') is False and value.get('provider_allocation_performed') is False,
                  'provider_refusal_invalid', **_work_kwargs(work_budget))
        return field, 'provider_refusal'
    c.require(all(value.get(k) is False for k in (_work_items(('provider_mutation_performed', 'provider_allocation_performed',
              'provider_zero_required', 'paid_execution_requested', 'automatic_retry_authorized', 'automatic_retry_performed', 'provider_call_reached'), work_budget) if work_budget is not None else ('provider_mutation_performed', 'provider_allocation_performed',
              'provider_zero_required', 'paid_execution_requested', 'automatic_retry_authorized', 'automatic_retry_performed', 'provider_call_reached')))
              and value.get('provider_zero_not_applicable') is True and type(value.get('retry_cap')) is int and value['retry_cap'] == 0
              and not {'provider_zero', 'policy_canary_result_projection', 'terminal_result_publication'}.intersection(value), 'nonexecution_invalid', **_work_kwargs(work_budget))
    return field, 'nonexecution'


def _positive_ref(ref, *, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
    c.require(isinstance(ref, dict) and {'path', 'sha256', 'size_bytes'} <= set(ref)
              and c.matches(ref.get('sha256'), **_work_kwargs(work_budget)) and type(ref.get('size_bytes')) is int and ref['size_bytes'] > 0, 'dispatch_reference_invalid', **_work_kwargs(work_budget))
    c.path(ref['path'], **_work_kwargs(work_budget))


def _validate(context, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    paths, dispatch_types, pointers = {}, {}, {}
    roots = context.roots
    for role in (_work_items(('canary_dispatches', *SCHEMAS, 'allocator_results'), work_budget) if work_budget is not None else ('canary_dispatches', *SCHEMAS, 'allocator_results')):
        paths[role] = {}
        for row in (_work_items(context.decoded[role], work_budget) if work_budget is not None else context.decoded[role]):
            value, proof = row
            p = PurePosixPath(proof['path'])
            if role == 'canary_offload_pointers':
                c.require(p.name.endswith('.offloaded.v1.json') and c.under(proof['path'], roots['policy_canary_root'], **_work_kwargs(work_budget)), 'terminal_path_invalid', **_work_kwargs(work_budget))
            elif role == 'terminal_publications':
                c.require(p.name == 'terminal_result_publication.json' and c.under(proof['path'], roots['terminal_result_root'], **_work_kwargs(work_budget)), 'terminal_path_invalid', **_work_kwargs(work_budget))
            else:
                c.require(c.under(proof['path'], roots['policy_canary_root'], **_work_kwargs(work_budget)) or c.under(proof['path'], roots['terminal_result_root'], **_work_kwargs(work_budget)), 'terminal_path_invalid', **_work_kwargs(work_budget))
            paths[role].setdefault(proof['path'], []).append(row)
            schema = value.get('schema_version')
            if role == 'canary_dispatches':
                c.require(p.name in {'dispatch_receipt.json', 'preprovider_blocked.json', 'no_provider_allocation_blocked.json',
                          'policy_canary_nonexecution.json'}, 'terminal_path_invalid', **_work_kwargs(work_budget))
                if schema not in {'task_evaluation_policy_canary_dispatch.v1', 'task_evaluation_policy_canary_preprovider_blocked.v1'}:
                    continue
                field, kind = _nonexecution(row, **_work_kwargs(work_budget))
                if p.name == 'preprovider_blocked.json':
                    c.require(field == 'blocked_result_digest', 'dispatch_role_invalid', **_work_kwargs(work_budget))
                if p.name in {'dispatch_receipt.json', 'no_provider_allocation_blocked.json'}:
                    c.require(field == 'receipt_digest', 'dispatch_role_invalid', **_work_kwargs(work_budget))
                dispatch_types[proof['sha256']] = (field, kind)
                if kind == 'executed':
                    for name in (_work_items(('policy_canary_result_projection', 'policy_canary_webapp_sync', 'provider_zero'), work_budget) if work_budget is not None else ('policy_canary_result_projection', 'policy_canary_webapp_sync', 'provider_zero')):
                        _positive_ref(value.get(name), **_work_kwargs(work_budget))
                if value.get('allocator_result') is not None:
                    _positive_ref(value['allocator_result'], **_work_kwargs(work_budget))
                if kind == 'nonexecution' and value.get('allocator_result') is not None and c.under(proof['path'], roots['policy_canary_root'], **_work_kwargs(work_budget)):
                    allocator = value['allocator_result']
                    c.require(isinstance(allocator, dict) and c.under(allocator.get('path'), str(p.parent), **_work_kwargs(work_budget)), 'allocator_reference_invalid', **_work_kwargs(work_budget))
            elif schema == SCHEMAS.get(role):
                if role == 'canary_projections':
                    c.seal(row, 'projection_digest', cross=True, **_work_kwargs(work_budget))
                    c.require(c.matches(value.get('run_id'), c.LAUNCH_ID, **_work_kwargs(work_budget)) and isinstance(value.get('result_status'), str)
                              and all(c.matches(value.get(k), **_work_kwargs(work_budget)) for k in (_work_items(('request_digest', 'configuration_digest', 'result_delivery_digest'), work_budget) if work_budget is not None else ('request_digest', 'configuration_digest', 'result_delivery_digest'))), 'projection_invalid', **_work_kwargs(work_budget))
                elif role == 'provider_zero_receipts':
                    c.seal(row, 'receipt_digest', **_work_kwargs(work_budget))
                    c.require(value.get('status') == 'provider_zero_confirmed' and value.get('api_confirmed') is True
                              and value.get('provider_zero_verified') is True and type(value.get('live_instance_count')) is int
                              and value['live_instance_count'] == 0 and value.get('blockers') == [], 'provider_zero_invalid', **_work_kwargs(work_budget))
                elif role == 'canary_syncs':
                    c.require(c.matches(value.get('run_id'), c.LAUNCH_ID, **_work_kwargs(work_budget)) and isinstance(value.get('status'), str)
                              and all(c.matches(value.get(k), **_work_kwargs(work_budget)) for k in (_work_items(('request_digest', 'configuration_digest', 'policy_canary_projection_digest'), work_budget) if work_budget is not None else ('request_digest', 'configuration_digest', 'policy_canary_projection_digest')))
                              and isinstance(value.get('result_status'), str), 'sync_invalid', **_work_kwargs(work_budget))
                elif role == 'canary_offload_pointers':
                    c.seal(row, 'pointer_digest', **_work_kwargs(work_budget))
                    c.require(value.get('status') == 'offloaded' and c.matches(value.get('directory'), c.ACTIVATION_ID, **_work_kwargs(work_budget))
                              and p.name == value['directory'] + '.offloaded.v1.json'
                              and value.get('terminal_receipt') == 'dispatch_receipt.json'
                              and isinstance(value.get('uri'), str) and value['uri'].startswith('s3://') and '?' not in value['uri']
                              and c.matches(value.get('digest'), **_work_kwargs(work_budget)) and type(value.get('size_bytes')) is int and value['size_bytes'] > 0, 'pointer_invalid', **_work_kwargs(work_budget))
                    pointers[proof['sha256']] = _members(context, value, **_work_kwargs(work_budget))
                elif role == 'terminal_publications':
                    c.seal(row, 'publication_digest', **_work_kwargs(work_budget))
                    c.require(c.matches(value.get('run_id'), c.LAUNCH_ID, **_work_kwargs(work_budget)) and all(c.matches(value.get(k), **_work_kwargs(work_budget)) for k in (_work_items(('digest', 'archive_digest', 'pointer_digest'), work_budget) if work_budget is not None else ('digest', 'archive_digest', 'pointer_digest')))
                              and isinstance(value.get('uri'), str) and value['uri'].startswith('s3://')
                              and '?' not in value['uri'] and not any(char.isspace() for char in (_work_items(value['uri'], work_budget) if work_budget is not None else value['uri']))
                              and type(value.get('size_bytes')) is int and value['size_bytes'] > 0
                              and type(value.get('archive_member_count')) is int and value['archive_member_count'] >= 0
                              and value.get('provider_allocated') is False, 'publication_invalid', **_work_kwargs(work_budget))
    # Validate every available exact projection edge, even when its dispatch
    # version is not the one selected by a terminal state.
    edges = {role: {(p['path'], p['sha256'], p['size_bytes']): (v, p) for v, p in (_work_items(context.decoded[role], work_budget) if work_budget is not None else context.decoded[role])}
             for role in (_work_items(('canary_projections', 'canary_syncs'), work_budget) if work_budget is not None else ('canary_projections', 'canary_syncs'))}
    for value, proof in (_work_items(context.decoded['canary_dispatches'], work_budget) if work_budget is not None else context.decoded['canary_dispatches']):
        if dispatch_types.get(proof['sha256'], (None, None))[1] != 'executed':
            continue
        projection = None
        matches = _available_edge(context, edges, 'canary_projections', value['policy_canary_result_projection'], row=(value, proof), **_work_kwargs(work_budget))
        if matches:
            v = matches[0][0]
            if v.get('schema_version') == SCHEMAS['canary_projections']:
                projection = v
                c.require(v.get('run_id') == value['run_id'] and v.get('projection_digest') == value.get('policy_canary_projection_digest')
                          and v.get('result_delivery_digest') == value.get('result_delivery_digest') and v.get('result_status') == value.get('status'), 'projection_binding_invalid', **_work_kwargs(work_budget))
        matches = _available_edge(context, edges, 'canary_syncs', value['policy_canary_webapp_sync'], row=(value, proof), **_work_kwargs(work_budget))
        sync = matches[0][0] if matches else None
        if sync is not None and sync.get('schema_version') == SCHEMAS['canary_syncs']:
            c.require(sync['status'] == 'succeeded' and sync['run_id'] == value['run_id'] and sync['result_status'] == value['status']
                      and sync['policy_canary_projection_digest'] == value.get('policy_canary_projection_digest')
                      and sync.get('notification_delivery') == value.get('notification_delivery'), 'sync_binding_invalid', **_work_kwargs(work_budget))
            if projection:
                c.require(all(sync[k] == projection[k] for k in (_work_items(('request_digest', 'configuration_digest'), work_budget) if work_budget is not None else ('request_digest', 'configuration_digest'))), 'sync_binding_invalid', **_work_kwargs(work_budget))
    paths['_raw_index'] = {role: {(p['path'], p['sha256'], p['size_bytes']): (v, p)
                                 for v, p in (_work_items(context.decoded[role], work_budget) if work_budget is not None else context.decoded[role])} for role in (_work_items(('canary_projections', 'canary_syncs', 'provider_zero_receipts', 'allocator_results', 'canary_dispatches'), work_budget) if work_budget is not None else ('canary_projections', 'canary_syncs', 'provider_zero_receipts', 'allocator_results', 'canary_dispatches'))}
    paths['_dispatch_index'] = {}
    for row in (_work_items(context.decoded['canary_dispatches'], work_budget) if work_budget is not None else context.decoded['canary_dispatches']):
        value, proof = row
        field = dispatch_types.get(proof['sha256'], (None, None))[0]
        if field:
            paths['_dispatch_index'].setdefault((proof['path'], value[field]), []).append(row)
    paths['_path_runs'] = {role: {} for role in (_work_items(('canary_projections', 'canary_syncs'), work_budget) if work_budget is not None else ('canary_projections', 'canary_syncs'))}
    for role in (_work_items(paths['_path_runs'], work_budget) if work_budget is not None else paths['_path_runs']):
        for value, proof in (_work_items(context.decoded[role], work_budget) if work_budget is not None else context.decoded[role]):
            if value.get('schema_version') == SCHEMAS[role]:
                paths['_path_runs'][role].setdefault(proof['path'], set()).add(value['run_id'])
    paths['_archives'] = _archive_index(context, paths, pointers, dispatch_types, **_work_kwargs(work_budget))
    return paths, dispatch_types, pointers


def _available_edge(context, indexes, role, ref, *, row, work_budget=None):
    """Exact bytes at the original path or either finite terminal-copy layout."""
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    value, proof = row
    directory = c.child(context.roots['terminal_result_root'], context.intent_id, **_work_kwargs(work_budget))
    copied = 'policy_canary_result_projection.json' if role == 'canary_projections' else 'policy_canary_webapp_sync.json'
    names = {ref['path'], c.child(directory, copied, **_work_kwargs(work_budget)),
             c.child(directory, 'runs', (_work_hash(work_budget, hashlib.sha256, value['run_id'].encode()) if work_budget is not None else hashlib.sha256(value['run_id'].encode())).hexdigest(), copied, **_work_kwargs(work_budget))}
    matches = [indexes[role][key] for path in (_work_items(sorted(names), work_budget) if work_budget is not None else sorted(names))
               if (key := (path, ref['sha256'], ref['size_bytes'])) in indexes[role]]
    copies = [r[1] for r in (_work_items(matches, work_budget) if work_budget is not None else matches) if r[1]['path'] != ref['path']]
    if copies:
        context.missing(role, 'indexed_bytes_match_original_presence_unverified', [proof, *copies],
                        ref['path'], {'sha256': ref['sha256'], 'size_bytes': ref['size_bytes']})
    return matches


def _raw_select(paths, role, ref, expected, copied, *, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
    c.require(isinstance(ref, dict) and {'path', 'sha256', 'size_bytes'} <= set(ref)
              and ref['path'] == expected, 'dispatch_reference_path_invalid', **_work_kwargs(work_budget))
    # A byte-identical indexed copy proves retained bytes, never original presence.
    index = paths['_raw_index'][role]
    return [index[key] for key in (_work_items(sorted({(expected, ref['sha256'], ref['size_bytes']), (copied, ref['sha256'], ref['size_bytes'])}), work_budget) if work_budget is not None else sorted({(expected, ref['sha256'], ref['size_bytes']), (copied, ref['sha256'], ref['size_bytes'])})) if key in index]


def _archive_index(context, paths, pointer_members, dispatch_types, *, work_budget=None):
    """Validate each historical pair once; never compare it to an unrelated state."""
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    seals, paired, result, pointer_paths = {}, set(), {}, {}
    for row in (_work_items(context.decoded['canary_offload_pointers'], work_budget) if work_budget is not None else context.decoded['canary_offload_pointers']):
        if row[0].get('schema_version') == SCHEMAS['canary_offload_pointers']:
            seals.setdefault(row[0]['pointer_digest'], []).append(row)
            pointer_paths.setdefault(row[1]['path'], []).append(row)
    state_selectors = set()
    for state, proof in (_work_items(context.decoded['terminal_states'], work_budget) if work_budget is not None else context.decoded['terminal_states']):
        if state.get('schema_version') == 'task_evaluation_scene_terminal_index_state.v1' and all(
                isinstance(state.get(k), str) for k in (_work_items(('canary_run_root', 'run_id', 'projection_digest', 'dispatch_receipt_digest'), work_budget) if work_budget is not None else ('canary_run_root', 'run_id', 'projection_digest', 'dispatch_receipt_digest'))):
            state_selectors.add((state['canary_run_root'], str(PurePosixPath(proof['path']).parent), state['run_id'],
                                 state['projection_digest'], state['dispatch_receipt_digest']))

    def add(pointer_row, publication):
        if work_budget is not None:
            _work(work_budget)
        pointer, proof = pointer_row
        canary = proof['path'][:-len('.offloaded.v1.json')]
        directory = str(PurePosixPath(publication[1]['path']).parent) if publication else None
        sources = [proof] + ([publication[1]] if publication else [])
        if publication:
            value = publication[0]
            c.require(value['archive_digest'] == pointer['digest'] and value.get('uri') == pointer['uri']
                      and value['size_bytes'] == pointer['size_bytes']
                      and value['archive_member_count'] == pointer['member_count'], 'publication_binding_invalid', **_work_kwargs(work_budget))
        selected = []
        for role, name, copied in (_work_items((('canary_projections', PROJECTION_PATH, 'policy_canary_result_projection.json'),
                                   ('canary_dispatches', 'dispatch_receipt.json', 'dispatch_receipt.json')), work_budget) if work_budget is not None else (('canary_projections', PROJECTION_PATH, 'policy_canary_result_projection.json'),
                                   ('canary_dispatches', 'dispatch_receipt.json', 'dispatch_receipt.json'))):
            member = pointer_members[proof['sha256']].get(name)
            c.require(member is not None, 'pointer_member_binding_invalid', **_work_kwargs(work_budget))
            names = [c.child(canary, name, **_work_kwargs(work_budget))] + ([c.child(directory, copied, **_work_kwargs(work_budget))] if directory else [])
            rows = [paths['_raw_index'][role][key] for path in (_work_items(names, work_budget) if work_budget is not None else names)
                    if (key := (path, member['sha256'], member['size_bytes'])) in paths['_raw_index'][role]]
            selected.append(rows[0] if rows else None)
        if not all(selected):
            context.missing('archive_member_bytes', 'archive_member_bytes_unavailable', sources,
                            selector={'pointer_digest': pointer['pointer_digest']})
            return
        projection, dispatch = selected
        if (projection[0].get('schema_version') != SCHEMAS['canary_projections']
                or dispatch_types.get(dispatch[1]['sha256'], (None, None))[1] != 'executed'):
            context.missing('archive_member_identity', 'archive_member_identity_unproven', sources)
            return
        v, d = projection[0], dispatch[0]
        c.require(d['run_id'] == v['run_id'] and d['policy_canary_projection_digest'] == v['projection_digest']
                  and d['result_delivery_digest'] == v['result_delivery_digest'] and d['status'] == v['result_status']
                  and all(d['policy_canary_result_projection'][k] == projection[1][k] for k in (_work_items(('sha256', 'size_bytes'), work_budget) if work_budget is not None else ('sha256', 'size_bytes'))),
                  'pointer_member_binding_invalid', **_work_kwargs(work_budget))
        if publication:
            c.require(publication[0]['run_id'] == v['run_id'] and publication[0]['digest'] == v['projection_digest'], 'publication_binding_invalid', **_work_kwargs(work_budget))
            selector = (canary, directory, v['run_id'], v['projection_digest'], d['receipt_digest'])
            if selector not in state_selectors:
                context.missing('archive_terminal_state', 'archive_terminal_state_selector_unavailable',
                                sources + [projection[1], dispatch[1]], selector={'projection_digest': v['projection_digest'],
                                'dispatch_receipt_digest': d['receipt_digest'], 'run_id': v['run_id']})
        key = (canary, directory, projection[1]['sha256'], dispatch[1]['sha256'])
        result.setdefault(key, []).append(sources)

    for publication in (_work_items(context.decoded['terminal_publications'], work_budget) if work_budget is not None else context.decoded['terminal_publications']):
        if publication[0].get('schema_version') != SCHEMAS['terminal_publications']:
            continue
        selected = seals.get(publication[0]['pointer_digest'], [])
        paired.update(row[1]['sha256'] for row in (_work_items(selected, work_budget) if work_budget is not None else selected))
        if len(selected) != 1:
            context.missing('publication_pointer_version', 'publication_pointer_version_unavailable' if not selected else 'publication_pointer_version_ambiguous',
                            [publication[1]], selector={'pointer_digest': publication[0]['pointer_digest']})
        else:
            add(selected[0], publication)
    for rows in (_work_items(seals.values(), work_budget) if work_budget is not None else seals.values()):
        for row in (_work_items(rows, work_budget) if work_budget is not None else rows):
            if row[1]['sha256'] not in paired:
                if len(pointer_paths[row[1]['path']]) == 1:
                    add(row, None)
                else:
                    context.missing('terminal_publication', 'terminal_publication_selector_unavailable', [row[1]],
                                    selector={'pointer_digest': row[0]['pointer_digest']})
    return result


def _archive(context, paths, state, dispatch, projection, directory, sources, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    canary = state['canary_run_root']
    key = (canary, directory, projection[1]['sha256'], dispatch[1]['sha256'])
    paired = paths['_archives'].get(key, [])
    fallback = paths['_archives'].get((canary, None, *key[2:]), []) if not paired else []
    if paired or fallback:
        for proofs in (_work_items(paired or fallback, work_budget) if work_budget is not None else paired or fallback):
            sources += proofs
        if fallback:
            context.missing('terminal_publication', 'terminal_publication_unavailable', sources,
                            c.child(directory, 'terminal_result_publication.json', **_work_kwargs(work_budget)))
        return True
    context.missing('canary_offload_pointer', 'canary_offload_pointer_unavailable_or_ambiguous'
                    if paths['canary_offload_pointers'].get(canary + '.offloaded.v1.json') else 'canary_offload_pointer_unavailable',
                    sources, canary + '.offloaded.v1.json')
    return False


def terminal(context, launches, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    paths, types, members = _validate(context, **_work_kwargs(work_budget))
    observations, indexed = context.rows(), set()
    for row in (_work_items(context.decoded['terminal_states'], work_budget) if work_budget is not None else context.decoded['terminal_states']):
        state, proof = row
        p = PurePosixPath(proof['path'])
        c.require(c.under(proof['path'], context.roots['terminal_result_root'], **_work_kwargs(work_budget)), 'terminal_path_invalid', **_work_kwargs(work_budget))
        schema = state.get('schema_version')
        if schema not in {'task_evaluation_scene_terminal_index_state.v1', 'task_evaluation_scene_nonexecution_terminal_state.v1'}:
            observations.append(c.observation(row, reason='terminal_state_schema_unproven', **_work_kwargs(work_budget)))
            continue
        c.seal(row, 'state_digest', **_work_kwargs(work_budget))
        nonexecution = schema == 'task_evaluation_scene_nonexecution_terminal_state.v1'
        c.require(p.name == ('nonexecution_terminal_state.json' if nonexecution else 'terminal_index_state.json')
                  and c.matches(state.get('run_id'), c.LAUNCH_ID, **_work_kwargs(work_budget)) and c.under(state.get('canary_run_root'), context.roots['policy_canary_root'], **_work_kwargs(work_budget)), 'terminal_state_invalid', **_work_kwargs(work_budget))
        owner_directory = c.child(context.roots['terminal_result_root'], context.intent_id, **_work_kwargs(work_budget))
        c.require(str(p.parent) in {owner_directory, c.child(owner_directory, 'runs', (_work_hash(work_budget, hashlib.sha256, state['run_id'].encode()) if work_budget is not None else hashlib.sha256(state['run_id'].encode())).hexdigest(), **_work_kwargs(work_budget))}, 'terminal_path_invalid', **_work_kwargs(work_budget))
        field = 'record_digest' if nonexecution else 'dispatch_receipt_digest'
        c.require(c.matches(state.get(field), **_work_kwargs(work_budget)), 'terminal_state_invalid', **_work_kwargs(work_budget))
        if not nonexecution:
            c.require(c.matches(state.get('projection_digest'), **_work_kwargs(work_budget)), 'terminal_state_invalid', **_work_kwargs(work_budget))
        directory, canary = str(p.parent), state['canary_run_root']
        for role, original, copied in (_work_items((('canary_projections', PROJECTION_PATH, 'policy_canary_result_projection.json'),
                                      ('canary_syncs', SYNC_PATH, 'policy_canary_webapp_sync.json')), work_budget) if work_budget is not None else (('canary_projections', PROJECTION_PATH, 'policy_canary_result_projection.json'),
                                      ('canary_syncs', SYNC_PATH, 'policy_canary_webapp_sync.json'))):
            for name in (_work_items((c.child(canary, original, **_work_kwargs(work_budget)), c.child(directory, copied, **_work_kwargs(work_budget))), work_budget) if work_budget is not None else (c.child(canary, original), c.child(directory, copied))):
                runs = paths['_path_runs'][role].get(name, set())
                c.require(not runs or runs == {state['run_id']}, 'terminal_identity_invalid', **_work_kwargs(work_budget))
        original_names = ('preprovider_blocked.json', 'no_provider_allocation_blocked.json', 'dispatch_receipt.json') if nonexecution else ('dispatch_receipt.json',)
        candidate_paths = [c.child(canary, name, **_work_kwargs(work_budget)) for name in (_work_items(original_names, work_budget) if work_budget is not None else original_names)] + [c.child(directory, 'policy_canary_nonexecution.json' if nonexecution else 'dispatch_receipt.json', **_work_kwargs(work_budget))]
        candidates = [r for path in (_work_items(candidate_paths, work_budget) if work_budget is not None else candidate_paths) for r in (_work_items(paths['_dispatch_index'].get((path, state[field]), []), work_budget) if work_budget is not None else paths['_dispatch_index'].get((path, state[field]), []))]
        # Same raw bytes at original+indexed names retain both provenances.
        variants = {r[1]['sha256'] for r in (_work_items(candidates, work_budget) if work_budget is not None else candidates)}
        sources, reason, archive = context.provenance(p for group in (_work_items(((proof,), (r[1] for r in (_work_items(candidates, work_budget) if work_budget is not None else candidates))), work_budget) if work_budget is not None else ((proof,), (r[1] for r in candidates))) for p in (_work_items(group, work_budget) if work_budget is not None else group)), None, False
        if len(variants) != 1:
            reason = 'terminal_dispatch_unavailable_or_ambiguous'
        dispatch = candidates[0] if len(variants) == 1 else None
        if dispatch:
            indexed.update(r[1]['sha256'] for r in (_work_items(candidates, work_budget) if work_budget is not None else candidates))
            value = dispatch[0]
            kind = types[dispatch[1]['sha256']][1]
            c.require(value['run_id'] == state['run_id'], 'terminal_identity_invalid', **_work_kwargs(work_budget))
            if nonexecution:
                c.require(value.get('status') == state.get('status'), 'nonexecution_identity_invalid', **_work_kwargs(work_budget))
                if kind != 'nonexecution':
                    reason = 'kept_unresolved_provider_refusal' if kind == 'provider_refusal' else 'nonexecution_scope_unproven'
                allocator = value.get('allocator_result')
                if allocator is not None:
                    c.require(isinstance(allocator, dict) and c.under(allocator.get('path'), canary, **_work_kwargs(work_budget)), 'allocator_reference_invalid', **_work_kwargs(work_budget))
                    matches = _raw_select(paths, 'allocator_results', allocator, allocator['path'], allocator['path'], **_work_kwargs(work_budget))
                    if not matches:
                        reason = reason or 'allocator_result_bytes_unavailable'
            elif kind != 'executed':
                reason = 'terminal_execution_status_unproven'
            else:
                projection_ref, sync_ref, zero_ref = (value.get(k) for k in (_work_items(('policy_canary_result_projection', 'policy_canary_webapp_sync', 'provider_zero'), work_budget) if work_budget is not None else ('policy_canary_result_projection', 'policy_canary_webapp_sync', 'provider_zero')))
                selected = [_raw_select(paths, role, ref, c.child(canary, original, **_work_kwargs(work_budget)), c.child(directory, copied, **_work_kwargs(work_budget)), **_work_kwargs(work_budget))
                            for role, ref, original, copied in (_work_items((
                            ('canary_projections', projection_ref, PROJECTION_PATH, 'policy_canary_result_projection.json'),
                            ('canary_syncs', sync_ref, SYNC_PATH, 'policy_canary_webapp_sync.json'),
                            ('provider_zero_receipts', zero_ref, 'post_teardown_global_provider_zero.json', 'provider_zero_closure.json')), work_budget) if work_budget is not None else (
                            ('canary_projections', projection_ref, PROJECTION_PATH, 'policy_canary_result_projection.json'),
                            ('canary_syncs', sync_ref, SYNC_PATH, 'policy_canary_webapp_sync.json'),
                            ('provider_zero_receipts', zero_ref, 'post_teardown_global_provider_zero.json', 'provider_zero_closure.json')))]
                known_selected = [matches if matches and matches[0][0].get('schema_version') == SCHEMAS[role] else []
                                  for role, matches in (_work_items(zip(('canary_projections', 'canary_syncs', 'provider_zero_receipts'), selected), work_budget) if work_budget is not None else zip(('canary_projections', 'canary_syncs', 'provider_zero_receipts'), selected))]
                if any(matches and not known for matches, known in (_work_items(zip(selected, known_selected), work_budget) if work_budget is not None else zip(selected, known_selected))):
                    reason = 'terminal_reference_schema_unproven'
                if known_selected[1]:
                    sync = known_selected[1][0][0]
                    c.require(sync['run_id'] == value['run_id'] and sync['result_status'] == value.get('status')
                              and sync['policy_canary_projection_digest'] == value.get('policy_canary_projection_digest')
                              and sync.get('notification_delivery') == value.get('notification_delivery'), 'sync_binding_invalid', **_work_kwargs(work_budget))
                if known_selected[0]:
                    projection = selected[0][0]
                    v = projection[0]
                    c.require(v.get('schema_version') == SCHEMAS['canary_projections'] and v['run_id'] == state['run_id']
                              and v['projection_digest'] == state['projection_digest'] == value.get('policy_canary_projection_digest')
                              and v['result_delivery_digest'] == value.get('result_delivery_digest') and v['result_status'] == value.get('status'), 'projection_binding_invalid', **_work_kwargs(work_budget))
                    if known_selected[1]:
                        sync = selected[1][0][0]
                        c.require(sync.get('schema_version') == SCHEMAS['canary_syncs'] and sync.get('status') == 'succeeded'
                                  and all(sync.get(k) == v[k] for k in (_work_items(('run_id', 'request_digest', 'configuration_digest', 'result_status'), work_budget) if work_budget is not None else ('run_id', 'request_digest', 'configuration_digest', 'result_status')))
                                  and sync.get('policy_canary_projection_digest') == v['projection_digest']
                                  and sync.get('notification_delivery') == value.get('notification_delivery'), 'sync_binding_invalid', **_work_kwargs(work_budget))
                    archive = _archive(context, paths, state, dispatch, projection, directory, sources, **_work_kwargs(work_budget))
                if known_selected[2]:
                    c.require(selected[2][0][0].get('schema_version') == SCHEMAS['provider_zero_receipts']
                              and zero_ref.get('provider_zero_verified') is True, 'provider_zero_binding_invalid', **_work_kwargs(work_budget))
                if not all(selected):
                    reason = reason or 'terminal_reference_bytes_unavailable'
                for matches in (_work_items(selected, work_budget) if work_budget is not None else selected):
                    sources += context.provenance(r[1] for r in (_work_items(matches, work_budget) if work_budget is not None else matches))
        bridges = [r for r in (_work_items(launches.get(state['run_id'], []), work_budget) if work_budget is not None else launches.get(state['run_id'], [])) if str(PurePosixPath(r[0][1]['path']).parent) == directory]
        if not bridges:
            reason = reason or 'terminal_owner_bridge_unavailable'
        else:
            sources += bridges[0][1]
        if reason:
            context.missing('terminal_join', reason, sources, selector={'run_id': state['run_id'], 'canary_run_root': canary})
        else:
            context.member(canary, 'canary_evidence_workspace', {'intent_id': context.intent_id, 'run_id': state['run_id']}, sources)
        observations.append(c.observation(row, status='matched_retained_bytes' if reason is None else 'kept_unresolved', reason=reason,
                                          run_id=state['run_id'], canary_run_root=canary, archive_binding_verified=archive,
                                          source_provenance=c.unique(sources, context.limits['MAX_OUTPUT_BYTES'], **_work_kwargs(work_budget)), **_work_kwargs(work_budget)))
    for row in (_work_items(context.decoded['canary_dispatches'], work_budget) if work_budget is not None else context.decoded['canary_dispatches']):
        if row[1]['sha256'] not in indexed:
            kind = types.get(row[1]['sha256'], (None, 'unknown'))[1]
            observations.append(c.observation(row, reason='kept_unresolved_provider_refusal' if kind == 'provider_refusal' else 'terminal_owner_state_unavailable', **_work_kwargs(work_budget)))
            context.missing('terminal_state', 'terminal_owner_state_unavailable', [row[1]], selector={'run_id': row[0].get('run_id')})
    return observations


def compilations(context, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    envelopes, observations = {}, context.rows()
    for row in (_work_items(context.decoded['compilation_envelopes'], work_budget) if work_budget is not None else context.decoded['compilation_envelopes']):
        value, proof = row
        c.require(c.under(proof['path'], context.roots['compilation_queue_root'], **_work_kwargs(work_budget)), 'compilation_path_invalid', **_work_kwargs(work_budget))
        if value.get('schema_version') != 'task_evaluation_episode_compilation_envelope.v1':
            observations.append(c.observation(row, reason='compilation_schema_unproven', **_work_kwargs(work_budget)))
            continue
        c.seal(row, 'envelope_digest', **_work_kwargs(work_budget))
        request = value.get('request')
        c.require(c.matches(value.get('compilation_id'), c.LAUNCH_ID, **_work_kwargs(work_budget)) and value.get('preparation_id') == value['compilation_id']
                  and c.matches(value.get('expected_production_commit'), c.COMMIT, **_work_kwargs(work_budget)) and c.matches(value.get('preparation_result_digest'), **_work_kwargs(work_budget))
                  and c.matches(value.get('configured_scene_revision_digest'), **_work_kwargs(work_budget)) and isinstance(request, dict)
                  and request.get('preparation_id') == value['preparation_id']
                  and request.get('run_mode') in {'episode_evaluation', 'destination_qualification'}
                  and isinstance(request.get('construction'), dict) and request['construction'].get('mode') == 'reuse_configured_scene'
                  and isinstance(request.get('task'), dict) and request['task'].get('binding_mode') == 'reuse_configured_template'
                  and request['task'].get('configured_scene_revision_digest') == value['configured_scene_revision_digest'], 'compilation_envelope_invalid', **_work_kwargs(work_budget))
        filename = value['compilation_id'] + '-' + value['envelope_digest'][7:] + '.json'
        p = PurePosixPath(proof['path'])
        c.require(p.name == filename and p.parent.name in {'pending', 'processing', 'completed', 'blocked'}
                  and str(p.parent.parent) == context.roots['compilation_queue_root'], 'compilation_path_invalid', **_work_kwargs(work_budget))
        envelopes.setdefault(filename, []).append(row)
    for row in (_work_items(context.decoded['compilation_results'], work_budget) if work_budget is not None else context.decoded['compilation_results']):
        value, proof = row
        p = PurePosixPath(proof['path'])
        c.require(str(p.parent) == c.child(context.roots['compilation_queue_root'], 'results', **_work_kwargs(work_budget)), 'compilation_path_invalid', **_work_kwargs(work_budget))
        if value.get('schema_version') != 'task_evaluation_episode_compilation_result.v1':
            observations.append(c.observation(row, reason='compilation_schema_unproven', **_work_kwargs(work_budget)))
            continue
        c.seal(row, 'result_digest', **_work_kwargs(work_budget))
        c.require(c.matches(value.get('compilation_id'), c.LAUNCH_ID, **_work_kwargs(work_budget)) and isinstance(value.get('status'), str), 'compilation_result_invalid', **_work_kwargs(work_budget))
        if value['status'] == 'compiled_for_production_launch':
            c.require(c.matches(value.get('source_commit'), c.COMMIT, **_work_kwargs(work_budget))
                      and all(c.matches(value.get(k), **_work_kwargs(work_budget)) for k in (_work_items(('configured_scene_revision_digest', 'compiled_episode_packet_digest',
                              'adapter_result_digest', 'compiler_output_digest'), work_budget) if work_budget is not None else ('configured_scene_revision_digest', 'compiled_episode_packet_digest',
                              'adapter_result_digest', 'compiler_output_digest')))
                      and type(value.get('compiled_episode_packet_size_bytes')) is int and value['compiled_episode_packet_size_bytes'] > 0
                      and all(c.under(value.get(k), c.child(context.roots['compilation_output_root'], value['compilation_id'], **_work_kwargs(work_budget)), **_work_kwargs(work_budget))
                              for k in (_work_items(('compiled_episode_packet_path', 'adapter_result_path'), work_budget) if work_budget is not None else ('compiled_episode_packet_path', 'adapter_result_path')))
                      and value.get('provider_mutation_performed') is False and value.get('paid_execution_requested') is False
                      and value.get('compiled_by_production') is True and value.get('customer_supplied_prebuilt_episode_packet') is False,
                      'compilation_result_invalid', **_work_kwargs(work_budget))
        for envelope, _ in (_work_items(envelopes.get(p.name, []), work_budget) if work_budget is not None else envelopes.get(p.name, [])):
            c.require(all(value.get(a) == envelope[b] for a, b in (_work_items((('compilation_id', 'compilation_id'), ('run_id', 'run_id'),
                      ('team_namespace', 'team_namespace')), work_budget) if work_budget is not None else (('compilation_id', 'compilation_id'), ('run_id', 'run_id'),
                      ('team_namespace', 'team_namespace')))) and (value['status'] != 'compiled_for_production_launch'
                      or value['source_commit'] == envelope['expected_production_commit']
                      and value['configured_scene_revision_digest'] == envelope['configured_scene_revision_digest']), 'compilation_identity_invalid', **_work_kwargs(work_budget))
        observations.append(c.observation(row, reason='compilation_owner_join_unproven', compilation_id=value['compilation_id'], **_work_kwargs(work_budget)))
    for role in (_work_items(('compilation_envelopes', 'compilation_results'), work_budget) if work_budget is not None else ('compilation_envelopes', 'compilation_results')):
        for row in (_work_items(context.decoded[role], work_budget) if work_budget is not None else context.decoded[role]):
            context.missing('compilation_owner_bridge', 'compilation_owner_join_unproven', [row[1]],
                            selector={'compilation_id': row[0].get('compilation_id'), 'required_proofs':
                            ['non_scene_configuration_owner_preparation', 'exact_pre_handoff_result', 'final_handoff_result',
                             'compilation_adapter_result', 'owner_bound_activation_profile_selection']})
    return observations
