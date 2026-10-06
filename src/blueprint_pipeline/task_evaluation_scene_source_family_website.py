"""Retained website capture/publication joins, never a capture or remote reader."""
from __future__ import annotations

from .task_evaluation_scene_lineage_budget import _work_collect, _work, _work_call, _work_items, _work_kwargs

from . import task_evaluation_scene_source_family_contracts as c

_CAPTURE_SCHEMAS = {
    'website_handoffs': ('website_scene_handoff.v1', 'digest'),
    'website_preparations': ('website_scene_preparation.v1', 'digest'),
    'website_runtime_inputs': ('website_scene_runtime_inputs.v1', 'digest'),
    'website_task_contexts': ('website_site_task_context.v1', 'context_digest'),
}


def _layout(context, row, *, identity=True, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    path, root = row[1]['path'], context.roots['pubsub_root']
    c.require(c.under(path, root, **_work_kwargs(work_budget)), 'capture_path_invalid', **_work_kwargs(work_budget))
    parts = path[len(root) + 1:].split('/')
    c.require(len(parts) >= 8 and parts[1] == 'scenes' and parts[3] == 'captures'
        and parts[5:7] == ['pipeline', 'website_scene_preparation'], 'capture_path_invalid', **_work_kwargs(work_budget))
    capture = c.child(root, *parts[:5], **_work_kwargs(work_budget))
    base = c.child(capture, 'pipeline', 'website_scene_preparation', **_work_kwargs(work_budget))
    role = row[1]['role']
    allowed = {'website_handoffs': ['handoff.json'], 'website_preparations': ['preparation.json', 'development_test/preparation.json'],
        'website_task_contexts': ['task_context.json', 'development_test/task_context.json'],
        'website_runtime_inputs': ['native/runtime_inputs.json', 'development_test/runtime_inputs.json']}
    c.require('/'.join(parts[7:]) in allowed[role], 'capture_path_invalid', **_work_kwargs(work_budget))
    if identity and role in {'website_handoffs', 'website_task_contexts'}:
        c.require(row[0].get('scene_id') == parts[2] and row[0].get('capture_id') == parts[4], 'capture_identity_invalid', **_work_kwargs(work_budget))
    return capture, base


def capture(context, old, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    rows, layouts, by_capture = {}, {}, {}
    for role, (schema, seal) in (_work_items(_CAPTURE_SCHEMAS.items(), work_budget) if work_budget is not None else _CAPTURE_SCHEMAS.items()):
        rows[role] = context.known(role, schema, seal)
        for row in (_work_items(rows[role], work_budget) if work_budget is not None else rows[role]):
            layout = _layout(context, row, **_work_kwargs(work_budget))
            layouts[id(row)] = layout
            by_capture.setdefault((layout[0], role), []).append(row)
    for value, proof in (_work_items(rows['website_handoffs'], work_budget) if work_budget is not None else rows['website_handoffs']):
        if value.get('source_registration') is not None:
            context.selected(value['source_registration'], proof, {'website_registrations'})
    _available_capture_edges(context, rows, **_work_kwargs(work_budget))
    intent = context.decoded['intent'][0][0]
    registrations = context.known('website_registrations', 'website_scene_source_registration.v1', 'registration_digest')
    bindings = context.known('website_bindings', 'website_scene_source_binding.v1', 'binding_digest')
    registration_index, binding_index = {}, {}
    for row in (_work_items(registrations, work_budget) if work_budget is not None else registrations):
        value, proof = row
        c.require(value.get('provider_mutation_performed') is False and value.get('execution_authority_granted') is False
            and value.get('claim_ceiling') == 'development_only' and c.matches(value.get('request_digest'), **_work_kwargs(work_budget)),
            'registration_invalid', **_work_kwargs(work_budget))
        c.require(proof['path'] == c.child(context.roots['website_source_binding_root'], value['request_digest'][7:] + '.json', **_work_kwargs(work_budget)),
                  'registration_path_invalid', **_work_kwargs(work_budget))
        refs = value.get('references')
        c.require(isinstance(refs, dict) and (_work_collect(work_budget, set, refs) if work_budget is not None else set(refs)) == {'preparation', 'runtime_inputs', 'task_context'}, 'registration_invalid', **_work_kwargs(work_budget))
        selected = {role: context.selected(ref, proof, {'website_' + {'preparation': 'preparations',
            'runtime_inputs': 'runtime_inputs', 'task_context': 'task_contexts'}[role]}) for role, ref in (_work_items(refs.items(), work_budget) if work_budget is not None else refs.items())}
        selected_layouts = {_layout(context, ({}, {'path': reference['path'],
            'role': {'preparation': 'website_preparations', 'runtime_inputs': 'website_runtime_inputs',
                     'task_context': 'website_task_contexts'}[role]}), identity=False, **_work_kwargs(work_budget))[0]
            for role, reference in (_work_items(refs.items(), work_budget) if work_budget is not None else refs.items())}
        # Context lexical path identity can be checked without pretending absent
        # bytes exist; actual supplied context owns its typed identity check.
        context_path = refs['task_context']['path']
        capture_root = next(iter(selected_layouts))
        c.require(len(selected_layouts) == 1 and c.under(context_path, c.child(capture_root, 'pipeline', 'website_scene_preparation', **_work_kwargs(work_budget)), **_work_kwargs(work_budget)),
                  'capture_reference_invalid', **_work_kwargs(work_budget))
        if selected['preparation']:
            preparation = selected['preparation'][0]
            if preparation.get('schema_version') == 'website_scene_preparation.v1':
                c.require(preparation.get('intake_request') is not None
                    and (_work_call(work_budget, c.cross_runtime_canonical_digest, preparation['intake_request']) if work_budget is not None else c.cross_runtime_canonical_digest(preparation['intake_request'])) == value['request_digest'],
                    'registration_request_invalid', **_work_kwargs(work_budget))
            else:
                context.missing('website_preparation', 'unsupported_retained_schema', [selected['preparation'][1]])
                selected['preparation'] = None
        for selected_role, schema in (_work_items((('runtime_inputs', 'website_scene_runtime_inputs.v1'),
                                      ('task_context', 'website_site_task_context.v1')), work_budget) if work_budget is not None else (('runtime_inputs', 'website_scene_runtime_inputs.v1'),
                                      ('task_context', 'website_site_task_context.v1'))):
            if selected[selected_role] and selected[selected_role][0].get('schema_version') != schema:
                context.missing('website_' + selected_role, 'unsupported_retained_schema', [selected[selected_role][1]])
                selected[selected_role] = None
        registration_index[(proof['path'], proof['sha256'], proof['size_bytes'])] = row, selected, capture_root
    for row in (_work_items(bindings, work_budget) if work_budget is not None else bindings):
        value, proof = row
        c.require(c.matches(value.get('binding_id'), c.ID, **_work_kwargs(work_budget))
            and proof['path'] == c.child(context.roots['factory_output_root'], context.intent_id,
                                        'website-source', value['binding_digest'][7:] + '.json', **_work_kwargs(work_budget)), 'binding_path_invalid', **_work_kwargs(work_budget))
        selected = context.selected(value.get('registration'), proof, {'website_registrations'})
        c.require(isinstance(value.get('references'), dict) and (_work_collect(work_budget, set, value['references']) if work_budget is not None else set(value['references'])) == {'preparation', 'runtime_inputs', 'task_context'},
                  'binding_invalid', **_work_kwargs(work_budget))
        if selected:
            c.require(value['references'] == selected[0]['references'], 'binding_reference_invalid', **_work_kwargs(work_budget))
        binding_index.setdefault(value['binding_digest'], []).append(row)
    # Check supported available runtime/preparation edges even when their exact
    # registration byte version or the unrelated handoff is missing.
    for row in (_work_items(rows['website_runtime_inputs'], work_budget) if work_budget is not None else rows['website_runtime_inputs']):
        value, proof = row
        capture_root, base = layouts[id(row)]
        prep_path = c.child(base, 'development_test', 'preparation.json', **_work_kwargs(work_budget)) if '/development_test/' in proof['path'] else c.child(base, 'preparation.json', **_work_kwargs(work_budget))
        candidates = [r for r in (_work_items(by_capture.get((capture_root, 'website_preparations'), []), work_budget) if work_budget is not None else by_capture.get((capture_root, 'website_preparations'), [])) if r[1]['path'] == prep_path]
        c.require(c.matches(value.get('preparation_digest'), **_work_kwargs(work_budget)), 'runtime_preparation_invalid', **_work_kwargs(work_budget))
        if candidates:
            c.require(any(r[0]['digest'] == value['preparation_digest'] for r in (_work_items(candidates, work_budget) if work_budget is not None else candidates)), 'runtime_preparation_invalid', **_work_kwargs(work_budget))
        else:
            context.missing('website_preparation', 'runtime_preparation_bytes_unavailable', [proof], prep_path,
                            {'digest': value['preparation_digest']})
    bound = {row['workspace_path']: row for row in (_work_items(old['seed']['source_attempt_obligations'], work_budget) if work_budget is not None else old['seed']['source_attempt_obligations']) if row['seed_disposition'] == 'matched_retained_bytes'}
    owner_bindings = {}
    for value, proof in (_work_items(context.decoded['source_snapshots'], work_budget) if work_budget is not None else context.decoded['source_snapshots']):
        if value.get('schema_version') != 'website_scene_source_binding.v1':
            continue
        workspace = proof['path'].rsplit('/', 1)[0]
        if workspace not in bound:
            continue
        c.require(value.get('intent_digest') == intent['intent_digest'] and value.get('owner') == intent['request']['owner'],
                  'binding_owner_invalid', **_work_kwargs(work_budget))
        if 'registration' not in value:
            context.missing('website_registration', 'snapshot_registration_unavailable', [proof])
            continue
        registration = context.selected(value['registration'], proof, {'website_registrations'})
        if registration:
            c.require(registration[0]['request_digest'] == (_work_call(work_budget, c.cross_runtime_canonical_digest, intent['request']) if work_budget is not None else c.cross_runtime_canonical_digest(intent['request']))
                and value.get('references') == registration[0]['references'], 'registration_owner_invalid', **_work_kwargs(work_budget))
        for standalone in (_work_items(binding_index.get(value['binding_digest'], []), work_budget) if work_budget is not None else binding_index.get(value['binding_digest'], [])):
            c.require(standalone[0] == value, 'binding_snapshot_invalid', **_work_kwargs(work_budget))
        owner_bindings.setdefault(tuple(value['registration'][k] for k in (_work_items(('path', 'sha256', 'size_bytes'), work_budget) if work_budget is not None else ('path', 'sha256', 'size_bytes'))), []).append((value, proof))
    observations = context.rows()
    for row in (_work_items(registrations, work_budget) if work_budget is not None else registrations):
        value, proof = row
        key = (proof['path'], proof['sha256'], proof['size_bytes'])
        _, selected, capture_root = registration_index[key]
        handoffs = by_capture.get((capture_root, 'website_handoffs'), [])
        matching = []
        for handoff in (_work_items(handoffs, work_budget) if work_budget is not None else handoffs):
            hvalue, hproof = handoff
            if hvalue.get('source_registration') is not None:
                reference = hvalue['source_registration']
                context.selected(reference, hproof, {'website_registrations'})
                if tuple(reference[k] for k in (_work_items(('path', 'sha256', 'size_bytes'), work_budget) if work_budget is not None else ('path', 'sha256', 'size_bytes'))) != key:
                    continue
            else:
                continue
            _handoff_edges(context, handoff, selected, **_work_kwargs(work_budget))
            matching.append(handoff)
        owners = owner_bindings.get(key, [])
        unique = len(owners) == 1 and len(matching) == 1 and all(selected.values())
        provenance = context.provenance(p for group in (_work_items(((proof,), (r[1] for r in (_work_items(selected.values(), work_budget) if work_budget is not None else selected.values()) if r),
            (r[1] for r in (_work_items(matching, work_budget) if work_budget is not None else matching)), (r[1] for r in (_work_items(owners, work_budget) if work_budget is not None else owners))), work_budget) if work_budget is not None else ((proof,), (r[1] for r in selected.values() if r),
            (r[1] for r in matching), (r[1] for r in owners))) for p in (_work_items(group, work_budget) if work_budget is not None else group))
        if unique:
            context.member(capture_root, 'capture_dependency', {'intent_id': context.intent_id,
                'binding_strength': 'owner_registration_original_handoff_exact_bytes'}, provenance)
        else:
            context.missing('website_capture', 'capture_join_unavailable_or_ambiguous', provenance,
                            selector={'registration_digest': value['registration_digest']})
        observations.append(c.observation(row, capture_binding_verified=bool(unique), capture_root=capture_root, **_work_kwargs(work_budget)))
    return observations


def _handoff_edges(context, handoff, selected, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    value, proof = handoff
    capture_root, base = _layout(context, handoff, **_work_kwargs(work_budget))
    c.require(value.get('preparation_path') in (None, c.child(base, 'preparation.json', **_work_kwargs(work_budget))), 'handoff_preparation_invalid', **_work_kwargs(work_budget))
    if selected['task_context'] and 'task_context_digest' in value:
        c.require(value['task_context_digest'] == selected['task_context'][0]['context_digest'], 'handoff_context_invalid', **_work_kwargs(work_budget))
    if selected['preparation']:
        preparation = selected['preparation']
        if '/development_test/' in preparation[1]['path']:
            development = value.get('development_test')
            c.require(isinstance(development, dict) and development.get('preparation_path') == preparation[1]['path'], 'handoff_development_invalid', **_work_kwargs(work_budget))
            source = preparation[0].get('development_test')
            c.require(isinstance(source, dict) and source.get('source_preparation_digest') == value.get('preparation_digest'),
                      'handoff_development_invalid', **_work_kwargs(work_budget))
            context.missing('website_original_preparation', 'original_preparation_selector_retained', [proof],
                c.child(base, 'preparation.json', **_work_kwargs(work_budget)), {'digest': value.get('preparation_digest')})
        else:
            c.require(value.get('preparation_digest') == preparation[0]['digest'], 'handoff_preparation_invalid', **_work_kwargs(work_budget))
    runtime = value.get('runtime_inputs')
    if selected['runtime_inputs'] and runtime is not None:
        c.require(isinstance(runtime, dict), 'handoff_runtime_invalid', **_work_kwargs(work_budget))
        if 'path' in runtime:
            c.require(runtime['path'] == selected['runtime_inputs'][1]['path'], 'handoff_runtime_invalid', **_work_kwargs(work_budget))
        if 'digest' in runtime:
            c.require(runtime['digest'] == selected['runtime_inputs'][0]['digest'], 'handoff_runtime_invalid', **_work_kwargs(work_budget))


def _available_capture_edges(context, rows, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    indexes = {role: {} for role in (_work_items(rows, work_budget) if work_budget is not None else rows)}
    for role, supplied in (_work_items(rows.items(), work_budget) if work_budget is not None else rows.items()):
        for row in (_work_items(supplied, work_budget) if work_budget is not None else supplied):
            indexes[role].setdefault(row[1]['path'], []).append(row)
    for row in (_work_items(rows['website_preparations'], work_budget) if work_budget is not None else rows['website_preparations']):
        value, proof = row
        development = value.get('development_test')
        if development is None:
            continue
        c.require(isinstance(development, dict), 'development_preparation_invalid', **_work_kwargs(work_budget))
        _, base = _layout(context, row, **_work_kwargs(work_budget))
        for field, role, expected, seal in (_work_items((
            ('source_preparation_digest', 'website_preparations', c.child(base, 'preparation.json', **_work_kwargs(work_budget)), 'digest'),
            ('source_task_context_digest', 'website_task_contexts', c.child(base, 'development_test', 'task_context.json', **_work_kwargs(work_budget)), 'context_digest')), work_budget) if work_budget is not None else (
            ('source_preparation_digest', 'website_preparations', c.child(base, 'preparation.json'), 'digest'),
            ('source_task_context_digest', 'website_task_contexts', c.child(base, 'development_test', 'task_context.json'), 'context_digest'))):
            if field in development:
                c.require(c.matches(development[field], **_work_kwargs(work_budget)), 'development_preparation_invalid', **_work_kwargs(work_budget))
                candidates = indexes[role].get(expected, [])
                if candidates:
                    c.require(any(r[0][seal] == development[field] for r in (_work_items(candidates, work_budget) if work_budget is not None else candidates)), 'development_preparation_invalid', **_work_kwargs(work_budget))
                else:
                    context.missing(role, 'development_selector_unavailable', [proof], expected, {'digest': development[field]})
        if ('source_task_context_digest' in development and isinstance(value.get('binding'), dict)
                and 'task_context_digest' in value['binding']):
            c.require(value['binding']['task_context_digest'] == development['source_task_context_digest'],
                      'development_context_binding_invalid', **_work_kwargs(work_budget))
    for row in (_work_items(rows['website_handoffs'], work_budget) if work_budget is not None else rows['website_handoffs']):
        value, proof = row
        _, base = _layout(context, row, **_work_kwargs(work_budget))
        original = c.child(base, 'preparation.json', **_work_kwargs(work_budget))
        if 'preparation_path' in value:
            c.require(value['preparation_path'] == original, 'handoff_preparation_invalid', **_work_kwargs(work_budget))
        if 'preparation_digest' in value:
            c.require(c.matches(value['preparation_digest'], **_work_kwargs(work_budget)), 'handoff_preparation_invalid', **_work_kwargs(work_budget))
            candidates = indexes['website_preparations'].get(original, [])
            if candidates:
                c.require(any(r[0]['digest'] == value['preparation_digest'] for r in (_work_items(candidates, work_budget) if work_budget is not None else candidates)), 'handoff_preparation_invalid', **_work_kwargs(work_budget))
        runtime = value.get('runtime_inputs')
        if runtime is not None:
            c.require(isinstance(runtime, dict), 'handoff_runtime_invalid', **_work_kwargs(work_budget))
            expected_paths = (c.child(base, 'native', 'runtime_inputs.json', **_work_kwargs(work_budget)), c.child(base, 'development_test', 'runtime_inputs.json', **_work_kwargs(work_budget)))
            if 'path' in runtime:
                c.require(runtime['path'] in expected_paths, 'handoff_runtime_invalid', **_work_kwargs(work_budget))
            if 'digest' in runtime:
                c.require(c.matches(runtime['digest'], **_work_kwargs(work_budget)), 'handoff_runtime_invalid', **_work_kwargs(work_budget))
                candidates = indexes['website_runtime_inputs'].get(runtime.get('path'), [])
                if candidates:
                    c.require(any(r[0]['digest'] == runtime['digest'] for r in (_work_items(candidates, work_budget) if work_budget is not None else candidates)), 'handoff_runtime_invalid', **_work_kwargs(work_budget))


def _manifest(context, row, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    value, proof = row
    c.c.seal(row, 'manifest_digest', **_work_kwargs(work_budget))
    c.require(value.get('status') == 'validated_pending_production_publication_and_submission'
        and c.matches(value.get('source_commit'), c.COMMIT, **_work_kwargs(work_budget)) and c.matches(value.get('request_digest'), **_work_kwargs(work_budget))
        and c.matches(value.get('input_namespace'), c.ID, **_work_kwargs(work_budget)) and value.get('raw_source_upload_allowed') is False
        and value.get('provider_allocated') is False, 'manifest_invalid', **_work_kwargs(work_budget))
    inventory = value.get('files')
    c.require(isinstance(inventory, list) and 1 <= len(inventory) <= 1024, 'manifest_inventory_invalid', **_work_kwargs(work_budget))
    paths, uris, by_uri = set(), set(), {}
    prefix = 's3://blueprint/task-evaluation/production-inputs/' + value['input_namespace'] + '/'
    for item in (_work_items(inventory, work_budget) if work_budget is not None else inventory):
        c.require(isinstance(item, dict) and (_work_collect(work_budget, set, item) if work_budget is not None else set(item)) == {'relative_path', 'uri', 'digest', 'size_bytes', 'publication_allowed'},
                  'manifest_row_invalid', **_work_kwargs(work_budget))
        relative = c.relative(item['relative_path'], **_work_kwargs(work_budget))
        c.require(isinstance(item['uri'], str) and len(item['uri']) <= 4096 and not any(ch.isspace() for ch in (_work_items(item['uri'], work_budget) if work_budget is not None else item['uri']))
            and c.matches(item['digest'], **_work_kwargs(work_budget)) and type(item['size_bytes']) is int and item['size_bytes'] > 0
            and type(item['publication_allowed']) is bool, 'manifest_row_invalid', **_work_kwargs(work_budget))
        c.require(relative not in paths and item['uri'] not in uris, 'manifest_duplicate', **_work_kwargs(work_budget))
        paths.add(relative)
        uris.add(item['uri'])
        by_uri[item['uri']] = item
        context.size(item['digest'], item['size_bytes'])
        if item['publication_allowed']:
            c.require(not relative.startswith('source/') and item['uri'] == prefix + relative, 'manifest_policy_invalid', **_work_kwargs(work_budget))
        else:
            c.require(relative.startswith('source/') and value.get('source') != 'website_capture_derivatives'
                and item['uri'].startswith(('https://', 's3://', 'gs://')), 'manifest_policy_invalid', **_work_kwargs(work_budget))
    return by_uri, prefix


def publication(context, old, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    receipts = context.known('submission_publications', 'task_evaluation_scene_configuration_submission_publication.v1', 'receipt_digest')
    rows = context.rows()
    bound = {r['workspace_path']: r for r in (_work_items(old['seed']['source_attempt_obligations'], work_budget) if work_budget is not None else old['seed']['source_attempt_obligations']) if r['seed_disposition'] == 'matched_retained_bytes'}
    manifests, request_index = {}, {}
    for row in (_work_items(context.decoded['source_submissions'], work_budget) if work_budget is not None else context.decoded['source_submissions']):
        schema = row[0].get('schema_version')
        if schema == 'task_evaluation_scene_configuration_submission_manifest.v1' and 'files' in row[0]:
            inventory, prefix = _manifest(context, row, **_work_kwargs(work_budget))
            manifests.setdefault((row[1]['sha256'], row[0]['manifest_digest']), []).append((row, inventory, prefix))
        elif schema == 'task_evaluation_launch_preparation_request.v1':
            request_index.setdefault((_work_call(work_budget, c.canonical_digest, row[0]) if work_budget is not None else c.canonical_digest(row[0])), []).append(row)
    for copies in (_work_items(manifests.values(), work_budget) if work_budget is not None else manifests.values()):
      for row, inventory, _ in (_work_items(copies, work_budget) if work_budget is not None else copies):
        for request in (_work_items(request_index.get(row[0]['request_digest'], []), work_budget) if work_budget is not None else request_index.get(row[0]['request_digest'], [])):
            if request[1]['path'].rsplit('/', 1)[0] != row[1]['path'].rsplit('/', 1)[0]:
                continue
            stack = [request[0]]
            while stack:
                if work_budget is not None:
                    work_budget.charge("values")
                item = stack.pop()
                if isinstance(item, dict):
                    if {'uri', 'digest', 'size_bytes'} <= (_work_collect(work_budget, set, item) if work_budget is not None else set(item)):
                        ref = inventory.get(item['uri'])
                        c.require(ref is not None and all(ref[k] == item[k] for k in (_work_items(('uri', 'digest', 'size_bytes'), work_budget) if work_budget is not None else ('uri', 'digest', 'size_bytes'))),
                                  'manifest_request_invalid', **_work_kwargs(work_budget))
                    stack.extend(_work_items(item.values(), work_budget) if work_budget is not None else item.values())
                elif isinstance(item, list):
                    stack.extend(_work_items(item, work_budget) if work_budget is not None else item)
    for row in (_work_items(receipts, work_budget) if work_budget is not None else receipts):
        value, proof = row
        c.require(value.get('status') == 'published_and_read_back' and c.matches(value.get('source_commit'), c.COMMIT, **_work_kwargs(work_budget))
            and c.matches(value.get('input_namespace'), c.ID, **_work_kwargs(work_budget)) and c.matches(value.get('manifest_sha256'), **_work_kwargs(work_budget))
            and c.matches(value.get('manifest_digest'), **_work_kwargs(work_budget)) and c.matches(value.get('request_digest'), **_work_kwargs(work_budget))
            and all(value.get(flag) is False for flag in (_work_items(('raw_source_uploaded', 'provider_allocated', 'run_submitted', 'global_atomic_create_claimed'), work_budget) if work_budget is not None else ('raw_source_uploaded', 'provider_allocated', 'run_submitted', 'global_atomic_create_claimed')))
            and value.get('full_byte_service_account_readback_passed') is True, 'publication_invalid', **_work_kwargs(work_budget))
        objects = value.get('published_objects')
        c.require(isinstance(objects, list) and 1 <= len(objects) <= 1025
            and isinstance(value.get('host_only_source_objects'), list)
            and len(value['host_only_source_objects']) <= 1024, 'publication_inventory_invalid', **_work_kwargs(work_budget))
        seen_uris, seen_paths = set(), set()
        prefix = 's3://blueprint/task-evaluation/production-inputs/' + value['input_namespace'] + '/'
        for item in (_work_items(objects, work_budget) if work_budget is not None else objects):
            c.require(isinstance(item, dict), 'publication_inventory_invalid', **_work_kwargs(work_budget))
            relative = c.relative(item.get('relative_path'), **_work_kwargs(work_budget))
            uri = item.get('uri')
            c.require(isinstance(uri, str) and len(uri) <= 4096 and uri == prefix + relative and not relative.startswith('source/')
                and not any(ch.isspace() for ch in (_work_items(uri, work_budget) if work_budget is not None else uri)) and c.matches(item.get('digest'), **_work_kwargs(work_budget))
                and type(item.get('size_bytes')) is int and item['size_bytes'] > 0
                and type(item.get('upload_performed')) is bool and item.get('full_byte_service_account_readback_passed') is True
                and uri not in seen_uris and relative not in seen_paths, 'publication_inventory_invalid', **_work_kwargs(work_budget))
            seen_uris.add(uri)
            seen_paths.add(relative)
        host_seen, host_uris = set(), set()
        for item in (_work_items(value['host_only_source_objects'], work_budget) if work_budget is not None else value['host_only_source_objects']):
            c.require(isinstance(item, dict) and (_work_collect(work_budget, set, item) if work_budget is not None else set(item)) == {'relative_path', 'uri', 'digest', 'size_bytes', 'publication_allowed'},
                      'publication_host_source_invalid', **_work_kwargs(work_budget))
            relative = c.relative(item['relative_path'], **_work_kwargs(work_budget))
            c.require(relative.startswith('source/') and isinstance(item['uri'], str) and len(item['uri']) <= 4096
                and item['uri'].startswith(('https://', 'gs://', 's3://')) and not any(ch.isspace() for ch in (_work_items(item['uri'], work_budget) if work_budget is not None else item['uri']))
                and c.matches(item['digest'], **_work_kwargs(work_budget)) and type(item['size_bytes']) is int and item['size_bytes'] > 0
                and item['publication_allowed'] is False and relative not in host_seen and item['uri'] not in host_uris,
                'publication_host_source_invalid', **_work_kwargs(work_budget))
            host_seen.add(relative)
            host_uris.add(item['uri'])
        workspace = proof['path'].rsplit('/', 1)[0]
        c.require(proof['path'].endswith('/publication.json') and c.under(proof['path'], c.child(context.roots['factory_output_root'], context.intent_id, **_work_kwargs(work_budget)), **_work_kwargs(work_budget)),
                  'publication_path_invalid', **_work_kwargs(work_budget))
        factory_rows = context.by_path['factories'].get(c.child(workspace, 'factory.json', **_work_kwargs(work_budget)), [])
        factory_ref = factory_rows[0][0].get('submission_manifest') if len(factory_rows) == 1 else None
        copies = manifests.get((value['manifest_sha256'], value['manifest_digest']), [])
        selected_rows = [r for r in (_work_items(copies, work_budget) if work_budget is not None else copies) if factory_ref == {k: r[0][1][k] for k in (_work_items(('path', 'sha256', 'size_bytes'), work_budget) if work_budget is not None else ('path', 'sha256', 'size_bytes'))}]
        selected = selected_rows[0] if len(selected_rows) == 1 else None
        # Raw SHA + document seal validates supplied content independently of
        # owner proof. Byte-identical copied content does not select a raw path:
        # only the exact factory tuple below can establish that provenance.
        observed_manifest = selected if selected is not None else copies[0] if copies else None
        if observed_manifest:
            manifest, inventory, prefix = observed_manifest
            c.require(all(value[k] == manifest[0][k] for k in (_work_items(('source_commit', 'input_namespace', 'request_digest'), work_budget) if work_budget is not None else ('source_commit', 'input_namespace', 'request_digest'))), 'publication_binding_invalid', **_work_kwargs(work_budget))
            expected = {uri: {k: r[k] for k in (_work_items(('relative_path', 'uri', 'digest', 'size_bytes'), work_budget) if work_budget is not None else ('relative_path', 'uri', 'digest', 'size_bytes'))} for uri, r in (_work_items(inventory.items(), work_budget) if work_budget is not None else inventory.items()) if r['publication_allowed']}
            expected[prefix + 'bundle_manifest.v1.json'] = {'relative_path': 'bundle_manifest.v1.json',
                'uri': prefix + 'bundle_manifest.v1.json', 'digest': manifest[1]['sha256'], 'size_bytes': manifest[1]['size_bytes']}
            objects = value.get('published_objects')
            c.require(isinstance(objects, list) and len(objects) == len(expected) and len(objects) <= 1025, 'publication_inventory_invalid', **_work_kwargs(work_budget))
            seen = set()
            for item in (_work_items(objects, work_budget) if work_budget is not None else objects):
                c.require(isinstance(item, dict) and item.get('uri') in expected and item['uri'] not in seen
                    and all(item.get(k) == v for k, v in (_work_items(expected[item['uri']].items(), work_budget) if work_budget is not None else expected[item['uri']].items()))
                    and type(item.get('upload_performed')) is bool and item.get('full_byte_service_account_readback_passed') is True,
                    'publication_inventory_invalid', **_work_kwargs(work_budget))
                seen.add(item['uri'])
            retained = value.get('host_only_source_objects')
            c.require(isinstance(retained, list) and retained == [r for r in (_work_items(manifest[0]['files'], work_budget) if work_budget is not None else manifest[0]['files']) if not r['publication_allowed']],
                      'publication_host_source_invalid', **_work_kwargs(work_budget))
        exact_factory = selected is not None and len(factory_rows) == 1 and factory_rows[0][0].get('submission_manifest') == {
            k: selected[0][1][k] for k in (_work_items(('path', 'sha256', 'size_bytes'), work_budget) if work_budget is not None else ('path', 'sha256', 'size_bytes'))}
        unique = exact_factory and workspace in bound
        if not unique:
            context.missing('submission_manifest', 'publication_owner_or_manifest_unavailable', [proof], selector={
                'manifest_sha256': value['manifest_sha256'], 'manifest_digest': value['manifest_digest']})
        rows.append(c.observation(row, historical_publication_binding_verified=unique, current_remote_readback_verified=False, **_work_kwargs(work_budget)))
    return rows
