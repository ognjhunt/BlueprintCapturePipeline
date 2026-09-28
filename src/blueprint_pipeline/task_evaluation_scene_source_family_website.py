"""Retained website capture/publication joins, never a capture or remote reader."""
from __future__ import annotations

from . import task_evaluation_scene_source_family_contracts as c

_CAPTURE_SCHEMAS = {
    'website_handoffs': ('website_scene_handoff.v1', 'digest'),
    'website_preparations': ('website_scene_preparation.v1', 'digest'),
    'website_runtime_inputs': ('website_scene_runtime_inputs.v1', 'digest'),
    'website_task_contexts': ('website_site_task_context.v1', 'context_digest'),
}


def _layout(context, row, *, identity=True):
    path, root = row[1]['path'], context.roots['pubsub_root']
    c.require(c.under(path, root), 'capture_path_invalid')
    parts = path[len(root) + 1:].split('/')
    c.require(len(parts) >= 8 and parts[1] == 'scenes' and parts[3] == 'captures'
        and parts[5:7] == ['pipeline', 'website_scene_preparation'], 'capture_path_invalid')
    capture = c.child(root, *parts[:5])
    base = c.child(capture, 'pipeline', 'website_scene_preparation')
    role = row[1]['role']
    allowed = {'website_handoffs': ['handoff.json'], 'website_preparations': ['preparation.json', 'development_test/preparation.json'],
        'website_task_contexts': ['task_context.json', 'development_test/task_context.json'],
        'website_runtime_inputs': ['native/runtime_inputs.json', 'development_test/runtime_inputs.json']}
    c.require('/'.join(parts[7:]) in allowed[role], 'capture_path_invalid')
    if identity and role in {'website_handoffs', 'website_task_contexts'}:
        c.require(row[0].get('scene_id') == parts[2] and row[0].get('capture_id') == parts[4], 'capture_identity_invalid')
    return capture, base


def capture(context, old):
    rows, layouts, by_capture = {}, {}, {}
    for role, (schema, seal) in _CAPTURE_SCHEMAS.items():
        rows[role] = context.known(role, schema, seal)
        for row in rows[role]:
            layout = _layout(context, row)
            layouts[id(row)] = layout
            by_capture.setdefault((layout[0], role), []).append(row)
    for value, proof in rows['website_handoffs']:
        if value.get('source_registration') is not None:
            context.selected(value['source_registration'], proof, {'website_registrations'})
    _available_capture_edges(context, rows)
    intent = context.decoded['intent'][0][0]
    registrations = context.known('website_registrations', 'website_scene_source_registration.v1', 'registration_digest')
    bindings = context.known('website_bindings', 'website_scene_source_binding.v1', 'binding_digest')
    registration_index, binding_index = {}, {}
    for row in registrations:
        value, proof = row
        c.require(value.get('provider_mutation_performed') is False and value.get('execution_authority_granted') is False
            and value.get('claim_ceiling') == 'development_only' and c.matches(value.get('request_digest')),
            'registration_invalid')
        c.require(proof['path'] == c.child(context.roots['website_source_binding_root'], value['request_digest'][7:] + '.json'),
                  'registration_path_invalid')
        refs = value.get('references')
        c.require(isinstance(refs, dict) and set(refs) == {'preparation', 'runtime_inputs', 'task_context'}, 'registration_invalid')
        selected = {role: context.selected(ref, proof, {'website_' + {'preparation': 'preparations',
            'runtime_inputs': 'runtime_inputs', 'task_context': 'task_contexts'}[role]}) for role, ref in refs.items()}
        selected_layouts = {_layout(context, ({}, {'path': reference['path'],
            'role': {'preparation': 'website_preparations', 'runtime_inputs': 'website_runtime_inputs',
                     'task_context': 'website_task_contexts'}[role]}), identity=False)[0]
            for role, reference in refs.items()}
        # Context lexical path identity can be checked without pretending absent
        # bytes exist; actual supplied context owns its typed identity check.
        context_path = refs['task_context']['path']
        capture_root = next(iter(selected_layouts))
        c.require(len(selected_layouts) == 1 and c.under(context_path, c.child(capture_root, 'pipeline', 'website_scene_preparation')),
                  'capture_reference_invalid')
        if selected['preparation']:
            preparation = selected['preparation'][0]
            c.require(preparation.get('intake_request') is not None
                and c.cross_runtime_canonical_digest(preparation['intake_request']) == value['request_digest'],
                'registration_request_invalid')
        registration_index[(proof['path'], proof['sha256'], proof['size_bytes'])] = row, selected, capture_root
    for row in bindings:
        value, proof = row
        c.require(c.matches(value.get('binding_id'), c.ID)
            and proof['path'] == c.child(context.roots['factory_output_root'], context.intent_id,
                                        'website-source', value['binding_digest'][7:] + '.json'), 'binding_path_invalid')
        selected = context.selected(value.get('registration'), proof, {'website_registrations'})
        c.require(isinstance(value.get('references'), dict) and set(value['references']) == {'preparation', 'runtime_inputs', 'task_context'},
                  'binding_invalid')
        if selected:
            c.require(value['references'] == selected[0]['references'], 'binding_reference_invalid')
        binding_index.setdefault(value['binding_digest'], []).append(row)
    # Check supported available runtime/preparation edges even when their exact
    # registration byte version or the unrelated handoff is missing.
    for row in rows['website_runtime_inputs']:
        value, proof = row
        capture_root, base = layouts[id(row)]
        prep_path = c.child(base, 'development_test', 'preparation.json') if '/development_test/' in proof['path'] else c.child(base, 'preparation.json')
        candidates = [r for r in by_capture.get((capture_root, 'website_preparations'), []) if r[1]['path'] == prep_path]
        c.require(c.matches(value.get('preparation_digest')), 'runtime_preparation_invalid')
        if candidates:
            c.require(any(r[0]['digest'] == value['preparation_digest'] for r in candidates), 'runtime_preparation_invalid')
        else:
            context.missing('website_preparation', 'runtime_preparation_bytes_unavailable', [proof], prep_path,
                            {'digest': value['preparation_digest']})
    bound = {row['workspace_path']: row for row in old['seed']['source_attempt_obligations'] if row['seed_disposition'] == 'matched_retained_bytes'}
    owner_bindings = {}
    for value, proof in context.decoded['source_snapshots']:
        if value.get('schema_version') != 'website_scene_source_binding.v1':
            continue
        workspace = proof['path'].rsplit('/', 1)[0]
        if workspace not in bound:
            continue
        c.require(value.get('intent_digest') == intent['intent_digest'] and value.get('owner') == intent['request']['owner'],
                  'binding_owner_invalid')
        if 'registration' not in value:
            context.missing('website_registration', 'snapshot_registration_unavailable', [proof])
            continue
        registration = context.selected(value['registration'], proof, {'website_registrations'})
        if registration:
            c.require(registration[0]['request_digest'] == c.cross_runtime_canonical_digest(intent['request'])
                and value.get('references') == registration[0]['references'], 'registration_owner_invalid')
        for standalone in binding_index.get(value['binding_digest'], []):
            c.require(standalone[0] == value, 'binding_snapshot_invalid')
        owner_bindings.setdefault(tuple(value['registration'][k] for k in ('path', 'sha256', 'size_bytes')), []).append((value, proof))
    observations = context.rows()
    for row in registrations:
        value, proof = row
        key = (proof['path'], proof['sha256'], proof['size_bytes'])
        _, selected, capture_root = registration_index[key]
        handoffs = by_capture.get((capture_root, 'website_handoffs'), [])
        matching = []
        for handoff in handoffs:
            hvalue, hproof = handoff
            if hvalue.get('source_registration') is not None:
                reference = hvalue['source_registration']
                context.selected(reference, hproof, {'website_registrations'})
                if tuple(reference[k] for k in ('path', 'sha256', 'size_bytes')) != key:
                    continue
            else:
                continue
            _handoff_edges(context, handoff, selected)
            matching.append(handoff)
        owners = owner_bindings.get(key, [])
        unique = len(owners) == 1 and len(matching) == 1 and all(selected.values())
        provenance = [proof, *[r[1] for r in selected.values() if r], *[r[1] for r in matching], *[r[1] for r in owners]]
        if unique:
            context.member(capture_root, 'capture_dependency', {'intent_id': context.intent_id,
                'binding_strength': 'owner_registration_original_handoff_exact_bytes'}, provenance)
        else:
            context.missing('website_capture', 'capture_join_unavailable_or_ambiguous', provenance,
                            selector={'registration_digest': value['registration_digest']})
        observations.append(c.observation(row, capture_binding_verified=bool(unique), capture_root=capture_root))
    return observations


def _handoff_edges(context, handoff, selected):
    value, proof = handoff
    capture_root, base = _layout(context, handoff)
    c.require(value.get('preparation_path') in (None, c.child(base, 'preparation.json')), 'handoff_preparation_invalid')
    if selected['task_context'] and 'task_context_digest' in value:
        c.require(value['task_context_digest'] == selected['task_context'][0]['context_digest'], 'handoff_context_invalid')
    if selected['preparation']:
        preparation = selected['preparation']
        if '/development_test/' in preparation[1]['path']:
            development = value.get('development_test')
            c.require(isinstance(development, dict) and development.get('preparation_path') == preparation[1]['path'], 'handoff_development_invalid')
            source = preparation[0].get('development_test')
            c.require(isinstance(source, dict) and source.get('source_preparation_digest') == value.get('preparation_digest'),
                      'handoff_development_invalid')
            context.missing('website_original_preparation', 'original_preparation_selector_retained', [proof],
                c.child(base, 'preparation.json'), {'digest': value.get('preparation_digest')})
        else:
            c.require(value.get('preparation_digest') == preparation[0]['digest'], 'handoff_preparation_invalid')
    runtime = value.get('runtime_inputs')
    if selected['runtime_inputs'] and runtime is not None:
        c.require(isinstance(runtime, dict), 'handoff_runtime_invalid')
        if 'path' in runtime:
            c.require(runtime['path'] == selected['runtime_inputs'][1]['path'], 'handoff_runtime_invalid')
        if 'digest' in runtime:
            c.require(runtime['digest'] == selected['runtime_inputs'][0]['digest'], 'handoff_runtime_invalid')


def _available_capture_edges(context, rows):
    indexes = {role: {} for role in rows}
    for role, supplied in rows.items():
        for row in supplied:
            indexes[role].setdefault(row[1]['path'], []).append(row)
    for row in rows['website_preparations']:
        value, proof = row
        development = value.get('development_test')
        if development is None:
            continue
        c.require(isinstance(development, dict), 'development_preparation_invalid')
        _, base = _layout(context, row)
        for field, role, expected, seal in (
            ('source_preparation_digest', 'website_preparations', c.child(base, 'preparation.json'), 'digest'),
            ('task_context_digest', 'website_task_contexts', c.child(base, 'development_test', 'task_context.json'), 'context_digest')):
            if field in development:
                c.require(c.matches(development[field]), 'development_preparation_invalid')
                candidates = indexes[role].get(expected, [])
                if candidates:
                    c.require(any(r[0][seal] == development[field] for r in candidates), 'development_preparation_invalid')
                else:
                    context.missing(role, 'development_selector_unavailable', [proof], expected, {'digest': development[field]})
    for row in rows['website_handoffs']:
        value, proof = row
        _, base = _layout(context, row)
        original = c.child(base, 'preparation.json')
        if 'preparation_path' in value:
            c.require(value['preparation_path'] == original, 'handoff_preparation_invalid')
        if 'preparation_digest' in value:
            c.require(c.matches(value['preparation_digest']), 'handoff_preparation_invalid')
            candidates = indexes['website_preparations'].get(original, [])
            if candidates:
                c.require(any(r[0]['digest'] == value['preparation_digest'] for r in candidates), 'handoff_preparation_invalid')
        runtime = value.get('runtime_inputs')
        if runtime is not None:
            c.require(isinstance(runtime, dict), 'handoff_runtime_invalid')
            expected_paths = (c.child(base, 'native', 'runtime_inputs.json'), c.child(base, 'development_test', 'runtime_inputs.json'))
            if 'path' in runtime:
                c.require(runtime['path'] in expected_paths, 'handoff_runtime_invalid')
            if 'digest' in runtime:
                c.require(c.matches(runtime['digest']), 'handoff_runtime_invalid')
                candidates = indexes['website_runtime_inputs'].get(runtime.get('path'), [])
                if candidates:
                    c.require(any(r[0]['digest'] == runtime['digest'] for r in candidates), 'handoff_runtime_invalid')


def _manifest(context, row):
    value, proof = row
    c.c.seal(row, 'manifest_digest')
    c.require(value.get('status') == 'validated_pending_production_publication_and_submission'
        and c.matches(value.get('source_commit'), c.COMMIT) and c.matches(value.get('request_digest'))
        and c.matches(value.get('input_namespace'), c.ID) and value.get('raw_source_upload_allowed') is False
        and value.get('provider_allocated') is False, 'manifest_invalid')
    inventory = value.get('files')
    c.require(isinstance(inventory, list) and 1 <= len(inventory) <= 1024, 'manifest_inventory_invalid')
    paths, uris, by_uri = set(), set(), {}
    prefix = 's3://blueprint/task-evaluation/production-inputs/' + value['input_namespace'] + '/'
    for item in inventory:
        c.require(isinstance(item, dict) and set(item) == {'relative_path', 'uri', 'digest', 'size_bytes', 'publication_allowed'},
                  'manifest_row_invalid')
        relative = c.relative(item['relative_path'])
        c.require(isinstance(item['uri'], str) and len(item['uri']) <= 4096 and not any(ch.isspace() for ch in item['uri'])
            and c.matches(item['digest']) and type(item['size_bytes']) is int and item['size_bytes'] > 0
            and type(item['publication_allowed']) is bool, 'manifest_row_invalid')
        c.require(relative not in paths and item['uri'] not in uris, 'manifest_duplicate')
        paths.add(relative)
        uris.add(item['uri'])
        by_uri[item['uri']] = item
        context.size(item['digest'], item['size_bytes'])
        if item['publication_allowed']:
            c.require(not relative.startswith('source/') and item['uri'] == prefix + relative, 'manifest_policy_invalid')
        else:
            c.require(relative.startswith('source/') and value.get('source') != 'website_capture_derivatives'
                and item['uri'].startswith(('https://', 's3://', 'gs://')), 'manifest_policy_invalid')
    return by_uri, prefix


def publication(context, old):
    receipts = context.known('submission_publications', 'task_evaluation_scene_configuration_submission_publication.v1', 'receipt_digest')
    rows = context.rows()
    bound = {r['workspace_path']: r for r in old['seed']['source_attempt_obligations'] if r['seed_disposition'] == 'matched_retained_bytes'}
    manifests, requests = {}, {}
    for row in context.decoded['source_submissions']:
        schema = row[0].get('schema_version')
        if schema == 'task_evaluation_scene_configuration_submission_manifest.v1' and 'files' in row[0]:
            inventory, prefix = _manifest(context, row)
            manifests.setdefault((row[1]['sha256'], row[0]['manifest_digest']), []).append((row, inventory, prefix))
        elif schema == 'task_evaluation_launch_preparation_request.v1':
            requests.setdefault(c.canonical_digest(row[0]), []).append(row)
    for copies in manifests.values():
      for row, inventory, _ in copies:
        for request in requests.get(row[0]['request_digest'], []):
            if request[1]['path'].rsplit('/', 1)[0] != row[1]['path'].rsplit('/', 1)[0]:
                continue
            stack = [request[0]]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    if {'uri', 'digest', 'size_bytes'} <= set(item):
                        ref = inventory.get(item['uri'])
                        c.require(ref is not None and all(ref[k] == item[k] for k in ('uri', 'digest', 'size_bytes')),
                                  'manifest_request_invalid')
                    stack.extend(item.values())
                elif isinstance(item, list):
                    stack.extend(item)
    for row in receipts:
        value, proof = row
        c.require(value.get('status') == 'published_and_read_back' and c.matches(value.get('source_commit'), c.COMMIT)
            and c.matches(value.get('input_namespace'), c.ID) and c.matches(value.get('manifest_sha256'))
            and c.matches(value.get('manifest_digest')) and c.matches(value.get('request_digest'))
            and all(value.get(flag) is False for flag in ('raw_source_uploaded', 'provider_allocated', 'run_submitted', 'global_atomic_create_claimed'))
            and value.get('full_byte_service_account_readback_passed') is True, 'publication_invalid')
        objects = value.get('published_objects')
        c.require(isinstance(objects, list) and 1 <= len(objects) <= 1025
            and isinstance(value.get('host_only_source_objects'), list)
            and len(value['host_only_source_objects']) <= 1024, 'publication_inventory_invalid')
        seen_uris, seen_paths = set(), set()
        for item in objects:
            c.require(isinstance(item, dict), 'publication_inventory_invalid')
            relative = c.relative(item.get('relative_path'))
            uri = item.get('uri')
            c.require(isinstance(uri, str) and len(uri) <= 4096 and uri.startswith('s3://blueprint/task-evaluation/production-inputs/')
                and not any(ch.isspace() for ch in uri) and c.matches(item.get('digest'))
                and type(item.get('size_bytes')) is int and item['size_bytes'] > 0
                and type(item.get('upload_performed')) is bool and item.get('full_byte_service_account_readback_passed') is True
                and uri not in seen_uris and relative not in seen_paths, 'publication_inventory_invalid')
            seen_uris.add(uri)
            seen_paths.add(relative)
        host_seen = set()
        for item in value['host_only_source_objects']:
            c.require(isinstance(item, dict) and set(item) == {'relative_path', 'uri', 'digest', 'size_bytes', 'publication_allowed'},
                      'publication_host_source_invalid')
            relative = c.relative(item['relative_path'])
            c.require(relative.startswith('source/') and isinstance(item['uri'], str) and len(item['uri']) <= 4096
                and item['uri'].startswith(('https://', 'gs://', 's3://')) and not any(ch.isspace() for ch in item['uri'])
                and c.matches(item['digest']) and type(item['size_bytes']) is int and item['size_bytes'] > 0
                and item['publication_allowed'] is False and relative not in host_seen, 'publication_host_source_invalid')
            host_seen.add(relative)
        workspace = proof['path'].rsplit('/', 1)[0]
        c.require(proof['path'].endswith('/publication.json') and c.under(proof['path'], c.child(context.roots['factory_output_root'], context.intent_id)),
                  'publication_path_invalid')
        factory_rows = context.by_path['factories'].get(c.child(workspace, 'factory.json'), [])
        factory_ref = factory_rows[0][0].get('submission_manifest') if len(factory_rows) == 1 else None
        copies = manifests.get((value['manifest_sha256'], value['manifest_digest']), [])
        selected_rows = [r for r in copies if factory_ref == {k: r[0][1][k] for k in ('path', 'sha256', 'size_bytes')}]
        selected = selected_rows[0] if len(selected_rows) == 1 else None
        if selected:
            manifest, inventory, prefix = selected
            c.require(all(value[k] == manifest[0][k] for k in ('source_commit', 'input_namespace', 'request_digest')), 'publication_binding_invalid')
            expected = {uri: {k: r[k] for k in ('relative_path', 'uri', 'digest', 'size_bytes')} for uri, r in inventory.items() if r['publication_allowed']}
            expected[prefix + 'bundle_manifest.v1.json'] = {'relative_path': 'bundle_manifest.v1.json',
                'uri': prefix + 'bundle_manifest.v1.json', 'digest': manifest[1]['sha256'], 'size_bytes': manifest[1]['size_bytes']}
            objects = value.get('published_objects')
            c.require(isinstance(objects, list) and len(objects) == len(expected) and len(objects) <= 1025, 'publication_inventory_invalid')
            seen = set()
            for item in objects:
                c.require(isinstance(item, dict) and item.get('uri') in expected and item['uri'] not in seen
                    and all(item.get(k) == v for k, v in expected[item['uri']].items())
                    and type(item.get('upload_performed')) is bool and item.get('full_byte_service_account_readback_passed') is True,
                    'publication_inventory_invalid')
                seen.add(item['uri'])
            retained = value.get('host_only_source_objects')
            c.require(isinstance(retained, list) and retained == [r for r in manifest[0]['files'] if not r['publication_allowed']],
                      'publication_host_source_invalid')
        exact_factory = selected is not None and len(factory_rows) == 1 and factory_rows[0][0].get('submission_manifest') == {
            k: selected[0][1][k] for k in ('path', 'sha256', 'size_bytes')}
        unique = exact_factory and workspace in bound
        if not unique:
            context.missing('submission_manifest', 'publication_owner_or_manifest_unavailable', [proof], selector={
                'manifest_sha256': value['manifest_sha256'], 'manifest_digest': value['manifest_digest']})
        rows.append(c.observation(row, historical_publication_binding_verified=unique, current_remote_readback_verified=False))
    return rows
