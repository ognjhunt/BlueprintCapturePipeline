"""Pure activation and historical owner-attempt launch joins; no runtime imports."""
from __future__ import annotations

from .task_evaluation_scene_lineage_budget import _work_collect, _work_order, _work, _work_hash, _work_items, _work_kwargs

from pathlib import PurePosixPath

from . import task_evaluation_scene_downstream_contracts as c
from .task_evaluation_scene_attempt_binding import scene_execution_binding_blockers


def activation(context, seed, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    rows, matched, envelopes = context.rows(), {}, {}
    for row in (_work_items(context.decoded['activation_envelopes'], work_budget) if work_budget is not None else context.decoded['activation_envelopes']):
        envelopes.setdefault(PurePosixPath(row[1]['path']).name, []).append(row)
    seed_paths = {m['path'] for m in (_work_items(seed['members'], work_budget) if work_budget is not None else seed['members']) if m['kind'] == 'configuration_progression_workspace'}
    for row in (_work_items(context.decoded['activation_results'], work_budget) if work_budget is not None else context.decoded['activation_results']):
        value, proof = row
        c.require(c.under(proof['path'], c.child(context.roots['activation_queue_root'], 'results', **_work_kwargs(work_budget)), **_work_kwargs(work_budget))
                  and PurePosixPath(proof['path']).parent == PurePosixPath(context.roots['activation_queue_root']) / 'results', 'activation_path_invalid', **_work_kwargs(work_budget))
        if value.get('schema_version') != 'task_evaluation_launch_activation_result.v1':
            rows.append(c.observation(row, reason='activation_schema_unproven', **_work_kwargs(work_budget)))
            continue
        c.seal(row, 'result_digest', **_work_kwargs(work_budget))
        c.require(c.matches(value.get('activation_id'), c.ACTIVATION_ID, **_work_kwargs(work_budget)) and isinstance(value.get('status'), str), 'activation_invalid', **_work_kwargs(work_budget))
        candidates = envelopes.get(PurePosixPath(proof['path']).name, [])
        envelope = candidates[0] if len(candidates) == 1 else None
        status = value['status']
        known = status in {'profile_authority_materialized_no_execution', 'policy_campaign_queue_materialized_no_execution'}
        if known or status == 'blocked':
            c.require(value.get('provider_mutation_performed') is False and value.get('paid_execution_requested') is False
                      and isinstance(value.get('blockers'), list), 'activation_scope_invalid', **_work_kwargs(work_budget))
        sources = context.provenance((proof,))
        if known:
            for field in (_work_items(('preparation_id', 'team_namespace', 'lane'), work_budget) if work_budget is not None else ('preparation_id', 'team_namespace', 'lane')):
                c.require(c.matches(value.get(field), c.ID, **_work_kwargs(work_budget)), 'activation_invalid', **_work_kwargs(work_budget))
            c.require(c.matches(value.get('source_commit'), c.COMMIT, **_work_kwargs(work_budget)), 'activation_invalid', **_work_kwargs(work_budget))
            for field in (_work_items(('preparation_result_digest', 'release_window_digest'), work_budget) if work_budget is not None else ('preparation_result_digest', 'release_window_digest')):
                c.require(c.matches(value.get(field), **_work_kwargs(work_budget)), 'activation_invalid', **_work_kwargs(work_budget))
            if status == 'profile_authority_materialized_no_execution':
                c.require(c.matches(value.get('profile_id'), c.LAUNCH_ID, **_work_kwargs(work_budget)), 'activation_invalid', **_work_kwargs(work_budget))
                for field in (_work_items(('profile_digest', 'profile_publication_receipt_digest', 'standing_authorization_digest'), work_budget) if work_budget is not None else ('profile_digest', 'profile_publication_receipt_digest', 'standing_authorization_digest')):
                    c.require(c.matches(value.get(field), **_work_kwargs(work_budget)), 'activation_invalid', **_work_kwargs(work_budget))
                c.require(all(value.get(k) is True for k in (_work_items(('full_byte_activation_reference_readback_passed',
                          'profile_publication_performed', 'catalog_mutation_performed', 'standing_authorization_published'), work_budget) if work_budget is not None else ('full_byte_activation_reference_readback_passed',
                          'profile_publication_performed', 'catalog_mutation_performed', 'standing_authorization_published'))), 'activation_scope_invalid', **_work_kwargs(work_budget))
                for role, field in (_work_items((('profile_publication_receipt', 'profile_publication_receipt_digest'),
                                    ('standing_authorization', 'standing_authorization_digest')), work_budget) if work_budget is not None else (('profile_publication_receipt', 'profile_publication_receipt_digest'),
                                    ('standing_authorization', 'standing_authorization_digest'))):
                    context.missing(role, 'activation_selector_bytes_unavailable', [proof],
                                    selector={'sha256': value[field], 'profile_id': value['profile_id']})
            if envelope:
                request = envelope[0]['request']
                c.require(value['activation_id'] == request['activation_id'] and value['preparation_id'] == request['preparation']['preparation_id']
                          and value['preparation_result_digest'] == request['preparation']['result_digest']
                          and value['team_namespace'] == request['team_namespace'] and value['lane'] == request['lane']
                          and value['source_commit'] == request['expected_production_commit'], 'activation_identity_invalid', **_work_kwargs(work_budget))
                sources.append(envelope[1])
        elif envelope:
            c.require(value['activation_id'] == envelope[0]['request']['activation_id'], 'activation_identity_invalid', **_work_kwargs(work_budget))
        expected = c.child(context.roots['configuration_progression_root'], 'scene-configuration-activations', value.get('preparation_id', 'unproven'), **_work_kwargs(work_budget))
        bound = status == 'profile_authority_materialized_no_execution' and envelope is not None and expected in seed_paths
        reason = None if bound else 'activation_envelope_ambiguous' if len(candidates) > 1 else 'activation_output_scope_unproven'
        if len(candidates) > 1:
            sources += context.provenance(r[1] for r in (_work_items(candidates, work_budget) if work_budget is not None else candidates))
        if bound:
            context.member(c.child(context.roots['activation_output_root'], value['activation_id'], **_work_kwargs(work_budget)), 'activation_workspace',
                           {'activation_id': value['activation_id'], 'preparation_id': value['preparation_id']}, sources)
            matched.setdefault(value['result_digest'], []).append(row)
        else:
            context.missing('activation_owner_join', reason, sources, selector={'activation_id': value['activation_id']})
        if status == 'policy_campaign_queue_materialized_no_execution':
            context.missing('campaign_outputs', 'campaign_output_identity_unproven', sources)
        rows.append(c.observation(row, status='matched_retained_bytes' if bound else 'kept_unresolved', reason=reason,
                                  activation_id=value['activation_id'], release_window_digest=value.get('release_window_digest'),
                                  window_binding_verified=False, source_provenance=sources, **_work_kwargs(work_budget)))
    result_names = {PurePosixPath(p['path']).name for _, p in (_work_items(context.decoded['activation_results'], work_budget) if work_budget is not None else context.decoded['activation_results'])}
    for filename, candidates in (_work_items(envelopes.items(), work_budget) if work_budget is not None else envelopes.items()):
        if filename not in result_names:
            context.missing('activation_result', 'activation_result_unavailable', context.provenance(r[1] for r in (_work_items(candidates, work_budget) if work_budget is not None else candidates)),
                            c.child(context.roots['activation_queue_root'], 'results', filename, **_work_kwargs(work_budget)))
    return rows, matched


def owner(context, row, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    profile, proof = row
    intent = context.decoded['intent'][0][0]
    direct = profile.get('scene_attempt_binding')
    plan = profile.get('internal_policy_canary_execution_plan')
    policy = plan.get('scene_policy_binding') if isinstance(plan, dict) else None
    if direct is not None or 'scene_attempt_id' in profile or ('scene_intent_digest' in profile and policy is None):
        c.require(not scene_execution_binding_blockers(profile), 'owner_binding_invalid', **_work_kwargs(work_budget))
    if policy is not None:
        c.require(isinstance(policy, dict) and (_work_collect(work_budget, set, policy) if work_budget is not None else set(policy)) == {'schema_version', 'scene_intent_digest', 'attempt_id',
                  'policy_candidates', 'runtime_digest', 'input_digest', 'binding_digest'}
                  and policy.get('schema_version') == 'task_evaluation_scene_policy_binding.v1', 'owner_binding_invalid', **_work_kwargs(work_budget))
        c.seal((policy, {}), 'binding_digest', **_work_kwargs(work_budget))
        for k in (_work_items(('scene_intent_digest', 'runtime_digest', 'input_digest'), work_budget) if work_budget is not None else ('scene_intent_digest', 'runtime_digest', 'input_digest')):
            c.require(c.matches(policy[k], **_work_kwargs(work_budget)), 'owner_binding_invalid', **_work_kwargs(work_budget))
        c.require(c.matches(policy['attempt_id'], c.ID, **_work_kwargs(work_budget)), 'owner_binding_invalid', **_work_kwargs(work_budget))
        if 'scene_intent_digest' in profile:
            c.require(profile['scene_intent_digest'] == policy['scene_intent_digest'], 'owner_binding_invalid', **_work_kwargs(work_budget))
        candidates = policy['policy_candidates']
        c.require(isinstance(candidates, list) and len(candidates) == 2 and all(isinstance(r, dict) and (_work_collect(work_budget, set, r) if work_budget is not None else set(r)) == {'id', 'artifact_digest'}
                  and c.matches(r['id'], c.ID, **_work_kwargs(work_budget)) and c.matches(r['artifact_digest'], **_work_kwargs(work_budget)) for r in (_work_items(candidates, work_budget) if work_budget is not None else candidates))
                  and candidates[0]['id'] != candidates[1]['id'], 'owner_binding_invalid', **_work_kwargs(work_budget))
        if direct is not None:
            c.require(all(direct[a] == policy[b] for a, b in (_work_items((('intent_digest', 'scene_intent_digest'), ('attempt_id', 'attempt_id'),
                      ('runtime_digest', 'runtime_digest'), ('input_digest', 'input_digest')), work_budget) if work_budget is not None else (('intent_digest', 'scene_intent_digest'), ('attempt_id', 'attempt_id'),
                      ('runtime_digest', 'runtime_digest'), ('input_digest', 'input_digest')))), 'owner_binding_invalid', **_work_kwargs(work_budget))
    binding = direct or policy
    if not binding:
        return None, 'launch_owner_unproven', [proof]
    digest = binding.get('intent_digest', binding.get('scene_intent_digest'))
    if digest != intent['intent_digest']:
        return None, 'foreign_owner_unproven', [proof]
    if direct is not None:
        c.require(direct['intent_id'] == context.intent_id, 'owner_binding_invalid', **_work_kwargs(work_budget))
    if policy is not None:
        expected = intent['request'].get('execution', {}).get('policy_candidates')
        c.require(isinstance(expected, list) and (_work_order(work_budget, sorted, candidates, key=lambda r: r['id']) if work_budget is not None else sorted(candidates, key=lambda r: r['id'])) == (_work_order(work_budget, sorted, expected, key=lambda r: r['id']) if work_budget is not None else sorted(expected, key=lambda r: r['id'])), 'owner_pair_invalid', **_work_kwargs(work_budget))
    attempt_id = binding['attempt_id']
    expected_path = c.child(context.roots['intent_root'], context.intent_id, 'attempts', attempt_id + '.json', **_work_kwargs(work_budget))
    matches = context.by_path['attempts'].get(expected_path, [])
    if not matches:
        context.missing('owner_attempt', 'owner_attempt_bytes_unavailable', [proof], expected_path)
        return None, 'owner_attempt_bytes_unavailable', [proof]
    c.require(len(matches) == 1, 'owner_attempt_ambiguous', **_work_kwargs(work_budget))
    attempt, provenance = matches[0]
    c.require(attempt.get('schema_version') == 'task_evaluation_scene_attempt.v1', 'owner_attempt_invalid', **_work_kwargs(work_budget))
    c.seal(matches[0], 'attempt_digest', cross=True, **_work_kwargs(work_budget))
    c.require(all(attempt.get(k) == binding[k] for k in (_work_items(('attempt_id', 'runtime_digest', 'input_digest'), work_budget) if work_budget is not None else ('attempt_id', 'runtime_digest', 'input_digest')))
              and attempt.get('intent_digest') == digest and attempt.get('intent_id') == context.intent_id
              and attempt.get('source_commit') == profile['source_commit'], 'owner_attempt_invalid', **_work_kwargs(work_budget))
    return attempt_id, None, [proof, provenance]


def launches(context, activations, *, work_budget=None):
    if work_budget is None:
        work_budget = getattr(context, "work_budget", None)
    if work_budget is not None:
        _work(work_budget)
    profiles, available_profiles, request_index, request_launch, observations, bound = {}, {}, {}, {}, context.rows(), {}
    for row in (_work_items(context.decoded['launch_profiles'], work_budget) if work_budget is not None else context.decoded['launch_profiles']):
        profile, proof = row
        p = PurePosixPath(proof['path'])
        c.require(p.name == 'launch_profile.json' and (c.under(proof['path'], context.roots['launch_execution_root'], **_work_kwargs(work_budget))
                  or c.under(proof['path'], context.roots['terminal_result_root'], **_work_kwargs(work_budget))), 'launch_path_invalid', **_work_kwargs(work_budget))
        if profile.get('schema_version') != 'task_evaluation_launch_profile.v1':
            observations.append(c.observation(row, reason='launch_profile_schema_unproven', **_work_kwargs(work_budget)))
            continue
        c.require(c.matches(profile.get('source_commit'), c.COMMIT, **_work_kwargs(work_budget)) and c.matches(profile.get('profile_id'), c.LAUNCH_ID, **_work_kwargs(work_budget)), 'profile_invalid', **_work_kwargs(work_budget))
        c.seal(row, 'profile_digest', **_work_kwargs(work_budget))
        identity = owner(context, row, **_work_kwargs(work_budget))  # Validate supplied owner even if no request exists.
        profiles.setdefault((str(p.parent), profile['profile_digest']), []).append((row, identity))
        available_profiles.setdefault(str(p.parent), []).append(row)
    for row in (_work_items(context.decoded['launch_requests'], work_budget) if work_budget is not None else context.decoded['launch_requests']):
        request, proof = row
        if request.get('schema_version') != 'task_evaluation_launch_request.v1':
            c.require(PurePosixPath(proof['path']).name == 'launch_request.json'
                      and (c.under(proof['path'], context.roots['launch_execution_root'], **_work_kwargs(work_budget)) or c.under(proof['path'], context.roots['terminal_result_root'], **_work_kwargs(work_budget))), 'launch_path_invalid', **_work_kwargs(work_budget))
            observations.append(c.observation(row, reason='launch_request_schema_unproven', **_work_kwargs(work_budget)))
            continue
        c.require(all(c.matches(request.get(k), c.LAUNCH_ID, **_work_kwargs(work_budget)) for k in (_work_items(('launch_id', 'run_id', 'launch_profile_id'), work_budget) if work_budget is not None else ('launch_id', 'run_id', 'launch_profile_id')))
                  and c.matches(request.get('source_commit'), c.COMMIT, **_work_kwargs(work_budget)) and c.matches(request.get('launch_profile_digest'), **_work_kwargs(work_budget)), 'launch_request_invalid', **_work_kwargs(work_budget))
        c.seal(row, 'request_digest', **_work_kwargs(work_budget))
        launch_path = c.child(context.roots['launch_execution_root'], request['launch_id'], 'launch_request.json', **_work_kwargs(work_budget))
        directory = str(PurePosixPath(proof['path']).parent)
        import hashlib
        terminal = c.child(context.roots['terminal_result_root'], context.intent_id, **_work_kwargs(work_budget))
        allowed = {launch_path, c.child(terminal, 'launch_request.json', **_work_kwargs(work_budget)),
                   c.child(terminal, 'runs', (_work_hash(work_budget, hashlib.sha256, request['run_id'].encode()) if work_budget is not None else hashlib.sha256(request['run_id'].encode())).hexdigest(), 'launch_request.json', **_work_kwargs(work_budget))}
        c.require(proof['path'] in allowed, 'launch_path_invalid', **_work_kwargs(work_budget))
        candidates = profiles.get((directory, request['launch_profile_digest']), [])
        available = available_profiles.get(directory, [])
        c.require(not available or candidates, 'launch_profile_identity_invalid', **_work_kwargs(work_budget))
        reason, sources, attempt = 'launch_profile_unavailable', [proof], None
        if len(candidates) == 1:
            profile, identity = candidates[0]
            c.require(profile[0]['source_commit'] == request['source_commit']
                      and profile[0]['profile_id'] == request['launch_profile_id'], 'launch_profile_identity_invalid', **_work_kwargs(work_budget))
            attempt, reason, owner_sources = identity
            sources += owner_sources
        elif candidates:
            reason = 'launch_profile_ambiguous'
        if reason:
            context.missing('launch_join', reason, sources, selector={'run_id': request['run_id']})
        else:
            bound.setdefault(request['run_id'], []).append((row, sources))
            kind = 'launch_workspace' if proof['path'] == launch_path else 'terminal_index_workspace'
            context.member(directory, kind, {'intent_id': context.intent_id, 'attempt_id': attempt,
                                             'launch_id': request['launch_id'], 'run_id': request['run_id']}, sources)
        request_index.setdefault(request['request_digest'], []).append(row)
        request_launch.setdefault(request['launch_id'], []).append(row)
        observations.append(c.observation(row, status='matched_retained_bytes' if reason is None else 'kept_unresolved', reason=reason,
                                          launch_id=request['launch_id'], run_id=request['run_id'], source_provenance=sources, **_work_kwargs(work_budget)))
    for row in (_work_items(context.decoded['launch_receipts'], work_budget) if work_budget is not None else context.decoded['launch_receipts']):
        value, proof = row
        c.require(PurePosixPath(proof['path']).name == 'launch_receipt.json' and c.under(proof['path'], context.roots['launch_execution_root'], **_work_kwargs(work_budget)), 'launch_path_invalid', **_work_kwargs(work_budget))
        if value.get('schema_version') != 'task_evaluation_launch_receipt.v1':
            observations.append(c.observation(row, reason='launch_receipt_schema_unproven', **_work_kwargs(work_budget)))
            continue
        if value.get('receipt_digest_canonicalization') != 'rfc8785':
            observations.append(c.observation(row, reason='launch_receipt_canonicalization_unproven', **_work_kwargs(work_budget)))
            continue
        c.seal(row, 'receipt_digest', cross=True, **_work_kwargs(work_budget))
        c.require(all(c.matches(value.get(k), c.LAUNCH_ID, **_work_kwargs(work_budget)) for k in (_work_items(('launch_id', 'run_id'), work_budget) if work_budget is not None else ('launch_id', 'run_id')))
                  and c.matches(value.get('source_commit'), c.COMMIT, **_work_kwargs(work_budget)), 'launch_receipt_invalid', **_work_kwargs(work_budget))
        c.require(proof['path'] == c.child(context.roots['launch_execution_root'], value['launch_id'],
                                        'launch_receipt.json', **_work_kwargs(work_budget)), 'launch_receipt_identity_invalid', **_work_kwargs(work_budget))
        for field in (_work_items(('request_digest', 'launch_profile_digest'), work_budget) if work_budget is not None else ('request_digest', 'launch_profile_digest')):
            c.require(c.matches(value.get(field), **_work_kwargs(work_budget)), 'launch_receipt_invalid', **_work_kwargs(work_budget))
        selected = request_index.get(value['request_digest'], [])
        for request, _ in (_work_items(selected, work_budget) if work_budget is not None else selected):
            c.require(all(value.get(k) == request[k] for k in (_work_items(('launch_id', 'run_id', 'source_commit', 'launch_profile_digest'), work_budget) if work_budget is not None else ('launch_id', 'run_id', 'source_commit', 'launch_profile_digest')))
                      and proof['path'] == c.child(context.roots['launch_execution_root'], request['launch_id'], 'launch_receipt.json', **_work_kwargs(work_budget)), 'launch_receipt_identity_invalid', **_work_kwargs(work_budget))
        if not selected:
            context.missing('launch_request', 'launch_request_unavailable', [proof], selector={'request_digest': value['request_digest']})
    activation_digest_index = {}
    for row in (_work_items(context.decoded['activation_results'], work_budget) if work_budget is not None else context.decoded['activation_results']):
        activation_digest_index.setdefault(row[0].get('result_digest'), []).append(row)
    for row in (_work_items(context.decoded['launch_progressions'], work_budget) if work_budget is not None else context.decoded['launch_progressions']):
        value, proof = row
        c.require(value.get('schema_version') == 'task_evaluation_scene_configuration_activation_progression.v1'
                  and value.get('status') == 'scene_configuration_launch_queued'
                  and c.matches(value.get('preparation_id'), c.ID, **_work_kwargs(work_budget)), 'launch_progression_invalid', **_work_kwargs(work_budget))
        c.seal(row, 'progression_digest', **_work_kwargs(work_budget))
        c.require(proof['path'] == c.child(context.roots['configuration_progression_root'], 'scene-configuration-activations',
                  value['preparation_id'], 'launch_progression.json', **_work_kwargs(work_budget)) and value.get('paid_execution_requested') is True
                  and value.get('provider_mutation_performed_inside_progression') is False
                  and value.get('submitted_through_webapp') is True, 'launch_progression_invalid', **_work_kwargs(work_budget))
        for field in (_work_items(('profile_digest', 'activation_result_digest', 'standing_authorization_digest'), work_budget) if work_budget is not None else ('profile_digest', 'activation_result_digest', 'standing_authorization_digest')):
            c.require(c.matches(value.get(field), **_work_kwargs(work_budget)), 'launch_progression_invalid', **_work_kwargs(work_budget))
        c.require(c.matches(value.get('activation_id'), c.ACTIVATION_ID, **_work_kwargs(work_budget)) and c.matches(value.get('expected_production_commit'), c.COMMIT, **_work_kwargs(work_budget))
                  and all(c.matches(value.get(k), c.LAUNCH_ID, **_work_kwargs(work_budget)) for k in (_work_items(('launch_id', 'run_id', 'profile_id'), work_budget) if work_budget is not None else ('launch_id', 'run_id', 'profile_id'))), 'launch_progression_invalid', **_work_kwargs(work_budget))
        candidates = activation_digest_index.get(value['activation_result_digest'], [])
        for activation, _ in (_work_items(candidates, work_budget) if work_budget is not None else candidates):
            c.require(all(activation.get(a) == value[b] for a, b in (_work_items((('activation_id', 'activation_id'), ('preparation_id', 'preparation_id'),
                      ('source_commit', 'expected_production_commit'), ('profile_id', 'profile_id'), ('profile_digest', 'profile_digest'),
                      ('standing_authorization_digest', 'standing_authorization_digest')), work_budget) if work_budget is not None else (('activation_id', 'activation_id'), ('preparation_id', 'preparation_id'),
                      ('source_commit', 'expected_production_commit'), ('profile_id', 'profile_id'), ('profile_digest', 'profile_digest'),
                      ('standing_authorization_digest', 'standing_authorization_digest')))), 'launch_progression_identity_invalid', **_work_kwargs(work_budget))
        for request, _ in (_work_items(request_launch.get(value['launch_id'], []), work_budget) if work_budget is not None else request_launch.get(value['launch_id'], [])):
            c.require(request['run_id'] == value['run_id'] and request['launch_profile_digest'] == value['profile_digest']
                      and request['launch_profile_id'] == value['profile_id'] and request['source_commit'] == value['expected_production_commit'], 'launch_progression_identity_invalid', **_work_kwargs(work_budget))
        if len(activations.get(value['activation_result_digest'], [])) != 1:
            context.missing('activation_result', 'launch_activation_result_unavailable_or_ambiguous', [proof], selector={'result_digest': value['activation_result_digest']})
    return observations, bound
