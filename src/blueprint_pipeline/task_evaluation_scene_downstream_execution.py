"""Pure activation and historical owner-attempt launch joins; no runtime imports."""
from __future__ import annotations

from pathlib import PurePosixPath

from . import task_evaluation_scene_downstream_contracts as c
from .task_evaluation_scene_attempt_binding import scene_execution_binding_blockers


def activation(context, seed):
    rows, matched, envelopes = context.rows(), {}, {}
    for row in context.decoded['activation_envelopes']:
        envelopes.setdefault(PurePosixPath(row[1]['path']).name, []).append(row)
    seed_paths = {m['path'] for m in seed['members'] if m['kind'] == 'configuration_progression_workspace'}
    for row in context.decoded['activation_results']:
        value, proof = row
        c.require(c.under(proof['path'], c.child(context.roots['activation_queue_root'], 'results'))
                  and PurePosixPath(proof['path']).parent == PurePosixPath(context.roots['activation_queue_root']) / 'results', 'activation_path_invalid')
        if value.get('schema_version') != 'task_evaluation_launch_activation_result.v1':
            rows.append(c.observation(row, reason='activation_schema_unproven'))
            continue
        c.seal(row, 'result_digest')
        c.require(c.matches(value.get('activation_id'), c.ACTIVATION_ID) and isinstance(value.get('status'), str), 'activation_invalid')
        candidates = envelopes.get(PurePosixPath(proof['path']).name, [])
        envelope = candidates[0] if len(candidates) == 1 else None
        status = value['status']
        known = status in {'profile_authority_materialized_no_execution', 'policy_campaign_queue_materialized_no_execution'}
        if known or status == 'blocked':
            c.require(value.get('provider_mutation_performed') is False and value.get('paid_execution_requested') is False
                      and isinstance(value.get('blockers'), list), 'activation_scope_invalid')
        sources = context.provenance((proof,))
        if known:
            for field in ('preparation_id', 'team_namespace', 'lane'):
                c.require(c.matches(value.get(field), c.ID), 'activation_invalid')
            c.require(c.matches(value.get('source_commit'), c.COMMIT), 'activation_invalid')
            for field in ('preparation_result_digest', 'release_window_digest'):
                c.require(c.matches(value.get(field)), 'activation_invalid')
            if status == 'profile_authority_materialized_no_execution':
                c.require(c.matches(value.get('profile_id'), c.LAUNCH_ID), 'activation_invalid')
                for field in ('profile_digest', 'profile_publication_receipt_digest', 'standing_authorization_digest'):
                    c.require(c.matches(value.get(field)), 'activation_invalid')
                c.require(all(value.get(k) is True for k in ('full_byte_activation_reference_readback_passed',
                          'profile_publication_performed', 'catalog_mutation_performed', 'standing_authorization_published')), 'activation_scope_invalid')
                for role, field in (('profile_publication_receipt', 'profile_publication_receipt_digest'),
                                    ('standing_authorization', 'standing_authorization_digest')):
                    context.missing(role, 'activation_selector_bytes_unavailable', [proof],
                                    selector={'sha256': value[field], 'profile_id': value['profile_id']})
            if envelope:
                request = envelope[0]['request']
                c.require(value['activation_id'] == request['activation_id'] and value['preparation_id'] == request['preparation']['preparation_id']
                          and value['preparation_result_digest'] == request['preparation']['result_digest']
                          and value['team_namespace'] == request['team_namespace'] and value['lane'] == request['lane']
                          and value['source_commit'] == request['expected_production_commit'], 'activation_identity_invalid')
                sources.append(envelope[1])
        elif envelope:
            c.require(value['activation_id'] == envelope[0]['request']['activation_id'], 'activation_identity_invalid')
        expected = c.child(context.roots['configuration_progression_root'], 'scene-configuration-activations', value.get('preparation_id', 'unproven'))
        bound = status == 'profile_authority_materialized_no_execution' and envelope is not None and expected in seed_paths
        reason = None if bound else 'activation_envelope_ambiguous' if len(candidates) > 1 else 'activation_output_scope_unproven'
        if len(candidates) > 1:
            sources += context.provenance(r[1] for r in candidates)
        if bound:
            context.member(c.child(context.roots['activation_output_root'], value['activation_id']), 'activation_workspace',
                           {'activation_id': value['activation_id'], 'preparation_id': value['preparation_id']}, sources)
            matched.setdefault(value['result_digest'], []).append(row)
        else:
            context.missing('activation_owner_join', reason, sources, selector={'activation_id': value['activation_id']})
        if status == 'policy_campaign_queue_materialized_no_execution':
            context.missing('campaign_outputs', 'campaign_output_identity_unproven', sources)
        rows.append(c.observation(row, status='matched_retained_bytes' if bound else 'kept_unresolved', reason=reason,
                                  activation_id=value['activation_id'], release_window_digest=value.get('release_window_digest'),
                                  window_binding_verified=False, source_provenance=sources))
    result_names = {PurePosixPath(p['path']).name for _, p in context.decoded['activation_results']}
    for filename, candidates in envelopes.items():
        if filename not in result_names:
            context.missing('activation_result', 'activation_result_unavailable', context.provenance(r[1] for r in candidates),
                            c.child(context.roots['activation_queue_root'], 'results', filename))
    return rows, matched


def owner(context, row):
    profile, proof = row
    intent = context.decoded['intent'][0][0]
    direct = profile.get('scene_attempt_binding')
    plan = profile.get('internal_policy_canary_execution_plan')
    policy = plan.get('scene_policy_binding') if isinstance(plan, dict) else None
    if direct is not None or 'scene_attempt_id' in profile or ('scene_intent_digest' in profile and policy is None):
        c.require(not scene_execution_binding_blockers(profile), 'owner_binding_invalid')
    if policy is not None:
        c.require(isinstance(policy, dict) and set(policy) == {'schema_version', 'scene_intent_digest', 'attempt_id',
                  'policy_candidates', 'runtime_digest', 'input_digest', 'binding_digest'}
                  and policy.get('schema_version') == 'task_evaluation_scene_policy_binding.v1', 'owner_binding_invalid')
        c.seal((policy, {}), 'binding_digest')
        for k in ('scene_intent_digest', 'runtime_digest', 'input_digest'):
            c.require(c.matches(policy[k]), 'owner_binding_invalid')
        c.require(c.matches(policy['attempt_id'], c.ID), 'owner_binding_invalid')
        if 'scene_intent_digest' in profile:
            c.require(profile['scene_intent_digest'] == policy['scene_intent_digest'], 'owner_binding_invalid')
        candidates = policy['policy_candidates']
        c.require(isinstance(candidates, list) and len(candidates) == 2 and all(isinstance(r, dict) and set(r) == {'id', 'artifact_digest'}
                  and c.matches(r['id'], c.ID) and c.matches(r['artifact_digest']) for r in candidates)
                  and candidates[0]['id'] != candidates[1]['id'], 'owner_binding_invalid')
        if direct is not None:
            c.require(all(direct[a] == policy[b] for a, b in (('intent_digest', 'scene_intent_digest'), ('attempt_id', 'attempt_id'),
                      ('runtime_digest', 'runtime_digest'), ('input_digest', 'input_digest'))), 'owner_binding_invalid')
    binding = direct or policy
    if not binding:
        return None, 'launch_owner_unproven', [proof]
    digest = binding.get('intent_digest', binding.get('scene_intent_digest'))
    if digest != intent['intent_digest']:
        return None, 'foreign_owner_unproven', [proof]
    if direct is not None:
        c.require(direct['intent_id'] == context.intent_id, 'owner_binding_invalid')
    if policy is not None:
        expected = intent['request'].get('execution', {}).get('policy_candidates')
        c.require(isinstance(expected, list) and sorted(candidates, key=lambda r: r['id']) == sorted(expected, key=lambda r: r['id']), 'owner_pair_invalid')
    attempt_id = binding['attempt_id']
    expected_path = c.child(context.roots['intent_root'], context.intent_id, 'attempts', attempt_id + '.json')
    matches = context.by_path['attempts'].get(expected_path, [])
    if not matches:
        context.missing('owner_attempt', 'owner_attempt_bytes_unavailable', [proof], expected_path)
        return None, 'owner_attempt_bytes_unavailable', [proof]
    c.require(len(matches) == 1, 'owner_attempt_ambiguous')
    attempt, provenance = matches[0]
    c.require(attempt.get('schema_version') == 'task_evaluation_scene_attempt.v1', 'owner_attempt_invalid')
    c.seal(matches[0], 'attempt_digest', cross=True)
    c.require(all(attempt.get(k) == binding[k] for k in ('attempt_id', 'runtime_digest', 'input_digest'))
              and attempt.get('intent_digest') == digest and attempt.get('intent_id') == context.intent_id
              and attempt.get('source_commit') == profile['source_commit'], 'owner_attempt_invalid')
    return attempt_id, None, [proof, provenance]


def launches(context, activations):
    profiles, available_profiles, requests, request_launch, observations, bound = {}, {}, {}, {}, context.rows(), {}
    for row in context.decoded['launch_profiles']:
        profile, proof = row
        p = PurePosixPath(proof['path'])
        c.require(p.name == 'launch_profile.json' and (c.under(proof['path'], context.roots['launch_execution_root'])
                  or c.under(proof['path'], context.roots['terminal_result_root'])), 'launch_path_invalid')
        if profile.get('schema_version') != 'task_evaluation_launch_profile.v1':
            observations.append(c.observation(row, reason='launch_profile_schema_unproven'))
            continue
        c.require(c.matches(profile.get('source_commit'), c.COMMIT) and c.matches(profile.get('profile_id'), c.LAUNCH_ID), 'profile_invalid')
        c.seal(row, 'profile_digest')
        identity = owner(context, row)  # Validate supplied owner even if no request exists.
        profiles.setdefault((str(p.parent), profile['profile_digest']), []).append((row, identity))
        available_profiles.setdefault(str(p.parent), []).append(row)
    for row in context.decoded['launch_requests']:
        request, proof = row
        if request.get('schema_version') != 'task_evaluation_launch_request.v1':
            c.require(PurePosixPath(proof['path']).name == 'launch_request.json'
                      and (c.under(proof['path'], context.roots['launch_execution_root']) or c.under(proof['path'], context.roots['terminal_result_root'])), 'launch_path_invalid')
            observations.append(c.observation(row, reason='launch_request_schema_unproven'))
            continue
        c.require(all(c.matches(request.get(k), c.LAUNCH_ID) for k in ('launch_id', 'run_id', 'launch_profile_id'))
                  and c.matches(request.get('source_commit'), c.COMMIT) and c.matches(request.get('launch_profile_digest')), 'launch_request_invalid')
        c.seal(row, 'request_digest')
        launch_path = c.child(context.roots['launch_execution_root'], request['launch_id'], 'launch_request.json')
        directory = str(PurePosixPath(proof['path']).parent)
        import hashlib
        terminal = c.child(context.roots['terminal_result_root'], context.intent_id)
        allowed = {launch_path, c.child(terminal, 'launch_request.json'),
                   c.child(terminal, 'runs', hashlib.sha256(request['run_id'].encode()).hexdigest(), 'launch_request.json')}
        c.require(proof['path'] in allowed, 'launch_path_invalid')
        candidates = profiles.get((directory, request['launch_profile_digest']), [])
        available = available_profiles.get(directory, [])
        c.require(not available or candidates, 'launch_profile_identity_invalid')
        reason, sources, attempt = 'launch_profile_unavailable', [proof], None
        if len(candidates) == 1:
            profile, identity = candidates[0]
            c.require(profile[0]['source_commit'] == request['source_commit']
                      and profile[0]['profile_id'] == request['launch_profile_id'], 'launch_profile_identity_invalid')
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
        requests.setdefault(request['request_digest'], []).append(row)
        request_launch.setdefault(request['launch_id'], []).append(row)
        observations.append(c.observation(row, status='matched_retained_bytes' if reason is None else 'kept_unresolved', reason=reason,
                                          launch_id=request['launch_id'], run_id=request['run_id'], source_provenance=sources))
    for row in context.decoded['launch_receipts']:
        value, proof = row
        c.require(PurePosixPath(proof['path']).name == 'launch_receipt.json' and c.under(proof['path'], context.roots['launch_execution_root']), 'launch_path_invalid')
        if value.get('schema_version') != 'task_evaluation_launch_receipt.v1':
            observations.append(c.observation(row, reason='launch_receipt_schema_unproven'))
            continue
        if value.get('receipt_digest_canonicalization') != 'rfc8785':
            observations.append(c.observation(row, reason='launch_receipt_canonicalization_unproven'))
            continue
        c.seal(row, 'receipt_digest', cross=True)
        c.require(all(c.matches(value.get(k), c.LAUNCH_ID) for k in ('launch_id', 'run_id'))
                  and c.matches(value.get('source_commit'), c.COMMIT), 'launch_receipt_invalid')
        c.require(proof['path'] == c.child(context.roots['launch_execution_root'], value['launch_id'],
                                        'launch_receipt.json'), 'launch_receipt_identity_invalid')
        for field in ('request_digest', 'launch_profile_digest'):
            c.require(c.matches(value.get(field)), 'launch_receipt_invalid')
        selected = requests.get(value['request_digest'], [])
        for request, _ in selected:
            c.require(all(value.get(k) == request[k] for k in ('launch_id', 'run_id', 'source_commit', 'launch_profile_digest'))
                      and proof['path'] == c.child(context.roots['launch_execution_root'], request['launch_id'], 'launch_receipt.json'), 'launch_receipt_identity_invalid')
        if not selected:
            context.missing('launch_request', 'launch_request_unavailable', [proof], selector={'request_digest': value['request_digest']})
    activation_digest_index = {}
    for row in context.decoded['activation_results']:
        activation_digest_index.setdefault(row[0].get('result_digest'), []).append(row)
    for row in context.decoded['launch_progressions']:
        value, proof = row
        c.require(value.get('schema_version') == 'task_evaluation_scene_configuration_activation_progression.v1'
                  and value.get('status') == 'scene_configuration_launch_queued'
                  and c.matches(value.get('preparation_id'), c.ID), 'launch_progression_invalid')
        c.seal(row, 'progression_digest')
        c.require(proof['path'] == c.child(context.roots['configuration_progression_root'], 'scene-configuration-activations',
                  value['preparation_id'], 'launch_progression.json') and value.get('paid_execution_requested') is True
                  and value.get('provider_mutation_performed_inside_progression') is False
                  and value.get('submitted_through_webapp') is True, 'launch_progression_invalid')
        for field in ('profile_digest', 'activation_result_digest', 'standing_authorization_digest'):
            c.require(c.matches(value.get(field)), 'launch_progression_invalid')
        c.require(c.matches(value.get('activation_id'), c.ACTIVATION_ID) and c.matches(value.get('expected_production_commit'), c.COMMIT)
                  and all(c.matches(value.get(k), c.LAUNCH_ID) for k in ('launch_id', 'run_id', 'profile_id')), 'launch_progression_invalid')
        candidates = activation_digest_index.get(value['activation_result_digest'], [])
        for activation, _ in candidates:
            c.require(all(activation.get(a) == value[b] for a, b in (('activation_id', 'activation_id'), ('preparation_id', 'preparation_id'),
                      ('source_commit', 'expected_production_commit'), ('profile_id', 'profile_id'), ('profile_digest', 'profile_digest'),
                      ('standing_authorization_digest', 'standing_authorization_digest'))), 'launch_progression_identity_invalid')
        for request, _ in request_launch.get(value['launch_id'], []):
            c.require(request['run_id'] == value['run_id'] and request['launch_profile_digest'] == value['profile_digest']
                      and request['launch_profile_id'] == value['profile_id'] and request['source_commit'] == value['expected_production_commit'], 'launch_progression_identity_invalid')
        if len(activations.get(value['activation_result_digest'], [])) != 1:
            context.missing('activation_result', 'launch_activation_result_unavailable_or_ambiguous', [proof], selector={'result_digest': value['activation_result_digest']})
    return observations, bound
