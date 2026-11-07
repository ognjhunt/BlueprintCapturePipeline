"""Bounded family source adoption rotates across durable retries and restarts.

Intake, registry discovery, plan validation and queue persistence are real.
Scientific source/current-website reads and capability facts are hermetic doubles;
these scheduling regressions establish no native runtime qualification.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import beta_policy_screening_family_sources as sources
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_attempt_evidence import record
from blueprint_pipeline.team_candidate_quick10_registry import REGISTRY_ENV, REGISTRY_SCHEMA
from tests.test_website_navigation_scene_inputs import navigation_scene_inputs_arguments
from tests.test_website_task_family_packet import family_preparation


def source_world(tmp_path, monkeypatch, count):
    from blueprint_pipeline import beta_policy_screening_capabilities as capabilities
    from blueprint_pipeline import beta_policy_screening_worker as worker
    from blueprint_pipeline import task_evaluation_scene_owner_authority as authority
    from blueprint_pipeline import website_task_family_scene_inputs as scene_inputs
    from blueprint_pipeline.task_evaluation_scene_intake import stage_scene_intent
    from tests.test_beta_policy_screening_plan import inputs, scene

    fixture = tmp_path / 'scientific-fixture'
    fixture.mkdir()
    args, preparation, _ = family_preparation(fixture)
    intents, registry_path = tmp_path / 'intents', tmp_path / 'registry.json'
    derived = tmp_path / 'derived'
    derived.mkdir()
    registry = {'schema_version': REGISTRY_SCHEMA, 'profiles': [], 'derived_profiles': {
        'root': str(derived), 'defaults': {'allowed_founder_approvers': ['owner-1'],
            'max_hourly_rate_usd': '1.00', 'open_loop_horizon': 1}}}
    monkeypatch.setenv(REGISTRY_ENV, str(registry_path))
    monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT', str(intents))
    world = SimpleNamespace(queue=tmp_path / 'screening', intents=intents, registry=registry,
        registry_path=registry_path, derived=derived, captures={}, inputs={}, intent_paths={},
        names={}, planning_calls=[], intent_checks=[], current_checks=[], scientific_checks=[],
        current_blocked=set(), scientific_blocked=set())

    def persist_registry():
        registry['registry_digest'] = canonical_digest(registry, digest_field='registry_digest')
        write_json(registry_path, registry)
        registry_path.chmod(0o600)

    def add(name):
        capture = tmp_path / 'captures' / ('capture-' + name)
        request = copy.deepcopy(preparation['intake_request'])
        request['submission_id'] = 'capture-' + name
        request['task']['task_id'] = 'website-source-' + name
        accepted = stage_scene_intent(value=request, queue_root=intents,
            authenticated_client='blueprint-webapp', trusted_clients={'blueprint-webapp'}, now=args['now'])
        intent_path = intents / accepted['intent_id'] / 'intent.json'
        source = {'scene_intent_id': accepted['intent_id'], 'scene_intent_digest': accepted['intent_digest'],
            'capture_id': request['submission_id'], 'task_id': request['task']['task_id'],
            'task_context_digest': canonical_digest({'context': name}),
            'scene_revision_digest': canonical_digest({'revision': name})}
        value = {'source': source, 'task_success_contract_digest': request['task']['task_success_contract_digest'],
            'references': {'scene_intent': record(intent_path)}}
        value['inputs_digest'] = canonical_digest(value, digest_field='inputs_digest')
        path = tmp_path / 'scene-inputs' / (name + '.json')
        write_json(path, value)
        profile = {'capture_root': str(capture), 'scene_id': 'site-family-' + name,
                   'family_scene_inputs_path': str(path)}
        registry['profiles'].append(profile)
        world.captures[name], world.inputs[name], world.intent_paths[name] = capture, path, intent_path
        world.names['family-' + canonical_digest({'capture_root': str(capture)})[7:39]] = name
        return profile

    for index in range(count):
        add(f'{index:03}')
    persist_registry()
    world.add, world.persist_registry = add, persist_registry
    reopen = authority.reopen_scene_intent

    def check_intent(reference):
        name = next(name for name, path in world.intent_paths.items() if str(path) == reference['path'])
        world.intent_checks.append(name)
        return reopen(reference)

    def read_scientific(path):
        name = Path(path).stem
        world.scientific_checks.append(name)
        if name in world.scientific_blocked:
            raise ValueError('scene_progression_family_scene_runtime_source_required')
        return json.loads(Path(path).read_text())

    def current(source, *, site_id):
        name = source['capture_id'].removeprefix('capture-')
        world.current_checks.append(name)
        if name in world.current_blocked:
            raise ValueError('beta_screening_current_capture_consent_required')
        return {'scene_id': site_id}

    def planning(**kwargs):
        name = kwargs['source']['capture_id'].removeprefix('capture-')
        world.planning_calls.append(name)
        catalog, admitted, quotes = inputs()
        assert kwargs['catalog'] == catalog
        facts = {**scene(), **kwargs['source'], 'site_id': kwargs['profile']['scene_id'],
                 'scene_id': kwargs['profile']['scene_id']}
        for quote in quotes.values():
            quote['scene_revision_digest'] = facts['scene_revision_digest']
            quote['valid_until'] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            quote['quote_digest'] = canonical_digest(quote, digest_field='quote_digest')
        return {'scene': facts, 'capabilities': admitted, 'cost_quotes': quotes}

    monkeypatch.setattr(authority, 'reopen_scene_intent', check_intent)
    monkeypatch.setattr(scene_inputs, 'read_family_scene_inputs', read_scientific)
    monkeypatch.setattr(worker, 'require_current_source', current)
    monkeypatch.setattr(capabilities, 'build_planning_inputs', planning)
    from blueprint_pipeline import beta_policy_screening_job_ceiling as job_ceiling
    from tests.test_beta_policy_screening_job_ceiling import ceiling_summary
    monkeypatch.setattr(job_ceiling, 'read_job_ceiling_summary', lambda **_: ceiling_summary(40))
    # The deployed commit is the one the planner fixture binds: a replay never re-plans.
    monkeypatch.setattr(worker, 'implementation_commit', lambda: 'a' * 40)
    return world


def _names(world, result):
    return [world.names[row['request_key']] for row in result['sources']]


def test_more_than_default_limit_rotates_after_restart_and_retries_authority_holds(tmp_path, monkeypatch):
    from blueprint_pipeline import team_candidate_quick10_registry as registry

    world = source_world(tmp_path, monkeypatch, 65)
    resolve = registry.profile_for_capture
    resolved = []

    def profile(capture):
        name = Path(capture).name.removeprefix('capture-')
        resolved.append(name)
        if name == '000':
            raise ValueError('team_candidate_profile_ambiguous')
        return resolve(capture)

    monkeypatch.setattr(registry, 'profile_for_capture', profile)
    write_json(world.intent_paths['001'].parent / 'revoked.json', {'revoked': True})
    world.current_blocked.add('002')
    world.scientific_blocked.add('003')
    # Duplicate operator/derived discovery and malformed retained rows must not
    # duplicate attempts or prevent the page from rotating.
    write_json(world.derived / 'duplicate.json', {'capture_root': str(world.captures['004']),
        'family_scene_inputs_path': str(world.inputs['004'])})
    write_json(world.derived / 'malformed.json', {'capture_root': {'invalid': True},
        'family_scene_inputs_path': '/unused'})
    first = sources.enqueue_registered_family_scenes(queue_root=world.queue)
    assert _names(world, first) == [f'{i:03}' for i in range(32)]
    assert len(resolved) == 32 and len(first['sources']) == 32
    assert all(row['status'] == 'blocked' for row in first['sources'][:4])
    assert all(row['status'] == 'enqueued' for row in first['sources'][4:])
    cursor = world.queue / 'family_source_cursor.json'
    assert json.loads(cursor.read_text()) == {'last': str(world.captures['031'])}
    importlib.reload(sources)
    second = sources.enqueue_registered_family_scenes(queue_root=world.queue)
    assert _names(world, second) == [f'{i:03}' for i in range(32, 64)]
    assert all(row['status'] == 'enqueued' for row in second['sources'])
    assert len(resolved) == 64 and len(world.planning_calls) == 60
    # The resumed third page reaches the 65th source before wrapping to retry
    # blocked and already-enqueued profiles. No process-memory cursor is used.
    world.current_blocked.clear()
    importlib.reload(sources)
    third = sources.enqueue_registered_family_scenes(queue_root=world.queue)
    assert _names(world, third) == ['064'] + [f'{i:03}' for i in range(31)]
    assert third['sources'][0]['status'] == 'enqueued'
    assert third['sources'][1]['blockers'] == ['team_candidate_profile_ambiguous']
    assert third['sources'][2]['blockers'] == ['scene_configuration_submission_scene_owner_authority_revoked']
    assert third['sources'][3]['status'] == 'enqueued'
    assert all(row['status'] == 'already_enqueued' for row in third['sources'][5:])
    assert '001' not in world.scientific_checks and '001' not in world.current_checks
    assert not {'000', '001', '003'} & set(world.planning_calls)
    assert world.current_checks.count('004') == 2  # Existing plans still reopen current consent.
    assert len(world.planning_calls) == len(set(world.planning_calls)) == 62
    assert len(list((world.queue / 'plans').glob('*.json'))) == 62
    assert not list((world.queue / 'attempts').iterdir())
    assert len(resolved) == 96


def test_cursor_survives_deleted_profile_and_visits_new_profiles_on_both_sides(tmp_path, monkeypatch):
    world = source_world(tmp_path, monkeypatch, 5)
    first = sources.enqueue_registered_family_scenes(queue_root=world.queue, max_sources=2)
    assert _names(world, first) == ['000', '001']
    world.registry['profiles'] = [profile for profile in world.registry['profiles']
        if profile['capture_root'] != str(world.captures['001'])]
    world.add('000a')
    world.add('001a')
    world.persist_registry()
    importlib.reload(sources)
    second = sources.enqueue_registered_family_scenes(queue_root=world.queue, max_sources=2)
    third = sources.enqueue_registered_family_scenes(queue_root=world.queue, max_sources=2)
    fourth = sources.enqueue_registered_family_scenes(queue_root=world.queue, max_sources=2)
    assert _names(world, second) == ['001a', '002']
    assert _names(world, third) == ['003', '004']
    assert _names(world, fourth) == ['000', '000a']
    assert fourth['sources'][0]['status'] == 'already_enqueued'
    assert fourth['sources'][1]['status'] == 'enqueued'
    assert len(list((world.queue / 'plans').glob('*.json'))) == 7


@pytest.mark.parametrize('limit', [0, 101, True, 1.5])
def test_family_source_limit_rejects_unbounded_or_noninteger_batches(tmp_path, limit):
    queue = tmp_path / 'screening'
    with pytest.raises(ValueError, match='beta_screening_family_source_limit_invalid'):
        sources.enqueue_registered_family_scenes(queue_root=queue, max_sources=limit)
    assert not queue.exists()


def test_malformed_cursor_fails_before_any_source_admission(tmp_path, monkeypatch):
    world = source_world(tmp_path, monkeypatch, 1)
    world.queue.mkdir()
    write_json(world.queue / 'family_source_cursor.json', {'last': []})
    with pytest.raises(ValueError, match='beta_screening_family_source_cursor_invalid'):
        sources.enqueue_registered_family_scenes(queue_root=world.queue)
    assert not world.intent_checks and not world.planning_calls


def _navigation_source(world, tmp_path, monkeypatch, *, derived):
    """A navigation job's real scene-inputs record, registered as an operator row or a derived row."""
    from blueprint_pipeline.team_candidate_quick10_registry import write_derived_profile
    from blueprint_pipeline.website_navigation_scene_inputs import write_navigation_scene_inputs

    written = write_navigation_scene_inputs(**navigation_scene_inputs_arguments(tmp_path, monkeypatch))
    record = written['record']
    capture = tmp_path / 'captures' / 'capture-navigation'
    row = {'capture_root': str(capture), 'scene_id': record['scene_id'], 'task_id': record['task_id'],
           'scenario_id': 'nominal', 'robot_free_probe_root': str(Path(record['robot_free_probe']['path']).parent),
           'scene_revision_digest': record['scene_revision_digest'], 'navigation_scene_inputs_path': written['path']}
    if derived:
        capture.mkdir(parents=True)
        assert write_derived_profile(row)['status'] == 'written'
    else:
        world.registry['profiles'].append(row)
        world.persist_registry()
    world.names['family-' + canonical_digest({'capture_root': str(capture)})[7:39]] = 'navigation'
    return written


@pytest.mark.parametrize('derived', [False, True])
def test_a_navigation_source_is_picked_up_and_holds_by_name_until_its_intake_receipt_exists(
        tmp_path, monkeypatch, derived):
    from blueprint_pipeline.website_navigation_scene_inputs import SOURCE_UNBOUND

    from blueprint_pipeline import website_navigation_scene_inputs as navigation_inputs

    world = source_world(tmp_path, monkeypatch, 2)
    _navigation_source(world, tmp_path, monkeypatch, derived=derived)
    # Refused by name before the record and its packet are read (and hashed) again on every pass:
    # no intake outbox, so nothing to bind and no receipt.
    monkeypatch.setattr(navigation_inputs, 'read_navigation_scene_inputs',
                        lambda _path: pytest.fail('an unbindable source is never read'))
    result = sources.enqueue_registered_family_scenes(queue_root=world.queue)
    assert _names(world, result) == ['000', '001', 'navigation']
    navigation = result['sources'][2]
    assert navigation['status'] == 'blocked' and navigation['blockers'] == [SOURCE_UNBOUND]
    assert [row['status'] for row in result['sources'][:2]] == ['enqueued', 'enqueued']
    # Nothing about the navigation source reached consent, an intent or the planner.
    assert world.current_checks == world.planning_calls == ['000', '001']
    assert sorted(world.intent_checks) == ['000', '001']
    assert json.loads((world.queue / 'family_source_cursor.json').read_text()) == {
        'last': str(tmp_path / 'captures' / 'capture-navigation')}


def _bound_navigation_source(world, tmp_path, monkeypatch):
    """A navigation job's real record whose website intake is staged, its outbox beside the record."""
    from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
    from blueprint_pipeline.task_evaluation_scene_intake import stage_scene_intent
    from blueprint_pipeline.website_navigation_intake import OUTBOX_FILENAME, OUTBOX_SCHEMA, navigation_task
    from blueprint_pipeline.website_navigation_scene_inputs import write_navigation_scene_inputs

    arguments = navigation_scene_inputs_arguments(tmp_path, monkeypatch)
    written = write_navigation_scene_inputs(**arguments)
    record, capture = written['record'], arguments['capture_root']
    row = {'capture_root': str(capture), 'scene_id': record['scene_id'], 'task_id': record['task_id'],
           'scenario_id': 'nominal', 'robot_free_probe_root': str(Path(record['robot_free_probe']['path']).parent),
           'scene_revision_digest': record['scene_revision_digest'], 'navigation_scene_inputs_path': written['path']}
    world.registry['profiles'].append(row)
    world.persist_registry()
    world.names['family-' + canonical_digest({'capture_root': str(capture)})[7:39]] = 'navigation'
    now = datetime.now(timezone.utc).timestamp()
    request = {'schema_version': 'task_evaluation_scene_intake_request.v1', 'submission_id': capture.name,
               'owner': {'user_id': 'blueprint', 'organization_id': 'blueprint'},
               'source': {'kind': 'gaussian_splat', 'binding_id': 'website-splat-' + 'a' * 32,
                          'content_digest': 'sha256:' + 'a' * 64},
               'task': navigation_task(record),
               'execution': {'purpose': 'scene_preparation', 'max_total_spend_usd': 20, 'max_paid_attempts': 2,
                             'max_retries': 0, 'expires_at_epoch': now + 3600,
                             'allowed_providers': ['vast', 'openai'], 'policy_candidates': [],
                             'claim_scope': 'development_only'},
               'consent': {'accepted_by': 'blueprint', 'accepted_at_epoch': now - 60,
                           'rights_reference': 'sha256:' + 'b' * 64, 'provider_terms_reference': 'sha256:' + 'c' * 64,
                           'private_processing_authorized': True, 'provider_training_authorized': False,
                           'task_confirmed': True, 'spend_authorized': True}}
    staged = stage_scene_intent(value=request, queue_root=world.intents, authenticated_client='blueprint-webapp',
                                trusted_clients={'blueprint-webapp'}, now=now)
    outbox = {'schema_version': OUTBOX_SCHEMA, 'claim_ceiling': 'development_only',
              'scene_inputs_digest': record['scene_inputs_digest'],
              'scene_revision_digest': record['scene_revision_digest'], 'request': request,
              'request_digest': cross_runtime_canonical_digest(request),
              'website_intake': {'id': staged['intent_id'], 'request_digest': cross_runtime_canonical_digest(request),
                                 'state': 'forward_pending'},
              'intent_id': staged['intent_id'], 'reserved_usd': 0, 'construction_allowance_granted': False}
    outbox['outbox_digest'] = canonical_digest(outbox, digest_field='outbox_digest')
    write_json(Path(written['path']).parent / OUTBOX_FILENAME, outbox)
    return written, capture, staged


@pytest.mark.parametrize('g1', ['registered', 'failed'])
def test_a_navigation_source_registers_its_sites_g1_rows_before_the_catalog_is_read(tmp_path, monkeypatch, g1):
    """The family pass registers a navigation site's G1 candidates itself, as the packet pass does for a team scene.

    A new row changes the catalog's revision, so registration comes before the catalog read; a G1 problem is
    recorded and never stops the site.
    """
    from blueprint_pipeline import beta_policy_screening_team as team
    from blueprint_pipeline import native_g1_site_registration as registration
    from blueprint_pipeline import team_scene_packet_worker as packets

    world = source_world(tmp_path, monkeypatch, 1)
    written, capture, staged = _bound_navigation_source(world, tmp_path, monkeypatch)
    order, calls = [], []

    def ensure(**kwargs):
        order.append('register')
        calls.append(kwargs)
        if g1 == 'failed':
            raise RuntimeError('g1 shared inputs unreadable')
        return {'status': 'registered'}

    real_catalog = team.catalog_for_profile
    from blueprint_pipeline import team_candidate_quick10_registry as quick10
    real_profile = quick10.profile_for_capture

    def profile_for_capture(capture_root):
        # The registry reader merges the site's new G1 rows once registration has written them.
        profile = real_profile(capture_root)
        return {**profile, 'registered_after': True} if profile and 'register' in order else profile

    def catalog(profile, base):
        order.append('catalog:' + ('navigation' if profile.get('navigation_scene_inputs_path') else 'family'))
        if profile.get('navigation_scene_inputs_path'):
            seen.append(profile.get('registered_after', False))
        return real_catalog(profile, base)

    seen = []
    monkeypatch.setattr(quick10, 'profile_for_capture', profile_for_capture)
    monkeypatch.setattr(registration, 'ensure_site_registration', ensure)
    monkeypatch.setattr(packets, 'current_release_commit', lambda: 'a' * 40)
    monkeypatch.setattr(team, 'catalog_for_profile', catalog)
    packet_root = tmp_path / 'team-scene-packets'
    result = sources.enqueue_registered_family_scenes(queue_root=world.queue, packet_queue_root=packet_root)
    navigation = result['sources'][_names(world, result).index('navigation')]
    assert navigation['status'] == 'enqueued'
    record = json.loads(Path(written['path']).read_text())
    assert calls == [{'site_id': record['scene_id'], 'source': {
        'scene_intent_id': staged['intent_id'], 'scene_intent_digest': staged['intent_digest'],
        'capture_id': capture.name, 'task_id': record['task_id'],
        'task_context_digest': record['source_gap']['known']['task_context_digest'],
        'scene_revision_digest': record['scene_revision_digest']},
        'source_queue': packet_root, 'release': 'a' * 40}]
    assert order.index('register') < order.index('catalog:navigation')
    # The catalog and plan read the profile as registration left it, not the one read before it.
    assert seen == [True]
    assert navigation['g1_registration'] == ({'status': 'registered'} if g1 == 'registered' else {
        'status': 'failed', 'error': navigation['g1_registration']['error']})


def test_a_navigation_source_binds_its_staged_intake_and_goes_on_through_current_consent_and_planning(
        tmp_path, monkeypatch):
    """Once the job's navigation intake is staged, its receipt binds the source and the family path goes on."""
    from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
    from blueprint_pipeline.task_evaluation_scene_intake import stage_scene_intent
    from blueprint_pipeline.website_navigation_intake import (
        OUTBOX_FILENAME, OUTBOX_SCHEMA, RECEIPT_FILENAME, navigation_task, read_navigation_intake_receipt,
    )
    from blueprint_pipeline.website_navigation_scene_inputs import screening_source, write_navigation_scene_inputs

    world = source_world(tmp_path, monkeypatch, 1)
    arguments = navigation_scene_inputs_arguments(tmp_path, monkeypatch)
    written = write_navigation_scene_inputs(**arguments)
    record, capture = written['record'], arguments['capture_root']
    row = {'capture_root': str(capture), 'scene_id': record['scene_id'], 'task_id': record['task_id'],
           'scenario_id': 'nominal', 'robot_free_probe_root': str(Path(record['robot_free_probe']['path']).parent),
           'scene_revision_digest': record['scene_revision_digest'], 'navigation_scene_inputs_path': written['path']}
    world.registry['profiles'].append(row)
    world.persist_registry()
    world.names['family-' + canonical_digest({'capture_root': str(capture)})[7:39]] = 'navigation'
    # The intake the handoff sent (website_navigation_intake.enqueue_navigation_intake), as its outbox keeps it.
    now = datetime.now(timezone.utc).timestamp()
    request = {'schema_version': 'task_evaluation_scene_intake_request.v1', 'submission_id': capture.name,
               'owner': {'user_id': 'blueprint', 'organization_id': 'blueprint'},
               'source': {'kind': 'gaussian_splat', 'binding_id': 'website-splat-' + 'a' * 32,
                          'content_digest': 'sha256:' + 'a' * 64},
               'task': navigation_task(record),
               'execution': {'purpose': 'scene_preparation', 'max_total_spend_usd': 20, 'max_paid_attempts': 2,
                             'max_retries': 0, 'expires_at_epoch': now + 3600,
                             'allowed_providers': ['vast', 'openai'], 'policy_candidates': [],
                             'claim_scope': 'development_only'},
               'consent': {'accepted_by': 'blueprint', 'accepted_at_epoch': now - 60,
                           'rights_reference': 'sha256:' + 'b' * 64, 'provider_terms_reference': 'sha256:' + 'c' * 64,
                           'private_processing_authorized': True, 'provider_training_authorized': False,
                           'task_confirmed': True, 'spend_authorized': True}}
    staged = stage_scene_intent(value=request, queue_root=world.intents, authenticated_client='blueprint-webapp',
                                trusted_clients={'blueprint-webapp'}, now=now)
    outbox = {'schema_version': OUTBOX_SCHEMA, 'claim_ceiling': 'development_only',
              'scene_inputs_digest': record['scene_inputs_digest'],
              'scene_revision_digest': record['scene_revision_digest'], 'request': request,
              'request_digest': cross_runtime_canonical_digest(request),
              'website_intake': {'id': staged['intent_id'], 'request_digest': cross_runtime_canonical_digest(request),
                                 'state': 'forward_pending'},
              'intent_id': staged['intent_id'], 'reserved_usd': 0, 'construction_allowance_granted': False}
    outbox['outbox_digest'] = canonical_digest(outbox, digest_field='outbox_digest')
    write_json(Path(written['path']).parent / OUTBOX_FILENAME, outbox)
    result = sources.enqueue_registered_family_scenes(queue_root=world.queue)
    assert _names(world, result) == ['000', 'navigation']
    assert [row['status'] for row in result['sources']] == ['enqueued', 'enqueued']
    # The pass bound the receipt beside the record; its source names the staged intent.
    receipt = read_navigation_intake_receipt(written['path'])
    assert (Path(written['path']).parent / RECEIPT_FILENAME).is_file()
    assert receipt['source'] == {'scene_intent_id': staged['intent_id'], 'scene_intent_digest': staged['intent_digest'],
                                 'capture_id': capture.name, 'task_id': record['task_id'],
                                 'task_context_digest': record['source_gap']['known']['task_context_digest'],
                                 'scene_revision_digest': record['scene_revision_digest']}
    assert screening_source(record) == receipt['source']
    # The world's doubles name a source by its capture: current consent and planning saw the navigation one.
    assert world.current_checks == world.planning_calls == ['000', capture.name]
    assert world.intent_checks == ['000']  # the navigation source is bound by its receipt, not a family intent
