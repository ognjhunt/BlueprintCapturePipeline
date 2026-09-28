# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_source_family_inventory.py
#   src/blueprint_pipeline/task_evaluation_scene_source_family_contracts.py
#   src/blueprint_pipeline/task_evaluation_scene_source_family_website.py
"""ADP-009D/day-28: retained capture/publication bytes never grant cleanup."""
from __future__ import annotations

import copy
import importlib
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from tests.test_scene_inventory_history import fixture as seed_fixture, pair, ref, seal
from tests.test_scene_source_attempt_lineage import fixture as source_fixture
from tests.test_scene_downstream_execution import ROLES as DOWNSTREAM_ROLES

ROLES = ('website_registrations', 'website_bindings', 'website_handoffs', 'website_preparations',
         'website_runtime_inputs', 'website_task_contexts', 'submission_publications', 'sam_parent_envelopes',
         'source_progress', 'source_resume_signals', 'sam_plans', 'sam_profiles', 'sam_recipes',
         'sam_stage_configurations', 'sam_jobs', 'sam_results', 'sam_execution_receipts',
         'sam_execution_progress', 'sam_adoptions', 'sam_prefix_selections', 'sam_host_tasks',
         'sam_host_evidence', 'sam_artifact_metadata', 'opaque_evidence')


def api():
    return importlib.import_module('blueprint_pipeline.task_evaluation_scene_source_family_inventory')


def fixture(*, website=False, development=False, status='intake_ready'):
    base = seed_fixture()
    roots = dict(base['roots'], activation_output_root='/retained/activation-output',
        launch_execution_root='/retained/launches', terminal_result_root='/retained/terminal',
        policy_canary_root='/retained/canaries', compilation_queue_root='/retained/compilation-queue',
        compilation_output_root='/retained/compiled', pubsub_root='/retained/pubsub',
        website_source_binding_root='/retained/website-bindings', sam_queue_root='/retained/sam-queue',
        sam_execution_root='/retained/sam-execution', host_input_root='/retained/host-inputs')
    args = dict(intent_id=base['intent_id'], seed_records=base['records'],
        downstream_records={r: [] for r in DOWNSTREAM_ROLES}, source_records={r: [] for r in ROLES}, roots=roots,
        parent_routes=[{'queue_root': roots['preparation_queue_root'], 'input_root': roots['preparation_input_root']}],
        retained_metadata_roots=['/retained/metadata'])
    if not website:
        return args
    child = source_fixture('website')
    seed = args['seed_records']
    for old, new in (('intent_record', 'intent'), ('attempt_records', 'attempts'),
                     ('snapshot_records', 'source_snapshots'), ('factory_records', 'factories'),
                     ('submission_records', 'source_submissions')):
        seed[new] = child[old]
    intent = json.loads(seed['intent'][1])
    capture = roots['pubsub_root'] + '/bucket/scenes/scene-1/captures/capture-1'
    preparation_root = capture + '/pipeline/website_scene_preparation'
    selected_root = preparation_root + ('/development_test' if development else '')
    context = seal({'schema_version': 'website_site_task_context.v1', 'scene_id': 'scene-1',
                    'capture_id': 'capture-1'}, 'context_digest')
    original = seal({'schema_version': 'website_scene_preparation.v1', 'status': status,
                    'intake_request': intent['request']}, 'digest')
    preparation = original
    if development:
        preparation = seal(dict(original, development_test={'source_preparation_digest': original['digest'],
            'source_task_context_digest': context['context_digest']}, binding={'task_context_digest': context['context_digest']}), 'digest')
    prep_pair = pair(selected_root + '/preparation.json', preparation)
    context_pair = pair(selected_root + '/task_context.json', context)
    runtime = seal({'schema_version': 'website_scene_runtime_inputs.v1', 'status': 'native_inputs_ready',
                    'preparation_digest': preparation['digest']}, 'digest')
    runtime_pair = pair(selected_root + ('/runtime_inputs.json' if development else '/native/runtime_inputs.json'), runtime)
    registration = seal({'schema_version': 'website_scene_source_registration.v1',
        'request_digest': cross_runtime_canonical_digest(intent['request']),
        'references': {'preparation': ref(prep_pair), 'runtime_inputs': ref(runtime_pair), 'task_context': ref(context_pair)},
        'provider_mutation_performed': False, 'execution_authority_granted': False,
        'claim_ceiling': 'development_only'}, 'registration_digest')
    registration_pair = pair(roots['website_source_binding_root'] + '/' + registration['request_digest'][7:] + '.json', registration)
    binding = json.loads(seed['source_snapshots'][0][1])
    binding.update(references=registration['references'], registration=ref(registration_pair),
                   provider_mutation_performed=False, claim_ceiling='development_only')
    binding = seal(binding, 'binding_digest')
    seed['source_snapshots'][0] = pair(seed['source_snapshots'][0][0], binding)
    attempt = json.loads(seed['attempts'][0][1])
    attempt['input_digest'] = binding['binding_digest']
    attempt = seal(attempt, 'attempt_digest', cross=True)
    seed['attempts'][0] = pair(seed['attempts'][0][0], attempt)
    factory = json.loads(seed['factories'][0][1])
    factory['attempt_digest'] = attempt['attempt_digest']
    seed['factories'][0] = pair(seed['factories'][0][0], seal(factory, 'factory_digest'))
    handoff = {'schema_version': 'website_scene_handoff.v1', 'scene_id': 'scene-1', 'capture_id': 'capture-1',
        'status': status, 'preparation_path': preparation_root + '/preparation.json',
        'preparation_digest': original['digest'], 'source_registration': ref(registration_pair),
        'task_context_digest': context['context_digest'],
        'runtime_inputs': {'path': runtime_pair[0], 'digest': runtime['digest'], 'status': runtime['status']}}
    if development:
        handoff['development_test'] = dict(preparation['development_test'], preparation_path=prep_pair[0])
    records = args['source_records']
    records['website_preparations'] = [prep_pair]
    if development:
        records['website_preparations'].append(pair(preparation_root + '/preparation.json', original))
    records['website_task_contexts'] = [context_pair]
    records['website_runtime_inputs'] = [runtime_pair]
    records['website_registrations'] = [registration_pair]
    records['website_bindings'] = [pair(roots['factory_output_root'] + '/' + args['intent_id'] +
        '/website-source/' + binding['binding_digest'][7:] + '.json', binding)]
    records['website_handoffs'] = [pair(preparation_root + '/handoff.json', seal(handoff, 'digest'))]
    return args


def change(args, role, edits, field):
    path, raw = args['source_records'][role][0]
    value = json.loads(raw)
    edits(value) if callable(edits) else value.update(edits)
    args['source_records'][role][0] = pair(path, seal(value, field))


def refuses(args):
    with pytest.raises(api().SceneSourceFamilyInventoryError) as raised:
        api().join_retained_scene_source_family_inventory(**args)
    assert str(raised.value).startswith('scene_source_family_') and len(str(raised.value)) < 100


def test_empty_source_keeps_exact_downstream_result_and_no_authority():
    args = fixture()
    before = copy.deepcopy(args)
    result = api().join_retained_scene_source_family_inventory(**args)
    from blueprint_pipeline.task_evaluation_scene_downstream_inventory import join_retained_scene_downstream_inventory
    roots = {k: v for k, v in args['roots'].items() if k not in api().EXTRA_ROOTS}
    assert result['downstream_inventory'] == join_retained_scene_downstream_inventory(
        intent_id=args['intent_id'], seed_records=args['seed_records'], downstream_records=args['downstream_records'], roots=roots)
    assert result['raw_versions'] == [] and result['status'] == 'kept_unresolved'
    assert args == before and result['mutations'] == 0
    for flag in api().FALSE_FLAGS:
        assert result[flag] is False


@pytest.mark.parametrize('development', [False, True])
def test_original_handoff_and_successor_capture_join_retains_provenance(development):
    args = fixture(website=True, development=development)
    result = api().join_retained_scene_source_family_inventory(**args)
    member = next(r for r in result['lexical_members'] if r['kind'] == 'capture_dependency')
    assert member['path'].endswith('/captures/capture-1')
    assert not member['presence_checked'] and not member['exclusive_ownership_proven']
    assert result['website_observations'][0]['capture_binding_verified'] is True
    assert result['raw_versions'] and result['source_family_complete'] is False


@pytest.mark.parametrize('role,field,updates', [
    ('website_registrations', 'registration_digest', {'provider_mutation_performed': True}),
    ('website_registrations', 'registration_digest', {'execution_authority_granted': 0}),
    ('website_registrations', 'registration_digest', {'request_digest': 'sha256:' + 'f'*64}),
    ('website_handoffs', 'digest', {'scene_id': 'foreign'}),
    ('website_handoffs', 'digest', {'capture_id': 'foreign'}),
    ('website_runtime_inputs', 'digest', {'preparation_digest': 'sha256:' + 'f'*64}),
    ('website_task_contexts', 'context_digest', {'capture_id': 'foreign'}),
])
def test_available_malformed_identity_refuses_even_when_unrelated_bytes_missing(role, field, updates):
    args = fixture(website=True)
    change(args, role, updates, field)
    # Removing the handoff must not hide available registration/runtime/context contradictions.
    if role != 'website_handoffs':
        args['source_records']['website_handoffs'] = []
    refuses(args)


@pytest.mark.parametrize('status', ['awaiting_reconstruction', 'awaiting_inputs', 'needs_input'])
def test_partial_capture_stays_retained_with_no_finish_claim(status):
    args = fixture(website=True, status=status)
    args['source_records']['website_runtime_inputs'] = []
    result = api().join_retained_scene_source_family_inventory(**args)
    assert result['website_observations'][0]['status'] == 'kept_unresolved'
    assert not result['website_observations'][0]['capture_binding_verified']
    assert not result['scene_finished']


def test_opaque_empty_bytes_and_unknown_json_versions_remain_discoverable():
    args = fixture()
    args['source_records']['opaque_evidence'] = [('/retained/metadata/empty.txt', b'')]
    args['source_records']['sam_plans'] = [pair('/retained/metadata/future.json', {'schema_version': 'future.v2'})]
    result = api().join_retained_scene_source_family_inventory(**args)
    assert len(result['raw_versions']) == 2
    assert next(r for r in result['raw_versions'] if r['role'] == 'opaque_evidence')['size_bytes'] == 0


def test_combined_limits_refuse_before_child_and_hash(monkeypatch):
    module = api()
    args = fixture()
    args['source_records']['sam_plans'] = [pair('/retained/metadata/future.json', {'a': {'b': 1}})]
    monkeypatch.setattr(module, 'MAX_NODES', 1)
    monkeypatch.setattr(module, '_downstream_join', lambda **_: pytest.fail('child reached'))
    monkeypatch.setattr(module.contracts, 'raw_digest', lambda _: pytest.fail('hash reached'))
    refuses(args)


def test_input_permutations_and_mutable_versions_are_deterministic():
    args = fixture(website=True)
    old = copy.deepcopy(args)
    change(old, 'website_handoffs', {'status': 'awaiting_inputs'}, 'digest')
    args['source_records']['website_handoffs'] += old['source_records']['website_handoffs']
    first = api().join_retained_scene_source_family_inventory(**args)
    permuted = copy.deepcopy(args)
    for rows in permuted['source_records'].values():
        rows.reverse()
    assert first == api().join_retained_scene_source_family_inventory(**permuted)
    assert len([p for p in first['raw_versions'] if p['role'] == 'website_handoffs']) == 2
    assert not any(p['kind'] == 'capture_dependency' for p in first['lexical_members'])


def publication_fixture(*, host_only=False):
    args = fixture(website=True)
    seed = args['seed_records']
    request_path, raw = seed['source_submissions'][0]
    request = json.loads(raw)
    manifest_path, raw = seed['source_submissions'][1]
    manifest = json.loads(raw)
    namespace = manifest['input_namespace']
    object_row = {'relative_path': 'derived/config.json',
        'uri': 's3://blueprint/task-evaluation/production-inputs/' + namespace + '/derived/config.json',
        'digest': 'sha256:' + 'd'*64, 'size_bytes': 7, 'publication_allowed': True}
    request['runtime'] = {'mounts': [{'source': {k: object_row[k] for k in ('uri', 'digest', 'size_bytes')}}]}
    manifest.update(source='website_capture_derivatives', files=[object_row],
        raw_source_upload_allowed=False, provider_allocated=False, request_digest=canonical_digest(request))
    if host_only:
        manifest['source'] = 'owner_provided_completed_asset'
        manifest['files'].append({'relative_path': 'source/input.glb', 'uri': 'https://owner.example/input.glb',
            'digest': 'sha256:' + 'e'*64, 'size_bytes': 11, 'publication_allowed': False})
    manifest = seal(manifest, 'manifest_digest')
    seed['source_submissions'] = [pair(request_path, request), pair(manifest_path, manifest)]
    factory_path, raw = seed['factories'][0]
    factory = json.loads(raw)
    factory.update(submission_request=ref(seed['source_submissions'][0]), submission_manifest=ref(seed['source_submissions'][1]))
    seed['factories'][0] = pair(factory_path, seal(factory, 'factory_digest'))
    manifest_ref = ref(seed['source_submissions'][1])
    manifest_object = {'relative_path': 'bundle_manifest.v1.json',
        'uri': 's3://blueprint/task-evaluation/production-inputs/' + namespace + '/bundle_manifest.v1.json',
        'digest': manifest_ref['sha256'], 'size_bytes': manifest_ref['size_bytes']}
    publication = {'schema_version': 'task_evaluation_scene_configuration_submission_publication.v1',
        'status': 'published_and_read_back', 'source_commit': manifest['source_commit'],
        'input_namespace': namespace, 'manifest_sha256': manifest_ref['sha256'],
        'manifest_digest': manifest['manifest_digest'], 'request_digest': manifest['request_digest'],
        'published_objects': [dict(r, upload_performed=False, full_byte_service_account_readback_passed=True)
                              for r in [object_row, manifest_object]],
        'host_only_source_objects': [r for r in manifest['files'] if not r['publication_allowed']],
        'raw_source_uploaded': False, 'provider_allocated': False, 'run_submitted': False,
        'full_byte_service_account_readback_passed': True, 'global_atomic_create_claimed': False}
    args['source_records']['submission_publications'] = [pair(factory_path.rsplit('/', 1)[0] + '/publication.json',
        seal(publication, 'receipt_digest'))]
    return args


@pytest.mark.parametrize('host_only', [False, True])
def test_publication_retains_manifest_raw_and_canonical_selectors_with_host_policy(host_only):
    result = api().join_retained_scene_source_family_inventory(**publication_fixture(host_only=host_only))
    observation = result['publication_observations'][0]
    assert observation['historical_publication_binding_verified']
    assert observation['current_remote_readback_verified'] is False
    assert result['fresh_remote_readback_verified'] is False


@pytest.mark.parametrize('edit', [
    lambda r: r.update(raw_source_uploaded=True),
    lambda r: r.update(global_atomic_create_claimed=0),
    lambda r: r['published_objects'].append(dict(r['published_objects'][0])),
    lambda r: r['published_objects'].pop(),
    lambda r: r['published_objects'][0].update(size_bytes=True),
    lambda r: r['published_objects'][0].update(full_byte_service_account_readback_passed=1),
    lambda r: r.update(host_only_source_objects=[r['published_objects'][0]]),
])
def test_available_publication_contradictions_refuse_without_handoff(edit):
    args = publication_fixture()
    args['source_records']['website_handoffs'] = []
    change(args, 'submission_publications', edit, 'receipt_digest')
    refuses(args)


@pytest.mark.parametrize('edit', [
    lambda m: m['files'][0].update(relative_path='../escape'),
    lambda m: m['files'][0].update(relative_path='derived//file'),
    lambda m: m['files'][0].update(size_bytes=True),
    lambda m: m['files'][0].update(publication_allowed=1),
    lambda m: m['files'][0].update(publication_allowed=False),
    lambda m: m['files'][0].update(uri='s3://foreign/file'),
    lambda m: m['files'].append(dict(m['files'][0])),
    lambda m: m.update(raw_source_upload_allowed=0),
])
def test_manifest_own_supported_grammar_refuses_before_missing_receipt(edit):
    args = publication_fixture()
    path, raw = args['seed_records']['source_submissions'][1]
    manifest = json.loads(raw)
    edit(manifest)
    args['seed_records']['source_submissions'][1] = pair(path, seal(manifest, 'manifest_digest'))
    # Keep the available factory's byte binding coherent; no old raw mismatch
    # should stand in for the source-family's manifest grammar validation.
    path, raw = args['seed_records']['factories'][0]
    factory = json.loads(raw)
    factory['submission_manifest'] = ref(args['seed_records']['source_submissions'][1])
    args['seed_records']['factories'][0] = pair(path, seal(factory, 'factory_digest'))
    args['source_records']['submission_publications'] = []
    refuses(args)


@pytest.mark.parametrize('edit', [
    lambda r: r['published_objects'][0].update(size_bytes=True),
    lambda r: r['published_objects'][0].update(uri='not-a-uri'),
    lambda r: r['published_objects'][0].update(full_byte_service_account_readback_passed=0),
    lambda r: r['published_objects'].append(dict(r['published_objects'][0])),
])
def test_publication_own_rows_validate_even_without_manifest_or_owner(edit):
    args = publication_fixture()
    change(args, 'submission_publications', edit, 'receipt_digest')
    args['seed_records']['source_submissions'] = args['seed_records']['source_submissions'][:1]
    args['seed_records']['factories'] = []
    refuses(args)


def test_handoff_own_reference_positive_size_validates_without_registration():
    args = fixture(website=True)
    change(args, 'website_handoffs', lambda r: r['source_registration'].update(size_bytes=0), 'digest')
    args['source_records']['website_registrations'] = []
    refuses(args)


@pytest.mark.parametrize('edit', [
    lambda h: h.update(preparation_path='/retained/pubsub/foreign/preparation.json'),
    lambda h: h.update(preparation_digest='sha256:' + 'f'*64),
    lambda h: h['runtime_inputs'].update(path='/retained/pubsub/foreign/runtime_inputs.json'),
    lambda h: h['runtime_inputs'].update(digest='sha256:' + 'f'*64),
])
def test_handoff_available_edges_validate_without_registration_bytes(edit):
    args = fixture(website=True)
    change(args, 'website_handoffs', edit, 'digest')
    args['source_records']['website_registrations'] = []
    refuses(args)


def test_status_only_error_runtime_handoff_is_legal_protected_history():
    args = fixture(website=True)
    change(args, 'website_handoffs', {'runtime_inputs': {'status': 'awaiting_inputs'}}, 'digest')
    result = api().join_retained_scene_source_family_inventory(**args)
    assert result['source_family_complete'] is False


@pytest.mark.parametrize('target', ['original', 'context'])
def test_development_original_and_context_available_selector_contradictions_refuse(target):
    args = fixture(website=True, development=True)
    if target == 'original':
        path, raw = args['source_records']['website_preparations'][1]
        value = json.loads(raw)
        value['status'] = 'needs_input'
        args['source_records']['website_preparations'][1] = pair(path, seal(value, 'digest'))
    else:
        change(args, 'website_preparations', lambda p: p['development_test'].update(source_task_context_digest='sha256:' + 'f'*64), 'digest')
        args['source_records']['website_runtime_inputs'] = []
        args['source_records']['website_handoffs'] = []
    args['source_records']['website_registrations'] = []
    refuses(args)


@pytest.mark.parametrize('edit', [
    lambda r: r['host_only_source_objects'][0].update(size_bytes=True),
    lambda r: r['host_only_source_objects'][0].update(relative_path='../bad'),
    lambda r: r['host_only_source_objects'][0].update(publication_allowed=True),
    lambda r: r['host_only_source_objects'].append(dict(r['host_only_source_objects'][0])),
])
def test_host_only_publication_own_rows_refuse_without_manifest(edit):
    args = publication_fixture(host_only=True)
    change(args, 'submission_publications', edit, 'receipt_digest')
    args['seed_records']['source_submissions'] = args['seed_records']['source_submissions'][:1]
    args['seed_records']['factories'] = []
    refuses(args)


def test_copied_manifest_provenance_selected_by_exact_factory_path_is_permutation_stable():
    args = publication_fixture()
    path, raw = args['seed_records']['source_submissions'][1]
    # The old child deliberately refuses foreign submissions, so its own valid
    # independent historical attempt carries the byte-identical manifest copy.
    from tests.test_scene_source_attempt_lineage import fixture as historical
    extra = historical('website', attempt_id='other-attempt')
    for old, new in (('attempt_records', 'attempts'), ('snapshot_records', 'source_snapshots'),
                     ('factory_records', 'factories'), ('submission_records', 'source_submissions')):
        args['seed_records'][new] += extra[old]
    other_manifest_path = extra['submission_records'][1][0]
    args['seed_records']['source_submissions'][-1] = (other_manifest_path, raw)
    other_request = args['seed_records']['source_submissions'][0]
    args['seed_records']['source_submissions'][-2] = (extra['submission_records'][0][0], other_request[1])
    factory_path, factory_raw = args['seed_records']['factories'][-1]
    factory = json.loads(factory_raw)
    factory.update(submission_manifest=ref(args['seed_records']['source_submissions'][-1]),
                   submission_request=ref(args['seed_records']['source_submissions'][-2]))
    args['seed_records']['factories'][-1] = pair(factory_path, seal(factory, 'factory_digest'))
    first = api().join_retained_scene_source_family_inventory(**args)
    assert first['publication_observations'][0]['historical_publication_binding_verified']
    args['seed_records']['source_submissions'].reverse()
    assert first == api().join_retained_scene_source_family_inventory(**args)


def test_unknown_selected_preparation_stays_raw_protected_without_known_semantics():
    args = fixture(website=True)
    path, raw = args['source_records']['website_preparations'][0]
    unknown = pair(path, {'schema_version': 'website_scene_preparation.v2', 'future_field': 'protected'})
    args['source_records']['website_preparations'] = [unknown]
    change(args, 'website_registrations', lambda r: r['references'].update(preparation=ref(unknown)), 'registration_digest')
    args['seed_records']['source_snapshots'] = []
    args['seed_records']['factories'] = []
    args['source_records']['website_bindings'] = []
    args['source_records']['website_handoffs'] = []
    result = api().join_retained_scene_source_family_inventory(**args)
    assert result['website_observations'][0]['capture_binding_verified'] is False
    assert any(r['path'] == path for r in result['raw_versions'])


@pytest.mark.parametrize('host_only', [False, True])
def test_available_manifest_receipt_edges_validate_without_factory_owner(host_only):
    args = publication_fixture(host_only=host_only)
    args['seed_records']['factories'] = []
    if host_only:
        change(args, 'submission_publications', {'host_only_source_objects': []}, 'receipt_digest')
    else:
        change(args, 'submission_publications', lambda p: p['published_objects'][0].update(relative_path='derived/other.json'), 'receipt_digest')
    refuses(args)


def test_host_only_duplicate_uri_refuses_without_manifest_even_with_distinct_paths():
    args = publication_fixture(host_only=True)
    def edit(p):
        p['host_only_source_objects'].append(dict(p['host_only_source_objects'][0], relative_path='source/other.glb'))
    change(args, 'submission_publications', edit, 'receipt_digest')
    args['seed_records']['source_submissions'] = args['seed_records']['source_submissions'][:1]
    args['seed_records']['factories'] = []
    refuses(args)


def test_known_owner_binding_selected_future_registration_keeps_raw_without_interpretation():
    args = fixture(website=True)
    rows, seed = args['source_records'], args['seed_records']
    path, raw = rows['website_registrations'][0]
    prior = json.loads(raw)
    future = pair(path, {'schema_version': 'website_scene_source_registration.v2', 'references': prior['references']})
    rows['website_registrations'] = [future]
    binding_path, raw = seed['source_snapshots'][0]
    binding = json.loads(raw)
    binding['registration'] = ref(future)
    binding = seal(binding, 'binding_digest')
    seed['source_snapshots'][0] = pair(binding_path, binding)
    rows['website_bindings'] = [pair(args['roots']['factory_output_root'] + '/' + args['intent_id'] +
        '/website-source/' + binding['binding_digest'][7:] + '.json', binding)]
    path, raw = seed['attempts'][0]
    attempt = json.loads(raw)
    attempt['input_digest'] = binding['binding_digest']
    attempt = seal(attempt, 'attempt_digest', cross=True)
    seed['attempts'][0] = pair(path, attempt)
    path, raw = seed['factories'][0]
    factory = json.loads(raw)
    factory['attempt_digest'] = attempt['attempt_digest']
    seed['factories'][0] = pair(path, seal(factory, 'factory_digest'))
    rows['website_handoffs'] = []
    result = api().join_retained_scene_source_family_inventory(**args)
    assert any(r['path'] == future[0] for r in result['raw_versions'])
    assert not any(m['kind'] == 'capture_dependency' for m in result['lexical_members'])
