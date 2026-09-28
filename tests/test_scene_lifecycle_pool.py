# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_pool.py
"""Discovery labels never substitute for exact queue-root/parent selectors."""
import hashlib
import json

from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
from blueprint_pipeline.task_evaluation_scene_lifecycle_pool import select
from tests.test_scene_compilation_owner_preparations import fixture


def decoded(role, pair):
    path, raw = pair
    return {'role': role, 'path': path, 'raw': raw, 'value': json.loads(raw),
            'sha256': 'sha256:' + hashlib.sha256(raw).hexdigest()}


def test_native_preparation_result_shared_schema_uses_exact_parent_mode():
    args = fixture()
    roots = args['roots']
    parent = args['bridge_records']['native_preparation_envelopes'][0]
    result = args['bridge_records']['native_preparation_results'][0]
    rows = [decoded('intent', args['seed_records']['intent']),
            decoded('parent_envelopes', parent), decoded('parent_results', result)]
    _, _, _, bridge, _ = select(rows, {'roots': roots,
        'parent_routes': args['parent_routes'], 'retained_metadata_files': [
            {'role': 'native_preparation_envelopes', 'path': parent[0]}]}, args['intent_id'], ReferenceCollectionBudget(monotonic=lambda: 0))
    assert bridge['native_preparation_results'] == [result]


def test_result_filename_never_selects_same_basename_from_another_queue():
    args = fixture()
    roots = args['roots']
    # Actual supported preparation link selects only its canonical queue.
    from tests.test_scene_inventory_preparations import fixture as legacy_fixture
    legacy = legacy_fixture()
    link = decoded('preparation_links', legacy['records']['preparation_links'][0])
    selected = legacy['records']['preparation_results'][0]
    foreign = ('/retained/foreign/results/' + selected[0].rsplit('/', 1)[1], selected[1])
    rows = [decoded('intent', legacy['records']['intent']), link,
            decoded('parent_envelopes', legacy['records']['preparation_envelopes'][0]),
            decoded('parent_results', selected), decoded('parent_results', foreign)]
    seed, _, _, bridge, protected = select(rows, {'roots': roots,
        'parent_routes': args['parent_routes'], 'retained_metadata_files': []}, args['intent_id'], ReferenceCollectionBudget(monotonic=lambda: 0))
    assert not any(pair[0] == foreign[0] for pairs in bridge.values() for pair in pairs)
    assert not any(pair[0] == foreign[0] for pair in seed['preparation_results'])
    assert any(row['path'] == foreign[0] for row in protected)


def test_publication_discovery_is_actual_attempt_sibling_not_submission_child():
    from blueprint_pipeline.task_evaluation_scene_lifecycle_pool import Pool
    from blueprint_pipeline.task_evaluation_scene_lifecycle_acquisition import AcquisitionError
    args = fixture()
    factory = args['roots']['factory_output_root']+'/'+args['intent_id']
    seen = []
    class Reader:
        budget = ReferenceCollectionBudget(monotonic=lambda: 0)
        def entries(self, path):
            return ('attempt-1',) if path == factory else ()
        def read_json(self, path):
            seen.append(path)
            raise AcquisitionError('scene_lifecycle_metadata_unavailable')
    context = {'roots': args['roots'], 'parent_routes': args['parent_routes'],
               'retained_metadata_files': [], 'progression_config': None}
    Pool(Reader(), context, args['intent_id']).discovery()
    assert factory+'/attempt-1/publication.json' in seen
    assert factory+'/attempt-1/materialized/submission/publication.json' not in seen


def test_fixed_family_discovery_covers_receipts_and_terminal_copies_without_payload_scan():
    from blueprint_pipeline.task_evaluation_scene_lifecycle_pool import Pool
    from blueprint_pipeline.task_evaluation_scene_lifecycle_acquisition import AcquisitionError
    args = fixture()
    roots, seen, directories = args['roots'], [], []
    h = 'a'*64
    owner = roots['terminal_result_root']+'/'+args['intent_id']
    layouts = {
        roots['configuration_progression_root']+'/scene-configuration-activations': ('prep-1',),
        roots['sam_execution_root']: (h,),
        roots['sam_execution_root']+'/'+h: ('sam31-'+h,),
        roots['launch_execution_root']: ('run-1',),
        owner+'/runs': (h,),
        roots['policy_canary_root']: ('canary-1', 'canary-1.offloaded.v1.json'),
        roots['activation_output_root']: ('activation-1',),
        roots['compilation_output_root']: ('compilation-1',),
    }
    class Reader:
        budget = ReferenceCollectionBudget(monotonic=lambda: 0)
        def entries(self, path):
            directories.append(path)
            return layouts.get(path, ())
        def read_json(self, path):
            seen.append(path)
            raise AcquisitionError('scene_lifecycle_metadata_unavailable')
    context = {'roots': roots, 'parent_routes': args['parent_routes'],
               'retained_metadata_files': [], 'progression_config': None}
    Pool(Reader(), context, args['intent_id']).discovery()
    expected = [
        roots['configuration_progression_root']+'/scene-configuration-activations/prep-1/activation_progression.json',
        roots['configuration_progression_root']+'/scene-configuration-activations/prep-1/launch_progression.json',
        roots['sam_execution_root']+'/'+h+'/sam31-'+h+'/phase_execution_receipt.v1.json',
        roots['launch_execution_root']+'/run-1/launch_request.json',
        owner+'/terminal_index_state.json',
        owner+'/runs/'+h+'/nonexecution_terminal_state.json',
        owner+'/runs/'+h+'/policy_canary_webapp_sync.json',
        roots['policy_canary_root']+'/canary-1/artifacts/result_delivery/policy_canary_result_projection.json',
        roots['policy_canary_root']+'/canary-1.offloaded.v1.json',
        roots['activation_output_root']+'/activation-1/scene_owner_attempt.json',
        roots['compilation_output_root']+'/compilation-1/native-arena-adapter/task_evaluation_native_arena_adapter_result.v1.json',
    ]
    assert set(expected) <= set(seen)
    assert all('/payload' not in path and '/frames' not in path for path in directories)


def test_unknown_selected_schema_keeps_raw_without_following_fictional_selectors():
    args = fixture()
    parent = args['bridge_records']['native_preparation_envelopes'][0]
    future = json.loads(parent[1])
    future['schema_version'] = 'future_preparation.v99'
    future['request_digest'] = 'sha256:'+'b'*64
    pair = parent[0], json.dumps(future).encode()
    rows = [decoded('intent', args['seed_records']['intent']), decoded('parent_envelopes', pair)]
    seed, _, _, bridge, protected = select(rows, {'roots': args['roots'],
        'retained_metadata_files': [{'role': 'native_preparation_envelopes', 'path': pair[0]}]},
        args['intent_id'], ReferenceCollectionBudget(monotonic=lambda: 0))
    assert all(not group for group in bridge.values())
    assert not seed['preparation_envelopes']
    assert any(p['path'] == pair[0] and p['status'] == 'kept_unsupported_schema' for p in protected)


def test_native_and_scene_activation_roles_use_actual_lane_not_identifier_length():
    from tests.test_scene_compilation_native_owner import fixture as native_fixture
    args = native_fixture(activation_id='scene-short')  # Same ID as acquired scene_configuration fixture.
    parent = args['bridge_records']['native_activation_envelopes'][0]
    result = args['bridge_records']['native_activation_results'][0]
    records = [decoded('intent', args['seed_records']['intent']),
               decoded('activation_envelopes', parent), decoded('activation_results', result)]
    _, downstream, _, bridge, _ = select(records, {'roots': args['roots'], 'retained_metadata_files':
        [{'role': 'native_activation_envelopes', 'path': parent[0]}]}, args['intent_id'],
        ReferenceCollectionBudget(monotonic=lambda: 0))
    assert bridge['native_activation_results'] == [result] and not downstream['activation_results']
    value = json.loads(parent[1])
    value['request']['lane'] = 'future_lane'
    unknown = parent[0], json.dumps(value).encode()
    records[1] = decoded('activation_envelopes', unknown)
    _, _, _, bridge, protected = select(records[:2], {'roots': args['roots'], 'retained_metadata_files':
        [{'role': 'native_activation_envelopes', 'path': parent[0]}]}, args['intent_id'],
        ReferenceCollectionBudget(monotonic=lambda: 0))
    assert not bridge['native_activation_envelopes']
    assert any(p['path'] == parent[0] and p['status'] == 'kept_activation_lane_unproven' for p in protected)


def test_capture_metadata_discovery_uses_fixed_layout_not_payload_directories():
    from blueprint_pipeline.task_evaluation_scene_lifecycle_pool import Pool
    from blueprint_pipeline.task_evaluation_scene_lifecycle_acquisition import AcquisitionError
    args = fixture()
    root = args['roots']['pubsub_root']
    observed, visited = [], []
    layouts = {root: ('bucket',), root+'/bucket/scenes': ('scene-1',),
               root+'/bucket/scenes/scene-1/captures': ('capture-1',)}
    class Reader:
        budget = ReferenceCollectionBudget(monotonic=lambda: 0)
        def entries(self, path):
            visited.append(path)
            return layouts.get(path, ())
        def read_json(self, path):
            observed.append(path)
            raise AcquisitionError('scene_lifecycle_metadata_unavailable')
    Pool(Reader(), {'roots': args['roots'], 'parent_routes': args['parent_routes'],
         'retained_metadata_files': [], 'progression_config': None}, args['intent_id']).discovery()
    prefix = root+'/bucket/scenes/scene-1/captures/capture-1/pipeline/website_scene_preparation/'
    assert prefix+'handoff.json' in observed and prefix+'native/runtime_inputs.json' in observed
    assert prefix+'development_test/preparation.json' in observed
    assert all('/frames' not in path and '/videos' not in path for path in visited)
