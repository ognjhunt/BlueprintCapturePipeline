# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_downstream_inventory.py
#   src/blueprint_pipeline/task_evaluation_scene_downstream_contracts.py
#   src/blueprint_pipeline/task_evaluation_scene_downstream_execution.py
#   src/blueprint_pipeline/task_evaluation_scene_downstream_terminal.py
"""Tiny preallocation, history preservation, cold-purity and compatibility proofs."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import pytest

from tests.test_scene_downstream_execution import D, api, edit, fixture, refuses
from tests.test_scene_downstream_terminal import fixture as terminal_fixture
from tests.test_scene_inventory_history import pair, ref, seal


def test_oversized_root_path_refuses_before_utf8_allocation():
    class Guarded(str):
        def encode(self, *args, **kwargs):
            pytest.fail('oversized path encoded before owned length guard')
    args = fixture()
    args['roots']['policy_canary_root'] = Guarded('/' + 'x' * 4097)
    refuses(args)


@pytest.mark.parametrize('family', ['seed', 'downstream'])
@pytest.mark.parametrize('cap', ['MAX_DEPTH', 'MAX_NODES'])
def test_combined_lexical_caps_precede_json_record_hash_and_seed(monkeypatch, family, cap):
    args, module = fixture(), api()
    raw = b'{"x":[[[[0]]]]}' if cap == 'MAX_DEPTH' else b'{"x":[0,0,0,0]}'
    if family == 'seed':
        args['seed_records']['intent'] = (args['seed_records']['intent'][0], raw)
    else:
        args['downstream_records']['allocator_results'] = [('/retained/canaries/run/allocator_result.json', raw)]
    monkeypatch.setattr(module, cap, 1)
    def forbidden(*args, **kwargs):
        pytest.fail('parser/hash/seed before lexical limit')
    monkeypatch.setattr(module.contracts.retained, '_record', forbidden)
    monkeypatch.setattr(json, 'loads', forbidden)
    monkeypatch.setattr(module.seed_module, 'join_retained_scene_inventory_seed', forbidden)
    refuses(args)


def test_actual_scalar_depth_precedes_any_raw_hash_or_seed(monkeypatch):
    args, module = fixture(), api()
    # Container depth2 but scalar/object-key decoded depth3.
    args['downstream_records']['allocator_results'] = [('/retained/canaries/run/allocator_result.json', b'{"x":{"y":0}}')]
    # Exercise only the additional record so the existing fixture does not trip
    # the earlier lexical guard, and prove all parsed records precede hashing.
    groups = {'allocator_results': args['downstream_records']['allocator_results']}
    limits = {name: getattr(module, name) for name in ('MAX_RECORD_BYTES', 'MAX_TOTAL_BYTES', 'MAX_RECORDS', 'MAX_NODES', 'MAX_DEPTH')}
    limits['MAX_DEPTH'] = 2
    monkeypatch.setattr(module.contracts.retained, '_record', lambda *a: pytest.fail('hashed before actual depth'))
    monkeypatch.setattr(module.contracts, 'raw_digest', lambda *a: pytest.fail('hashed before actual depth'), raising=False)
    with pytest.raises(module.SceneDownstreamInventoryError):
        module.contracts.decode(groups, limits)


def test_string_delimiters_and_escapes_do_not_count_as_json_depth():
    args = fixture()
    args['downstream_records']['allocator_results'] = [pair('/retained/canaries/run/allocator_result.json', {'text': '{[\\\"},] ' * 20})]
    assert api().join_retained_scene_downstream_inventory(**args)['mutations'] == 0


@pytest.mark.parametrize('cap', ['MAX_RECORDS', 'MAX_RECORD_BYTES', 'MAX_TOTAL_BYTES'])
def test_input_caps_before_decode(monkeypatch, cap):
    args = fixture()
    monkeypatch.setattr(api(), cap, 0)
    monkeypatch.setattr(api().contracts.retained, '_record', lambda *a: pytest.fail('parsed before cap'))
    refuses(args)


@pytest.mark.parametrize('raw', [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', br'{"x":"\ud800"}', b'[]', b'\xff'])
def test_invalid_retained_json_uses_fixed_error(raw):
    args = fixture()
    args['downstream_records']['allocator_results'] = [('/retained/canaries/run/allocator_result.json', raw)]
    refuses(args)


def test_row_and_output_caps_refuse_before_bulk_output_serialization(monkeypatch):
    for cap in ('MAX_ROWS', 'MAX_OUTPUT_BYTES'):
        with monkeypatch.context() as scoped:
            scoped.setattr(api(), cap, 0)
            scoped.setattr(api().contracts, 'encoded', lambda *a: pytest.fail('serialized before output cap'))
            refuses(fixture())


def test_raw_output_cap_exact_inclusive(monkeypatch):
    args, module = fixture(), api()
    result = module.join_retained_scene_downstream_inventory(**args)
    size = len(module.contracts.encoded(result))
    monkeypatch.setattr(module, 'MAX_OUTPUT_BYTES', size)
    assert module.join_retained_scene_downstream_inventory(**args) == result
    monkeypatch.setattr(module, 'MAX_OUTPUT_BYTES', size - 1)
    refuses(args)


def test_envelope_state_alternatives_never_overwrite_or_promote_ambiguity(monkeypatch):
    args, module = fixture(), api()
    path, raw = args['seed_records']['activation_envelopes'][0]
    args['seed_records']['activation_envelopes'].append((path.replace('/pending/', '/processing/'), raw))
    # Public11c remains unchanged and refuses ambiguous envelope input. Isolate
    # the new join to prove it never silently reduces supplied alternatives.
    original = module.seed_module.join_retained_scene_inventory_seed
    seed = original(intent_id=args['intent_id'], records={**args['seed_records'], 'activation_envelopes': [(path, raw)]},
                    roots={k: v for k, v in args['roots'].items() if k not in module.EXTRA_ROOTS})
    monkeypatch.setattr(module.seed_module, 'join_retained_scene_inventory_seed', lambda **kwargs: seed)
    result = module.join_retained_scene_downstream_inventory(**args)
    assert not any(row['kind'] == 'activation_workspace' for row in result['lexical_members'])
    assert result['activation_observations'][0]['reason'] == 'activation_envelope_ambiguous'
    assert len([p for p in result['raw_versions'] if p['role'] == 'activation_envelopes']) == 2


def test_missing_related_edges_do_not_allow_impossible_terminal_layout():
    args = terminal_fixture()
    path, raw = args['downstream_records']['terminal_states'][0]
    args['downstream_records']['terminal_states'][0] = (path.replace('/intent-1/', '/wrong-layout/'), raw)
    args['downstream_records']['launch_requests'] = []
    args['downstream_records']['launch_profiles'] = []
    refuses(args)


@pytest.mark.parametrize('role', ['canary_projections', 'canary_syncs', 'provider_zero_receipts'])
def test_exact_selected_unknown_schema_keeps_raw_version_unresolved(role):
    args = terminal_fixture()
    rows = args['downstream_records']
    field = {'canary_projections': 'projection_digest', 'canary_syncs': None, 'provider_zero_receipts': 'receipt_digest'}[role]
    for i, (path, raw) in enumerate(rows[role]):
        v = json.loads(raw)
        v['schema_version'] = 'future_retained_record.v2'
        rows[role][i] = pair(path, seal(v, field, cross=role == 'canary_projections') if field else v)
    reference_name = {'canary_projections': 'policy_canary_result_projection', 'canary_syncs': 'policy_canary_webapp_sync', 'provider_zero_receipts': 'provider_zero'}[role]
    for i in (0, 1):
        edit(args, 'canary_dispatches', {reference_name: {**ref(rows[role][0]), **({'provider_zero_verified': True} if role == 'provider_zero_receipts' else {})}}, 'receipt_digest', position=i)
    edit(args, 'terminal_states', {'dispatch_receipt_digest': json.loads(rows['canary_dispatches'][0][1])['receipt_digest']}, 'state_digest')
    result = api().join_retained_scene_downstream_inventory(**args)
    assert any(r['reason'] == 'terminal_reference_schema_unproven' for r in result['terminal_observations'])
    assert len([p for p in result['raw_versions'] if p['role'] == role]) == 2
    assert not any(r['kind'] == 'canary_evidence_workspace' for r in result['lexical_members'])


def test_available_sync_notification_mismatch_refuses_when_projection_absent():
    args = terminal_fixture()
    args['downstream_records']['canary_projections'] = []
    for i, (path, raw) in enumerate(args['downstream_records']['canary_syncs']):
        value = json.loads(raw)
        value['notification_delivery'] = {'status': 'wrong'}
        args['downstream_records']['canary_syncs'][i] = pair(path, value)
    for i in (0, 1):
        edit(args, 'canary_dispatches', {'policy_canary_webapp_sync': ref(args['downstream_records']['canary_syncs'][0])}, 'receipt_digest', position=i)
    edit(args, 'terminal_states', {'dispatch_receipt_digest': json.loads(args['downstream_records']['canary_dispatches'][0][1])['receipt_digest']}, 'state_digest')
    refuses(args)


@pytest.mark.parametrize('role', ['activation_results', 'launch_profiles', 'launch_requests', 'launch_receipts'])
def test_unknown_execution_record_versions_are_never_dropped_or_promoted(role):
    args = fixture()
    path, raw = args['downstream_records'][role][0]
    value = json.loads(raw)
    value['schema_version'] = 'future_retained_record.v2'
    args['downstream_records'][role][0] = pair(path, value)
    # Remove related known selectors so this is an unsupported supplied record,
    # not a made-up exact supported identity contradiction.
    args['downstream_records']['launch_progressions'] = []
    if role == 'launch_profiles':
        args['downstream_records']['launch_receipts'] = []
    result = api().join_retained_scene_downstream_inventory(**args)
    assert any(p['role'] == role for p in result['raw_versions'])


def test_all_mutable_versions_survive_without_owner_or_state_projection():
    args = terminal_fixture()
    args['downstream_records']['terminal_states'] = []
    path, raw = args['downstream_records']['canary_dispatches'][0]
    historical = json.loads(raw)
    historical['historical_note'] = 'old byte version'
    args['downstream_records']['canary_dispatches'].append(pair(path, seal(historical, 'receipt_digest')))
    result = api().join_retained_scene_downstream_inventory(**args)
    assert len([p for p in result['raw_versions'] if p['role'] == 'canary_dispatches']) == 3
    assert not any(r['kind'] == 'canary_evidence_workspace' for r in result['lexical_members'])


def test_full_terminal_permutations_and_caller_nonmutation():
    args = terminal_fixture(pointer=True)
    saved = copy.deepcopy(args)
    result = api().join_retained_scene_downstream_inventory(**args)
    assert args == saved
    for rows in args['downstream_records'].values():
        rows.reverse()
    for key, rows in args['seed_records'].items():
        if key not in ('intent', 'projection'):
            rows.reverse()
    assert api().join_retained_scene_downstream_inventory(**args) == result


@pytest.mark.slow
@pytest.mark.parametrize('nonexecution', [None, 'prepared_no_execution'])
def test_first_downstream_join_in_cold_process_has_no_runtime_effects(tmp_path, nonexecution):
    args = terminal_fixture(nonexecution=nonexecution, pointer=nonexecution is None)
    packet = copy.deepcopy(args)
    for key in ('intent', 'projection'):
        p = packet['seed_records'][key]
        if p is not None:
            packet['seed_records'][key] = [p[0], p[1].decode()]
    for name in ('seed_records', 'downstream_records'):
        for role, rows in packet[name].items():
            if role not in ('intent', 'projection'):
                packet[name][role] = [[p, raw.decode()] for p, raw in rows]
    packet_path = tmp_path / 'packet.json'
    packet_path.write_text(json.dumps(packet))
    script = r'''
import json,pathlib,sys,os,subprocess,socket
packet=json.loads(pathlib.Path(sys.argv[1]).read_text())
for role in ('intent','projection'):
 row=packet['seed_records'][role]
 if row is not None: packet['seed_records'][role]=(row[0],row[1].encode())
for name in ('seed_records','downstream_records'):
 for role,rows in packet[name].items():
  if role not in ('intent','projection'): packet[name][role]=[(p,r.encode()) for p,r in rows]
def forbidden(*a,**k): raise AssertionError('cold runtime effect')
for name in ('resolve','stat','lstat','read_bytes','read_text','write_bytes','write_text','glob','mkdir'):
 setattr(pathlib.Path,name,forbidden)
subprocess.run=forbidden
subprocess.Popen=forbidden
socket.create_connection=forbidden
os.getenv=forbidden
from blueprint_pipeline.task_evaluation_scene_downstream_inventory import join_retained_scene_downstream_inventory
result=join_retained_scene_downstream_inventory(**packet)
assert result['mutations']==0 and result['cleanup_authorized'] is False
for name in ('task_evaluation_scene_intake','task_evaluation_scene_policy_binding','task_evaluation_scene_terminal_result_index',
 'task_evaluation_scene_terminal_reconciler','task_evaluation_launch_dispatcher','task_evaluation_launch_activation_worker',
 'task_evaluation_policy_canary_result','task_evaluation_controls_worker'):
 assert 'blueprint_pipeline.'+name not in sys.modules,name
print('pure')
'''
    result = subprocess.run([sys.executable, '-c', script, str(packet_path)], env={**os.environ, 'PYTHONPATH': 'src'},
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'pure'


def test_direct_binding_intent_id_contradiction_refuses_even_with_selected_digest():
    args = fixture()
    edit(args, 'launch_profiles', lambda p: p['scene_attempt_binding'].update(intent_id='foreign-intent'), 'profile_digest')
    digest = json.loads(args['downstream_records']['launch_profiles'][0][1])['profile_digest']
    edit(args, 'launch_requests', {'launch_profile_digest': digest}, 'request_digest')
    args['downstream_records']['launch_progressions'] = []
    args['downstream_records']['launch_receipts'] = []
    refuses(args)


def test_owner_bound_activation_retains_both_absent_raw_digest_selectors():
    result = api().join_retained_scene_downstream_inventory(**fixture())
    rows = [r for r in result['structural_join_obligations'] if r['role'] in ('profile_publication_receipt', 'standing_authorization')]
    assert {r['role'] for r in rows} == {'profile_publication_receipt', 'standing_authorization'}
    assert all(r['selector']['sha256'] == D and 'size_bytes' not in r['selector'] for r in rows)


def test_coherent_unknown_executed_status_keeps_records_without_canary_promotion():
    args = terminal_fixture()
    rows = args['downstream_records']
    for i in (0, 1):
        edit(args, 'canary_projections', {'result_status': 'future_status'}, 'projection_digest', cross=True, position=i)
        path, raw = rows['canary_syncs'][i]
        sync = json.loads(raw)
        sync.update(result_status='future_status', policy_canary_projection_digest=json.loads(rows['canary_projections'][i][1])['projection_digest'])
        rows['canary_syncs'][i] = pair(path, sync)
    for i in (0, 1):
        edit(args, 'canary_dispatches', {'status': 'future_status',
             'policy_canary_result_projection': ref(rows['canary_projections'][0]),
             'policy_canary_webapp_sync': ref(rows['canary_syncs'][0]),
             'policy_canary_projection_digest': json.loads(rows['canary_projections'][0][1])['projection_digest']}, 'receipt_digest', position=i)
    edit(args, 'terminal_states', {'dispatch_receipt_digest': json.loads(rows['canary_dispatches'][0][1])['receipt_digest'],
                                  'projection_digest': json.loads(rows['canary_projections'][0][1])['projection_digest']}, 'state_digest')
    result = api().join_retained_scene_downstream_inventory(**args)
    assert any(r['reason'] == 'terminal_execution_status_unproven' for r in result['terminal_observations'])
    assert not any(r['kind'] == 'canary_evidence_workspace' for r in result['lexical_members'])


def test_historical_pointer_publication_pairs_use_exact_seals_and_preserve_proof():
    args = terminal_fixture(pointer=True)
    rows = args['downstream_records']
    path, raw = rows['canary_offload_pointers'][0]
    older = json.loads(raw)
    older.update(uri='s3://evidence/older.tar.gz', digest='sha256:' + 'e' * 64, size_bytes=90)
    older = seal(older, 'pointer_digest')
    rows['canary_offload_pointers'].append(pair(path, older))
    pubpath, pubraw = rows['terminal_publications'][0]
    publication = json.loads(pubraw)
    publication.update(uri=older['uri'], archive_digest=older['digest'], size_bytes=90, pointer_digest=older['pointer_digest'])
    rows['terminal_publications'].append(pair(pubpath, seal(publication, 'publication_digest')))
    result = api().join_retained_scene_downstream_inventory(**args)
    row = result['terminal_observations'][0]
    assert row['archive_binding_verified'] is True
    assert {p['role'] for p in row['source_provenance']} >= {'canary_offload_pointers', 'terminal_publications'}
    assert len([p for p in row['source_provenance'] if p['role'] == 'canary_offload_pointers']) == 2
    for role in ('canary_offload_pointers', 'terminal_publications'):
        rows[role].reverse()
    assert api().join_retained_scene_downstream_inventory(**args) == result
    rows['canary_offload_pointers'] = rows['canary_offload_pointers'][:1]
    result = api().join_retained_scene_downstream_inventory(**args)
    assert any(r['reason'] == 'publication_pointer_version_unavailable' for r in result['structural_join_obligations'])


@pytest.mark.parametrize('size', [0, True, -1])
def test_known_dispatch_positive_byte_reference_grammar_refuses_even_without_state(size):
    args = terminal_fixture()
    for i in (0, 1):
        edit(args, 'canary_dispatches', lambda r: r['policy_canary_result_projection'].update(size_bytes=size), 'receipt_digest', position=i)
    args['downstream_records']['terminal_states'] = []
    refuses(args)


def test_reference_length_contradicts_available_raw_identity_instead_of_missing():
    args = terminal_fixture(nonexecution='prepared_no_execution')
    for i in (0, 1):
        edit(args, 'canary_dispatches', lambda r: r['allocator_result'].update(size_bytes=r['allocator_result']['size_bytes'] + 1), 'receipt_digest', position=i)
    args['downstream_records']['terminal_states'] = []
    refuses(args)
