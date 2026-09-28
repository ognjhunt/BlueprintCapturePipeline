# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_compilation_native_owner_inventory.py
#   src/blueprint_pipeline/task_evaluation_scene_compilation_owner_contracts.py
#   src/blueprint_pipeline/task_evaluation_scene_compilation_owner_preparations.py
#   src/blueprint_pipeline/task_evaluation_scene_compilation_owner_outputs.py
#   src/blueprint_pipeline/task_evaluation_scene_compilation_native_owners.py
"""ADP-009D/day28: one bounded pure invocation over old and new records."""
import copy
import json
import os
import subprocess
import sys

import pytest

from tests.test_scene_compilation_owner_preparations import api, change, fixture as prep_fixture, refuses
from tests.test_scene_compilation_native_owner import fixture


@pytest.mark.parametrize('family', ['seed', 'downstream', 'source', 'bridge'])
@pytest.mark.parametrize('cap', ['MAX_DEPTH', 'MAX_NODES', 'MAX_RECORDS', 'MAX_TOTAL_BYTES', 'MAX_RECORD_BYTES'])
def test_global_input_preflight_precedes_parsing_hashing_and_child(monkeypatch, family, cap):
    args, module = prep_fixture(), api()
    raw = b'{"x":[[[[0]]]]}'
    if family == 'seed':
        args['seed_records']['intent'] = (args['seed_records']['intent'][0], raw)
    elif family == 'downstream':
        args['downstream_records']['allocator_results'] = [('/retained/a.json', raw)]
    elif family == 'source':
        args['source_records']['sam_artifact_metadata'] = [('/retained/a.json', raw)]
    else:
        args['bridge_records']['compiler_outputs'] = [('/retained/a.json', raw)]
    monkeypatch.setattr(module, cap, 1)
    def forbidden(*a, **k):
        pytest.fail('parsed/hashed/called child before global bound')
    monkeypatch.setattr(json, 'loads', forbidden)
    monkeypatch.setattr(module.c.retained.c, 'raw_digest', forbidden)
    monkeypatch.setattr(module.prior, '_join', forbidden)
    refuses(args)


@pytest.mark.parametrize('cap', ['MAX_ROWS', 'MAX_OUTPUT_BYTES', 'MAX_REFERENCES'])
def test_wrapper_output_and_reference_caps_precede_child(monkeypatch, cap):
    module = api()
    monkeypatch.setattr(module, cap, 0)
    monkeypatch.setattr(module.prior, '_join', lambda *a, **k: pytest.fail('child before parent refusal'))
    refuses(prep_fixture())


def test_inverse_refusal_precedes_canonical_hashing(monkeypatch):
    module = api()
    from blueprint_pipeline.task_evaluation_scene_lineage_budget import RetainedEmissionBudget, RetainedEmissionBudgetError
    reserve, digest = RetainedEmissionBudget.reserve_row, module.c.canonical_digest
    def guard(self, value, **kwargs):
        if isinstance(value, dict) and value.get('status') == 'inputs_materialized_awaiting_construction_adapter':
            raise RetainedEmissionBudgetError('retained_lineage_emission_limit')
        return reserve(self, value, **kwargs)
    def hash_guard(value, **kwargs):
        if value.get('status') == 'inputs_materialized_awaiting_construction_adapter':
            pytest.fail('inverse encoded/hashed before reservation')
        return digest(value, **kwargs)
    monkeypatch.setattr(RetainedEmissionBudget, 'reserve_row', guard)
    monkeypatch.setattr(module.c, 'canonical_digest', hash_guard)
    refuses(prep_fixture())


@pytest.mark.parametrize('raw', [b'{"x":1,"x":2}', b'{"x":NaN}', b'[]', b'{"x":"\\ud800"}', b''])
def test_bridge_json_failures_are_fixed_and_private(raw):
    args = prep_fixture()
    args['bridge_records']['compiler_outputs'] = [('/retained/metadata/private.json', raw)]
    refuses(args)


@pytest.mark.parametrize('role,field', [('native_preparation_results', 'result_digest'),
    ('compiler_outputs', 'compiler_output_digest'), ('native_owner_records', 'owner_attempt_digest'),
    ('native_activation_results', 'result_digest')])
def test_future_schema_raw_versions_remain_without_supported_semantics(role, field):
    args = fixture()
    change(args, role, {'schema_version': 'future.v2'}, field)
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    assert any(r['role'] == role for r in result['raw_versions'])
    assert any(r['reason'] == 'unsupported_retained_schema' for r in result['structural_join_obligations'])


def test_all_family_permutations_are_exactly_deterministic_and_no_input_mutation():
    args = fixture()
    # Alternate raw formatting of each mutable role is retained as a historical
    # proof, not a latest selection; finite inverse and native request stay exact.
    for role in ('native_preparation_results', 'native_activation_results', 'compiler_outputs', 'native_owner_records'):
        path, raw = args['bridge_records'][role][0]
        args['bridge_records'][role].append((path, json.dumps(json.loads(raw), indent=2).encode()))
    before = copy.deepcopy(args)
    first = api().join_retained_scene_compilation_native_owner_inventory(**args)
    assert args == before
    for mapping in ('seed_records', 'downstream_records', 'source_records', 'bridge_records'):
        for rows in args[mapping].values():
            if isinstance(rows, list):
                rows.reverse()
    assert first == api().join_retained_scene_compilation_native_owner_inventory(**args)
    assert all(first[flag] is False for flag in api().prior.FALSE_FLAGS)


@pytest.mark.parametrize('missing', ['native_preparation_envelopes', 'native_preparation_results', 'compiler_outputs',
    'compilation_adapter_results', 'native_owner_records', 'native_activation_results'])
def test_missing_edges_preserve_all_other_raw_proofs_and_no_authority(missing):
    args = fixture()
    args['bridge_records'][missing] = []
    result = api().join_retained_scene_compilation_native_owner_inventory(**args)
    expected = {(path, api().c.retained.raw_digest(raw)) for role, rows in args['bridge_records'].items() for path, raw in rows}
    assert {(r['path'], r['sha256']) for r in result['raw_versions']} == expected
    assert result['complete_scene_inventory'] is False and result['cleanup_authorized'] is False


@pytest.mark.slow
@pytest.mark.parametrize('kind', ['preparation', 'native_owner'])
def test_cold_first_call_denies_host_runtime_environment_network_and_process(tmp_path, kind):
    packet = prep_fixture() if kind == 'preparation' else fixture()
    for role in ('intent', 'projection'):
        row = packet['seed_records'][role]
        if row is not None:
            packet['seed_records'][role] = [row[0], row[1].decode()]
    for family in ('seed_records', 'downstream_records', 'source_records', 'bridge_records'):
        for role, rows in packet[family].items():
            if role not in ('intent', 'projection'):
                packet[family][role] = [[p, raw.decode()] for p, raw in rows]
    path = tmp_path/'packet.json'
    path.write_text(json.dumps(packet))
    script = r'''
import json,pathlib,sys,os,subprocess,socket
packet=json.loads(pathlib.Path(sys.argv[1]).read_text())
for role in ('intent','projection'):
 row=packet['seed_records'][role]
 if row is not None: packet['seed_records'][role]=(row[0],row[1].encode())
for family in ('seed_records','downstream_records','source_records','bridge_records'):
 for role,rows in packet[family].items():
  if role not in ('intent','projection'): packet[family][role]=[(p,r.encode()) for p,r in rows]
def forbidden(*a,**k): raise AssertionError('cold runtime effect')
for name in ('resolve','stat','lstat','read_bytes','read_text','write_bytes','write_text','glob','mkdir'):
 setattr(pathlib.Path,name,forbidden)
subprocess.run=forbidden
subprocess.Popen=forbidden
socket.create_connection=forbidden
os.getenv=forbidden
from blueprint_pipeline.task_evaluation_scene_compilation_native_owner_inventory import join_retained_scene_compilation_native_owner_inventory
result=join_retained_scene_compilation_native_owner_inventory(**packet)
assert result['mutations']==0 and result['cleanup_authorized'] is False
for name in ('task_evaluation_episode_compilation_worker','task_evaluation_launch_activation_worker',
 'task_evaluation_native_arena_episode_compiler','task_evaluation_native_arena_preparation_adapter',
 'task_evaluation_scene_owner_attempt_profiles','task_evaluation_scene_execution_authority',
 'task_evaluation_controls_autoprovision','task_evaluation_configured_scene_revision'):
 assert 'blueprint_pipeline.'+name not in sys.modules,name
print('pure')
'''
    result = subprocess.run([sys.executable, '-c', script, str(path)], env={**os.environ, 'PYTHONPATH': 'src'},
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'pure'
