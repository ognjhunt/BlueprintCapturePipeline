# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_source_family_inventory.py
#   src/blueprint_pipeline/task_evaluation_scene_source_family_contracts.py
#   src/blueprint_pipeline/task_evaluation_scene_source_family_sam.py
#   src/blueprint_pipeline/task_evaluation_scene_source_family_adoption.py
#   src/blueprint_pipeline/task_evaluation_scene_lineage_budget.py
"""ADP-009D/day28: tiny combined bounds and fresh-process purity evidence."""
import copy
import json
import os
import subprocess
import sys

import pytest

from tests.test_scene_source_family_website import api, fixture, refuses
from tests.test_scene_source_family_adoption import fixture as adoption_fixture, inherited_fixture


@pytest.mark.parametrize('family', ['seed', 'downstream', 'source'])
@pytest.mark.parametrize('cap', ['MAX_DEPTH', 'MAX_NODES'])
def test_combined_preparse_refusal_precedes_every_parser_hash_or_child(monkeypatch, family, cap):
    args, module = fixture(), api()
    raw = b'{"x":[[[[0]]]]}' if cap == 'MAX_DEPTH' else b'{"x":[0,0,0,0]}'
    if family == 'seed':
        args['seed_records']['intent'] = (args['seed_records']['intent'][0], raw)
    elif family == 'downstream':
        args['downstream_records']['allocator_results'] = [('/retained/canaries/allocator.json', raw)]
    else:
        args['source_records']['sam_artifact_metadata'] = [('/retained/metadata/record.json', raw)]
    monkeypatch.setattr(module, cap, 1)
    def forbidden(*a, **k):
        pytest.fail('parser/hash/child before combined bound')
    monkeypatch.setattr(json, 'loads', forbidden)
    monkeypatch.setattr(module.contracts.c.retained, '_record', forbidden)
    monkeypatch.setattr(module, '_downstream_join', forbidden)
    refuses(args)


@pytest.mark.parametrize('cap', ['MAX_RECORDS', 'MAX_RECORD_BYTES', 'MAX_TOTAL_BYTES'])
def test_input_limits_precede_decode(monkeypatch, cap):
    module = api()
    monkeypatch.setattr(module, cap, 0)
    monkeypatch.setattr(module.contracts.c.retained, '_record', lambda *a: pytest.fail('decoded before input bound'))
    refuses(fixture())


@pytest.mark.parametrize('cap', ['MAX_ROWS', 'MAX_OUTPUT_BYTES'])
def test_wrapper_output_caps_precede_allocating_child_and_bulk_encoding(monkeypatch, cap):
    module = api()
    monkeypatch.setattr(module, cap, 0)
    monkeypatch.setattr(module, '_downstream_join', lambda **k: pytest.fail('child allocated before shared output cap'))
    monkeypatch.setattr(module.contracts.c, 'encoded', lambda *a: pytest.fail('encoded before output cap'))
    refuses(fixture())


@pytest.mark.parametrize('cap', ['MAX_ADOPTION_DEPTH', 'MAX_ADOPTION_NODES'])
def test_recursive_adoption_budget_refuses_with_tiny_historical_graph(monkeypatch, cap):
    args = inherited_fixture(through='sam31_tracking')
    monkeypatch.setattr(api(), cap, 1)
    refuses(args)


def test_graph_active_identity_cycle_refuses_before_second_visit(monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_source_family_adoption import Graph
    class Context:
        limits = {'MAX_ADOPTION_DEPTH': 4, 'MAX_ADOPTION_NODES': 4}
        def rows(self):
            return []
    graph = Graph(Context())
    row = ({}, {'path': '/retained/a', 'sha256': 'sha256:'+'1'*64, 'size_bytes': 1})
    monkeypatch.setattr(graph, '_visit', lambda row, depth: graph.visit(row, depth + 1))
    with pytest.raises(api().SceneSourceFamilyInventoryError, match='adoption_cycle'):
        graph.visit(row)


@pytest.mark.slow
@pytest.mark.parametrize('kind', ['website', 'adoption'])
def test_cold_import_and_call_cannot_use_host_runtime_environment_or_network(tmp_path, kind):
    args = fixture(website=True, development=True) if kind == 'website' else adoption_fixture()
    packet = copy.deepcopy(args)
    for role in ('intent', 'projection'):
        row = packet['seed_records'][role]
        if row is not None:
            packet['seed_records'][role] = [row[0], row[1].decode()]
    for name in ('seed_records', 'downstream_records', 'source_records'):
        for role, rows in packet[name].items():
            if role not in ('intent', 'projection'):
                packet[name][role] = [[p, r.decode()] for p, r in rows]
    path = tmp_path / 'packet.json'
    path.write_text(json.dumps(packet))
    script = r'''
import json,pathlib,sys,os,subprocess,socket
packet=json.loads(pathlib.Path(sys.argv[1]).read_text())
for role in ('intent','projection'):
 row=packet['seed_records'][role]
 if row is not None: packet['seed_records'][role]=(row[0],row[1].encode())
for name in ('seed_records','downstream_records','source_records'):
 for role,rows in packet[name].items():
  if role not in ('intent','projection'): packet[name][role]=[(p,r.encode()) for p,r in rows]
def forbidden(*a,**k): raise AssertionError('cold runtime effect')
for name in ('resolve','stat','lstat','read_bytes','read_text','write_bytes','write_text','glob','mkdir'):
 setattr(pathlib.Path,name,forbidden)
subprocess.run=forbidden
subprocess.Popen=forbidden
socket.create_connection=forbidden
os.getenv=forbidden
from blueprint_pipeline.task_evaluation_scene_source_family_inventory import join_retained_scene_source_family_inventory
result=join_retained_scene_source_family_inventory(**packet)
assert result['mutations']==0 and result['cleanup_authorized'] is False
for name in ('website_scene_dispatch','website_handoff','task_evaluation_sam31_phase_queue',
 'task_evaluation_sam31_prefix_adoption','task_evaluation_sam31_preparation_execution',
 'task_evaluation_scene_configuration_sam31_preparation_driver','public_scene_sam31_task_inputs',
 'task_evaluation_controls_worker','task_evaluation_retained_preparation_contract'):
 assert 'blueprint_pipeline.'+name not in sys.modules,name
print('pure')
'''
    result = subprocess.run([sys.executable, '-c', script, str(path)], env={**os.environ, 'PYTHONPATH': 'src'},
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'pure'
