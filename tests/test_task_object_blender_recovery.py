"""Retained Blender replay keeps physics and receipt authority outside the model."""
import pytest

from blueprint_pipeline import task_object_astra_authoring as author
from blueprint_pipeline import task_object_astra_worker as worker
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_object_blender_runtime import mesh_facts


def test_only_rigidbody_object_add_is_admitted_for_superseded_appearance_dynamics():
    author.validate_blender_program('import bpy\nbpy.ops.rigidbody.object_add()\nCAD_BASE.rigid_body.mass = 1.2')
    for operation in ('object_remove', 'world_add', 'world_remove', 'bake_to_keyframes'):
        with pytest.raises(author.AssetAuthoringError, match='operator_forbidden'):
            author.validate_blender_program(f'bpy.ops.rigidbody.{operation}()')
    with pytest.raises(author.AssetAuthoringError, match='operator_forbidden'):
        author.validate_blender_program('bpy.ops.ptcache.bake_all()')


def test_final_mesh_facts_detect_disconnection_open_edges_and_inverted_winding():
    vertices = [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]
    faces = [[0,2,1],[0,1,3],[0,3,2],[1,2,3]]
    facts = mesh_facts(vertices, faces)
    assert facts['signed_volume_m3'] == pytest.approx(1/6)
    assert facts['watertight'] and facts['consistent_winding']
    assert facts['connected_component_count'] == 1
    assert facts['dimensions_m'] == [1,1,1]
    assert not mesh_facts(vertices, faces[:-1])['watertight']
    assert not mesh_facts(vertices, [faces[0][::-1], *faces[1:]])['consistent_winding']
    double = vertices + [[x+2,y,z] for x,y,z in vertices]
    assert mesh_facts(double, faces + [[a+4,b+4,c+4] for a,b,c in faces])['connected_component_count'] == 2


def test_blender_program_adoption_requires_unchanged_request_and_completed_sdk_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'validate_request', lambda value: value)
    prior = tmp_path / 'prior'
    (prior / 'appearance-00').mkdir(parents=True)
    budget = tmp_path / 'budget'
    completed = budget / 'inference_reservations/completed'
    completed.mkdir(parents=True)
    request = {'run_id': 'same-run', 'object_id': 'book', 'source_frames': [],
               'request_digest': 'prior-digest', 'expected_production_commit': 'prior-commit',
               'dimensions_m': [1,2,3]}
    program = {'program': 'import bpy\nbpy.ops.rigidbody.object_add()', 'explanation': 'candidate',
               'generated_surface_assumptions': ['unknown underside']}
    phase = {'request_digest': request['request_digest'], 'model': 'gpt-6-astra', 'provider': 'openai',
             'references': [], 'output': program}
    completion = {'run_id': 'same-run', 'capability': 'book_blender_author_0',
                  'provider': 'openai', 'model': 'gpt-6-astra',
                  'structured_output_digest': canonical_digest(program)}
    completion['inference_completion_digest'] = canonical_digest(completion, digest_field='inference_completion_digest')
    author.save_json(prior / 'request.json', request)
    author.save_json(prior / 'appearance-00/blender_author_0.json', phase)
    author.save_json(prior / 'cad_result.json', {'readback': {'volume_mm3': 6000}})
    author.save_json(completed / 'receipt.json', completion)
    current = {**request, 'expected_production_commit': 'fixed-commit', 'request_digest': 'new-digest'}
    output, receipt = worker.verify_blender_program_adoption(prior_root=prior, request_value=current, budget_root=budget)
    assert output.program == program['program'] and receipt['new_provider_call'] is False
    assert receipt['cad_readback_digest'] == canonical_digest({'volume_mm3': 6000})
    with pytest.raises(author.AssetAuthoringError, match='source_inputs_changed'):
        worker.verify_blender_program_adoption(prior_root=prior, request_value={**current,'dimensions_m':[2,2,3]}, budget_root=budget)
    phase['output']['program'] += '\nCAD_BASE.rigid_body.mass=1000'
    author.save_json(prior / 'appearance-00/blender_author_0.json', phase)
    with pytest.raises(author.AssetAuthoringError, match='completed_response_missing'):
        worker.verify_blender_program_adoption(prior_root=prior, request_value=current, budget_root=budget)
