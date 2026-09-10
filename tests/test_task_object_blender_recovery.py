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


def test_globals_only_reads_named_public_blender_inputs():
    for name in ('CAD_BASE', 'DIMENSIONS', 'SOURCE_IMAGES'):
        author.validate_blender_program(f"obj = globals().get('{name}')")
    for code in ("x = globals()", "x = globals().get('secret')", "globals().update(x=1)",
                 "x = globals().get(variable)", "x = globals().get('CAD_BASE', open)"):
        with pytest.raises(author.AssetAuthoringError, match='operation_forbidden'):
            author.validate_blender_program(code)


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


@pytest.mark.parametrize('round_index', [0, 1])
def test_blender_program_adoption_requires_unchanged_request_and_completed_sdk_digest(tmp_path, monkeypatch, round_index):
    monkeypatch.setattr(worker, 'validate_request', lambda value: value)
    prior = tmp_path / 'prior'
    phase_path = prior / f'appearance-{round_index:02d}/blender_author_{round_index}.json'
    phase_path.parent.mkdir(parents=True)
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
    completion = {'run_id': 'same-run', 'capability': f'book_blender_author_{round_index}',
                  'provider': 'openai', 'model': 'gpt-6-astra',
                  'structured_output_digest': canonical_digest(program)}
    completion['inference_completion_digest'] = canonical_digest(completion, digest_field='inference_completion_digest')
    author.save_json(prior / 'request.json', request)
    author.save_json(phase_path, phase)
    author.save_json(prior / 'cad_result.json', {'readback': {'volume_mm3': 6000}})
    author.save_json(completed / 'receipt.json', completion)
    current = {**request, 'expected_production_commit': 'fixed-commit', 'request_digest': 'new-digest'}
    def adopt(value):
        return worker.verify_blender_program_adoption(prior_root=prior, request_value=value,
            budget_root=budget, round_index=round_index)
    output, receipt = adopt(current)
    assert output.program == program['program'] and receipt['new_provider_call'] is False
    assert receipt['cad_readback_digest'] == canonical_digest({'volume_mm3': 6000})
    with pytest.raises(author.AssetAuthoringError, match='source_inputs_changed'):
        adopt({**current,'dimensions_m':[2,2,3]})
    phase['output']['program'] += '\nCAD_BASE.rigid_body.mass=1000'
    author.save_json(phase_path, phase)
    with pytest.raises(author.AssetAuthoringError, match='completed_response_missing'):
        adopt(current)
