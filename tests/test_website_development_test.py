import copy
import json

import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_development_test import ENV, prepare_development_test, LABEL, DRAWER_KIND, DRAWER_LABEL
from blueprint_pipeline.website_native_background import prepare_construction_stages
from tests.test_website_native_appearance import inputs


def setup(tmp_path, monkeypatch):
    args, original, _ = inputs(tmp_path)
    original['status'] = 'needs_input'
    original['blockers'] = ['support_surface_not_found_under_subject']
    original['support'] = None
    original['digest'] = canonical_digest(original, digest_field='digest')
    monkeypatch.setenv(ENV, json.dumps([args['task_context']['context_digest']]))
    return args, original


def test_authored_surface_keeps_original_refusal_dimensions_rights_budget_and_source(tmp_path, monkeypatch):
    args, original = setup(tmp_path, monkeypatch)
    before = copy.deepcopy(original)
    root = tmp_path / 'component-test'
    prepared, runtime = prepare_development_test(preparation=original, source_geometry=args['source_geometry'],
        task_masks=args['task_masks'], output_root=root)
    assert original == before
    assert prepared['status'] == 'intake_ready'
    assert prepared['development_test']['captured_scene_evaluation_allowed'] is False
    assert prepared['development_test']['label'] == LABEL
    assert prepared['intake_request']['execution'] == original['intake_request']['execution']
    assert prepared['intake_request']['submission_id'] == original['intake_request']['submission_id']
    assert prepared['physics'] == original['physics']
    assert prepared['intake_request']['source']['kind'] == 'mesh'
    assert 'splat_digest' not in prepared['binding']
    assert runtime['simulator_ready'] is False
    assert runtime['object_authoring']['configuration']['scene_id'].endswith('-development')
    np.testing.assert_allclose(np.subtract(prepared['subject']['aabb_max_xyz'], prepared['subject']['aabb_min_xyz']),
                               np.subtract(original['subject']['aabb_max_xyz'], original['subject']['aabb_min_xyz']))
    stages = prepare_construction_stages(runtime_inputs_path=root / 'runtime_inputs.json', preparation_path=root / 'preparation.json')
    assert len(stages['stage_sequence']) == 6
    assert stages['configurations'][-1]['support_plane']['authority'] == 'authored_development_surface'
    # Real source views survive the normal native authoring transport contract.
    assert any('.frames.' in row['contract_path'] for row in stages['references'])


@pytest.mark.parametrize('problem', ['not_authorized', 'rights', 'changed'])
def test_no_automatic_fallback_or_bypass_of_unrelated_refusals(tmp_path, monkeypatch, problem):
    args, original = setup(tmp_path, monkeypatch)
    if problem == 'not_authorized':
        monkeypatch.delenv(ENV)
    elif problem == 'rights':
        original['blockers'].append('website_scene_processing_rights_required')
        original['digest'] = canonical_digest(original, digest_field='digest')
    else:
        original['physics']['basis'] = 'measured'
    with pytest.raises(ValueError, match='not_authorized|not_admitted|source_changed'):
        prepare_development_test(preparation=original, source_geometry=args['source_geometry'],
            task_masks=args['task_masks'], output_root=tmp_path / 'no-output')


def test_development_surface_reaches_real_native_submission_with_separate_identity(tmp_path, monkeypatch):
    from pathlib import Path
    from tests.test_website_native_submission import setup as native_setup, SHA
    from blueprint_pipeline.website_native_submission import materialize_website_submission
    from blueprint_pipeline.task_evaluation_scene_configuration_submission_publication import _validated_inventory
    kwargs, _ = native_setup(tmp_path, monkeypatch, development=True)
    result = materialize_website_submission(**kwargs)
    assert materialize_website_submission(**kwargs) == result
    root = kwargs['staging_root']
    _validated_inventory(root, SHA)
    request = json.loads((root / 'scene_configuration_preparation_request.v1.json').read_text())
    assert request['scene']['identity']['id'].endswith('-development')
    assert request['scene']['appearance']['kind'] == 'textured_usd'
    assert request['replacement_authoring_backend'] == 'astra_cad_blender_v1'
    assert request['spend']['external_service_caps']['openai']['stage_max_cost_usd']['content_agents'] >= 5
    assert request['spend']['hard_cap_usd'] <= 20
    assert 'public_display_authorization' not in request
    task = json.loads((root / 'configuration/task.json').read_text())
    assert task['test_environment']['captured_scene_evaluation_allowed'] is False
    assert task['instruction'].startswith(LABEL)
    assert task['physical_world_truth_claimed'] is False
    # Original source task context is retained, not re-confirmed or relabeled.
    context = json.loads(Path(kwargs['task']['task_context']['path']).read_text())
    assert context['scene_id'] != request['scene']['identity']['id']


def test_development_scope_cannot_be_stripped_or_published_as_capture(tmp_path, monkeypatch):
    from blueprint_pipeline.website_development_test import environment
    args, original = setup(tmp_path, monkeypatch)
    value, _ = prepare_development_test(preparation=original, source_geometry=args['source_geometry'],
        task_masks=args['task_masks'], output_root=tmp_path / 'development')
    del value['development_test']
    with pytest.raises(ValueError, match='binding_invalid'):
        environment(value)


def test_registered_room_failure_does_not_block_authorized_object_preparation(tmp_path, monkeypatch):
    from blueprint_pipeline import website_task_preparation as compiler
    from tests.test_website_task_preparation import _arguments
    args = _arguments(tmp_path)
    monkeypatch.setenv(ENV, json.dumps([args['task_context']['context_digest']]))
    def refused(**_):
        raise ValueError('website_registration_conflicts_provider_anchor')
    monkeypatch.setattr(compiler, 'register_source_to_runtime', refused)
    original = compiler.compile_website_scene_preparation(**args)
    assert original['status'] == 'needs_input'
    assert 'website_registration_conflicts_provider_anchor' in original['blockers']
    assert original['support'] is None
    assert original['registration']['physical_registration_proven'] is False
    assert original['coordinate_frame']['declared_meters_per_unit'] == 1
    assert original['registration']['basis'] == 'estimated_masked_object_principal_axes'
    prepared, runtime = prepare_development_test(preparation=original, source_geometry=args['source_geometry'],
        task_masks=args['task_masks'], output_root=tmp_path / 'independent-test')
    assert prepared['status'] == 'intake_ready'
    assert prepared['development_test']['captured_scene_evaluation_allowed'] is False
    assert prepared['development_test']['source_scene_blockers'] == original['blockers']
    assert prepared['intake_request']['execution'] == original['intake_request']['execution']
    assert runtime['simulator_ready'] is False
    before = np.subtract(prepared['subject']['aabb_max_xyz'], prepared['subject']['aabb_min_xyz'])
    # Marble's scale must not size the independently observed development object.
    args['base_scene']['meters_per_unit'] = 500.0
    again = compiler.compile_website_scene_preparation(**args)
    np.testing.assert_allclose(before, np.subtract(again['subject']['aabb_max_xyz'], again['subject']['aabb_min_xyz']))


def test_missing_marble_anchor_uses_separately_named_drawer_fixture(tmp_path, monkeypatch):
    from blueprint_pipeline import website_task_preparation as compiler
    from tests.test_website_task_preparation import _arguments, _masks, ARTICULATED_REMOVAL
    args = _arguments(tmp_path)
    args['task_masks'] = _masks(args['source_geometry'], destination=False, articulated=True)
    args['removal_manifest'] = ARTICULATED_REMOVAL
    monkeypatch.setenv(ENV, json.dumps([args['task_context']['context_digest']]))
    def refused(**_):
        raise ValueError('website_registration_anchor_frame_missing')
    monkeypatch.setattr(compiler, 'register_source_to_runtime', refused)
    original = compiler.compile_website_scene_preparation(**args)
    assert original['status'] == 'needs_input'
    assert original['registration']['captured_scene_integration'] == 'pending'
    assert 'website_registration_anchor_frame_missing' in original['blockers']
    prepared, runtime = prepare_development_test(preparation=original, source_geometry=args['source_geometry'],
        task_masks=args['task_masks'], output_root=tmp_path / 'drawer-fixture')
    task = prepared['intake_request']['task']
    assert prepared['development_test']['kind'] == DRAWER_KIND
    assert prepared['development_test']['label'] == DRAWER_LABEL
    assert prepared['development_test']['captured_scene_evaluation_allowed'] is False
    assert task['strategy'] == 'articulated_open_close' and 'destination' not in task
    assert task['articulation']['part_label'] == 'middle drawer'
    assert prepared['destination'] is None
    assert runtime['simulator_ready'] is False
    assert runtime['object_authoring']['configuration']['schema_version'] == 'articulated_replacement_authoring_configuration.v1'


@pytest.mark.parametrize('authorized,reason', [
    (False, 'website_registration_conflicts_provider_anchor'),
    (True, 'website_registration_anchor_frame_invalid'),
])
def test_independent_object_frame_cannot_bypass_scope_or_corrupt_input(tmp_path, monkeypatch, authorized, reason):
    from blueprint_pipeline import website_task_preparation as compiler
    from tests.test_website_task_preparation import _arguments
    args = _arguments(tmp_path)
    monkeypatch.setenv(ENV, json.dumps([args['task_context']['context_digest']] if authorized else []))
    def refused(**_):
        raise ValueError(reason)
    monkeypatch.setattr(compiler, 'register_source_to_runtime', refused)
    with pytest.raises(ValueError, match=reason):
        compiler.compile_website_scene_preparation(**args)


@pytest.mark.parametrize('unit', [None, 'arbitrary', 'millimeters'])
def test_independent_object_frame_requires_estimated_metric_source(tmp_path, unit):
    from blueprint_pipeline.website_object_local_frame import estimated_object_frame
    from tests.test_website_task_preparation import _arguments
    args = _arguments(tmp_path)
    source = {**args['source_geometry'], 'unit': unit}
    with pytest.raises(ValueError, match='website_object_frame_scale_invalid'):
        estimated_object_frame(track=args['task_masks']['targets'][0]['track'],
            source_geometry=source, registration_blocker='website_registration_conflicts_provider_anchor')
