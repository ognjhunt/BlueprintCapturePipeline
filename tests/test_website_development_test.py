import copy
import json

import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_development_test import ENV, prepare_development_test, LABEL
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
