"""Resume the real SDK loop after a completed candidate, without repeating tools."""
import json

import pytest

from blueprint_pipeline import task_object_agent_session as session
from blueprint_pipeline import task_evaluation_scene_configuration_astra_phase_adoption as adoption
from blueprint_pipeline import task_evaluation_scene_configuration_astra_driver as driver
from blueprint_pipeline.task_object_agent_resume import inspect_agent_candidate
from blueprint_pipeline.task_object_astra_authoring import AssetAuthoringError, validate_request
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_task_object_agent_session import (  # noqa: F401
    agent_fixture, bounded, ScriptedModel, reply,
    authoring_fixture, retained, component,
)


def interrupted(f):
    invoker, audit = bounded(f)
    invoker.maximum_calls = 5  # render handoff and physics; final visual review is next
    with pytest.raises(driver.AstraStageError, match='inference_boundary_refused'):
        session.execute_agent_authoring(**f.kwargs, invoker=invoker, model=f.model)
    assert audit.manifest()['reservation_count'] == 5
    assert audit.manifest()['in_flight_unknown_count'] == 0
    return audit.manifest()


def prepare(f, output):
    descriptor = adoption.materialize_automatic_phase_adoption(prior_runtime=f.runtime)
    return adoption.prepare_phase_adoption(value=descriptor, request_value=f.request.model_dump(mode='json'),
        package=f.package, budget_root=output / 'inference')


def test_resume_calls_only_missing_reviewer_and_preserves_spend_history(agent_fixture):  # noqa: F811
    f = agent_fixture
    before = interrupted(f)
    original_executions = list(f.executed)
    snapshot = adoption._inventory(f.runtime)
    output = f.runtime.parent / 'resume-agent'
    prepared = prepare(f, output)
    f.kwargs.update(output_root=output / 'authoring', budget_root=output / 'inference')
    f.model = ScriptedModel([], f.model.physics)
    invoker, audit = bounded(f)
    invoker.prior_calls = prepared['prior_call_count']
    result = session.execute_agent_authoring(**f.kwargs, **prepared['authoring_kwargs'], invoker=invoker, model=f.model)
    assert result['status'] == 'candidate_authored_pending_native_qualification'
    assert f.executed == original_executions
    assert len(f.model.calls) == 1
    assert f.model.calls[0]['output_schema'].json_schema()['title'] == 'AppearanceReview'
    assert audit.manifest()['reservation_count'] == before['reservation_count'] + 1
    assert audit.manifest()['reserved_max_cost_usd'] > before['reserved_max_cost_usd']
    assert prepared['retained_inference_cost_usd'] == before['reserved_max_cost_usd']
    assert adoption._inventory(f.runtime) == snapshot


@pytest.mark.parametrize('change', ['program', 'usd', 'physics', 'history', 'render', 'wal', 'cad_program', 'source_evidence'])
def test_changed_candidate_refuses_before_new_model_or_tool_calls(agent_fixture, change):  # noqa: F811
    f = agent_fixture
    interrupted(f)
    if change == 'program':
        (f.runtime / 'authoring/appearance-00/asset_program.py').write_text('changed')
    elif change == 'usd':
        (f.runtime / 'authoring/appearance-00/candidate.usdc').write_bytes(b'changed')
    elif change == 'physics':
        path = f.runtime / 'authoring/physical_property_review_result.json'
        value = json.loads(path.read_text())
        value['accepted']['properties']['mass_kg']['value'] *= 2
        path.write_text(json.dumps(value))
    elif change == 'render':
        (f.runtime / 'authoring/appearance-00/top.png').write_bytes(b'changed')
    elif change == 'wal':
        (f.runtime / 'inference/asset_session/conversation.sqlite-wal').write_bytes(b'uncheckpointed')
    elif change == 'cad_program':
        (f.runtime / 'authoring/cad-01/asset.py').write_text('changed')
    elif change == 'source_evidence':
        path = f.runtime / 'authoring/physical_review_input.json'
        value = json.loads(path.read_text())
        value['evidence'][-1]['sha256'] = '0' * 64
        path.write_text(json.dumps(value))
    else:
        path = next((f.runtime / 'inference/asset_session/tools').glob('*.started.json'))
        value = json.loads(path.read_text())
        value['arguments'] = {}
        path.write_text(json.dumps(value))
    with pytest.raises((AssetAuthoringError, ValueError), match='agent_resume'):
        inspect_agent_candidate(f.runtime, f.request.model_dump(mode='json'))


def test_resume_preserves_parent_request_limit_before_review(agent_fixture):  # noqa: F811
    f = agent_fixture
    interrupted(f)
    output = f.runtime.parent / 'budget-refusal'
    prepared = prepare(f, output)
    f.kwargs.update(output_root=output / 'authoring', budget_root=output / 'inference')
    f.model = ScriptedModel([], f.model.physics)
    invoker, _ = bounded(f)
    invoker.prior_calls = prepared['prior_call_count']
    invoker.maximum_calls = prepared['prior_call_count']
    with pytest.raises(driver.AstraStageError, match='inference_boundary_refused'):
        session.execute_agent_authoring(**f.kwargs, **prepared['authoring_kwargs'], invoker=invoker, model=f.model)
    assert not f.model.calls


def test_accepted_sdk_candidate_is_adopted_without_another_model_call(agent_fixture):  # noqa: F811
    f = agent_fixture
    invoker, audit = bounded(f)
    result = session.execute_agent_authoring(**f.kwargs, invoker=invoker, model=f.model)
    original_calls = len(f.model.calls)
    prepared = prepare(f, f.runtime.parent / 'accepted-resume')
    assert prepared['completed_authoring_result']['status'] == result['status']
    assert prepared['authoring_kwargs'] == {}
    assert prepared['prior_call_count'] == audit.manifest()['reservation_count']
    assert len(f.model.calls) == original_calls


def test_candidate_can_resume_again_after_release_change_and_budget_refusal(agent_fixture):  # noqa: F811
    f = agent_fixture
    interrupted(f)
    value = f.request.model_dump(mode='json')
    value['expected_production_commit'] = 'a' * 40
    value['request_digest'] = canonical_digest(value, digest_field='request_digest')
    f.request = validate_request(value)
    output = f.runtime.parent / 'new-release-budget-refusal'
    prepared = prepare(f, output)
    f.kwargs.update(request_value=value, output_root=output / 'authoring', budget_root=output / 'inference')
    f.model = ScriptedModel([], f.model.physics)
    invoker, _ = bounded(f)
    invoker.prior_calls = invoker.maximum_calls = prepared['prior_call_count']
    with pytest.raises(driver.AstraStageError, match='inference_boundary_refused'):
        session.execute_agent_authoring(**f.kwargs, **prepared['authoring_kwargs'], invoker=invoker, model=f.model)
    state = inspect_agent_candidate(output, value)
    assert state['author_calls'] == 4
    assert not f.model.calls


def test_completed_review_can_finish_result_without_repeating_inference(agent_fixture):  # noqa: F811
    f = agent_fixture
    invoker, _ = bounded(f)
    session.execute_agent_authoring(**f.kwargs, invoker=invoker, model=f.model)
    (f.runtime / 'authoring/result.json').unlink()  # crash after review, before result publication
    (f.runtime / 'inference/asset_session/completion.json').unlink()
    output = f.runtime.parent / 'completed-review-resume'
    prepared = prepare(f, output)
    f.kwargs.update(output_root=output / 'authoring', budget_root=output / 'inference')
    f.model = ScriptedModel([], f.model.physics)
    invoker, _ = bounded(f)
    invoker.prior_calls = prepared['prior_call_count']
    result = session.execute_agent_authoring(**f.kwargs, **prepared['authoring_kwargs'], invoker=invoker, model=f.model)
    assert result['status'] == 'candidate_authored_pending_native_qualification'
    assert not f.model.calls


def test_retained_rejection_returns_to_author_without_repeating_review(agent_fixture):  # noqa: F811
    f = agent_fixture
    original = f.model.get_response
    async def reject(**kwargs):
        if kwargs['output_schema'].json_schema()['title'] == 'AppearanceReview':
            return reply(dict(source_object_recognizable=True, source_color_and_material_preserved=False,
                opaque_surfaces_opaque=True, required_parts_present=True, no_obvious_geometry_artifacts=True,
                blockers=['wrong blue'], repair_instructions='Darken the blue', unobserved_surface_limitations=[]), 100)
        return await original(**kwargs)
    f.model.get_response = reject
    invoker, _ = bounded(f)
    invoker.maximum_calls = 6  # completed rejection; next author call cannot start
    with pytest.raises(driver.AstraStageError, match='inference_boundary_refused'):
        session.execute_agent_authoring(**f.kwargs, invoker=invoker, model=f.model)
    output = f.runtime.parent / 'rejected-review-resume'
    prepared = prepare(f, output)
    f.kwargs.update(output_root=output / 'authoring', budget_root=output / 'inference')
    f.model = ScriptedModel([f.steps[3]], f.model.physics)
    resumed_response = f.model.get_response
    async def unique_call_ids(**kwargs):
        response = await resumed_response(**kwargs)
        for item in response.output:
            if item.type == 'function_call':
                item.call_id = 'resumed_' + item.call_id
        return response
    f.model.get_response = unique_call_ids
    invoker, _ = bounded(f)
    invoker.prior_calls = prepared['prior_call_count']
    result = session.execute_agent_authoring(**f.kwargs, **prepared['authoring_kwargs'], invoker=invoker, model=f.model)
    assert result['status'] == 'candidate_authored_pending_native_qualification'
    assert 'Darken the blue' in json.dumps(f.model.calls[0]['input'])
    assert len(f.model.calls) == 3  # author render and two new reviews
    assert f.executed.count('cad') == 1


def test_controller_successor_resumes_sdk_candidate_with_original_costs(agent_fixture):  # noqa: F811
    from blueprint_pipeline import task_evaluation_partial_astra_successor as partial
    from blueprint_pipeline.task_object_astra_inherited_inference import inherited_balance
    f = agent_fixture
    before = interrupted(f)
    executions = list(f.executed)
    old = f.request.model_dump(mode='json')
    new = dict(old, run_id=old['run_id'] + '-successor', expected_production_commit='b' * 40)
    new['request_digest'] = canonical_digest(new, digest_field='request_digest')
    binding = json.loads((f.runtime / 'stage_source_binding.json').read_text())
    # The SDK fixture replaces the source PNG; bind that fixture image just as the driver does.
    binding['authoring_input_digest'] = canonical_digest({k: v for k, v in old.items()
        if k not in {'request_digest', 'expected_production_commit'}})
    binding['binding_digest'] = canonical_digest(binding, digest_field='binding_digest')
    (f.runtime / 'stage_source_binding.json').write_text(json.dumps(binding))
    identity = dict(source_run_id=old['run_id'], successor_run_id=new['run_id'], owner_id='owner',
                    stable_intent_id='intent', stable_intent_digest='sha256:' + 'c' * 64)
    descriptor = dict(schema_version=partial.SCHEMA_VERSION, source_run_id=old['run_id'],
        successor_run_id=new['run_id'], source_request_digest=old['request_digest'],
        original_runtime_root=str(f.runtime), semantic_request_digest=canonical_digest(partial.semantic_request(new)),
        source_stage_binding_digest=binding['binding_digest'], owner_intent_lineage=identity,
        retained_files=adoption._inventory(f.runtime))
    descriptor['adoption_digest'] = canonical_digest(descriptor, digest_field='adoption_digest')
    output = f.runtime.parent / 'controller-successor'
    prepared = partial.prepare_partial_astra_successor(value=descriptor, request_value=new,
        source_binding=binding, verified_lineage=identity, package=f.package, budget_root=output / 'inference')
    inherited = inherited_balance(output / 'inference', new['run_id'])
    assert inherited['call_count'] == before['reservation_count']
    assert inherited['cost_usd'] == before['reserved_max_cost_usd']
    f.request = validate_request(new)
    f.kwargs.update(request_value=new, output_root=output / 'authoring', budget_root=output / 'inference')
    f.model = ScriptedModel([], f.model.physics)
    invoker, audit = bounded(f)
    invoker.prior_calls = prepared['prior_call_count']
    result = session.execute_agent_authoring(**f.kwargs, **prepared['authoring_kwargs'], invoker=invoker, model=f.model)
    assert result['status'] == 'candidate_authored_pending_native_qualification'
    assert f.executed == executions
    assert len(f.model.calls) == 1
    assert audit.manifest()['reservation_count'] == 1
    assert inspect_agent_candidate(output, new)['completed_result']['result_digest'] == result['result_digest']
    assert adoption._inventory(f.runtime) == descriptor['retained_files']


def test_obsolete_rejection_gets_current_review_without_rebuilding(agent_fixture, monkeypatch):  # noqa: F811
    from blueprint_pipeline import task_object_agent_tools as tools
    f = agent_fixture
    original_invoke = tools.invoke_vision
    def old_scope(*args, capability, **kwargs):
        return original_invoke(*args, capability=capability.removesuffix('_' + tools.APPEARANCE_SCOPE), **kwargs)
    monkeypatch.setattr(tools, 'invoke_vision', old_scope)
    original_model = f.model.get_response
    async def old_rejection(**kwargs):
        if kwargs['output_schema'].json_schema()['title'] == 'AppearanceReview':
            return reply(dict(source_object_recognizable=False, source_color_and_material_preserved=True,
                opaque_surfaces_opaque=True, required_parts_present=True, no_obvious_geometry_artifacts=True,
                blockers=['Missing native import proof'], repair_instructions='Provide simulator proof',
                unobserved_surface_limitations=['underside']), 100)
        return await original_model(**kwargs)
    f.model.get_response = old_rejection
    invoker, audit = bounded(f)
    invoker.maximum_calls = 6
    with pytest.raises(driver.AstraStageError, match='inference_boundary_refused'):
        session.execute_agent_authoring(**f.kwargs, invoker=invoker, model=f.model)
    calls, executions = audit.manifest()['reservation_count'], list(f.executed)
    snapshot = adoption._inventory(f.runtime)
    monkeypatch.setattr(tools, 'invoke_vision', original_invoke)
    state = inspect_agent_candidate(f.runtime, f.request.model_dump(mode='json'))
    assert state['retained_visual_review'] is None
    assert state['current_scope_visual_reviews'] == 0
    assert state['completed_visual_reviews'] == 1
    output = f.runtime.parent / 'current-review-scope'
    prepared = prepare(f, output)
    f.kwargs.update(output_root=output / 'authoring', budget_root=output / 'inference')
    f.model = ScriptedModel([], f.model.physics)
    invoker, audit = bounded(f)
    invoker.prior_calls = prepared['prior_call_count']
    result = session.execute_agent_authoring(**f.kwargs, **prepared['authoring_kwargs'], invoker=invoker, model=f.model)
    assert result['status'] == 'candidate_authored_pending_native_qualification'
    assert len(f.model.calls) == 1
    assert f.model.calls[0]['output_schema'].json_schema()['title'] == 'AppearanceReview'
    assert f.executed == executions
    assert audit.manifest()['reservation_count'] == calls + 1
    assert adoption._inventory(f.runtime) == snapshot
