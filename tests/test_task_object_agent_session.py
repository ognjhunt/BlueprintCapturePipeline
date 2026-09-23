"""ADP-009B/day-21: real SDK tool repairs, independent acceptance and per-call spend."""
import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from agents.items import ModelResponse
from agents.models.interface import Model
from agents import ModelProvider
from agents.tool_context import ToolContext
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
from PIL import Image

from blueprint_pipeline import task_object_agent_session as session
from blueprint_pipeline import task_object_astra_authoring as author
from blueprint_pipeline import task_evaluation_scene_configuration_astra_driver as driver
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_object_agent_model import context_ceiling
from blueprint_pipeline.task_object_agent_tools import AssetTools
from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import AgentsSDKInvocationBlocked
from tests.test_astra_automatic_resume import authoring_fixture, FixtureInvoker  # noqa: F401
from tests.test_task_evaluation_scene_configuration_astra_driver import retained, component  # noqa: F401


def reply(value, index):
    return ModelResponse(output=[ResponseOutputMessage(id=f'msg_{index}', role='assistant', status='completed',
        type='message', content=[ResponseOutputText(type='output_text', text=json.dumps(value), annotations=[])])],
        usage=Usage(requests=1, input_tokens=500, output_tokens=300, total_tokens=800), response_id=f'resp_{index}')


class ScriptedModel(Model):
    def __init__(self, steps, physics):
        self.steps, self.physics, self.calls = list(steps), physics, []

    async def get_response(self, **kwargs):
        self.calls.append(kwargs)
        title = kwargs['output_schema'].json_schema()['title']
        if title == 'PhysicalPropertyReviewProposal':
            return reply(self.physics, len(self.calls))
        if title == 'AppearanceReview':
            return reply(dict(source_object_recognizable=True, source_color_and_material_preserved=True,
                opaque_surfaces_opaque=True, required_parts_present=True, no_obvious_geometry_artifacts=True,
                blockers=[], repair_instructions='', unobserved_surface_limitations=['underside']), len(self.calls))
        step = self.steps.pop(0)
        if isinstance(step, dict):
            return reply(step, len(self.calls))
        name, arguments = step
        return ModelResponse(output=[ResponseFunctionToolCall(id=f'fc_{len(self.calls)}', call_id=f'call_{len(self.calls)}',
            type='function_call', name=name, arguments=json.dumps(arguments), status='completed')],
            usage=Usage(requests=1, input_tokens=500, output_tokens=300, total_tokens=800),
            response_id=f'resp_{len(self.calls)}')

    def stream_response(self, *args, **kwargs):
        raise AssertionError('No streaming')


class Provider(ModelProvider):
    def __init__(self, model): self.model = model
    def get_model(self, model_name): return self.model


@pytest.fixture
def agent_fixture(authoring_fixture, monkeypatch):  # noqa: F811
    f = authoring_fixture
    path = Path(f.request.source_frames[0].path)
    Image.new('RGB', (64, 64), 'blue').save(path)
    f.request.source_frames[0].sha256 = author.file_record(path)['sha256']
    value = f.request.model_dump(mode='json')
    value['request_digest'] = canonical_digest(value, digest_field='request_digest')
    f.request = author.validate_request(value)
    brief = author.VisualBrief(object_identity='blue object', observed_parts=['body'], appearance_requirements=['blue'],
        unknown_regions=['underside'], cad_brief_markdown='Keep the supplied dimensions.', proposed_material='paper',
        proposed_appearance='opaque').model_dump(mode='json')
    fixture = FixtureInvoker(f.request, f.runtime / 'review-fixture')
    spec = SimpleNamespace(capability=f.request.object_id + '_physical_property_review', run_id=f.request.run_id)
    physics = fixture.invoke(spec, {'fixture': True}).output.model_dump(mode='json')
    base_blender = f.blender

    class Blender:
        def preflight(self): base_blender.preflight()
        def __call__(self, argv, **kwargs):
            result = base_blender(argv, **kwargs)
            for name in ('perspective.png', 'top.png', 'side.png'):
                Image.new('RGB', (96, 64), 'blue').save(Path(kwargs['cwd']) / name)
            return result

    def cad(*, program, output_root, request):
        f.executed.append(program)
        if program == 'broken':
            raise author.AssetAuthoringError('CAD compiler failed: unknown fillet radius')
        result = f.cad(brief='agent program', output_root=output_root, dimensions_m=request.dimensions_m)
        script = output_root / 'asset.py'
        script.write_text(program)
        result.update(program=author.file_record(script), execution='agent_program_through_pinned_cad_cli')
        return result

    monkeypatch.setenv('BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS', '1')
    f.kwargs = dict(request_value=value, output_root=f.runtime / 'authoring', budget_root=f.runtime / 'inference',
        cad_executor=cad, blender_runner=Blender(), blender_executable='fixture', authoring_instructions='Pinned CAD instructions.')
    f.steps = [('observe_object', {'brief': brief}), ('build_cad', {'program': 'broken'}),
        ('build_cad', {'program': 'repaired'}), ('render_candidate', {'program': dict(program='# valid fixture',
        explanation='blue material', generated_surface_assumptions=['underside'])}),
        ('inspect_candidate', {}), {'summary': 'Candidate ready for independent inspection.'}]
    f.model = ScriptedModel(f.steps, physics)
    return f


def bounded(f, cap=5):
    invoker, audit = author.budgeted_invoker(root=f.kwargs['budget_root'], run_id=f.request.run_id, maximum_cost_usd=cap)
    invoker._model_provider = Provider(f.model)
    return driver._StageInvoker(invoker, f.request.run_id, 15), audit


def test_real_sdk_keeps_images_error_and_repair_in_one_session(agent_fixture):
    f = agent_fixture
    invoker, audit = bounded(f)
    result = session.execute_agent_authoring(**f.kwargs, invoker=invoker, model=f.model)
    assert result['status'] == 'candidate_authored_pending_native_qualification'
    assert not result['scene_placement_qualified'] and not result['native_import_qualified']
    assert f.executed == ['broken', 'repaired', 'cad', 'blender']
    assert len(f.model.calls) == invoker.calls == 6  # four author turns and two independent reviewers
    assert 'unknown fillet radius' in json.dumps(f.model.calls[2]['input'])
    # The render hands off directly. Original and all three generated views go
    # to the independent reviewer before the author can invalidate the asset.
    review_input = json.dumps(f.model.calls[5]['input'])
    assert review_input.count('data:image/png;base64,') == 4
    assert f.model.calls[4]['output_schema'].json_schema()['title'] == 'PhysicalPropertyReviewProposal'
    assert f.model.calls[5]['output_schema'].json_schema()['title'] == 'AppearanceReview'
    manifest = audit.manifest()
    assert manifest['reserved_max_cost_usd'] < 5
    records = list((f.kwargs['budget_root'] / 'inference_reservations/reserved').glob('*.json'))
    assert len(records) == 6
    assert all(json.loads(p.read_text())['max_turns'] == 1 for p in records)
    assert (f.kwargs['budget_root'] / 'asset_session/conversation.sqlite').is_file()
    assert (f.kwargs['budget_root'] / 'asset_session/completion.json').is_file()


def test_render_hands_off_before_later_brief_mutation_can_clear_candidate(agent_fixture):
    f = agent_fixture
    # This sequence reproduces the drawer CPU failure: a valid render followed
    # by observe_object would clear asset.candidate before independent review.
    later_brief = f.steps[0]
    f.model.steps = f.steps[:4] + [later_brief]
    invoker, _ = bounded(f)
    result = session.execute_agent_authoring(**f.kwargs, invoker=invoker, model=f.model)
    assert result['status'] == 'candidate_authored_pending_native_qualification'
    assert f.model.steps == [later_brief]
    tools = list((f.kwargs['budget_root'] / 'asset_session/tools').glob('*.started.json'))
    assert len(tools) == 4


def test_budget_refuses_before_unaffordable_model_request(agent_fixture):
    f = agent_fixture
    invoker, audit = bounded(f, .0001)
    with pytest.raises(AgentsSDKInvocationBlocked, match='budget_ceiling'):
        session.execute_agent_authoring(**f.kwargs, invoker=invoker, model=f.model)
    assert not f.model.calls and not f.executed
    assert audit.manifest()['reserved_max_cost_usd'] == 0
    assert not (f.kwargs['output_root'] / 'result.json').exists()


def test_model_cannot_declare_an_unbuilt_asset_complete(agent_fixture):
    f = agent_fixture
    f.model.steps = [{'summary': 'It is complete, trust me.'}] * 3
    invoker, _ = bounded(f)
    with pytest.raises(author.AssetAuthoringError, match='independent_review_limit'):
        session.execute_agent_authoring(**f.kwargs, invoker=invoker, model=f.model)
    assert len(f.model.calls) == 3 and not f.executed
    assert not (f.kwargs['output_root'] / 'result.json').exists()


def test_unchanged_rejected_candidate_reuses_review_without_another_reservation(agent_fixture):
    f = agent_fixture
    class RejectedModel(ScriptedModel):
        async def get_response(self, **kwargs):
            if kwargs['output_schema'].json_schema()['title'] == 'AppearanceReview':
                self.calls.append(kwargs)
                return reply(dict(source_object_recognizable=True, source_color_and_material_preserved=False,
                    opaque_surfaces_opaque=True, required_parts_present=True, no_obvious_geometry_artifacts=True,
                    blockers=['Wrong color'], repair_instructions='Correct the blue material.',
                    unobserved_surface_limitations=['underside']), len(self.calls))
            return await super().get_response(**kwargs)
    f.model = RejectedModel(f.steps + [{'summary': 'No changes.'}] * 2, f.model.physics)
    invoker, _ = bounded(f)
    with pytest.raises(author.AssetAuthoringError, match='independent_review_limit'):
        session.execute_agent_authoring(**f.kwargs, invoker=invoker, model=f.model)
    titles = [call['output_schema'].json_schema()['title'] for call in f.model.calls]
    assert titles.count('AppearanceReview') == titles.count('PhysicalPropertyReviewProposal') == 1
    assert not (f.kwargs['output_root'] / 'result.json').exists()



def test_rebuilt_candidate_invalidates_rejected_review_cache(agent_fixture):
    f = agent_fixture
    class RepairModel(ScriptedModel):
        async def get_response(self, **kwargs):
            if kwargs['output_schema'].json_schema()['title'] == 'AppearanceReview' and not getattr(self, 'rejected', False):
                self.rejected = True
                self.calls.append(kwargs)
                return reply(dict(source_object_recognizable=True, source_color_and_material_preserved=False,
                    opaque_surfaces_opaque=True, required_parts_present=True, no_obvious_geometry_artifacts=True,
                    blockers=['Wrong color'], repair_instructions='Correct the blue material.',
                    unobserved_surface_limitations=['underside']), len(self.calls))
            return await super().get_response(**kwargs)
    f.model = RepairModel(f.steps + [f.steps[3], f.steps[4], f.steps[5]], f.model.physics)
    invoker, _ = bounded(f)
    result = session.execute_agent_authoring(**f.kwargs, invoker=invoker, model=f.model)
    assert result['status'] == 'candidate_authored_pending_native_qualification'
    titles = [call['output_schema'].json_schema()['title'] for call in f.model.calls]
    assert titles.count('AppearanceReview') == titles.count('PhysicalPropertyReviewProposal') == 2

def test_duplicate_tool_call_reuses_retained_outcome_and_rejects_changed_arguments(tmp_path):
    calls = []
    asset = SimpleNamespace(observe_object=lambda value: calls.append(value) or {'status': 'recorded'})
    tool = session.tool_definitions(asset, tmp_path)[0]
    context = ToolContext(context=None, tool_name=tool.name, tool_call_id='fixed', tool_arguments='{}')
    arguments = json.dumps({'brief': {'name': 'fixture'}})
    async def invoke():
        first = await tool.on_invoke_tool(context, arguments)
        assert await tool.on_invoke_tool(context, arguments) == first
        with pytest.raises(author.AssetAuthoringError, match='identity_changed'):
            await tool.on_invoke_tool(context, '{"brief": {"name": "different"}}')
    asyncio.run(invoke())
    assert len(calls) == 1


def test_image_cost_uses_dimensions_and_rejects_remote_context(tmp_path):
    path = tmp_path / 'frame.png'
    Image.new('RGB', (1920, 1080)).save(path)
    url = 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()
    assert context_ceiling([{'type': 'input_image', 'image_url': url}]) > 6500
    with pytest.raises(ValueError, match='source_invalid'):
        context_ceiling([{'type': 'input_image', 'image_url': 'https://unbounded.example/img'}])


def test_changed_candidate_rejected_before_any_review(agent_fixture):
    f = agent_fixture
    asset = AssetTools(**{k: v for k, v in f.kwargs.items() if k not in {'budget_root', 'authoring_instructions'}})
    asset.observe_object(f.steps[0][1]['brief'])
    asset.build_cad('repaired')
    asset.render_candidate(f.steps[3][1]['program'])
    Path(asset.cad['step']['path']).write_text('changed after render')
    with pytest.raises(author.AssetAuthoringError, match='changed_during_review'):
        asset.independent_review(SimpleNamespace(invoke=lambda *args: pytest.fail('must not call model')))


def test_independent_rejection_returns_to_same_conversation(agent_fixture):
    f = agent_fixture
    original = f.model.get_response
    reviews = []
    async def rejecting_first(**kwargs):
        if kwargs['output_schema'].json_schema()['title'] == 'AppearanceReview':
            reviews.append(True)
            if len(reviews) == 1:
                return reply(dict(source_object_recognizable=True, source_color_and_material_preserved=False,
                    opaque_surfaces_opaque=True, required_parts_present=True, no_obvious_geometry_artifacts=True,
                    blockers=['wrong blue tone'], repair_instructions='Darken the blue',
                    unobserved_surface_limitations=['underside']), 100)
        return await original(**kwargs)
    f.model.get_response = rejecting_first
    f.model.steps += [f.steps[3], f.steps[4], {'summary': 'Corrected the blue.'}]
    invoker, _ = bounded(f)
    result = session.execute_agent_authoring(**f.kwargs, invoker=invoker, model=f.model)
    assert result['status'] == 'candidate_authored_pending_native_qualification'
    assert len(reviews) == 2
    assert f.executed.count('cad') == 1 and f.executed.count('blender') == 2
    assert any('Darken the blue' in json.dumps(call['input']) for call in f.model.calls)


def test_shared_stage_call_cap_stops_tool_loop(agent_fixture):
    f = agent_fixture
    invoker, _ = bounded(f)
    invoker.maximum_calls = 2
    with pytest.raises(driver.AstraStageError, match='inference_boundary_refused'):
        session.execute_agent_authoring(**f.kwargs, invoker=invoker, model=f.model)
    assert len(f.model.calls) == 2
    assert f.executed == ['broken']


@pytest.mark.parametrize('resume', [False, True])
def test_website_driver_uses_sdk_session_inside_existing_cost_gate(component, monkeypatch, resume):  # noqa: F811
    path = Path(component.environment[driver._INPUT_ENV])
    value = json.loads(path.read_text())
    value['configuration']['source_observation_kind'] = 'website_capture_frames'
    value['configuration']['dimension_authority'] = 'estimated'
    monkeypatch.setattr('blueprint_pipeline.website_native_inputs.validate_website_authoring_disclosure', lambda **kw: None)
    path.write_text(json.dumps(value))
    if resume:
        monkeypatch.setattr(driver, 'prepare_phase_adoption', lambda **kw: {
            'authoring_kwargs': {'adopted_agent_root': '/fixture/retained-sdk'},
            'cad_kwargs': {}, 'prior_call_count': 7, 'adoption_digest': 'sha256:' + 'a' * 64,
            'retained_inference_cost_usd': 0.01})
    prior = component.kwargs.pop('authoring_executor')
    seen = []
    def execute(**kwargs):
        assert 'mac_executor' not in kwargs
        assert kwargs['budget_root'].name == 'inference'
        assert callable(kwargs['cad_executor'])
        assert kwargs.get('adopted_agent_root') == ('/fixture/retained-sdk' if resume else None)
        seen.append(True)
        return prior(**kwargs)
    monkeypatch.setattr(session, 'execute_agent_authoring', execute)
    result = driver.execute_astra_component(**component.kwargs)
    assert seen == [True] and result['status'] == 'completed'
    assert component.events[-4:] == ['reserve', 'sdk', 'complete', 'package']


def test_long_retained_session_compacts_before_reserved_request(agent_fixture):
    f = agent_fixture
    root = f.kwargs['budget_root'] / 'asset_session'
    root.mkdir(parents=True)
    history = session.SQLiteSession(f.request.object_id, db_path=root / 'conversation.sqlite')
    prior_messages = [
        {'role': 'user', 'content': 'Original task: retain uncertainty and blue material.'},
        {'type': 'function_call', 'call_id': 'old', 'name': 'build_cad', 'arguments': json.dumps({'program': 'x' * 90000})},
        {'type': 'function_call_output', 'call_id': 'old', 'output': 'obsolete compiler error'},
        {'type': 'function_call', 'call_id': 'latest', 'name': 'build_cad', 'arguments': json.dumps({'program': 'latest retained program'})},
        {'type': 'function_call_output', 'call_id': 'latest', 'output': 'latest compiler error'},
        {'role': 'user', 'content': 'Independent review: correct shape, no native qualification yet.'},
    ]
    asyncio.run(history.add_items(prior_messages))
    history.close()
    invoker, _ = bounded(f)
    result = session.execute_agent_authoring(**f.kwargs, invoker=invoker, model=f.model)
    assert result['status'] == 'candidate_authored_pending_native_qualification'
    first = f.model.calls[0]['input']
    assert 'x' * 90000 not in json.dumps(first)
    assert 'latest retained program' in json.dumps(first)
    assert 'latest compiler error' in json.dumps(first)
    assert 'correct shape' in json.dumps(first)
    assert 'data:image/png;base64,' in json.dumps(first)
    calls = {row['call_id'] for row in first if row.get('type') == 'function_call'}
    assert calls == {row['call_id'] for row in first if row.get('type') == 'function_call_output'}
    history = session.SQLiteSession(f.request.object_id, db_path=root / 'conversation.sqlite')
    assert 'x' * 90000 in json.dumps(asyncio.run(history.get_items()))
    history.close()
    assert len(f.model.calls) == 6  # no summarization model calls


def test_uncompactable_current_evidence_fails_before_model(agent_fixture):
    f = agent_fixture
    invoker, audit = bounded(f)
    f.kwargs['authoring_instructions'] = 'required' * 15000
    with pytest.raises(author.AssetAuthoringError, match='context_ceiling_exceeded'):
        session.execute_agent_authoring(**f.kwargs, invoker=invoker, model=f.model)
    assert not f.model.calls and not f.executed
    assert audit.manifest()['reserved_max_cost_usd'] == 0


def test_context_preserves_entire_current_reasoning_tool_turn():
    from blueprint_pipeline.task_object_agent_context import compact_authoring_history
    original = {'role': 'user', 'content': 'original task and source evidence'}
    older = [original,
        {'type': 'reasoning', 'id': 'old-reasoning', 'encrypted_content': 'old opaque state'},
        {'type': 'function_call', 'id': 'old-call', 'call_id': 'old', 'name': 'build_cad', 'arguments': '{"program":"exact prior program"}'},
        {'type': 'function_call_output', 'call_id': 'old', 'output': 'prior CAD readback'}]
    current = [
        {'role': 'user', 'content': 'independent review requires repair'},
        {'type': 'reasoning', 'id': 'current-reasoning', 'encrypted_content': 'required opaque state'},
        {'type': 'function_call', 'id': 'new-call', 'call_id': 'new', 'name': 'build_cad', 'arguments': '{"program":"new program"}'},
        {'type': 'function_call_output', 'call_id': 'new', 'output': 'current compiler error'}]
    rows = older + current
    before = json.dumps(rows)
    compact = compact_authoring_history(rows)
    assert compact[-len(current):] == current
    assert compact[0] == original
    assert 'exact prior program' in json.dumps(compact)
    assert 'old-reasoning' not in json.dumps(compact)
    assert not any(row.get('call_id') == 'old' for row in compact)
    assert json.dumps(rows) == before
    assert compact_authoring_history(older) == older  # No boundary: do not prune.
