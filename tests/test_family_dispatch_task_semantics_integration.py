"""Source-bound derived routing reaches canonical family dispatch without owner rewrites."""
from __future__ import annotations

import copy
import json

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest, cross_runtime_canonical_digest, cross_runtime_canonical_json,
)
from blueprint_pipeline.task_evaluation_scene_intake import stage_scene_intent
from blueprint_pipeline.website_task_context_contracts import retains_task_family_preparation
from blueprint_pipeline.website_task_semantics import retain_task_semantics
from tests.test_task_family_preparation import _evaluation_inputs
from tests.test_task_semantics_catalog_composition import catalog_semantics_fixture
from tests.test_website_preparation_contract_v2 import VECTOR
from tests.test_website_task_family_packet import family_preparation


@pytest.mark.parametrize("authoring_provider", ["openai", "anthropic"])
def test_source_bound_general_task_dispatches_unconfirmed_family_under_retained_authority(tmp_path, monkeypatch, authoring_provider):
    from blueprint_pipeline.website_scene_handoff import prepare_website_scene_handoff

    args, _old_preparation, _runtime = family_preparation(tmp_path)
    context = args['task_context']
    context.update(task_family='general_task', description='Press the start button.',
        purpose='scene_preparation', confirmed=False, confirmed_at=None,
        capture_rights=copy.deepcopy(VECTOR['task_context_response_preparation']['capture_rights']))
    context['context_digest'] = canonical_digest(context, digest_field='context_digest')
    _other_context, source_video_digest, semantic_template = catalog_semantics_fixture('button_press')
    semantic_receipt = copy.deepcopy(semantic_template['planner_receipt'])
    semantic_receipt['binding']['task_context_digest'] = context['context_digest']
    semantic_receipt['binding_digest'] = canonical_digest(semantic_receipt['binding'])
    geometry = args['source_geometry']
    geometry['binding'] = {'source_video_digest': source_video_digest}
    geometry['digest'] = canonical_digest(geometry, digest_field='digest')
    args['task_masks']['source_geometry_digest'] = geometry['digest']
    args['task_masks']['digest'] = canonical_digest(args['task_masks'], digest_field='digest')
    original = args['evaluation_inputs']
    evaluation = _evaluation_inputs(args)
    evaluation['native_runtime_inputs']['assets'] = copy.deepcopy(original['native_runtime_inputs']['assets'])
    evaluation['asset_sources'] = copy.deepcopy(original['asset_sources'])
    evaluation['task_family_runtime_selection'] = original['task_family_runtime_selection']
    args['base_scene']['splat_binding_id'] = 'website-splat-' + args['base_scene']['splat_digest'][7:39]
    grant = copy.deepcopy(VECTOR['sponsorship_response'])
    grant.update({key: context[key] for key in ('request_id', 'scene_id', 'capture_id')})
    grant['task_context_digest'] = context['context_digest']
    grant['consent']['rights_reference'] = cross_runtime_canonical_digest(context['capture_rights'])
    grant['consent']['accepted_at_epoch'] = args['now'] - 1
    grant['expires_at_epoch'] = args['now'] + 3600
    grant['assessment_preparation_proposal'].update({key: context[key] for key in ('request_id', 'capture_id')})
    grant['authoring_provider'] = authoring_provider
    if authoring_provider == 'anthropic':
        grant['anthropic_provider_terms_reference'] = 'sha256:' + 'e' * 64
    grant['authority_digest'] = canonical_digest(grant, digest_field='authority_digest')
    original_context = copy.deepcopy(context)
    pipeline = tmp_path / 'pipeline'
    write_json(pipeline / 'atlas/selection_binding.json', {'planner_receipt': semantic_receipt})
    semantics = retain_task_semantics(job_dir=pipeline / 'atlas', receipt=semantic_receipt,
        task_context=context, source_video_digest=source_video_digest)
    removal = pipeline / 'removal.json'
    write_json(removal, args['removal_manifest'])
    monkeypatch.setattr('blueprint_pipeline.website_atlas_scene_source.atlas_scene_source', lambda **_: {
        'status': 'ready', 'base_scene': args['base_scene'], 'source_geometry': geometry,
        'source_pose_digest': 'sha256:' + 'b' * 64, 'background_pose_digest': 'sha256:' + 'c' * 64,
        'alignment': 'anchored'})
    monkeypatch.setattr('blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry', lambda **kw: kw['task_masks'])
    queue = tmp_path / 'intents'
    monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT', str(queue))
    monkeypatch.setenv('BLUEPRINT_WEBSITE_SCENE_BINDING_ROOT', str(tmp_path / 'bindings'))
    accepted, reads = {}, []

    def website_transport(*, capture_id, operation, payload):
        assert capture_id == context['capture_id']
        if operation == 'task-context':
            reads.append(operation)
            return copy.deepcopy(context)
        if operation == 'scene-sponsorship':
            reads.append(operation)
            assert payload['create'] is False
            return copy.deepcopy(grant)
        assert operation == 'prepared-scene'
        request = json.loads(cross_runtime_canonical_json(payload['request']))
        receipt = stage_scene_intent(value=request, queue_root=queue,
            authenticated_client='blueprint-webapp', trusted_clients={'blueprint-webapp'}, now=args['now'])
        accepted.update(receipt)
        return {'id': receipt['intent_id'], 'state': 'forward_pending',
            'request_digest': cross_runtime_canonical_digest(request)}

    monkeypatch.setattr('blueprint_pipeline.website_task_context.website_webapp_request', website_transport)
    result = prepare_website_scene_handoff(descriptor={'scene_id': context['scene_id'],
        'capture_id': context['capture_id'], 'metadata': {'site_task_context': context,
            'website_scene_execution_authority': grant, 'native_task_evaluation': evaluation}},
        clean_plate={'privacy_verified': True, 'status': 'objects_removed', 'removal_manifest_path': str(removal),
            'task_masks': args['task_masks']}, capture_root=tmp_path, now=args['now'])
    assert result['status'] == 'intake_ready', result
    assert result['source_registration'] and result['website_intake_outbox']['id'] == accepted['intent_id']
    assert context == original_context and context['task_family'] == 'general_task' and context['confirmed'] is False
    assert result['task_evaluation_inputs']['task_family'] == 'button_press'
    assert result['task_evaluation_inputs']['task_semantics'] == semantics
    assert result['task_evaluation_inputs']['source_video_digest'] == source_video_digest
    assert reads == ['task-context', 'scene-sponsorship']
    assert retains_task_family_preparation(result, context)
    intent = json.loads((queue / accepted['intent_id'] / 'intent.json').read_text())
    assert intent['request']['task']['strategy'] == 'button_press'
    assert intent['request']['consent']['task_confirmed'] is False
    assert intent['request']['execution']['policy_candidates'] == []
    assert intent['request']['execution']['allowed_providers'] == ['vast', 'openai'] + (
        ['anthropic'] if authoring_provider == 'anthropic' else [])
