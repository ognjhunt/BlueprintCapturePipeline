"""Retained family dispatch reopens the original source-bound operation order."""
from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_family_scoring import seal_task_family_contract
from blueprint_pipeline.website_task_family_packet import family_source_references
from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
from blueprint_pipeline.website_task_semantics import task_semantics_from_receipt
from tests.test_task_family_preparation import _evaluation_inputs
from tests.test_task_semantics_catalog_composition import catalog_semantics_fixture
from tests.test_website_task_family_packet import family_preparation


def semantic_source(tmp_path, *, multi=False):
    args, _old, runtime_path = family_preparation(tmp_path)
    family = 'multi_step' if multi else 'button_press'
    template, video, artifact = catalog_semantics_fixture(family,
        operations=['button_press', 'wiping'] if multi else ['button_press'])
    context = args['task_context']
    context.update(task_family='general_task', description=template['description'])
    context['context_digest'] = canonical_digest(context, digest_field='context_digest')
    receipt = copy.deepcopy(artifact['planner_receipt'])
    receipt['binding']['task_context_digest'] = context['context_digest']
    receipt['binding_digest'] = canonical_digest(receipt['binding'])
    artifact = task_semantics_from_receipt(receipt=receipt, task_context=context, source_video_digest=video)
    geometry = args['source_geometry']
    geometry['binding'] = {'source_video_digest': video}
    geometry['digest'] = canonical_digest(geometry, digest_field='digest')
    args['task_masks']['source_geometry_digest'] = geometry['digest']
    args['task_masks']['digest'] = canonical_digest(args['task_masks'], digest_field='digest')
    old_evaluation = args['evaluation_inputs']
    evaluation = _evaluation_inputs(args)
    evaluation['native_runtime_inputs']['assets'] = old_evaluation['native_runtime_inputs']['assets']
    evaluation['asset_sources'] = old_evaluation['asset_sources']
    evaluation['task_family_runtime_selection'] = old_evaluation['task_family_runtime_selection']
    if multi:
        button = evaluation['task_success_contract']
        evaluation['task_success_contract'] = seal_task_family_contract(task_family='multi_step',
            scope=button['scope'], runtime_identity_digest=button['runtime_identity_digest'],
            stages=[button['stages'][0], {'stage_id': 'stage2', 'task_family': 'wiping', 'minimum_hold_seconds': 0,
                'criteria': {'minimum_cleaned_fraction': .5, 'maximum_remaining_dirty_area_m2': 1}}],
            reset={**button['reset'], 'stage2': {'dirty_area_m2': 2}})
        evaluation['native_runtime_inputs']['task_state_binding']['stage_bindings']['stage2'] = {
            'native_readback_kind': 'wiping', 'asset_ids': ['button1'], 'asset_roles': {'surface': 'button1'},
            'measurement_geometry': {'cell_areas_m2': [1, 1]}}
    args.update(evaluation_inputs=evaluation, task_semantics=artifact)
    preparation = compile_website_scene_preparation(**args)
    assert preparation['status'] == 'intake_ready', preparation
    write_json(runtime_path, preparation['task_evaluation_inputs']['native_runtime_contract'])
    return preparation, runtime_path, context


def test_retained_inferred_family_requires_original_context_and_unchanged_source_binding(tmp_path):
    preparation, runtime_path, context = semantic_source(tmp_path)
    assert family_source_references(preparation=preparation, runtime_inputs_path=runtime_path, task_context=context)
    with pytest.raises(ValueError, match='semantics_context_required'):
        family_source_references(preparation=preparation, runtime_inputs_path=runtime_path)
    stale = {**context, 'description': context['description'] + ' Changed job.'}
    stale['context_digest'] = canonical_digest(stale, digest_field='context_digest')
    with pytest.raises(ValueError, match='source_context_changed'):
        family_source_references(preparation=preparation, runtime_inputs_path=runtime_path, task_context=stale)
    preparation['binding']['source_geometry_digest'] = 'sha256:' + 'f' * 64
    with pytest.raises(ValueError, match='preparation_binding_changed'):
        family_source_references(preparation=preparation, runtime_inputs_path=runtime_path, task_context=context)


@pytest.mark.parametrize('changed', ['compiled_order', 'semantic_order'])
def test_resealed_native_or_semantic_operation_reordering_refuses_before_dispatch(tmp_path, changed):
    preparation, runtime_path, context = semantic_source(tmp_path, multi=True)
    assert family_source_references(preparation=preparation, runtime_inputs_path=runtime_path, task_context=context)
    inputs = preparation['task_evaluation_inputs']
    if changed == 'compiled_order':
        contract = inputs['task_success_contract']
        contract['stages'].reverse()
        contract['contract_digest'] = canonical_digest(contract, digest_field='contract_digest')
        native = inputs['native_runtime_contract']
        native['task_spec']['task_success_contract'] = contract
        native['contract_digest'] = canonical_digest(native, digest_field='contract_digest')
        preparation['intake_request']['task'].update(task_success_contract_digest=contract['contract_digest'],
            native_runtime_contract_digest=native['contract_digest'])
        write_json(runtime_path, native)
    else:
        receipt = inputs['task_semantics']['planner_receipt']
        receipt['proposal']['task_semantics']['steps'].reverse()
        receipt['proposal_digest'] = canonical_digest(receipt['proposal'])
        inputs['task_semantics'] = task_semantics_from_receipt(receipt=receipt, task_context=context,
            source_video_digest=inputs['source_video_digest'])
    preparation['digest'] = canonical_digest(preparation, digest_field='digest')
    with pytest.raises(ValueError, match='semantic_stages_mismatch'):
        family_source_references(preparation=preparation, runtime_inputs_path=runtime_path, task_context=context)
