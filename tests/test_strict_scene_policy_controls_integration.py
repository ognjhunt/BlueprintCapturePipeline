"""Actual minimal-task assembly preserves controls and delegated provenance."""
from __future__ import annotations

from copy import deepcopy

import blueprint_pipeline.task_evaluation_scene_configuration_submission_records as records
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_policy_canary_control_gate import controls_required
from blueprint_pipeline.native_task_arena_policy_canary_session import execute_paired_session, validate_runtime_input_manifest
from blueprint_pipeline.task_evaluation_policy_canary_scene_setup import _quick_cells
from blueprint_pipeline.task_evaluation_rigid_destination_geometry import bind_destination_trajectory
from blueprint_pipeline.task_evaluation_rigid_owner_contract import _derive_configured_owner_success_contract
from blueprint_pipeline.task_evaluation_rigid_relocation_native_adapter import adapt_rigid_relocation_task_template
from tests.test_native_task_arena_policy_canary_session import _preload_gate_session_kwargs
from tests.test_task_evaluation_scene_configuration_submission_native_contract import _assembled_native_case


def _delegated_request(task):
    task['success'].update(
        retreat_clearance_m=.05, drop_minimum_fall_m=.005, maximum_task_contact_force_n=10.,
        collision_failure_minimum_force_n=1.,
        robot_workspace_position_bounds_world_m={'minimum': [-10., -10., -1.], 'maximum': [10., 10., 3.]},
        forbidden_contact_classes=['robot_background', 'robot_object', 'object_background', 'destination_background'])
    proposal = {'success': deepcopy(task['success']), 'model': 'fixture-agents-sdk'}
    task['success_contract_authority'] = {
        'author_source': 'agent_proposal', 'author_id': 'fixture:task-author', 'agent_proposal': proposal,
        'proposal_digest': canonical_digest(proposal, digest_field='proposal_digest'),
        'delegation_authority_reference': 'fixture:standing-user-delegation',
        'confirmed_by_team_id': task['team_namespace'], 'confirmation_status': 'confirmed',
        'accepted_by': 'fixture-team', 'authority_reference': 'fixture:retained-owner-task-request'}


def test_minimal_request_reaches_worker_without_losing_required_controls_or_agent_provenance(tmp_path, monkeypatch):
    recorded = []
    original = records.pick_and_place_task_records
    def observed_records(**kwargs):
        recorded.append(True)
        return original(**kwargs)
    monkeypatch.setattr(records, 'pick_and_place_task_records', observed_records)
    launch, configured, references, _, _ = _assembled_native_case(tmp_path, _delegated_request)
    adapted = adapt_rigid_relocation_task_template(request=launch, configured_revision=configured, materialized_references=references)
    native = adapted['native_task_definition']['task_spec']
    assert recorded == [True]
    assert native['configured_success_criteria']['per_cell_controls_required'] is True
    # Synthetic downstream destination qualification joins only; no real
    # native measurement or source-dataset qualification is claimed by this test.
    target = native['target_position_world_m']
    native = bind_destination_trajectory(native, {
        'intended_support_prim_paths': ['/Tray/Bottom'], 'insertion_withdrawal_unit_world': [0., 0., 1.],
        'target_position_world_m': target, 'destination_orientation_world_xyzw': [0., 0., 0., 1.],
        'support_height_interval_m': native['support_height_interval_m'],
        'visible_label': 'blue document tray', 'relation': 'inside'})
    native.update(destination_relation='inside', destination_pose_world=[*target, 0., 0., 0., 1.],
                  subject_collision_bounds_scoring_frame_m={'minimum': [-.1, -.1, -.01], 'maximum': [.1, .1, .01]})
    contract = _derive_configured_owner_success_contract(native, site_id=launch['scene']['identity']['id'],
        task_id=launch['task']['identity']['id'], team_namespace=launch['team_namespace'])
    assert controls_required(contract)
    assert contract['provenance']['author_source'] == 'agent_proposal'
    assert contract['provenance']['confirmed_by_team_id'] == launch['team_namespace']
    assert contract['provenance']['proposal_digest'].startswith('sha256:')

    calls = {'open': 0, 'close': 0, 'loads': []}
    runtime_root = tmp_path/'runtime'
    runtime_root.mkdir()
    kwargs = _preload_gate_session_kwargs(runtime_root, calls)
    inputs = deepcopy(kwargs['runtime_inputs'])
    inputs.update(task_success_contract=contract, task_success_contract_digest=contract['contract_digest'])
    inputs['cells'] = [{**cell, 'resolved_scenario_digest': canonical_digest(cell['resolved_scenario']),
                       'control_diagnostic': {'mode': 'required_before_policy',
                                              'typed_gap': 'controls_pending_at_submission',
                                              'policy_execution_blocked': True}}
                      for cell in _quick_cells(configured['revision_digest'], scene_id='841757')]
    inputs['runtime_inputs_digest'] = canonical_digest(inputs, digest_field='runtime_inputs_digest')
    validate_runtime_input_manifest(inputs)
    authority = deepcopy(kwargs['authority'])
    authority.update(runtime_inputs_digest=inputs['runtime_inputs_digest'], task_success_contract_digest=contract['contract_digest'])
    authority['authority_digest'] = canonical_digest(authority, digest_field='authority_digest')
    kwargs.update(runtime_inputs=inputs, authority=authority)
    result = execute_paired_session(**kwargs,
        prepolicy_observation_gate=lambda session: {'policy_observation_integrity_passed': True})
    assert result['status'] == 'blocked'
    assert result['policy_loads'] == [] and calls['loads'] == []
    assert result['candidate_policy_queried'] is False
    assert result['task_success_contract'] == contract
