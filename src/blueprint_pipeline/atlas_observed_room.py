"""Retain useful observed-room reconstruction when the task reading abstains.

This is a projection of independently usable room observations, never acceptance
of the original task proposal. It creates no task object, destination or grant.
"""
from copy import deepcopy
import json
import math

from .common import write_json
from .decision_evidence_contracts import canonical_digest

SELECTION = 'observed_room/selection.json'


def observation_review(*, proposal, survey, policy, pose_budget):
    from .atlas_scene_planner import Coverage, SelectionPolicy
    if proposal.get('status') != 'abstained':
        raise ValueError('atlas_observed_room_task_abstention_required')
    intervals = proposal.get('static_intervals_seconds')
    if (not isinstance(intervals, list) or not intervals or any(not isinstance(row, list) or len(row) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in row)
            or not 0 <= row[0] < row[1] for row in intervals)):
        raise ValueError('atlas_observed_room_static_observations_required')
    state = proposal.get('scene_state')
    if not isinstance(state, str) or not state.strip():
        raise ValueError('atlas_observed_room_state_required')
    frames = {row['frame_id']: row for row in survey['frames']}
    if (not isinstance(proposal.get('unsafe_frame_ids'), list)
            or not isinstance(proposal.get('transient_frame_ids', []), list)):
        raise ValueError('atlas_observed_room_privacy_selection_invalid')
    unsafe, transient = set(proposal['unsafe_frame_ids']), set(proposal.get('transient_frame_ids', []))
    if not (unsafe | transient) <= frames.keys():
        raise ValueError('atlas_observed_room_privacy_selection_invalid')
    thresholds = SelectionPolicy.model_validate(proposal['selection_policy']).model_dump(mode='json')
    excluded = unsafe | transient | {key for key, row in frames.items()
        if row.get('sharpness', float('inf')) < thresholds['minimum_sharpness']
        or row.get('clipped_fraction', 0) > thresholds['maximum_clipped_fraction']}
    selected = {key for key, row in frames.items() if key not in excluded
        and any(start <= row['timestamp_seconds'] <= end for start, end in intervals)}
    # Missing/uncertain task coverage is quarantined, not promoted to room truth.
    room = [Coverage.model_validate(row).model_dump(mode='json')
            for row in proposal.get('coverage_requirements', []) if row.get('role') == 'surrounding_scene']
    required = {key for row in room for key in row['required_frame_ids']}
    if len(selected) < 2 or not room or not required <= selected:
        raise ValueError('atlas_observed_room_anchors_required')
    original_anchor = proposal.get('anchor_frame_id')
    anchor = original_anchor if original_anchor in required else room[0]['required_frame_ids'][0]
    if pose_budget < 2 * policy['pose_input_cost_usd']:
        raise ValueError('atlas_runtime_budget_insufficient')
    return {'schema_version': 'atlas_scene_state_review.v1', 'review_scope': 'observed_room_only',
        'survey_digest': survey['digest'], 'original_task_proposal_digest': canonical_digest(proposal),
        'scene_state': state, 'static_intervals_seconds': deepcopy(intervals), 'anchor_frame_id': anchor,
        'original_task_anchor_frame_id': original_anchor,
        'coverage_requirements': room, 'uncaptured_regions': deepcopy(proposal.get('uncaptured_regions', [])),
        'selection_policy': thresholds, 'excluded_frame_ids': sorted(excluded), 'transient_frame_ids': [],
        'moving_people_excluded': True, 'same_scene_state': True,
        'subject_origin': proposal.get('subject_origin'),
        'automation': {'objects': [], 'scale_references': [], 'removal_instruction': '', 'mask_margin_fraction': .01},
        'reviewed_by': 'retained_production_agents_sdk:' + policy['model'],
        'review_basis': 'retained_room_observations_not_task_acceptance_or_privacy_proof',
        'selection_budget': {'maximum_pose_cost_usd': pose_budget, 'input_cost_usd': policy['pose_input_cost_usd'],
            **{key: policy[key] for key in ('rate_card_reference', 'reviewed_by', 'reviewed_on')}},
        'task_and_surrounding_scene_covered': False, 'simulation_qualified': False, 'physical_evidence': False}


def _read(path):
    if path.is_symlink():
        raise ValueError('atlas_observed_room_path_unsafe')
    return json.loads(path.read_text())


def retain_selection(*, output_root, survey_path, survey, receipt, task_context, configuration):
    source = survey['source_video_digest']
    binding = receipt.get('binding') or {}
    if (receipt.get('status') != 'completed' or receipt.get('binding_digest') != canonical_digest(binding)
            or binding.get('source_video_digest') != source or binding.get('survey_digest') != survey['digest']
            or binding.get('task_context_digest') != task_context['context_digest']):
        raise ValueError('atlas_observed_room_planner_receipt_changed')
    review = observation_review(proposal=receipt['proposal'], survey=survey, policy=configuration['policy'],
                                pose_budget=configuration['remaining_cost_usd'])
    relative = str(survey_path.resolve().relative_to(output_root.resolve()))
    value = {'schema_version': 'atlas_observed_room_selection.v1', 'source_video_digest': source,
        'task_context_digest': task_context['context_digest'], 'survey_relative_path': relative,
        'survey_digest': survey['digest'], 'planner_receipt': deepcopy(receipt), 'review': review,
        'authority_issued': False, 'task_accepted': False, 'physical_evidence': False}
    value['digest'] = canonical_digest(value, digest_field='digest')
    path = output_root / SELECTION
    if path.exists():
        if _read(path) != value:
            raise ValueError('atlas_observed_room_selection_changed')
    else:
        write_json(path, value)
    return value


def run_observed_room(*, output_root, source_digest, task_context, max_polls, checkpoint, experimental=False):
    from . import website_atlas_pipeline as pipeline
    from .atlas_video_inputs import prepare_video_inputs
    from .website_atlas_pose import ATLAS_PROVIDER_RETENTION_ENV, atlas_api_config
    import os
    selection = _read(output_root / SELECTION)
    if (selection.get('digest') != canonical_digest(selection, digest_field='digest')
            or selection.get('source_video_digest') != source_digest
            or selection.get('task_context_digest') != task_context['context_digest']
            or selection.get('task_accepted') is not False or selection.get('authority_issued') is not False):
        raise ValueError('atlas_observed_room_selection_changed')
    survey_path = output_root / selection['survey_relative_path']
    if survey_path.is_symlink() or not survey_path.resolve().is_relative_to(output_root.resolve()):
        raise ValueError('atlas_observed_room_path_unsafe')
    survey = _read(survey_path)
    if (survey.get('digest') != selection['survey_digest'] or survey.get('digest') != canonical_digest(survey, digest_field='digest')
            or survey.get('source_video_digest') != source_digest or survey.get('heldout_pixels_included') is not False):
        raise ValueError('atlas_observed_room_source_changed')
    current = pipeline.load_configuration(task_context, source_digest, create=False)
    original = _read(output_root / 'configuration.json')
    current = {**current, **pipeline.resumed_policy(original, current)}
    atlas_api_config()
    os.environ[ATLAS_PROVIDER_RETENTION_ENV] = current['authority']['provider_retention']
    check = checkpoint or (lambda _stage: None)
    check('observed_room_source')
    root = output_root / 'observed_room'
    review = selection['review']
    receipt = selection['planner_receipt']
    binding = receipt.get('binding') or {}
    if (receipt.get('status') != 'completed' or receipt.get('binding_digest') != canonical_digest(binding)
            or binding.get('source_video_digest') != source_digest or binding.get('survey_digest') != survey['digest']
            or binding.get('task_context_digest') != task_context['context_digest']
            or observation_review(proposal=receipt['proposal'], survey=survey, policy=original['policy'],
                pose_budget=review['selection_budget']['maximum_pose_cost_usd']) != review):
        raise ValueError('atlas_observed_room_planner_receipt_changed')
    write_json(root / 'atlas_scene_review.json', review)
    inputs = prepare_video_inputs(survey_path=survey_path, review=review, output_root=root / 'pose')
    limitation = {'schema_version': 'atlas_task_evidence_limitation.v1', 'status': 'abstained',
        'reason': 'atlas_scene_planner_abstained', 'planner_receipt_digest': canonical_digest(selection['planner_receipt']),
        'original_task_context_digest': task_context['context_digest'], 'source_video_digest': source_digest,
        'summary': selection['planner_receipt']['proposal']['summary']}
    limitation['digest'] = canonical_digest(limitation, digest_field='digest')
    result = pipeline._preserved_scene(pose_root=root / 'pose', output_root=root, inputs=inputs, review=review,
        policy=current['policy'], task_context=task_context, source_digest=source_digest, configuration=current,
        check=check, experimental=experimental, max_polls=max_polls, model=selection['planner_receipt']['model'],
        reconstruction_only=True, task_evidence_limitation=limitation)
    if result['status'] == 'processing' or not result.get('download'):
        return result
    result = deepcopy(result)
    if result.get('source_pose', {}).get('pose_relative_path'):
        result['source_pose']['pose_relative_path'] = 'observed_room/' + result['source_pose']['pose_relative_path']
    result.update(observed_scene_only=True, task_and_surrounding_scene_covered=False,
        native_preparation_eligible=False, execution_authority_granted=False,
        geometry_authority='captured_rgb_derived_depth_estimates',
        task_evidence_limitation=limitation)
    result['blockers'] = list(dict.fromkeys([*result.get('blockers', []), 'atlas_task_evidence_incomplete']))
    result['digest'] = canonical_digest(result, digest_field='digest')
    check('observed_room_publish')
    write_json(root / 'result.json', result)
    write_json(output_root / 'result.json', result)
    return result
