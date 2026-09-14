"""Apply explicit user exclusions to real AI-reviewed training targets.

The original provider review stays immutable. This derived review retains its
provider-call provenance and identifies the additional operator veto separately;
it never claims a new provider call or promotes a rejected target to accepted.
Only the existing count and camera-coverage selector can admit the result.
"""
from copy import deepcopy

from .decision_evidence_contracts import canonical_digest


def apply_user_exclusions(*, review, excluded_frames, authorization_reference):
    if (review.get('execution_digest') != canonical_digest(review, digest_field='execution_digest')
            or review.get('status') != 'completed' or review.get('provider_called') is not True
            or review.get('review_phase') != 'pre_training_semantic_targets'
            or review.get('schema_version') != 'task_evaluation_artifixer_ai_visual_review_execution.v1'
            or not authorization_reference or not excluded_frames):
        raise ValueError('operator_selection_source_review_invalid')
    inventory = {row['camera_id']: row for row in review['frames']}
    if len(inventory) != len(review['frames']):
        raise ValueError('operator_selection_duplicate_camera')
    for camera, digest in excluded_frames.items():
        if camera not in inventory or inventory[camera]['frame_sha256'] != digest:
            raise ValueError('operator_selection_frame_binding_invalid')
    value = deepcopy(review)
    value['source_provider_review'] = deepcopy(review)
    value['operator_selection'] = {
        'schema_version': 'artifixer_user_training_exclusions.v1',
        'authorization_reference': authorization_reference,
        'excluded_frames': dict(excluded_frames),
        'source_execution_digest': review['execution_digest'],
        'new_provider_call_performed': False,
        'scope': 'training_targets_only_no_final_appearance_qualification',
    }
    value['reviewer'] = {**review.get('reviewer', {}),
                         'kind': 'ai_review_with_explicit_user_exclusions'}
    value['decision'] = 'rejected'
    value['summary'] = 'Use the original AI review subject to the explicit user exclusions recorded here.'
    for row in value['frames']:
        if row['camera_id'] in excluded_frames:
            row.update(decision='rejected', preserves_non_target_content=False,
                       repair_priority=3, rationale='Explicit user exclusion of this exact frame; see operator_selection.authorization_reference.')
    value['execution_digest'] = canonical_digest(value, digest_field='execution_digest')
    return value
