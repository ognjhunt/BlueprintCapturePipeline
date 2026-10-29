"""Source-bound object corrections for the existing Atlas worker.

A new mask is not evidence of a newly edited background. Original source views,
selection and result stay retained; an owner correction has its own provenance.
This module cannot create or extend a preparation allowance.
"""
from copy import deepcopy
from pathlib import Path
import json
import re

from .decision_evidence_contracts import canonical_digest

SCHEMA = 'website_target_background_correction.v1'


def derive_review(*, review, targets, corrections, task_context, parent_result, task_masks_digest):
    """Derive only observed regions from the selected instance, never new task truth."""
    from .website_task_target_choice import _candidate_rows
    from .website_track_mask_codec import decode_track_mask
    from .atlas_automation_plan import AutomationPlan
    result = deepcopy(review)
    removed = [row for row in result['automation']['objects'] if row['disposition'] == 'remove']
    # The existing Atlas bridge represents one observed rigid manipulated subject.
    # An unrelated target cannot acquire that subject's source binding.
    if len(corrections) != 1 or len(removed) != 1:
        raise ValueError('website_target_background_correction_selection_unresolved')
    correction = corrections[0]
    selected = [row for row in targets if row.get('target_id') == correction['target_id']
                and row.get('task_effect') == 'manipulated' and row.get('disposition') == 'remove']
    if len(selected) != 1 or not isinstance(selected[0].get('source_track'), dict):
        raise ValueError('website_target_background_correction_track_missing')
    target = selected[0]
    identity = _candidate_rows([target['source_track']])
    if len(identity) != 1 or identity[0]['candidate_id'] != correction['candidate_id']:
        raise ValueError('website_target_background_correction_candidate_changed')
    # Native depth uses a resized, uncropped grid. Keep the original source
    # mask for image editing; never resize a depth-grid mask onto cropped RGB.
    from PIL import Image
    import numpy as np
    source = {canonical_digest(row): row for row in target['source_track']['observations']}
    for observation in target['track']['observations']:
        original = source.get(observation.get('source_mask_digest'))
        if original is None:
            original = source.get(canonical_digest(observation))
        if original is None or original['source_frame_id'] != observation['source_frame_id']:
            raise ValueError('website_target_background_correction_mask_binding_changed')
        expected = np.asarray(Image.fromarray(decode_track_mask(original)).resize(
            (observation['width'], observation['height']), Image.Resampling.NEAREST))
        if not np.array_equal(expected, decode_track_mask(observation)):
            raise ValueError('website_target_background_correction_mask_binding_changed')
    observed_ids = {row['source_frame_id'] for row in target['track']['observations']}
    track = {**deepcopy(target['source_track']), 'observations': [deepcopy(row)
        for row in target['source_track']['observations'] if row['source_frame_id'] in observed_ids]}
    regions = []
    for observation in track['observations']:
        mask = decode_track_mask(observation)
        ys, xs = mask.nonzero()
        if not len(xs):
            continue
        height, width = mask.shape
        regions.append({'frame_id': observation['source_frame_id'], 'box_xywh_normalized': [
            int(xs.min()) / width, int(ys.min()) / height,
            (int(xs.max()) - int(xs.min()) + 1) / width,
            (int(ys.max()) - int(ys.min()) + 1) / height]})
    if not regions:
        raise ValueError('website_target_background_correction_observation_missing')
    removed[0]['regions'] = regions
    removed[0]['segmentation_prompt'] = 'the selected object inside the supplied source masks'
    result['automation']['removal_instruction'] = 'Remove only the object inside the supplied source masks; preserve other objects.'
    result['automation'] = AutomationPlan.model_validate(result['automation']).model_dump(mode='json', exclude_none=True)
    value = {'schema_version': SCHEMA, 'claim_ceiling': 'development_only',
        'basis': 'owner_selected_source_instance', 'capture_id': task_context['capture_id'],
        'task_context_digest': task_context['context_digest'],
        'source_video_digest': parent_result['source_video_digest'],
        'parent_result_digest': parent_result['digest'],
        'source_pose': deepcopy(parent_result['source_pose']),
        'original_review_digest': canonical_digest(review),
        'task_masks_digest': task_masks_digest, 'corrections': deepcopy(corrections),
        'corrections_digest': canonical_digest({'corrections': corrections}),
        'review': result, 'selected_tracks': {removed[0]['target_id']: track},
        'owner_confirmation': False, 'metric_authority': False, 'physical_evidence': False,
        'authority_issued': False}
    value['digest'] = canonical_digest(value, digest_field='digest')
    return value


def correction_root(job_dir, request_digest):
    if not isinstance(request_digest, str) or not re.fullmatch(r'sha256:[a-f0-9]{64}', request_digest):
        raise ValueError('website_target_background_correction_digest_invalid')
    return Path(job_dir) / 'target_corrections' / request_digest[7:]


def retain_rejected_selection(*, capture_root, task_context, source_video):
    """Retire the guessed estimate without buying another answer or requiring footage."""
    from .website_task_target_choice import rejected_questions, correction_binding_digest
    from .local_reconstruction_adapters import _sha256_file
    from .core.handoff_lease import guarded_handoff_write_json
    rows = [row for row in rejected_questions(Path(capture_root) / 'pipeline')
            if row['answer']['task_context_digest'] == task_context.get('context_digest')]
    if not rows:
        return None
    source_digest = _sha256_file(Path(source_video))
    if any(any(row['question'].get(key) != task_context[key] for key in ('scene_id', 'capture_id'))
           or row['question']['source_video_digest'] != source_digest for row in rows):
        raise ValueError('website_target_rejection_source_changed')
    rejection = {'schema_version': 'website_task_target_rejection.v1',
        'task_context_digest': task_context['context_digest'], 'source_video_digest': source_digest,
        'corrections_digest': correction_binding_digest(Path(capture_root) / 'pipeline'),
        'rejections': rows, 'provider_mutation_performed': False, 'execution_authority_granted': False,
        'claim_ceiling': 'development_only', 'additional_footage_required': False,
        'evidence_limit': 'owner_rejected_retained_candidates_no_supported_selection'}
    rejection['digest'] = canonical_digest(rejection, digest_field='digest')
    handoff = {'schema_version': 'website_scene_handoff.v1', 'status': 'needs_input',
        'scene_id': task_context['scene_id'], 'capture_id': task_context['capture_id'],
        'blockers': ['website_task_target_selection_rejected'], 'selection_rejection': rejection,
        'provider_mutation_performed': False, 'simulator_ready': False, 'claim_ceiling': 'development_only'}
    handoff['digest'] = canonical_digest(handoff, digest_field='digest')
    root = Path(capture_root) / 'pipeline/website_scene_preparation'
    prior = root / 'handoff.json'
    if prior.is_file():
        old = json.loads(prior.read_text())
        old_digest = canonical_digest(old)
        _retain(root / 'history' / old_digest[7:] / 'handoff.json', old, guarded_handoff_write_json)
    _retain(root / 'rejections' / rejection['digest'][7:] / 'handoff.json', handoff, guarded_handoff_write_json)
    guarded_handoff_write_json(prior, handoff)
    return handoff


def _retain(path, value, writer=None):
    from .common import write_json
    if path.is_symlink():
        raise ValueError('website_target_correction_retained_path_unsafe')
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('website_target_correction_retained_record_changed')
    else:
        (writer or write_json)(path, value)


def retains_rejected_selection(handoff, context):
    """Read-only classification; every owner/source/commit admission remains separate."""
    from .website_task_target_choice import validate_answer
    value = handoff.get('selection_rejection') or {}
    if not isinstance(value, dict):
        return False
    if (handoff.get('schema_version') != 'website_scene_handoff.v1' or handoff.get('status') != 'needs_input'
            or handoff.get('digest') != canonical_digest(handoff, digest_field='digest')
            or any(handoff.get(key) != context.get(key) for key in ('scene_id', 'capture_id'))
            or handoff.get('provider_mutation_performed') is not False
            or handoff.get('simulator_ready') is not False
            or value.get('schema_version') != 'website_task_target_rejection.v1'
            or value.get('digest') != canonical_digest(value, digest_field='digest')
            or value.get('task_context_digest') != context.get('context_digest')
            or value.get('provider_mutation_performed') is not False
            or value.get('execution_authority_granted') is not False
            or value.get('additional_footage_required') is not False):
        return False
    rows = value.get('rejections')
    if not isinstance(rows, list) or not rows:
        return False
    try:
        for row in rows:
            answer = validate_answer(row['answer'], question=row['question'])
            if (answer['choice']['kind'] != 'none_of_these'
                    or answer['task_context_digest'] != context.get('context_digest')
                    or row['question']['source_video_digest'] != value.get('source_video_digest')):
                return False
    except (ValueError, KeyError, TypeError):
        return False
    # Keep this classifier pure: current on-disk choices are checked by the
    # caller's status/source read, not inferred as new execution authority.
    return isinstance(value.get('corrections_digest'), str)
