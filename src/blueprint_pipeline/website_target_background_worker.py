"""Repair a corrected target through the capture's existing, capped Atlas worker.

The intake only queues retained evidence. This worker edits original screened
views, never the previous object's cleaned background. Original job clocks,
authority, operations and exports remain retained.
"""
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import time

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .website_target_background_correction import correction_root, derive_review, _retain


def _read(path):
    if path.is_symlink():
        raise ValueError('website_target_correction_path_unsafe')
    return json.loads(path.read_text())


def _inside(root, relative):
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('website_target_correction_path_unsafe')
    return path


def queue_correction(*, capture_root, task_context, task_masks, now=None):
    """Return ready, pending or needs_input. This function never calls a provider."""
    from .core.handoff_lease import handoff_lease_checkpoint, guarded_handoff_write_json
    from .website_task_target_choice import corrections
    from .website_atlas_preview_store import JOB_FILE, LOCK_FILE, MAXIMUM_JOB_SECONDS
    fixes = corrections(capture_root / 'pipeline')
    if not fixes:
        return {'status': 'ready'}
    if any(row.get('kind') == 'none_of_these' for row in fixes):
        return {'status': 'needs_input', 'reason': 'website_task_target_selection_rejected'}
    digest = canonical_digest({'corrections': fixes})
    job_dir = capture_root / 'pipeline/atlas'
    current = int(time.time() if now is None else now)
    with (job_dir / LOCK_FILE).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'pending', 'reason': 'website_atlas_job_processing'}
        job = _read(job_dir / JOB_FILE)
        active = job.get('target_correction') or {}
        if active:
            if (active.get('corrections_digest') != digest
                    or active.get('task_context_digest') != task_context['context_digest']):
                return {'status': 'needs_input', 'reason': 'website_target_correction_selection_changed'}
            if job['state'] == 'processing':
                return {'status': 'pending', 'reason': 'website_atlas_job_processing'}
            if job['state'] != 'ready':
                return {'status': 'needs_input', 'reason': job.get('reason') or 'website_target_correction_incomplete'}
        result = _read(job_dir / 'result.json')
        if (result.get('digest') != canonical_digest(result, digest_field='digest')
                or result['digest'] != job.get('result_digest')
                or result.get('task_context_digest') != task_context['context_digest']
                or task_masks.get('source_video_digest') != result.get('source_video_digest')
                or task_masks.get('digest') != canonical_digest(task_masks, digest_field='digest')):
            raise ValueError('website_target_correction_source_changed')
        completed = result.get('target_correction') or {}
        if completed.get('corrections_digest') == digest and completed.get('task_context_digest') == task_context['context_digest']:
            _bound_request(job_dir, completed)
            return {'status': 'ready'}
        if (job['state'] != 'ready' or current - job['enqueued_at_epoch'] > MAXIMUM_JOB_SECONDS
                or (job.get('provider_cleanup') or {}).get('state') == 'failed'):
            return {'status': 'needs_input', 'reason': 'website_target_correction_original_window_unavailable'}
        selection = _read(job_dir / 'selection_binding.json')
        if (result.get('source_pose') or {}).get('status') != 'completed':
            raise ValueError('website_target_correction_source_pose_unavailable')
        request = derive_review(review=selection['review'], targets=task_masks['targets'], corrections=fixes,
            task_context=task_context, parent_result=result, task_masks_digest=task_masks['digest'])
        root = correction_root(job_dir, request['digest'])
        handoff_lease_checkpoint('website_target_correction_queue')
        _retain(root / 'request.json', request, guarded_handoff_write_json)
        _retain(root / 'parent_result.json', result, guarded_handoff_write_json)
        _retain(root / 'parent_job.json', job, guarded_handoff_write_json)
        cleanup = job.get('provider_cleanup') or {}
        # Preserve prior deletion evidence and attempts; new assets still need cleanup.
        updated = {**job, 'state': 'processing', 'reason': None, 'stage': 'target_correction',
            'target_correction': _descriptor(request, job_dir), 'archive': None,
            'next_attempt_at_epoch': current, 'retry_at_epoch': None,
            'revision': job['revision'] + 1, 'revised_at_epoch': current, 'updated_at_epoch': current,
            'provider_cleanup': {**cleanup, 'state': 'pending', 'next_attempt_at_epoch': current}}
        guarded_handoff_write_json(job_dir / JOB_FILE, updated)
    return {'status': 'pending', 'reason': 'website_atlas_job_processing'}


def _descriptor(request, job_dir):
    root = correction_root(job_dir, request['digest'])
    return {'request_digest': request['digest'], 'request_relative_path': str((root / 'request.json').relative_to(job_dir)),
        'automation_relative_path': str((root / 'automation').relative_to(job_dir)),
        'corrections_digest': request['corrections_digest'], 'task_context_digest': request['task_context_digest']}


def _bound_request(job_dir, descriptor):
    root = correction_root(job_dir, descriptor['request_digest'])
    request = _read(root / 'request.json')
    if (request.get('digest') != canonical_digest(request, digest_field='digest')
            or request['digest'] != descriptor['request_digest'] or _descriptor(request, job_dir) != descriptor):
        raise ValueError('website_target_correction_request_changed')
    return root, request


def _source_inputs(job_dir, request):
    """Use the original screened full-view pixels, with no crop/depth-grid substitution."""
    from .atlas_context_fov import full_view_sources
    from .atlas_video_inputs import GEOMETRY_INPUTS_FILE
    from .local_reconstruction_adapters import _sha256_file
    from .website_track_mask_codec import decode_track_mask
    from PIL import Image
    pose_root = job_dir / 'pose'
    inputs = _read(pose_root / GEOMETRY_INPUTS_FILE)
    if (inputs.get('digest') != canonical_digest(inputs, digest_field='digest')
            or inputs['binding']['source_video_digest'] != request['source_video_digest']):
        raise ValueError('website_target_correction_source_inputs_changed')
    frames = full_view_sources(inputs, pose_root)
    tracks = deepcopy(request['selected_tracks'])
    for track in tracks.values():
        observations = {row['source_frame_id']: row for row in track['observations']}
        selected = []
        for frame in frames:
            observation = observations.get(frame['frame_id'])
            if observation is None:
                raise ValueError('website_target_correction_observed_view_missing')
            path = _inside(job_dir, Path(frame['image_path']))
            if _sha256_file(path) != frame['image_digest']:
                raise ValueError('website_target_correction_source_pixels_changed')
            with Image.open(path) as image:
                if decode_track_mask(observation).shape != (image.height, image.width):
                    raise ValueError('website_target_correction_source_grid_changed')
            selected.append(observation)
        track['observations'] = selected
    inputs = {**inputs, 'frames': frames}
    inputs.pop('full_view_sources', None)
    inputs['digest'] = canonical_digest(inputs, digest_field='digest')
    review = deepcopy(request['review'])
    wanted = {row['frame_id'] for row in frames}
    for obj in review['automation']['objects']:
        obj['regions'] = [row for row in obj['regions'] if row['frame_id'] in wanted]
    return inputs, tracks, review


def _owner_checkpoint(*, job_dir, capture_root, context, request, checkpoint):
    """A withdrawal, changed task or changed owner selection stops the next effect."""
    from .website_preparation_status import DELIVERY_FILE, validate_selectors
    from .website_task_context import website_webapp_request
    from .website_task_target_choice import read_question, validate_answer
    selectors = validate_selectors(_read(capture_root / DELIVERY_FILE)['selectors'])
    if selectors['task_context_digest'] != context['context_digest']:
        raise ValueError('website_target_correction_task_changed')
    for correction in request['corrections']:
        question = read_question(capture_root / 'pipeline', correction['target_id'])
        if not question or question['question_digest'] != correction['question_digest']:
            raise ValueError('website_target_correction_question_changed')
        answer = validate_answer(website_webapp_request(capture_id=context['capture_id'],
            operation='task-target-choice', payload={**selectors, 'question_digest': question['question_digest']}),
            question=question, selectors=selectors)
        if answer['choice'] != {'kind': 'other_candidate', 'candidate_id': correction['candidate_id']}:
            raise ValueError('website_target_correction_selection_changed')
    checkpoint('target_correction_current_source')


def advance_correction(*, job, job_dir, capture_root, context, checkpoint, max_polls=1):
    """Called only under the existing Atlas job lock, with its unchanged expiry."""
    from .website_atlas_pipeline import load_configuration, finalize_background_result, _REVIEW_RECORD
    from .paid_resource_allocator import run_sponsored_website_atlas_automation
    from .website_atlas_pose import ATLAS_PROVIDER_RETENTION_ENV
    import os
    root, request = _bound_request(job_dir, job['target_correction'])
    from .local_reconstruction_adapters import _sha256_file
    video = _inside(capture_root, job['source_video'])
    if _sha256_file(video) != request['source_video_digest']:
        raise ValueError('website_target_correction_source_video_changed')
    parent = _read(root / 'parent_result.json')
    if (parent.get('digest') != request['parent_result_digest']
            or parent.get('digest') != canonical_digest(parent, digest_field='digest')
            or context['context_digest'] != request['task_context_digest']):
        raise ValueError('website_target_correction_source_changed')
    current = load_configuration(context, request['source_video_digest'], create=False)
    original = _read(job_dir / 'configuration.json')
    if (current['authority'] != original['authority']
            or any(current['policy'].get(key) != value for key, value in original['policy'].items() if key not in _REVIEW_RECORD)):
        raise ValueError('atlas_runtime_authority_or_policy_changed')
    inputs, tracks, review = _source_inputs(job_dir, request)
    _retain(root / 'effective_review.json', review)

    def check(stage):
        _owner_checkpoint(job_dir=job_dir, capture_root=capture_root, context=context, request=request, checkpoint=checkpoint)
        checkpoint(stage)

    check('target_correction_start')
    os.environ[ATLAS_PROVIDER_RETENTION_ENV] = current['authority']['provider_retention']
    automation = run_sponsored_website_atlas_automation(pose_root=None, inputs=inputs, review=review,
        output_root=root / 'automation', task_context=context, configuration=current, max_polls=max_polls,
        checkpoint=check, edit_scope=parent.get('edit_scope') or 'required_and_additional', selected_tracks=tracks)
    if automation['status'] == 'processing':
        return {**automation, 'stage': 'target_correction'}
    if automation['status'] != 'completed' or (automation.get('background') or {}).get('status') != 'completed':
        return {'status': 'needs_input', 'stage': 'target_correction', 'reason': 'website_target_correction_background_incomplete'}
    check('target_correction_publish')
    result = finalize_background_result(output_root=root, source_digest=request['source_video_digest'],
        task_context=context, automation=automation, source_pose=parent['source_pose'], model=parent['model'],
        edit_scope=parent.get('edit_scope'))
    result['target_correction'] = job['target_correction']
    result['digest'] = canonical_digest(result, digest_field='digest')
    write_json(root / 'result.json', result)
    # The preview controller archives this result before publishing its current
    # pointer. Failed archive/withdrawal cannot expose a completed corrected scene.
    return result


def publish_correction(*, job_dir, result, capture_root, context, checkpoint):
    descriptor = result.get('target_correction')
    if descriptor:
        root, request = _bound_request(job_dir, descriptor)
        _owner_checkpoint(job_dir=job_dir, capture_root=capture_root, context=context,
                          request=request, checkpoint=checkpoint)
        _retain(root / 'published_result.json', result)
        write_json(job_dir / 'result.json', result)
