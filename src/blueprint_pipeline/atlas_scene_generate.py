"""Photograph-and-depth conditioned novel views, before final reconstruction.

ADP-040/day-28 development seam. Reuses the admitted task allocator, full-FOV
photo preparation, native result reader and source-priority geometry merge.
Meshing is deliberately downstream: it cannot generate missing observations.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import write_json
from .atlas_pricing import render_cost, final_output_cost, completion_request_preflight
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file


def prepare_rgbd_context(*, frames, root):
    """Upload aligned RGB/depth/cameras, with every approved photo ray retained."""
    from .atlas_scene_chisel import prepare_source_photographs
    from .atlas_depth_splat import write_exr_depth
    from .atlas_scene_chisel import project_source_depth

    photos = prepare_source_photographs(frames=frames, root=root / 'photographs')
    by_id = {frame['frame_id']: frame for frame in frames}
    contexts, uploads = [], {}
    for index, photo in enumerate(photos):
        key = f'context-{index}'
        uploads[key + '-rgb'] = {'path': photo['image_path'], 'sha256': photo['image_digest']}
        linear = write_exr_depth(root / key / 'depth.exr', project_source_depth(
            [by_id[photo['source_frame_id']]], photo['camera']))
        uploads[key + '-depth'] = {'path': str(linear), 'sha256': _sha256_file(linear)}
        contexts.append({'imageAsset': {'assetId': 'upload:' + key + '-rgb'},
                         'depth': {'depthAsset': {'assetId': 'upload:' + key + '-depth'}},
                         'camera': photo['camera']})
    return contexts, uploads, photos


def generate_rgbd_completion(*, frames, selected, candidates, root, pose, context,
                             configuration, task_runner, max_polls, scene_caption='',
                             checkpoint=None, geometry_frames=None, reference_frames=None,
                             exported_gap_feedback=False, extent_frames=None, **_):
    """Generate native RGBD, then admit additions only in explicit missing regions."""
    from .atlas_result_bundle import _completed_task
    from .atlas_scene_chisel import prepare_depth_context, target_batches
    from .atlas_scene_completion import load_generated_views
    from .atlas_scene_preservation import (
        prepare_completion_views, preserve_generated_view, observed_scene_bounds, _save_image,
    )
    from .website_atlas_tasks import validate_task_request

    if not selected or any(not row.get('completion_gap_mask_path') for row in selected):
        raise ValueError('atlas_generate_named_gap_masks_required')
    root.mkdir(parents=True, exist_ok=True)
    context_frames = frames if reference_frames is None else reference_frames
    contexts, uploads, photos = prepare_rgbd_context(frames=context_frames, root=root / 'rgbd-context')
    groups = target_batches(selected=selected, candidates=candidates, source_count=len(contexts))
    chosen = list({row['camera_id']: row for group in groups for row in group}.values())
    references = frames if geometry_frames is None else geometry_frames
    extent = observed_scene_bounds(frames if extent_frames is None else extent_frames)
    rows = prepare_completion_views(frames=context_frames, selected=chosen, root=root / 'source-footprints')
    depths = prepare_depth_context(frames=references, selected=chosen, root=root / 'depth-context')
    selected_by_id = {row['camera_id']: row for row in selected}
    for row in rows:
        depth = depths[row['frame_id']]
        row.update(source_depth_context_path=depth['path'], source_depth_context_digest=depth['sha256'])
        target = selected_by_id.get(row['frame_id'])
        if target is None:
            path = root / 'source-footprints' / (row['frame_id'] + '-no-fill.png')
            _save_image(path, np.zeros((720, 1280), np.uint8))
            row.update(completion_gap_mask_path=str(path), completion_gap_mask_digest=_sha256_file(path),
                       completion_gap_ids=[], completion_gap_pixel_count=0)
        else:
            row.update({key: target[key] for key in ('completion_gap_mask_path', 'completion_gap_mask_digest',
                                                    'completion_gap_ids', 'completion_gap_pixel_count')})
            if exported_gap_feedback:
                from PIL import Image

                path = Path(target['completion_gap_mask_path'])
                if _sha256_file(path) != target['completion_gap_mask_digest']:
                    raise ValueError('atlas_completion_gap_mask_changed')
                gap = np.asarray(Image.open(path).convert('L')) == 255
                known = np.asarray(Image.open(depth['path']).convert('L')).copy()
                if known.shape != gap.shape:
                    raise ValueError('atlas_completion_gap_mask_grid_changed')
                # A photograph can supply correct colour while its foreground
                # geometry is absent from the export. Depth of a farther wall
                # must not freeze that named foreground hole. Native context
                # depth and source free-space rejection remain unchanged.
                known[gap] = 0
                output = root / 'depth-context' / (row['frame_id'] + '-export-coverage.png')
                _save_image(output, known)
                row.update(source_depth_context_path=str(output),
                           source_depth_context_digest=_sha256_file(output))
    by_id = {row['frame_id']: row for row in rows}
    # Refuse before reserving/uploading if a selected new view has effectively
    # no photographed surroundings. All reference images still accompany every
    # request; this checks overlap from the requested target camera itself.
    for target in selected:
        known = by_id[target['camera_id']]['protected_pixel_count']
        if known <= 0 or known < target['completion_gap_pixel_count']:
            raise ValueError('atlas_generate_target_context_insufficient')
    caption = scene_caption.strip() + '\n' + (
        'The exact photographed room with the same layout, materials, lighting, fixed fixtures '
        'and existing possessions in their photographed positions. Missing areas continue only '
        'the bare surfaces at their visible boundaries. Cleared openings remain open and empty. '
        'No additional rooms, fixtures, furniture, possessions or decorations.')
    policy, specs, needed = configuration['policy'], [], 0.
    for index, group in enumerate(groups):
        request = {'contextFrames': contexts, 'targetCameras': [row['camera'] for row in group],
                   'prompt': caption, 'enhancePrompt': False, 'returnDepth': True}
        validate_task_request('atlasGenerate', request)
        stage = root / 'generated' / str(index)
        cost = render_cost(policy, len(group))
        quote = {'schema_version': 'website_atlas_task_price_review.v1', 'task': 'atlasGenerate',
                 'request_digest': canonical_digest(request), 'maximum_cost_usd': cost,
                 **{key: policy[key] for key in ('rate_card_reference', 'reviewed_by', 'reviewed_on')}}
        if (stage / 'result.json').is_file():
            _completed_task(stage, pose['binding']['source_video_digest'], 'generated')
        else:
            needed += cost
        specs.append((group, request, stage, quote))
    # Reserve room for joint registration before buying novel RGB. Count the
    # real source photographs and distinct selected cameras, not conditioning
    # anchors repeated in every output batch.
    source_ids = {frame['frame_id'] for frame in frames}
    registration_images = sum(photo['source_frame_id'] in source_ids for photo in photos) + len(selected_by_id)
    registration_quote = registration_images * policy.get('pose_input_cost_usd', 0)
    if needed + registration_quote + final_output_cost(policy) + 1 > configuration['remaining_cost_usd']:
        raise ValueError('atlas_completion_remaining_budget_insufficient')
    binding = {'schema_version': 'atlas_generate_completion_plan.v1', 'source_pose_digest': pose['digest'],
               'source_photographs': photos, 'prepared_views': rows, 'native_keep_masks_sent': False,
               'selected_camera_ids': list(selected_by_id), 'generation_extent': extent,
               'batches': [{'frame_ids': [row['camera_id'] for row in group],
                           'request_digest': canonical_digest(request), 'maximum_cost_usd': quote['maximum_cost_usd']}
                          for group, request, _, quote in specs]}
    binding['digest'] = canonical_digest(binding, digest_field='digest')
    plan = root / 'rgbd-completion-plan.json'
    if plan.exists() and json.loads(plan.read_text()) != binding:
        raise ValueError('atlas_completion_generate_plan_changed')
    write_json(plan, binding)
    assets = [plan, *[Path(row['path']) for row in uploads.values()],
              *[Path(row[key]) for row in rows for key in ('image_path', 'keep_mask_path', 'completion_gap_mask_path')]]
    generated, images, roots, digests, receipts, seen = [], [], [], [], [], set()
    for batch_index, (group, request, stage, quote) in enumerate(specs):
        if not (stage / 'atlas_pose/submission.json').exists() and not (stage / 'result.json').exists():
            forecast = completion_request_preflight(configuration,
                generation_roots=[spec[2] for spec in specs[batch_index:]],
                final_output_root=root.parent, context=context,
                source_digest=pose['binding']['source_video_digest'])
            write_json(root / 'request-budget-preflight.json', forecast)
            if not forecast['sufficient_for_known_calls']:
                raise ValueError('atlas_completion_request_budget_insufficient:'
                    f"required={forecast['minimum_remaining_requests']},remaining={forecast['remaining_requests']}")
        if checkpoint:
            checkpoint('automation_completion_generate_rgbd')
        state = task_runner(task='atlasGenerate', request=request, uploads=uploads, output_root=stage,
            source_video_digest=pose['binding']['source_video_digest'], task_context=context,
            price_review=quote, max_polls=max_polls)
        if state['status'] != 'completed':
            return None, {'status': 'processing', 'stage': 'scene_completion', 'operation_id': state.get('operation_id')}
        native, returned, native_images = load_generated_views(stage, ['completion-' + root.name + '-' + row['camera_id'] for row in group])
        roots.append(str(stage))
        digests.append(native['digest'])
        for native_index, (frame, target) in enumerate(zip(returned, group, strict=True)):
            if target['camera_id'] in seen:
                continue
            seen.add(target['camera_id'])
            fixed, image, receipt, paths = preserve_generated_view(generated=frame,
                prepared=by_id[target['camera_id']], references=references,
                output_root=root / 'preserved' / f'{batch_index:04d}-{native_index:04d}',
                native_result_digest=native['digest'], native_frame_index=native_index, extent_bounds=extent)
            generated.append(fixed)
            images.append({**image, 'source_frame_id': target['source_frame_id'],
                           'native_image_path': native_images[native_index]['image_path'],
                           'native_image_digest': native_images[native_index]['image_digest']})
            receipts.append(receipt['digest'])
            assets.extend(paths)
    report = {'status': 'completed', 'generation_method': 'atlasGenerate_posed_rgbd_photographs',
              'generated_roots': roots, 'native_result_digests': digests,
              'preservation_receipt_digests': receipts, 'mask_plan_digest': binding['digest'],
              'known_pixels_changed': 0, 'source_geometry_modified': False,
              'generated_geometry_pixel_count': int(sum(np.count_nonzero(row['confidence']) for row in generated)),
              'preservation_paths': [str(path) for path in assets]}
    report['digest'] = canonical_digest(report, digest_field='digest')
    return (generated, images), report
