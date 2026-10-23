"""Photograph-conditioned scene completion without locking projected fragments.

ADP-040 / Day 28: reuse admitted Atlas tasks and the existing RGBD consumer.
Full photograph coverage is tiled onto the native grid with calibrated crops.
Native outputs remain retained; source geometry remains the reconstruction base.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file


def prepare_source_photographs(*, frames, root):
    """Cover every source pixel with overlapping 16:9 photographs, without padding.

    Chisel center-crops other aspect ratios. Explicit crops and transformed K
    prevent that hidden loss. A centered crop also supplies a reconstruction
    anchor where the source principal point permits one.
    """
    from PIL import Image
    from .atlas_scene_preservation import _save_image

    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, frame in enumerate(frames):
        k = frame['intrinsics']
        width, height = k['width'], k['height']
        if frame['rgb'].shape != (height, width, 3):
            raise ValueError('atlas_chisel_source_grid_changed')
        crop_width = min(width, height * 1280 / 720)
        crop_height = crop_width * 720 / 1280
        xs = np.linspace(0, width - crop_width, max(1, math.ceil((width - crop_width) / (.75 * crop_width)) + 1))
        ys = np.linspace(0, height - crop_height, max(1, math.ceil((height - crop_height) / (.75 * crop_height)) + 1))
        boxes = [(float(x), float(y), float(x + crop_width), float(y + crop_height)) for x in xs for y in ys]
        # Ensure an eligible centered source photograph without replacing the
        # edge tiles: all original coverage must survive the native crop.
        x, y = k['cx'] - crop_width / 2, k['cy'] - crop_height / 2
        if 0 <= x <= width - crop_width and 0 <= y <= height - crop_height:
            centered = (x, y, x + crop_width, y + crop_height)
            if not any(np.allclose(centered, box, atol=1e-8, rtol=0) for box in boxes):
                boxes.append(centered)
        image = Image.fromarray(frame['rgb'])
        for tile, (left, top, right, bottom) in enumerate(boxes):
            pixels = np.asarray(image.resize((1280, 720), Image.Resampling.LANCZOS,
                                             box=(left, top, right, bottom)))
            path = root / f'{index:04d}-{tile:04d}.png'
            _save_image(path, pixels)
            scale = 1280 / (right - left)
            camera = deepcopy({'intrinsics': k, 'extrinsics': frame['extrinsics']})
            camera['intrinsics'].update(width=1280, height=720, fx=k['fx'] * scale,
                fy=k['fy'] * scale, cx=(k['cx'] - left) * scale, cy=(k['cy'] - top) * scale)
            rows.append({'frame_id': f'{frame["frame_id"]}-photo-{tile}', 'source_frame_id': frame['frame_id'],
                'image_path': str(path), 'image_digest': _sha256_file(path), 'camera': camera,
                'source_grid': [width, height], 'source_rectangle': [left, top, right, bottom],
                'pixel_scale': scale, 'input_basis': 'approved_cleaned_photograph'})
    return rows


def target_batches(*, selected, candidates, source_count):
    """Respect the native rollout limit while retaining every photo per batch."""
    capacity = min(32, 64 - source_count)
    wide = [row for row in candidates if row['purpose'] == 'wide']
    if capacity < 2 or not wide:
        raise ValueError('atlas_chisel_source_rollout_capacity_insufficient')
    first = wide[0]
    center = first['camera']['extrinsics']['position']
    second = next((row for row in wide if math.dist(center, row['camera']['extrinsics']['position']) >= 1e-6), None)
    if second is None:
        raise ValueError('atlas_chisel_observed_translation_missing')
    anchors = [first, second]
    ids = {row['camera_id'] for row in anchors}
    others = [row for row in selected if row['camera_id'] not in ids]
    if others and capacity == 2:
        raise ValueError('atlas_chisel_source_rollout_capacity_insufficient')
    groups = [[*anchors, *others[i:i + capacity - 2]] for i in range(0, len(others), max(1, capacity - 2))]
    return groups or [anchors]


def generate_chisel_completion(*, frames, selected, candidates, root, pose, context, configuration,
                               task_runner, max_polls, removed_objects=(), checkpoint=None):
    """Use the existing admitted task lifecycle, then the existing RGBD fusion seam."""
    from .atlas_result_bundle import _completed_task
    from .atlas_scene_completion import load_generated_views
    from .atlas_scene_preservation import prepare_completion_views, preserve_generated_view
    from .website_atlas_tasks import validate_task_request

    sources = prepare_source_photographs(frames=frames, root=root / 'source-photographs')
    groups = target_batches(selected=selected, candidates=candidates, source_count=len(sources))
    chosen = list({row['camera_id']: row for group in groups for row in group}.values())
    # These footprints are derived locally for source-priority fusion and
    # inspection only. No masks or projected RGB canvases are sent to Chisel.
    rows = prepare_completion_views(frames=frames, selected=chosen, root=root / 'source-footprints')
    by_id = {row['frame_id']: row for row in rows}
    uploads = {f'photo-{i}': {'path': row['image_path'], 'sha256': row['image_digest']} for i, row in enumerate(sources)}
    source_frames = [{'imageAsset': {'assetId': f'upload:photo-{i}'}, 'camera': row['camera']}
                     for i, row in enumerate(sources)]
    caption = ('The exact environment in the source photographs, preserving its architecture, materials, '
               'lighting, layout, fixtures and existing objects. Continue unseen surfaces naturally. '
               'Do not add appliances, furniture, clutter or decorations. Empty installation openings remain empty. '
               'Absent task objects: ' + json.dumps(list(removed_objects)) + '.')
    policy = configuration['policy']
    specs, remaining = [], 0.
    for index, group in enumerate(groups):
        request = {'sourceFrames': source_frames, 'contextFrames': [], 'targetCameras': [row['camera'] for row in group],
                   'prompt': caption, 'enhancePrompt': False, 'returnDepth': True}
        validate_task_request('atlasChisel', request)
        stage = root / 'generated' / str(index)
        # Reviewed RGBD rate already includes depth; charge output views once.
        cost = round(len(group) * policy['rgbd_view_cost_usd'], 6)
        quote = {'schema_version': 'website_atlas_task_price_review.v1', 'task': 'atlasChisel',
                 'request_digest': canonical_digest(request), 'maximum_cost_usd': cost,
                 **{k: policy[k] for k in ('rate_card_reference', 'reviewed_by', 'reviewed_on')}}
        if (stage / 'result.json').is_file():
            _completed_task(stage, pose['binding']['source_video_digest'], 'generated')
        else:
            remaining += cost
        specs.append((group, request, stage, quote))
    if remaining + policy.get('mesh_export_cost_usd', 0) + 1 > configuration['remaining_cost_usd']:
        raise ValueError('atlas_completion_remaining_budget_insufficient')
    binding = {'schema_version': 'atlas_chisel_completion_plan.v1', 'source_pose_digest': pose['digest'],
               'selected_camera_ids': [row['camera_id'] for row in selected], 'source_photographs': sources,
               'prepared_views': rows, 'native_keep_masks_sent': False,
               'batches': [{'frame_ids': [row['camera_id'] for row in group], 'request_digest': canonical_digest(request),
                            'maximum_cost_usd': quote['maximum_cost_usd']} for group, request, _, quote in specs]}
    binding['digest'] = canonical_digest(binding, digest_field='digest')
    plan_path = root / 'chisel-completion-plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != binding:
        raise ValueError('atlas_completion_chisel_plan_changed')
    write_json(plan_path, binding)
    assets = [plan_path, *[Path(row['image_path']) for row in sources],
              *[Path(row[key]) for row in rows for key in ('image_path', 'keep_mask_path')]]
    generated, images, roots, digests, receipts, seen = [], [], [], [], [], set()
    for index, (group, request, stage, quote) in enumerate(specs):
        if checkpoint:
            checkpoint('automation_completion_chisel_rgbd')
        state = task_runner(task='atlasChisel', request=request, uploads=uploads, output_root=stage,
            source_video_digest=pose['binding']['source_video_digest'], task_context=context,
            price_review=quote, max_polls=max_polls)
        if state['status'] != 'completed':
            return None, {'status': 'processing', 'stage': 'scene_completion', 'operation_id': state.get('operation_id')}
        result, returned, _ = load_generated_views(stage, ['completion-' + row['camera_id'] for row in group])
        roots.append(str(stage))
        digests.append(result['digest'])
        for native_index, (frame, target) in enumerate(zip(returned, group, strict=True)):
            if target['camera_id'] in seen:
                continue
            seen.add(target['camera_id'])
            row = by_id[target['camera_id']]
            fixed, image, receipt, paths = preserve_generated_view(generated=frame, prepared=row, references=frames,
                output_root=root / 'preserved' / f'{index:04d}-{native_index:04d}',
                native_result_digest=result['digest'], native_frame_index=native_index)
            image['source_frame_id'] = row['source_frame_id']
            generated.append(fixed)
            images.append(image)
            receipts.append(receipt['digest'])
            assets.extend(paths)
    report = {'status': 'completed', 'native_result_digests': digests, 'generated_roots': roots,
              'preservation_receipt_digests': receipts, 'mask_plan_digest': binding['digest'],
              'generation_method': 'atlasChisel_posed_photographs_no_keep_masks', 'known_pixels_changed': 0,
              'source_geometry_modified': False, 'preservation_paths': [str(p) for p in assets]}
    report['digest'] = canonical_digest(report, digest_field='digest')
    return (generated, images), report
