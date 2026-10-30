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
from .atlas_pricing import render_cost, final_output_cost
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file


def oriented_photograph(frame):
    """Rotate portrait pixels and camera together; preserve every captured ray."""
    from scipy.spatial.transform import Rotation

    camera = deepcopy({'intrinsics': frame['intrinsics'], 'extrinsics': frame['extrinsics']})
    rgb, k = frame['rgb'], camera['intrinsics']
    rotated = k['height'] > k['width']
    if rotated:
        width, height, fx, fy, cx, cy = (k[key] for key in ('width', 'height', 'fx', 'fy', 'cx', 'cy'))
        rgb = np.rot90(rgb, -1)
        k.update(width=height, height=width, fx=fy, fy=fx, cx=height - cy, cy=cx)
        camera['extrinsics']['quaternion'] = Rotation.from_matrix(
            frame['rotation'] @ Rotation.from_euler('Z', 90, degrees=True).as_matrix()).as_quat().tolist()
    return rgb, camera, rotated


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
        rgb, oriented_camera, rotated = oriented_photograph(frame)
        k = oriented_camera['intrinsics']
        width, height = k['width'], k['height']
        if rgb.shape != (height, width, 3):
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
        image = Image.fromarray(rgb)
        for tile, (left, top, right, bottom) in enumerate(boxes):
            pixels = np.asarray(image.resize((1280, 720), Image.Resampling.LANCZOS,
                                             box=(left, top, right, bottom)))
            path = root / f'{index:04d}-{tile:04d}.png'
            _save_image(path, pixels)
            scale = 1280 / (right - left)
            camera = deepcopy(oriented_camera)
            camera['intrinsics'].update(width=1280, height=720, fx=k['fx'] * scale,
                fy=k['fy'] * scale, cx=(k['cx'] - left) * scale, cy=(k['cy'] - top) * scale)
            rows.append({'frame_id': f'{frame["frame_id"]}-photo-{tile}', 'source_frame_id': frame['frame_id'],
                'image_path': str(path), 'image_digest': _sha256_file(path), 'camera': camera,
                'source_grid': [frame['intrinsics']['width'], frame['intrinsics']['height']],
                'oriented_grid': [width, height], 'source_rectangle': [left, top, right, bottom],
                'clockwise_quarter_turn_with_camera': rotated,
                'pixel_scale': scale, 'input_basis': 'approved_cleaned_photograph'})
    return rows


def target_batches(*, selected, candidates, source_count):
    """Respect the native rollout limit while retaining every photo per batch."""
    # One depth-context view accompanies each generated camera.
    capacity = min(32, (64 - source_count) // 2)
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


def project_source_depth(frames, camera):
    """Render retained estimated depth into the target; unknown pixels stay zero.

    A same-centre inverse warp keeps continuous surfaces continuous without
    forward-projection holes. Other observed centres contribute nearest depth.
    This is soft geometric conditioning, never an RGB keep mask.
    """
    import cv2
    from .atlas_scene_preservation import _same_center, _valid, _world, _project
    from .atlas_depth_splat import _rotation_matrix

    k = camera['intrinsics']
    width, height = k['width'], k['height']
    target_k = np.array([[k['fx'], 0, k['cx']], [0, k['fy'], k['cy']], [0, 0, 1.]])
    target_rotation = _rotation_matrix(camera['extrinsics']['quaternion'])
    zbuffer = np.full((height, width), np.inf, np.float32)
    for frame in frames:
        if not _same_center(frame, camera):
            points, _, _ = _world(frame, _valid(frame))
            z, u, v, inside = _project(points, camera)
            pixels = np.floor(v[inside]).astype(int) * width + np.floor(u[inside]).astype(int)
            np.minimum.at(zbuffer.reshape(-1), pixels, z[inside])
            continue
        source = frame['intrinsics']
        source_k = np.array([[source['fx'], 0, source['cx']], [0, source['fy'], source['cy']], [0, 0, 1.]])
        optical_to_rub = np.diag([1., -1., -1.])
        rotation = optical_to_rub @ target_rotation.T @ frame['rotation'] @ optical_to_rub
        to_center = np.array([[1, 0, .5], [0, 1, .5], [0, 0, 1.]])
        mapping = np.linalg.inv(to_center) @ target_k @ rotation @ np.linalg.inv(source_k) @ to_center
        source_depth = np.where(_valid(frame), frame['depth'], 0).astype(np.float32)
        sampled = cv2.warpPerspective(source_depth, mapping, (width, height), flags=cv2.INTER_NEAREST)
        yy, xx = np.indices((height, width))
        ray = (rotation.T @ np.linalg.inv(target_k))[2]
        source_z = ray[0] * (xx + .5) + ray[1] * (yy + .5) + ray[2]
        valid = (sampled > 0) & (source_z > 0)
        projected = np.full((height, width), np.inf, np.float32)
        projected[valid] = sampled[valid] / source_z[valid]
        np.minimum(zbuffer, projected, out=zbuffer)
    return np.where(np.isfinite(zbuffer), zbuffer, 0).astype(np.float32)


def prepare_depth_context(*, frames, selected, root):
    from .atlas_scene_preservation import _save_image

    root.mkdir(parents=True, exist_ok=True)
    rows = {}
    for index, row in enumerate(selected):
        depth = project_source_depth(frames, row['camera'])
        valid = depth > 0
        zmin, zmax = ((float(depth[valid].min()), float(depth[valid].max())) if valid.any() else (1., 1.))
        pixels = np.zeros(depth.shape, np.uint8)
        pixels[valid] = (np.clip(np.round(255 * (1 - np.log(depth[valid] / zmin) / math.log(zmax / zmin))), 1, 255)
                         .astype(np.uint8) if zmax > zmin else 255)
        path = root / f'{index:04d}-depth.png'
        _save_image(path, pixels)
        rows[row['camera_id']] = {'path': str(path), 'sha256': _sha256_file(path),
            'zMin': zmin, 'zMax': zmax, 'known_depth_pixel_count': int(valid.sum()),
            'total_pixel_count': int(valid.size), 'basis': 'retained_source_estimated_planar_depth'}
    return rows


def generate_chisel_completion(*, frames, selected, candidates, root, pose, context, configuration,
                               task_runner, max_polls, removed_objects=(), checkpoint=None,
                               scene_caption='', view_gap_reasons=None, geometry_frames=None):
    """Use the existing admitted task lifecycle, then the existing RGBD fusion seam."""
    from .atlas_result_bundle import _completed_task
    from .atlas_scene_completion import load_generated_views
    from .atlas_scene_preservation import prepare_completion_views, preserve_generated_view, observed_scene_bounds
    from .website_atlas_tasks import validate_task_request

    sources = prepare_source_photographs(frames=frames, root=root / 'source-photographs')
    references = frames if geometry_frames is None else geometry_frames
    extent = observed_scene_bounds(frames)
    groups = target_batches(selected=selected, candidates=candidates, source_count=len(sources))
    chosen = list({row['camera_id']: row for group in groups for row in group}.values())
    # These footprints are derived locally for source-priority fusion and
    # inspection only. No masks or projected RGB canvases are sent to Chisel.
    rows = prepare_completion_views(frames=frames, selected=chosen, root=root / 'source-footprints')
    depth_context = prepare_depth_context(frames=references, selected=chosen, root=root / 'depth-context')
    selected_by_id = {row['camera_id']: row for row in selected}
    for row in rows:
        source_depth = depth_context[row['frame_id']]
        row.update(source_depth_context_path=source_depth['path'], source_depth_context_digest=source_depth['sha256'])
        if any(view.get('completion_gap_mask_path') for view in selected):
            target = selected_by_id.get(row['frame_id'])
            if target is None:
                # Extra translated anchors condition depth; they are not fill requests.
                from .atlas_scene_preservation import _save_image
                path = root / 'source-footprints' / (row['frame_id'] + '-no-fill.png')
                _save_image(path, np.zeros((720, 1280), np.uint8))
                row.update(completion_gap_mask_path=str(path), completion_gap_mask_digest=_sha256_file(path),
                           completion_gap_ids=[], completion_gap_pixel_count=0)
            else:
                row.update({key: target[key] for key in ('completion_gap_mask_path', 'completion_gap_mask_digest',
                                                        'completion_gap_ids', 'completion_gap_pixel_count')})
    by_id = {row['frame_id']: row for row in rows}
    photo_uploads = {f'photo-{i}': {'path': row['image_path'], 'sha256': row['image_digest']} for i, row in enumerate(sources)}
    source_frames = [{'imageAsset': {'assetId': f'upload:photo-{i}'}, 'camera': row['camera']}
                     for i, row in enumerate(sources)]
    caption = (scene_caption.strip() + '\n' if scene_caption.strip() else '') + (
        'Photorealistic continuation of this exact captured room. Existing geometry, materials, lighting, '
        'fixtures and visible objects remain in their photographed positions. '
        'Newly revealed gaps contain only bare continuous architectural and support surfaces matching '
        'their visible edges: floor, walls, ceiling and empty work or storage surfaces where applicable. '
        'Cleared openings remain open, unobstructed space. Newly revealed areas are empty of movable objects. '
        'The scene stays inside the captured room, without additional rooms, objects or fixtures.')
    policy = configuration['policy']
    specs, remaining = [], 0.
    for index, group in enumerate(groups):
        uploads, contexts = dict(photo_uploads), []
        for i, target in enumerate(group):
            depth = depth_context[target['camera_id']]
            uploads[f'depth-{i}'] = {key: depth[key] for key in ('path', 'sha256')}
            contexts.append({'camera': target['camera'], 'depth': {'logDepthAsset': {'assetId': f'upload:depth-{i}'},
                'zMin': depth['zMin'], 'zMax': depth['zMax']}})
        request = {'sourceFrames': source_frames, 'contextFrames': contexts, 'targetCameras': [row['camera'] for row in group],
                   'prompt': caption, 'enhancePrompt': False, 'returnDepth': True}
        validate_task_request('atlasChisel', request)
        stage = root / 'generated' / str(index)
        # Reviewed RGBD rate already includes depth; charge output views once.
        cost = render_cost(policy, len(group))
        quote = {'schema_version': 'website_atlas_task_price_review.v1', 'task': 'atlasChisel',
                 'request_digest': canonical_digest(request), 'maximum_cost_usd': cost,
                 **{k: policy[k] for k in ('rate_card_reference', 'reviewed_by', 'reviewed_on')}}
        if (stage / 'result.json').is_file():
            _completed_task(stage, pose['binding']['source_video_digest'], 'generated')
        else:
            remaining += cost
        specs.append((group, request, uploads, stage, quote))
    if remaining + final_output_cost(policy) + 1 > configuration['remaining_cost_usd']:
        raise ValueError('atlas_completion_remaining_budget_insufficient')
    binding = {'schema_version': 'atlas_chisel_completion_plan.v1', 'source_pose_digest': pose['digest'],
               'selected_camera_ids': [row['camera_id'] for row in selected], 'source_photographs': sources,
               'prepared_views': rows, 'native_keep_masks_sent': False, 'depth_context': depth_context,
               'generation_extent': extent, 'view_gap_reasons': view_gap_reasons or {},
               'additional_conditioning_camera_ids': {row['camera_id']: 'observed_pose_baseline_for_depth'
                   for row in chosen if row['camera_id'] not in {view['camera_id'] for view in selected}},
               'batches': [{'frame_ids': [row['camera_id'] for row in group], 'request_digest': canonical_digest(request),
                            'maximum_cost_usd': quote['maximum_cost_usd']} for group, request, _, _, quote in specs]}
    binding['digest'] = canonical_digest(binding, digest_field='digest')
    plan_path = root / 'chisel-completion-plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != binding:
        raise ValueError('atlas_completion_chisel_plan_changed')
    write_json(plan_path, binding)
    assets = [plan_path, *[Path(row['image_path']) for row in sources],
              *[Path(row['path']) for row in depth_context.values()],
              *[Path(row[key]) for row in rows for key in ('image_path', 'keep_mask_path')],
              *[Path(row['completion_gap_mask_path']) for row in rows if row.get('completion_gap_mask_path')]]
    generated, images, roots, digests, receipts, seen = [], [], [], [], [], set()
    source_depth_hole_pixels = 0
    for index, (group, request, uploads, stage, quote) in enumerate(specs):
        if checkpoint:
            checkpoint('automation_completion_chisel_rgbd')
        state = task_runner(task='atlasChisel', request=request, uploads=uploads, output_root=stage,
            source_video_digest=pose['binding']['source_video_digest'], task_context=context,
            price_review=quote, max_polls=max_polls)
        if state['status'] != 'completed':
            return None, {'status': 'processing', 'stage': 'scene_completion', 'operation_id': state.get('operation_id')}
        result, returned, _ = load_generated_views(stage, ['completion-' + root.name + '-' + row['camera_id'] for row in group])
        roots.append(str(stage))
        digests.append(result['digest'])
        for native_index, (frame, target) in enumerate(zip(returned, group, strict=True)):
            if target['camera_id'] in seen:
                continue
            seen.add(target['camera_id'])
            row = by_id[target['camera_id']]
            fixed, image, receipt, paths = preserve_generated_view(generated=frame, prepared=row, references=references,
                output_root=root / 'preserved' / f'{index:04d}-{native_index:04d}',
                native_result_digest=result['digest'], native_frame_index=native_index, extent_bounds=extent)
            image['source_frame_id'] = row['source_frame_id']
            generated.append(fixed)
            images.append(image)
            receipts.append(receipt['digest'])
            source_depth_hole_pixels += receipt['generated_source_depth_hole_pixel_count']
            assets.extend(paths)
    report = {'status': 'completed', 'native_result_digests': digests, 'generated_roots': roots,
              'preservation_receipt_digests': receipts, 'mask_plan_digest': binding['digest'],
              'generation_method': 'atlasChisel_posed_photographs_and_source_depth_no_keep_masks', 'known_pixels_changed': 0,
              'source_geometry_modified': False, 'preservation_paths': [str(p) for p in assets]}
    report['generated_source_depth_hole_pixel_count'] = source_depth_hole_pixels
    report['digest'] = canonical_digest(report, digest_field='digest')
    return (generated, images), report
