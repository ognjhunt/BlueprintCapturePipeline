"""Project approved context, keep its RGB fixed, and fill only unobserved space.

These are source-derived candidate views, not new capture observations. Native
provider outputs remain untouched. Derived masks state which generated depth
may contribute; known surfaces and observed free space always take precedence.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .atlas_depth_splat import _rotation_matrix, MAX_MEDIAN_CROSS_VIEW_RESIDUAL
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file


def _world(frame, valid):
    y, x = np.nonzero(valid)
    k = frame['intrinsics']
    z = frame['depth'][y, x]
    local = np.column_stack(((x + .5 - k['cx']) / k['fx'] * z,
                             -(y + .5 - k['cy']) / k['fy'] * z, -z))
    return local @ frame['rotation'].T + frame['position'], y, x


def _project(points, camera):
    k, e = camera['intrinsics'], camera['extrinsics']
    local = (points - np.asarray(e['position'])) @ _rotation_matrix(e['quaternion'])
    z = -local[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = local[:, 0] / z * k['fx'] + k['cx']
        v = -local[:, 1] / z * k['fy'] + k['cy']
    inside = ((z > 0) & np.isfinite(u) & np.isfinite(v) &
              (u >= 0) & (u < k['width']) & (v >= 0) & (v < k['height']))
    return z, u, v, inside


def _valid(frame):
    return (np.isfinite(frame['depth']) & (frame['depth'] > 0) &
            np.isfinite(frame['confidence']) & (frame['confidence'] > 0))


def _same_camera(frame, camera):
    return (np.allclose(frame['position'], camera['extrinsics']['position'], atol=1e-8, rtol=0)
            and np.allclose(frame['rotation'], _rotation_matrix(camera['extrinsics']['quaternion']),
                            atol=1e-8, rtol=0))


def supported_projection_frames(frames):
    """Conservative RGBD projection candidates, never a modification of source.

    A valid depth alone is insufficient to lock a colour into a new viewpoint.
    Drop depth boundaries and require another view to agree in depth and RGB.
    The RGB tolerance is a projection heuristic, not calibrated confidence.
    Actual source viewpoints still preserve every source RGB pixel.
    """
    from .atlas_depth_splat import EDGE_RELATIVE_JUMP

    filtered = []
    for source in frames:
        valid = _valid(source)
        depth = source['depth']
        center = depth[1:-1, 1:-1]
        interior = valid[1:-1, 1:-1].copy()
        for neighbour in (depth[1:-1, 2:], depth[1:-1, :-2], depth[2:, 1:-1], depth[:-2, 1:-1]):
            interior &= np.isfinite(neighbour) & (np.abs(neighbour - center) <= EDGE_RELATIVE_JUMP * center)
        valid = np.zeros(valid.shape, bool)
        valid[1:-1, 1:-1] = interior
        points, y, x = _world(source, valid)
        supported = np.zeros(len(points), bool)
        colors = source['rgb'][y, x].astype(np.int16)
        for other in frames:
            if other is source:
                continue
            camera = {'intrinsics': other['intrinsics'], 'extrinsics': other['extrinsics']}
            z, u, v, inside = _project(points, camera)
            ids = np.flatnonzero(inside)
            rows, cols = np.floor(v[inside]).astype(int), np.floor(u[inside]).astype(int)
            observed = other['depth'][rows, cols]
            agrees = (_valid(other)[rows, cols]
                      & (np.abs(z[inside] - observed) <= EDGE_RELATIVE_JUMP * observed)
                      & (np.max(np.abs(colors[ids] - other['rgb'][rows, cols].astype(np.int16)), axis=-1) <= 32))
            supported[ids[agrees]] = True
        keep = np.zeros(valid.shape, bool)
        keep[y[supported], x[supported]] = True
        filtered.append({**source, 'confidence': source['confidence'] * keep})
    return filtered


def _save_image(path, pixels):
    from PIL import Image

    if path.is_symlink():
        raise ValueError('atlas_completion_preservation_inputs_changed')
    if path.exists():
        if not np.array_equal(np.asarray(Image.open(path)), pixels):
            raise ValueError('atlas_completion_preservation_inputs_changed')
    else:
        Image.fromarray(pixels).save(path)


def project_known_context(frames, camera):
    """Nearest-surface RGB projection; an exact source camera also keeps all RGB.

    The latter is independent of missing depth: no depth is necessary to know
    what an already photographed pixel looks like at the same camera centre
    and orientation. Intrinsics alone define that resampling, with no crop.
    """
    import cv2

    k = camera['intrinsics']
    height, width = k['height'], k['width']
    rgb = np.full((height, width, 3), 127, np.uint8)
    zbuffer = np.full(height * width, np.inf)
    keep = np.zeros(height * width, bool)
    same = [frame for frame in frames if _same_camera(frame, camera)]
    # An actual source viewpoint needs no estimated-depth warp. Keep its
    # authoritative RGB footprint clean; the other paired views provide
    # cross-view context to the inpainting model instead of noisy edge splats.
    for frame in ([] if same else frames):
        points, y, x = _world(frame, _valid(frame))
        z, u, v, inside = _project(points, camera)
        source_indices = np.flatnonzero(inside)
        flat = np.floor(v[inside]).astype(int) * width + np.floor(u[inside]).astype(int)
        order = np.lexsort((z[inside], flat))
        sorted_flat = flat[order]
        first = np.r_[True, sorted_flat[1:] != sorted_flat[:-1]] if len(order) else np.array([], bool)
        chosen = source_indices[order[first]]
        pixels = sorted_flat[first]
        nearer = z[chosen] < zbuffer[pixels]
        chosen, pixels = chosen[nearer], pixels[nearer]
        rgb.reshape(-1, 3)[pixels] = frame['rgb'][y[chosen], x[chosen]]
        zbuffer[pixels] = z[chosen]
        keep[pixels] = True
    keep = keep.reshape(height, width)
    for frame in same[:1]:
        source = frame['intrinsics']
        source_k = np.array([[source['fx'], 0, source['cx']], [0, source['fy'], source['cy']], [0, 0, 1.]])
        target_k = np.array([[k['fx'], 0, k['cx']], [0, k['fy'], k['cy']], [0, 0, 1.]])
        # Camera matrices use pixel centres; OpenCV uses integer pixel indices.
        to_center = np.array([[1, 0, .5], [0, 1, .5], [0, 0, 1.]])
        mapping = np.linalg.inv(to_center) @ target_k @ np.linalg.inv(source_k) @ to_center
        known = cv2.warpPerspective(np.ones(frame['rgb'].shape[:2], np.uint8), mapping, (width, height),
                                    flags=cv2.INTER_NEAREST).astype(bool)
        sampled = cv2.warpPerspective(frame['rgb'], mapping, (width, height),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rgb[known], keep[known] = sampled[known], True
    return rgb, keep


def unobserved_geometry_mask(generated, references):
    """Reject generated surfaces in known geometry or observed free space.

    A point behind an observed surface may be unseen/occluded; a point in
    front of it contradicts the observation. No source geometry is rewritten.
    """
    valid = _valid(generated)
    points, y, x = _world(generated, valid)
    allowed = np.ones(len(points), bool)
    for reference in references:
        camera = {'intrinsics': reference['intrinsics'], 'extrinsics': reference['extrinsics']}
        z, u, v, inside = _project(points, camera)
        indices = np.flatnonzero(inside)
        rows, cols = np.floor(v[inside]).astype(int), np.floor(u[inside]).astype(int)
        observed = reference['depth'][rows, cols]
        known = _valid(reference)[rows, cols]
        blocked = known & (z[inside] <= observed * (1 + MAX_MEDIAN_CROSS_VIEW_RESIDUAL))
        allowed[indices[blocked]] = False
    mask = np.zeros(valid.shape, bool)
    mask[y[allowed], x[allowed]] = True
    return mask


def prepare_completion_views(*, frames, selected, root):
    """Prepare camera-paired grayscale keep masks without a provider call."""
    import cv2

    root.mkdir(parents=True, exist_ok=True)
    supported = supported_projection_frames(frames)
    rows = []
    for index, view in enumerate(selected):
        same = any(_same_camera(frame, view['camera']) for frame in frames)
        rgb, keep = project_known_context(supported, view['camera'])
        if not same:
            # Isolated forward-projected pixels and narrow jagged edges are
            # not stable image patches. Leave them fillable and neutral; a
            # speckled keep mask otherwise forces Atlas to reproduce noise.
            keep = cv2.erode(keep.astype(np.uint8), np.ones((3, 3), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
            rgb[~keep] = 127
        image, mask = root / f'{index:04d}-context.png', root / f'{index:04d}-keep.png'
        _save_image(image, rgb)
        _save_image(mask, keep.astype(np.uint8) * 255)
        rows.append({'frame_id': view['camera_id'], 'source_frame_id': view['source_frame_id'],
                     'camera': view['camera'], 'image_path': str(image), 'image_digest': _sha256_file(image),
                     'keep_mask_path': str(mask), 'keep_mask_digest': _sha256_file(mask),
                     'projection_basis': ('source_camera_rgb' if same else 'cross_view_supported_rgbd_dense_patches'),
                     'protected_pixel_count': int(keep.sum()), 'total_pixel_count': int(keep.size),
                     'clean_anchor': view.get('preservation_anchor', False),
                     'edit_pixels': int(keep.sum())})
    return rows


def preserve_generated_view(*, generated, prepared, references, output_root, native_result_digest,
                            native_frame_index):
    """Keep source-derived RGB exactly; exclude known regions from new geometry."""
    from PIL import Image

    output_root.mkdir(parents=True, exist_ok=True)
    for key in ('image', 'keep_mask'):
        if _sha256_file(Path(prepared[key + '_path'])) != prepared[key + '_digest']:
            raise ValueError('atlas_completion_preservation_inputs_changed')
    rgb = np.asarray(Image.open(prepared['image_path']).convert('RGB'))
    keep = np.asarray(Image.open(prepared['keep_mask_path']).convert('L')) == 255
    if rgb.shape != generated['rgb'].shape or keep.shape != generated['depth'].shape:
        raise ValueError('atlas_completion_preservation_grid_changed')
    composite = generated['rgb'].copy()
    composite[keep] = rgb[keep]
    allowed = unobserved_geometry_mask(generated, references) & ~keep
    frame = {**generated, 'rgb': composite, 'confidence': generated['confidence'] * allowed}
    image, known_mask, geometry_mask = (output_root / n for n in ('image.png', 'known.png', 'generated-geometry.png'))
    _save_image(image, composite)
    _save_image(known_mask, keep.astype(np.uint8) * 255)
    _save_image(geometry_mask, allowed.astype(np.uint8) * 255)
    assets = [{'role': role, 'path': str(path), 'sha256': _sha256_file(path)} for role, path in
              (('composite_rgb', image), ('known_pixels', known_mask), ('generated_geometry_mask', geometry_mask))]
    report = {'schema_version': 'atlas_scene_preservation.v1', 'frame_id': generated['frame_id'],
              'native_result_digest': native_result_digest, 'native_frame_index': native_frame_index,
              'camera': prepared['camera'], 'context_image_digest': prepared['image_digest'],
              'keep_mask_digest': prepared['keep_mask_digest'], 'protected_pixel_count': int(keep.sum()),
              'projection_basis': prepared.get('projection_basis'),
              'native_changed_protected_pixel_count': int(np.any(generated['rgb'] != rgb, axis=-1)[keep].sum()),
              'native_protected_mean_absolute_difference': (
                  float(np.abs(generated['rgb'].astype(float) - rgb.astype(float))[keep].mean()) if keep.any() else None),
              'known_pixels_changed': int(np.any(composite != rgb, axis=-1)[keep].sum()),
              'generated_geometry_pixel_count': int(allowed.sum()),
              'native_valid_geometry_pixel_count': int(_valid(generated).sum()),
              'depth_basis': 'unchanged_native_depth_with_generated_geometry_mask',
              'confidence_rule': 'native_confidence_times_generated_geometry_mask',
              'source_geometry_modified': False, 'native_outputs_modified': False,
              'claim_ceiling': 'development_only', 'assets': assets}
    report['digest'] = canonical_digest(report, digest_field='digest')
    receipt = output_root / 'preservation.json'
    write_json(receipt, report)
    paths = [image, known_mask, geometry_mask, receipt]
    return frame, {'frame_id': frame['frame_id'], 'image_path': str(image), 'image_digest': _sha256_file(image),
                   'novel_view': True}, report, paths


def generate_masked_completion(*, frames, selected, candidates, root, pose, context, configuration,
                               task_runner, max_polls, removed_objects=(), checkpoint=None):
    """Use the existing inpainting/batching adapter; preserve every native result."""
    import json
    from .atlas_background_automation import batches, masked_request
    from .atlas_result_bundle import _completed_task
    from .atlas_scene_completion import load_generated_views

    # An observed camera anchors every batch. It is a conditioning view, not
    # an input-frame ceiling; all planner-selected cameras remain in the plan.
    anchor = next(row for row in candidates if row['purpose'] == 'wide')
    anchor = {**anchor, 'preservation_anchor': True}
    chosen = [anchor, *[row for row in selected if row['camera_id'] != anchor['camera_id']]]
    center = np.asarray(anchor['camera']['extrinsics']['position'])
    if all(np.linalg.norm(np.asarray(row['camera']['extrinsics']['position']) - center) < 1e-6 for row in chosen):
        translated = next((row for row in candidates if
            np.linalg.norm(np.asarray(row['camera']['extrinsics']['position']) - center) >= 1e-6), None)
        if translated is None:
            raise ValueError('atlas_completion_depth_requires_translation')
        chosen.append(translated)
    rows = prepare_completion_views(frames=frames, selected=chosen, root=root / 'masked-context')
    groups = batches(rows)
    policy = configuration['policy']
    specs = []
    caption = ('The exact environment shown in the context photographs, with the same architecture, materials, '
               'lighting, layout, fixtures and existing objects. Observed surfaces and objects retain their '
               'original appearance and positions. Previously unseen space continues the existing surfaces '
               'naturally, without additional objects, appliances, furniture, clutter or decorations. '
               'The following task objects are absent: ' + json.dumps(list(removed_objects)) + '.')
    remaining = 0.
    for index, group in enumerate(groups):
        request, uploads = masked_request(group, caption)
        # Supply a finished-scene caption ourselves. The default enhancer
        # misdescribed the real countertop and must not rewrite this boundary.
        request['enhancePrompt'] = False
        if request['returnDepth'] is not True:
            raise ValueError('atlas_completion_depth_requires_translation')
        stage = root / 'generated' / str(index)
        cost = round(len(group) * policy['rgbd_view_cost_usd'] * 2, 6)
        quote = {'schema_version': 'website_atlas_task_price_review.v1', 'task': 'atlasMasked',
                 'request_digest': canonical_digest(request), 'maximum_cost_usd': cost,
                 **{k: policy[k] for k in ('rate_card_reference', 'reviewed_by', 'reviewed_on')}}
        if (stage / 'result.json').is_file():
            _completed_task(stage, pose['binding']['source_video_digest'], 'generated')
        else:
            remaining += cost
        specs.append((group, request, uploads, stage, quote))
    if remaining + policy.get('mesh_export_cost_usd', 0) + 1 > configuration['remaining_cost_usd']:
        raise ValueError('atlas_completion_remaining_budget_insufficient')
    binding = {'schema_version': 'atlas_masked_completion_plan.v1', 'source_pose_digest': pose['digest'],
               'selected_camera_ids': [row['camera_id'] for row in selected],
               'anchor_camera_id': anchor['camera_id'], 'prepared_views': rows,
               'additional_conditioning_camera_ids': [row['camera_id'] for row in chosen if
                   row['camera_id'] not in {view['camera_id'] for view in selected}],
               'batches': [{'frame_ids': [row['frame_id'] for row in group], 'request_digest': canonical_digest(request),
                            'maximum_cost_usd': quote['maximum_cost_usd']} for group, request, _, _, quote in specs]}
    binding['digest'] = canonical_digest(binding, digest_field='digest')
    plan_path = root / 'masked-completion-plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != binding:
        raise ValueError('atlas_completion_masked_plan_changed')
    write_json(plan_path, binding)
    assets = [plan_path, *[Path(row[key]) for row in rows for key in ('image_path', 'keep_mask_path')]]
    generated, images, roots, digests, receipts, seen = [], [], [], [], [], set()
    for index, (group, request, uploads, stage, quote) in enumerate(specs):
        if checkpoint:
            checkpoint('automation_completion_masked_rgbd')
        state = task_runner(task='atlasMasked', request=request, uploads=uploads, output_root=stage,
            source_video_digest=pose['binding']['source_video_digest'], task_context=context,
            price_review=quote, max_polls=max_polls)
        if state['status'] != 'completed':
            return None, {'status': 'processing', 'stage': 'scene_completion', 'operation_id': state.get('operation_id')}
        result, returned, _ = load_generated_views(stage, ['completion-' + row['frame_id'] for row in group])
        roots.append(str(stage))
        digests.append(result['digest'])
        for native_index, (frame, row) in enumerate(zip(returned, group, strict=True)):
            if row['frame_id'] in seen:
                continue
            seen.add(row['frame_id'])
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
              'generation_method': 'atlasMasked_unknown_regions_only', 'known_pixels_changed': 0,
              'source_geometry_modified': False, 'preservation_paths': [str(p) for p in assets]}
    report['digest'] = canonical_digest(report, digest_field='digest')
    return (generated, images), report
