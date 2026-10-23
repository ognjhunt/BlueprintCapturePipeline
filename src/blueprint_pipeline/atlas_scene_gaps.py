"""ADP-040/day-28: identify missing geometry before purchasing completion.

These CPU projections survey the retained RGBD reconstruction, not splat alpha
or raw observations. Outside background and sub-pixel sampling noise are not
generation targets. Every admitted contribution must land in a named gap.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file


def gap_labels(depth, photographed):
    """Connected missing surfaces inside photographed or enclosed coverage.

    A mask never turns the exterior black canvas into a whole-room request.
    Resolution-relative opening removes raster pinholes, not large scene gaps.
    """
    import cv2
    from scipy.ndimage import binary_fill_holes

    covered = np.isfinite(depth) & (depth > 0)
    inside = photographed | binary_fill_holes(covered)
    missing = (~covered & inside).astype(np.uint8)
    missing = cv2.morphologyEx(missing, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(missing, connectivity=8)
    minimum = max(9, round(depth.size / 4096))
    selected = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= minimum]
    result = np.zeros(depth.shape, np.uint16)
    regions = []
    for index, old in enumerate(selected, 1):
        result[labels == old] = index
        x, y, w, h, area = stats[old].tolist()
        regions.append({'label': index, 'pixel_count': area,
                        'box_xywh_normalized': [x / depth.shape[1], y / depth.shape[0],
                                                w / depth.shape[1], h / depth.shape[0]]})
    return result, regions


def survey_gaps(*, frames, geometry_frames, candidates, root):
    """Survey all known directions; the existing production agent chooses gaps."""
    from .atlas_scene_chisel import project_source_depth
    from .atlas_scene_preservation import project_known_context, _save_image

    root.mkdir(parents=True, exist_ok=True)
    binding = {'candidates': candidates, 'geometry_frames': [{
        'frame_id': f['frame_id'], 'intrinsics': f['intrinsics'], 'extrinsics': f['extrinsics'],
        **{key: hashlib.sha256(np.ascontiguousarray(f[key]).tobytes()).hexdigest()
           for key in ('rgb', 'depth', 'confidence')}} for f in geometry_frames],
        'photograph_frame_ids': [f['frame_id'] for f in frames]}
    binding_digest = canonical_digest(binding)
    saved = root / 'survey.json'
    if saved.is_file():
        report = json.loads(saved.read_text())
        if (report.get('digest') != canonical_digest(report, digest_field='digest')
                or report.get('binding_digest') != binding_digest):
            raise ValueError('atlas_completion_gap_survey_changed')
        assets = [(r['label_path'], r['label_sha256']) for r in report['views']]
        assets += [(r['path'], r['sha256']) for r in report['previews']]
        for path, digest in assets:
            if Path(path).is_symlink() or _sha256_file(Path(path)) != digest:
                raise ValueError('atlas_completion_gap_survey_changed')
        return report
    rows, previews = [], []
    for index, candidate in enumerate(candidates):
        rgb, photographed = project_known_context(frames, candidate['camera'])
        depth = project_source_depth(geometry_frames, candidate['camera'])
        labels, regions = gap_labels(depth, photographed)
        path = root / f'{index:04d}-gaps.png'
        _save_image(path, labels)
        rows.append({'camera_id': candidate['camera_id'], 'source_frame_id': candidate['source_frame_id'],
                     'label_path': str(path), 'label_sha256': _sha256_file(path),
                     'gaps': [{**r, 'gap_id': candidate['camera_id'] + '-gap-' + str(r['label'])} for r in regions],
                     'gap_pixel_count': int(np.count_nonzero(labels)),
                     'total_pixel_count': int(labels.size)})
        if candidate['purpose'] in {'wide', 'translated'}:
            preview = np.zeros_like(rgb)
            covered = depth > 0
            preview[covered] = rgb[covered]
            preview[labels > 0] = (255, 0, 200)
            image = root / f'{index:04d}-survey.png'
            _save_image(image, preview)
            previews.append({'path': str(image), 'sha256': _sha256_file(image),
                             'source_frame_id': candidate['source_frame_id'],
                             'camera_id': candidate['camera_id'], 'kind': 'geometry_gap_survey'})
    report = {'schema_version': 'atlas_geometry_gap_survey.v1', 'views': rows, 'previews': previews,
              'binding_digest': binding_digest,
              'basis': 'retained_rgbd_geometry_coverage_not_splat_alpha',
              'legend': 'Magenta identifies candidate missing geometry; black exterior is not a fill target.',
              'renderer': 'atlas_planar_depth_projection.v1', 'claim_ceiling': 'development_only'}
    report['digest'] = canonical_digest(report, digest_field='digest')
    write_json(root / 'survey.json', report)
    return report


def select_gap_mask(view, gap_ids, output):
    """Bind the agent's named regions to an exact downstream geometry mask."""
    from PIL import Image
    from .atlas_scene_preservation import _save_image

    path = Path(view['label_path'])
    if _sha256_file(path) != view['label_sha256']:
        raise ValueError('atlas_completion_gap_survey_changed')
    regions = {row['gap_id']: row['label'] for row in view['gaps']}
    if not gap_ids or len(set(gap_ids)) != len(gap_ids) or not set(gap_ids) <= regions.keys():
        raise ValueError('atlas_completion_named_gaps_required')
    labels = np.asarray(Image.open(path))
    mask = np.isin(labels, [regions[key] for key in gap_ids])
    _save_image(output, mask.astype(np.uint8) * 255)
    return {'completion_gap_mask_path': str(output), 'completion_gap_mask_digest': _sha256_file(output),
            'completion_gap_ids': list(gap_ids), 'completion_gap_pixel_count': int(mask.sum())}


def completion_plan_path(root):
    """New RGBD generation and retained Chisel runs keep their own receipts."""
    paths = [root / name for name in ('rgbd-completion-plan.json', 'chisel-completion-plan.json')]
    found = [path for path in paths if path.is_file()]
    if len(found) != 1 or found[0].is_symlink():
        raise ValueError('atlas_completion_prior_plan_missing_or_ambiguous')
    return found[0]


def retained_completion(*, root, pose_digest):
    """Adopt a passed prefix's exact fused pixels; never repurchase it."""
    from PIL import Image
    from .atlas_scene_completion import load_generated_views

    report = json.loads((root / 'completion.json').read_text())
    plan = json.loads(completion_plan_path(root).read_text())
    for value in (report, plan):
        if value.get('digest') != canonical_digest(value, digest_field='digest'):
            raise ValueError('atlas_completion_prior_changed')
    if report['status'] != 'passed' or plan['source_pose_digest'] != pose_digest:
        raise ValueError('atlas_completion_prior_incompatible')
    frames, seen_cameras = [], set()
    receipts = []
    for path in root.glob('preserved/*/preservation.json'):
        if path.is_symlink():
            raise ValueError('atlas_completion_prior_changed')
        receipts.append(json.loads(path.read_text()))
    for native_root, batch in zip(report['generated_roots'], plan['batches'], strict=True):
        result, native, _ = load_generated_views(Path(native_root), ['completion-' + i for i in batch['frame_ids']])
        for index, (frame, camera_id) in enumerate(zip(native, batch['frame_ids'], strict=True)):
            # Repeated native conditioning frames were fused only once.
            if camera_id in seen_cameras:
                continue
            seen_cameras.add(camera_id)
            matched = [row for row in receipts if row.get('native_result_digest') == result['digest']
                       and row.get('native_frame_index') == index]
            if len(matched) != 1:
                raise ValueError('atlas_completion_prior_preservation_missing')
            receipt = matched[0]
            if (receipt.get('digest') != canonical_digest(receipt, digest_field='digest')
                    or receipt['native_result_digest'] != result['digest']
                    or receipt['digest'] not in report['generation']['preservation_receipt_digests']):
                raise ValueError('atlas_completion_prior_changed')
            frame['frame_id'] = receipt['frame_id']
            assets = {}
            for row in receipt['assets']:
                path = Path(row['path'])
                if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or _sha256_file(path) != row['sha256']:
                    raise ValueError('atlas_completion_prior_changed')
                assets[row['role']] = path
            frame['rgb'] = np.asarray(Image.open(assets['composite_rgb']).convert('RGB'))
            mask = np.asarray(Image.open(assets['generated_geometry_mask']).convert('L')) == 255
            frame['confidence'] = frame['confidence'] * mask
            frames.append(frame)
    return frames, report
