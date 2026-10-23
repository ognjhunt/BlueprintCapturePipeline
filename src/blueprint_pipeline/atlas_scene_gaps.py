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


def gap_labels(depth, photographed, *, scene_outline=False, candidate_mask=None,
               expected_depth=None):
    """Connected missing surfaces inside photographed or enclosed coverage.

    A mask never turns the exterior black canvas into a whole-room request.
    Resolution-relative opening removes raster pinholes, not large scene gaps.
    """
    import cv2
    from scipy.ndimage import binary_fill_holes

    covered = np.isfinite(depth) & (depth > 0)
    if expected_depth is not None:
        from .atlas_depth_splat import MAX_MEDIAN_CROSS_VIEW_RESIDUAL

        expected = np.asarray(expected_depth)
        if expected.shape != depth.shape:
            raise ValueError('atlas_completion_expected_depth_grid_changed')
        expected_valid = np.isfinite(expected) & (expected > 0)
        # A wall/floor behind a missing foreground surface does not close its
        # hole. Reviewed generated depth is a planning hypothesis, never an
        # authority to overwrite a nearer retained surface or source free space.
        covered &= ~(expected_valid &
                     (depth > expected * (1 + MAX_MEDIAN_CROSS_VIEW_RESIDUAL)))
        photographed = photographed | expected_valid
    inside = photographed | binary_fill_holes(covered)
    if scene_outline:
        # Only a candidate planning domain: source free-space/extent rejection
        # and native RGBD review still decide whether generated depth is usable.
        # It bridges open disocclusions between projected scene surfaces, without
        # making the exterior canvas an instruction to invent an entire room.
        y, x = np.nonzero(covered)
        if len(x) >= 3:
            hull = cv2.convexHull(np.column_stack((x, y)).astype(np.int32))
            outline = np.zeros(depth.shape, np.uint8)
            cv2.fillConvexPoly(outline, hull, 1)
            inside |= outline.astype(bool)
    if candidate_mask is not None:
        inside &= candidate_mask
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


def project_exported_splat(splat, camera):
    """CPU coverage diagnostic of retained export centers, not a beauty render.

    A one-pixel support border suppresses point-raster pinholes. Large missing
    patches remain visible; no triangle or room surface is invented here.
    """
    import cv2
    from .atlas_scene_preservation import _project
    from .splat_scene_analysis import DEFAULT_VISIBLE_OPACITY

    k = camera['intrinsics']
    shape = (int(k['height']), int(k['width']))
    points = splat.xyz[np.isfinite(splat.xyz).all(axis=1) &
                       (splat.opacity_sigmoid >= DEFAULT_VISIBLE_OPACITY)]
    z, u, v, inside = _project(points, camera)
    pixels = np.floor(v[inside]).astype(int) * shape[1] + np.floor(u[inside]).astype(int)
    depth = np.full(shape, np.inf, np.float32)
    np.minimum.at(depth.reshape(-1), pixels, z[inside])
    depth = cv2.erode(depth, np.ones((3, 3), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=float('inf'))
    return np.where(np.isfinite(depth), depth, 0).astype(np.float32)


def expected_depth_for_candidate(frames, candidate):
    """Keep the surface hypothesis paired with its reviewed reference view."""
    from .atlas_scene_chisel import project_source_depth

    identity = candidate.get('expected_frame_id')
    chosen = [frame for frame in frames if frame['frame_id'] == identity] if identity else frames
    if identity and len(chosen) != 1:
        raise ValueError('atlas_completion_expected_reference_missing')
    return project_source_depth(chosen, candidate['camera'])


def survey_gaps(*, frames, geometry_frames, candidates, root, expected_frames=(),
                splat_root=None):
    """Survey all known directions; the existing production agent chooses gaps."""
    from .atlas_scene_chisel import project_source_depth
    from .atlas_scene_preservation import prepare_completion_views, _save_image
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    binding = {'candidates': candidates, 'geometry_frames': [{
        'frame_id': f['frame_id'], 'intrinsics': f['intrinsics'], 'extrinsics': f['extrinsics'],
        **{key: hashlib.sha256(np.ascontiguousarray(f[key]).tobytes()).hexdigest()
           for key in ('rgb', 'depth', 'confidence')}} for f in geometry_frames],
        'photograph_frame_ids': [f['frame_id'] for f in frames]}
    if expected_frames:
        binding['expected_frames'] = [{
            'frame_id': f['frame_id'], 'intrinsics': f['intrinsics'], 'extrinsics': f['extrinsics'],
            **{key: hashlib.sha256(np.ascontiguousarray(f[key]).tobytes()).hexdigest()
               for key in ('rgb', 'depth', 'confidence')}} for f in expected_frames]
    splat = None
    if splat_root is not None:
        from .atlas_depth_splat import decode_spz

        manifest = json.loads((splat_root / 'atlas_depth_splat_manifest.json').read_text())
        if manifest.get('digest') != canonical_digest(manifest, digest_field='digest'):
            raise ValueError('atlas_completion_export_changed')
        asset = next(r for r in manifest['outputs'] if r['role'] == 'splat_spz')
        path = splat_root / asset['path']
        if path.is_symlink() or not path.resolve().is_relative_to(splat_root.resolve()) or _sha256_file(path) != asset['sha256']:
            raise ValueError('atlas_completion_export_changed')
        binding['export_manifest_digest'] = manifest['digest']
        splat = decode_spz(path.read_bytes())
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
    # Use the very same supported patches/erosion as the generation handoff.
    # Raw projected points can look like context while the usable patches are
    # actually empty, especially when looking at the back of a surface.
    prepared = prepare_completion_views(frames=frames, selected=candidates, root=root / 'context')
    rows, previews = [], []
    for index, (candidate, source) in enumerate(zip(candidates, prepared, strict=True)):
        rgb = np.asarray(Image.open(source['image_path']).convert('RGB'))
        photographed = np.asarray(Image.open(source['keep_mask_path']).convert('L')) == 255
        depth = (project_exported_splat(splat, candidate['camera']) if splat is not None
                 else project_source_depth(geometry_frames, candidate['camera']))
        expected = (expected_depth_for_candidate(expected_frames, candidate)
                    if expected_frames else None)
        outline = candidate.get('gap_domain') == 'projected_scene_outline_hypothesis'
        domain = None
        if candidate.get('gap_target_polygon_pixels') is not None:
            import cv2
            domain = np.zeros(depth.shape, np.uint8)
            cv2.fillConvexPoly(domain, np.rint(candidate['gap_target_polygon_pixels']).astype(np.int32), 1)
        labels, regions = gap_labels(depth, photographed, scene_outline=outline,
                                     candidate_mask=None if domain is None else domain.astype(bool),
                                     expected_depth=expected)
        path = root / f'{index:04d}-gaps.png'
        _save_image(path, labels)
        rows.append({'camera_id': candidate['camera_id'], 'source_frame_id': candidate['source_frame_id'],
                     'label_path': str(path), 'label_sha256': _sha256_file(path),
                     'gaps': [{**r, 'gap_id': candidate['camera_id'] + '-gap-' + str(r['label'])} for r in regions],
                     'gap_domain': ('projected_scene_outline_hypothesis' if outline else 'photographed_or_enclosed'),
                     'gap_pixel_count': int(np.count_nonzero(labels)),
                     'known_context_pixel_count': int(photographed.sum()),
                     'total_pixel_count': int(labels.size)})
        if candidate['purpose'] in {'wide', 'translated', 'disocclusion', 'gap_target'}:
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
              'basis': ('exported_splat_centers_and_reviewed_depth_hypotheses' if splat is not None
                        else 'retained_rgbd_geometry_coverage_not_splat_alpha'),
              'legend': 'Magenta identifies candidate missing geometry; black exterior is not a fill target.',
              'renderer': ('atlas_exported_center_projection.v1' if splat is not None
                           else 'atlas_planar_depth_projection.v1'),
              'claim_ceiling': 'development_only'}
    report['digest'] = canonical_digest(report, digest_field='digest')
    write_json(root / 'survey.json', report)
    return report


def context_supported_candidates(candidates, survey):
    """Keep observed surroundings at least as large as the requested fill.

    The requirement scales with each gap, rather than imposing a frame count
    or a fixed scene-wide percentage. Unsupported directions remain in the
    full topology survey, but cannot be purchased as grounded RGBD views.
    """
    views = {row['camera_id']: row for row in survey['views']}
    accepted, rejected = [], []
    for candidate in candidates:
        row = views[candidate['camera_id']]
        known, missing = row['known_context_pixel_count'], row['gap_pixel_count']
        # A camera may cover several gaps; the agent can request a supported
        # subset. The purchase boundary checks the exact selected mask area.
        smallest = min((region['pixel_count'] for region in row.get('gaps', [])), default=missing)
        if known <= 0 or known < smallest:
            rejected.append(candidate['camera_id'])
            continue
        accepted.append({**candidate,
                         'known_context_fraction': known / row['total_pixel_count'],
                         'gap_fraction': missing / row['total_pixel_count']})
    return accepted, rejected


def target_gap_ids(view, candidate, requested):
    """Separate a new camera's actual fill regions from declared source lineage.

    A planner may cite both the originating gap and its projection into the
    new camera. Only the latter indexes this camera's mask. Unknown references,
    duplicates, or a plan with no actual target regions still refuse.
    """
    regions = {row['gap_id'] for row in view['gaps']}
    lineage = set(candidate.get('source_gap_ids', []))
    if (not requested or len(requested) != len(set(requested))
            or not set(requested) <= regions | lineage):
        raise ValueError('atlas_completion_named_gaps_required')
    targets = [key for key in requested if key in regions]
    if not targets:
        raise ValueError('atlas_completion_named_gaps_required')
    return targets


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
