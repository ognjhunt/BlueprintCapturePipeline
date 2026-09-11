"""Fixed-state native mesh/ParticleField visibility diagnostic; no policy or provider."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from blueprint_pipeline.decision_evidence_contracts import canonical_digest

REQUEST_SCHEMA = 'native_task_composition_diagnostic_request.v1'
RESULT_SCHEMA = 'native_task_composition_diagnostic.v1'
PASSES = ('full', 'appearance_only', 'native_meshes_only')


class CompositionDiagnosticError(ValueError):
    pass


def seal(value, field='receipt_digest'):
    value[field] = canonical_digest(value, digest_field=field)
    return value


def file_record(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise CompositionDiagnosticError('composition_input_missing_or_symlink')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'sha256': 'sha256:' + digest, 'size_bytes': path.stat().st_size}


def verify_record(path, expected):
    actual = file_record(path)
    if (actual['size_bytes'] != expected.get('size_bytes')
            or actual['sha256'].removeprefix('sha256:') != str(expected.get('sha256', '')).removeprefix('sha256:')):
        raise CompositionDiagnosticError('composition_input_digest_mismatch')
    return Path(path)


def validate_request(request):
    if (request.get('schema_version') != REQUEST_SCHEMA
            or request.get('request_digest') != canonical_digest(request, digest_field='request_digest')
            or request.get('passes') != list(PASSES)
            or request.get('camera_role') not in {'external', 'overview'}
            or request.get('target_semantic_class') not in {'task_support', 'task_object', 'task_target_marker'}
            or request.get('policy_queries_permitted') != 0
            or request.get('physics_steps_between_passes_permitted') != 0
            or request.get('source_asset_mutation_permitted') is not False
            or type(request.get('render_refresh_count')) is not int
            or not 2 <= request['render_refresh_count'] <= 64):
        raise CompositionDiagnosticError('composition_request_invalid')
    return request


@dataclass(frozen=True)
class CompositionAdapters:
    """Visibility operations target only geometry; lights and physics stay fixed."""
    snapshot_visibility: Callable[[], Any]
    apply_visibility: Callable[[str], None]
    restore_visibility: Callable[[Any], None]
    read_fixed_state: Callable[[], dict]
    render_and_capture: Callable[[str, Path], dict]
    sensor_generation: Callable[[], int]


def _capture_records(row, root):
    paths = []
    for key in ('rgb_png', 'metric_depth', 'semantic_segmentation'):
        record = row.get(key, {})
        relative = PurePosixPath(str(record.get('path', '')))
        if relative.is_absolute() or '..' in relative.parts or str(relative) in {'', '.'}:
            raise CompositionDiagnosticError('composition_aov_path_invalid')
        path = root / relative
        actual = file_record(path)
        if actual['sha256'].removeprefix('sha256:') != str(record.get('sha256', '')).removeprefix('sha256:'):
            raise CompositionDiagnosticError('composition_aov_digest_mismatch')
        paths.append({'role': key, 'relative_path': str(relative), **actual})
    if (row['metric_depth'].get('status') not in {'valid', 'invalid_native_aov'}
            or row['metric_depth'].get('aov') != 'distance_to_camera'):
        raise CompositionDiagnosticError('composition_metric_depth_required')
    return paths


def compare_pass_pixels(rows, output_root, *, target_class):
    """Report paired raw pixels; Gaussian distance AOV semantics remain unqualified."""
    import numpy as np
    from PIL import Image
    from .native_task_camera_observability import measure_native_task_semantic_label_pixels
    by_pass = {row['pass']: row['camera'] for row in rows}
    full = by_pass['full']
    info = full['semantic_segmentation']['id_to_labels'] or {}
    labels = info.get('idToLabels', info)
    semantic = np.load(output_root / 'full' / full['semantic_segmentation']['path'], allow_pickle=False)
    measured = measure_native_task_semantic_label_pixels(
        semantic_ids=semantic, id_to_labels=labels, target_label=target_class)
    ids = measured['target_semantic_ids']
    mask = np.isin(semantic, ids)
    if not mask.any():
        raise CompositionDiagnosticError('composition_full_pass_target_semantic_pixels_missing')
    mesh_camera = by_pass['native_meshes_only']
    mesh_semantic = np.load(output_root / 'native_meshes_only' / mesh_camera['semantic_segmentation']['path'], allow_pickle=False)
    mesh_info = mesh_camera['semantic_segmentation']['id_to_labels'] or {}
    mesh_measurement = measure_native_task_semantic_label_pixels(
        semantic_ids=mesh_semantic, id_to_labels=mesh_info.get('idToLabels', mesh_info),
        target_label=target_class)
    mesh_mask = np.isin(mesh_semantic, mesh_measurement['target_semantic_ids'])
    occluded = mesh_mask & ~mask
    rgb, depth = {}, {}
    for name, row in by_pass.items():
        rgb[name] = np.asarray(Image.open(output_root / name / row['rgb_png']['path']))
        depth[name] = np.squeeze(np.load(output_root / name / row['metric_depth']['path'], allow_pickle=False))
        if rgb[name].shape[:2] != mask.shape or depth[name].shape != mask.shape:
            raise CompositionDiagnosticError('composition_paired_aov_shape_mismatch')
    changed = np.any(rgb['full'] != rgb['native_meshes_only'], axis=-1) & mask
    valid = mask & np.isfinite(depth['appearance_only']) & np.isfinite(depth['native_meshes_only'])
    delta = np.full(mask.shape, np.nan, dtype=np.float32)
    np.subtract(depth['appearance_only'], depth['native_meshes_only'], out=delta,
                where=np.isfinite(depth['appearance_only']) & np.isfinite(depth['native_meshes_only']))
    output = output_root / 'pixel_comparison'
    output.mkdir()
    artifacts = []
    for name, values in [('full_target_semantic_mask', mask), ('full_vs_mesh_rgb_changed', changed),
                         ('native_mesh_target_semantic_mask', mesh_mask),
                         ('mesh_target_occluded_in_full', occluded),
                         ('appearance_minus_mesh_distance_m', delta)]:
        path = output / (name + '.npy')
        np.save(path, values, allow_pickle=False)
        artifacts.append({'relative_path': str(path.relative_to(output_root)), **file_record(path)})
    return {'target_semantic_class': target_class, 'target_semantic_ids': ids,
        'target_pixel_count': int(mask.sum()), 'rgb_changed_target_pixel_count': int(changed.sum()),
        'native_mesh_target_pixel_count': int(mesh_mask.sum()),
        'native_mesh_target_pixels_occluded_in_full': int(occluded.sum()),
        'finite_paired_distance_target_pixel_count': int(valid.sum()),
        'changed_pixels_appearance_distance_nearer': int((changed & valid & (delta < 0)).sum()),
        'changed_pixels_appearance_distance_deeper': int((changed & valid & (delta > 0)).sum()),
        'changed_pixels_distances_equal': int((changed & valid & (delta == 0)).sum()),
        'gaussian_distance_aov_surface_equivalence_qualified': False,
        'rgb_difference_is_not_primary_ray_contribution_proof': True, 'artifacts': artifacts}


def run_composition_diagnostic(request, *, output_root, adapters: CompositionAdapters):
    """Retain all three raw native passes and restore even when a capture fails."""
    validate_request(request)
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise CompositionDiagnosticError('composition_output_must_be_fresh')
    output_root.mkdir(parents=True, exist_ok=True)
    visibility = adapters.snapshot_visibility()
    baseline = adapters.read_fixed_state()
    fixed_digest = canonical_digest(baseline)
    rows = []
    generation = adapters.sensor_generation()
    failure = None
    try:
        for index, label in enumerate(PASSES):
            adapters.restore_visibility(visibility)
            adapters.apply_visibility(label)
            if canonical_digest(adapters.read_fixed_state()) != fixed_digest:
                raise CompositionDiagnosticError('composition_fixed_state_changed')
            pass_root = output_root / label
            pass_root.mkdir()
            row = adapters.render_and_capture(label, pass_root)
            next_generation = adapters.sensor_generation()
            if next_generation <= generation:
                raise CompositionDiagnosticError('composition_sensor_buffer_not_refreshed')
            generation = next_generation
            if canonical_digest(adapters.read_fixed_state()) != fixed_digest:
                raise CompositionDiagnosticError('composition_fixed_state_changed')
            calibration = {key: row.get(key) for key in ('intrinsic_matrix', 'position_world_m', 'quaternion_world_opengl_xyzw', 'resolution_hw')}
            if any(value is None for value in calibration.values()):
                raise CompositionDiagnosticError('composition_native_camera_calibration_missing')
            if rows and calibration != rows[0]['camera_calibration']:
                raise CompositionDiagnosticError('composition_camera_changed_between_passes')
            records = _capture_records(row, pass_root)
            # The shared writer also serves episode captures; correct its generic
            # synchronization label for this render-only, no-step diagnostic.
            row = {**row, 'synchronization': 'native_render_and_forced_sensor_refresh_without_physics_step'}
            rows.append(seal({'pass': label, 'ordinal': index, 'camera': row,
                'camera_calibration': calibration, 'sensor_generation': generation,
                'artifacts': records, 'fixed_state_digest': fixed_digest}))
            (pass_root / 'capture.json').write_text(json.dumps(rows[-1], indent=2, sort_keys=True) + '\n')
    except Exception as exc:
        failure = type(exc).__name__ + ':' + str(exc)
    finally:
        adapters.restore_visibility(visibility)
    if adapters.snapshot_visibility() != visibility:
        raise CompositionDiagnosticError('composition_visibility_restoration_failed')
    final_state = adapters.read_fixed_state()
    comparison = None
    if failure is None:
        try:
            comparison = compare_pass_pixels(rows, output_root, target_class=request['target_semantic_class'])
        except Exception as exc:
            failure = type(exc).__name__ + ':' + str(exc)
    result = seal({'schema_version': RESULT_SCHEMA, 'status': 'captured' if failure is None else 'blocked',
        'request_digest': request['request_digest'], 'passes': rows, 'fixed_state': baseline,
        'fixed_state_digest': fixed_digest, 'visibility_restored': True, 'pixel_comparison': comparison,
        'policy_queries': 0,
        'physics_steps_between_passes': final_state['physics_step_index'] - baseline['physics_step_index'],
        'final_fixed_state': final_state, 'source_assets_mutated': False,
        'metric_depth_usable_for_occlusion': len(rows) == len(PASSES) and all(row['camera']['metric_depth']['status'] == 'valid' for row in rows),
        'pixel_cause_proven': False, 'physical_truth_claimed': False,
        'blockers': [] if failure is None else [failure]})
    (output_root / 'composition_diagnostic.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result
