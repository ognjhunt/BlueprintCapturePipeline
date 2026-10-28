"""Bounded Atlas completion without removing captured objects or changing source geometry."""
from __future__ import annotations

import json
from pathlib import Path

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file

APPEARANCE_BASIS = 'preserved_capture_with_bounded_completion'


def _splat_assets(root, manifest):
    paths = []
    for asset in manifest.get('outputs', []):
        path = Path(root) / asset['path']
        if (path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve())
                or not path.is_file() or _sha256_file(path) != asset['sha256']):
            raise ValueError('atlas_preserved_completion_export_changed')
        paths.append(path)
    if not paths:
        raise ValueError('atlas_preserved_completion_export_missing')
    return paths


def validate_preserved_completion(*, report, job_root, source_pose_digest,
                                  source_video_digest, task_context_digest, splat):
    """A geometry handoff rechecks the retained proof and exact consumed export."""
    expected = {'source_pose_digest': source_pose_digest, 'source_video_digest': source_video_digest,
                'task_context_digest': task_context_digest}
    if (report.get('schema_version') != 'atlas_preserved_completion.v1'
            or report.get('digest') != canonical_digest(report, digest_field='digest')
            or report.get('status') != 'completed' or report.get('context_basis') != 'preserved_capture'
            or any(report.get(key) != value for key, value in expected.items())
            or report.get('source_geometry_modified') is not False
            or type(report.get('known_pixels_changed')) is not int or report['known_pixels_changed'] != 0
            or report.get('claim_ceiling') != 'development_only'
            or any(report.get(key) is not False for key in ('whole_scene_complete',
                   'metric_measurement_proven', 'simulation_qualified', 'physical_evidence'))
            or not report.get('generated_frame_ids')
            or splat.get('digest') != report.get('splat_manifest_digest')
            or splat.get('digest') != canonical_digest(splat, digest_field='digest')
            or splat.get('pose_result_digest') != source_pose_digest
            or splat.get('generated_content') is not True
            or splat.get('appearance_basis') != APPEARANCE_BASIS
            or splat.get('generated_result_digests') != report.get('geometry_provenance')
            or splat.get('visual_review_digest') != report.get('visual_review_digest')
            or splat.get('rebuild_source_manifest_digest') != report.get('source_splat_manifest_digest')):
        raise ValueError('atlas_preserved_completion_not_bound')
    exported = {row['frame_id'] for row in splat.get('frames', [])}
    if exported != set(report['source_frame_ids']) | set(report['generated_frame_ids']):
        raise ValueError('atlas_preserved_completion_not_consumed')
    completion = report.get('completion') or {}
    registration = report.get('generated_view_registration') or {}
    recovery = report.get('geometry_recovery') or {}
    generation = completion.get('generation') or {}
    if (any(value.get('digest') != canonical_digest(value, digest_field='digest')
            for value in (completion, registration, recovery, generation))
            or completion.get('status') != 'passed'
            or completion.get('generated_geometry_ready') is not True
            or completion.get('review', {}).get('status') != 'passed'
            or any(completion.get(key) != value for key, value in expected.items())
            or completion.get('context_basis') != 'preserved_capture'
            or completion.get('generation_result_digest') != generation['digest']
            or generation.get('status') != 'completed'
            or type(generation.get('known_pixels_changed')) is not int or generation['known_pixels_changed'] != 0
            or generation.get('source_geometry_modified') is not False
            or not generation.get('native_result_digests')
            or registration.get('status') != 'completed'
            or registration.get('source_geometry_modified') is not False
            or registration.get('native_outputs_modified') is not False
            or registration.get('depth_basis') != 'joint_pose_depth_registered_to_retained_source_world'
            or set(registration.get('source_frame_ids', [])) != set(report['source_frame_ids'])
            or recovery.get('source_geometry_modified') is not False
            or type(recovery.get('known_pixels_changed')) is not int or recovery['known_pixels_changed'] != 0
            or type(recovery.get('exportable_samples')) is not int or recovery['exportable_samples'] <= 0
            or report['geometry_provenance'] != [generation['digest'], registration['digest'], recovery['digest']]
            or report['visual_review_digest'] != canonical_digest(completion['review'])):
        raise ValueError('atlas_preserved_completion_geometry_not_admitted')
    assets = report.get('assets')
    if not isinstance(assets, list) or not assets:
        raise ValueError('atlas_preserved_completion_evidence_missing')
    for asset in assets:
        path = Path(job_root) / asset['relative_path']
        if (path.is_symlink() or not path.resolve().is_relative_to(Path(job_root).resolve())
                or not path.is_file() or _sha256_file(path) != asset['sha256']):
            raise ValueError('atlas_preserved_completion_evidence_changed')


def run_preserved_completion(*, pose_root, pose, frames, review, root, job_root, context,
                             configuration, splat_root, source_survey_path, task_runner,
                             planner_runner, pose_runner, release_blockers, max_polls, checkpoint=None):
    """Survey source first, complete named gaps, register native views, then fuse real additions."""
    from .atlas_scene_completion import complete_scene
    from .atlas_scene_recovery import register_generated_completion, recover_reviewed_geometry, _frame_binding
    from .atlas_depth_splat import surface_samples, write_depth_splat

    root, job_root, splat_root = Path(root), Path(job_root), Path(splat_root)
    baseline = json.loads((splat_root / 'atlas_depth_splat_manifest.json').read_text())
    if (baseline.get('digest') != canonical_digest(baseline, digest_field='digest')
            or baseline.get('pose_result_digest') != pose['digest'] or baseline.get('generated_content')):
        raise ValueError('atlas_preserved_completion_source_export_changed')
    baseline_assets = _splat_assets(splat_root, baseline)
    source_bindings = [_frame_binding(frame) for frame in frames]
    generated, completion = complete_scene(pose_root=pose_root, pose=pose, frames=frames, review=review,
        root=root / 'completion', context=context, configuration=configuration, task_runner=task_runner,
        planner_runner=planner_runner, release_blockers=release_blockers, max_polls=max_polls,
        checkpoint=checkpoint, splat_root=splat_root, reference_frames=frames,
        source_survey_path=source_survey_path, context_basis='preserved_capture')
    if generated is None:
        return {**completion, 'stage': completion.get('stage', 'preserved_scene_completion'),
                'failure_class': ('generic_harness_gap' if completion.get('status') == 'source_recovery_required'
                                  else 'missing_evidence')}
    if (completion.get('digest') != canonical_digest(completion, digest_field='digest')
            or completion.get('status') != 'passed'
            or completion.get('generated_geometry_ready') is not True
            or completion.get('context_basis') != 'preserved_capture'
            or completion.get('source_pose_digest') != pose['digest']
            or completion.get('source_video_digest') != pose['binding']['source_video_digest']
            or completion.get('task_context_digest') != context['context_digest']):
        raise ValueError('atlas_preserved_completion_review_not_bound')
    if not generated:
        if completion.get('generated_roots') or completion.get('review', {}).get('status') != 'not_required':
            raise ValueError('atlas_preserved_completion_empty_geometry_not_bound')
        return {'status': 'completed', 'generated_geometry_admitted': False,
                'splat_root': str(splat_root), 'completion': completion, 'generated_roots': []}
    if completion.get('review', {}).get('status') != 'passed':
        raise ValueError('atlas_preserved_completion_review_not_bound')
    registered, registration = register_generated_completion(frames=frames, completion=completion,
        root=root / 'joint-registration', pose=pose, context=context, configuration=configuration,
        pose_runner=pose_runner, max_polls=max_polls, checkpoint=checkpoint)
    if registered is None:
        return {**registration, 'completion': completion,
                **({'failure_class': 'missing_evidence'} if registration.get('status') == 'needs_input' else {}),
                'stage': registration.get('stage', 'generated_view_registration')}
    additions, recovery = recover_reviewed_geometry(frames=frames, geometry_frames=frames,
        reviewed_views=registered, root=root / 'registered-geometry-recovery', checkpoint=checkpoint)
    additions = [frame for frame in additions if len(surface_samples(frame)['points'])]
    if not additions:
        return {'status': 'needs_input', 'reason': 'atlas_preserved_completion_no_exportable_geometry',
                'stage': 'preserved_scene_completion', 'completion': completion,
                'generated_view_registration': registration, 'geometry_recovery': recovery,
                'failure_class': 'missing_evidence'}
    if source_bindings != [_frame_binding(frame) for frame in frames]:
        raise ValueError('atlas_preserved_completion_modified_source')
    provenance = [completion['generation_result_digest'], registration['digest'], recovery['digest']]
    visual = canonical_digest(completion['review'])
    binding = {'source_pose_digest': pose['digest'], 'source_splat_manifest_digest': baseline['digest'],
               'geometry_provenance': provenance, 'source_frame_bindings': source_bindings}
    destination = root / ('splat-' + canonical_digest(binding)[7:23])
    saved = destination / 'atlas_depth_splat_manifest.json'
    manifest = (json.loads(saved.read_text()) if saved.is_file() else write_depth_splat(
        frames=[*frames, *additions], result=pose, output_root=destination,
        generated_result_digests=provenance, visual_review_digest=visual,
        rebuild_source_manifest_digest=baseline['digest'], generated_appearance_basis=APPEARANCE_BASIS))
    paths = [root / 'completion/completion.json', *map(Path, completion.get('preservation_paths', [])),
             *map(Path, registration['preservation_paths']), *map(Path, recovery['preservation_paths']),
             splat_root / 'atlas_depth_splat_manifest.json', saved,
             *baseline_assets, *_splat_assets(destination, manifest)]
    assets = []
    for path in sorted(set(paths)):
        if path.is_symlink() or not path.resolve().is_relative_to(job_root.resolve()) or not path.is_file():
            raise ValueError('atlas_preserved_completion_evidence_path_invalid')
        assets.append({'relative_path': str(path.resolve().relative_to(job_root.resolve())),
                       'sha256': _sha256_file(path)})
    report = {'schema_version': 'atlas_preserved_completion.v1', 'status': 'completed',
        'context_basis': 'preserved_capture', 'source_pose_digest': pose['digest'],
        'source_video_digest': pose['binding']['source_video_digest'],
        'task_context_digest': context['context_digest'], 'source_splat_manifest_digest': baseline['digest'],
        'splat_manifest_digest': manifest['digest'], 'geometry_provenance': provenance,
        'visual_review_digest': visual, 'source_frame_ids': [f['frame_id'] for f in frames],
        'generated_frame_ids': [f['frame_id'] for f in additions], 'completion': completion,
        'generated_view_registration': registration, 'geometry_recovery': recovery,
        'source_geometry_modified': False, 'known_pixels_changed': 0, 'assets': assets,
        'whole_scene_complete': False, 'metric_measurement_proven': False,
        'simulation_qualified': False, 'physical_evidence': False, 'claim_ceiling': 'development_only'}
    report['digest'] = canonical_digest(report, digest_field='digest')
    validate_preserved_completion(report=report, job_root=job_root, source_pose_digest=pose['digest'],
        source_video_digest=pose['binding']['source_video_digest'],
        task_context_digest=context['context_digest'], splat=manifest)
    admission = root / ('admission-' + report['digest'][7:23] + '.json')
    write_json(admission, report)
    return {'status': 'completed', 'generated_geometry_admitted': True, 'splat_root': str(destination),
            'splat_manifest': manifest, 'completion': completion, 'admission': report,
            'admission_path': str(admission), 'generated_roots': completion['generated_roots'],
            'analysis_paths': [str(admission), *[str(job_root / a['relative_path']) for a in assets]]}
