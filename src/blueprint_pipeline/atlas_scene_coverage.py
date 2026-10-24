"""ADP-040/day-28: close task-relevant gaps after exporting the reconstruction.

Reuse the existing Atlas camera planner, RGBD adapter and independent review.
Exported geometry, rather than the mere existence of a generated image, decides
what the next pass sees. All generated surfaces remain development candidates.
"""
from __future__ import annotations

import json

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest


def reviewed_reference_frames(reviewed_views, region_plan):
    """Keep every approved reference; remove explicitly rejected pixels/depth."""
    from .atlas_scene_task_regions import task_region_masks

    masks = task_region_masks(region_plan, reviewed_views)
    references = []
    for row in reviewed_views:
        frame = row['frame']
        mask = masks[frame['frame_id']]
        if not np.any(mask & (frame['confidence'] > 0) & np.isfinite(frame['depth']) & (frame['depth'] > 0)):
            continue
        rgb = frame['rgb'].copy()
        rgb[~mask] = 127
        references.append({**frame, 'rgb': rgb, 'confidence': frame['confidence'] * mask})
    return references


def refine_scene_coverage(*, pose_root, pose, frames, geometry_frames, reviewed_views,
                          region_plan, splat_root, output_root, review, context,
                          configuration, task_runner, planner_runner, release_blockers,
                          max_polls, pose_runner, checkpoint=None, source_survey_path=None,
                          source_recovery_runner=None, recorded_frames=(), recorded_source_reports=(),
                          context_pose_root=None, context_pose_digest=None):
    """Survey -> choose necessary gaps -> generate -> fuse -> re-survey.

    Repeated calls adopt identical completed stages. The shared provider allowance
    authorizes every paid request. A pass with no exportable contribution stops
    with an explicit gap, rather than buying the same unsuccessful repair again.
    """
    from .atlas_depth_splat import surface_samples, write_depth_splat
    from .atlas_scene_completion import complete_scene
    from .atlas_recorded_source_recovery import merge_source_frames

    if pose_runner is None:
        raise ValueError('atlas_generated_registration_canonical_pose_runner_required')
    frames = list(frames)
    references = reviewed_reference_frames(reviewed_views, region_plan)
    geometry = list(geometry_frames)
    manifest = json.loads((splat_root / 'atlas_depth_splat_manifest.json').read_text())
    if (manifest.get('digest') != canonical_digest(manifest, digest_field='digest')
            or manifest.get('pose_result_digest') != pose['digest']):
        raise ValueError('atlas_coverage_export_changed')
    reports, paths, generated_roots, depth_outcomes = [], [], [], []
    provenance = list(manifest.get('generated_result_digests', []))
    visual = manifest.get('visual_review_digest')
    recorded, source_reports = list(recorded_frames), list(recorded_source_reports)
    references.extend(recorded)
    paths.extend(p for report in source_reports for p in report.get('preservation_paths', []))
    while references:
        binding = {'revision': 'v3-complete-surface-review', 'export_digest': manifest['digest'],
                   'task_region_plan_digest': region_plan['digest'],
                   'task_context_digest': context['context_digest']}
        identity = canonical_digest(binding)
        root = output_root / 'completion_coverage' / identity[7:23]
        if checkpoint:
            checkpoint('automation_export_coverage', export_digest=manifest['digest'],
                       gaussian_count=manifest['gaussian_count'], completed_repair_passes=len(reports))
        sources = merge_source_frames(frames, recorded)
        source_ids = {f['frame_id'] for f in sources}
        generated, report = complete_scene(
            pose_root=pose_root, pose=pose, frames=sources,
            existing_geometry_frames=[f for f in geometry if f['frame_id'] not in source_ids],
            expected_frames=references,
            splat_root=splat_root, reference_frames=merge_source_frames(sources, references),
            coverage_feedback={**binding,
                'required_surface_hypotheses': region_plan['proposal']['remaining_required_surfaces'],
                'task_extent_summary': region_plan['proposal']['task_extent_summary'],
                'previous_repair_outcomes': depth_outcomes,
                **({'recorded_source_outcomes': [{'recovered_frame_ids': r['recovered_frame_ids'],
                    'rejected_views': r.get('rejected_views', [])} for r in source_reports]}
                   if source_reports else {}),
                'small_distortions_acceptable': True, 'scene_complete_is_not_implied': True},
            review=review, root=root, context=context, configuration=configuration,
            task_runner=task_runner, planner_runner=planner_runner, release_blockers=release_blockers,
            max_polls=max_polls, checkpoint=checkpoint, source_survey_path=source_survey_path,
            **({'context_pose_root': context_pose_root, 'context_pose_digest': context_pose_digest}
               if context_pose_root else {}))
        if report.get('status') == 'processing':
            return None, report
        if report.get('status') == 'source_recovery_required' and source_recovery_runner is not None:
            recovered, source_report = source_recovery_runner(report, [*frames, *recorded])
            if recovered is None:
                return None, source_report
            known = {f['frame_id'] for f in [*frames, *recorded]}
            added = set(source_report.get('recovered_frame_ids', []))
            refreshed = set(source_report.get('refreshed_frame_ids', []))
            if (not recovered or len({f['frame_id'] for f in recovered}) != len(recovered)
                    or added & known or not refreshed <= known or added & refreshed
                    or source_report.get('status') != 'completed'
                    or source_report.get('digest') != canonical_digest(source_report, digest_field='digest')
                    or source_report.get('target_pose_digest') != pose['digest']
                    or added | refreshed != {f['frame_id'] for f in recovered}):
                raise ValueError('atlas_coverage_recorded_source_recovery_invalid')
            frames = merge_source_frames(frames, [f for f in recovered if f['frame_id'] in {
                old['frame_id'] for old in frames}])
            recorded = merge_source_frames(recorded, [f for f in recovered if f['frame_id'] not in {
                old['frame_id'] for old in frames}])
            context_pose_root = source_report.get('context_pose_root', context_pose_root)
            context_pose_digest = source_report.get('pose_result_digest', context_pose_digest)
            source_reports.append(source_report)
            references = merge_source_frames(references, recovered)
            geometry = merge_source_frames(geometry, recovered)
            provenance.append(source_report['digest'])
            visual = canonical_digest({'previous': visual, 'recorded_source_review': source_report['digest']})
            paths.extend(source_report.get('preservation_paths', []))
            splat_root = output_root / ('background_source_recovered_splat-' + source_report['digest'][7:23])
            saved = splat_root / 'atlas_depth_splat_manifest.json'
            manifest = (json.loads(saved.read_text()) if saved.is_file() else write_depth_splat(
                frames=geometry, result=pose, output_root=splat_root,
                generated_result_digests=provenance, visual_review_digest=visual))
            if (manifest.get('digest') != canonical_digest(manifest, digest_field='digest')
                    or manifest.get('pose_result_digest') != pose['digest']
                    or manifest.get('generated_result_digests') != provenance
                    or {r['frame_id'] for r in manifest['frames']} != {f['frame_id'] for f in geometry}):
                raise ValueError('atlas_coverage_export_did_not_consume_recorded_sources')
            continue
        if generated is None:
            return None, report
        from .atlas_source_gap_evidence import repair_depth_outcomes
        depth_outcomes.extend(repair_depth_outcomes(generated, report))
        registered_references = []
        if generated and report.get('generated_roots'):
            from .atlas_scene_recovery import register_generated_completion, recover_reviewed_geometry
            from .atlas_scene_task_regions import plan_task_regions

            registered, registration = register_generated_completion(frames=sources, completion=report,
                root=root / 'joint-registration', pose=pose, context=context, configuration=configuration,
                pose_runner=pose_runner, max_polls=max_polls, checkpoint=checkpoint)
            if registered is None:
                return None, registration
            regions = plan_task_regions(frames=sources, reviewed_views=registered, pose=pose,
                context=context, configuration=configuration, root=root / 'registered-task-regions',
                planner_runner=planner_runner, checkpoint=checkpoint)
            generated, recovery = recover_reviewed_geometry(frames=sources, geometry_frames=geometry,
                reviewed_views=registered, region_plan=regions, exported_splat_root=splat_root,
                root=root / 'registered-export-recovery', checkpoint=checkpoint)
            registered_references = reviewed_reference_frames(registered, regions)
            report = {**report, 'native_completion_report_digest': report['digest'],
                      'generated_view_registration': registration, 'registered_geometry_recovery': recovery,
                      'preservation_paths': [*report.get('preservation_paths', []),
                          *registration['preservation_paths'], *recovery['preservation_paths']],
                      'generated_frame_ids': [f['frame_id'] for f in generated]}
            report['digest'] = canonical_digest(report, digest_field='digest')
            depth_outcomes.append({'registration_digest': registration['digest'],
                'registered_geometry_recovery_digest': recovery['digest'],
                'added_geometry_pixels': recovery['added_geometry_pixels'],
                'exportable_samples': recovery['exportable_samples'],
                'depth_basis': registration['depth_basis'], 'native_depth_is_unmodified': True})
        reports.append(report)
        paths.extend(report.get('preservation_paths', []))
        paths.append(str(root / 'completion.json'))
        # Keep the agent's exact input/decision and coverage overlays alongside
        # the generated RGBD, so future sessions can resume this repair.
        paths.extend(str(p) for name in ('plan-inputs.json', 'survey.json', 'context-fov-audit.json',
                                        'gap-target-survey.json') if (p := root / name).is_file())
        paths.extend(str(p) for p in root.glob('plan-inputs-source-*.json'))
        paths.extend(str(p) for p in (root / 'planner').glob('*.json'))
        paths.extend(str(p) for p in (root / 'reviewed-references').glob('*.png'))
        for directory in ('gaps', 'gap-targets', 'gap-target-context-fallback'):
            survey_path = root / directory / 'survey.json'
            if survey_path.is_file():
                paths.append(str(survey_path))
                surveyed = json.loads(survey_path.read_text())
                paths.extend(row['path'] for row in surveyed['previews'])
                paths.extend(row['label_path'] for row in surveyed['views'])
        generated_roots.extend(report.get('generated_roots', []))
        if not generated:
            stop = report['review']['reason']
            break
        contributions = [frame for frame in generated if len(surface_samples(frame)['points'])]
        if not contributions:
            stop = 'generated_views_added_no_exportable_geometry'
            break
        geometry.extend(contributions)
        references.extend(registered_references or contributions)
        provenance.append(report['generation_result_digest'])
        if report.get('generated_view_registration'):
            provenance.extend([report['generated_view_registration']['digest'],
                               report['registered_geometry_recovery']['digest']])
        visual = canonical_digest({'previous': visual, 'review': report['review']})
        splat_root = output_root / ('background_refined_splat-' + identity[7:23])
        saved = splat_root / 'atlas_depth_splat_manifest.json'
        if checkpoint:
            checkpoint('automation_export_coverage_fuse', reuse_cached_export=saved.is_file(),
                       retained_frame_count=len(geometry),
                       added_generated_frame_ids=[frame['frame_id'] for frame in contributions])
        manifest = (json.loads(saved.read_text()) if saved.is_file() else write_depth_splat(
            frames=geometry, result=pose, output_root=splat_root,
            generated_result_digests=provenance, visual_review_digest=visual))
        if (manifest.get('digest') != canonical_digest(manifest, digest_field='digest')
                or manifest.get('generated_result_digests') != provenance
                or {r['frame_id'] for r in manifest['frames']} != {f['frame_id'] for f in geometry}):
            raise ValueError('atlas_coverage_export_did_not_consume_repair')
    else:
        stop = 'no_reviewed_depth_references'
    result = {'schema_version': 'atlas_export_coverage_refinement.v1', 'status': 'completed',
              'binding': binding if references else {'task_region_plan_digest': region_plan['digest']},
              'splat_root': str(splat_root), 'splat_manifest_digest': manifest['digest'],
              'completion_reports': reports, 'generated_roots': generated_roots,
              **({'recorded_source_recoveries': source_reports} if source_reports else {}),
              'preservation_paths': list(dict.fromkeys(paths)), 'stop_reason': stop,
              'task_coverage_sufficient': stop in {'agent_found_no_material_task_gap',
                                                   'source_review_found_physical_openings'},
              'whole_scene_complete': False, 'metric_measurement_proven': False,
              'simulation_qualified': False, 'claim_ceiling': 'development_only'}
    result['digest'] = canonical_digest(result, digest_field='digest')
    write_json(output_root / 'completion_coverage' / 'latest.json', result)
    return (geometry, manifest, splat_root), result
