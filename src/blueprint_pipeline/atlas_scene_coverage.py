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
                          max_polls, checkpoint=None, source_survey_path=None):
    """Survey -> choose necessary gaps -> generate -> fuse -> re-survey.

    Repeated calls adopt identical completed stages. The shared provider allowance
    authorizes every paid request. A pass with no exportable contribution stops
    with an explicit gap, rather than buying the same unsuccessful repair again.
    """
    from .atlas_depth_splat import surface_samples, write_depth_splat
    from .atlas_scene_completion import complete_scene

    references = reviewed_reference_frames(reviewed_views, region_plan)
    geometry = list(geometry_frames)
    manifest = json.loads((splat_root / 'atlas_depth_splat_manifest.json').read_text())
    if (manifest.get('digest') != canonical_digest(manifest, digest_field='digest')
            or manifest.get('pose_result_digest') != pose['digest']):
        raise ValueError('atlas_coverage_export_changed')
    reports, paths, generated_roots, depth_outcomes = [], [], [], []
    provenance = list(manifest.get('generated_result_digests', []))
    visual = manifest.get('visual_review_digest')
    while references:
        binding = {'revision': 'v2-paired-depth', 'export_digest': manifest['digest'],
                   'task_region_plan_digest': region_plan['digest'],
                   'task_context_digest': context['context_digest']}
        identity = canonical_digest(binding)
        root = output_root / 'completion_coverage' / identity[7:23]
        if checkpoint:
            checkpoint('automation_export_coverage', export_digest=manifest['digest'],
                       gaussian_count=manifest['gaussian_count'], completed_repair_passes=len(reports))
        generated, report = complete_scene(
            pose_root=pose_root, pose=pose, frames=frames,
            existing_geometry_frames=geometry[len(frames):], expected_frames=references,
            splat_root=splat_root, reference_frames=[*frames, *references],
            coverage_feedback={**binding,
                'required_surface_hypotheses': region_plan['proposal']['remaining_required_surfaces'],
                'task_extent_summary': region_plan['proposal']['task_extent_summary'],
                'previous_repair_outcomes': depth_outcomes,
                'small_distortions_acceptable': True, 'scene_complete_is_not_implied': True},
            review=review, root=root, context=context, configuration=configuration,
            task_runner=task_runner, planner_runner=planner_runner, release_blockers=release_blockers,
            max_polls=max_polls, checkpoint=checkpoint, source_survey_path=source_survey_path)
        if report.get('status') == 'processing':
            return None, report
        if generated is None:
            return None, report
        reports.append(report)
        from .atlas_source_gap_evidence import repair_depth_outcomes
        depth_outcomes.extend(repair_depth_outcomes(generated, report))
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
        references.extend(contributions)
        provenance.append(report['generation_result_digest'])
        visual = canonical_digest({'previous': visual, 'review': report['review']})
        splat_root = output_root / ('background_refined_splat-' + identity[7:23])
        saved = splat_root / 'atlas_depth_splat_manifest.json'
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
              'preservation_paths': list(dict.fromkeys(paths)), 'stop_reason': stop,
              'task_coverage_sufficient': stop in {'agent_found_no_material_task_gap', 'no_eligible_geometry_gaps',
                                                   'source_review_found_physical_openings'},
              'whole_scene_complete': False, 'metric_measurement_proven': False,
              'simulation_qualified': False, 'claim_ceiling': 'development_only'}
    result['digest'] = canonical_digest(result, digest_field='digest')
    write_json(output_root / 'completion_coverage' / 'latest.json', result)
    return (geometry, manifest, splat_root), result
