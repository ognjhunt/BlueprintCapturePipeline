"""Recover reviewed Atlas RGBD omitted by an earlier, narrower gap selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file

RECOVERY_REVISION = "v1"


def reviewed_completion_views(reports, *, selected_only=False):
    """Read completed, visually reviewed native outputs without repurchasing."""
    from .atlas_scene_completion import load_generated_views

    views, seen, seen_cameras = [], set(), set()
    for report in reports:
        if (report.get("status") != "passed"
                or report.get("digest") != canonical_digest(report, digest_field="digest")):
            raise ValueError("atlas_recovery_review_not_bound")
        if not report.get("generated_roots") and report.get("review", {}).get("status") == "not_required":
            continue
        if report.get("review", {}).get("status") != "passed":
            raise ValueError("atlas_recovery_review_not_bound")
        admitted = set(report.get("generation", {}).get("native_result_digests", []))
        for batch_index, root in enumerate(report.get("generated_roots", [])):
            root = Path(root)
            result = json.loads((root / "result.json").read_text())
            if result["digest"] not in admitted:
                raise ValueError("atlas_recovery_native_result_not_reviewed")
            ids = [f"recovered-{result['digest'][7:23]}-{i}" for i in
                   range(len(result["operation"]["response"]["frames"]))]
            native, frames, _ = load_generated_views(root, ids)
            selected = None
            if selected_only:
                plan_path = root.parent.parent / 'rgbd-completion-plan.json'
                plan = json.loads(plan_path.read_text())
                if (plan_path.is_symlink() or plan.get('digest') != canonical_digest(plan, digest_field='digest')
                        or plan['digest'] != report['generation'].get('mask_plan_digest')
                        or set(plan['selected_camera_ids']) != set(report['selected_camera_ids'])):
                    raise ValueError('atlas_registration_selected_views_not_bound')
                batch = plan['batches'][batch_index]['frame_ids']
                if len(batch) != len(frames):
                    raise ValueError('atlas_registration_selected_views_not_bound')
                selected = {i for i, key in enumerate(batch) if key in plan['selected_camera_ids']}
            for index, frame in enumerate(frames):
                key = (native["digest"], index)
                if (key in seen or (selected is not None and index not in selected)
                        or (selected is not None and batch[index] in seen_cameras)):
                    continue
                seen.add(key)
                if selected is not None:
                    seen_cameras.add(batch[index])
                views.append({"frame": frame, "native_result_digest": native["digest"],
                              "native_frame_index": index, "review_digest": report["digest"],
                              **({'requested_camera_id': batch[index]} if selected is not None else {})})
    return views


def register_generated_completion(*, frames, completion, root, pose, context,
                                  configuration, pose_runner, max_polls, checkpoint=None):
    """Estimate a generated image's actual pose with all cleaned source views.

    Requested camera metadata is not independent evidence that generated RGB
    follows those rays. Reuse the pose adapter and shared-source registration;
    keep original geometry fixed and re-mask the registered candidate afterward.
    """
    from .atlas_scene_generate import prepare_rgbd_context
    from .atlas_scene_automation import _fallback_pose
    from .atlas_shared_view_alignment import shared_view_alignment
    from .atlas_recorded_source_recovery import aligned_recorded_frames
    from .atlas_depth_splat import posed_frame, read_exr_depth
    from .atlas_scene_preservation import _save_image

    if pose_runner is None:
        raise ValueError('atlas_generated_registration_canonical_pose_runner_required')
    views = reviewed_completion_views([completion], selected_only=True)
    if not views:
        raise ValueError('atlas_generated_registration_selected_views_required')
    binding = {'revision': 'v1', 'source_pose_digest': pose['digest'],
               'source_frames': [_frame_binding(f) for f in frames],
               'reviewed_completion_digest': completion['digest']}
    identity = canonical_digest(binding)
    root = root / identity[7:23]
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'binding.json', binding)
    if checkpoint:
        checkpoint('automation_completion_register_generated')
    _, uploads, photographs = prepare_rgbd_context(frames=frames, root=root / 'source-context')
    inputs, targets, seen = [], [], set()
    for index, photo in enumerate(photographs):
        inputs.append({'frame_id': photo['frame_id'], 'image_path': photo['image_path'],
                       'image_digest': photo['image_digest']})
        # All tiles inform the joint pose, but one physical view cannot count
        # as several independent source-registration anchors.
        if photo['source_frame_id'] in seen:
            continue
        seen.add(photo['source_frame_id'])
        depth = read_exr_depth(Path(uploads[f'context-{index}-depth']['path']))
        targets.append(posed_frame(frame_id=photo['frame_id'],
            rgb=np.asarray(Image.open(photo['image_path']).convert('RGB')), depth=depth,
            confidence=(np.isfinite(depth) & (depth > 0)).astype(np.float32), camera=photo['camera']))
    for index, row in enumerate(views):
        path = root / f'generated-{index:04d}.png'
        _save_image(path, row['frame']['rgb'])
        inputs.append({'frame_id': row['frame']['frame_id'], 'image_path': str(path),
                       'image_digest': _sha256_file(path)})
    quote = len(inputs) * configuration['policy']['pose_input_cost_usd']
    pose_root = root / 'fallback_pose-v3'
    if (not (pose_root / 'atlas_pose/submission.json').exists()
            and not (pose_root / 'atlas_posed_rgbd_result.json').exists()
            and quote + configuration['policy'].get('mesh_export_cost_usd', 0) > configuration['remaining_cost_usd']):
        raise ValueError('atlas_generated_registration_remaining_allowance_insufficient')
    derived, state = _fallback_pose({'frames': inputs}, root, pose['binding']['source_video_digest'],
        context, configuration['policy'], pose_runner, max_polls)
    if derived is None:
        return None, {**state, 'stage': 'generated_view_registration'}
    joint, posed, pose_root = derived
    fit = shared_view_alignment(posed, targets, estimated_depth_tolerance=True)
    if fit is None:
        return None, {'status': 'needs_input', 'reason': 'atlas_generated_view_registration_unresolved',
                      'joint_pose_digest': joint['digest'], 'claim_ceiling': 'development_only'}
    by_id = {f['frame_id']: f for f in aligned_recorded_frames(posed, fit)}
    registered = [{**row, 'frame': by_id[row['frame']['frame_id']]} for row in views]
    paths = [root / 'binding.json', pose_root / 'atlas_posed_rgbd_result.json',
             *[Path(row['path']) for row in uploads.values()],
             *[Path(row['image_path']) for row in inputs],
             *[pose_root / asset['path'] for row in joint['operation_receipt']['input_usage']['frames']
               for asset in row['outputs']]]
    report = {'schema_version': 'atlas_generated_view_registration.v1', 'status': 'completed',
              'binding_digest': identity, 'joint_pose_digest': joint['digest'],
              'source_frame_ids': sorted(seen), 'conditioned_photograph_count': len(photographs),
              'registered_frame_bindings': [_frame_binding(r['frame']) for r in registered],
              'fit': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fit.items()},
              'source_geometry_modified': False, 'native_outputs_modified': False,
              'depth_basis': 'joint_pose_depth_registered_to_retained_source_world',
              'preservation_paths': [str(p) for p in paths] + [str(root / 'registration.json')],
              'claim_ceiling': 'development_only', 'metric_measurement_proven': False}
    report['digest'] = canonical_digest(report, digest_field='digest')
    write_json(root / 'registration.json', report)
    return [{**row, 'registration_digest': report['digest']} for row in registered], report


def _frame_binding(frame):
    return {"frame_id": frame["frame_id"], "intrinsics": frame["intrinsics"],
            "extrinsics": frame["extrinsics"], **{key: hashlib.sha256(
                np.ascontiguousarray(frame[key]).tobytes()).hexdigest()
                for key in ("rgb", "depth", "confidence")}}


def recover_reviewed_geometry(*, frames, geometry_frames, reviewed_views, root, checkpoint=None,
                             region_plan=None, exported_splat_root=None):
    """Fill remaining bounded gaps from already reviewed full native views.

    Earlier masks selected where to request generation. They must not permanently
    hide other recoverable holes in those same reviewed outputs. Re-survey each
    camera against all retained geometry and admit only named missing regions.
    Existing RGB, geometry, observed free space, and scene bounds still win.
    """
    from .atlas_scene_chisel import project_source_depth
    from .atlas_scene_gaps import gap_labels, select_gap_mask
    from .atlas_scene_preservation import (
        prepare_completion_views, preserve_generated_view, observed_scene_bounds, _save_image,
    )
    from .atlas_depth_splat import surface_samples

    exported = None
    if exported_splat_root is not None:
        from .atlas_scene_gaps import load_exported_splat
        if region_plan is None:
            raise ValueError('atlas_export_geometry_recovery_region_plan_required')
        exported_manifest, exported = load_exported_splat(exported_splat_root)

    task_masks = None
    if region_plan is not None:
        from .atlas_scene_task_regions import task_region_masks
        task_masks = task_region_masks(region_plan, reviewed_views)
    if any(not view.get("review_digest") for view in reviewed_views):
        raise ValueError("atlas_recovery_review_not_bound")
    binding = {"revision": RECOVERY_REVISION,
               "sources": [_frame_binding(f) for f in frames],
               "geometry": [_frame_binding(f) for f in geometry_frames],
               "native": [{**{k: v for k, v in row.items() if k != "frame"},
                           "frame": _frame_binding(row["frame"])} for row in reviewed_views]}
    if region_plan is not None:
        binding["task_region_plan_digest"] = region_plan["digest"]
    if exported is not None:
        binding.update(revision='v2-export-depth', exported_splat_digest=exported_manifest['digest'])
    digest = canonical_digest(binding)
    root = root / digest[7:23]
    root.mkdir(parents=True, exist_ok=True)
    saved = root / "recovery.json"
    by_id = {row["frame"]["frame_id"]: row for row in reviewed_views}
    if len(by_id) != len(reviewed_views):
        raise ValueError("atlas_recovery_duplicate_frame")
    if saved.is_file():
        report = json.loads(saved.read_text())
        if (report.get("digest") != canonical_digest(report, digest_field="digest")
                or report.get("binding_digest") != digest):
            raise ValueError("atlas_recovery_inputs_changed")
        for row in report["assets"]:
            if Path(row["path"]).is_symlink() or _sha256_file(Path(row["path"])) != row["sha256"]:
                raise ValueError("atlas_recovery_output_changed")
        recovered = []
        for row in report["views"]:
            if not row.get("exportable_samples"):
                continue
            native = by_id[row["frame_id"]]["frame"]
            recovered.append({**native,
                "rgb": np.asarray(Image.open(row["image_path"]).convert("RGB")),
                "confidence": native["confidence"] *
                    (np.asarray(Image.open(row["mask_path"]).convert("L")) > 0)})
        return recovered, report

    extent = observed_scene_bounds(frames)
    candidates = [{"camera_id": row["frame"]["frame_id"],
                   "source_frame_id": min(frames, key=lambda f: np.linalg.norm(
                       f["position"] - row["frame"]["position"]))["frame_id"],
                   "camera": {"intrinsics": row["frame"]["intrinsics"],
                              "extrinsics": row["frame"]["extrinsics"]}}
                  for row in reviewed_views]
    prepared = prepare_completion_views(frames=frames, selected=candidates, root=root / "context")
    recovered, rows = [], []
    paths = [Path(region_plan["plan_path"])] if region_plan is not None else []
    geometry = list(geometry_frames)
    for index, (view, source) in enumerate(zip(reviewed_views, prepared, strict=True)):
        if checkpoint:
            checkpoint("automation_completion_recover_geometry")
        frame = view["frame"]
        stage = root / f"{index:04d}"
        stage.mkdir(exist_ok=True)
        if exported is not None:
            from .atlas_scene_gaps import project_exported_splat
            covered = project_exported_splat(exported, source['camera'])
        else:
            covered = project_source_depth(geometry, source["camera"])
        expected = (np.where(np.isfinite(frame['depth']) & (frame['confidence'] > 0), frame['depth'], 0)
                    if exported is not None else None)
        known = np.asarray(Image.open(source["keep_mask_path"]).convert("L")) == 255
        if task_masks is None:
            labels, regions = gap_labels(covered, known, scene_outline=True, expected_depth=expected)
        else:
            # A reviewed task surface can be outside the old 2D silhouette.
            # Semantic selection defines the domain; existing depth/free-space
            # and world bounds still control which 3D points may be added.
            labels, regions = gap_labels(covered, task_masks[frame["frame_id"]],
                                         candidate_mask=task_masks[frame["frame_id"]], expected_depth=expected)
        # This is a bounded per-region continuation, not permission to fill
        # the entire new camera or extend the capture's world-space bounds.
        # Context/fill ratio is a pre-generation heuristic, not grounds to
        # discard an already reviewed, task-relevant large surface.
        eligible = (regions if task_masks is not None
                    else [r for r in regions if r["pixel_count"] <= int(known.sum())])
        row = {"frame_id": frame["frame_id"], "candidate_gap_pixels": int(np.count_nonzero(labels)),
               "eligible_gap_pixels": sum(r["pixel_count"] for r in eligible),
               "native_result_digest": view["native_result_digest"],
               "native_frame_index": view["native_frame_index"], "review_digest": view["review_digest"],
               "exportable_samples": 0, "added_geometry_pixels": 0}
        if eligible:
            label_path, depth_path = stage / "labels.png", stage / "known-depth.png"
            _save_image(label_path, labels)
            known_depth = covered > 0
            if exported is not None:
                # A farther visible plane is not the missing foreground face.
                # Keep its source colour but let the explicitly labeled depth
                # gap contribute, just as a genuinely empty export pixel can.
                known_depth &= labels == 0
            _save_image(depth_path, known_depth.astype(np.uint8) * 255)
            regions = [{**r, "gap_id": frame["frame_id"] + "-gap-" + str(r["label"])} for r in eligible]
            gaps = {"label_path": str(label_path), "label_sha256": _sha256_file(label_path), "gaps": regions}
            source.update(select_gap_mask(gaps, [r["gap_id"] for r in regions], stage / "selected.png"),
                          source_depth_context_path=str(depth_path),
                          source_depth_context_digest=_sha256_file(depth_path))
            fixed, image, receipt, assets = preserve_generated_view(
                generated=frame, prepared=source, references=frames if exported is not None else geometry,
                output_root=stage / "preserved",
                native_result_digest=view["native_result_digest"], native_frame_index=view["native_frame_index"],
                extent_bounds=extent, repair_exported_gap=exported is not None)
            samples = len(surface_samples(fixed)["points"])
            mask_path = next(a["path"] for a in receipt["assets"] if a["role"] == "generated_geometry_mask")
            row.update(exportable_samples=samples, added_geometry_pixels=receipt["generated_geometry_pixel_count"],
                       image_path=image["image_path"], mask_path=mask_path,
                       requested_gap_ids=source["completion_gap_ids"])
            paths.extend([label_path, depth_path, Path(source["completion_gap_mask_path"]), *assets])
            if samples:
                recovered.append(fixed)
                geometry.append(fixed)
        rows.append(row)
        paths.extend([Path(source["image_path"]), Path(source["keep_mask_path"])])
    report = {"schema_version": "atlas_geometry_recovery.v1", "binding_digest": digest,
              "revision": binding['revision'], "views": rows,
              "new_provider_calls": 0, "source_geometry_modified": False, "known_pixels_changed": 0,
              "added_geometry_pixels": sum(r["added_geometry_pixels"] for r in rows),
              "exportable_samples": sum(r["exportable_samples"] for r in rows),
              "generation_extent": extent, "whole_scene_complete": False,
              "claim_ceiling": "development_only",
              "assets": [{"path": str(p), "sha256": _sha256_file(p)} for p in paths],
              "preservation_paths": [str(p) for p in paths] + [str(saved)]}
    if exported is not None:
        report['exported_splat_digest'] = exported_manifest['digest']
    if region_plan is not None:
        report["task_region_plan"] = region_plan
    report["digest"] = canonical_digest(report, digest_field="digest")
    write_json(saved, report)
    return recovered, report
