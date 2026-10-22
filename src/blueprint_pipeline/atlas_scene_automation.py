"""Reusable production automation for Atlas size estimates and edited backgrounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file


def _unposed_source_views(inputs):
    """Read reviewed source RGB without inventing cameras or purchasing depth."""
    import numpy as np
    from PIL import Image

    frames = []
    for row in inputs["frames"]:
        path = Path(row["image_path"])
        if _sha256_file(path) != row["image_digest"]:
            raise ValueError("atlas_automation_source_changed")
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        h, w = rgb.shape[:2]
        frames.append({"frame_id": row["frame_id"], "image_path": str(path),
                       "image_digest": row["image_digest"], "rgb": rgb,
                       "intrinsics": {"width": w, "height": h}, "camera_calibrated": False,
                       "pixel_alignment": {"source_grid": [w, h], "method": "same_source_pixels",
                                           "matrix": np.eye(3).tolist()}})
    return {"digest": canonical_digest({"unposed_source_inputs": inputs["digest"]}),
            "binding": inputs["binding"], "geometry_kind": "unposed_source_rgb"}, frames


def _edited_frames(background):
    from .atlas_depth_splat import posed_frame, read_exr_depth
    import cv2

    seen = set()
    frames = []
    for root, ids in zip(background["stage_roots"], background["batch_frame_ids"], strict=True):
        root = Path(root)
        result = json.loads((root / "result.json").read_text())
        if result.get("digest") != canonical_digest(result, digest_field="digest"):
            raise ValueError("atlas_edited_result_changed")
        for row in result["outputs"]:
            if _sha256_file(root / row["path"]) != row["sha256"]:
                raise ValueError("atlas_edited_output_changed")
        for i, (frame_id, response) in enumerate(
            zip(ids, result["operation"]["response"]["frames"], strict=True)
        ):
            if frame_id in seen:
                continue
            seen.add(frame_id)

            def path(role):
                row = next(r for r in result["outputs"] if r["role"] == f"frames-{i}-{role}")
                return root / row["path"]

            image = cv2.imread(str(path("imageAsset")))
            depth = read_exr_depth(path("depth-depthAsset"))
            confidence = cv2.imread(str(path("depth-confidenceAsset")), cv2.IMREAD_UNCHANGED)
            frames.append(
                posed_frame(
                    frame_id=frame_id,
                    rgb=image[..., ::-1],
                    depth=depth,
                    confidence=confidence,
                    camera=response["camera"],
                )
            )
    return frames


def _fallback_pose(background, root, source_digest, context, policy, pose_runner, max_polls):
    """Re-estimate edited RGB's geometry; never paste original depth under a fill."""
    from .website_worldlabs import prepare_atlas_pose_inputs
    from .website_scene_geometry import INPUT_SCHEMA
    from .website_atlas_pose import ATLAS_PRICE_REVIEW_ENV, request_options, load_pending_atlas_pose_inputs
    import os

    legacy = root / "fallback_pose/atlas_pose/submission.json"
    if legacy.exists():
        raise ValueError("atlas_fallback_legacy_pose_requires_reconciliation")
    # V1 put RGB in a sibling directory, outside the canonical producer's
    # capture boundary. Preserve that refused handoff; v2 owns its own inputs.
    pose_root = root / "fallback_pose-v2"
    input_root = pose_root / "inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, row in enumerate(background["frames"]):
        src = Path(row["image_path"])
        if _sha256_file(src) != row["image_digest"]:
            raise ValueError("atlas_fallback_rgb_changed")
        dst = input_root / f"{i:04d}.png"
        shutil.copyfile(src, dst)
        rows.append(
            {
                "frame_id": row["frame_id"],
                "image_path": str(dst),
                "image_digest": _sha256_file(dst),
                "source_image_path": str(dst),
                "source_image_digest": _sha256_file(dst),
                "generated_content": True,
            }
        )
    inputs = {
        "schema_version": INPUT_SCHEMA,
        "heldout_pixels_included": False,
        "frames": rows,
        "binding": {
            "source_video_digest": source_digest,
            "content_basis": "generated_background_after_independent_visual_review",
        },
        "generated_content": True,
        "source_sample_count": len(rows),
        "heldout_frame_count": 0,
    }
    inputs["digest"] = canonical_digest(inputs, digest_field="digest")
    prepare_atlas_pose_inputs(inputs=inputs, input_root=input_root, output_root=pose_root)
    load_pending_atlas_pose_inputs(pose_root)
    quote = {
        "schema_version": "website_atlas_pose_price_review.v1",
        "endpoint": "/api/v2/tasks:images2PosedRGBD",
        "maximum_cost_usd": round(len(rows) * policy["pose_input_cost_usd"], 6),
        "max_input_images": len(rows),
        "target_resolutions": [[720, 1280], [1280, 720]],
        "request_options": request_options(),
        **{k: policy[k] for k in ["rate_card_reference", "reviewed_by", "reviewed_on"]},
    }
    os.environ[ATLAS_PRICE_REVIEW_ENV] = json.dumps(quote)
    state = pose_runner(
        output_root=pose_root,
        capture_id=context["capture_id"],
        task_context=context,
        max_polls=max_polls,
    )
    if state["status"] != "completed":
        return None, state
    from .atlas_depth_splat import load_posed_rgbd

    result, frames = load_posed_rgbd(pose_root)
    return (result, frames, pose_root), state


def measurement_plan(review):
    """The reviewed automation plan with the scene facts size estimates are bound to."""
    from .atlas_automation_plan import AutomationPlan

    automation = AutomationPlan.model_validate(review["automation"]).model_dump(mode="json")
    return {
        **automation,
        "uncaptured_regions": review.get("uncaptured_regions", []),
        "anchor_frame_id": review.get("anchor_frame_id"),
        "reference_sources": review.get("scale_reference_sources", []),
    }


def run_automation(
    *,
    pose_root,
    inputs,
    review,
    output_root,
    task_context,
    configuration,
    task_runner,
    pose_runner,
    release_blockers,
    planner_runner=None,
    max_polls=2,
    execute=True,
    checkpoint=None,
):
    from .atlas_scene_measurements import load_views, estimate_sizes
    from .atlas_background_automation import run_background

    if review.get("automation") is None:
        return {"status": "legacy_plan_without_automation", "new_paid_calls": 0}
    automation = measurement_plan(review)
    output_root.mkdir(parents=True, exist_ok=True)
    pose, frames = load_views(pose_root, inputs) if pose_root is not None else _unposed_source_views(inputs)
    measurements = estimate_sizes(
        frames=frames,
        automation=automation,
        pose_digest=pose["digest"],
        output=output_root / "scene_measurements.json",
    ) if pose_root is not None else {
        "status": "unavailable", "reason": "pre_edit_observed_depth_not_requested",
        "metric_measurement_proven": False, "physical_evidence": False,
    }
    if not execute:
        return {
            "status": "measurements_complete",
            "measurements": measurements,
            "new_paid_calls": 0,
        }
    source = pose["binding"]["source_video_digest"]
    policy = configuration["policy"]
    requested = list(dict.fromkeys([
        key for group in review.get("coverage_requirements", [])
        for key in group["required_frame_ids"]
    ] + review.get("additional_input_frame_ids", [])))
    selected_ids = [row["frame_id"] for row in frames if row["frame_id"] in requested] if requested else None
    if requested and set(selected_ids) != set(requested):
        raise ValueError("atlas_background_reviewed_selection_missing")
    if pose_root is None and selected_ids is not None:
        frames = [row for row in frames if row["frame_id"] in selected_ids]
    try:
        background = run_background(
            frames=frames,
            objects=automation["objects"],
            automation=automation,
            inputs=inputs,
            pose=pose,
            root=output_root / "background",
            context=task_context,
            policy=policy,
            source_digest=source,
            task_runner=task_runner,
            release_blockers=release_blockers,
            max_polls=max_polls,
            configuration=configuration,
            checkpoint=checkpoint,
            selected_frame_ids=selected_ids,
            transients=review.get("transients", []),
        )
        if background["status"] == "processing":
            return background
        if background["status"] in {"passed", "fallback_prepared"}:
            from .atlas_depth_splat import write_depth_splat

            if background["status"] == "fallback_prepared":
                if checkpoint is not None:
                    checkpoint("automation_fallback_pose")
                derived, state = _fallback_pose(
                    background, output_root, source, task_context, policy, pose_runner, max_polls
                )
                if derived is None:
                    return {
                        "status": "processing",
                        "stage": "fallback_pose",
                        "operation_id": state.get("operation_id"),
                    }
                effective_pose, derived_frames, effective_root = derived
                if pose_root is None:
                    # Removed appliances cannot become scale references in the
                    # generated background. Report relative scene coverage only.
                    _, measured_frames = load_views(effective_root, json.loads(
                        (effective_root / "atlas_pose_inputs.json").read_text()))
                    measurements = estimate_sizes(frames=measured_frames,
                        automation={**automation, "scale_references": []}, pose_digest=effective_pose["digest"],
                        output=output_root / "scene_measurements.json")
                provenance = [canonical_digest({"frames": background["frames"]})]
                visual_digest = canonical_digest(background["review"])
            else:
                effective_pose, effective_root = pose, pose_root
                derived_frames = _edited_frames(background)
                provenance = background["result_digests"]
                visual_digest = background["visual_review_digest"]
            generated_roots = [Path(p) for p in background.get("stage_roots", [])]
            completion = None
            if background["status"] == "fallback_prepared" and "rgbd_view_cost_usd" in policy:
                from .atlas_scene_completion import complete_scene, COMPLETION_REVISION

                generated, completion = complete_scene(
                    pose_root=effective_root, pose=effective_pose, frames=derived_frames,
                    review=review, root=output_root / f"scene_completion-{COMPLETION_REVISION}", context=task_context,
                    configuration=configuration, task_runner=task_runner, release_blockers=release_blockers,
                    planner_runner=planner_runner, max_polls=max_polls, checkpoint=checkpoint)
                if completion["status"] == "processing":
                    return completion
                if generated is None:
                    raise ValueError(completion["reason"])
                derived_frames = [*derived_frames, *generated]
                provenance = [*provenance, completion["generation_result_digest"]]
                visual_digest = canonical_digest({"reviews": [visual_digest, completion["review"]]})
                generated_roots.append(Path(completion["generated_root"]))
                background["completion"] = completion
                background["reconstruction_frame_count"] = len(derived_frames)
                background["generated_completion_frame_count"] = len(generated)
            # Retain the earlier baseline rather than letting its existing
            # manifest silently bypass the newly generated geometry.
            splat_root = output_root / (f"background_completed_splat-{COMPLETION_REVISION}" if completion else "background_splat")
            manifest_path = splat_root / "atlas_depth_splat_manifest.json"
            splat = (
                json.loads(manifest_path.read_text())
                if manifest_path.is_file()
                else write_depth_splat(
                    frames=derived_frames,
                    result=effective_pose,
                    output_root=splat_root,
                    generated_result_digests=provenance,
                    visual_review_digest=visual_digest,
                )
            )
            if completion is not None and (
                    splat.get("generated_result_digests") != provenance
                    or {row["frame_id"] for row in splat.get("frames", [])}
                    != {row["frame_id"] for row in derived_frames}):
                raise ValueError("atlas_completion_splat_did_not_consume_generated_depth")
            mesh_root = None
            if "mesh_export_cost_usd" in policy:
                spec = json.loads((splat_root / "splats2mesh_task_spec.json").read_text())
                mesh_root = output_root / (f"background_completed_mesh-{COMPLETION_REVISION}" if completion else "background_mesh")
                quote = {
                    "schema_version": "website_atlas_task_price_review.v1",
                    "task": "splats2Mesh",
                    "request_digest": canonical_digest(spec["request"]),
                    "maximum_cost_usd": policy["mesh_export_cost_usd"],
                    **{k: policy[k] for k in ["rate_card_reference", "reviewed_by", "reviewed_on"]},
                }
                if checkpoint is not None:
                    checkpoint("automation_mesh")
                mesh = task_runner(
                    task="splats2Mesh",
                    request=spec["request"],
                    uploads=spec["uploads"],
                    output_root=mesh_root,
                    source_video_digest=source,
                    task_context=task_context,
                    price_review=quote,
                    max_polls=max_polls,
                )
                if mesh["status"] != "completed":
                    return {
                        "status": "processing",
                        "stage": "background_mesh",
                        "operation_id": mesh.get("operation_id"),
                    }
                from .atlas_mesh_registration import register_atlas_mesh

                if not (mesh_root / "registered/atlas_mesh_registration.json").is_file():
                    glb = next(o for o in mesh["outputs"] if o["role"] == "mesh")
                    register_atlas_mesh(
                        glb=mesh_root / glb["path"],
                        splat_root=splat_root,
                        output_root=mesh_root / "registered",
                    )
            from .atlas_result_bundle import export_pose_bundle

            write_json(output_root / "background_review.json", background)
            bundle = export_pose_bundle(
                pose_root=effective_root,
                output_root=output_root / "downloads",
                splat_root=splat_root,
                mesh_root=mesh_root,
                generated_roots=generated_roots,
                analysis_paths=[
                    output_root / "scene_measurements.json",
                    output_root / "background_review.json",
                ] + ([output_root / f"scene_completion-{COMPLETION_REVISION}/completion.json"] if completion else []),
            )
            background.update(
                status="completed",
                download=bundle,
                splat_manifest_digest=splat["digest"],
                mesh_available=mesh_root is not None,
            )
    except (ValueError, StopIteration) as exc:
        # Keep source reconstruction and usable size estimates even when consent,
        # budget, pixels, review, depth agreement or export blocks this candidate.
        background = {
            "status": "needs_input",
            "reason": str(exc),
            "retained_root": str(output_root / "background"),
        }
    value = {
        "schema_version": "atlas_scene_automation.v1",
        "status": background["status"],
        "measurements": measurements,
        "background": background,
        "source_pose_digest": pose["digest"] if pose_root is not None else None,
        "automation_plan_digest": canonical_digest(automation),
        "raw_capture_preserved": True,
        "metric_measurement_proven": False,
        "physical_evidence": False,
        "simulation_qualified": False,
        "claim_ceiling": "development_only",
    }
    value["digest"] = canonical_digest(value, digest_field="digest")
    write_json(output_root / "result.json", value)
    return value


def main():
    """Offline retained-run measurements; does not load secrets or call providers."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["pose-root", "inputs", "review", "output-root"]:
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()

    def forbidden(**_):
        raise AssertionError("offline_measurement_provider_call")

    result = run_automation(
        pose_root=args.pose_root,
        inputs=json.loads(args.inputs.read_text()),
        review=json.loads(args.review.read_text()),
        output_root=args.output_root,
        task_context={},
        configuration={},
        task_runner=forbidden,
        pose_runner=forbidden,
        release_blockers=lambda: ("", []),
        execute=False,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
