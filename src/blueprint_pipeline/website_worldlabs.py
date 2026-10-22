"""Atlas pose inputs and generation requests for website captures.

Website scenes no longer buy Marble worlds (owner decision 2026-10-22); the
capture's Atlas job is the only reconstruction lane.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from copy import deepcopy

from .common import write_json
from .decision_evidence_contracts import canonical_digest


def prepare_atlas_pose_inputs(*, inputs: Mapping[str, Any], input_root: Path,
                              output_root: Path) -> dict[str, Any]:
    """CPU-only handoff for the documented unposed-image operation. Never submits."""
    from .website_scene_geometry import _bound_frames, INPUT_SCHEMA

    if inputs.get("schema_version") != INPUT_SCHEMA or inputs.get("heldout_pixels_included") is not False:
        raise ValueError("website_atlas_candidate_inputs_required")
    frames = _bound_frames(inputs, input_root, geometry=False)
    result = {"schema_version": "website_atlas_pose_inputs.v1", "status": "pending",
              "backend": "atlas", "endpoint": "/api/v2/tasks:images2PosedRGBD",
              "binding": {"source_video_digest": inputs["binding"]["source_video_digest"],
                          "geometry_input_digest": inputs["digest"]},
              "frames": [{"frame_id": row["frame_id"], "image_path": row["image_path"],
                          "image_digest": row["image_digest"]} for row in frames],
              "source_sample_count": inputs.get("source_sample_count"),
              "heldout_frame_count": inputs.get("heldout_frame_count"),
              "planned_provider_input_count": len(frames), "actual_provider_input_count": 0,
              "heldout_pixels_included": False, "claim_ceiling": "development_only",
              "metric_measurement_proven": False, "physical_evidence": False,
              "provider_mutation_performed": False,
              "blockers": ["website_atlas_admitted_pose_operation_required"]}
    if inputs.get("generated_content") is True:
        result["generated_input_content"] = True
        result["input_content_basis"] = inputs["binding"].get("content_basis")
    result["digest"] = canonical_digest(result, digest_field="digest")
    path = output_root / "atlas_pose_inputs.json"
    if path.is_file():
        if json.loads(path.read_text()) != result:
            raise ValueError("website_atlas_pose_input_identity_conflict")
    else:
        write_json(path, result)
    pose_path = output_root / "atlas_posed_rgbd_result.json"
    if pose_path.is_file():
        # A producer must retain the real admitted operation with its exact
        # source/input binding. A bare operation from another capture is not
        # sufficient, and this reader grants no upload/spending authority.
        pose = json.loads(pose_path.read_text())
        if (pose.get("schema_version") != "website_atlas_pose_result.v1"
                or pose.get("digest") != canonical_digest(pose, digest_field="digest")
                or pose.get("binding") != result["binding"]
                or pose.get("endpoint") != result["endpoint"]):
            raise ValueError("website_atlas_pose_result_binding_mismatch")
        operation = pose.get("operation")
        if not isinstance(operation, Mapping):
            raise ValueError("website_atlas_pose_operation_not_succeeded")
        if pose.get("operation_receipt") is not None:
            from .website_atlas_pose import atlas_pose_result_completed
            if not atlas_pose_result_completed(pose=pose, output_root=output_root):
                # The producer has not recorded completion: an interrupted
                # pass, not a result. Report what it has done so far.
                return _with_atlas_producer_status(result, output_root)
        posed_context = _atlas_posed_context(operation=operation, expected_frame_count=len(frames))
        # A returned bundle proves output count, not provider input usage.
        # Only the admitted producer's receipt records usage, and only while
        # it still binds these frames and its retained output bytes.
        actual_inputs = None
        if pose.get("operation_receipt") is not None:
            from .website_atlas_pose import verify_atlas_pose_receipt
            actual_inputs = verify_atlas_pose_receipt(pose=pose, frames=frames, output_root=output_root)
        context = {**result, "pose_operation_id": operation["id"],
                   "pose_result_digest": pose["digest"], "posed_rgbd_available": True,
                   "actual_provider_input_count": actual_inputs,
                   # A verified receipt is an admitted, bought operation.
                   "provider_mutation_performed": actual_inputs is not None,
                   "returned_pose_frame_count": len(posed_context),
                   "native_geometry_available": False,
                   "blockers": ["website_atlas_final_scene_assets_required",
                                "website_atlas_native_geometry_binding_required"]}
        if pose.get("target_cameras") is not None:
            body = atlas_generate_from_pose_operation(operation=operation,
                expected_frame_count=len(frames), target_cameras=pose["target_cameras"],
                prompt=pose.get("prompt"))
            context.update(generation_endpoint="/api/v2/tasks:atlasGenerate", generation_request=body)
            context["blockers"].insert(0, "website_atlas_admitted_generation_operation_required")
        context["digest"] = canonical_digest(context, digest_field="digest")
        context_path = output_root / ("atlas_generation_inputs.json" if "generation_request" in context
                                      else "atlas_posed_rgbd_handoff.json")
        if context_path.is_file():
            if json.loads(context_path.read_text()) != context:
                raise ValueError("website_atlas_generation_input_identity_conflict")
        else:
            write_json(context_path, context)
        return context
    return _with_atlas_producer_status(result, output_root)


def _with_atlas_producer_status(result: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    """The pending record, plus what an admitted producer has already done.

    Once frames are uploaded or a task POST is sent, "no provider mutation"
    and "zero inputs used" would be false. The recorded pending file is left
    unchanged; only the returned view says so.
    """
    from .website_atlas_pose import atlas_pose_producer_status

    status = atlas_pose_producer_status(output_root)
    if not status:
        return dict(result)
    view = {**result, **status}
    view["digest"] = canonical_digest(view, digest_field="digest")
    return view


def _atlas_posed_context(*, operation: Mapping[str, Any], expected_frame_count: int,
                         target_cameras: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Validate the returned bundle and any explicitly requested target grid.

    The caller owns operation/source admission. This is a generation payload,
    not native source geometry, a splat, a collider, or measured metric evidence.
    https://atlas-beta.worldlabs.ai/docs/images-to-posed-images
    https://atlas-beta.worldlabs.ai/docs/cameras-and-posed-images
    """
    if (operation.get("done") is not True or operation.get("error") is not None
            or not isinstance(operation.get("id"), str) or not operation["id"].strip()):
        raise ValueError("website_atlas_pose_operation_not_succeeded")
    response = operation.get("response")
    frames = response.get("frames") if isinstance(response, Mapping) else None
    if (isinstance(expected_frame_count, bool) or not isinstance(expected_frame_count, int)
            or expected_frame_count < 1 or not isinstance(frames, list)
            or len(frames) != expected_frame_count):
        raise ValueError("website_atlas_pose_frames_invalid")

    def camera_grid(camera: Mapping[str, Any]) -> tuple[int, int]:
        if not isinstance(camera, Mapping):
            raise ValueError("website_atlas_camera_invalid")
        k, pose = camera.get("intrinsics") or {}, camera.get("extrinsics") or {}
        if not isinstance(k, Mapping) or not isinstance(pose, Mapping):
            raise ValueError("website_atlas_camera_invalid")
        width, height = k.get("width"), k.get("height")
        position, quaternion = pose.get("position"), pose.get("quaternion")
        def numeric(x: Any) -> bool:
            return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
        if (any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in (width, height))
                or not all(numeric(k.get(key)) for key in ("fx", "fy", "cx", "cy"))
                or k["fx"] <= 0 or k["fy"] <= 0
                or pose.get("coordinateSystem") not in ("rub", "rdf")
                or not isinstance(position, (list, tuple)) or len(position) != 3
                or not all(numeric(x) for x in position)
                or not isinstance(quaternion, (list, tuple)) or len(quaternion) != 4
                or not all(numeric(x) for x in quaternion)
                or not math.isclose(sum(x * x for x in quaternion), 1, abs_tol=1e-3)):
            raise ValueError("website_atlas_camera_invalid")
        return width, height

    def asset(reference: Any) -> bool:
        # Retained project asset IDs avoid exporting transient access URLs.
        return isinstance(reference, Mapping) and isinstance(reference.get("assetId"), str) and bool(reference["assetId"].strip())

    grids = set()
    context = []
    for frame in frames:
        if (not isinstance(frame, Mapping) or not asset(frame.get("imageAsset"))
                or not isinstance(frame.get("depth"), Mapping)
                or not asset(frame["depth"].get("depthAsset"))):
            raise ValueError("website_atlas_posed_rgbd_bundle_required")
        grids.add(camera_grid(frame.get("camera")))
        # Never resize, crop, relabel axes, substitute source RGB or infer a
        # source-pixel transform for the provider's reframed output.
        depth = {"depthAsset": {"assetId": frame["depth"]["depthAsset"]["assetId"]}}
        if frame["depth"].get("confidenceAsset") is not None:
            if not asset(frame["depth"]["confidenceAsset"]):
                raise ValueError("website_atlas_confidence_asset_invalid")
            depth["confidenceAsset"] = {"assetId": frame["depth"]["confidenceAsset"]["assetId"]}
        context.append({"imageAsset": {"assetId": frame["imageAsset"]["assetId"]},
                        "camera": deepcopy(frame["camera"]), "depth": depth})
    if len(grids) != 1:
        raise ValueError("website_atlas_context_grid_mismatch")
    if target_cameras is not None:
        if not target_cameras:
            raise ValueError("website_atlas_pose_frames_invalid")
        for camera in target_cameras:
            if camera_grid(camera) != (1280, 720):
                raise ValueError("website_atlas_target_grid_invalid")
    return context


def atlas_generate_from_pose_operation(*, operation: Mapping[str, Any],
                                       expected_frame_count: int,
                                       target_cameras: Sequence[Mapping[str, Any]],
                                       prompt: str | None = None) -> dict[str, Any]:
    """Build, never submit, from the unchanged returned bundle and explicit cameras."""
    if prompt is not None and not isinstance(prompt, str):
        raise ValueError("website_atlas_generation_prompt_invalid")
    context = _atlas_posed_context(operation=operation, expected_frame_count=expected_frame_count,
                                  target_cameras=target_cameras)
    return {"contextFrames": context, "targetCameras": deepcopy(list(target_cameras)), "prompt": prompt}
