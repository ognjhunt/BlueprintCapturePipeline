"""Original conservative mask and receipt continuation for prepared scene views."""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .decision_evidence_contracts import canonical_digest, canonical_json


def finish_prepared_inputs(context: Mapping[str, Any], *, sealed_render_manifests,
        rgb_run, support_run, background_run, render_frame_subdir,
        render_execution_evidence=None, validate_only=False) -> dict[str, Any]:
    from .public_scene_inpainting_inputs import (
        PublicSceneInpaintingInputError, SEALED_SOURCE_ADAPTERS, RECEIPT_SCHEMA_V2,
        RECEIPT_SCHEMA, RENDER_HARNESS_REL, RENDER_ENTRY_REL, _record, _project_obb, _sha256,
    )
    paths = context["paths"]
    repo, output = Path(paths["repo"]), Path(paths["output"])
    data = Path(paths["data"])
    standard_ply, target_ply = Path(paths["standard_ply"]), Path(paths["target_ply"])
    background_ply, camera_file = Path(paths["background_ply"]), Path(paths["camera_file"])
    request_file = Path(paths["request_file"])
    retained_receipt = Path(paths["retained_receipt"]) if paths["retained_receipt"] else None
    request, repository = context["request"], context["repository"]
    rendering, cameras = request["rendering"], context["cameras"]
    corners, target_center = np.asarray(context["corners"]), np.asarray(context["target_center"])
    source_adapter, source_identity = context["source_adapter"], context["source_identity"]
    observed_sources, target_count = context["observed_sources"], context["target_count"]
    manifest, component_receipt = context["manifest"], context["component_receipt"]
    conversion, decode_command = context["conversion"], context["decode_command"]
    width, height = int(rendering["width"]), int(rendering["height"])
    dilation = int(request["mask_policy"]["dilation_pixels"])
    mask_rows = []
    image_rows = []
    for camera in cameras:
        camera_id = camera["camera_id"]
        rgb = output / "images" / render_frame_subdir / f"{camera_id}.png"
        support = output / "target_support" / render_frame_subdir / f"{camera_id}.png"
        background = (
            output / "scene_without_target" / render_frame_subdir / f"{camera_id}.png"
        )
        if not rgb.is_file() or not support.is_file() or not background.is_file():
            raise PublicSceneInpaintingInputError([f"edit_input_render_missing:{camera_id}"])
        rgb_pixels = np.asarray(Image.open(rgb).convert("RGB"))
        support_pixels = np.asarray(Image.open(support).convert("RGB"))
        background_pixels = np.asarray(Image.open(background).convert("RGB"))
        if (
            rgb_pixels.shape[:2] != (height, width)
            or support_pixels.shape[:2] != (height, width)
            or background_pixels.shape[:2] != (height, width)
        ):
            raise PublicSceneInpaintingInputError([f"edit_input_render_size_mismatch:{camera_id}"])
        if float(rgb_pixels.std()) < 1.0:
            raise PublicSceneInpaintingInputError([f"edit_input_rgb_blank:{camera_id}"])
        support_mask = Image.fromarray(
            (np.max(support_pixels, axis=2) >= int(request["mask_policy"]["support_threshold_8bit"]))
            .astype(np.uint8) * 255, mode="L",
        )
        if dilation:
            support_mask = support_mask.filter(ImageFilter.MaxFilter(2 * dilation + 1))
        obb_mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(obb_mask).polygon(_project_obb(corners, camera), fill=255)
        final = Image.fromarray(
            np.maximum(np.asarray(obb_mask), np.asarray(support_mask)).astype(np.uint8), mode="L"
        )
        mask_path = output / "masks" / f"{camera_id}.png"
        if validate_only:
            if mask_path.is_symlink() or not mask_path.is_file():
                raise PublicSceneInpaintingInputError(["edit_input_retained_mask_missing"])
            with Image.open(mask_path) as saved_mask:
                if not np.array_equal(np.asarray(saved_mask.convert("L")), np.asarray(final)):
                    raise PublicSceneInpaintingInputError(["edit_input_retained_mask_changed"])
        else:
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            final.save(mask_path, format="PNG", optimize=False)
        final_pixels = np.asarray(final) > 0
        support_binary = np.asarray(support_mask) > 0
        coverage = float(final_pixels.mean())
        maximum_fraction = float(request["mask_policy"].get("maximum_image_fraction", 0.2))
        if not 0.00001 < coverage < maximum_fraction or int(support_binary.sum()) == 0:
            raise PublicSceneInpaintingInputError([f"edit_input_mask_invalid:{camera_id}"])
        support_inside = float((support_binary & final_pixels).sum() / support_binary.sum())
        if support_inside < float(request["mask_policy"]["minimum_support_inside_final_fraction"]):
            raise PublicSceneInpaintingInputError([f"edit_input_mask_support_mismatch:{camera_id}"])
        contribution = np.max(
            np.abs(rgb_pixels.astype(np.int16) - background_pixels.astype(np.int16)), axis=2
        ) >= int(request["mask_policy"].get("visual_contribution_threshold_8bit", 8))
        visible_pixels = int((contribution & final_pixels).sum())
        visible_fraction = float(visible_pixels / final_pixels.sum())
        if visible_fraction < float(
            request["mask_policy"].get("minimum_visible_target_fraction", 0.01)
        ):
            raise PublicSceneInpaintingInputError(
                [f"edit_input_target_occluded_or_unrenderable:{camera_id}"]
            )
        image_rows.append({"camera_id": camera_id, **_record(rgb, output)})
        mask_rows.append(
            {"camera_id": camera_id, **_record(mask_path, output),
             "masked_pixel_count": int(final_pixels.sum()), "image_fraction": round(coverage, 9),
             "gaussian_support_inside_fraction": round(support_inside, 9),
             "visible_target_contribution_pixel_count": visible_pixels,
             "visible_target_contribution_fraction": round(visible_fraction, 9),
             "scene_without_target_render": _record(background, output)}
        )
    if source_adapter in SEALED_SOURCE_ADAPTERS:
        renderer = {
            "name": "reference_spark_renderer_exact_camera",
            "authorization_class": "method_input",
            "purpose_bound": True,
            "render_manifest_digests": {
                label: row["sealed_camera_render_manifest_digest"]
                for label, row in sealed_render_manifests.items()
            },
            "render_settings": sealed_render_manifests["images"]["render_settings"],
            "renderer_identity": sealed_render_manifests["images"][
                "renderer_identity"
            ],
        }
        standard_splat_record = next(
            row for row in observed_sources if row["role"] == "standard_splat"
        )
        source_admission = {
            "adapter": source_adapter,
            "scene_freeze_digest": source_identity["scene_freeze_digest"],
            "task_freeze_digest": source_identity["task_freeze_digest"],
            "standard_splat_conversion_receipt_digest": source_identity[
                "conversion_receipt_digest"
            ],
            "registered_frame_receipt_digest": source_identity[
                "registered_frame_receipt_digest"
            ],
            "registered_frame_status": source_identity["registered_frame_status"],
        }
    else:
        renderer = {
            "name": "reference_spark_renderer_exact_camera",
            "authorization_class": "legacy_unqualified",
            "harness_sha256": _sha256(repo / RENDER_HARNESS_REL),
            "entry_sha256": _sha256(repo / RENDER_ENTRY_REL),
            "node_version": subprocess.run(
                ["node", "--version"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            "graphics_backend": rendering["graphics_backend"],
            "width": width,
            "height": height,
            "warmup_ms": rendering["warmup_ms"],
            "settle_frames": rendering["settle_frames"],
            "settle_ms": rendering["settle_ms"],
        }
        standard_splat_record = _record(standard_ply, output)
        source_admission = {
            "adapter": source_adapter,
            "scene_component_manifest_digest": manifest["manifest_digest"],
            "scene_component_receipt_digest": component_receipt["receipt_digest"],
        }
    receipt = {
        "schema_version": (
            RECEIPT_SCHEMA_V2
            if source_adapter in SEALED_SOURCE_ADAPTERS
            else RECEIPT_SCHEMA
        ),
        "status": "render_derived_input_packet_materialized",
        "program_id": "arm-decision-proof-v1",
        "adp_item": request["adp_item"],
        "repository": repository,
        "request_digest": request["request_digest"],
        "source_admission": source_admission,
        "scene": {
            "publisher_scene_id": source_identity["scene_id"],
            "task_id": source_identity["task_id"],
            "target_instance_id": source_identity["target_instance_id"],
            "target_semantic_label": source_identity["target_semantic_label"],
            "mask_set_id": source_identity["mask_set_id"],
            "removal_id": source_identity["removal_id"],
            "target_obb_corners_m": corners.tolist(), "target_gaussian_count": target_count,
            "scene_gaussian_count": context["scene_gaussian_count"],
        },
        "source_artifacts": observed_sources,
        "derived_artifacts": {
            "standard_splat": standard_splat_record,
            "target_gaussian_support": _record(target_ply, output),
            "scene_without_target_obb_gaussians": _record(background_ply, output),
            "cameras": _record(camera_file, output), "images": image_rows, "masks": mask_rows,
        },
        "camera_policy": {
            "generator": "translated_target_coverage_v1", "orbit_only": False,
            "camera_count": len(cameras),
            "radii_m": [
                round(float(np.linalg.norm(np.asarray(row["T_world_camera_opencv"])[:3, 3] - target_center)), 6)
                for row in cameras
            ],
        },
        "camera_pose_contract": {
            "schema_version": "public_scene_inpainting_camera_pose_contract.v1",
            "camera_file_pose_field": (
                "T_world_camera_provider_frame"
                if source_adapter in SEALED_SOURCE_ADAPTERS
                else "T_world_camera_opencv"
            ),
            "semantic_pose_field": "T_world_camera_opencv",
            "camera_coordinate_convention": "opencv_x_right_y_down_z_forward",
            "provider_frame_aliases_opencv": (
                source_adapter in SEALED_SOURCE_ADAPTERS
            ),
        },
        "mask_policy": {
            "authority": request["mask_policy"]["authority"],
            "dilation_pixels": dilation,
            "maximum_image_fraction": float(
                request["mask_policy"].get("maximum_image_fraction", 0.2)
            ),
            "visual_contribution_threshold_8bit": int(
                request["mask_policy"].get("visual_contribution_threshold_8bit", 8)
            ),
            "minimum_visible_target_fraction": float(
                request["mask_policy"].get("minimum_visible_target_fraction", 0.01)
            ),
        },
        "renderer": renderer,
        "executed_commands": {
            "decode": conversion.get("command") or decode_command, "rgb_render": rgb_run["command"],
            "target_support_render": support_run["command"],
            "scene_without_target_render": background_run["command"],
        },
        "method_execution": {
            "inpaint360gs_executed": False, "infusion_executed": False,
            "aurafusion360_executed": False,
        },
        "proof_boundaries": {
            "uses_original_capture_frames": False, "uses_rendered_scene_consistent_rgb": True,
            "hidden_background_truth_available": False,
            "source_target_obb_visual_contribution_measured": True,
            "source_object_removed_from_appearance": False, "source_collider_removed": False,
            "simready_replacement_inserted": False, "inpainting_result": False,
            "mask_is_calibrated_candidate_not_owned_gaussian_classification": True,
            "gaussian_ownership_qualified": False,
        },
        "smallest_next_blocker": (
            "independent_gaussian_contribution_ownership_and_replacement_depth_coverage"
            if source_adapter in SEALED_SOURCE_ADAPTERS
            else "method_native_interiorgs_adapter_and_unchanged_author_runtime_required"
        ),
        "claim_ceiling": "synthetic_public_scene_inpainting_input_candidate",
        "replay_command": shlex.join([
            "python", "-m", "blueprint_pipeline.public_scene_inpainting_inputs",
            "--request", str(request_file), "--repo-root", str(repo),
            "--data-root", str(data), "--output-root", str(output),
        ]),
    }
    if render_execution_evidence is not None:
        receipt["source_calibration_render"] = dict(render_execution_evidence)
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    if validate_only:
        return receipt
    output_receipt_name = (
        "public_scene_interiorgs_edit_input_receipt.v2.json"
        if source_adapter in SEALED_SOURCE_ADAPTERS
        else "adp009b_interiorgs_edit_input_receipt.v1.json"
    )
    (output / output_receipt_name).write_text(
        canonical_json(receipt) + "\n", encoding="utf-8"
    )
    if retained_receipt is not None:
        retained_receipt.parent.mkdir(parents=True, exist_ok=True)
        retained_receipt.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt
