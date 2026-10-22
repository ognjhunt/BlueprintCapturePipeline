"""Source-bound SAM -> reviewed editor anchor -> Atlas multiview background.

No generated background replaces capture truth. Every paid helper reuses its
existing durable receipt and the same source allowance. Bad edits are retained
for inspection and cannot enter background reconstruction.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file
from .atlas_scene_measurements import mapped_pixels


def _provider_preflight():
    """Resolve all deterministic execute dependencies before buying segmentation."""
    import shutil
    from .meta_sam31 import meta_api_key, _sam_parser_node
    from .task_evaluation_supervisor.agents_sdk import _file_based_openai_api_key
    from .website_image_completion import _api_key
    from google import genai  # noqa: F401

    meta_api_key()
    if not _file_based_openai_api_key() or not _api_key()[0]:
        raise ValueError("atlas_automation_edit_or_review_secret_missing")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe") or _sam_parser_node() is None:
        raise ValueError("atlas_automation_segmentation_runtime_missing")


def object_evidence(*, frames, objects, inputs):
    """Validate/clip source boxes in the returned grid before a paid SAM call."""
    timestamps = {
        r["frame_id"]: r.get("timestamp_seconds", i) for i, r in enumerate(inputs["frames"])
    }
    by_id = {f["frame_id"]: f for f in frames}
    result = {}
    for obj in objects:
        evidence = []
        for region in obj["regions"]:
            f = by_id[region["frame_id"]]
            x, y, w, h = region["box_xywh_normalized"]
            corners = mapped_pixels(f, [[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
            grid = np.array([f["intrinsics"]["width"], f["intrinsics"]["height"]])
            lo = np.clip(corners.min(axis=0), 0, grid)
            hi = np.clip(corners.max(axis=0), 0, grid)
            if not np.isfinite(corners).all() or np.any(hi - lo < 1):
                raise ValueError("atlas_object_region_outside_returned_view")
            evidence.append(
                {
                    "timestamp_seconds": timestamps[f["frame_id"]],
                    "box_xywh_normalized": np.r_[lo / grid, (hi - lo) / grid].tolist(),
                }
            )
        result[obj["target_id"]] = evidence
    return result


def segment_objects(*, frames, objects, inputs, task_context, source_digest, root):
    from .meta_sam31 import run_meta_sam31
    from .website_task_masks import select_task_track

    spatial = object_evidence(frames=frames, objects=objects, inputs=inputs)
    registry = [
        {
            "source_frame_id": f["frame_id"],
            "width": f["intrinsics"]["width"],
            "height": f["intrinsics"]["height"],
        }
        for f in frames
    ]
    artifacts = [
        {"source_frame_id": f["frame_id"], "path": f["image_path"], "sha256": f["image_digest"]}
        for f in frames
    ]
    prompts = [
        {
            "prompt_id": o["target_id"],
            "output_label": o["target_id"],
            "text": o["segmentation_prompt"],
        }
        for o in objects
    ]
    tracks = run_meta_sam31(
        frame_registry=registry,
        frame_artifacts=artifacts,
        prompts=prompts,
        output_root=root,
        admission={},
        task_context=task_context,
        source_video_digest=source_digest,
    )["tracks"]
    timestamps = {
        r["frame_id"]: r.get("timestamp_seconds", i) for i, r in enumerate(inputs["frames"])
    }
    identity_frames = [
        {"frame_id": f["frame_id"], "timestamp_seconds": timestamps[f["frame_id"]]} for f in frames
    ]
    by_id = {f["frame_id"]: f for f in frames}
    selected = []
    for obj in objects:
        target = {**obj, "spatial_evidence": spatial[obj["target_id"]]}
        track = select_task_track(target=target, tracks=tracks, frames=identity_frames)
        # A track must explicitly report every selected view, even an empty mask.
        if {o["source_frame_id"] for o in track["observations"]} != set(by_id):
            raise ValueError("atlas_object_mask_view_coverage_incomplete")
        selected.append({**obj, "track": track})
    write_json(
        root / "selected_objects.json", {"objects": selected, "source_video_digest": source_digest}
    )
    return selected


def prepare_masked_frames(*, frames, targets, margin, root):
    from .website_task_masks import decode_track_mask

    root.mkdir(parents=True, exist_ok=True)
    prepared = []
    for index, frame in enumerate(frames):
        h, w = frame["depth"].shape
        mask = np.zeros((h, w), bool)
        shown = []
        for target in targets:
            if target["disposition"] != "remove":
                continue
            observation = next(
                o
                for o in target["track"]["observations"]
                if o["source_frame_id"] == frame["frame_id"]
            )
            part = decode_track_mask(observation)
            if part.shape != mask.shape:
                raise ValueError("atlas_object_mask_grid_mismatch")
            mask |= part
            if part.any():
                shown.append(target["target_id"])
        if mask.any():
            mask = binary_dilation(mask, iterations=max(1, round(min(h, w) * margin)))
        image = root / f"{index:04d}.png"
        Image.fromarray(frame["rgb"]).save(image)
        path = root / f"{index:04d}-edit.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        prepared.append(
            {
                "frame_id": frame["frame_id"],
                "image_path": str(image),
                "image_digest": _sha256_file(image),
                "original_image_path": str(image),
                "original_image_digest": _sha256_file(image),
                "remaining_mask_path": str(path),
                "remaining_mask_digest": _sha256_file(path),
                "remaining_pixel_count": int(mask.sum()),
                "removed_task_object_ids": shown,
                "generated_pixels_present": False,
            }
        )
    return prepared


def tile_views(*, frames, prepared, anchor, root):
    """Overlap cover-crops across a long image: no stretching or lost portrait FOV."""
    root.mkdir(parents=True, exist_ok=True)
    tiles = []
    by_id = {f["frame_id"]: f for f in frames}
    for row in prepared:
        frame = by_id[row["frame_id"]]
        source = Path(row["image_path"])
        mask = Path(row["remaining_mask_path"])
        if (
            _sha256_file(source) != row["image_digest"]
            or _sha256_file(mask) != row["remaining_mask_digest"]
        ):
            raise ValueError("atlas_edit_source_changed")
        original = Image.open(source).convert("RGB")
        edit = Image.open(mask).convert("L")
        is_anchor = row["frame_id"] == anchor["frame_id"]
        if is_anchor:
            if _sha256_file(Path(anchor["image_path"])) != anchor["image_digest"]:
                raise ValueError("atlas_clean_anchor_changed")
            edited = Image.open(anchor["image_path"]).convert("RGB")
            if edited.size != original.size:
                raise ValueError("atlas_anchor_grid_changed")
        else:
            edited = original
        w, h = original.size
        scale = max(1280 / w, 720 / h)
        rw, rh = round(w * scale), round(h * scale)
        original = original.resize((rw, rh), Image.Resampling.LANCZOS)
        edited = edited.resize((rw, rh), Image.Resampling.LANCZOS)
        edit = edit.resize((rw, rh), Image.Resampling.NEAREST)
        xs = (
            np.linspace(0, rw - 1280, max(1, math.ceil((rw - 1280) / (1280 * 0.75)) + 1))
            .round()
            .astype(int)
        )
        ys = (
            np.linspace(0, rh - 720, max(1, math.ceil((rh - 720) / (720 * 0.75)) + 1))
            .round()
            .astype(int)
        )
        for x in xs:
            for y in ys:
                identity = f"{row['frame_id']}-t{len(tiles):04d}"
                image_path = root / f"{identity}.png"
                original_path = root / f"{identity}-original.png"
                mask_path = root / f"{identity}-keep.png"
                rectangle = (int(x), int(y), int(x + 1280), int(y + 720))
                edited.crop(rectangle).save(image_path)
                original.crop(rectangle).save(original_path)
                edit_tile = np.asarray(edit.crop(rectangle))
                keep = np.full((720, 1280), 255, np.uint8) if is_anchor else 255 - edit_tile
                Image.fromarray(keep).save(mask_path)
                camera = deepcopy(
                    {"intrinsics": frame["intrinsics"], "extrinsics": frame["extrinsics"]}
                )
                k = camera["intrinsics"]
                sx, sy = rw / w, rh / h
                k.update(
                    width=1280,
                    height=720,
                    fx=k["fx"] * sx,
                    fy=k["fy"] * sy,
                    cx=k["cx"] * sx - int(x),
                    cy=k["cy"] * sy - int(y),
                )
                tiles.append(
                    {
                        "frame_id": identity,
                        "source_frame_id": row["frame_id"],
                        "image_path": str(image_path),
                        "image_digest": _sha256_file(image_path),
                        "original_image_path": str(original_path),
                        "original_image_digest": _sha256_file(original_path),
                        "keep_mask_path": str(mask_path),
                        "keep_mask_digest": _sha256_file(mask_path),
                        "camera": camera,
                        "clean_anchor": is_anchor,
                        "edit_pixels": int(np.count_nonzero(edit_tile)),
                        "pixel_transform": {
                            "resize_xy": [sx, sy],
                            "crop_xy": [int(x), int(y)],
                            "source_grid": [w, h],
                        },
                    }
                )
    if not any(r["clean_anchor"] for r in tiles):
        raise ValueError("atlas_clean_anchor_missing")
    return tiles


def batches(tiles):
    # Sixteen is the vendor's practical context per request, not a scene/frame ceiling.
    anchor = max((r for r in tiles if r["clean_anchor"]), key=lambda r: r["edit_pixels"])
    others = [r for r in tiles if r is not anchor]
    result = []
    for i in range(0, len(others), 14):
        group = [anchor, *others[i : i + 14]]
        center = anchor["camera"]["extrinsics"]["position"]
        if all(math.dist(center, r["camera"]["extrinsics"]["position"]) < 1e-6 for r in group):
            translated = next(
                (
                    r
                    for r in tiles
                    if math.dist(center, r["camera"]["extrinsics"]["position"]) >= 1e-6
                ),
                None,
            )
            if translated is not None:
                group.append(translated)
        result.append(group)
    return result or [[anchor]]


def masked_request(rows, instruction):
    context = []
    uploads = {}
    for i, row in enumerate(rows):
        for role, path, digest in [
            ("image", row["image_path"], row["image_digest"]),
            ("mask", row["keep_mask_path"], row["keep_mask_digest"]),
        ]:
            uploads[f"{role}-{i}"] = {"path": path, "sha256": digest}
        context.append(
            {
                "imageAsset": {"assetId": f"upload:image-{i}"},
                "maskAsset": {"assetId": f"upload:mask-{i}"},
                "camera": row["camera"],
            }
        )
    centers = [r["camera"]["extrinsics"]["position"] for r in rows]
    translated = len(rows) > 1 and max(math.dist(centers[0], p) for p in centers) > 1e-6
    request = {
        "contextFrames": context,
        "targetCameras": [r["camera"] for r in rows],
        "prompt": instruction,
        "enhancePrompt": True,
        "returnDepth": translated,
    }
    from .website_atlas_tasks import validate_task_request

    validate_task_request("atlasMasked", request)
    return request, uploads


def run_background(
    *,
    frames,
    objects,
    automation,
    inputs,
    pose,
    root,
    context,
    policy,
    source_digest,
    task_runner,
    release_blockers,
    max_polls=2,
    configuration=None,
):
    from .website_image_completion import (
        complete_background_images,
        verify_completed_background,
        _validated_backend,
        REGISTRY_PATH,
        BACKEND_ID,
    )
    from .atlas_scene_measurements import estimate_sizes

    root.mkdir(parents=True, exist_ok=True)
    if not objects:
        return {"status": "not_requested", "reason": "no_object_regions_proposed"}
    commit, blockers = release_blockers()
    if blockers:
        raise ValueError("atlas_automation_release_not_admitted:" + ",".join(blockers))
    # Complete deterministic geometry and cost checks before the first upload.
    object_evidence(frames=frames, objects=objects, inputs=inputs)
    if any(o["disposition"] == "remove" for o in objects):
        centers = [f["extrinsics"]["position"] for f in frames]
        if not centers or max(math.dist(centers[0], p) for p in centers) < 1e-6:
            raise ValueError("atlas_background_depth_requires_translated_source_views")
    _, execution, _ = _validated_backend(REGISTRY_PATH, backend_id=BACKEND_ID)
    tile_count = 0
    for f in frames:
        k = f["intrinsics"]
        factor = max(1280 / k["width"], 720 / k["height"])
        tile_count += max(
            1, math.ceil((round(k["width"] * factor) - 1280) / (1280 * 0.75)) + 1
        ) * max(1, math.ceil((round(k["height"] * factor) - 720) / (720 * 0.75)) + 1)
    groups = max(1, math.ceil(max(0, tile_count - 1) / 14))
    # Reserve forecasting space for repeated shared anchors and independent reviews.
    forecast = round(
        (tile_count + 2 * groups) * policy["rgbd_view_cost_usd"] * 2
        + float(execution["pricing_binding"]["max_cost_per_request_usd"])
        + max(50, len(frames)) * 0.0002 * len(objects)
        + 0.10
        + groups * 0.10
        + policy.get("mesh_export_cost_usd", 0),
        6,
    )
    write_json(
        root / "preflight.json",
        {
            "estimated_maximum_path_cost_usd": forecast,
            "tiles": tile_count,
            "batches": groups,
            "original_views": len(frames),
            "same_source_allowance": True,
            "fallback_requires_remaining_allowance": True,
        },
    )
    if (
        not (root / "anchor").exists()
        and configuration is not None
        and configuration["remaining_cost_usd"] + 1e-9 < forecast
    ):
        raise ValueError("atlas_automation_budget_insufficient:" + str(forecast))
    _provider_preflight()
    selected = segment_objects(
        frames=frames,
        objects=objects,
        inputs=inputs,
        task_context=context,
        source_digest=source_digest,
        root=root / "sam",
    )
    measurements = estimate_sizes(
        frames=frames,
        automation=automation,
        pose_digest=pose["digest"],
        tracks=selected,
        output=root / "object_measurements.json",
    )
    removed = [r for r in selected if r["disposition"] == "remove"]
    if not removed:
        return {
            "status": "not_requested",
            "measurements": measurements,
            "reason": "all_objects_kept",
        }
    prepared = prepare_masked_frames(
        frames=frames,
        targets=selected,
        margin=automation["mask_margin_fraction"],
        root=root / "originals",
    )
    if not any(r["remaining_pixel_count"] for r in prepared):
        raise ValueError("atlas_removal_masks_empty")
    targets = [
        {
            **r,
            "semantic_label": r["label"],
            "task_effect": "manipulated" if r["disposition"] == "remove" else "static_contact",
        }
        for r in objects
    ]
    task_digest = sha256(
        json.dumps(dict(context), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    plan = {"targets": targets, "task_context_sha256": task_digest}
    anchor_source = max(
        prepared, key=lambda r: (len(r["removed_task_object_ids"]), r["remaining_pixel_count"])
    )
    preferred = next(
        (
            r
            for r in prepared
            if r["frame_id"] == automation.get("anchor_frame_id") and r["remaining_pixel_count"]
        ),
        None,
    )
    if preferred is not None:
        anchor_source = preferred
    anchor = complete_background_images(
        frames=[anchor_source],
        task_digest=task_digest,
        output_root=root / "anchor",
        admission={},
        token="",
        targets=targets,
        task_context=context,
        source_video_digest=source_digest,
        repair_instruction=automation["removal_instruction"],
    )[0]
    anchor_review = verify_completed_background(
        frames=[anchor],
        original_frames=[anchor_source],
        plan=plan,
        output_root=root / "anchor_review",
        task_context=context,
        source_video_digest=source_digest,
    )
    if anchor_review["status"] != "passed":
        return {
            "status": "quality_rejected",
            "stage": "anchor",
            "review": anchor_review,
            "measurements": measurements,
            "retained_root": str(root),
        }
    tiles = tile_views(frames=frames, prepared=prepared, anchor=anchor, root=root / "tiles")
    groups = batches(tiles)
    stage_roots = []
    results = []
    reviewed = []
    for i, rows in enumerate(groups):
        request, uploads = masked_request(rows, automation["removal_instruction"])
        cost = round(len(rows) * policy["rgbd_view_cost_usd"] * 2, 6)
        quote = {
            "schema_version": "website_atlas_task_price_review.v1",
            "task": "atlasMasked",
            "request_digest": canonical_digest(request),
            "maximum_cost_usd": cost,
            **{k: policy[k] for k in ["rate_card_reference", "reviewed_by", "reviewed_on"]},
        }
        stage = root / "masked" / str(i)
        result = task_runner(
            task="atlasMasked",
            request=request,
            uploads=uploads,
            output_root=stage,
            source_video_digest=source_digest,
            task_context=context,
            price_review=quote,
            max_polls=max_polls,
        )
        if result["status"] != "completed":
            return {
                "status": "processing",
                "stage": "atlasMasked",
                "operation_id": result.get("operation_id"),
            }
        stage_roots.append(stage)
        results.append(result)
        candidates = []
        originals = []
        for j, row in enumerate(rows):
            output = next(o for o in result["outputs"] if o["role"] == f"frames-{j}-imageAsset")
            candidates.append(
                {
                    "frame_id": row["frame_id"],
                    "image_path": str(stage / output["path"]),
                    "image_digest": output["sha256"],
                }
            )
            originals.append(
                {
                    "frame_id": row["frame_id"],
                    "image_path": row["original_image_path"],
                    "image_digest": row["original_image_digest"],
                }
            )
        review = verify_completed_background(
            frames=candidates,
            original_frames=originals,
            plan=plan,
            output_root=root / "reviews" / str(i),
            task_context=context,
            source_video_digest=source_digest,
        )
        reviewed.append(review)
        if review["status"] != "passed":
            # Keep the existing image-editor method as the bounded fallback. Its
            # own receipts and independent review still gate reconstruction.
            fallback = []
            # The shared reservation API accepts <=32 requests per transaction;
            # retain separate batches instead of imposing that limit on a scene.
            for offset in range(0, len(prepared), 32):
                fallback.extend(
                    complete_background_images(
                        frames=prepared[offset : offset + 32],
                        task_digest=task_digest,
                        output_root=root / "fallback" / str(offset // 32),
                        admission={},
                        token="",
                        targets=targets,
                        task_context=context,
                        source_video_digest=source_digest,
                        repair_instruction=automation["removal_instruction"],
                        repair_reference_path=Path(anchor["image_path"]),
                    )
                )
            fallback_review = verify_completed_background(
                frames=fallback,
                original_frames=prepared,
                plan=plan,
                output_root=root / "fallback_review",
                task_context=context,
                source_video_digest=source_digest,
            )
            return {
                "status": "fallback_prepared"
                if fallback_review["status"] == "passed"
                else "quality_rejected",
                "provider": "existing_image_editor",
                "frames": fallback,
                "review": fallback_review,
                "atlas_review": review,
                "measurements": measurements,
                "retained_root": str(root),
                "ready_for_reconstruction": fallback_review["status"] == "passed",
            }
    write_json(
        root / "tile_lineage.json",
        {"tiles": tiles, "batches": [[r["frame_id"] for r in g] for g in groups]},
    )
    return {
        "status": "passed",
        "provider": "atlas",
        "measurements": measurements,
        "stage_roots": [str(p) for p in stage_roots],
        "result_digests": [r["digest"] for r in results],
        "visual_review_digest": canonical_digest({"reviews": reviewed}),
        "tiles": tiles,
        "batch_frame_ids": [[r["frame_id"] for r in g] for g in groups],
        "ready_for_reconstruction": True,
        "source_commit": commit,
    }
