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


def _prepared_background(path, binding, value=None):
    """Retain reviewed RGB independently of later pose/mesh jobs and sessions."""
    if value is None:
        if not path.is_file():
            return None
        saved = json.loads(path.read_text())
        if (saved.get("digest") != canonical_digest(saved, digest_field="digest")
                or saved.get("binding") != binding):
            raise ValueError("atlas_prepared_background_binding_changed")
        value = saved["background"]
    review = value.get("review") or {}
    rows = [{"frame_id": row["frame_id"], "image_digest": row["image_digest"]}
            for row in value.get("frames", [])]
    if (value.get("status") != "fallback_prepared" or not rows
            or value.get("ready_for_reconstruction") is not True
            or review.get("status") != "passed"
            or review.get("binding", {}).get("frames") != rows
            or review.get("request_digest") != canonical_digest(review.get("binding"))
            or not all(review.get("review", {}).get(key) is True for key in (
                "consistent_background", "task_objects_removed", "unrelated_objects_preserved"))):
        raise ValueError("atlas_prepared_background_review_invalid")
    for row in value["frames"]:
        for prefix in ("image", "original_image"):
            if _sha256_file(Path(row[f"{prefix}_path"])) != row[f"{prefix}_digest"]:
                raise ValueError("atlas_prepared_background_pixels_changed")
    if not path.is_file():
        saved = {"schema_version": "atlas_prepared_background.v1", "binding": binding,
                 "background": value, "raw_capture_preserved": True,
                 "claim_ceiling": "development_only"}
        saved["digest"] = canonical_digest(saved, digest_field="digest")
        temporary = path.with_suffix(".tmp")
        write_json(temporary, saved)
        temporary.replace(path)
    return value


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


def grounded_region_guide(*, target, frames):
    """Source-grounded location hint for image editing, never a segmentation mask."""
    evidence = list(zip(target['regions'], target['spatial_evidence'], strict=True))
    observations = []
    for frame in frames:
        h, w = frame['rgb'].shape[:2]
        mask = np.zeros((h, w), bool)
        for region, row in evidence:
            if region['frame_id'] != frame['frame_id']:
                continue
            x, y, width, height = row['box_xywh_normalized']
            left, top = math.floor(x * w), math.floor(y * h)
            right, bottom = math.ceil((x + width) * w), math.ceil((y + height) * h)
            mask[max(0, top):min(h, bottom), max(0, left):min(w, right)] = True
        edges = np.diff(np.pad(mask.reshape(-1).astype(np.int8), (1, 1)))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        observations.append({'source_frame_id': frame['frame_id'], 'width': w, 'height': h,
            'runs': [{'start': int(start), 'length': int(end - start)} for start, end in zip(starts, ends, strict=True)]})
    if not any(row['runs'] for row in observations):
        raise ValueError('atlas_grounded_image_guide_empty')
    return {'track_id': target['target_id'] + '-grounded-region-guide', 'label': target['target_id'],
        'observations': observations, 'guide_only': True, 'physical_evidence': False,
        'selection_basis': 'agent_grounded_region_not_segmentation',
        'spatial_evidence_digest': canonical_digest({'regions': target['spatial_evidence']})}


def complete_track_coverage(*, track, target, frames):
    """Report every selected view for a SAM track, which names only the views it found.

    A view the planner marked for this object that the track misses takes the
    planner's box as an image-edit hint, so the object is never left in that
    edit; any other unreported view has nothing to remove. 2026-10-24: the chosen
    dish track covered 3 of 10 views and missed one the planner had marked.
    """
    observed = {row['source_frame_id']: row for row in track['observations']}
    missing = [frame for frame in frames if frame['frame_id'] not in observed]
    if not missing:
        return track
    marked = {region['frame_id'] for region in target['regions']}
    hints = ({row['source_frame_id']: row for row in grounded_region_guide(target=target, frames=frames)['observations']}
             if any(frame['frame_id'] in marked for frame in missing) else {})
    observations = []
    for frame in frames:
        h, w = frame['rgb'].shape[:2]
        observations.append(observed.get(frame['frame_id']) or hints.get(frame['frame_id'])
                            or {'source_frame_id': frame['frame_id'], 'width': w, 'height': h, 'runs': []})
    return {**track, 'observations': observations,
            **({'guide_only': True, 'physical_evidence': False,
                'selection_basis': 'sam31_track_with_agent_grounded_regions'} if hints else {})}


def segment_objects(*, frames, objects, inputs, task_context, source_digest, root, transients=(), selected_tracks=None):
    """Segment reviewed objects; region fallbacks remain image-edit hints only.

    A transient (people, carried items, moving equipment) is never a single
    chosen instance: every one the concept finds is merged per view and erased.
    """
    from .meta_sam31 import run_meta_sam31
    from .website_task_masks import select_task_track, union_person_tracks

    selected_tracks = selected_tracks or {}
    if set(selected_tracks) - {row['target_id'] for row in objects}:
        raise ValueError('atlas_selected_source_track_target_changed')
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
        for o in [*objects, *transients] if o["target_id"] not in selected_tracks
    ]
    tracks = run_meta_sam31(
        frame_registry=registry,
        frame_artifacts=artifacts,
        prompts=prompts,
        output_root=root,
        admission={},
        task_context=task_context,
        source_video_digest=source_digest,
    )["tracks"] if prompts else []
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
        if obj['target_id'] in selected_tracks:
            from .website_track_mask_codec import decode_track_mask
            track = selected_tracks[obj['target_id']]
            observations = track.get('observations') or []
            if (track.get('guide_only') or len(observations) != len(by_id)
                    or {row.get('source_frame_id') for row in observations} != set(by_id)):
                raise ValueError('atlas_selected_source_track_coverage_changed')
            for observation in observations:
                frame = by_id[observation['source_frame_id']]
                if decode_track_mask(observation).shape != frame['rgb'].shape[:2]:
                    raise ValueError('atlas_selected_source_track_pixels_changed')
        else:
            try:
                track = select_task_track(target=target, tracks=tracks, frames=identity_frames)
            except ValueError as exc:
                if str(exc) != f"task_target_track_ambiguous:{obj['target_id']}":
                    raise
                # The image editor needs a location hint, not a single physical
                # instance. Keep strict track selection unchanged for geometry.
                track = grounded_region_guide(target=target, frames=frames)
            else:
                track = complete_track_coverage(track=track, target=target, frames=frames)
        # A track must explicitly report every selected view, even an empty mask.
        if {o["source_frame_id"] for o in track["observations"]} != set(by_id):
            raise ValueError("atlas_object_mask_view_coverage_incomplete")
        selected.append({**obj, "track": track})
    for transient in transients:
        members = [t for t in tracks if t.get("label") == transient["target_id"]]
        selected.append({**transient, "disposition": "remove", "task_effect": "transient",
                         "track": union_person_tracks(members, label=transient["target_id"])})
    write_json(
        root / "selected_objects.json", {"objects": selected, "source_video_digest": source_digest}
    )
    return selected


def prepare_masked_frames(*, frames, targets, margin, root):
    from .website_task_masks import decode_track_mask

    root.mkdir(parents=True, exist_ok=True)
    prepared = []
    for index, frame in enumerate(frames):
        h, w = frame["rgb"].shape[:2]
        mask = np.zeros((h, w), bool)
        shown, transients = [], []
        for target in targets:
            if target["disposition"] != "remove":
                continue
            observation = next(
                (o for o in target["track"]["observations"]
                 if o["source_frame_id"] == frame["frame_id"]),
                None,
            )
            if observation is None:
                # Object tracks report every view; a transient is simply absent.
                if target.get("task_effect") != "transient":
                    raise ValueError("atlas_object_mask_view_coverage_incomplete")
                continue
            part = decode_track_mask(observation)
            if part.shape != mask.shape:
                raise ValueError("atlas_object_mask_grid_mismatch")
            mask |= part
            if part.any():
                (transients if target.get("task_effect") == "transient" else shown).append(target["target_id"])
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
                **({"removed_transient_ids": transients} if transients else {}),
                "generated_pixels_present": False,
                **({'mask_guide_basis': 'agent_grounded_region',
                    'mask_guide_evidence_digest': canonical_digest({'targets': [
                        {'target_id': target['target_id'], 'track_digest': canonical_digest(target['track'])}
                        for target in targets if target['disposition'] == 'remove' and target['track'].get('guide_only')]})}
                   if any(target['disposition'] == 'remove' and target['track'].get('guide_only') for target in targets) else {}),
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


def batches(tiles, *, fixed_context=()):
    # Sixteen is the vendor's practical context per request, not a scene/frame ceiling.
    if fixed_context:
        context_ids = {row['frame_id'] for row in fixed_context}
        others = [row for row in tiles if row['frame_id'] not in context_ids]
        # Keep all source context when a larger scene exceeds that practical
        # batch size. Actual request prices still pass the same total cap.
        stride = max(1, 16 - len(fixed_context))
        return [[*fixed_context, *others[i:i + stride]] for i in range(0, len(others), stride)] or [list(fixed_context)]
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


def reusable_background(path, *, source_digest, context, frames, objects):
    """Adopt reviewed RGB only for identical source pixels and removal targets.

    Selection may grow and poses may change. The old manifest and its image
    bytes stay untouched, and the expanded set still receives a fresh review.
    """
    if path is None:
        return [], None
    path = Path(path)
    saved = json.loads(path.read_text())
    binding = saved['binding']
    if (binding['source_video_digest'] != source_digest
            or binding['task_context_digest'] != canonical_digest(dict(context))):
        raise ValueError('atlas_prior_background_source_or_context_changed')
    prior = _prepared_background(path, binding)
    by_id = {f['frame_id']: f for f in frames}
    removed = {obj['target_id'] for obj in objects if obj['disposition'] == 'remove'}
    result = []
    for row in prior['frames']:
        if row['frame_id'] not in by_id:
            continue
        with Image.open(row['original_image_path']) as image:
            matches = np.array_equal(np.asarray(image.convert('RGB')), by_id[row['frame_id']]['rgb'])
        if not matches:
            raise ValueError('atlas_prior_background_source_pixels_changed')
        if not set(row.get('removed_task_object_ids', [])) <= removed:
            raise ValueError('atlas_prior_background_removal_changed')
        result.append(row)
    return result, saved['digest']


def _reviewed_editor_background(*, prepared, anchor, plan, automation, root, context,
                                source_digest, measurements, selected_frame_ids, check,
                                prepared_binding, atlas_review=None, reused_frames=(), prior_digest=None):
    from .website_image_completion import complete_background_images, verify_completed_background
    from .website_image_repair_agent import image_repair_enabled, repair_rejected_views

    wanted = set(selected_frame_ids or [row["frame_id"] for row in prepared])
    chosen = [row for row in prepared if row["frame_id"] in wanted]
    if len(chosen) != len(wanted) or not chosen:
        raise ValueError("atlas_background_selected_frames_missing")
    options = dict(task_digest=plan["task_context_sha256"], admission={}, token="",
                   targets=plan["targets"], task_context=context, source_video_digest=source_digest,
                   repair_instruction=automation["removal_instruction"],
                   repair_reference_path=Path(anchor["image_path"]), mask_guidance=True)
    candidates = []
    # A prior run may have edited a larger set. Read its exact completed batch
    # without authority to buy a missing image, then review only the current set.
    for offset in range(0, len(prepared), 32):
        try:
            candidates.extend(complete_background_images(
                frames=prepared[offset:offset + 32], output_root=root / "fallback" / str(offset // 32),
                require_cached=True, **options))
        except ValueError as exc:
            if str(exc) != "website_image_completion_cached_batch_missing":
                raise
    by_id = {row["frame_id"]: row for row in candidates}
    by_id.update({row['frame_id']: row for row in reused_frames})
    missing = [row for row in chosen if row["frame_id"] not in by_id]
    for offset in range(0, len(missing), 32):
        check("automation_editor_cleanup")
        edited = complete_background_images(frames=missing[offset:offset + 32],
            output_root=root / "selected_editor" / str(offset // 32), **options)
        by_id.update({row["frame_id"]: row for row in edited})
    candidates = [by_id[row["frame_id"]] for row in chosen]
    check("automation_editor_review")
    review = verify_completed_background(frames=candidates, original_frames=chosen, plan=plan,
        output_root=root / "selected_editor_review", task_context=context, source_video_digest=source_digest)
    seen = {tuple(row["image_digest"] for row in candidates)}
    repair_round = 0
    while review["status"] != "passed" and image_repair_enabled():
        check("automation_targeted_repairs")
        # Each round consumes only current source-bound grants. Completed plans,
        # edits and reviews replay from receipts after a restart. A new defect
        # named by the reviewer becomes the next repair, never an approval.
        repair_root = root / ("fallback_repairs" if repair_round == 0 else
                              f"fallback_repairs-{repair_round + 1:04d}")
        candidates, review = repair_rejected_views(
            selected=candidates, object_removal_frames=[{
                **row, "original_image_path": row["image_path"], "original_image_digest": row["image_digest"],
            } for row in chosen], original_frames=chosen, plan=plan, failed_review=review,
            output_root=repair_root, task_context=context,
            source_video_digest=source_digest, max_repairs=len(chosen),
            repair_reference_path=Path(anchor["image_path"]))
        repair_round += 1
        signature = tuple(row["image_digest"] for row in candidates)
        if review["status"] != "passed" and signature in seen:
            break
        seen.add(signature)
    background = {"status": "fallback_prepared" if review["status"] == "passed" else "quality_rejected",
                  "provider": "existing_image_editor", "frames": candidates, "review": review,
                  "measurements": measurements, "retained_root": str(root),
                  "ready_for_reconstruction": review["status"] == "passed", "repair_rounds": repair_round}
    if prior_digest:
        background.update(prior_prepared_background_digest=prior_digest,
                          reused_prepared_frame_ids=[row['frame_id'] for row in reused_frames
                              if any(final['frame_id'] == row['frame_id'] and final['image_digest'] == row['image_digest']
                                     for final in candidates)])
    if atlas_review is not None:
        background["atlas_review"] = atlas_review
    if background["ready_for_reconstruction"]:
        _prepared_background(root / "prepared_background.json", prepared_binding, background)
    return background


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
    checkpoint=None,
    selected_frame_ids=None,
    editor_first=True,
    transients=(),
    prior_background_path=None,
    require_cached=False,
    selected_tracks=None,
):
    from .website_image_completion import (
        complete_background_images,
        verify_completed_background,
        _validated_backend,
        REGISTRY_PATH,
        BACKEND_ID,
    )
    from .atlas_scene_measurements import estimate_sizes

    def check(stage):
        if checkpoint is not None:
            checkpoint(stage)

    root.mkdir(parents=True, exist_ok=True)
    transients = list(transients)
    if not objects and not transients:
        return {"status": "not_requested", "reason": "no_object_regions_proposed"}
    commit, blockers = release_blockers()
    if blockers:
        raise ValueError("atlas_automation_release_not_admitted:" + ",".join(blockers))
    prepared_path = root / "prepared_background.json"
    prepared_binding = {"source_video_digest": source_digest, "source_pose_digest": pose.get("digest"),
                        "inputs_digest": inputs.get("digest"),
                        "automation_digest": canonical_digest(automation),
                        "task_context_digest": canonical_digest(dict(context)),
                        "selected_frame_ids": selected_frame_ids,
                        "editor_first": editor_first,
                        **({"transients": transients} if transients else {}),
                        **({"selected_tracks_digest": canonical_digest(selected_tracks)} if selected_tracks else {})}
    retained = _prepared_background(prepared_path, prepared_binding)
    if retained is not None:
        check("automation_reuse_prepared_background")
        return retained
    if require_cached:
        raise ValueError("atlas_export_replay_prepared_background_missing")
    reused, prior_digest = reusable_background(prior_background_path, source_digest=source_digest,
        context=context, frames=frames, objects=objects)
    reused_ids = {row['frame_id'] for row in reused}
    edit_frames = [frame for frame in frames if frame['frame_id'] not in reused_ids]
    edit_ids = {frame['frame_id'] for frame in edit_frames}
    edit_objects = [{**obj, 'regions': [r for r in obj['regions'] if r['frame_id'] in edit_ids]}
                    for obj in objects if any(r['frame_id'] in edit_ids for r in obj['regions'])]
    # Complete deterministic geometry and cost checks before the first upload.
    object_evidence(frames=edit_frames, objects=edit_objects, inputs=inputs)
    if not editor_first and any(o["disposition"] == "remove" for o in objects):
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
        + max(50, len(frames)) * 0.0002 * (len(objects) + len(transients))
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
    check("automation_segmentation")
    selected = segment_objects(
        frames=edit_frames,
        objects=edit_objects,
        inputs=inputs,
        task_context=context,
        source_digest=source_digest,
        root=root / "sam",
        **({"transients": transients} if transients else {}),
        **({"selected_tracks": selected_tracks} if selected_tracks is not None else {}),
    ) if edit_frames else []
    found = [r for r in selected if r.get("task_effect") == "transient"]
    selected = [r for r in selected if r.get("task_effect") != "transient"]
    measurements = estimate_sizes(
        frames=frames,
        automation=automation,
        pose_digest=pose["digest"],
        tracks=selected,
        output=root / "object_measurements.json",
    ) if all("depth" in frame for frame in frames) and not any(r['track'].get('guide_only') for r in selected) else {
        "status": "unavailable", "reason": ("image_guides_are_not_measurement_masks"
            if any(r['track'].get('guide_only') for r in selected) else "pre_edit_observed_depth_not_requested"),
        "metric_measurement_proven": False, "physical_evidence": False,
    }
    removed = [r for r in selected if r["disposition"] == "remove"]
    if not removed and not reused and not any(o["runs"] for r in found for o in r["track"]["observations"]):
        return {
            "status": "not_requested",
            "measurements": measurements,
            "reason": "all_objects_kept",
        }
    prepared = prepare_masked_frames(
        frames=edit_frames,
        targets=[*selected, *found],
        margin=automation["mask_margin_fraction"],
        root=root / "originals",
    )
    prepared.extend({**row, 'image_path': row['original_image_path'], 'image_digest': row['original_image_digest']}
                    for row in reused)
    order = {frame['frame_id']: i for i, frame in enumerate(frames)}
    prepared.sort(key=lambda row: order[row['frame_id']])
    if not any(r["remaining_pixel_count"] for r in prepared) and not reused:
        raise ValueError("atlas_removal_masks_empty")
    prepared_by_id = {row['frame_id']: row for row in prepared}
    reused = [row for row in reused if set(prepared_by_id[row['frame_id']]['removed_task_object_ids'])
              <= set(row.get('removed_task_object_ids', []))]
    targets = [
        {
            **r,
            "semantic_label": r["label"],
            "task_effect": "manipulated" if r["disposition"] == "remove" else "static_contact",
        }
        for r in objects
    ] + [
        {key: r[key] for key in ("target_id", "label", "segmentation_prompt", "disposition", "task_effect")}
        | {"semantic_label": r["label"]}
        for r in found
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
            if r["frame_id"] == automation.get("anchor_frame_id") and (r["remaining_pixel_count"] or r['frame_id'] in reused_ids)
        ),
        None,
    )
    if preferred is not None:
        anchor_source = preferred
    check("automation_anchor")
    anchor = next((row for row in reused if row['frame_id'] == anchor_source['frame_id']), None)
    reused_anchor = anchor is not None
    if not reused_anchor:
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
            mask_guidance=True,
        )[0]
    check("automation_anchor_review")
    if reused_anchor:
        anchor_review = {'status': 'passed', 'basis': 'prior_reviewed_background', 'prior_digest': prior_digest}
    else:
        anchor_review = verify_completed_background(
            frames=[anchor],
            original_frames=[anchor_source],
            plan=plan,
            output_root=root / "anchor_review",
            task_context=context,
            source_video_digest=source_digest,
        )
        if anchor_review.get("basis") == "deterministic_view_geometry":
            # The editor reframed the anchor, which every other view would copy.
            # One re-edit names the camera to keep; the review still decides.
            from .website_image_repair_agent import CAMERA_CHANGED_INSTRUCTION

            check("automation_anchor_camera_reedit")
            anchor = complete_background_images(
                frames=[anchor_source], task_digest=task_digest, output_root=root / "anchor",
                admission={}, token="", targets=targets, task_context=context,
                source_video_digest=source_digest, mask_guidance=True,
                repair_instruction=automation["removal_instruction"] + " " + CAMERA_CHANGED_INSTRUCTION,
            )[0]
            anchor_review = verify_completed_background(
                frames=[anchor], original_frames=[anchor_source], plan=plan,
                output_root=root / "anchor_review", task_context=context, source_video_digest=source_digest)
        findings = anchor_review.get("review") or {}
        if (anchor_review["status"] != "passed" and findings.get("unrelated_objects_preserved") is False
                and findings.get("unrelated_object_loss_frame_ids")):
            # 2026-10-29 capture-4b60431e: the anchor edit removed the tray but also
            # erased a dishwasher across the room, and the whole job stopped. One
            # re-edit names what to keep; the independent review still decides.
            from .website_image_repair_agent import CAMERA_CHANGED_INSTRUCTION, unrelated_objects_kept_instruction

            check("automation_anchor_content_reedit")
            anchor = complete_background_images(
                frames=[anchor_source], task_digest=task_digest, output_root=root / "anchor",
                admission={}, token="", targets=targets, task_context=context,
                source_video_digest=source_digest, mask_guidance=True,
                repair_instruction=" ".join((automation["removal_instruction"], CAMERA_CHANGED_INSTRUCTION,
                                             unrelated_objects_kept_instruction(findings))),
            )[0]
            anchor_review = verify_completed_background(
                frames=[anchor], original_frames=[anchor_source], plan=plan,
                output_root=root / "anchor_review", task_context=context, source_video_digest=source_digest)
    if anchor_review["status"] != "passed":
        return {
            "status": "quality_rejected",
            "stage": "anchor",
            "review": anchor_review,
            "measurements": measurements,
            "retained_root": str(root),
        }
    if editor_first:
        return _reviewed_editor_background(
            prepared=prepared, anchor=anchor, plan=plan, automation=automation, root=root,
            context=context, source_digest=source_digest, measurements=measurements,
            selected_frame_ids=selected_frame_ids, check=check, prepared_binding=prepared_binding,
            reused_frames=reused, prior_digest=prior_digest)
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
        check("automation_masked")
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
        check("automation_masked_review")
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
                check("automation_fallback")
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
            check("automation_fallback_review")
            fallback_review = verify_completed_background(
                frames=fallback,
                original_frames=prepared,
                plan=plan,
                output_root=root / "fallback_review",
                task_context=context,
                source_video_digest=source_digest,
            )
            from .website_image_repair_agent import image_repair_enabled, repair_rejected_views

            if fallback_review["status"] != "passed" and image_repair_enabled():
                check("automation_targeted_repairs")
                fallback, fallback_review = repair_rejected_views(
                    selected=fallback,
                    object_removal_frames=[{
                        **row, "original_image_path": row["image_path"],
                        "original_image_digest": row["image_digest"],
                    } for row in prepared],
                    original_frames=prepared, plan=plan, failed_review=fallback_review,
                    output_root=root / "fallback_repairs", task_context=context,
                    source_video_digest=source_digest, max_repairs=len(prepared),
                    repair_reference_path=Path(anchor["image_path"]),
                )
            background = {
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
            if background["ready_for_reconstruction"]:
                _prepared_background(prepared_path, prepared_binding, background)
            return background
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
