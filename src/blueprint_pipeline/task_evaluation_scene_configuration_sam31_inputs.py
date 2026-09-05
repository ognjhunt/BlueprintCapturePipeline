"""Consume reviewed SAM masks and measured global cutouts without rerendering.

This is a byte-checked bridge into the existing ArtiFixer envelope. It cannot
accept a review, generate masks, select an AABB, or authorize a policy episode.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json
from .gaussian_splat_decode import (
    read_standard_3dgs_ply, verify_standard_3dgs_ply_subset_exact,
)
from .public_scene_calibrated_object_masks import (
    _decode_union, _frame_map, _verified_source_tracks,
)
from .public_scene_gaussian_excision_audit import (
    CONTRIBUTION_CLASS_ORDER, CONTRIBUTION_EVIDENCE_SCHEMA, FREEZE_SCHEMA,
)
from .public_scene_sam31_track_selection_review import (
    load_validated_sam31_track_selection_inputs, validate_sam31_track_selection_review,
)
from .public_scene_segment_contribution_cutout import (
    CUTOUT_SET_SCHEMA, SELECTION_RULE, SWEEP_KIND, _load_arrays,
)
from .public_scene_segment_mask_repair_preflight import _camera_rows
from .task_evaluation_scene_configuration_disclosure import resolve_scene_configuration_disclosure

MASK_SOURCE = "sam31_reviewed_calibrated_object_masks"
EVIDENCE_FIELDS = (
    "selection_inputs", "track_selection_review", "calibrated_mask_set",
    "segment_cutout_set", "standard_splat_conversion",
)


class Sam31ExactMaskInputsError(ValueError):
    """Exact source, selection, camera or cutout evidence failed its join."""


def _require(condition, reason):
    if not condition:
        raise Sam31ExactMaskInputsError("scene_configuration_sam31_" + reason)


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def _file(row, root=None):
    _require(isinstance(row, Mapping), "file_record_invalid")
    relative = row.get("relative_path")
    if relative is not None:
        _require(root is not None and isinstance(relative, str)
                 and not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                 "file_path_invalid")
        path = Path(root) / relative
    else:
        path = Path(str(row.get("path") or ""))
    _require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents))
             and path.is_file(), "file_path_invalid")
    _require(path.stat().st_size == row.get("size_bytes")
             and _sha(path) == row.get("sha256", row.get("digest")), "file_bytes_changed")
    return path


def _read(path, field):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict) and value.get(field)
             == canonical_digest(value, digest_field=field), "receipt_digest_invalid")
    return value


def _copy(path, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    return {"path": str(destination), "digest": _sha(destination),
            "size_bytes": destination.stat().st_size}


def _cutout(candidate_path, candidate, source, task, camera_ids, mask_paths):
    _require(candidate.get("schema_version") == CUTOUT_SET_SCHEMA
             and candidate.get("selection", {}).get("rule") == SELECTION_RULE
             and candidate.get("selection", {}).get("learned_policy_or_simulator_output_used") is False
             and candidate.get("claim_boundary", {}).get("canonical_source_altered") is False,
             "cutout_invalid")
    rows = candidate.get("task_candidates")
    _require(isinstance(rows, list) and len(rows) == 1, "task_count_invalid")
    row = rows[0]
    _require(row.get("task_id") == task["task_id"]
             and row.get("task_freeze_digest") == task["task_freeze_digest"], "task_join_invalid")
    sweep_path = _file(row["sweep_freeze"])
    sweep = _read(sweep_path, "freeze_digest")
    meta = sweep.get("segment_contribution_sweep", {})
    split = sweep.get("camera_split", {})
    _require(sweep.get("schema_version") == FREEZE_SCHEMA
             and sweep.get("status") == "frozen_before_excision_execution"
             and sweep.get("learned_policy_outcomes_observed") is False
             and sweep.get("replacement_usd_inserted") is False
             and meta.get("kind") == SWEEP_KIND
             and meta.get("all_frozen_cameras_included") is True
             and meta.get("selection_classes") == ["target_core", "uncertain"]
             and split.get("calibration_camera_ids") == camera_ids
             and split.get("heldout_camera_ids") == []
             and split.get("camera_count") == len(camera_ids), "sweep_invalid")
    _require(_file(sweep["source_standard_splat"]) == source
             and sweep["scene"]["target_instance_id"] == task["source_object"]["instance_id"]
             and sweep["scene"]["task_id"] == task["task_id"]
             and sweep["target_collision_prim_path"] == task["removal_plan"]["source_collider_prim_path"],
             "sweep_source_join_invalid")
    original_path = _file(meta["source_excision_freeze"])
    original = _read(original_path, "freeze_digest")
    _require(original["freeze_digest"] == meta["source_excision_freeze"]["freeze_digest"]
             and original.get("schema_version") == FREEZE_SCHEMA
             and original.get("status") == "frozen_before_excision_execution"
             and original.get("source_standard_splat") == sweep.get("source_standard_splat")
             and original.get("target_collision_prim_path") == sweep.get("target_collision_prim_path")
             and original.get("scene") == sweep.get("scene"),
             "excision_freeze_invalid")
    mask_rows = {item["camera_id"]: item for item in sweep["masks"]}
    _require(set(mask_rows) == set(camera_ids), "mask_camera_join_invalid")
    for camera_id in camera_ids:
        # Sweep keeps historical mask references anchored at the original freeze.
        record = mask_rows[camera_id]["historical_outer_mask"]
        mask = _file(record, original_path.parent)
        _require(_sha(mask) == _sha(mask_paths[camera_id]), "excision_mask_join_invalid")
        for zone in CONTRIBUTION_CLASS_ORDER:
            _file(mask_rows[camera_id]["zones"][zone], sweep_path.parent)
    manifest_path = _file(row["contribution_manifest"])
    manifest = _read(manifest_path, "manifest_digest")
    _require(manifest.get("schema_version") == CONTRIBUTION_EVIDENCE_SCHEMA
             and manifest.get("freeze_digest") == sweep["freeze_digest"]
             and manifest.get("class_order") == list(CONTRIBUTION_CLASS_ORDER)
             and manifest.get("camera_ids") == camera_ids
             and manifest.get("heldout_cameras_accessed_for_classification") is False
             and manifest.get("method", {}).get("released_code_executed") is True,
             "contribution_manifest_invalid")
    frozen_method = sweep.get("contribution_method")
    measured_method = manifest["method"]
    _require(isinstance(frozen_method, Mapping)
             and original.get("contribution_method") == frozen_method, "contribution_method_missing")
    for field in ("name", "repository", "commit", "rasterizer_repository", "rasterizer_commit",
                  "contribution_semantics", "source_modified", "depth_anything_3_used"):
        _require(field in frozen_method and measured_method.get(field) == frozen_method[field],
                 "contribution_method_mismatch")
    decimals = sweep.get("policy", {}).get("contribution_quantization_decimals")
    threshold = sweep.get("policy", {}).get("minimum_per_view_contribution")
    _require(type(decimals) is int and 3 <= decimals <= 12
             and type(threshold) in (float, int) and np.isfinite(threshold)
             and 0 < threshold <= 1, "contribution_policy_invalid")
    count = read_standard_3dgs_ply(source).count
    arrays = _load_arrays(manifest_path=manifest_path, manifest=manifest,
                         shape=(len(camera_ids), len(CONTRIBUTION_CLASS_ORDER), count),
                         decimals=decimals)
    selected = np.logical_or.reduce([
        np.any(array[:, 1, :] + array[:, 2, :] >= threshold, axis=0)
        for array in arrays
    ])
    deleted_indices, retained_indices = np.flatnonzero(selected), np.flatnonzero(~selected)
    _require(deleted_indices.size > 0 and retained_indices.size > 0, "cutout_empty")
    shared = candidate["shared_scene_union"]
    outputs = shared["outputs"]
    for key, expected in (("deleted_source_indices", deleted_indices),
                          ("retained_source_indices", retained_indices)):
        actual = np.load(_file(outputs[key], candidate_path.parent), allow_pickle=False)
        _require(actual.dtype.kind in "iu" and np.array_equal(actual, expected),
                 "cutout_indices_mismatch")
    deleted = _file(outputs["deleted_source_gaussians"], candidate_path.parent)
    retained = _file(outputs["retained_scene_gaussians"], candidate_path.parent)
    for path, indices in ((deleted, deleted_indices), (retained, retained_indices)):
        proof = verify_standard_3dgs_ply_subset_exact(source, path, indices)
        _require(proof.get("retained_rows_byte_exact") is True, "cutout_rows_changed")
    _require(shared["counts"] == {"source": count, "deleted_total": len(deleted_indices),
                                  "retained_total": len(retained_indices)}, "cutout_counts_mismatch")
    return deleted, retained, shared["counts"], sweep, original


def materialize_sam31_exact_mask_render_inputs(
    *, envelope, stage_one_configuration, output_root,
):
    config = stage_one_configuration
    bindings = config.get("sam31_exact_mask_evidence")
    evidence = envelope.get("sam31_exact_mask_inputs")
    _require(isinstance(evidence, Mapping) and set(evidence) == set(EVIDENCE_FIELDS),
             "evidence_missing")
    if "sam31_preparation_plan" in config:
        plan = config["sam31_preparation_plan"]
        prepared = envelope.get("sam31_preparation_result")
        _require(bindings is None and isinstance(prepared, Mapping)
                 and prepared.get("schema_version") == "task_evaluation_sam31_preparation_result.v1"
                 and prepared.get("status") == "exact_mask_inputs_ready"
                 and prepared.get("source_commit") == envelope["request"].get("expected_production_commit")
                 and prepared.get("plan_digest") == plan["digest"]
                 and prepared.get("evidence") == evidence
                 and prepared.get("result_digest") == canonical_digest(prepared, digest_field="result_digest"),
                 "preparation_result_invalid")
    else:
        _require(isinstance(bindings, Mapping), "evidence_missing")
    paths, receipts = {}, {}
    for key in EVIDENCE_FIELDS:
        paths[key] = _file(evidence[key])
        receipts[key] = _read(paths[key], "receipt_digest")
        if bindings is not None:
            _require(bindings.get(key + "_digest") == receipts[key]["receipt_digest"],
                     "configuration_evidence_mismatch")
    bindings = {key + "_digest": receipts[key]["receipt_digest"] for key in EVIDENCE_FIELDS}
    freezes, inputs, selected = load_validated_sam31_track_selection_inputs(paths["selection_inputs"])
    review = validate_sam31_track_selection_review(
        receipt_path=paths["track_selection_review"], task_freeze_paths=freezes,
        task_inputs=inputs, selected_track_ids_by_task=selected,
    )
    review_kind = config.get("sam31_review_kind")
    actual_kind = ("human" if review["schema_version"] == "public_scene_sam31_track_selection_review.v1"
                   else "ai")
    _require(review_kind in {"human", "ai"} and actual_kind == review_kind,
             "reviewer_kind_mismatch")
    if review_kind == "human":
        _require(bool(str(review.get("reviewed_by") or "").strip())
                 and bool(str(review.get("reviewed_on") or "").strip()), "human_review_required")
    _require(len(freezes) == 1, "task_count_invalid")
    task = _read(freezes[0], "task_freeze_digest")
    task_id = task["task_id"]
    source_object = config["source_object"]
    _require(task["source_object"]["instance_id"] == source_object["publisher_instance_id"]
             and task["removal_plan"]["source_collider_prim_path"] == source_object["collision_candidate_prim"],
             "task_source_identity_mismatch")
    if isinstance(task.get("scene_selection"), Mapping):
        scene_selection = _read(_file(task["scene_selection"]), "scene_freeze_digest")
        _require(scene_selection["selected_scene_id"] == source_object["scene_id"],
                 "task_scene_identity_mismatch")
        original_source = scene_selection["source_components"]["interiorgs"]
    else:
        original_source = None
    mask_set = receipts["calibrated_mask_set"]
    selection = mask_set.get("selection_authority", {})
    _require(mask_set.get("schema_version") == "public_scene_calibrated_object_mask_set.v1"
             and len(mask_set.get("tasks", [])) == 1
             and selection.get("review_receipt_digest") == review["receipt_digest"]
             and selection.get("reviewer_kind") == review_kind
             and selection.get("all_selected_tracks_" + ("human_review" if review_kind == "human" else "ai_visual_review") + "_accepted") is True
             and selection.get("mask_dilation_pixels") == 0, "mask_set_invalid")
    masks = mask_set["tasks"][0]
    _require(masks["task_id"] == task_id and masks["selected_track_ids"] == selected[task_id]
             and masks["source_object_instance_id"] == source_object["publisher_instance_id"],
             "mask_task_join_invalid")
    camera_file = _file(masks["camera_contract"], paths["calibrated_mask_set"].parent)
    _require(_sha(camera_file) == _sha(Path(inputs[task_id]["camera_contract_path"])),
             "camera_contract_changed")
    calibrations = _camera_rows(camera_file)
    camera_ids = sorted(calibrations)
    _require(config["required_views"]["minimum"] <= len(camera_ids) <= 16,
             "camera_count_invalid")
    _require(masks["camera_frame_map"] == inputs[task_id]["camera_frame_map"],
             "camera_frame_map_changed")
    tracks = _verified_source_tracks(Path(inputs[task_id]["source_track_result_path"]))
    frames = _frame_map(tracks)
    frame_rows = {row["camera_id"]: row for row in masks["source_images"]}
    mask_rows = {row["camera_id"]: row for row in masks["masks"]}
    _require(set(frame_rows) == set(mask_rows) == set(camera_ids), "frame_camera_join_invalid")
    frame_paths, mask_paths = {}, {}
    for camera_id in camera_ids:
        frame = frames[masks["camera_frame_map"][camera_id]]
        frame_paths[camera_id] = _file(frame_rows[camera_id]["image"], paths["calibrated_mask_set"].parent)
        mask_paths[camera_id] = _file(mask_rows[camera_id]["mask"], paths["calibrated_mask_set"].parent)
        _require(_sha(frame_paths[camera_id]) == frame["source_frame_digest"], "source_frame_changed")
        expected = _decode_union(frame, selected_track_ids=set(selected[task_id]),
                                 code="exact_mask_decode_invalid", allow_empty_selected_tracks=True)
        with Image.open(mask_paths[camera_id]) as image:
            actual = np.asarray(image.convert("L"))
        with Image.open(frame_paths[camera_id]) as image:
            image_size = image.size
        _require(np.array_equal(actual, expected) and image_size == (actual.shape[1], actual.shape[0]),
                 "exact_mask_changed")
    source_rows = [row for row in envelope["materialized_references"]
                   if row.get("contract_path") == "scene.appearance.representation"]
    _require(len(source_rows) == 1, "source_reference_missing")
    source_row = source_rows[0]
    _file({**source_row, "path": source_row["materialized_path"]})
    if original_source is not None:
        _require(original_source["sha256"] == source_row["digest"]
                 and original_source["size_bytes"] == source_row["size_bytes"],
                 "task_appearance_identity_mismatch")
    conversion = receipts["standard_splat_conversion"]
    _require(conversion.get("schema_version") == "standard_splat_conversion_receipt.v1"
             and conversion.get("status") == "standard_splat_conversion_materialized"
             and conversion.get("source", {}).get("sha256") == source_row["digest"]
             and conversion["source"].get("size_bytes") == source_row["size_bytes"]
             and conversion.get("raw_source_uploaded") is False
             and conversion["output"].get("standard_3dgs_schema_validated") is True
             and conversion["output"].get("gaussian_count_preserved") is True, "conversion_invalid")
    candidate = receipts["segment_cutout_set"]
    standard = _file(candidate["source_standard_splat"])
    _require(_sha(standard) == conversion["output"]["sha256"]
             and standard.stat().st_size == conversion["output"]["size_bytes"], "conversion_output_changed")
    deleted, retained, counts, sweep, original = _cutout(
        paths["segment_cutout_set"], candidate, standard, task, camera_ids, mask_paths)
    upstream_render = _file(original["render_input_packet"]["receipt"])
    upstream = _read(upstream_render, "receipt_digest")
    renderer = upstream.get("renderer", {})
    _require(upstream.get("schema_version") == "public_scene_interiorgs_edit_input_receipt.v2"
             and upstream.get("status") == "render_derived_input_packet_materialized"
             and renderer.get("authorization_class") == "method_input"
             and renderer.get("purpose_bound") is True
             and renderer.get("render_manifest_digests")
             and upstream["scene"]["target_instance_id"] == source_object["publisher_instance_id"]
             and upstream["scene"]["publisher_scene_id"] == source_object["scene_id"],
             "renderer_qualification_missing")
    _require(upstream["scene"].get("task_id") == task_id
             and upstream.get("source_admission", {}).get("standard_splat_conversion_receipt_digest")
             == conversion["receipt_digest"], "renderer_source_join_invalid")
    rendered = upstream.get("derived_artifacts", {})
    _require(rendered.get("cameras", {}).get("sha256") == _sha(camera_file),
             "renderer_camera_join_invalid")
    images = {row["camera_id"]: row for row in rendered.get("images", [])}
    _require(set(images) == set(camera_ids), "renderer_camera_join_invalid")
    for camera_id in camera_ids:
        _require(images[camera_id]["sha256"] == _sha(frame_paths[camera_id])
                 and images[camera_id]["size_bytes"] == frame_paths[camera_id].stat().st_size,
                 "renderer_frame_join_invalid")
    sealed_path = upstream_render.parent / "images" / "sealed_camera_render_manifest.v1.json"
    _require(sealed_path.is_file() and not any(p.is_symlink() for p in (sealed_path, *sealed_path.parents)),
             "sealed_render_manifest_missing")
    sealed = _read(sealed_path, "sealed_camera_render_manifest_digest")
    _require(sealed.get("schema_version") == "sealed_camera_render_manifest.v1"
             and sealed.get("status") == "rendered_exact_cameras"
             and sealed.get("authorization_class") == "method_input"
             and sealed.get("splat_digest") == _sha(standard)
             and sealed.get("sealed_camera_render_manifest_digest") == renderer["render_manifest_digests"]["images"]
             and sealed.get("calibrated_camera_file", {}).get("digest") == _sha(camera_file)
             and sealed.get("calibrated_camera_file", {}).get("binding") == "caller_file_exact_match"
             and sealed.get("render_count") == len(camera_ids), "sealed_render_join_invalid")
    render_rows = {row["camera_id"]: row for row in sealed.get("renders", [])}
    _require(set(render_rows) == set(camera_ids), "sealed_render_join_invalid")
    for camera_id in camera_ids:
        _require(render_rows[camera_id]["digest"] == _sha(frame_paths[camera_id]),
                 "sealed_render_frame_changed")
    rights_rows = [row for row in envelope["materialized_references"]
                   if row.get("contract_path") == "scene.rights.admission"]
    _require(len(rights_rows) == 1, "rights_missing")
    rights_path = _file({**rights_rows[0], "path": rights_rows[0]["materialized_path"]})
    rights = json.loads(rights_path.read_text())
    disclosure = resolve_scene_configuration_disclosure(
        stage_one_configuration=config, rights_admission=rights)
    _require(disclosure["render_execution_site"] == "control_plane"
             and disclosure["source_appearance_bytes_to_provider"] is False, "raw_upload_forbidden")
    root = Path(output_root)
    _require(root.is_absolute() and not any(p.is_symlink() for p in (root, *root.parents))
             and not root.exists(), "output_exists")
    root.mkdir(parents=True)
    calibration_path = root / "artifixer_method_input_cameras.v1.json"
    calibration_path.write_text(canonical_json([calibrations[key] for key in camera_ids]) + "\n")
    derived = []
    for index, camera_id in enumerate(camera_ids):
        frame = _copy(frame_paths[camera_id], root / "frames" / f"{index:04d}.png")
        mask = _copy(mask_paths[camera_id], root / "masks" / f"{index:04d}.png")
        with Image.open(mask_paths[camera_id]) as image:
            mask["foreground_pixel_count"] = int(np.count_nonzero(np.asarray(image)))
        derived.append({"camera_id": camera_id, **frame, "source_object_mask": mask})
    result = {
        "schema_version": "task_evaluation_scene_configuration_render_inputs.v1",
        "status": "derived_method_inputs_materialized",
        "run_id": envelope["request"]["run_id"],
        "publisher_instance_id": source_object["publisher_instance_id"],
        "source_splat_digest": source_row["digest"],
        "source_splat_bytes_retained_on_control_plane": True,
        "raw_interiorgs_bytes_in_provider_packet": False,
        "provider_disclosure_scope": "derived_rendered_views_only",
        "disclosure_decision": disclosure, "render_execution_site": "control_plane",
        "source_appearance": {"path": source_row["materialized_path"], "digest": source_row["digest"],
                              "size_bytes": source_row["size_bytes"]},
        "camera_calibration": {"path": str(calibration_path), "digest": _sha(calibration_path),
                               "size_bytes": calibration_path.stat().st_size},
        "render_manifest": {**_copy(sealed_path, root / "sealed_camera_render_manifest.v1.json"),
                            "manifest_digest": sealed["sealed_camera_render_manifest_digest"]},
        "derived_frames": derived, "derived_frame_count": len(derived),
        "source_object_masks": {"count": len(derived), "source": MASK_SOURCE,
            "source_object_identity": {"publisher_instance_id": source_object["publisher_instance_id"]},
            "observed_segmentation_truth": False, "all_masks_digest_bound": True,
            "selection_review_digest": review["receipt_digest"], "reviewer_kind": review_kind,
            "human_review_completed": review_kind == "human", "mask_dilation_pixels": 0},
        "derived_gaussian_cutout": {"selection_rule": SELECTION_RULE,
            "source_count": counts["source"], "removed_count": counts["deleted_total"],
            "retained_count": counts["retained_total"],
            "source_object_candidate": _copy(deleted, root / "source_object_candidate_gaussians.ply"),
            "retained_scene_without_source_object": _copy(retained, root / "retained_scene_gaussians_without_source_object.ply"),
            "retained_rows_byte_exact": True,
            "selection_is_candidate_not_observed_object_ownership_truth": True,
            "raw_source_bytes_in_provider_packet": False,
            "segment_cutout_set_digest": candidate["receipt_digest"]},
        "sam31_exact_mask_evidence": dict(bindings),
        "sam31_evidence_records": {
            **{key: _copy(path, root / "provenance" / (key + ".json")) for key, path in paths.items()},
            "source_render_receipt": _copy(upstream_render, root / "provenance/source_render_receipt.json"),
        },
        "browser_preview_used_as_method_input": False, "sage_render_used_as_appearance": False,
        "provider_mutation_performed": False, "paid_execution_requested": False,
        "renderer_runtime": renderer, "provider_render_required": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    (root / (result["schema_version"] + ".json")).write_text(canonical_json(result) + "\n")
    return result
