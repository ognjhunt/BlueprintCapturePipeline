#!/usr/bin/env python3
"""Execute one exact InteriorGS edit with a digest-bound Inpaint360GS adapter."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

from PIL import Image


SCHEMA_VERSION = "adp_inpaint360_interiorgs_result.v1"
METHOD_RESOLUTION_ARGUMENT = 2
MASK_ASSOCIATION_MODE = "pre_registered_single_target_resolution_divisor_2"
COMMAND_TIMEOUT_SECONDS = 10_800
MIN_VIRTUAL_MASK_FILL_RATIO = 0.2
MIN_VIRTUAL_MASK_REFERENCE_RATIO = 0.1
MIN_QUALIFYING_VIRTUAL_VIEW_COUNT = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepend_pythonpath(env: dict[str, str], root: Path) -> dict[str, str]:
    """Put the checked-out method ahead of similarly named installed packages."""

    updated = dict(env)
    existing = updated.get("PYTHONPATH", "")
    updated["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(root.resolve()), existing) if part
    )
    return updated


def _run(
    command: Sequence[str], *, cwd: Path, log_path: Path, env: dict[str, str] | None = None
) -> dict[str, Any]:
    started = dt.datetime.now(dt.timezone.utc)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("wb") as log_stream:
            completed = subprocess.run(
                list(command),
                cwd=cwd,
                env=env,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                timeout=COMMAND_TIMEOUT_SECONDS,
                check=False,
            )
        returncode, timed_out = completed.returncode, False
    except subprocess.TimeoutExpired:
        returncode, timed_out = 124, True
    finished = dt.datetime.now(dt.timezone.utc)
    return {
        "command": [str(item) for item in command],
        "cwd": str(cwd),
        "returncode": returncode,
        "timed_out": timed_out,
        "runtime_seconds": (finished - started).total_seconds(),
        "stdout_stderr_sha256": _sha256(log_path),
        "log": log_path.name,
    }


def _source_identity(source: Path, spec: dict[str, Any]) -> dict[str, Any]:
    changed: list[str] = []
    for record in spec["source"]["files"]:
        path = source / str(record["path"])
        if record.get("type") == "symlink":
            if not path.is_symlink() or os.readlink(path) != record.get("target"):
                changed.append(str(record["path"]))
            continue
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or _sha256(path) != record["sha256"]
        ):
            changed.append(str(record["path"]))
    return {"matches": not changed, "changed_files": changed[:100]}


def _packet_identity(packet: Path, spec: dict[str, Any]) -> dict[str, Any]:
    changed: list[str] = []
    for record in spec["adapter"]["files"]:
        path = packet / str(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or _sha256(path) != record["sha256"]
        ):
            changed.append(str(record["path"]))
    return {"matches": not changed, "changed_files": changed[:100]}


def _nested_dependency_identity(source: Path, spec: dict[str, Any]) -> dict[str, Any]:
    changed: list[str] = []
    dependency = spec["nested_dependencies"]["lama"]
    for record in dependency["files"]:
        path = source / "LaMa" / str(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or _sha256(path) != record["sha256"]
        ):
            changed.append(str(record["path"]))
    return {
        "repository": dependency["repository"],
        "commit": dependency["commit"],
        "tree": dependency["tree"],
        "matches": not changed,
        "changed_files": changed[:100],
    }


def _artifact(path: Path, output: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    destination = output / "artifacts" / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return {
        "relative_path": destination.relative_to(output).as_posix(),
        "size_bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def _ply_vertex_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        for raw in stream:
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[-1])
            if line == "end_header":
                break
    return None


def _freeze_supplemental_fusion_view(
    *, selection: dict[str, Any], output: Path
) -> dict[str, Any]:
    candidates = [dict(row) for row in selection.get("selected_views") or []]
    positive = [row for row in candidates if int(row.get("foreground_pixels") or 0) > 0]
    selected = sorted(
        positive,
        key=lambda row: (-int(row["foreground_pixels"]), str(row["view_id"])),
    )[0] if positive else None
    blockers = [] if selected else ["inpaint360_supplemental_fusion_positive_mask_missing"]
    receipt = {
        "schema_version": "inpaint360_supplemental_fusion_view_selection.v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "accepted" if not blockers else "blocked",
        "selection_timing": "before_lama_color_depth_inpainting",
        "selection_basis": "max_pre_inpainting_quality_qualified_mask_coverage_tiebreak_lowest_view_id",
        "publisher_default_view_id": "00004",
        "selected_view": selected,
        "candidate_count": len(candidates),
        "positive_candidate_count": len(positive),
        "blockers": blockers,
    }
    _write_json(output / "supplemental_fusion_view_selection.json", receipt)
    return receipt


def _freeze_nonempty_virtual_views(
    *, handoff: dict[str, Any], mask_binding: dict[str, Any], output: Path
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    candidate_count = 0
    empty_view_count = 0
    bbox_ineligible_view_count = 0
    low_fill_ratio_view_count = 0
    low_reference_coverage_view_count = 0
    source_counts = [
        int(value)
        for value in (mask_binding.get("associated_target_pixel_counts") or {}).values()
        if isinstance(value, int) and value > 0
    ]
    reference_min_pixels = min(source_counts) if source_counts else 0
    minimum_foreground_pixels = max(
        1, round(reference_min_pixels * MIN_VIRTUAL_MASK_REFERENCE_RATIO)
    )
    for row in handoff.get("output_masks") or []:
        relative = str(row.get("relative_path") or "")
        view_id = Path(relative).stem
        foreground_pixels = row.get("foreground_pixels")
        bbox_width = row.get("foreground_bbox_width")
        bbox_height = row.get("foreground_bbox_height")
        if len(view_id) != 5 or not view_id.isdigit() or not isinstance(foreground_pixels, int):
            continue
        candidate_count += 1
        if foreground_pixels == 0:
            empty_view_count += 1
        elif not (
            isinstance(bbox_width, int)
            and bbox_width >= 2
            and isinstance(bbox_height, int)
            and bbox_height >= 2
        ):
            bbox_ineligible_view_count += 1
        bbox_area = (
            int(bbox_width) * int(bbox_height)
            if isinstance(bbox_width, int)
            and bbox_width > 0
            and isinstance(bbox_height, int)
            and bbox_height > 0
            else 0
        )
        fill_ratio = float(foreground_pixels) / float(bbox_area) if bbox_area else 0.0
        bbox_eligible = (
            foreground_pixels > 0
            and isinstance(bbox_width, int)
            and bbox_width >= 2
            and isinstance(bbox_height, int)
            and bbox_height >= 2
        )
        if bbox_eligible and fill_ratio < MIN_VIRTUAL_MASK_FILL_RATIO:
            low_fill_ratio_view_count += 1
        if bbox_eligible and foreground_pixels < minimum_foreground_pixels:
            low_reference_coverage_view_count += 1
        if (
            bbox_eligible
            and fill_ratio >= MIN_VIRTUAL_MASK_FILL_RATIO
            and foreground_pixels >= minimum_foreground_pixels
        ):
            selected.append(
                {
                    "view_id": view_id,
                    "foreground_pixels": foreground_pixels,
                    "foreground_bbox_width": bbox_width,
                    "foreground_bbox_height": bbox_height,
                    "foreground_bbox_fill_ratio": fill_ratio,
                    "mask_sha256": row.get("sha256"),
                }
            )
    selected.sort(key=lambda row: str(row["view_id"]))
    blockers: list[str] = []
    if len(selected) < MIN_QUALIFYING_VIRTUAL_VIEW_COUNT:
        blockers.append("inpaint360_target_visible_virtual_view_support_inadequate")
    receipt = {
        "schema_version": "inpaint360_nonempty_virtual_view_selection.v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "accepted" if not blockers else "blocked",
        "selection_timing": "before_lama_color_depth_inpainting",
        "selection_basis": "pre_inpainting_binary_target_mask_quality_and_source_coverage",
        "candidate_count": candidate_count,
        "selected_count": len(selected),
        "selected_views": selected,
        "empty_view_count": empty_view_count,
        "bbox_ineligible_view_count": bbox_ineligible_view_count,
        "low_fill_ratio_view_count": low_fill_ratio_view_count,
        "low_reference_coverage_view_count": low_reference_coverage_view_count,
        "minimum_qualifying_view_count": MIN_QUALIFYING_VIRTUAL_VIEW_COUNT,
        "minimum_bbox_fill_ratio": MIN_VIRTUAL_MASK_FILL_RATIO,
        "minimum_foreground_pixels": minimum_foreground_pixels,
        "minimum_foreground_pixels_derivation": "10_percent_of_smallest_frozen_source_mask_at_method_resolution",
        "source_reference_min_foreground_pixels": reference_min_pixels,
        "excluded_view_count": candidate_count - len(selected),
        "mask_pixels_or_images_modified": False,
        "blockers": blockers,
    }
    _write_json(output / "nonempty_virtual_view_selection.json", receipt)
    return receipt


def _validate_lama_depth_numerics(*, log_path: Path, output: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    non_finite_tokens = re.findall(
        r"(?<![A-Za-z0-9_])(?:nan|[+-]?inf)(?![A-Za-z0-9_])",
        text,
        flags=re.IGNORECASE,
    )
    blockers: list[str] = []
    if not log_path.is_file() or not text:
        blockers.append("inpaint360_lama_depth_log_missing")
    if non_finite_tokens:
        blockers.append("inpaint360_lama_depth_non_finite")
    receipt = {
        "schema_version": "inpaint360_lama_depth_numerical_validation.v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "accepted" if not blockers else "blocked",
        "log": log_path.name,
        "log_sha256": _sha256(log_path) if log_path.is_file() else None,
        "non_finite_token_count": len(non_finite_tokens),
        "fusion_allowed": not blockers,
        "blockers": blockers,
    }
    _write_json(output / "lama_depth_numerical_validation.json", receipt)
    return receipt


def _materialize_nonempty_virtual_view_adapter(
    *, source: Path, runtime: Path, selection: dict[str, Any], output: Path
) -> dict[str, Any]:
    source_script = source / "edit_object_inpaint.py"
    adapted_script = runtime / "adapted_edit_object_inpaint.py"
    retained_script = output / "adapter_overlays/edit_object_inpaint.nonempty_views.py"
    blockers: list[str] = []
    selected_ids = [
        str(row.get("view_id") or "") for row in selection.get("selected_views") or []
    ]
    if selection.get("status") != "accepted" or not selected_ids:
        blockers.append("inpaint360_nonempty_virtual_view_selection_not_accepted")
    if not source_script.is_file():
        blockers.append("inpaint360_publisher_inpaint_script_missing")
        source_text = ""
    else:
        source_text = source_script.read_text(encoding="utf-8")
    anchor = "        virtual_pose_list.append(view_tmp)\n\n    # 2. inpaint selected object"
    if source_text.count(anchor) != 1:
        blockers.append("inpaint360_nonempty_virtual_view_adapter_anchor_changed")
    if not blockers:
        selected_literal = json.dumps(selected_ids, separators=(",", ":"))
        replacement = (
            "        virtual_pose_list.append(view_tmp)\n\n"
            f"    blueprint_nonempty_view_ids = set({selected_literal})\n"
            "    virtual_pose_list = [\n"
            "        view_tmp for view_tmp in virtual_pose_list\n"
            "        if view_tmp.image_name in blueprint_nonempty_view_ids\n"
            "    ]\n"
            f"    if len(virtual_pose_list) != {len(selected_ids)}:\n"
            "        raise ValueError(\"inpaint360_nonempty_virtual_view_binding_changed\")\n\n"
            "    # 2. inpaint selected object"
        )
        adapted_text = source_text.replace(anchor, replacement)
        adapted_script.parent.mkdir(parents=True, exist_ok=True)
        adapted_script.write_text(adapted_text, encoding="utf-8")
        retained_script.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(adapted_script, retained_script)
    receipt = {
        "schema_version": "inpaint360_nonempty_virtual_view_adapter.v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "accepted" if not blockers else "blocked",
        "publisher_source_relative_path": "edit_object_inpaint.py",
        "publisher_source_sha256": _sha256(source_script) if source_script.is_file() else None,
        "adapted_script_sha256": _sha256(adapted_script) if adapted_script.is_file() else None,
        "retained_script_relative_path": (
            retained_script.relative_to(output).as_posix() if retained_script.is_file() else None
        ),
        "selected_view_ids": selected_ids,
        "selection_receipt": "nonempty_virtual_view_selection.json",
        "behavioral_change": "exclude_virtual_views_with_empty_frozen_target_masks_from_finetune_sampling",
        "publisher_source_files_modified": False,
        "mask_pixels_or_images_modified": False,
        "unchanged_source_execution_claimed": False,
        "blockers": blockers,
    }
    _write_json(output / "nonempty_virtual_view_adapter.json", receipt)
    return receipt


def _materialize_obb_removal_adapter(
    *, source: Path, runtime: Path, spec: dict[str, Any], output: Path
) -> dict[str, Any]:
    source_script = source / "edit_object_removal.py"
    adapted_script = runtime / "adapted_edit_object_removal.py"
    retained_script = output / "adapter_overlays/edit_object_removal.obb.py"
    corners = spec.get("target_obb_corners_m")
    blockers: list[str] = []
    if spec.get("target_removal_volume_contract") != (
        "gaussian_center_inside_exact_publisher_obb"
    ):
        blockers.append("inpaint360_target_removal_volume_contract_invalid")
    if (
        not isinstance(corners, list)
        or len(corners) != 8
        or any(not isinstance(row, list) or len(row) != 3 for row in corners)
    ):
        blockers.append("inpaint360_target_obb_corners_invalid")
    if not source_script.is_file():
        blockers.append("inpaint360_publisher_removal_script_missing")
        source_text = ""
    else:
        source_text = source_script.read_text(encoding="utf-8")
    anchor = (
        "            mask3d_convex, object_radius = points_inside_convex_hull("
        "gaussians._xyz.detach(), mask3d, remove_outliers=True, outlier_factor=1.0)\n"
        "           \n"
        "            mask3d = torch.logical_or(mask3d,mask3d_convex)"
    )
    if source_text.count(anchor) != 1:
        blockers.append("inpaint360_obb_removal_adapter_anchor_changed")
    if not blockers:
        corners_literal = json.dumps(corners, separators=(",", ":"))
        replacement = (
            f"            blueprint_target_obb = np.asarray({corners_literal}, dtype=np.float64)\n"
            "            blueprint_inside_obb = Delaunay(blueprint_target_obb).find_simplex(\n"
            "                gaussians._xyz.detach().cpu().numpy()\n"
            "            ) >= 0\n"
            "            mask3d = torch.tensor(\n"
            "                blueprint_inside_obb, device=gaussians._xyz.device, dtype=torch.bool\n"
            "            )\n"
            "            if not torch.any(mask3d):\n"
            "                raise ValueError(\"inpaint360_exact_obb_selected_no_gaussians\")\n"
            "            object_radius = get_hull_size(\n"
            "                gaussians._xyz.detach()[mask3d].cpu().numpy()\n"
            "            )"
        )
        adapted_text = source_text.replace(anchor, replacement)
        adapted_script.parent.mkdir(parents=True, exist_ok=True)
        adapted_script.write_text(adapted_text, encoding="utf-8")
        retained_script.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(adapted_script, retained_script)
    receipt = {
        "schema_version": "inpaint360_obb_removal_adapter.v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "accepted" if not blockers else "blocked",
        "publisher_source_relative_path": "edit_object_removal.py",
        "publisher_source_sha256": _sha256(source_script) if source_script.is_file() else None,
        "adapted_script_sha256": _sha256(adapted_script) if adapted_script.is_file() else None,
        "retained_script_relative_path": (
            retained_script.relative_to(output).as_posix() if retained_script.is_file() else None
        ),
        "target_obb_corners_m": corners,
        "removal_volume_contract": spec.get("target_removal_volume_contract"),
        "behavioral_change": "replace_semantic_convex_hull_removal_with_exact_publisher_obb_center_membership",
        "publisher_source_files_modified": False,
        "unchanged_source_execution_claimed": False,
        "blockers": blockers,
    }
    _write_json(output / "obb_removal_adapter.json", receipt)
    return receipt


def _materialize_supplemental_fusion_view(
    *, model: Path, selection: dict[str, Any], output: Path
) -> dict[str, Any]:
    selected = selection.get("selected_view") or {}
    selected_view_id = str(selected.get("view_id") or "")
    fused = model / "virtual/ours_object_removal/iteration_2000/fused_mask_col_dep_ply"
    source = fused / f"{selected_view_id}.ply"
    target = fused / "00004.ply"
    source_vertex_count = _ply_vertex_count(source)
    blockers: list[str] = []
    if selection.get("status") != "accepted":
        blockers.append("inpaint360_supplemental_fusion_view_not_frozen")
    if source_vertex_count is None:
        blockers.append("inpaint360_supplemental_fusion_selected_ply_missing")
    elif source_vertex_count <= 0:
        blockers.append("inpaint360_supplemental_fusion_selected_ply_empty")
    publisher_default_before = {
        "vertex_count": _ply_vertex_count(target),
        "sha256": _sha256(target) if target.is_file() else None,
    }
    if not blockers and source != target:
        shutil.copy2(source, target)
    materialized_count = _ply_vertex_count(target)
    if not blockers and materialized_count != source_vertex_count:
        blockers.append("inpaint360_supplemental_fusion_materialization_changed")
    receipt = {
        "schema_version": "inpaint360_supplemental_fusion_view_materialization.v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "accepted" if not blockers else "blocked",
        "selected_view_id": selected_view_id or None,
        "selection_basis": selection.get("selection_basis"),
        "source_vertex_count": source_vertex_count,
        "source_sha256": _sha256(source) if source.is_file() else None,
        "publisher_default_view_id": "00004",
        "publisher_default_before": publisher_default_before,
        "materialized_vertex_count": materialized_count,
        "materialized_sha256": _sha256(target) if target.is_file() else None,
        "publisher_source_files_modified": False,
        "blockers": blockers,
    }
    _write_json(output / "supplemental_fusion_view_materialization.json", receipt)
    return receipt


def _materialize_pre_registered_mask_binding(
    *, source_data: Path, target_method_instance_id: int, output: Path
) -> dict[str, Any]:
    """Bind one preregistered target across views without heuristic ID discovery."""

    raw_masks = sorted((source_data / "raw_hqsam").glob("*.png"))
    source_images = sorted((source_data / "images").glob("*.png"))
    source_images_by_name = {path.name: path for path in source_images}
    associated_dir = source_data / "associated_hqsam"
    existing_masks = sorted(associated_dir.glob("*.png"))
    target_pixel_counts: dict[str, int] = {}
    associated_target_pixel_counts: dict[str, int] = {}
    image_mask_dimensions: dict[str, dict[str, list[int]]] = {}
    source_image_sha256: dict[str, str] = {}
    raw_mask_sha256: dict[str, str] = {}
    associated_mask_sha256: dict[str, str] = {}
    derived_masks: dict[str, Image.Image] = {}
    invalid_masks: list[str] = []
    missing_source_images: list[str] = []
    for path in raw_masks:
        with Image.open(path) as image:
            grayscale = image.convert("L")
            histogram = grayscale.histogram()
            mask_size = list(grayscale.size)
        source_image = source_images_by_name.get(path.name)
        if source_image is None:
            missing_source_images.append(path.name)
            invalid_masks.append(path.name)
            source_size: list[int] = []
        else:
            with Image.open(source_image) as image:
                source_size = list(image.size)
            source_image_sha256[path.name] = _sha256(source_image)
            if source_size != mask_size:
                invalid_masks.append(path.name)
            width, height = source_size
            method_size = [
                (width + METHOD_RESOLUTION_ARGUMENT - 1) // METHOD_RESOLUTION_ARGUMENT,
                (height + METHOD_RESOLUTION_ARGUMENT - 1) // METHOD_RESOLUTION_ARGUMENT,
            ]
            derived = grayscale.resize(tuple(method_size), resample=Image.Resampling.NEAREST)
            derived_histogram = derived.histogram()
            derived_values = {
                index for index, count in enumerate(derived_histogram) if count
            }
            derived_target_count = int(derived_histogram[target_method_instance_id])
            associated_target_pixel_counts[path.name] = derived_target_count
            if (
                derived_values - {0, target_method_instance_id}
                or derived_target_count <= 0
            ):
                invalid_masks.append(path.name)
            derived_masks[path.name] = derived
        image_mask_dimensions[path.name] = {
            "source_image": source_size,
            "raw_mask": mask_size,
            "method_image_and_associated_mask": method_size if source_image else [],
        }
        populated_values = {index for index, count in enumerate(histogram) if count}
        target_count = (
            int(histogram[target_method_instance_id])
            if target_method_instance_id < len(histogram)
            else 0
        )
        target_pixel_counts[path.name] = target_count
        raw_mask_sha256[path.name] = _sha256(path)
        if populated_values - {0, target_method_instance_id} or target_count <= 0:
            invalid_masks.append(path.name)

    raw_mask_names = {path.name for path in raw_masks}
    unpaired_source_images = sorted(set(source_images_by_name) - raw_mask_names)
    invalid_masks = sorted(set(invalid_masks))
    valid = bool(raw_masks) and not existing_masks and not invalid_masks
    valid = valid and not unpaired_source_images and len(source_images) == len(raw_masks)
    if valid:
        associated_dir.mkdir(parents=True, exist_ok=True)
        for path in raw_masks:
            associated_path = associated_dir / path.name
            derived_masks[path.name].save(associated_path, format="PNG")
            associated_mask_sha256[path.name] = _sha256(associated_path)
            with Image.open(associated_path) as image:
                valid = valid and list(image.size) == image_mask_dimensions[path.name][
                    "method_image_and_associated_mask"
                ]
        _write_json(
            associated_dir / "scene.json",
            {
                "association_mode": MASK_ASSOCIATION_MODE,
                "num_classes": target_method_instance_id + 1,
                "raw_mask_folder": str(source_data / "raw_hqsam"),
                "associated_mask_folder": str(associated_dir),
                "target_method_instance_id": target_method_instance_id,
                "method_resolution_argument": METHOD_RESOLUTION_ARGUMENT,
                "categorical_resize_filter": "nearest",
            },
        )

    receipt = {
        "schema_version": "adp_inpaint360_pre_registered_mask_binding.v1",
        "status": "accepted" if valid else "blocked",
        "association_mode": MASK_ASSOCIATION_MODE,
        "target_method_instance_id": target_method_instance_id,
        "raw_mask_count": len(raw_masks),
        "source_image_count": len(source_images),
        "existing_associated_mask_count": len(existing_masks),
        "full_resolution_source_preserved": valid,
        "method_resolution_argument": METHOD_RESOLUTION_ARGUMENT,
        "categorical_resize_filter": "nearest",
        "image_mask_dimensions": image_mask_dimensions,
        "target_pixel_counts": target_pixel_counts,
        "associated_target_pixel_counts": associated_target_pixel_counts,
        "source_image_sha256": source_image_sha256,
        "raw_mask_sha256": raw_mask_sha256,
        "associated_mask_sha256": associated_mask_sha256,
        "invalid_masks": invalid_masks,
        "missing_source_images": sorted(missing_source_images),
        "unpaired_source_images": unpaired_source_images,
        "blockers": [] if valid else ["inpaint360_pre_registered_mask_binding_invalid"],
    }
    _write_json(output / "pre_registered_mask_binding.json", receipt)
    return receipt


def _validate_method_resolution_commands(
    commands: list[tuple[str, list[str], Path, dict[str, str]]], *, output: Path
) -> dict[str, Any]:
    """Bind model stages to 1.6K native loading and LaMa handoff to its folder contract."""

    required_stage_resolutions = {
        "distillation": str(METHOD_RESOLUTION_ARGUMENT),
        "baseline_render": str(METHOD_RESOLUTION_ARGUMENT),
        "removal": str(METHOD_RESOLUTION_ARGUMENT),
        "virtual_views": str(METHOD_RESOLUTION_ARGUMENT),
        "lama_prepare": "1",
        "lama_collect": "1",
        "ply_fusion": str(METHOD_RESOLUTION_ARGUMENT),
        "inpaint_3d": str(METHOD_RESOLUTION_ARGUMENT),
    }
    observed: dict[str, str | None] = {}
    violations: list[str] = []
    for stage, command, _cwd, _env in commands:
        if stage not in required_stage_resolutions:
            continue
        positions = [index for index, value in enumerate(command) if value == "--resolution"]
        value = (
            command[positions[0] + 1]
            if len(positions) == 1 and positions[0] + 1 < len(command)
            else None
        )
        observed[stage] = value
        if len(positions) != 1 or value != required_stage_resolutions[stage]:
            violations.append(stage)
    missing_stages = sorted(set(required_stage_resolutions) - set(observed))
    violations.extend(missing_stages)
    valid = not violations
    receipt = {
        "schema_version": "adp_inpaint360_method_resolution_command_contract.v1",
        "status": "accepted" if valid else "blocked",
        "method_resolution_argument": METHOD_RESOLUTION_ARGUMENT,
        "required_stage_resolutions": required_stage_resolutions,
        "observed_stage_resolutions": observed,
        "violating_or_missing_stages": sorted(set(violations)),
        "blockers": (
            [] if valid else ["inpaint360_method_resolution_command_contract_invalid"]
        ),
    }
    _write_json(output / "method_resolution_command_contract.json", receipt)
    return receipt


def _validate_baseline_depth_inventory(
    *, source_data: Path, model: Path, iteration: int, output: Path
) -> dict[str, Any]:
    """Require the author baseline render depths before object removal."""

    expected_names = sorted(path.stem for path in (source_data / "images").glob("*.png"))
    observed: dict[str, list[str]] = {}
    empty_files: list[str] = []
    for split in ("train", "test"):
        depth_dir = model / split / f"ours_{iteration}" / "depth"
        for path in sorted(depth_dir.glob("*.npy")):
            observed.setdefault(path.stem, []).append(split)
            if path.stat().st_size == 0:
                empty_files.append(path.relative_to(model).as_posix())
    observed_names = sorted(observed)
    missing_names = sorted(set(expected_names) - set(observed_names))
    unexpected_names = sorted(set(observed_names) - set(expected_names))
    duplicate_names = sorted(name for name, splits in observed.items() if len(splits) != 1)
    valid = bool(expected_names) and not (
        missing_names or unexpected_names or duplicate_names or empty_files
    )
    receipt = {
        "schema_version": "adp_inpaint360_baseline_depth_inventory.v1",
        "status": "accepted" if valid else "blocked",
        "iteration": iteration,
        "expected_camera_names": expected_names,
        "observed_camera_splits": observed,
        "missing_camera_names": missing_names,
        "unexpected_camera_names": unexpected_names,
        "duplicate_camera_names": duplicate_names,
        "empty_files": empty_files,
        "blockers": [] if valid else ["inpaint360_baseline_depth_inventory_invalid"],
    }
    _write_json(output / "baseline_depth_inventory.json", receipt)
    return receipt


def _retain_review_frames(
    render_dir: Path, output: Path, *, count: int = 8
) -> list[dict[str, Any]]:
    frames = sorted(render_dir.glob("[0-9][0-9][0-9][0-9][0-9].png"))
    if not frames:
        return []
    selected_indices = sorted(
        {round(index * (len(frames) - 1) / max(1, count - 1)) for index in range(count)}
    )
    retained: list[dict[str, Any]] = []
    review_dir = output / "review_frames"
    review_dir.mkdir(parents=True, exist_ok=True)
    for source_index in selected_indices:
        source = frames[source_index]
        with Image.open(source) as image:
            comparison = image.convert("RGB")
            comparison_size = comparison.size
            rgb = comparison.crop((0, 0, comparison.width // 2, comparison.height))
            rgb_name = f"frame_{source.stem}_rgb.png"
            comparison_name = f"frame_{source.stem}_rgb_and_mask.png"
            rgb_path = review_dir / rgb_name
            comparison_path = review_dir / comparison_name
            rgb.save(rgb_path, format="PNG")
            comparison.save(comparison_path, format="PNG")
        retained.append(
            {
                "source_frame": source.name,
                "source_frame_index": source_index,
                "rgb": {
                    "relative_path": rgb_path.relative_to(output).as_posix(),
                    "width": rgb.width,
                    "height": rgb.height,
                    "size_bytes": rgb_path.stat().st_size,
                    "sha256": _sha256(rgb_path),
                },
                "rgb_and_mask": {
                    "relative_path": comparison_path.relative_to(output).as_posix(),
                    "width": comparison_size[0],
                    "height": comparison_size[1],
                    "size_bytes": comparison_path.stat().st_size,
                    "sha256": _sha256(comparison_path),
                },
            }
        )
    return retained


def main() -> int:
    runtime = Path(__file__).resolve().parent
    source = runtime / "Inpaint360GS"
    packet = runtime / "interiorgs_adapter"
    output = Path(
        os.environ.get("BLUEPRINT_ADP_INPAINT360_OUTPUT_DIR", runtime.parent / "runtime_output")
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = _read_json(runtime / "execution_spec.json")
    main_python = str(source / ".venv/bin/python")
    lama_python = str(source / "LaMa/.venv/bin/python")
    source_before = _source_identity(source, spec)
    dependency_before = _nested_dependency_identity(source, spec)
    packet_before = _packet_identity(packet, spec)
    hardware = _run(["nvidia-smi", "-q"], cwd=source, log_path=output / "nvidia-smi.log")
    main_env = _prepend_pythonpath(dict(os.environ), source)
    main_env.update({"CUDA_HOME": "/usr/local/cuda", "PYTHONUNBUFFERED": "1"})
    lama_env = dict(main_env)
    lama_env["TORCH_HOME"] = str(source / "LaMa")
    lama_env["PYTHONPATH"] = str(source / "LaMa")
    config_removal = packet / f"config/object_removal/blueprint/{spec['scene_id']}.json"
    config_inpaint = packet / f"config/object_inpaint/blueprint/{spec['scene_id']}.json"
    source_data = packet / "source"
    model = packet / "inpaint360_model"
    vanilla = packet / "vanilla_3dgs"
    commands: list[tuple[str, list[str], Path, dict[str, str]]] = [
        (
            "camera_rasterizer_preflight",
            [
                main_python,
                str(runtime / "probe_inpaint360_camera_rasterizer.py"),
                "--source_path",
                str(source_data),
                "--model_path",
                str(model),
                "--vanilla_3dgs_path",
                str(vanilla),
                "--images",
                "images",
                "--object_path",
                "associated_hqsam",
                "--train_distill",
                "--eval",
                "--resolution",
                str(METHOD_RESOLUTION_ARGUMENT),
                "--expected-camera-count",
                str(spec["runtime"]["expected_input_camera_count"]),
                "--receipt",
                str(output / "camera_rasterizer_preflight.json"),
            ],
            source,
            main_env,
        ),
        (
            "distillation",
            [
                main_python,
                "seg/distillation.py",
                "--source_path",
                str(source_data),
                "--model_path",
                str(model),
                "--vanilla_3dgs_path",
                str(vanilla),
                "--images",
                "images",
                "--object_path",
                "associated_hqsam",
                "--eval",
                "--config_file",
                str(packet / "config/distill.json"),
                "--resolution",
                str(METHOD_RESOLUTION_ARGUMENT),
                "--save_iterations",
                "2000",
            ],
            source,
            main_env,
        ),
        (
            "baseline_render",
            [
                main_python,
                "render.py",
                "--source_path",
                str(source_data),
                "--model_path",
                str(model),
                "--images",
                "images",
                "--eval",
                "--resolution",
                str(METHOD_RESOLUTION_ARGUMENT),
            ],
            source,
            main_env,
        ),
        (
            "removal",
            [
                main_python,
                "edit_object_removal.py",
                "--source_path",
                str(source_data),
                "--model_path",
                str(model),
                "--images",
                "images",
                "--iteration",
                "2000",
                "--config_file",
                str(config_removal),
                "--resolution",
                str(METHOD_RESOLUTION_ARGUMENT),
                "--skip_test",
            ],
            source,
            main_env,
        ),
        (
            "virtual_views",
            [
                main_python,
                "tools/virtual_pose.py",
                "--source_path",
                str(source_data),
                "--model_path",
                str(model),
                "--images",
                "images",
                "--iteration",
                "2000",
                "--config_file",
                str(config_removal),
                "--resolution",
                str(METHOD_RESOLUTION_ARGUMENT),
            ],
            source,
            main_env,
        ),
        (
            "virtual_masks",
            [
                main_python,
                str(runtime / "materialize_inpaint360_virtual_masks.py"),
                "--runtime-root",
                str(runtime),
                "--evidence-root",
                str(output),
                "--predicted-mask-dir",
                str(model / "virtual/ours_2000/objects_pred"),
                "--output-dir",
                str(source / "Segment-and-Track-Anything/tracking_results/images/images_masks"),
                "--receipt",
                str(output / "virtual_mask_handoff.json"),
                "--target-instance-id",
                str(spec["target_method_instance_id"]),
                "--expected-count",
                str(spec["runtime"]["virtual_view_count"]),
            ],
            source,
            main_env,
        ),
        (
            "lama_prepare",
            [
                main_python,
                "tools/prepare_lama_data.py",
                "--source_path",
                str(source_data),
                "--model_path",
                str(model),
                "--resolution",
                "1",
                "--inpaint2lama",
            ],
            source,
            main_env,
        ),
        (
            "lama_color",
            [lama_python, "bin/predict_color.py", "--data_name", "360_source_virtual"],
            source / "LaMa",
            lama_env,
        ),
        (
            "lama_depth",
            [lama_python, "bin/predict_depth.py", "--data_name", "360_source_virtual"],
            source / "LaMa",
            lama_env,
        ),
        (
            "lama_collect",
            [
                main_python,
                "tools/prepare_lama_data.py",
                "--source_path",
                str(source_data),
                "--model_path",
                str(model),
                "--resolution",
                "1",
            ],
            source,
            main_env,
        ),
        (
            "ply_fusion",
            [
                main_python,
                "edit_object_removal_plyfusion.py",
                "--source_path",
                str(source_data),
                "--model_path",
                str(model),
                "--config_file",
                str(config_removal),
                "--resolution",
                str(METHOD_RESOLUTION_ARGUMENT),
            ],
            source,
            main_env,
        ),
        (
            "inpaint_3d",
            [
                main_python,
                "edit_object_inpaint.py",
                "--source_path",
                str(source_data),
                "--model_path",
                str(model),
                "--config_file",
                str(config_inpaint),
                "--resolution",
                str(METHOD_RESOLUTION_ARGUMENT),
                "--render_video",
            ],
            source,
            main_env,
        ),
    ]
    workflow: list[dict[str, Any]] = []
    method_resolution_validation = _validate_method_resolution_commands(
        commands, output=output
    )
    resolution_accepted = method_resolution_validation["status"] == "accepted"
    workflow.append(
        {
            "stage": "method_resolution_contract",
            "operation": "bind_author_supported_divisor_2_and_lama_folder_resolution",
            "cwd": str(packet),
            "returncode": 0 if resolution_accepted else 44,
            "timed_out": False,
            "receipt": "method_resolution_command_contract.json",
        }
    )
    mask_association_validation = _materialize_pre_registered_mask_binding(
        source_data=source_data,
        target_method_instance_id=int(spec["target_method_instance_id"]),
        output=output,
    )
    binding_accepted = mask_association_validation["status"] == "accepted"
    workflow.append(
        {
            "stage": "pre_registered_mask_binding",
            "operation": "copy_digest_bound_binary_target_masks",
            "cwd": str(packet),
            "returncode": 0 if binding_accepted else 43,
            "timed_out": False,
            "receipt": "pre_registered_mask_binding.json",
        }
    )
    depth_inventory: dict[str, Any] = {
        "schema_version": "adp_inpaint360_baseline_depth_inventory.v1",
        "status": "not_executed",
        "blockers": ["inpaint360_baseline_depth_inventory_not_executed"],
    }
    supplemental_selection: dict[str, Any] = {
        "status": "not_executed",
        "blockers": ["inpaint360_supplemental_fusion_view_selection_not_executed"],
    }
    supplemental_materialization: dict[str, Any] = {
        "status": "not_executed",
        "blockers": ["inpaint360_supplemental_fusion_view_materialization_not_executed"],
    }
    nonempty_view_selection: dict[str, Any] = {
        "status": "not_executed",
        "blockers": ["inpaint360_nonempty_virtual_view_selection_not_executed"],
    }
    nonempty_view_adapter: dict[str, Any] = {
        "status": "not_executed",
        "blockers": ["inpaint360_nonempty_virtual_view_adapter_not_executed"],
    }
    obb_removal_adapter: dict[str, Any] = {
        "status": "not_executed",
        "blockers": ["inpaint360_obb_removal_adapter_not_executed"],
    }
    lama_depth_validation: dict[str, Any] = {
        "status": "not_executed",
        "blockers": ["inpaint360_lama_depth_numerical_validation_not_executed"],
    }
    if resolution_accepted and binding_accepted:
        for stage, command, cwd, env in commands:
            print(f"BLUEPRINT_ADP_INPAINT360_STAGE_STARTED:{stage}", flush=True)
            if stage == "removal":
                obb_removal_adapter = _materialize_obb_removal_adapter(
                    source=source,
                    runtime=runtime,
                    spec=spec,
                    output=output,
                )
                obb_adapter_accepted = obb_removal_adapter["status"] == "accepted"
                workflow.append(
                    {
                        "stage": "obb_removal_adapter",
                        "operation": "bind_exact_publisher_obb_to_released_removal_runtime",
                        "cwd": str(runtime),
                        "returncode": 0 if obb_adapter_accepted else 51,
                        "timed_out": False,
                        "receipt": "obb_removal_adapter.json",
                    }
                )
                if not obb_adapter_accepted:
                    break
                command = list(command)
                command[1] = str(runtime / "adapted_edit_object_removal.py")
            if stage == "inpaint_3d":
                nonempty_view_adapter = _materialize_nonempty_virtual_view_adapter(
                    source=source,
                    runtime=runtime,
                    selection=nonempty_view_selection,
                    output=output,
                )
                adapter_accepted = nonempty_view_adapter["status"] == "accepted"
                workflow.append(
                    {
                        "stage": "nonempty_virtual_view_adapter",
                        "operation": "materialize_digest_bound_released_source_input_filter",
                        "cwd": str(runtime),
                        "returncode": 0 if adapter_accepted else 48,
                        "timed_out": False,
                        "receipt": "nonempty_virtual_view_adapter.json",
                    }
                )
                if not adapter_accepted:
                    break
                command = list(command)
                command[1] = str(runtime / "adapted_edit_object_inpaint.py")
            observed = _run(command, cwd=cwd, env=env, log_path=output / f"{stage}.log")
            observed["stage"] = stage
            workflow.append(observed)
            print(
                f"BLUEPRINT_ADP_INPAINT360_STAGE_FINISHED:{stage}:returncode={observed['returncode']}",
                flush=True,
            )
            if observed["returncode"] != 0:
                break
            if stage == "virtual_masks":
                nonempty_view_selection = _freeze_nonempty_virtual_views(
                    handoff=_read_json(output / "virtual_mask_handoff.json"),
                    mask_binding=mask_association_validation,
                    output=output,
                )
                nonempty_selection_accepted = (
                    nonempty_view_selection["status"] == "accepted"
                )
                workflow.append(
                    {
                        "stage": "nonempty_virtual_view_selection",
                        "operation": "freeze_pre_inpainting_positive_mask_view_set",
                        "cwd": str(model),
                        "returncode": 0 if nonempty_selection_accepted else 49,
                        "timed_out": False,
                        "receipt": "nonempty_virtual_view_selection.json",
                    }
                )
                if not nonempty_selection_accepted:
                    break
                supplemental_selection = _freeze_supplemental_fusion_view(
                    selection=nonempty_view_selection,
                    output=output,
                )
                selection_accepted = supplemental_selection["status"] == "accepted"
                workflow.append(
                    {
                        "stage": "supplemental_fusion_view_selection",
                        "operation": "freeze_pre_inpainting_max_mask_coverage_view",
                        "cwd": str(model),
                        "returncode": 0 if selection_accepted else 46,
                        "timed_out": False,
                        "receipt": "supplemental_fusion_view_selection.json",
                    }
                )
                if not selection_accepted:
                    break
            if stage == "lama_depth":
                lama_depth_validation = _validate_lama_depth_numerics(
                    log_path=output / "lama_depth.log",
                    output=output,
                )
                numerics_accepted = lama_depth_validation["status"] == "accepted"
                workflow.append(
                    {
                        "stage": "lama_depth_numerical_validation",
                        "operation": "reject_non_finite_depth_before_point_fusion",
                        "cwd": str(output),
                        "returncode": 0 if numerics_accepted else 50,
                        "timed_out": False,
                        "receipt": "lama_depth_numerical_validation.json",
                    }
                )
                if not numerics_accepted:
                    break
            if stage == "baseline_render":
                depth_inventory = _validate_baseline_depth_inventory(
                    source_data=source_data,
                    model=model,
                    iteration=2000,
                    output=output,
                )
                inventory_accepted = depth_inventory["status"] == "accepted"
                workflow.append(
                    {
                        "stage": "baseline_depth_inventory",
                        "operation": "bind_author_baseline_depths_to_frozen_cameras",
                        "cwd": str(model),
                        "returncode": 0 if inventory_accepted else 45,
                        "timed_out": False,
                        "receipt": "baseline_depth_inventory.json",
                    }
                )
                if not inventory_accepted:
                    break
            if stage == "ply_fusion":
                supplemental_materialization = _materialize_supplemental_fusion_view(
                    model=model,
                    selection=supplemental_selection,
                    output=output,
                )
                materialization_accepted = supplemental_materialization["status"] == "accepted"
                workflow.append(
                    {
                        "stage": "supplemental_fusion_view_materialization",
                        "operation": "bind_selected_fused_ply_to_publisher_fixed_input_slot",
                        "cwd": str(model),
                        "returncode": 0 if materialization_accepted else 47,
                        "timed_out": False,
                        "receipt": "supplemental_fusion_view_materialization.json",
                    }
                )
                if not materialization_accepted:
                    break
    completed = {row["stage"] for row in workflow if row["returncode"] == 0}
    final_ply = model / "point_cloud_object_inpaint_virtual/iteration_5000/point_cloud.ply"
    final_video = model / "video/ours__object_inpaint_virtual/iteration_5000/final_video.mp4"
    final_render_dir = final_video.parent
    source_after = _source_identity(source, spec)
    dependency_after = _nested_dependency_identity(source, spec)
    blockers: list[str] = []
    required = ["method_resolution_contract", "pre_registered_mask_binding"]
    for stage, _, _, _ in commands:
        if stage == "inpaint_3d":
            required.append("nonempty_virtual_view_adapter")
        required.append(stage)
        if stage == "removal":
            required.append("obb_removal_adapter")
        if stage == "lama_depth":
            required.append("lama_depth_numerical_validation")
        if stage == "baseline_render":
            required.append("baseline_depth_inventory")
        if stage == "virtual_masks":
            required.append("nonempty_virtual_view_selection")
            required.append("supplemental_fusion_view_selection")
        if stage == "ply_fusion":
            required.append("supplemental_fusion_view_materialization")
    for stage in required:
        if stage not in completed:
            blockers.append(f"inpaint360_{stage}_failed_or_not_executed")
            break
    if hardware["returncode"] != 0:
        blockers.append("inpaint360_nvidia_hardware_probe_failed")
    if not source_before["matches"] or not source_after["matches"]:
        blockers.append("inpaint360_author_source_modified")
    if not dependency_before["matches"] or not dependency_after["matches"]:
        blockers.append("inpaint360_lama_dependency_changed")
    if not packet_before["matches"]:
        blockers.append("inpaint360_adapter_input_changed_before_execution")
    if not final_ply.is_file() or final_ply.stat().st_size == 0:
        blockers.append("inpaint360_final_point_cloud_missing")
    review_frames = _retain_review_frames(final_render_dir, output)
    if "inpaint_3d" in completed and len(review_frames) != 8:
        blockers.append("inpaint360_review_frames_missing")
    main_freeze = output / "main-pip-freeze.txt"
    lama_freeze = output / "lama-pip-freeze.txt"
    vgg16_materialization = _read_json(output / "vgg16_materialization.json")
    if vgg16_materialization.get("status") != "accepted":
        blockers.append("inpaint360_vgg16_materialization_receipt_missing_or_blocked")
    for freeze, blocker in (
        (main_freeze, "inpaint360_main_environment_receipt_missing"),
        (lama_freeze, "inpaint360_lama_environment_receipt_missing"),
    ):
        if not freeze.is_file() or freeze.stat().st_size == 0:
            blockers.append(blocker)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "completed" if not blockers else "blocked",
        "scene_id": spec["scene_id"],
        "target_instance_id": spec["target_instance_id"],
        "target_method_instance_id": spec["target_method_instance_id"],
        "source_commit": spec["source"]["commit"],
        "source_tree": spec["source"]["tree"],
        "source_identity_before": source_before,
        "source_identity_after": source_after,
        "source_modified": not source_after["matches"],
        "nested_dependency_identity_before": dependency_before,
        "nested_dependency_identity_after": dependency_after,
        "vgg16_materialization": vgg16_materialization,
        "adapter_identity_before": packet_before,
        "hardware_probe": hardware,
        "workflow": workflow,
        "mask_association_validation": mask_association_validation,
        "method_resolution_validation": method_resolution_validation,
        "baseline_depth_inventory_validation": depth_inventory,
        "supplemental_fusion_view_selection": supplemental_selection,
        "supplemental_fusion_view_materialization": supplemental_materialization,
        "nonempty_virtual_view_selection": nonempty_view_selection,
        "nonempty_virtual_view_adapter": nonempty_view_adapter,
        "obb_removal_adapter": obb_removal_adapter,
        "lama_depth_numerical_validation": lama_depth_validation,
        "method_resolution_execution": resolution_accepted,
        "mask_association_executed": False,
        "mask_association_mode": MASK_ASSOCIATION_MODE,
        "pre_registered_mask_binding_materialized": (
            "pre_registered_mask_binding" in completed
        ),
        "virtual_masks_materialized": "virtual_masks" in completed,
        "lama_color_executed": "lama_color" in completed,
        "lama_depth_executed": "lama_depth" in completed,
        "inpaint_3d_executed": "inpaint_3d" in completed and final_ply.is_file(),
        "execution_source_class": "released_source_with_digest_bound_blueprint_obb_and_input_validity_adapters",
        "unchanged_source_execution_claimed": False,
        "final_point_cloud": _artifact(final_ply, output),
        "final_review_video": _artifact(final_video, output),
        "final_review_frames": review_frames,
        "rendered_frames_have_no_hidden_background_truth": True,
        "publisher_splat_is_not_metric_surface_truth": True,
        "replacement_or_physics_result": False,
        "retry_cap": 0,
        "blockers": sorted(set(blockers)),
        "raw_secret_values_recorded": False,
    }
    _write_json(output / "adp_inpaint360_interiorgs_result.json", result)
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
