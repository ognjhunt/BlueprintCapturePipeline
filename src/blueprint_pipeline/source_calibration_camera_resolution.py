"""Verify bounded camera replacement from retained pixels and frozen poses."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .task_evaluation_scene_configuration_submission_inputs import checked_file, sha

RECOVERY_SCHEMA = "source_calibration_camera_recovery.v1"
RESOLUTION_SCHEMA = "source_calibration_camera_resolution.v1"
ROLES = ("images", "target_support", "scene_without_target")


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError("source_calibration_camera_" + code)


def visibility_gate(prepared: dict) -> dict:
    policy = prepared["context"]["request"]["mask_policy"]
    return {
        "support_threshold_8bit": int(policy["support_threshold_8bit"]),
        "visual_contribution_threshold_8bit": int(policy.get("visual_contribution_threshold_8bit", 8)),
        "minimum_visible_target_fraction": float(policy.get("minimum_visible_target_fraction", .01)),
    }


def validate_recovery_contract(prepared: dict) -> dict | None:
    contract = prepared.get("camera_recovery")
    policy = prepared.get("context", {}).get("request", {}).get("camera_policy", {})
    reserves = policy.get("replacement_views", [])
    if contract is None:
        _require(not reserves, "replacement_contract_missing")
        return None
    from .public_scene_inpainting_inputs import _camera_rows
    _require(isinstance(contract, dict) and contract.get("schema_version") == RECOVERY_SCHEMA
             and contract.get("maximum_rounds") == 1
             and contract.get("visibility_gate") == visibility_gate(prepared), "recovery_contract_invalid")
    _require(isinstance(reserves, list) and 1 <= len(reserves) <= 16, "reserve_count_invalid")
    primary_ids = {row["camera_id"] for row in prepared["cameras"]}
    all_ids = [row.get("camera_id") for row in reserves]
    _require(len(set(all_ids)) == len(all_ids) and not primary_ids.intersection(all_ids), "reserve_ids_invalid")
    expected = _camera_rows({**prepared["context"]["request"], "camera_policy": {**policy, "views": reserves}},
                            np.asarray(prepared["context"]["corners"]).mean(axis=0))
    expected = [{"camera_id": row["camera_id"],
                 "T_world_camera_provider_frame": row["T_world_camera_opencv"],
                 "intrinsics": row["intrinsics"]} for row in expected]
    record = contract["replacement_camera_file"]
    path = checked_file(record["path"], record)
    _require(path.parent == Path(prepared["context"]["paths"]["output"])
             and not any(p.is_symlink() for p in (path, *path.parents)), "reserve_path_invalid")
    _require(json.loads(path.read_text()) == expected == contract.get("replacement_cameras"), "reserve_poses_changed")
    poses = [json.dumps(row["T_world_camera_provider_frame"], sort_keys=True)
             for row in [*prepared["cameras"], *expected]]
    _require(len(set(poses)) == len(poses), "duplicate_camera_pose")
    return contract


def measure_candidate(groups: dict, camera_id: str, gate: dict) -> dict:
    paths = {role: group["root"] / "frames" / f"{camera_id}.png" for role, group in groups.items()}
    pixels = {role: np.asarray(Image.open(path).convert("RGB")) for role, path in paths.items()}
    rgb, support, background = (pixels[role] for role in ROLES)
    _require(rgb.shape == support.shape == background.shape, "measurement_dimensions_invalid")
    support_mask = np.max(support, axis=2) >= gate["support_threshold_8bit"]
    changed = np.max(np.abs(rgb.astype(np.int16) - background.astype(np.int16)), axis=2)
    count = int((support_mask & (changed >= gate["visual_contribution_threshold_8bit"])).sum())
    total = int(rgb.shape[0] * rgb.shape[1])
    fraction, std = count / total, float(rgb.std())
    return {"candidate_camera_id": camera_id, "rgb_std": std,
            "support_contribution_pixels": count, "total_pixels": total,
            "visible_fraction": fraction,
            "passed": fraction >= gate["minimum_visible_target_fraction"] and std >= 1.0,
            "frame_digests": {role: sha(path) for role, path in paths.items()}}


def _with_cameras(prepared: dict, file_record: dict, cameras: list) -> dict:
    value = copy.deepcopy(prepared)
    value["camera_file"], value["cameras"] = file_record, cameras
    return value


def resolve_verified_cameras(prepared: dict, result: dict, result_root: Path) -> tuple[dict, dict | None]:
    """Recompute every attempt; accept only deterministic, bounded replacements."""
    from .source_calibration_render_return import _local_manifest, _verify_group, record
    contract = validate_recovery_contract(prepared)
    resolution = result.get("camera_resolution")
    if contract is None:
        _require(resolution is None, "unsolicited_resolution")
        return prepared, None
    _require(isinstance(resolution, dict) and resolution.get("schema_version") == RESOLUTION_SCHEMA
             and resolution.get("maximum_rounds") == 1, "resolution_missing_or_invalid")

    def group_set(rows, context):
        _require(isinstance(rows, list) and len(rows) == 3
                 and {row.get("role") for row in rows} == set(ROLES), "attempt_groups_invalid")
        return {row["role"]: _verify_group(context, row["role"], _local_manifest(result_root, row["manifest"]))
                for row in rows}

    original = group_set(resolution.get("original_render_groups"), prepared)
    primary = prepared["cameras"]
    gate = contract["visibility_gate"]
    metrics = [measure_candidate(original, row["camera_id"], gate) for row in primary]
    failed = [row for row in metrics if not row["passed"]]
    reserve_rows = contract["replacement_cameras"]
    replacements = resolution.get("replacement_render_groups")
    reserve_groups = None
    if failed:
        _require(resolution.get("rounds_used") == 1, "repair_round_required")
        reserve_context = _with_cameras(prepared, contract["replacement_camera_file"], reserve_rows)
        reserve_groups = group_set(replacements, reserve_context)
        metrics.extend(measure_candidate(reserve_groups, row["camera_id"], gate) for row in reserve_rows)
    else:
        _require(resolution.get("rounds_used") == 0 and replacements == [], "unneeded_repair_round")
    reported = resolution.get("measurement_rows")
    _require(isinstance(reported, list) and len(reported) == len(metrics), "measurement_inventory_invalid")
    for actual, claimed in zip(metrics, reported, strict=True):
        _require(all(claimed.get(key) == actual[key] for key in (
            "candidate_camera_id", "support_contribution_pixels", "total_pixels", "frame_digests", "passed")),
            "measurement_readback_mismatch")
        _require(all(isinstance(claimed.get(key), (float, int))
                     and abs(claimed[key] - actual[key]) <= 1e-8
                     for key in ("rgb_std", "visible_fraction")), "measurement_scalar_mismatch")
    by_id = {row["candidate_camera_id"]: row for row in metrics}
    available = iter(row for row in reserve_rows if by_id.get(row["camera_id"], {}).get("passed"))
    expected, selected_rows = [], []
    for original_row in primary:
        selected = original_row if by_id[original_row["camera_id"]]["passed"] else next(available, None)
        _require(selected is not None, "bounded_repair_exhausted")
        expected.append({"camera_id": original_row["camera_id"], "candidate_camera_id": selected["camera_id"]})
        selected_rows.append({**selected, "camera_id": original_row["camera_id"]})
    _require(resolution.get("selection") == expected, "selection_changed")
    file_path = _local_manifest(result_root, resolution["resolved_camera_file"])
    _require(json.loads(file_path.read_text()) == selected_rows, "resolved_pose_changed")
    effective = _with_cameras(prepared, record(file_path), selected_rows)
    accepted = group_set(result.get("render_groups"), effective)
    for row in expected:
        source = original if row["candidate_camera_id"] in {v["camera_id"] for v in primary} else reserve_groups
        for role in ROLES:
            _require(sha(accepted[role]["root"] / "frames" / f"{row['camera_id']}.png")
                     == sha(source[role]["root"] / "frames" / f"{row['candidate_camera_id']}.png"),
                     "selected_frame_changed")
    return effective, {"selection": expected, "rounds_used": resolution["rounds_used"],
                       "measurement_rows": metrics, "resolved_camera_file": record(file_path)}
