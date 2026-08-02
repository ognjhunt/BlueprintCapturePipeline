"""Independently qualify visual robot placement from signed Isaac artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .decision_evidence_contracts import canonical_digest
from .external_provider_nurec import (
    build_provider_nurec_isaac_request,
    build_provider_nurec_isaac_runtime_result,
)


SCHEMA_VERSION = "provider_robot_placement_evidence.v1"
MAX_DEPTH_BYTES = 64 * 1024**2
MIN_FOREGROUND_PIXELS = 32


class ProviderRobotPlacementEvidenceError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _artifact(root: Path, reference: Any, suffix: str) -> Path:
    text = str(reference or "").replace("\\", "/")
    relative = PurePosixPath(text)
    if (
        not text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or ":" in relative.parts[0]
        or relative.suffix.lower() != suffix
    ):
        raise ProviderRobotPlacementEvidenceError(
            ["provider_robot_placement_artifact_reference_unsafe"]
        )
    path = root.joinpath(*relative.parts)
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProviderRobotPlacementEvidenceError(
            ["provider_robot_placement_artifact_missing"]
        ) from exc
    if path.is_symlink() or not resolved.is_file() or root not in resolved.parents:
        raise ProviderRobotPlacementEvidenceError(["provider_robot_placement_artifact_invalid"])
    return resolved


def _build_signed_isaac_visual_placement_evidence(
    *,
    verification_request: Mapping[str, Any],
    runtime_result: Mapping[str, Any],
    runtime_artifact_root: str | Path,
    request_builder: Callable[[Mapping[str, Any]], dict[str, Any]],
    runtime_builder: Callable[..., dict[str, Any]],
    schema_version: str,
    digest_field: str,
) -> dict[str, Any]:
    """Rehash robot RGB/depth evidence and qualify only visual placement."""

    request = request_builder(verification_request)
    runtime = runtime_builder(
        runtime_result,
        verification_request=request,
    )
    root = Path(runtime_artifact_root)
    if root.is_symlink() or not root.is_dir():
        raise ProviderRobotPlacementEvidenceError(
            ["provider_robot_placement_artifact_root_invalid"]
        )
    root = root.resolve()
    robot = runtime.get("robot")
    robot = dict(robot) if isinstance(robot, Mapping) else {}
    blockers: list[str] = []
    runtime_blockers = sorted(
        str(code) for code in (runtime.get("blockers") or []) if str(code)
    )
    policy_only_abstention = bool(
        runtime.get("status") == "blocked"
        and runtime_blockers == ["isaac_articulated_policy_trace_pair_incomplete"]
    )
    if runtime.get("status") != "completed" and not policy_only_abstention:
        blockers.append("provider_robot_placement_runtime_not_completed")
    if (
        robot.get("requested") is not True
        or robot.get("composited") is not True
        or robot.get("geometry_streamed") is not True
        or not isinstance(robot.get("mesh_point_total"), int)
        or int(robot.get("mesh_point_total") or 0) < 100
    ):
        blockers.append("provider_robot_placement_geometry_not_observed")
    if (
        robot.get("robot_only_environment_hidden") is not True
        or not isinstance(robot.get("robot_only_hidden_environment_prim_paths"), list)
        or not robot.get("robot_only_hidden_environment_prim_paths")
    ):
        blockers.append("provider_robot_placement_robot_only_isolation_not_proven")
    pose = robot.get("robot_pose")
    bounds_min = robot.get("world_bound_min")
    bounds_max = robot.get("world_bound_max")
    for value, code, expected_length in (
        (pose, "provider_robot_placement_pose_invalid", 4),
        (bounds_min, "provider_robot_placement_bounds_invalid", 3),
        (bounds_max, "provider_robot_placement_bounds_invalid", 3),
    ):
        if not isinstance(value, list) or len(value) != expected_length:
            blockers.append(code)

    passes = robot.get("robot_only_pass")
    passes = passes if isinstance(passes, list) else []
    rows: list[dict[str, Any]] = []
    for item in passes:
        if not isinstance(item, Mapping):
            blockers.append("provider_robot_placement_camera_evidence_invalid")
            continue
        try:
            rgb = _artifact(root, item.get("rgb_artifact_reference"), ".png")
            depth = _artifact(
                root,
                item.get("distance_artifact_reference"),
                ".npy",
            )
            if (
                _sha256(rgb) != item.get("rgb_digest")
                or _sha256(depth) != item.get("distance_digest")
                or depth.stat().st_size > MAX_DEPTH_BYTES
            ):
                raise ProviderRobotPlacementEvidenceError(
                    ["provider_robot_placement_artifact_digest_mismatch"]
                )
            array = np.load(depth, allow_pickle=False)
            if array.ndim != 2 or not np.issubdtype(array.dtype, np.floating):
                raise ProviderRobotPlacementEvidenceError(
                    ["provider_robot_placement_depth_invalid"]
                )
            foreground = np.isfinite(array) & (array > 0)
            foreground_count = int(foreground.sum())
            values = array[foreground]
            depth_visible = foreground_count >= max(
                MIN_FOREGROUND_PIXELS,
                int(array.size * 0.00001),
            )
            if not depth_visible:
                blockers.append(
                    f"provider_robot_placement_depth_foreground_too_small:{item.get('id')}"
                )
            rows.append(
                {
                    "id": item.get("id"),
                    "rgb_artifact_reference": item.get("rgb_artifact_reference"),
                    "rgb_digest": item.get("rgb_digest"),
                    "distance_artifact_reference": item.get("distance_artifact_reference"),
                    "distance_digest": item.get("distance_digest"),
                    "depth_foreground_pixel_count": foreground_count,
                    "depth_foreground_fraction": round(
                        foreground_count / float(array.size),
                        12,
                    ),
                    "min_distance_m": round(float(values.min()), 6) if values.size else None,
                    "max_distance_m": round(float(values.max()), 6) if values.size else None,
                    "visual_geometry_observed": depth_visible,
                }
            )
        except (OSError, ValueError, ProviderRobotPlacementEvidenceError) as exc:
            if isinstance(exc, ProviderRobotPlacementEvidenceError):
                blockers.extend(exc.codes)
            else:
                blockers.append("provider_robot_placement_depth_invalid")
    if [row.get("id") for row in rows] != request["fixed_camera_ids"]:
        blockers.append("provider_robot_placement_camera_inventory_mismatch")

    evidence = {
        "schema_version": schema_version,
        "status": "verified_visual_placement_only" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "isaac_verification_request_digest": request["isaac_verification_request_digest"],
        "isaac_runtime_result_digest": runtime["isaac_runtime_result_digest"],
        "package_digest": request["package_digest"],
        "render_options_digest": request.get("render_options_digest"),
        "robot_prim_path": robot.get("prim_path"),
        "robot_usd": robot.get("robot_usd"),
        "resolved_robot_usd": robot.get("resolved_usd"),
        "robot_pose": pose,
        "world_bound_min": bounds_min,
        "world_bound_max": bounds_max,
        "mesh_point_total": robot.get("mesh_point_total"),
        "camera_evidence": rows,
        "visual_robot_placement_observed": not blockers,
        "policy_lane_abstained_without_invalidating_visual_evidence": policy_only_abstention,
        "collision_free_placement_proven": False,
        "kinematic_reachability_proven": False,
        "navigation_or_task_success_proven": False,
        "physical_robot_readiness_proven": False,
        "proof_effect": "visual_robot_placement_evidence_only",
        "claim_ceiling": "isaac_visual_robot_placement",
        "raw_secret_values_recorded": False,
    }
    evidence[digest_field] = canonical_digest(
        evidence,
        digest_field=digest_field,
    )
    return evidence


def build_provider_robot_placement_evidence(
    *,
    verification_request: Mapping[str, Any],
    runtime_result: Mapping[str, Any],
    runtime_artifact_root: str | Path,
) -> dict[str, Any]:
    return _build_signed_isaac_visual_placement_evidence(
        verification_request=verification_request,
        runtime_result=runtime_result,
        runtime_artifact_root=runtime_artifact_root,
        request_builder=build_provider_nurec_isaac_request,
        runtime_builder=build_provider_nurec_isaac_runtime_result,
        schema_version=SCHEMA_VERSION,
        digest_field="provider_robot_placement_evidence_digest",
    )


__all__ = [
    "ProviderRobotPlacementEvidenceError",
    "SCHEMA_VERSION",
    "build_provider_robot_placement_evidence",
]
