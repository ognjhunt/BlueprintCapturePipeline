"""Freeze and classify an auditable public-scene Gaussian object excision.

The source scene remains immutable.  A registered collision mesh supplies an
independent confident target core, the historical broad mask supplies a safety
envelope, and released-renderer ``alpha * transmittance`` totals supply the
appearance evidence.  Geometry and neighborhood consistency may make strong
evidence more conservative; neither may override protected-pixel evidence.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .decision_evidence_contracts import canonical_digest, canonical_json
from .gaussian_splat_decode import (
    read_standard_3dgs_ply,
    verify_standard_3dgs_ply_subset_exact,
    write_standard_3dgs_ply_subset_exact,
)


FREEZE_SCHEMA = "adp009b_gaussian_excision_audit_freeze.v1"
CONTRIBUTION_EVIDENCE_SCHEMA = "adp009b_gaussian_excision_contribution_evidence.v1"
OWNERSHIP_RECEIPT_SCHEMA = "adp009b_gaussian_excision_ownership_receipt.v1"
CONTRIBUTION_CLASS_ORDER = ("protected", "target_core", "uncertain")


class GaussianExcisionAuditError(ValueError):
    """Stable fail-closed Gaussian-excision errors."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path, root: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    value: dict[str, Any] = {
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }
    if root is None:
        value["path"] = str(resolved)
    else:
        value["relative_path"] = resolved.relative_to(root.resolve()).as_posix()
    return value


def _camera_vector(camera: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(camera.get("T_world_camera_opencv"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise GaussianExcisionAuditError(["excision_camera_transform_invalid"])
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0.0):
        raise GaussianExcisionAuditError(["excision_camera_transform_not_affine"])
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
        raise GaussianExcisionAuditError(["excision_camera_rotation_invalid"])
    forward = rotation[:, 2]
    return transform[:3, 3], forward / np.linalg.norm(forward)


def select_maximally_diverse_holdout_pair(
    cameras: Sequence[Mapping[str, Any]],
    *,
    projected_target_fraction: Mapping[str, float],
) -> dict[str, Any]:
    """Select two held-out cameras before outcomes using frozen geometric diversity."""

    if len(cameras) < 4:
        raise GaussianExcisionAuditError(["excision_camera_count_below_four"])
    rows: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    seen: set[str] = set()
    for camera in cameras:
        camera_id = str(camera.get("camera_id") or "").strip()
        if not camera_id or camera_id in seen:
            raise GaussianExcisionAuditError(["excision_camera_id_invalid"])
        seen.add(camera_id)
        if camera_id not in projected_target_fraction:
            raise GaussianExcisionAuditError(["excision_camera_target_fraction_missing"])
        fraction = float(projected_target_fraction[camera_id])
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise GaussianExcisionAuditError(["excision_camera_target_fraction_invalid"])
        position, forward = _camera_vector(camera)
        rows.append((camera_id, position, forward, fraction))

    raw: list[tuple[tuple[str, str], float, float, float]] = []
    for left, right in itertools.combinations(rows, 2):
        position_distance = float(np.linalg.norm(left[1] - right[1]))
        dot = float(np.clip(np.dot(left[2], right[2]), -1.0, 1.0))
        angular_distance = math.degrees(math.acos(dot))
        scale_distance = abs(left[3] - right[3])
        raw.append(
            (
                tuple(sorted((left[0], right[0]))),
                position_distance,
                angular_distance,
                scale_distance,
            )
        )
    maxima = np.max(np.asarray([row[1:] for row in raw], dtype=np.float64), axis=0)
    if maxima[0] <= 0.0 or maxima[1] <= 0.0:
        raise GaussianExcisionAuditError(["excision_camera_diversity_degenerate"])
    # Identical projected scale is valid; it simply contributes no diversity.
    maxima = np.where(maxima > 0.0, maxima, 1.0)
    scored = [
        (
            row[1] / maxima[0] + row[2] / maxima[1] + 0.25 * row[3] / maxima[2],
            row,
        )
        for row in raw
    ]
    score, selected = sorted(scored, key=lambda item: (-item[0], item[1][0]))[0]
    heldout = list(selected[0])
    calibration = sorted(seen - set(heldout))
    return {
        "method": "maximum_normalized_position_angle_and_quarter_scale_diversity.v1",
        "heldout_camera_ids": heldout,
        "calibration_camera_ids": calibration,
        "selected_score": round(float(score), 12),
        "selected_position_distance_m": round(selected[1], 12),
        "selected_view_direction_distance_deg": round(selected[2], 12),
        "selected_projected_target_fraction_distance": round(selected[3], 12),
        "outcome_fields_accessed": False,
    }


def _load_target_mesh(
    collision_path: Path, target_prim_path: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover
        raise GaussianExcisionAuditError(["excision_openusd_runtime_missing"]) from exc
    stage = Usd.Stage.Open(str(collision_path), load=Usd.Stage.LoadNone)
    if stage is None:
        raise GaussianExcisionAuditError(["excision_collision_stage_unreadable"])
    prim = stage.GetPrimAtPath(target_prim_path)
    mesh = UsdGeom.Mesh(prim)
    if not prim.IsValid() or not prim.IsActive() or not mesh:
        raise GaussianExcisionAuditError(["excision_target_collision_prim_missing"])
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not len(counts)
        or np.any(counts != 3)
        or len(indices) != 3 * len(counts)
    ):
        raise GaussianExcisionAuditError(["excision_target_collision_not_triangulated"])
    matrix = np.asarray(
        UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim),
        dtype=np.float64,
    ).T
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    world = (matrix @ homogeneous.T).T[:, :3]
    stage_info = {
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)).upper(),
        "target_point_count": int(len(points)),
        "target_triangle_count": int(len(counts)),
        "target_world_aabb_min_m": world.min(axis=0).tolist(),
        "target_world_aabb_max_m": world.max(axis=0).tolist(),
    }
    if stage_info["meters_per_unit"] != 1.0 or stage_info["up_axis"] != "Z":
        raise GaussianExcisionAuditError(["excision_collision_stage_metric_frame_invalid"])
    return world, indices.reshape((-1, 3)), stage_info


def _projected_mesh_mask(
    points_world: np.ndarray,
    faces: np.ndarray,
    camera: Mapping[str, Any],
    *,
    supersample: int,
) -> np.ndarray:
    intrinsics = camera.get("intrinsics")
    if not isinstance(intrinsics, Mapping):
        raise GaussianExcisionAuditError(["excision_camera_intrinsics_missing"])
    width = int(intrinsics.get("width") or 0)
    height = int(intrinsics.get("height") or 0)
    if width <= 0 or height <= 0 or supersample < 1:
        raise GaussianExcisionAuditError(["excision_mask_dimensions_invalid"])
    camera_to_world = np.asarray(camera.get("T_world_camera_opencv"), dtype=np.float64)
    _camera_vector(camera)
    homogeneous = np.concatenate(
        [points_world, np.ones((len(points_world), 1), dtype=np.float64)], axis=1
    )
    camera_points = (np.linalg.inv(camera_to_world) @ homogeneous.T).T[:, :3]
    depth = camera_points[:, 2]
    if np.any(depth <= 1e-6):
        raise GaussianExcisionAuditError(["excision_target_mesh_behind_camera"])
    pixels = np.column_stack(
        [
            float(intrinsics["fx"]) * camera_points[:, 0] / depth
            + float(intrinsics["cx"]),
            float(intrinsics["fy"]) * camera_points[:, 1] / depth
            + float(intrinsics["cy"]),
        ]
    )
    pixels *= supersample
    canvas = np.zeros((height * supersample, width * supersample), dtype=np.uint8)
    for face in faces:
        polygon = np.rint(pixels[face]).astype(np.int32)
        cv2.fillConvexPoly(canvas, polygon, 255, lineType=cv2.LINE_8)
    reduced = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
    return np.where(reduced >= 128, 255, 0).astype(np.uint8)


def materialize_excision_audit_freeze(
    *,
    source_standard_splat_path: str | Path,
    source_collision_path: str | Path,
    target_collision_prim_path: str,
    camera_contract_path: str | Path,
    source_image_root: str | Path,
    historical_outer_mask_root: str | Path,
    scene: Mapping[str, Any],
    policy: Mapping[str, Any],
    historical_baseline: Mapping[str, Any],
    output_root: str | Path,
    supersample: int = 2,
) -> dict[str, Any]:
    """Freeze independent mask zones and the six/two split before execution."""

    source_path = Path(source_standard_splat_path).expanduser().resolve()
    collision_path = Path(source_collision_path).expanduser().resolve()
    camera_path = Path(camera_contract_path).expanduser().resolve()
    image_root = Path(source_image_root).expanduser().resolve()
    outer_root = Path(historical_outer_mask_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    for path, code in (
        (source_path, "excision_source_splat_missing"),
        (collision_path, "excision_source_collision_missing"),
        (camera_path, "excision_camera_contract_missing"),
    ):
        if not path.is_file() or path.is_symlink():
            raise GaussianExcisionAuditError([code])
    try:
        cameras_value = json.loads(camera_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GaussianExcisionAuditError(["excision_camera_contract_invalid"]) from exc
    if not isinstance(cameras_value, list) or len(cameras_value) != 8:
        raise GaussianExcisionAuditError(["excision_exact_eight_cameras_required"])
    cameras = [dict(row) for row in cameras_value if isinstance(row, Mapping)]
    if len(cameras) != 8:
        raise GaussianExcisionAuditError(["excision_camera_contract_invalid"])

    splat = read_standard_3dgs_ply(source_path)
    world, faces, stage_info = _load_target_mesh(collision_path, target_collision_prim_path)
    output.mkdir(parents=True, exist_ok=True)
    mask_root = output / "masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    target_fractions: dict[str, float] = {}
    mask_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    for camera in sorted(cameras, key=lambda row: str(row.get("camera_id") or "")):
        camera_id = str(camera.get("camera_id") or "").strip()
        if not camera_id:
            raise GaussianExcisionAuditError(["excision_camera_id_invalid"])
        outer_path = outer_root / f"{camera_id}.png"
        image_path = image_root / f"{camera_id}.png"
        if not outer_path.is_file() or not image_path.is_file():
            raise GaussianExcisionAuditError([f"excision_camera_artifact_missing:{camera_id}"])
        with Image.open(outer_path) as image:
            outer = np.asarray(image.convert("L"), dtype=np.uint8) >= 128
        core = _projected_mesh_mask(
            world, faces, camera, supersample=supersample
        ) >= 128
        if core.shape != outer.shape or np.any(core & ~outer):
            raise GaussianExcisionAuditError(
                [f"excision_registered_core_not_inside_safety_envelope:{camera_id}"]
            )
        uncertain = outer & ~core
        protected = ~outer
        zones = {
            "target_core": core,
            "uncertain": uncertain,
            "protected": protected,
        }
        records: dict[str, Any] = {}
        for zone, values in zones.items():
            zone_path = mask_root / f"{camera_id}.{zone}.png"
            if not cv2.imwrite(str(zone_path), values.astype(np.uint8) * 255):
                raise GaussianExcisionAuditError(["excision_mask_write_failed"])
            records[zone] = _record(zone_path, output)
        fraction = float(core.mean())
        target_fractions[camera_id] = fraction
        mask_rows.append(
            {
                "camera_id": camera_id,
                "target_core_pixel_count": int(core.sum()),
                "uncertain_pixel_count": int(uncertain.sum()),
                "protected_pixel_count": int(protected.sum()),
                "target_core_fraction": fraction,
                "target_core_is_subset_of_historical_outer_mask": True,
                "historical_outer_mask": _record(outer_path),
                "zones": records,
            }
        )
        image_rows.append({"camera_id": camera_id, **_record(image_path)})

    split = select_maximally_diverse_holdout_pair(
        cameras, projected_target_fraction=target_fractions
    )
    baseline_method = historical_baseline.get("method")
    baseline_min = np.asarray(
        historical_baseline.get("center_aabb_min_m"), dtype=np.float64
    )
    baseline_max = np.asarray(
        historical_baseline.get("center_aabb_max_m"), dtype=np.float64
    )
    expected_baseline_count = historical_baseline.get("selected_gaussian_count")
    if (
        baseline_method != "center_inside_registered_target_aabb"
        or baseline_min.shape != (3,)
        or baseline_max.shape != (3,)
        or not np.isfinite(baseline_min).all()
        or not np.isfinite(baseline_max).all()
        or np.any(baseline_min >= baseline_max)
        or isinstance(expected_baseline_count, bool)
        or not isinstance(expected_baseline_count, int)
        or expected_baseline_count < 1
    ):
        raise GaussianExcisionAuditError(["excision_historical_baseline_invalid"])
    baseline_indices = np.flatnonzero(
        np.all((splat.xyz >= baseline_min) & (splat.xyz <= baseline_max), axis=1)
    ).astype(np.int64)
    if len(baseline_indices) != expected_baseline_count:
        raise GaussianExcisionAuditError(
            ["excision_historical_baseline_count_mismatch"]
        )
    baseline_path = output / "historical_obb_source_indices.npy"
    np.save(baseline_path, baseline_indices, allow_pickle=False)
    bounds_min, bounds_max = splat.aabb()
    freeze: dict[str, Any] = {
        "schema_version": FREEZE_SCHEMA,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009B",
        "status": "frozen_before_excision_execution",
        "scene": dict(scene),
        "source_standard_splat": _record(source_path),
        "source_collision": _record(collision_path),
        "target_collision_prim_path": target_collision_prim_path,
        "camera_contract": _record(camera_path),
        "source_images": image_rows,
        "masks": mask_rows,
        "camera_split": split,
        "scale_and_bounds": {
            "coordinate_frame": "InteriorGS_world_and_SAGE_world",
            "handedness": "right_handed",
            "up_axis": "Z",
            "meters_per_unit": 1.0,
            "interiorgs_to_sage_transform": np.eye(4).tolist(),
            "source_gaussian_count": int(splat.count),
            "source_center_aabb_min_m": bounds_min.astype(float).tolist(),
            "source_center_aabb_max_m": bounds_max.astype(float).tolist(),
            **stage_info,
            "claim_ceiling": "publisher_frame_metric_scale_not_physical_site_metrology",
        },
        "mask_method": {
            "target_core": "exact_registered_sage_target_triangle_projection",
            "uncertain": "historical_outer_mask_minus_registered_sage_target_core",
            "protected": "outside_historical_outer_mask",
            "renderer": "blueprint_deterministic_cpu_triangle_union_rasterizer.v1",
            "supersample": supersample,
            "binary_threshold_8bit": 128,
            "historical_outer_mask_is_final_ownership_authority": False,
            "collision_mask_is_exact_appearance_silhouette": False,
        },
        "contribution_method": {
            "name": "FlashSplat",
            "repository": "https://github.com/florinshen/FlashSplat",
            "commit": "3e3b14786333bf0163ba1b8541e86a3765112d7d",
            "rasterizer_repository": "https://github.com/florinshen/flashsplat-rasterization",
            "rasterizer_commit": "189c483ffa33dd6d5661343ce496df0c6eb80a0c",
            "contribution_semantics": "front_to_back_transmittance_times_alpha",
            "source_modified": False,
            "depth_anything_3_used": False,
        },
        "policy": dict(policy),
        "historical_baseline": {
            "method": baseline_method,
            "center_aabb_min_m": baseline_min.tolist(),
            "center_aabb_max_m": baseline_max.tolist(),
            "selected_gaussian_count": int(len(baseline_indices)),
            "indices": _record(baseline_path, output),
            "used_as_ground_truth": False,
        },
        "learned_policy_outcomes_observed": False,
        "replacement_usd_inserted": False,
    }
    freeze["freeze_digest"] = canonical_digest(freeze, digest_field="freeze_digest")
    freeze_path = output / f"{FREEZE_SCHEMA}.json"
    freeze_path.write_text(canonical_json(freeze) + "\n", encoding="utf-8")
    return freeze


def metric_geometry_score(
    xyz: np.ndarray,
    log_scales: np.ndarray,
    *,
    target_aabb_min_m: Sequence[float],
    target_aabb_max_m: Sequence[float],
    sigma_extent: float,
    margin_m: float,
) -> np.ndarray:
    """Return covariance-scale-aware proximity to an independently registered box."""

    xyz = np.asarray(xyz, dtype=np.float64)
    scales = np.asarray(log_scales, dtype=np.float64)
    lower = np.asarray(target_aabb_min_m, dtype=np.float64)
    upper = np.asarray(target_aabb_max_m, dtype=np.float64)
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or scales.shape != xyz.shape
        or lower.shape != (3,)
        or upper.shape != (3,)
        or np.any(lower >= upper)
        or not np.isfinite(xyz).all()
        or not np.isfinite(scales).all()
        or not math.isfinite(float(sigma_extent))
        or sigma_extent <= 0.0
        or not math.isfinite(float(margin_m))
        or margin_m < 0.0
    ):
        raise GaussianExcisionAuditError(["excision_metric_geometry_input_invalid"])
    delta = np.maximum(np.maximum(lower - xyz, xyz - upper), 0.0)
    distance = np.linalg.norm(delta, axis=1)
    radius = sigma_extent * np.exp(np.clip(np.max(scales, axis=1), -20.0, 20.0))
    denominator = np.maximum(radius + margin_m, 1e-9)
    return np.exp(-0.5 * np.square(distance / denominator))


def classify_excision_ownership(
    per_view_class_contribution: np.ndarray,
    *,
    xyz: np.ndarray,
    log_scales: np.ndarray,
    target_aabb_min_m: Sequence[float],
    target_aabb_max_m: Sequence[float],
    policy: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Fuse six-view contribution, metric proximity, and local consistency."""

    evidence = np.asarray(per_view_class_contribution, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    log_scales = np.asarray(log_scales, dtype=np.float64)
    if (
        evidence.ndim != 3
        or evidence.shape[0] < 2
        or evidence.shape[1] != len(CONTRIBUTION_CLASS_ORDER)
        or evidence.shape[2] != len(xyz)
        or log_scales.shape != xyz.shape
        or not np.isfinite(evidence).all()
        or np.any(evidence < 0.0)
    ):
        raise GaussianExcisionAuditError(["excision_contribution_evidence_invalid"])

    required = {
        "minimum_per_view_contribution": (0.0, 1e12),
        "owned_min_core_fraction": (0.5, 1.0),
        "retained_max_core_fraction": (0.0, 0.5),
        "minimum_geometry_score_owned": (0.0, 1.0),
        "geometry_sigma_extent": (1.0, 8.0),
        "geometry_margin_m": (0.0, 1.0),
        "neighbor_radius_m": (1e-6, 10.0),
        "neighbor_blend": (0.0, 1.0),
        "graph_owned_min_score": (0.5, 1.0),
        "graph_retained_max_score": (0.0, 0.5),
    }
    parsed: dict[str, float] = {}
    for key, (minimum, maximum) in required.items():
        value = policy.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GaussianExcisionAuditError([f"excision_policy_invalid:{key}"])
        number = float(value)
        if not math.isfinite(number) or not minimum <= number <= maximum:
            raise GaussianExcisionAuditError([f"excision_policy_invalid:{key}"])
        parsed[key] = number
    for key, minimum, maximum in (
        ("minimum_core_camera_count", 1, evidence.shape[0]),
        ("maximum_protected_camera_count_for_owned", 0, evidence.shape[0]),
        ("neighbor_count", 1, 64),
        ("neighbor_iterations", 1, 32),
    ):
        value = policy.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise GaussianExcisionAuditError([f"excision_policy_invalid:{key}"])
    if parsed["retained_max_core_fraction"] >= parsed["owned_min_core_fraction"]:
        raise GaussianExcisionAuditError(["excision_policy_fraction_order_invalid"])
    if parsed["graph_retained_max_score"] >= parsed["graph_owned_min_score"]:
        raise GaussianExcisionAuditError(["excision_policy_graph_order_invalid"])

    protected = evidence[:, 0, :]
    core = evidence[:, 1, :]
    uncertain = evidence[:, 2, :]
    minimum = parsed["minimum_per_view_contribution"]
    protected_count = np.sum(protected >= minimum, axis=0).astype(np.int16)
    core_count = np.sum(core >= minimum, axis=0).astype(np.int16)
    visible_count = np.sum((protected + core + uncertain) >= minimum, axis=0).astype(
        np.int16
    )
    protected_total = protected.sum(axis=0)
    core_total = core.sum(axis=0)
    denominator = protected_total + core_total
    core_fraction = np.zeros(len(xyz), dtype=np.float64)
    np.divide(core_total, denominator, out=core_fraction, where=denominator > 0.0)
    geometry = metric_geometry_score(
        xyz,
        log_scales,
        target_aabb_min_m=target_aabb_min_m,
        target_aabb_max_m=target_aabb_max_m,
        sigma_extent=parsed["geometry_sigma_extent"],
        margin_m=parsed["geometry_margin_m"],
    )
    score = 0.85 * core_fraction + 0.15 * geometry
    candidate = (core_total + uncertain.sum(axis=0) >= minimum) | (geometry >= 1e-3)
    candidate_indices = np.flatnonzero(candidate)
    if len(candidate_indices) > 1:
        count = min(int(policy["neighbor_count"]) + 1, len(candidate_indices))
        tree = cKDTree(xyz[candidate_indices])
        distances, neighbors = tree.query(
            xyz[candidate_indices],
            k=count,
            distance_upper_bound=parsed["neighbor_radius_m"],
            workers=1,
        )
        if distances.ndim == 1:
            distances = distances[:, None]
            neighbors = neighbors[:, None]
        local_score = score[candidate_indices].copy()
        radius = parsed["neighbor_radius_m"]
        for _ in range(int(policy["neighbor_iterations"])):
            sums = np.zeros_like(local_score)
            weights = np.zeros_like(local_score)
            for column in range(1, distances.shape[1]):
                valid = np.isfinite(distances[:, column]) & (
                    neighbors[:, column] < len(candidate_indices)
                )
                if not np.any(valid):
                    continue
                weight = np.exp(-0.5 * np.square(distances[valid, column] / radius))
                sums[valid] += weight * local_score[neighbors[valid, column]]
                weights[valid] += weight
            neighbor_mean = local_score.copy()
            np.divide(sums, weights, out=neighbor_mean, where=weights > 0.0)
            blend = parsed["neighbor_blend"]
            local_score = (1.0 - blend) * local_score + blend * neighbor_mean
        score[candidate_indices] = local_score

    owned = (
        (core_fraction >= parsed["owned_min_core_fraction"])
        & (core_count >= int(policy["minimum_core_camera_count"]))
        & (
            protected_count
            <= int(policy["maximum_protected_camera_count_for_owned"])
        )
        & (geometry >= parsed["minimum_geometry_score_owned"])
        & (score >= parsed["graph_owned_min_score"])
    )
    retained = (
        (~candidate)
        | (core_fraction <= parsed["retained_max_core_fraction"])
        | (score <= parsed["graph_retained_max_score"])
    ) & ~owned
    ambiguous = ~(owned | retained)
    if np.any(owned & retained) or np.any(owned & ambiguous) or np.any(retained & ambiguous):
        raise GaussianExcisionAuditError(["excision_ownership_partition_overlap"])
    if not np.all(owned | retained | ambiguous):
        raise GaussianExcisionAuditError(["excision_ownership_partition_incomplete"])
    return {
        "owned": owned,
        "retained": retained,
        "ambiguous": ambiguous,
        "core_fraction": core_fraction,
        "geometry_score": geometry,
        "neighborhood_score": score,
        "core_camera_count": core_count,
        "protected_camera_count": protected_count,
        "visible_camera_count": visible_count,
    }


def _load_contribution_array(path: Path, *, expected_shape: tuple[int, int, int]) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "per_view_class_contribution" not in archive:
                raise GaussianExcisionAuditError(
                    ["excision_contribution_array_key_missing"]
                )
            value = np.asarray(archive["per_view_class_contribution"], dtype=np.float64)
    except (OSError, ValueError) as exc:
        if isinstance(exc, GaussianExcisionAuditError):
            raise
        raise GaussianExcisionAuditError(["excision_contribution_array_invalid"]) from exc
    if (
        value.shape != expected_shape
        or not np.isfinite(value).all()
        or np.any(value < 0.0)
    ):
        raise GaussianExcisionAuditError(["excision_contribution_array_shape_invalid"])
    return value


def materialize_excision_ownership(
    *,
    freeze_path: str | Path,
    contribution_manifest_path: str | Path,
    source_standard_splat_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Create exact three-way index sets from two deterministic GPU repetitions."""

    freeze_file = Path(freeze_path).expanduser().resolve()
    manifest_file = Path(contribution_manifest_path).expanduser().resolve()
    source_path = Path(source_standard_splat_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    for path, code in (
        (freeze_file, "excision_freeze_missing"),
        (manifest_file, "excision_contribution_manifest_missing"),
        (source_path, "excision_source_splat_missing"),
    ):
        if not path.is_file() or path.is_symlink():
            raise GaussianExcisionAuditError([code])
    try:
        freeze = json.loads(freeze_file.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GaussianExcisionAuditError(["excision_bound_json_invalid"]) from exc
    if (
        freeze.get("schema_version") != FREEZE_SCHEMA
        or freeze.get("freeze_digest")
        != canonical_digest(freeze, digest_field="freeze_digest")
        or freeze.get("status") != "frozen_before_excision_execution"
    ):
        raise GaussianExcisionAuditError(["excision_freeze_invalid"])
    if (
        manifest.get("schema_version") != CONTRIBUTION_EVIDENCE_SCHEMA
        or manifest.get("manifest_digest")
        != canonical_digest(manifest, digest_field="manifest_digest")
        or manifest.get("freeze_digest") != freeze["freeze_digest"]
    ):
        raise GaussianExcisionAuditError(["excision_contribution_manifest_invalid"])
    if manifest.get("class_order") != list(CONTRIBUTION_CLASS_ORDER):
        raise GaussianExcisionAuditError(["excision_contribution_class_order_invalid"])
    calibration = freeze.get("camera_split", {}).get("calibration_camera_ids")
    if manifest.get("camera_ids") != calibration or not isinstance(calibration, list):
        raise GaussianExcisionAuditError(["excision_contribution_camera_split_mismatch"])
    method = manifest.get("method")
    frozen_method = freeze.get("contribution_method")
    if not isinstance(method, Mapping) or not isinstance(frozen_method, Mapping):
        raise GaussianExcisionAuditError(["excision_contribution_method_missing"])
    for key in (
        "name",
        "repository",
        "commit",
        "rasterizer_repository",
        "rasterizer_commit",
        "contribution_semantics",
        "source_modified",
        "depth_anything_3_used",
    ):
        if method.get(key) != frozen_method.get(key):
            raise GaussianExcisionAuditError(
                [f"excision_contribution_method_mismatch:{key}"]
            )
    if method.get("released_code_executed") is not True:
        raise GaussianExcisionAuditError(["excision_released_contribution_not_executed"])

    splat = read_standard_3dgs_ply(source_path)
    if _sha256(source_path) != freeze.get("source_standard_splat", {}).get("sha256"):
        raise GaussianExcisionAuditError(["excision_source_splat_changed"])
    if splat.count != freeze.get("scale_and_bounds", {}).get("source_gaussian_count"):
        raise GaussianExcisionAuditError(["excision_source_gaussian_count_changed"])
    baseline_binding = freeze.get("historical_baseline")
    if not isinstance(baseline_binding, Mapping):
        raise GaussianExcisionAuditError(["excision_historical_baseline_missing"])
    baseline_relative_path = baseline_binding.get("indices", {}).get("relative_path")
    if not isinstance(baseline_relative_path, str) or not baseline_relative_path:
        raise GaussianExcisionAuditError(["excision_historical_obb_indices_missing"])
    baseline_path = freeze_file.parent / baseline_relative_path
    if (
        not baseline_path.is_file()
        or baseline_path.is_symlink()
        or baseline_path.stat().st_size
        != baseline_binding.get("indices", {}).get("size_bytes")
        or _sha256(baseline_path)
        != baseline_binding.get("indices", {}).get("sha256")
    ):
        raise GaussianExcisionAuditError(["excision_historical_obb_indices_changed"])
    repetitions = manifest.get("repetitions")
    expected_repetitions = int(freeze.get("policy", {}).get("deterministic_repetitions", 0))
    if (
        not isinstance(repetitions, list)
        or len(repetitions) != expected_repetitions
        or expected_repetitions < 2
    ):
        raise GaussianExcisionAuditError(["excision_contribution_repetitions_invalid"])
    decimals = freeze.get("policy", {}).get("contribution_quantization_decimals")
    if isinstance(decimals, bool) or not isinstance(decimals, int) or not 3 <= decimals <= 12:
        raise GaussianExcisionAuditError(["excision_contribution_quantization_invalid"])
    arrays: list[np.ndarray] = []
    for row in repetitions:
        if not isinstance(row, Mapping):
            raise GaussianExcisionAuditError(["excision_contribution_repetition_invalid"])
        path = manifest_file.parent / str(row.get("relative_path") or "")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row.get("size_bytes")
            or _sha256(path) != row.get("sha256")
        ):
            raise GaussianExcisionAuditError(["excision_contribution_repetition_changed"])
        arrays.append(
            np.round(
                _load_contribution_array(
                    path,
                    expected_shape=(len(calibration), len(CONTRIBUTION_CLASS_ORDER), splat.count),
                ),
                decimals=decimals,
            )
        )
    if any(not np.array_equal(arrays[0], value) for value in arrays[1:]):
        raise GaussianExcisionAuditError(
            ["excision_quantized_contribution_repetitions_nondeterministic"]
        )

    scale = freeze["scale_and_bounds"]
    result = classify_excision_ownership(
        arrays[0],
        xyz=splat.xyz,
        log_scales=splat.scales,
        target_aabb_min_m=scale["target_world_aabb_min_m"],
        target_aabb_max_m=scale["target_world_aabb_max_m"],
        policy=freeze["policy"],
    )
    owned = np.flatnonzero(result["owned"]).astype(np.int64)
    retained = np.flatnonzero(result["retained"]).astype(np.int64)
    ambiguous = np.flatnonzero(result["ambiguous"]).astype(np.int64)
    if not len(owned):
        raise GaussianExcisionAuditError(["excision_owned_set_empty"])
    try:
        baseline = np.load(baseline_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise GaussianExcisionAuditError(["excision_historical_obb_indices_invalid"]) from exc
    baseline = np.asarray(baseline)
    if (
        baseline.ndim != 1
        or baseline.dtype.kind not in {"i", "u"}
        or len(baseline) != baseline_binding.get("selected_gaussian_count")
        or np.any(baseline[1:] <= baseline[:-1])
        or int(baseline[0]) < 0
        or int(baseline[-1]) >= splat.count
    ):
        raise GaussianExcisionAuditError(["excision_historical_obb_indices_invalid"])
    baseline = baseline.astype(np.int64, copy=False)

    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "owned_indices": output / "owned_source_indices.npy",
        "retained_indices": output / "retained_source_indices.npy",
        "ambiguous_indices": output / "ambiguous_source_indices.npy",
        "historical_obb_indices": output / "historical_obb_source_indices.npy",
        "ownership_labels": output / "gaussian_ownership_labels.npy",
    }
    labels = np.full(splat.count, 1, dtype=np.uint8)
    labels[owned] = 0
    labels[ambiguous] = 2
    for name, values in (
        ("owned_indices", owned),
        ("retained_indices", retained),
        ("ambiguous_indices", ambiguous),
        ("historical_obb_indices", baseline),
        ("ownership_labels", labels),
    ):
        np.save(paths[name], values, allow_pickle=False)
    retained_scene = np.sort(np.concatenate([retained, ambiguous]))
    ply_paths = {
        "owned_gaussians": write_standard_3dgs_ply_subset_exact(
            source_path, output / "owned_gaussians.ply", owned
        ),
        "retained_scene_gaussians": write_standard_3dgs_ply_subset_exact(
            source_path, output / "retained_scene_gaussians.ply", retained_scene
        ),
        "ambiguous_gaussians": write_standard_3dgs_ply_subset_exact(
            source_path, output / "ambiguous_gaussians.ply", ambiguous
        ),
        "historical_obb_gaussians": write_standard_3dgs_ply_subset_exact(
            source_path, output / "historical_obb_gaussians.ply", baseline
        ),
    }
    exact = {
        name: verify_standard_3dgs_ply_subset_exact(source_path, path, indices)
        for name, path, indices in (
            ("owned", ply_paths["owned_gaussians"], owned),
            ("retained_scene", ply_paths["retained_scene_gaussians"], retained_scene),
            ("ambiguous", ply_paths["ambiguous_gaussians"], ambiguous),
            ("historical_obb", ply_paths["historical_obb_gaussians"], baseline),
        )
    }
    if any(row.get("retained_rows_byte_exact") is not True for row in exact.values()):
        raise GaussianExcisionAuditError(["excision_partition_rows_changed"])

    diagnostic_paths: dict[str, Path] = {}
    for name, values in (
        ("core_fraction", result["core_fraction"]),
        ("geometry_score", result["geometry_score"]),
        ("neighborhood_score", result["neighborhood_score"]),
        ("core_camera_count", result["core_camera_count"]),
        ("protected_camera_count", result["protected_camera_count"]),
        ("visible_camera_count", result["visible_camera_count"]),
    ):
        path = output / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        diagnostic_paths[name] = path

    receipt: dict[str, Any] = {
        "schema_version": OWNERSHIP_RECEIPT_SCHEMA,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009B",
        "status": "three_way_ownership_materialized_heldout_not_evaluated",
        "freeze_digest": freeze["freeze_digest"],
        "contribution_manifest_digest": manifest["manifest_digest"],
        "method": method,
        "camera_split": freeze["camera_split"],
        "source_standard_splat": freeze["source_standard_splat"],
        "ownership": {
            "source_gaussian_count": splat.count,
            "owned_count": int(len(owned)),
            "retained_count": int(len(retained)),
            "ambiguous_count": int(len(ambiguous)),
            "historical_obb_count": int(len(baseline)),
            "exhaustive": len(owned) + len(retained) + len(ambiguous) == splat.count,
            "pairwise_disjoint": True,
        },
        "determinism": {
            "repetition_count": len(arrays),
            "quantization_decimals": decimals,
            "quantized_contribution_arrays_identical": True,
        },
        "preservation": exact,
        "outputs": {
            **{name: _record(path, output) for name, path in paths.items()},
            **{name: _record(path, output) for name, path in ply_paths.items()},
            **{name: _record(path, output) for name, path in diagnostic_paths.items()},
        },
        "heldout_cameras_accessed_for_classification": False,
        "replacement_usd_inserted": False,
        "claim_ceiling": "byte_exact_three_way_gaussian_ownership_candidate_pending_heldout_audit",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = output / f"{OWNERSHIP_RECEIPT_SCHEMA}.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


__all__ = [
    "CONTRIBUTION_EVIDENCE_SCHEMA",
    "CONTRIBUTION_CLASS_ORDER",
    "FREEZE_SCHEMA",
    "GaussianExcisionAuditError",
    "OWNERSHIP_RECEIPT_SCHEMA",
    "classify_excision_ownership",
    "materialize_excision_audit_freeze",
    "materialize_excision_ownership",
    "metric_geometry_score",
    "select_maximally_diverse_holdout_pair",
]
