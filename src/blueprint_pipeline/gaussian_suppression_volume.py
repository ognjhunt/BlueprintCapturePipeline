"""Hide the replaced object's Gaussians at draw time instead of deleting them.

The sealed 840796 cutout answered "which splats does the twin's space own?" by
writing a new PLY with those rows removed. That works, but it forks a
multi-hundred-megabyte scan per edit, cannot compose two task objects against
one scene, and edits bytes the capture contract says we protect.

A suppression volume records the same answer as geometry instead of surgery:
the box the twin's body occupies, one swept prism per articulated member taken
to its *authored* limits rather than the commanded maximum, and an optional
annex of indices contributed by an admitted evidence process. Renderers resolve
that receipt against the untouched canonical scan and skip the resulting rows.
Turning a task off removes one small file and the original object returns
exactly.

A swept region is the one place this can quietly destroy real content: if the
door sweeps through occupied space, suppressing that space deletes scene the
twin only hides while the door happens to be open. The capture ceiling makes
that fail closed rather than look clean.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .gaussian_splat_decode import read_standard_3dgs_ply


GAUSSIAN_SUPPRESSION_VOLUME_SCHEMA_VERSION = "gaussian_suppression_volume.v1"
SUPPRESSION_COMPOSITION_SCHEMA_VERSION = "gaussian_suppression_composition.v1"
MEMBERSHIP_MODES = ("center_in_volume", "support_overlap_k_sigma")
_ANGLE_EPSILON = 1e-9


class GaussianSuppressionVolumeError(ValueError):
    """Stable, sorted suppression-volume failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_indices(indices: np.ndarray) -> str:
    payload = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
    return "sha256:" + hashlib.sha256(payload.tobytes()).hexdigest()


def _finite_vector(value: Any, length: int, error: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise GaussianSuppressionVolumeError([error])
    if len(value) != length:
        raise GaussianSuppressionVolumeError([error])
    out: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise GaussianSuppressionVolumeError([error])
        number = float(item)
        if not math.isfinite(number):
            raise GaussianSuppressionVolumeError([error])
        out.append(number)
    return out


def _support_radius(splat, multiplier: float) -> np.ndarray:
    """Per-splat conservative extent: k times the largest Gaussian axis."""

    if multiplier <= 0.0:
        return np.zeros(splat.count, dtype=np.float64)
    scales = np.exp(np.asarray(splat.scales, dtype=np.float64))
    return float(multiplier) * scales.max(axis=1)


def _box_membership(
    points: np.ndarray, region: Mapping[str, Any], radius: np.ndarray
) -> np.ndarray:
    minimum = np.asarray(region["world_aabb_min_m"], dtype=np.float64)
    maximum = np.asarray(region["world_aabb_max_m"], dtype=np.float64)
    # Distance from each centre to the box; zero inside.
    outside = np.maximum(minimum - points, 0.0) + np.maximum(points - maximum, 0.0)
    return np.linalg.norm(outside, axis=1) <= radius + 1e-12


def _swept_membership(
    points: np.ndarray, region: Mapping[str, Any], radius: np.ndarray
) -> np.ndarray:
    hinge = np.asarray(region["hinge_origin_world_m"], dtype=np.float64)
    interval = region["vertical_interval_m"]
    sweep_radius = float(region["member_radius_m"]) + float(region["half_thickness_m"])
    source_angle = float(region["closed_angle_radians"])
    lower, upper = (math.radians(value) for value in region["limit_degrees"])
    if upper < lower:
        lower, upper = upper, lower

    within_z = (points[:, 2] >= interval[0] - radius) & (
        points[:, 2] <= interval[1] + radius
    )
    delta = points[:, :2] - hinge[:2]
    distance = np.linalg.norm(delta, axis=1)
    within_radius = distance <= sweep_radius + radius
    # Angle of each point relative to the closed member direction, wrapped to
    # [0, 2pi) so a positive-Z rotation sweeps forward through it.
    relative = np.arctan2(delta[:, 1], delta[:, 0]) - source_angle
    relative = np.mod(relative + math.pi, 2.0 * math.pi) - math.pi
    forward = np.mod(relative, 2.0 * math.pi)
    # Near the hinge a thickness pad covers a wider angle than far from it.
    pad = np.arctan2(
        float(region["half_thickness_m"]) + radius, np.maximum(distance, 1e-6)
    )
    within_angle = (forward >= lower - pad - _ANGLE_EPSILON) & (
        forward <= upper + pad + _ANGLE_EPSILON
    )
    return within_z & within_radius & within_angle


def _region_membership(
    points: np.ndarray, region: Mapping[str, Any], radius: np.ndarray
) -> np.ndarray:
    if region["kind"] == "axis_aligned_box":
        return _box_membership(points, region, radius)
    if region["kind"] == "revolute_swept_prism":
        return _swept_membership(points, region, radius)
    raise GaussianSuppressionVolumeError(
        [f"suppression_region_kind_unsupported:{region['kind']}"]
    )


def _normalize_regions(
    *,
    body_world_aabb_min_m: Sequence[float] | None,
    body_world_aabb_max_m: Sequence[float] | None,
    body_margin_m: float,
    articulated_members: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    margin = float(body_margin_m)
    if not math.isfinite(margin) or margin < 0.0:
        raise GaussianSuppressionVolumeError(["suppression_body_margin_invalid"])
    if body_world_aabb_min_m is not None or body_world_aabb_max_m is not None:
        minimum = _finite_vector(
            body_world_aabb_min_m, 3, "suppression_body_aabb_invalid"
        )
        maximum = _finite_vector(
            body_world_aabb_max_m, 3, "suppression_body_aabb_invalid"
        )
        if any(minimum[axis] >= maximum[axis] for axis in range(3)):
            raise GaussianSuppressionVolumeError(["suppression_body_aabb_invalid"])
        regions.append(
            {
                "region_id": "body",
                "role": "body",
                "kind": "axis_aligned_box",
                "world_aabb_min_m": [value - margin for value in minimum],
                "world_aabb_max_m": [value + margin for value in maximum],
                "margin_m": margin,
            }
        )
    for index, member in enumerate(articulated_members):
        if not isinstance(member, Mapping):
            raise GaussianSuppressionVolumeError(
                [f"suppression_member_{index}_invalid"]
            )
        error = f"suppression_member_{index}_invalid"
        hinge = _finite_vector(member.get("hinge_origin_world_m"), 3, error)
        endpoint = _finite_vector(member.get("closed_endpoint_world_m"), 3, error)
        interval = _finite_vector(member.get("vertical_interval_m"), 2, error)
        limits = _finite_vector(member.get("limit_degrees"), 2, error)
        half_thickness = float(member.get("half_thickness_m") or 0.0)
        if (
            interval[0] >= interval[1]
            or limits[0] == limits[1]
            or half_thickness < 0.0
            or not math.isfinite(half_thickness)
        ):
            raise GaussianSuppressionVolumeError([error])
        radius = math.hypot(endpoint[0] - hinge[0], endpoint[1] - hinge[1])
        if radius <= 0.0:
            raise GaussianSuppressionVolumeError([error])
        regions.append(
            {
                "region_id": f"swept_{member.get('member_id') or index}",
                "role": "swept_member",
                "kind": "revolute_swept_prism",
                "member_id": str(member.get("member_id") or f"member_{index}"),
                "hinge_origin_world_m": hinge,
                "closed_endpoint_world_m": endpoint,
                "member_radius_m": radius,
                "closed_angle_radians": math.atan2(
                    endpoint[1] - hinge[1], endpoint[0] - hinge[0]
                ),
                "vertical_interval_m": interval,
                "limit_degrees": sorted(limits),
                "half_thickness_m": half_thickness,
            }
        )
    if not regions:
        raise GaussianSuppressionVolumeError(["suppression_no_region_defined"])
    return regions


def _annex_indices(
    index_annex: Mapping[str, Any] | None, vertex_count: int
) -> tuple[np.ndarray, dict[str, Any] | None]:
    if index_annex is None:
        return np.zeros(0, dtype=np.int64), None
    if not isinstance(index_annex, Mapping):
        raise GaussianSuppressionVolumeError(["index_annex_invalid"])
    path = Path(str(index_annex.get("path") or "")).expanduser()
    if not path.is_file():
        raise GaussianSuppressionVolumeError(["index_annex_missing"])
    try:
        values = np.load(path)
    except (OSError, ValueError) as exc:
        raise GaussianSuppressionVolumeError(["index_annex_unreadable"]) from exc
    values = np.asarray(values).reshape(-1)
    if values.size and not np.issubdtype(values.dtype, np.integer):
        raise GaussianSuppressionVolumeError(["index_annex_invalid"])
    values = np.unique(values.astype(np.int64))
    if values.size and (values.min() < 0 or values.max() >= vertex_count):
        raise GaussianSuppressionVolumeError(["index_annex_out_of_range"])
    declared = index_annex.get("sha256")
    observed = _sha256_file(path)
    if declared is not None and str(declared) != observed:
        raise GaussianSuppressionVolumeError(["index_annex_digest_mismatch"])
    record = {
        "path": str(path.resolve()),
        "sha256": observed,
        "count": int(values.size),
        "provenance": str(index_annex.get("provenance") or "unspecified"),
        "justification": str(index_annex.get("justification") or ""),
        "is_evidence_derived_not_geometric": True,
    }
    return values, record


def derive_suppression_volume_from_twin(
    *,
    task_id: str,
    canonical_ply_path: str | Path,
    body_world_aabb_min_m: Sequence[float] | None = None,
    body_world_aabb_max_m: Sequence[float] | None = None,
    body_margin_m: float = 0.0,
    articulated_members: Sequence[Mapping[str, Any]] = (),
    membership_mode: str = "center_in_volume",
    support_sigma_multiplier: float = 2.0,
    index_annex: Mapping[str, Any] | None = None,
    swept_region_capture_ceiling: int | None = None,
    twin_usd_path: str | Path | None = None,
    twin_usd_sha256: str | None = None,
    clearance_receipt_digest: str | None = None,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Bind the space one twin owns, and which canonical rows fall inside it."""

    if membership_mode not in MEMBERSHIP_MODES:
        raise GaussianSuppressionVolumeError(
            [f"suppression_membership_mode_unsupported:{membership_mode}"]
        )
    multiplier = float(support_sigma_multiplier)
    if not math.isfinite(multiplier) or multiplier < 0.0:
        raise GaussianSuppressionVolumeError(
            ["suppression_support_sigma_multiplier_invalid"]
        )
    canonical = Path(canonical_ply_path).expanduser().resolve()
    if not canonical.is_file():
        raise GaussianSuppressionVolumeError(["suppression_canonical_scan_missing"])

    regions = _normalize_regions(
        body_world_aabb_min_m=body_world_aabb_min_m,
        body_world_aabb_max_m=body_world_aabb_max_m,
        body_margin_m=body_margin_m,
        articulated_members=articulated_members,
    )
    splat = read_standard_3dgs_ply(canonical)
    points = np.asarray(splat.xyz, dtype=np.float64)
    radius = (
        _support_radius(splat, multiplier)
        if membership_mode == "support_overlap_k_sigma"
        else np.zeros(splat.count, dtype=np.float64)
    )
    annex, annex_record = _annex_indices(index_annex, splat.count)

    body_mask = np.zeros(splat.count, dtype=bool)
    swept_mask = np.zeros(splat.count, dtype=bool)
    for region in regions:
        member = _region_membership(points, region, radius)
        region["captured_index_count"] = int(member.sum())
        if region["role"] == "body":
            body_mask |= member
        else:
            swept_mask |= member

    swept_only = swept_mask & ~body_mask
    swept_only_count = int(swept_only.sum())
    if swept_region_capture_ceiling is not None:
        ceiling = int(swept_region_capture_ceiling)
        if swept_only_count > ceiling:
            raise GaussianSuppressionVolumeError(
                [
                    "suppression_swept_region_capture_exceeds_ceiling:"
                    f"{swept_only_count}>{ceiling}"
                ]
            )

    geometric = body_mask | swept_mask
    geometric_indices = np.nonzero(geometric)[0].astype(np.int64)
    suppressed = np.union1d(geometric_indices, annex)
    annex_only = int(np.setdiff1d(annex, geometric_indices).size)

    receipt: dict[str, Any] = {
        "schema_version": GAUSSIAN_SUPPRESSION_VOLUME_SCHEMA_VERSION,
        "status": "suppression_volume_bound",
        "task_id": str(task_id),
        "canonical_scan": {
            "path": str(canonical),
            "sha256": _sha256_file(canonical),
            "vertex_count": int(splat.count),
        },
        "twin": {
            "usd_path": str(Path(twin_usd_path).expanduser().resolve())
            if twin_usd_path
            else None,
            "usd_sha256": twin_usd_sha256,
        },
        "regions": regions,
        "membership": {
            "mode": membership_mode,
            "support_sigma_multiplier": multiplier
            if membership_mode == "support_overlap_k_sigma"
            else None,
        },
        "index_annex": annex_record,
        "capture": {
            "body_index_count": int(body_mask.sum()),
            "swept_only_index_count": swept_only_count,
            "annex_only_index_count": annex_only,
            "suppressed_index_count": int(suppressed.size),
            "retained_index_count": int(splat.count - suppressed.size),
            "suppressed_index_digest": _sha256_indices(suppressed),
        },
        "swept_region_capture_ceiling": (
            int(swept_region_capture_ceiling)
            if swept_region_capture_ceiling is not None
            else None
        ),
        "clearance_receipt_digest": clearance_receipt_digest,
        "claim_boundary": {
            "canonical_scan_modified": False,
            "reversible": True,
            "suppression_is_visibility_not_ownership": True,
            "generated_geometry_is_observed_site_truth": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    if destination is not None:
        write_json(Path(destination).expanduser().resolve(), receipt)
    return json.loads(json.dumps(receipt))


def resolve_suppressed_indices(
    *,
    canonical_ply_path: str | Path,
    receipt: Mapping[str, Any],
    verify_canonical_digest: bool = True,
) -> tuple[np.ndarray, str]:
    """Re-derive the suppressed rows for one receipt against the canonical scan."""

    if receipt.get("schema_version") != GAUSSIAN_SUPPRESSION_VOLUME_SCHEMA_VERSION:
        raise GaussianSuppressionVolumeError(["suppression_receipt_schema_invalid"])
    canonical = Path(canonical_ply_path).expanduser().resolve()
    if not canonical.is_file():
        raise GaussianSuppressionVolumeError(["suppression_canonical_scan_missing"])
    bound = receipt.get("canonical_scan") or {}
    if verify_canonical_digest and _sha256_file(canonical) != bound.get("sha256"):
        raise GaussianSuppressionVolumeError(
            ["suppression_canonical_scan_digest_mismatch"]
        )

    splat = read_standard_3dgs_ply(canonical)
    if int(splat.count) != int(bound.get("vertex_count") or -1):
        raise GaussianSuppressionVolumeError(
            ["suppression_canonical_scan_vertex_count_mismatch"]
        )
    points = np.asarray(splat.xyz, dtype=np.float64)
    membership = receipt.get("membership") or {}
    radius = (
        _support_radius(splat, float(membership.get("support_sigma_multiplier") or 0.0))
        if membership.get("mode") == "support_overlap_k_sigma"
        else np.zeros(splat.count, dtype=np.float64)
    )
    mask = np.zeros(splat.count, dtype=bool)
    for region in receipt.get("regions") or []:
        mask |= _region_membership(points, region, radius)
    indices = np.nonzero(mask)[0].astype(np.int64)

    annex_record = receipt.get("index_annex")
    if annex_record:
        annex_path = Path(str(annex_record.get("path") or "")).expanduser()
        if not annex_path.is_file():
            raise GaussianSuppressionVolumeError(["index_annex_missing"])
        if _sha256_file(annex_path) != annex_record.get("sha256"):
            raise GaussianSuppressionVolumeError(["index_annex_digest_mismatch"])
        indices = np.union1d(
            indices, np.unique(np.load(annex_path).reshape(-1).astype(np.int64))
        )

    digest = _sha256_indices(indices)
    declared = ((receipt.get("capture") or {}).get("suppressed_index_digest"))
    if declared is not None and digest != declared:
        raise GaussianSuppressionVolumeError(
            ["suppression_resolved_indices_digest_mismatch"]
        )
    return indices, digest


def compose_suppression_volumes(
    *,
    canonical_ply_path: str | Path,
    receipts: Iterable[Mapping[str, Any]],
    verify_canonical_digest: bool = True,
) -> dict[str, Any]:
    """Union many task volumes against one untouched scan."""

    rows = list(receipts)
    if not rows:
        raise GaussianSuppressionVolumeError(["composition_no_receipts"])
    canonical = Path(canonical_ply_path).expanduser().resolve()
    canonical_digest_value = _sha256_file(canonical)
    combined = np.zeros(0, dtype=np.int64)
    members: list[dict[str, Any]] = []
    errors: list[str] = []
    for receipt in rows:
        bound = (receipt.get("canonical_scan") or {}).get("sha256")
        if bound != canonical_digest_value:
            errors.append("composition_canonical_scan_mismatch")
            continue
        indices, digest = resolve_suppressed_indices(
            canonical_ply_path=canonical,
            receipt=receipt,
            verify_canonical_digest=verify_canonical_digest,
        )
        combined = np.union1d(combined, indices)
        members.append(
            {
                "task_id": str(receipt.get("task_id")),
                "receipt_digest": receipt.get("receipt_digest"),
                "suppressed_index_count": int(indices.size),
                "suppressed_index_digest": digest,
            }
        )
    if errors:
        raise GaussianSuppressionVolumeError(errors)

    members.sort(key=lambda row: row["task_id"])
    composite: dict[str, Any] = {
        "schema_version": SUPPRESSION_COMPOSITION_SCHEMA_VERSION,
        "status": "suppression_composition_resolved",
        "canonical_scan": {
            "path": str(canonical),
            "sha256": canonical_digest_value,
        },
        "task_ids": [row["task_id"] for row in members],
        "members": members,
        "suppressed_index_count": int(combined.size),
        "suppressed_index_digest": _sha256_indices(combined),
        "claim_boundary": {
            "canonical_scan_modified": False,
            "reversible": True,
        },
        "composite_digest": "",
    }
    composite["composite_digest"] = canonical_digest(
        composite, digest_field="composite_digest"
    )
    return composite


__all__ = [
    "GAUSSIAN_SUPPRESSION_VOLUME_SCHEMA_VERSION",
    "SUPPRESSION_COMPOSITION_SCHEMA_VERSION",
    "GaussianSuppressionVolumeError",
    "compose_suppression_volumes",
    "derive_suppression_volume_from_twin",
    "resolve_suppressed_indices",
]
