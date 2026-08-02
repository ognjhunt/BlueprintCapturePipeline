"""Bind a visible 2D proposal box to a derived 3D Gaussian region.

This is a reduced-authority fallback for external splats whose renderer does not
emit exact per-pixel Gaussian contributions. It projects digest-bound Gaussian
centers into one digest-bound rendered view and selects the front depth band in
the proposed box. The result is a target candidate with explicit uncertainty,
not semantic ground truth, metric scale proof, or collision qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest, canonical_json
from .gaussian_splat_decode import read_standard_3dgs_ply


REQUEST_SCHEMA = "splat_bbox_target_binding_request.v1"
RESULT_SCHEMA = "splat_bbox_target_binding_result.v1"


class SplatBBoxTargetBindingError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _finite3(value: Any) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result).all() else None


def build_splat_bbox_target_binding_request(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        request = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SplatBBoxTargetBindingError(["bbox_binding_request_not_json"]) from exc
    supplied = request.pop("request_digest", None)
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA:
        errors.append("bbox_binding_request_schema_invalid")
    for key in (
        "source_scene_digest",
        "analysis_splat_digest",
        "camera_spec_digest",
        "rgb_digest",
    ):
        if not _digest(request.get(key)):
            errors.append(f"bbox_binding_{key}_invalid")
    if not str(request.get("view_id") or "").strip():
        errors.append("bbox_binding_view_id_missing")
    image = request.get("image_size")
    if (
        not isinstance(image, Mapping)
        or not isinstance(image.get("width"), int)
        or isinstance(image.get("width"), bool)
        or not isinstance(image.get("height"), int)
        or isinstance(image.get("height"), bool)
        or not 64 <= image.get("width", 0) <= 8192
        or not 64 <= image.get("height", 0) <= 8192
    ):
        errors.append("bbox_binding_image_size_invalid")
    bbox = request.get("bbox_xyxy_pixels")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in (bbox or [])
        )
    ):
        errors.append("bbox_binding_bbox_invalid")
    elif isinstance(image, Mapping):
        x0, y0, x1, y1 = [float(item) for item in bbox]
        if not (0 <= x0 < x1 <= float(image["width"]) and 0 <= y0 < y1 <= float(image["height"])):
            errors.append("bbox_binding_bbox_out_of_frame")
    camera = request.get("camera")
    if not isinstance(camera, Mapping):
        errors.append("bbox_binding_camera_missing")
    else:
        for key in ("pos", "target", "up"):
            if _finite3(camera.get(key)) is None:
                errors.append(f"bbox_binding_camera_{key}_invalid")
        fov = camera.get("fov")
        if (
            isinstance(fov, bool)
            or not isinstance(fov, (int, float))
            or not 10.0 <= float(fov) <= 150.0
        ):
            errors.append("bbox_binding_camera_fov_invalid")
    opacity = request.get("minimum_opacity")
    if (
        isinstance(opacity, bool)
        or not isinstance(opacity, (int, float))
        or not 0.0 <= float(opacity) <= 1.0
    ):
        errors.append("bbox_binding_opacity_invalid")
    fraction = request.get("front_depth_fraction")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not 0.01 <= float(fraction) <= 0.5
    ):
        errors.append("bbox_binding_front_fraction_invalid")
    minimum = request.get("minimum_projected_splats")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 16:
        errors.append("bbox_binding_minimum_splats_invalid")
    if request.get("binding_may_self_authorize") is not False:
        errors.append("bbox_binding_self_authorization_forbidden")
    expected = canonical_digest(request, digest_field="request_digest")
    if supplied is not None and supplied != expected:
        errors.append("bbox_binding_request_digest_mismatch")
    if errors:
        raise SplatBBoxTargetBindingError(errors)
    request["request_digest"] = expected
    return request


def bind_splat_bbox_target(
    *, analysis_splat_path: str | Path, request: Mapping[str, Any]
) -> dict[str, Any]:
    admitted = build_splat_bbox_target_binding_request(request)
    source = Path(analysis_splat_path)
    if source.is_symlink():
        raise SplatBBoxTargetBindingError(["bbox_binding_splat_symlink_forbidden"])
    try:
        source = source.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SplatBBoxTargetBindingError(["bbox_binding_splat_missing"]) from exc
    if source.suffix.lower() != ".ply" or _sha256(source) != admitted["analysis_splat_digest"]:
        raise SplatBBoxTargetBindingError(["bbox_binding_splat_digest_mismatch"])
    splat = read_standard_3dgs_ply(source)
    points = np.asarray(splat.xyz, dtype=np.float64)
    camera = admitted["camera"]
    position = _finite3(camera["pos"])
    target = _finite3(camera["target"])
    supplied_up = _finite3(camera["up"])
    assert position is not None and target is not None and supplied_up is not None
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, supplied_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-9:
        raise SplatBBoxTargetBindingError(["bbox_binding_camera_basis_degenerate"])
    right /= right_norm
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    relative = points - position
    depth = relative @ forward
    width = int(admitted["image_size"]["width"])
    height = int(admitted["image_size"]["height"])
    tangent = math.tan(math.radians(float(camera["fov"])) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pixel_x = ((relative @ right) / (depth * tangent * width / height) + 1.0) * width / 2.0
        pixel_y = (1.0 - (relative @ up) / (depth * tangent)) * height / 2.0
    x0, y0, x1, y1 = [float(item) for item in admitted["bbox_xyxy_pixels"]]
    mask = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(pixel_x)
        & np.isfinite(pixel_y)
        & (depth > 0)
        & (np.asarray(splat.opacity_sigmoid) >= float(admitted["minimum_opacity"]))
        & (pixel_x >= x0)
        & (pixel_x < x1)
        & (pixel_y >= y0)
        & (pixel_y < y1)
    )
    projected = points[mask]
    projected_depth = depth[mask]
    minimum = int(admitted["minimum_projected_splats"])
    if len(projected) < minimum:
        raise SplatBBoxTargetBindingError(["bbox_binding_projected_support_insufficient"])
    cutoff = float(np.quantile(projected_depth, float(admitted["front_depth_fraction"])))
    front = projected[projected_depth <= cutoff]
    if len(front) < max(8, minimum // 4):
        raise SplatBBoxTargetBindingError(["bbox_binding_front_support_insufficient"])
    center = np.median(front, axis=0)
    q25, q75 = np.quantile(front, [0.25, 0.75], axis=0)
    uncertainty = float(np.linalg.norm(q75 - q25) / 2.0)
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "candidate_bound",
        "request_digest": admitted["request_digest"],
        "source_scene_digest": admitted["source_scene_digest"],
        "analysis_splat_digest": admitted["analysis_splat_digest"],
        "view_id": admitted["view_id"],
        "rgb_digest": admitted["rgb_digest"],
        "bbox_xyxy_pixels": list(admitted["bbox_xyxy_pixels"]),
        "method": "rendered_depth_backprojection",
        "position_scene": [round(float(item), 9) for item in center],
        "spatial_uncertainty_scene_units": round(uncertainty, 9),
        "projected_splat_count": int(len(projected)),
        "front_surface_splat_count": int(len(front)),
        "front_depth_cutoff_scene_units": round(cutoff, 9),
        "metric_scale_proven": False,
        "collision_support_proven": False,
        "binding_may_self_authorize": False,
        "proof_effect": "derived_visual_to_splat_3d_binding_candidate",
        "claim_ceiling": "task_target_binding_candidate",
    }
    result["binding_evidence_digest"] = canonical_digest(
        result, digest_field="binding_evidence_digest"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-splat", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--result-out", required=True)
    args = parser.parse_args(argv)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    if not isinstance(request, Mapping):
        raise SplatBBoxTargetBindingError(["bbox_binding_request_not_json_object"])
    result = bind_splat_bbox_target(
        analysis_splat_path=args.analysis_splat,
        request=request,
    )
    write_json(Path(args.result_out), result)
    print(canonical_json(result))
    return 0


__all__ = [
    "REQUEST_SCHEMA",
    "RESULT_SCHEMA",
    "SplatBBoxTargetBindingError",
    "bind_splat_bbox_target",
    "build_splat_bbox_target_binding_request",
]


if __name__ == "__main__":
    raise SystemExit(main())
