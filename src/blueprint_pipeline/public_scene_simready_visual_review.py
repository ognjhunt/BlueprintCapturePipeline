"""Render the exact SimReady replacement into sealed Aura camera frames.

This is a deterministic software visual-review seam.  It proves the camera,
scale, and authored placement are mutually consistent; it is not OVRTX or
OVPhysX evidence and cannot establish contact dynamics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .common import utc_now_iso
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "adp009b_simready_visual_review_receipt.v1"


class SimReadyVisualReviewError(ValueError):
    """The bound replacement or camera evidence cannot be rendered truthfully."""


def _read(path: Path, *, error: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimReadyVisualReviewError(error) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.resolve().relative_to(root.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _bound_mesh(stage_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover
        raise SimReadyVisualReviewError("openusd_runtime_missing") from exc
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise SimReadyVisualReviewError("replacement_stage_open_failed")
    prim = stage.GetPrimAtPath("/World/BlueprintReplacement/visuals/body")
    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        raise SimReadyVisualReviewError("replacement_visual_mesh_missing")
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    matrix = np.asarray(
        UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim),
        dtype=np.float64,
    ).T
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    world = (matrix @ homogeneous.T).T[:, :3]
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if not len(counts) or np.any(counts != 3) or len(indices) != 3 * len(counts):
        raise SimReadyVisualReviewError("replacement_visual_mesh_not_triangulated")
    faces = indices.reshape((-1, 3))
    colors = mesh.GetDisplayColorAttr().Get()
    base_color = np.asarray(colors[0] if colors else (0.08, 0.7, 0.3), dtype=np.float64)
    return world, faces, base_color


def _project(
    points_world: np.ndarray, camera_to_world: np.ndarray, intrinsics: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(camera_to_world)
    homogeneous = np.concatenate(
        [points_world, np.ones((len(points_world), 1), dtype=np.float64)], axis=1
    )
    camera = (world_to_camera @ homogeneous.T).T[:, :3]
    z = camera[:, 2]
    if np.any(z <= 1e-6):
        raise SimReadyVisualReviewError("replacement_mesh_behind_camera")
    pixels = np.column_stack(
        [
            float(intrinsics["fx"]) * camera[:, 0] / z + float(intrinsics["cx"]),
            float(intrinsics["fy"]) * camera[:, 1] / z + float(intrinsics["cy"]),
        ]
    )
    return pixels, z


def _render_layer(
    *,
    points_world: np.ndarray,
    faces: np.ndarray,
    base_color: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: Mapping[str, Any],
    supersample: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    pixels, depth = _project(points_world, camera_to_world, intrinsics)
    pixels *= supersample
    canvas = np.zeros((height * supersample, width * supersample, 4), dtype=np.uint8)
    camera_position = camera_to_world[:3, 3]
    centers = points_world[faces].mean(axis=1)
    normals = np.cross(
        points_world[faces[:, 1]] - points_world[faces[:, 0]],
        points_world[faces[:, 2]] - points_world[faces[:, 0]],
    )
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-12)
    view = camera_position[None, :] - centers
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-12)
    facing = np.abs(np.sum(normals * view, axis=1))
    lighting = np.clip(0.42 + 0.58 * facing, 0.35, 1.0)
    face_depth = depth[faces].mean(axis=1)
    # Painter order is sufficient for this closed convex can mesh.
    for index in np.argsort(face_depth)[::-1]:
        polygon = np.rint(pixels[faces[index]]).astype(np.int32)
        if (
            polygon[:, 0].max() < 0
            or polygon[:, 1].max() < 0
            or polygon[:, 0].min() >= width * supersample
            or polygon[:, 1].min() >= height * supersample
        ):
            continue
        top_or_bottom = abs(normals[index, 2]) > 0.72
        if top_or_bottom:
            rgb = np.array([0.72, 0.75, 0.76]) * (0.65 + 0.35 * lighting[index])
        else:
            rgb = base_color * lighting[index]
            # A restrained camera-facing metallic highlight makes curvature legible.
            rgb += 0.16 * np.power(facing[index], 18.0)
        bgr = tuple(int(v) for v in np.clip(rgb[::-1] * 255.0, 0, 255))
        cv2.fillConvexPoly(canvas, polygon, (*bgr, 255), lineType=cv2.LINE_AA)
    layer = cv2.resize(
        canvas,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    alpha = layer[:, :, 3]
    return layer[:, :, :3], alpha


def _composite(background: np.ndarray, foreground: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    mask = alpha.astype(np.float32) / 255.0
    # A small, soft contact shadow is presentation-only and does not assert contact proof.
    shadow = cv2.GaussianBlur(alpha, (0, 0), 5.0)
    shadow = np.roll(shadow, shift=(4, 3), axis=(0, 1)).astype(np.float32) / 255.0
    darkened = background.astype(np.float32) * (1.0 - 0.16 * shadow[:, :, None])
    output = foreground.astype(np.float32) * mask[:, :, None] + darkened * (
        1.0 - mask[:, :, None]
    )
    return np.clip(output, 0, 255).astype(np.uint8)


def materialize_visual_review(
    *,
    replacement_receipt_path: str | Path,
    exact_camera_manifest_path: str | Path,
    cameras_path: str | Path,
    frame_root: str | Path,
    evidence_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    evidence = Path(evidence_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output != evidence and evidence not in output.parents:
        raise SimReadyVisualReviewError("visual_review_output_outside_evidence_root")
    replacement_path = Path(replacement_receipt_path).expanduser().resolve()
    exact_path = Path(exact_camera_manifest_path).expanduser().resolve()
    camera_path = Path(cameras_path).expanduser().resolve()
    frames = Path(frame_root).expanduser().resolve()
    replacement = _read(replacement_path, error="replacement_receipt_invalid")
    exact = _read(exact_path, error="exact_camera_manifest_invalid")
    cameras = _read(camera_path, error="camera_contract_invalid")
    if replacement.get("receipt_digest") != canonical_digest(
        replacement, digest_field="receipt_digest"
    ):
        raise SimReadyVisualReviewError("replacement_receipt_digest_invalid")
    if replacement.get("status") != "composed_static_candidate":
        raise SimReadyVisualReviewError("replacement_static_candidate_required")
    if not isinstance(cameras, list) or not cameras:
        raise SimReadyVisualReviewError("camera_contract_empty")
    camera_by_id = {str(row.get("camera_id")): row for row in cameras if isinstance(row, dict)}
    render_rows = exact.get("renders")
    if not isinstance(render_rows, list) or set(camera_by_id) != {
        str(row.get("camera_id")) for row in render_rows if isinstance(row, dict)
    }:
        raise SimReadyVisualReviewError("camera_identity_mismatch")
    composed_record = replacement.get("composition")
    if not isinstance(composed_record, Mapping):
        raise SimReadyVisualReviewError("replacement_composition_missing")
    stage_path = evidence / str(composed_record.get("relative_path") or "")
    if not stage_path.is_file() or _sha256(stage_path) != composed_record.get("sha256"):
        raise SimReadyVisualReviewError("replacement_stage_digest_mismatch")
    points, faces, color = _bound_mesh(stage_path)
    output.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for row in render_rows:
        camera_id = str(row["camera_id"])
        source = frames / Path(str(row["relative_path"])).name
        if not source.is_file() or _sha256(source) != row.get("digest"):
            raise SimReadyVisualReviewError(f"sealed_frame_digest_mismatch:{camera_id}")
        background = cv2.imread(str(source), cv2.IMREAD_COLOR)
        camera = camera_by_id[camera_id]
        intrinsics = camera.get("intrinsics")
        if not isinstance(intrinsics, Mapping) or background is None:
            raise SimReadyVisualReviewError(f"camera_or_frame_invalid:{camera_id}")
        if background.shape[:2] != (int(intrinsics["height"]), int(intrinsics["width"])):
            raise SimReadyVisualReviewError(f"frame_resolution_mismatch:{camera_id}")
        camera_to_world = np.asarray(camera.get("T_world_camera_opencv"), dtype=np.float64)
        if camera_to_world.shape != (4, 4) or not np.allclose(
            np.linalg.inv(camera_to_world) @ camera_to_world, np.eye(4), atol=1e-8
        ):
            raise SimReadyVisualReviewError(f"camera_transform_invalid:{camera_id}")
        layer, alpha = _render_layer(
            points_world=points,
            faces=faces,
            base_color=color,
            camera_to_world=camera_to_world,
            intrinsics=intrinsics,
        )
        after = _composite(background, layer, alpha)
        before_path = output / f"{camera_id}.before.png"
        after_path = output / f"{camera_id}.after.png"
        comparison_path = output / f"{camera_id}.before_after.png"
        mask_path = output / f"{camera_id}.replacement_mask.png"
        cv2.imwrite(str(before_path), background)
        cv2.imwrite(str(after_path), after)
        cv2.imwrite(str(comparison_path), np.concatenate([background, after], axis=1))
        cv2.imwrite(str(mask_path), alpha)
        ys, xs = np.nonzero(alpha)
        if not len(xs):
            raise SimReadyVisualReviewError(f"replacement_not_visible:{camera_id}")
        pad_x = max(80, int((xs.max() - xs.min()) * 1.8))
        pad_y = max(80, int((ys.max() - ys.min()) * 1.2))
        x0, x1 = max(0, xs.min() - pad_x), min(background.shape[1], xs.max() + pad_x)
        y0, y1 = max(0, ys.min() - pad_y), min(background.shape[0], ys.max() + pad_y)
        crop_path = output / f"{camera_id}.contact_crop.before_after.png"
        cv2.imwrite(
            str(crop_path),
            np.concatenate([background[y0:y1, x0:x1], after[y0:y1, x0:x1]], axis=1),
        )
        artifacts.append(
            {
                "camera_id": camera_id,
                "source_frame_sha256": row["digest"],
                "visible_pixel_count": int(np.count_nonzero(alpha)),
                "before": _record(before_path, output),
                "after": _record(after_path, output),
                "before_after": _record(comparison_path, output),
                "contact_crop_before_after": _record(crop_path, output),
                "replacement_mask": _record(mask_path, output),
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "rendered_visual_review_candidate",
        "replacement_receipt_digest": replacement["receipt_digest"],
        "exact_camera_manifest_digest": exact.get("sealed_camera_render_manifest_digest"),
        "renderer": "blueprint_deterministic_cpu_triangle_rasterizer_v1",
        "renderer_is_native_ovrtx": False,
        "contact_shadow_is_presentation_only": True,
        "artifacts": artifacts,
        "claim_ceiling": "camera_scale_pose_visual_review_only",
        "dynamic_contact_proven": False,
        "human_visual_acceptance": "pending",
        "blockers": [
            "human_visual_acceptance_pending",
            "native_ovrtx_render_missing",
            "native_ovphysx_drop_contact_settle_missing",
        ],
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = output / "adp009b_simready_visual_review_receipt.v1.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replacement-receipt", type=Path, required=True)
    parser.add_argument("--exact-camera-manifest", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    materialize_visual_review(
        replacement_receipt_path=args.replacement_receipt,
        exact_camera_manifest_path=args.exact_camera_manifest,
        cameras_path=args.cameras,
        frame_root=args.frame_root,
        evidence_root=args.evidence_root,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
