"""Construct explicit appearance support for a mixed foreground/background cutout.

The SAM/FlashSplat deletion proposal stays immutable. A separate, reviewable
candidate keeps its spatially distant or broad splats, removes only its bounded
foreground population, and seeds a registered support mesh with teacher colors.
Neither that reclassification nor the generated splats establishes object truth.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .artifixer_source_geometry_admission import _record
from .decision_evidence_contracts import canonical_digest, canonical_json
from .gaussian_splat_decode import _parse_ply_header, read_standard_3dgs_ply

SCHEMA = "artifixer_registered_background_initialization.v1"
GEOMETRY_MODE = "freeze_declared_appearance_initialization"
POLICY = {
    "selection": "registered_local_subset_of_immutable_segment_contribution_candidate",
    "foreground_margin_m": 0.05,
    "maximum_foreground_gaussian_scale_m": 0.08,
    "support_margin_m": 0.10,
    "surface_spacing_m": 0.001,
    "surface_tangent_scale_m": 0.001,
    "surface_normal_scale_m": 0.0001,
    "surface_opacity": 0.95,
    "minimum_color_views": 2,
    "maximum_surface_samples": 400_000,
    "maximum_support_bounds_deviation_m": 0.01,
    "original_appearance_frozen": True,
    "generated_geometry_and_opacity_frozen": True,
}


def bound_file(record: Mapping[str, Any]) -> Path:
    path = Path(str(record.get("path") or record.get("materialized_path") or ""))
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("artifixer_background_bound_file_missing")
    expected = record.get("sha256") or record.get("digest")
    actual = _record(path)
    if actual["sha256"] != expected or actual["size_bytes"] != record.get("size_bytes"):
        raise ValueError("artifixer_background_bound_file_changed")
    return path


def _bounds(lower, upper):
    lower, upper = np.asarray(lower, np.float64), np.asarray(upper, np.float64)
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or not np.isfinite([lower, upper]).all()
        or np.any(upper <= lower)
        or np.max(upper - lower) > 1.0
    ):
        raise ValueError("artifixer_background_target_bounds_invalid")
    return lower, upper


def local_foreground_mask(candidate, *, lower, upper):
    lower, upper = _bounds(lower, upper)
    positions = np.asarray(candidate.xyz, np.float64)
    scales = np.exp(np.asarray(candidate.scales, np.float64))
    if not np.isfinite(positions).all() or not np.isfinite(scales).all():
        raise ValueError("artifixer_background_candidate_nonfinite")
    margin = POLICY["foreground_margin_m"]
    selected = ((positions >= lower - margin) & (positions <= upper + margin)).all(axis=1)
    selected &= scales.max(axis=1) <= POLICY["maximum_foreground_gaussian_scale_m"]
    if not selected.any():
        raise ValueError("artifixer_background_local_foreground_empty")
    return selected


def sample_registered_top_surface(*, mesh_path: Path, support: Mapping, lower, upper):
    """Sample actual mesh triangles, rather than substituting the mesh AABB."""
    from pxr import Gf, Usd, UsdGeom

    lower, upper = _bounds(lower, upper)
    stage = Usd.Stage.Open(str(mesh_path))
    prim = stage.GetPrimAtPath(str(support["sage_prim_path"])) if stage else None
    if not prim or not prim.IsA(UsdGeom.Mesh):
        raise ValueError("artifixer_background_support_mesh_missing")
    mesh = UsdGeom.Mesh(prim)
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    vertices = np.asarray(
        [transform.Transform(Gf.Vec3d(*p)) for p in mesh.GetPointsAttr().Get()], np.float64
    )
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("artifixer_background_support_vertices_invalid")
    actual_lower, actual_upper = vertices.min(axis=0), vertices.max(axis=0)
    deviation = float(
        np.max(
            np.abs(
                np.asarray([actual_lower, actual_upper])
                - np.asarray([support["bounds_min_xyz_m"], support["bounds_max_xyz_m"]])
            )
        )
    )
    if deviation > POLICY["maximum_support_bounds_deviation_m"]:
        raise ValueError("artifixer_background_support_registration_mismatch")
    # Publisher instance boxes and matched SAGE vertices are separate records.
    # Sample the actual triangles and report their bounded disagreement.
    top = float(actual_upper[2])
    if abs(top - float(support["top_z_m"])) > POLICY["maximum_support_bounds_deviation_m"]:
        raise ValueError("artifixer_background_support_height_mismatch")
    if abs(top - lower[2]) > 0.01:
        raise ValueError("artifixer_background_target_support_contact_missing")
    margin, step = POLICY["support_margin_m"], POLICY["surface_spacing_m"]
    crop_lo = np.maximum(lower[:2] - margin, vertices[:, :2].min(axis=0))
    crop_hi = np.minimum(upper[:2] + margin, vertices[:, :2].max(axis=0))
    sizes = np.ceil((crop_hi - crop_lo) / step).astype(int)
    if np.any(sizes <= 0) or np.prod(sizes) > POLICY["maximum_surface_samples"]:
        raise ValueError("artifixer_background_support_sample_budget_exceeded")
    x = np.arange(crop_lo[0] + step / 2, crop_hi[0], step)
    y = np.arange(crop_lo[1] + step / 2, crop_hi[1], step)
    xx, yy = np.meshgrid(x, y)
    xy = np.c_[xx.ravel(), yy.ravel()]
    heights = np.full(len(xy), -np.inf)
    face_ids = np.full(len(xy), -1, np.int64)
    barycentrics = np.zeros((len(xy), 3))
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int64)
    offset = 0
    for face_id, count in enumerate(mesh.GetFaceVertexCountsAttr().Get()):
        ids = indices[offset : offset + count]
        offset += count
        for j in range(1, count - 1):
            triangle = vertices[[ids[0], ids[j], ids[j + 1]]]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            length = np.linalg.norm(normal)
            if (
                length < 1e-12
                or abs(normal[2]) / length < 0.999
                or np.max(np.abs(triangle[:, 2] - top)) > 0.003
            ):
                continue
            a, b, c = triangle[:, :2]
            u, v, d = b - a, c - a, xy - a
            denominator = u[0] * v[1] - u[1] * v[0]
            beta = (d[:, 0] * v[1] - d[:, 1] * v[0]) / denominator
            gamma = (u[0] * d[:, 1] - u[1] * d[:, 0]) / denominator
            bary = np.c_[1 - beta - gamma, beta, gamma]
            z = bary @ triangle[:, 2]
            take = (bary >= -1e-8).all(axis=1) & (z > heights)
            heights[take], face_ids[take], barycentrics[take] = z[take], face_id, bary[take]
    valid = face_ids >= 0
    if valid.mean() < 0.98:
        raise ValueError("artifixer_background_support_mesh_coverage_missing")
    points = np.c_[xy[valid], heights[valid]].astype(np.float32)
    return (
        points,
        face_ids[valid],
        barycentrics[valid],
        {
            "source_mesh_bounds_min_m": actual_lower.tolist(),
            "source_mesh_bounds_max_m": actual_upper.tolist(),
            "publisher_support_bounds_min_m": support["bounds_min_xyz_m"],
            "publisher_support_bounds_max_m": support["bounds_max_xyz_m"],
            "maximum_absolute_bounds_deviation_m": deviation,
            "admission_tolerance_m": POLICY["maximum_support_bounds_deviation_m"],
            "physical_alignment_qualified": False,
        },
    )


def sample_teacher_colors(points: np.ndarray, cameras: Sequence[Mapping], frames: Mapping):
    samples = []
    for camera in cameras:
        spec = camera["spec"]
        k = spec["intrinsics"]
        matrix = np.asarray(spec["pose"]["T_world_camera_opencv"], np.float64)
        if (
            matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1)
        ):
            raise ValueError("artifixer_background_camera_invalid")
        q = (points - matrix[:3, 3]) @ matrix[:3, :3]
        depth = np.maximum(q[:, 2], 1e-8)
        u = k["fx"] * q[:, 0] / depth + k["cx"]
        v = k["fy"] * q[:, 1] / depth + k["cy"]
        visible = (
            (q[:, 2] > 0.01)
            & (u >= 0)
            & (v >= 0)
            & (u < k["width"] - 1)
            & (v < k["height"] - 1)
            & (matrix[2, 3] > points[:, 2] + 0.01)
        )
        path = bound_file(frames[camera["id"]]["whole_frame_semantic_teacher"])
        with Image.open(path) as image:
            if image.size != (k["width"], k["height"]):
                raise ValueError("artifixer_background_teacher_camera_shape_mismatch")
            rgb = np.asarray(image.convert("RGB"), np.float32) / 255
        x, y = np.floor(u[visible]).astype(int), np.floor(v[visible]).astype(int)
        dx, dy = (u[visible] - x)[:, None], (v[visible] - y)[:, None]
        colors = np.full((len(points), 3), np.nan, np.float32)
        colors[visible] = (
            rgb[y, x] * (1 - dx) * (1 - dy)
            + rgb[y, x + 1] * dx * (1 - dy)
            + rgb[y + 1, x] * (1 - dx) * dy
            + rgb[y + 1, x + 1] * dx * dy
        )
        samples.append(colors)
    stack = np.stack(samples)
    counts = np.isfinite(stack[:, :, 0]).sum(axis=0)
    if counts.min(initial=POLICY["minimum_color_views"]) < POLICY["minimum_color_views"]:
        raise ValueError("artifixer_background_teacher_view_coverage_missing")
    return np.nanmedian(stack, axis=0), counts


def _raw_rows(path):
    with path.open("rb") as stream:
        fmt, count, properties, offset = _parse_ply_header(stream)
    if fmt != "binary_little_endian" or any(t not in {"float", "float32"} for t, _ in properties):
        raise ValueError("artifixer_background_source_ply_layout_invalid")
    body = path.read_bytes()
    if len(body) - offset != count * len(properties) * 4:
        raise ValueError("artifixer_background_source_ply_size_invalid")
    return (
        body[:offset],
        np.frombuffer(body[offset:], dtype=np.dtype([(n, "<f4") for _, n in properties])),
        properties,
    )


def write_initialization(
    *, retained: Path, deleted: Path, foreground_mask, points, colors, output_path: Path
):
    """Append new rows while preserving every reused source vertex byte."""
    header, source_rows, properties = _raw_rows(retained)
    _, candidate_rows, candidate_properties = _raw_rows(deleted)
    if candidate_properties != properties or len(foreground_mask) != len(candidate_rows):
        raise ValueError("artifixer_background_source_partition_layout_mismatch")
    restored = candidate_rows[~foreground_mask]
    seeds = np.zeros(len(points), dtype=source_rows.dtype)
    for axis, name in enumerate("xyz"):
        seeds[name] = points[:, axis]
    for axis in range(3):
        seeds[f"f_dc_{axis}"] = (colors[:, axis] - 0.5) / 0.28209479177387814
        scale = POLICY["surface_normal_scale_m"] if axis == 2 else POLICY["surface_tangent_scale_m"]
        seeds[f"scale_{axis}"] = np.log(scale)
    seeds["rot_0"] = 1
    opacity = POLICY["surface_opacity"]
    seeds["opacity"] = np.log(opacity / (1 - opacity))
    prefix = source_rows.tobytes() + restored.tobytes()
    count = len(source_rows) + len(restored) + len(seeds)
    header, replacements = re.subn(
        rb"(?m)^element vertex [0-9]+\r?$", f"element vertex {count}".encode(), header
    )
    if replacements != 1 or output_path.exists():
        raise ValueError("artifixer_background_output_invalid")
    with output_path.open("xb") as stream:
        stream.write(header)
        stream.write(prefix)
        stream.write(seeds.tobytes())
    _, check, _ = _raw_rows(output_path)
    if check[: len(source_rows) + len(restored)].tobytes() != prefix:
        raise ValueError("artifixer_background_source_prefix_changed")
    return {
        "frozen_source_count": len(source_rows) + len(restored),
        "generated_support_count": len(seeds),
        "total_count": count,
        "frozen_source_vertex_bytes_sha256": "sha256:" + hashlib.sha256(prefix).hexdigest(),
        "reused_source_vertex_rows_byte_exact": True,
    }


def materialize_background_initialization(
    *, envelope, configuration, candidate, teacher_receipt_path: Path, output_root: Path
):
    render = envelope["render_inputs_result"]
    cutout = render["derived_gaussian_cutout"]
    retained = bound_file(candidate["shared_retained_scene"])
    deleted = bound_file(cutout["source_object_candidate"])
    target = configuration["source_object"]
    lower, upper = _bounds(target["aabb_min_xyz_m"], target["aabb_max_xyz_m"])
    refs = {r["contract_path"]: r for r in envelope["materialized_references"]}
    collision = bound_file(refs["scene.geometry.collision"])
    support_path = bound_file(refs["scene.registration.support_plane"])
    support = json.loads(support_path.read_text())
    if support.get("scene_id") != configuration.get("scene_id"):
        raise ValueError("artifixer_background_support_scene_mismatch")
    teachers = json.loads(teacher_receipt_path.read_text())
    if (
        teachers.get("schema_version") != "public_scene_whole_frame_semantic_teacher_candidates.v1"
        or teachers.get("receipt_digest")
        != canonical_digest(teachers, digest_field="receipt_digest")
        or teachers.get("source_candidate_inputs_receipt", {}).get("receipt_digest")
        != candidate.get("receipt_digest")
    ):
        raise ValueError("artifixer_background_teacher_receipt_invalid")
    if teachers.get("editor_identity", {}).get("training_view_selection"):
        raise ValueError("artifixer_background_partial_teacher_set_not_supported")
    cameras = json.loads(bound_file(render["camera_calibration"]).read_text())
    frames = {r["camera_id"]: r for r in teachers["frames"]}
    if len(frames) < 8 or set(frames) != {c["id"] for c in cameras}:
        raise ValueError("artifixer_background_teacher_camera_set_mismatch")
    foreground = local_foreground_mask(read_standard_3dgs_ply(deleted), lower=lower, upper=upper)
    points, faces, barycentrics, registration = sample_registered_top_surface(
        mesh_path=collision, support=support, lower=lower, upper=upper
    )
    colors, counts = sample_teacher_colors(points, cameras, frames)
    output_root.mkdir(parents=True, exist_ok=False)
    sample_path = output_root / "registered_surface_samples.npz"
    np.savez_compressed(
        sample_path,
        points=points,
        source_face_indices=faces,
        barycentrics=barycentrics,
        color_view_counts=counts,
    )
    index_path = output_root / "foreground_indices_in_segment_candidate.npy"
    np.save(index_path, np.flatnonzero(foreground), allow_pickle=False)
    initialization = output_root / "declared_appearance_initialization.ply"
    partition = write_initialization(
        retained=retained,
        deleted=deleted,
        foreground_mask=foreground,
        points=points,
        colors=colors,
        output_path=initialization,
    )
    receipt = {
        "schema_version": SCHEMA,
        "status": "candidate_initialized_requires_training_and_review",
        "policy": POLICY,
        "geometry_mode": GEOMETRY_MODE,
        "source_retained": _record(retained),
        "segment_deletion_candidate": _record(deleted),
        "segment_cutout_set_digest": cutout["segment_cutout_set_digest"],
        "target_instance_id": target["publisher_instance_id"],
        "target_aabb_min_m": lower.tolist(),
        "target_aabb_max_m": upper.tolist(),
        "registered_mesh_sha256": _record(collision)["sha256"],
        "support_input": _record(support_path),
        "support_prim_path": support["sage_prim_path"],
        "support_instance_id": support["publisher_instance_id"],
        "support_registration": registration,
        "teacher_receipt": _record(teacher_receipt_path),
        "teacher_receipt_digest": teachers["receipt_digest"],
        "foreground_indices": _record(index_path),
        "surface_samples": _record(sample_path),
        "initialization": _record(initialization),
        "parameter_partition": partition,
        "original_segment_candidate_count": len(foreground),
        "local_foreground_deleted_count": int(foreground.sum()),
        "background_candidate_preserved_count": int((~foreground).sum()),
        "minimum_color_view_count": int(counts.min()),
        "original_segmentation_and_contribution_bytes_unchanged": True,
        "source_object_ownership_qualified": False,
        "source_object_absence_qualified": False,
        "appearance_qualified": False,
        "physical_or_capture_evidence": False,
        "generated_geometry_authority": "registered_mesh_backed_appearance_only",
        "provider_call_performed": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = output_root / "initialization_receipt.json"
    receipt_path.write_text(canonical_json(receipt) + "\n")
    return initialization, receipt_path, receipt


def prepare_background_supported_inputs(
    *,
    envelope,
    configuration,
    candidate,
    teacher_receipt_path: Path,
    preflight_path: Path,
    output_root: Path,
):
    """Rebind unchanged, already reviewed teacher bytes to the declared new seed."""
    import shutil
    from .public_scene_artifixer3d_candidate_inputs import materialize_artifixer3d_candidate_inputs
    from .public_scene_artifixer3d_dual_target_inputs import (
        materialize_whole_frame_semantic_teacher_receipt,
    )

    output_root.mkdir(parents=True, exist_ok=False)
    initialization, receipt_path, receipt = materialize_background_initialization(
        envelope=envelope,
        configuration=configuration,
        candidate=candidate,
        teacher_receipt_path=teacher_receipt_path,
        output_root=output_root / "initialization",
    )
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("preflight_digest") != canonical_digest(
        preflight, digest_field="preflight_digest"
    ):
        raise ValueError("artifixer_background_preflight_changed")
    preflight["shared_retained_scene"] = {
        **_record(initialization),
        "retained_gaussian_count": receipt["parameter_partition"]["total_count"],
    }
    preflight["appearance_initialization"] = {
        "receipt": _record(receipt_path),
        "receipt_digest": receipt["receipt_digest"],
        "geometry_mode": GEOMETRY_MODE,
        "parameter_partition": receipt["parameter_partition"],
    }
    preflight["preflight_digest"] = canonical_digest(preflight, digest_field="preflight_digest")
    new_preflight = output_root / "calibrated_preflight.v1.json"
    new_preflight.write_text(canonical_json(preflight) + "\n")
    new_root = output_root / "candidate_inputs"
    new_candidate = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=new_preflight, output_root=new_root
    )
    if new_candidate["shared_retained_scene"]["sha256"] != receipt["initialization"]["sha256"]:
        raise ValueError("artifixer_background_initialization_conditioning_changed_partition")
    new_candidate["appearance_initialization"] = preflight["appearance_initialization"]
    new_candidate["receipt_digest"] = canonical_digest(new_candidate, digest_field="receipt_digest")
    new_candidate_path = new_root / "public_scene_artifixer3d_candidate_inputs.v3.json"
    new_candidate_path.write_text(canonical_json(new_candidate) + "\n")
    teachers = json.loads(teacher_receipt_path.read_text())
    frames_root = output_root / "accepted_teacher_frames"
    frames_root.mkdir()
    for row in teachers["frames"]:
        source = bound_file(row["whole_frame_semantic_teacher"])
        target = frames_root / f"{row['frame_index']:05d}.png"
        shutil.copyfile(source, target)
        if _record(target)["sha256"] != _record(source)["sha256"]:
            raise ValueError("artifixer_background_teacher_copy_changed")
    new_teacher = output_root / "whole_frame_semantic_teacher.v1.json"
    materialize_whole_frame_semantic_teacher_receipt(
        source_candidate_inputs_receipt_path=new_candidate_path,
        task_id=teachers["task_id"],
        semantic_teacher_frames_root=frames_root,
        editor_identity={
            **teachers["editor_identity"],
            "unchanged_admitted_teacher_receipt_digest": teachers["receipt_digest"],
            "background_initialization_receipt_digest": receipt["receipt_digest"],
        },
        prompt_policy=teachers["prompt_policy"],
        output_path=new_teacher,
    )
    return new_candidate, new_candidate_path, new_teacher
