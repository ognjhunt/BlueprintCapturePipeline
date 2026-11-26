"""Mesh registration reads the splat and the GLB through memory maps, to the same record and GLB bytes.

The functions below ``# Frozen`` are main's before 2026-11-25: they decoded the whole splat PLY
(every column copied out of its body), read the whole GLB, took the support test over (faces, 3)
distances at once and sorted every kept corner (``np.unique``). The rework is checked against them
byte for byte, and its traced memory is bounded.
"""
from __future__ import annotations

import json
import struct
import tracemalloc
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from blueprint_pipeline import atlas_mesh_registration as registration
from blueprint_pipeline.atlas_depth_splat import PLY_PROPERTIES, SH_C0, posed_frame
from blueprint_pipeline.atlas_mesh_registration import (  # noqa: F401  (read by the frozen functions)
    _COMPONENTS, _WIDTH, COLLIDER_ESTIMATE_FLAG, FAILURE_RECORD, GLB_FRAMES, MAX_MEDIAN_RESIDUAL_VOXELS,
    REGISTRATION_MESH_EDGES, REGISTRATION_SAMPLES, SCHEMA_VERSION, SUPPORT_MESH_EDGES, SUPPORT_VOXELS, _refusal,
    mesh_depth_free_space_filter, write_glb_triangles)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gaussian_splat_decode import SplatData, read_standard_3dgs_ply, write_standard_3dgs_ply
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from tests.test_atlas_mesh_registration import VOXEL, exported_glb, grid

_BASE_FACE_BLOCK = 1 << 20


def _splat_root(root: Path, points: np.ndarray, *, pose_digest: str | None = None, seed: int = 0) -> Path:
    """A splat with a colour of its own at every Gaussian, so the colour gather is checked row by row."""
    rng = np.random.default_rng(seed)
    count = len(points)
    splat = SplatData(count=count, xyz=points.astype(np.float32), opacity=np.full(count, 3.0, np.float32),
                      f_dc=rng.normal(0.0, 0.8, (count, 3)).astype(np.float32),
                      scales=np.full((count, 3), np.log(VOXEL), np.float32),
                      quats=np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (count, 1)), properties=PLY_PROPERTIES)
    root.mkdir(parents=True, exist_ok=True)
    ply = write_standard_3dgs_ply(splat, root / "atlas_depth_splat.ply")
    manifest = {"voxel_size": VOXEL, "units": "provider_world_units",
                "outputs": [{"role": "splat_ply", "path": ply.name, "sha256": _sha256_file(ply)}],
                **({"pose_result_digest": pose_digest} if pose_digest else {})}
    manifest["digest"] = canonical_digest(manifest, digest_field="digest")
    (root / "atlas_depth_splat_manifest.json").write_text(json.dumps(manifest))
    return root


def _view(frame_id: str, x: float = 0.0):
    return posed_frame(frame_id=frame_id, rgb=np.zeros((64, 64, 3), np.uint8),
                       depth=np.full((64, 64), 2.0, np.float32), confidence=np.ones((64, 64), np.float32),
                       camera={"intrinsics": {"fx": 32.0, "fy": 32.0, "cx": 32.0, "cy": 32.0, "width": 64, "height": 64},
                               "extrinsics": {"position": [x, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0],
                                              "coordinateSystem": "rub"}})


def _case(tmp_path: Path, name: str) -> dict[str, Any]:
    pose = "sha256:" + "a" * 64
    if name == "registered":     # the rotated export, its closing sheet trimmed
        observed, phantom = grid(-2.0, extent=0.3), grid(-1.6, extent=0.15)
        return {"splat_root": _splat_root(tmp_path / "splat", grid(-2.0)[0]),
                "glb": exported_glb(tmp_path / "mesh.glb", observed, phantom)}
    if name == "estimated":      # fits no declared frame: its supported faces in the closest frame, flagged
        return {"splat_root": _splat_root(tmp_path / "splat", grid(-2.0)[0]),
                "glb": exported_glb(tmp_path / "mesh.glb", grid(-2.0, x0=0.6), sign=(1.0, 1.0, 1.0))}
    observed, phantom = grid(-2.0, extent=0.2), grid(-1.8, extent=0.2)   # the depth filter removes the phantom
    return {"splat_root": _splat_root(tmp_path / "splat", np.concatenate([observed[0], phantom[0]]), pose_digest=pose),
            "glb": exported_glb(tmp_path / "mesh.glb", observed, phantom),
            "reference_frames": [_view("left", x=-0.2), _view("right", x=0.2)], "reference_pose_digest": pose}


def _written(root: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(root.iterdir())}


@pytest.mark.parametrize("name", ["registered", "estimated", "depth_filtered"])
@pytest.mark.parametrize("block", [7, 1 << 18])
def test_registration_is_the_frozen_base_byte_for_byte(tmp_path, monkeypatch, name, block):
    case = _case(tmp_path, name)
    base = _base_register_atlas_mesh(**case, output_root=tmp_path / "base")
    monkeypatch.setattr(registration, "FACE_BLOCK", block)
    record = registration.register_atlas_mesh(**case, output_root=tmp_path / "rework")
    assert record == base
    assert _written(tmp_path / "rework") == _written(tmp_path / "base")
    assert record["faces_kept"] < record["faces_in"]


def test_glb_read_through_its_map_is_the_whole_file_read(tmp_path):
    vertices, faces = grid(-1.0, extent=0.05)
    path = write_glb_triangles(tmp_path / "m.glb", vertices, faces, np.zeros((len(vertices), 3)))
    for got, want in zip(registration.read_glb_triangles(path), _base_read_glb_triangles(path)):
        assert got.dtype == want.dtype and np.array_equal(got, want)
    for cut in (0, 12, 27, path.stat().st_size - 4):
        short = tmp_path / f"short-{cut}.glb"
        short.write_bytes(path.read_bytes()[:cut])
        with pytest.raises(ValueError, match="atlas_mesh_glb_invalid"):
            registration.read_glb_triangles(short)


def test_registration_holds_neither_the_decoded_splat_nor_every_corner_sorted(tmp_path):
    """Main traced 258 MB here (about 200 bytes a Gaussian and 120 a face); the rework 134 MB."""
    points, _ = grid(-2.0, extent=2.5, step=0.005)
    splat = _splat_root(tmp_path / "splat", points)
    glb = exported_glb(tmp_path / "mesh.glb", grid(-2.0, extent=2.5, step=0.01))
    tracemalloc.start()
    try:
        record = registration.register_atlas_mesh(glb=glb, splat_root=splat, output_root=tmp_path / "registered")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    gaussians, faces = len(points), record["faces_in"]
    assert gaussians > 10**6 and faces == 500_000
    assert peak < 64 * gaussians + 96 * faces + (48 << 20), peak


# Frozen: main's functions before 2026-11-25.


def _base_read_glb_triangles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Positions and triangle indices of a single-primitive, untransformed GLB."""
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data)
    json_length, json_type = struct.unpack_from("<I4s", data, 12)
    bin_length, bin_type = struct.unpack_from("<I4s", data, 20 + json_length)
    if (magic != b"glTF" or version != 2 or length != len(data) or json_type != b"JSON"
            or bin_type != b"BIN\0" or 28 + json_length + bin_length > len(data)):
        raise ValueError("atlas_mesh_glb_invalid")
    gltf = json.loads(data[20:20 + json_length])
    binary = data[28 + json_length:28 + json_length + bin_length]
    if any(key in node for node in gltf.get("nodes", []) for key in ("matrix", "rotation", "translation", "scale")):
        raise ValueError("atlas_mesh_node_transform_unsupported")
    if len(gltf.get("meshes", [])) != 1 or len(gltf["meshes"][0]["primitives"]) != 1:
        raise ValueError("atlas_mesh_primitive_count_unsupported")
    primitive = gltf["meshes"][0]["primitives"][0]
    if primitive.get("mode", 4) != 4 or "indices" not in primitive:
        raise ValueError("atlas_mesh_not_indexed_triangles")

    def accessor(index: int) -> np.ndarray:
        row = gltf["accessors"][index]
        view = gltf["bufferViews"][row["bufferView"]]
        dtype, width = np.dtype(_COMPONENTS[row["componentType"]]), _WIDTH[row["type"]]
        if view.get("byteStride", dtype.itemsize * width) != dtype.itemsize * width:
            raise ValueError("atlas_mesh_strided_accessor_unsupported")
        start = view.get("byteOffset", 0) + row.get("byteOffset", 0)
        return np.frombuffer(binary, dtype=dtype.newbyteorder("<"), count=row["count"] * width,
                             offset=start).reshape(row["count"], width)

    vertices = accessor(primitive["attributes"]["POSITION"]).astype(np.float64)
    faces = accessor(primitive["indices"]).reshape(-1, 3).astype(np.int64)
    if not len(faces) or faces.max() >= len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("atlas_mesh_geometry_invalid")
    return vertices, faces


def _base_register_atlas_mesh(*, glb: Path, splat_root: Path, output_root: Path,
                        reference_frames: Sequence[Mapping[str, Any]] | None = None,
                        reference_pose_digest: str | None = None) -> dict[str, Any]:
    """Fit the GLB to the observed splat, trim unsupported faces and write the result."""
    from scipy.spatial import cKDTree

    from blueprint_pipeline.atlas_native_splat import load_splat_manifest
    splat_manifest = load_splat_manifest(splat_root)
    if splat_manifest.get("digest") != canonical_digest(splat_manifest, digest_field="digest"):
        raise ValueError("atlas_mesh_splat_manifest_changed")
    if reference_frames is not None and (
            not reference_pose_digest or reference_pose_digest != splat_manifest.get("pose_result_digest")):
        raise ValueError("atlas_mesh_depth_filter_pose_binding_invalid")
    ply = splat_root / next(row["path"] for row in splat_manifest["outputs"] if row["role"] == "splat_ply")
    if ply.is_symlink() or not ply.resolve().is_relative_to(splat_root.resolve()):
        raise ValueError("atlas_mesh_splat_path_invalid")
    if _sha256_file(ply) != next(row["sha256"] for row in splat_manifest["outputs"] if row["role"] == "splat_ply"):
        raise ValueError("atlas_mesh_splat_changed")
    splat = read_standard_3dgs_ply(ply)
    voxel = float(splat_manifest["voxel_size"])
    if not np.isfinite(voxel) or voxel <= 0:
        raise ValueError("atlas_mesh_voxel_size_invalid")
    vertices, faces = _base_read_glb_triangles(glb)

    def per_face(rows: np.ndarray, value) -> np.ndarray:
        """``value`` of ``vertices[rows]``, gathered ``_BASE_FACE_BLOCK`` faces at a time: each face's own arithmetic."""
        return np.concatenate([value(vertices[rows[start:start + _BASE_FACE_BLOCK]])
                               for start in range(0, len(rows), _BASE_FACE_BLOCK)] or [value(vertices[rows[:0]])])

    # Edge length does not change under the declared axis flips, so it is measured once, before the search.
    edge = float(np.median(per_face(faces, lambda tri: np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1))))
    tree = cKDTree(splat.xyz)
    mesh_tree = cKDTree(vertices)
    sample = vertices[np.linspace(0, len(vertices) - 1, min(REGISTRATION_SAMPLES, len(vertices))).astype(np.int64)]
    source_sample = splat.xyz[np.linspace(0, len(splat.xyz) - 1,
                                         min(REGISTRATION_SAMPLES, len(splat.xyz))).astype(np.int64)]
    limit = max(MAX_MEDIAN_RESIDUAL_VOXELS * voxel, REGISTRATION_MESH_EDGES * edge)
    diagnostics, residuals, candidates = {}, {}, []
    for name, sign in GLB_FRAMES.items():
        # A partial mesh can cover less than half the source; closing sheets can
        # outnumber supported mesh vertices. Measure both directions at the same
        # fixed tolerance. These declared axis flips are their own inverses.
        mesh_distance = tree.query(sample * np.array(sign), distance_upper_bound=np.nextafter(limit, np.inf))[0]
        source_distance = mesh_tree.query(source_sample * np.array(sign),
                                          distance_upper_bound=np.nextafter(limit, np.inf))[0]
        medians = [float(np.median(value)) for value in (mesh_distance, source_distance)]
        best = min(medians)
        residuals[name] = best if np.isfinite(best) else None
        diagnostics[name] = {"mesh_to_source_median": medians[0] if np.isfinite(medians[0]) else None,
                             "source_to_mesh_median": medians[1] if np.isfinite(medians[1]) else None,
                             "mesh_supported_fraction": float(np.mean(mesh_distance <= limit)),
                             "source_supported_fraction": float(np.mean(source_distance <= limit))}
        if best <= limit:
            candidates.append(name)
    registration_estimate = None
    if len(candidates) != 1:
        # Counts and distances only, so the refusal can be read without the mesh or the splat.
        failure = {"schema_version": "atlas_mesh_registration_failure.v1",
                   "reason": "atlas_mesh_not_registered_to_splat" if not candidates else "atlas_mesh_frame_ambiguous",
                   "frame_support": diagnostics, "median_residual_by_frame": residuals,
                   "registration_distance_limit": limit, "voxel_size": voxel, "median_mesh_edge": edge,
                   "registration_limit_basis": "max(3 draft voxels, 1.5 median mesh edges)",
                   "mesh_vertices": int(len(vertices)), "mesh_faces": int(len(faces)),
                   "splat_points": int(len(splat.xyz)),
                   "mesh_bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
                   "splat_bounds": [splat.xyz.min(axis=0).astype(float).tolist(),
                                    splat.xyz.max(axis=0).astype(float).tolist()],
                   "source_glb_sha256": _sha256_file(glb), "splat_manifest_digest": splat_manifest["digest"]}
        failure["digest"] = canonical_digest(failure, digest_field="digest")
        Path(output_root).mkdir(parents=True, exist_ok=True)
        (Path(output_root) / FAILURE_RECORD).write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        unbounded = {}
        for name in (candidates or list(GLB_FRAMES)):
            sign = np.array(GLB_FRAMES[name])
            unbounded[name] = min(float(np.median(tree.query(sample * sign)[0])),
                                  float(np.median(mesh_tree.query(source_sample * sign)[0])))
        finite = {name: value for name, value in unbounded.items() if np.isfinite(value)}
        if not finite:
            raise _refusal(failure["reason"], glb, splat_manifest)
        frame = min(finite, key=lambda name: (finite[name], name))
        registration_estimate = {"code": COLLIDER_ESTIMATE_FLAG, "reason": failure["reason"], "glb_frame": frame,
                                 "median_residual": finite[frame], "registration_distance_limit": limit,
                                 "candidate_frames": sorted(candidates), "failure_record_digest": failure["digest"],
                                 "basis": "smallest_median_residual_frame_supported_faces_only",
                                 "claim": "development_only_estimate"}
    else:
        frame = candidates[0]
    del mesh_tree  # only the frame search reads it
    vertices = vertices * np.array(GLB_FRAMES[frame])
    support = max(SUPPORT_VOXELS * voxel, SUPPORT_MESH_EDGES * edge)
    vertex_distance, vertex_nearest = tree.query(vertices, distance_upper_bound=np.nextafter(support, np.inf))
    centroid_distance = per_face(faces, lambda tri: tree.query(
        tri.mean(axis=1), distance_upper_bound=np.nextafter(support, np.inf))[0])
    kept = faces[np.all(vertex_distance[faces] <= support, axis=1) & (centroid_distance <= support)]
    supported = kept
    depth_filter = None
    if reference_frames is not None:
        mask, depth_filter = mesh_depth_free_space_filter(
            vertices, faces=kept, frames=reference_frames, absolute_tolerance=support)
        depth_filter["reference_pose_digest"] = reference_pose_digest
        depth_filter["digest"] = canonical_digest(depth_filter, digest_field="digest")
        kept = kept[mask]
    support_estimate = None
    if not len(kept):
        if registration_estimate is not None:
            # An estimated frame with nothing near observed geometry: no face of it can be trusted,
            # and the mesh is as unregistered as it was.
            raise _refusal(registration_estimate["reason"], glb, splat_manifest)
        # A registered frame: its faces lie on the splat's frame. Keep those the support test kept,
        # else every registered face, and say so.
        kept = supported if len(supported) else faces
        support_estimate = {"code": COLLIDER_ESTIMATE_FLAG, "reason": "atlas_mesh_no_supported_faces",
                            "basis": ("support_filtered_faces_before_depth_filter" if len(supported)
                                      else "every_registered_face"),
                            "claim": "development_only_estimate"}
    used, remapped = np.unique(kept, return_inverse=True)
    nearest = vertex_nearest[used]
    colors = np.clip((0.5 + SH_C0 * splat.f_dc[nearest]) * 255.0, 0, 255).round()
    out = write_glb_triangles(output_root / "atlas_mesh_supported.glb", vertices[used], remapped.reshape(-1, 3), colors)

    def area(rows: np.ndarray) -> float:
        # One sum over every face's own area, as over the whole (faces, 3, 3) array.
        return float(0.5 * per_face(rows, lambda tri: np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)).sum())

    record = {"schema_version": SCHEMA_VERSION, "source_glb_sha256": _sha256_file(glb),
              "splat_manifest_digest": splat_manifest["digest"], "glb_frame": frame,
              "median_residual_by_frame": residuals, "frame_support": diagnostics,
              "registration_distance_limit": limit, "registration_basis": "majority_overlap_in_either_direction",
              "registration_limit_basis": "max(3 draft voxels, 1.5 median mesh edges)",
              "support_basis": "all_vertices_and_centroid", "voxel_size": voxel, "median_mesh_edge": edge,
              "support_distance": support, "faces_in": int(len(faces)), "faces_kept": int(len(kept)),
              "area_kept_fraction": area(kept) / area(faces),
              "outputs": [{"role": "supported_mesh_glb", "path": out.name, "bytes": out.stat().st_size,
                           "sha256": _sha256_file(out)}],
              "world_frame": "atlas_world_rub_y_up", "colour_source": "nearest_observed_gaussian",
              "units": splat_manifest.get("units"), "metric_measurement_proven": False,
              "claim_scope": "development_only"}
    if depth_filter is not None:
        record["depth_free_space_filter"] = depth_filter
    if registration_estimate is not None:
        record["registration_estimate"] = registration_estimate
    if support_estimate is not None:
        record["support_estimate"] = support_estimate
    record["digest"] = canonical_digest(record, digest_field="digest")
    (output_root / "atlas_mesh_registration.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
