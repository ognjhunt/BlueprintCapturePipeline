"""Register an Atlas splats2Mesh GLB to the observed splat and keep supported faces.

The live splats2Mesh export is not in the frame of the cameras and splat it was
given: its vertices match the splat only after a 180 degree rotation about X
(an OpenCV-style Y-down world). It also closes unobserved regions with smooth
sheets that no camera saw, which would become phantom collision walls. This
stage measures which declared frame fits the splat, refuses a mesh that fits
neither, keeps only faces near observed geometry, colours vertices from the
nearest observed Gaussian instead of the provider texture, and writes an
untextured GLB in the Atlas world frame. Everything stays development_only.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .atlas_depth_splat import SH_C0
from .decision_evidence_contracts import canonical_digest
from .gaussian_splat_decode import read_standard_3dgs_ply
from .local_reconstruction_adapters import _sha256_file

SCHEMA_VERSION = "atlas_mesh_registration.v2"
GLB_FRAMES = {"atlas_world": (1.0, 1.0, 1.0), "atlas_world_rotated_180_x": (1.0, -1.0, -1.0)}
MAX_MEDIAN_RESIDUAL_VOXELS = 3.0
SUPPORT_VOXELS = 4.0
SUPPORT_MESH_EDGES = 2.0
#: A mesh cannot sit closer to the splat than its own triangles allow: nearest-vertex distances grow
#: with the provider's mesh resolution, which the draft's fusion grid does not set. capture-447's
#: registered refined mesh measured a median residual of 0.81 median edges (0.0178 against a 0.0219
#: edge); the library-aisle mesh (613,408 faces against capture-447's 1.89 M) failed the grid-only
#: tolerance. A wrong declared frame misses by scene-scale distances, far beyond either bound.
REGISTRATION_MESH_EDGES = 1.5
FAILURE_RECORD = "atlas_mesh_registration_failure.json"
REGISTRATION_SAMPLES = 200_000
#: Faces gathered into (faces, 3, 3) at once. A room's mesh has millions of faces (capture-447's: 6.5 M), and
#: the whole array is 470 MB of float64 per copy; registration held two and its area temporaries (2.8 GB).
FACE_BLOCK = 1 << 20
_COMPONENTS = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}


def mesh_depth_free_space_filter(
    triangles: np.ndarray, *, frames: Sequence[Mapping[str, Any]],
    absolute_tolerance: float, relative_tolerance: float = 0.02, faces: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep surfaces unless multiple saved views contradict the entire face.

    The three vertices and centroid must each lie in front of confident depth
    in two views. Any agreeing sample protects the face. Occluded, unseen and
    invalid-depth samples never vote for deletion. This is a derived geometry
    screen, not native contact or captured metric truth.

    With ``faces``, ``triangles`` is the (finite) vertex array they index, and
    each batch of faces is gathered on its own instead of all of them at once.
    """
    import hashlib
    from .atlas_depth_splat import _rotation_matrix, _seen_ratio

    triangles = np.asarray(triangles, dtype=np.float64)
    if faces is None:
        shape_valid = triangles.ndim == 3 and triangles.shape[1:] == (3, 3)
        count = len(triangles)
    else:
        faces = np.asarray(faces)
        shape_valid = (triangles.ndim == 2 and triangles.shape[1:] == (3,) and faces.ndim == 2
                       and faces.shape[1:] == (3,) and np.issubdtype(faces.dtype, np.integer)
                       and (not len(faces) or (faces.min() >= 0 and faces.max() < len(triangles))))
        count = len(faces) if shape_valid else 0
    if (not shape_valid
            or not np.isfinite(triangles).all()
            or not np.isfinite(absolute_tolerance) or absolute_tolerance <= 0
            or not np.isfinite(relative_tolerance) or not 0 <= relative_tolerance < 1):
        raise ValueError("atlas_mesh_depth_filter_geometry_or_tolerance_invalid")
    identifiers = [str(f.get("frame_id") or "") for f in frames]
    if not frames or not all(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("atlas_mesh_depth_filter_reference_identity_invalid")
    references, voting_frames, seen_references = [], [], set()
    for frame in frames:
        camera, k = frame["extrinsics"], frame["intrinsics"]
        if (camera.get("coordinateSystem", "rub") != "rub"
                or np.shape(frame["depth"]) != (k["height"], k["width"])
                or np.shape(frame["confidence"]) != np.shape(frame["depth"])
                or not np.allclose(frame["position"], camera["position"], rtol=0, atol=1e-12)
                or not np.allclose(frame["rotation"], _rotation_matrix(camera["quaternion"]), rtol=0, atol=1e-12)):
            raise ValueError("atlas_mesh_depth_filter_camera_binding_invalid")
        arrays = {}
        for name in ("depth", "confidence"):
            value = np.ascontiguousarray(frame[name])
            arrays[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                           "sha256": "sha256:" + hashlib.sha256(value.tobytes()).hexdigest()}
        reference = {"intrinsics": dict(frame["intrinsics"]),
                     "extrinsics": dict(frame["extrinsics"]), **arrays}
        identity = canonical_digest(reference)
        references.append({"frame_id": frame["frame_id"], **reference})
        if identity not in seen_references:
            voting_frames.append(frame)
            seen_references.add(identity)
    keep = np.ones(count, dtype=bool)
    for start in range(0, count, 4096):
        batch = triangles[start:start + 4096] if faces is None else triangles[faces[start:start + 4096]]
        samples = np.concatenate([batch, batch.mean(axis=1)[:, None]], axis=1).reshape(-1, 3)
        clear = np.zeros(len(samples), dtype=np.uint32)
        agreement = np.zeros(len(samples), dtype=bool)
        for frame in voting_frames:
            local = (samples - frame["position"]) @ frame["rotation"]
            distance = -local[:, 2]
            ratio = _seen_ratio(frame, samples)
            residual = (ratio - 1.0) * distance
            valid = np.isfinite(residual) & (distance > 0)
            tolerance = np.maximum(absolute_tolerance, distance * relative_tolerance)
            clear += valid & (residual > tolerance)
            agreement |= valid & (np.abs(residual) <= tolerance)
        contradicted = np.all(clear.reshape(-1, 4) >= 2, axis=1)
        protected = np.any(agreement.reshape(-1, 4), axis=1)
        keep[start:start + len(batch)] = ~contradicted | protected
    report = {"schema_version": "atlas_mesh_depth_free_space_filter.v1",
              "method": "all_vertices_and_centroid_two_views_no_agreeing_sample",
              "absolute_tolerance_provider_units": float(absolute_tolerance),
              "relative_depth_tolerance": float(relative_tolerance),
              "references": references, "distinct_reference_count": len(voting_frames),
              "faces_in": count,
              "faces_kept": int(keep.sum()), "faces_removed": int((~keep).sum()),
              "native_collision_qualified": False, "metric_measurement_proven": False}
    report["digest"] = canonical_digest(report, digest_field="digest")
    return keep, report


def read_glb_triangles(path: Path) -> tuple[np.ndarray, np.ndarray]:
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


def write_glb_triangles(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> Path:
    """Minimal glTF 2.0 binary: positions, normalized RGBA8 vertex colours, uint32 indices."""
    position = np.ascontiguousarray(vertices, dtype="<f4")
    rgba = np.concatenate([colors.astype(np.uint8), np.full((len(colors), 1), 255, np.uint8)], axis=1)
    index = np.ascontiguousarray(faces, dtype="<u4")
    # Each array is written as it is, never joined into one buffer: a room's mesh is hundreds of MB of GLB.
    chunks, views, offset = [], [], 0
    for payload, target in ((position, 34962), (rgba, 34962), (index, 34963)):
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": payload.nbytes, "target": target})
        chunks.append((payload, -payload.nbytes % 4))
        offset += payload.nbytes + chunks[-1][1]
    gltf = {"asset": {"version": "2.0", "generator": "blueprint atlas_mesh_registration"}, "scene": 0,
            "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}], "buffers": [{"byteLength": offset}],
            "bufferViews": views, "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1},
                                                              "indices": 2, "mode": 4}]}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(position), "type": "VEC3",
                           "min": position.min(axis=0).tolist(), "max": position.max(axis=0).tolist()},
                          {"bufferView": 1, "componentType": 5121, "normalized": True, "count": len(rgba),
                           "type": "VEC4"},
                          {"bufferView": 2, "componentType": 5125, "count": index.size, "type": "SCALAR"}]}
    text = json.dumps(gltf, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<4sII", b"glTF", 2, 28 + len(text) + offset)
                     + struct.pack("<I4s", len(text), b"JSON") + text + struct.pack("<I4s", offset, b"BIN\0"))
        for payload, padding in chunks:
            stream.write(memoryview(np.ascontiguousarray(payload)).cast("B"))
            stream.write(b"\0" * padding)
    return path


def register_atlas_mesh(*, glb: Path, splat_root: Path, output_root: Path,
                        reference_frames: Sequence[Mapping[str, Any]] | None = None,
                        reference_pose_digest: str | None = None) -> dict[str, Any]:
    """Fit the GLB to the observed splat, trim unsupported faces and write the result."""
    from scipy.spatial import cKDTree

    from .atlas_native_splat import load_splat_manifest
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
    vertices, faces = read_glb_triangles(glb)

    def per_face(rows: np.ndarray, value) -> np.ndarray:
        """``value`` of ``vertices[rows]``, gathered ``FACE_BLOCK`` faces at a time: each face's own arithmetic."""
        return np.concatenate([value(vertices[rows[start:start + FACE_BLOCK]])
                               for start in range(0, len(rows), FACE_BLOCK)] or [value(vertices[rows[:0]])])

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
        raise ValueError(failure["reason"])
    frame = candidates[0]
    del mesh_tree  # only the frame search reads it
    vertices = vertices * np.array(GLB_FRAMES[frame])
    support = max(SUPPORT_VOXELS * voxel, SUPPORT_MESH_EDGES * edge)
    vertex_distance, vertex_nearest = tree.query(vertices, distance_upper_bound=np.nextafter(support, np.inf))
    centroid_distance = per_face(faces, lambda tri: tree.query(
        tri.mean(axis=1), distance_upper_bound=np.nextafter(support, np.inf))[0])
    kept = faces[np.all(vertex_distance[faces] <= support, axis=1) & (centroid_distance <= support)]
    depth_filter = None
    if reference_frames is not None:
        mask, depth_filter = mesh_depth_free_space_filter(
            vertices, faces=kept, frames=reference_frames, absolute_tolerance=support)
        depth_filter["reference_pose_digest"] = reference_pose_digest
        depth_filter["digest"] = canonical_digest(depth_filter, digest_field="digest")
        kept = kept[mask]
    if not len(kept):
        raise ValueError("atlas_mesh_no_supported_faces")
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
    record["digest"] = canonical_digest(record, digest_field="digest")
    (output_root / "atlas_mesh_registration.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
