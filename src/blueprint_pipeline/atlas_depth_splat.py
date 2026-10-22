"""Fuse Atlas posed RGBD into a draft Gaussian splat that splats2Mesh can mesh.

The public Atlas beta API poses images and returns per-frame depth, but it has no
task that creates a splat, and ``splats2Mesh`` (its only mesh export) accepts
nothing else. This stage back-projects every returned depth grid with its
returned camera, keeps finite, confident pixels away from depth discontinuities,
merges them on a voxel grid sized from the capture's own pixel footprint, and
writes one surface-aligned Gaussian per voxel as a standard 3DGS PLY and an SPZ
v2 upload. Cross-view depth agreement is measured before fusion, so poses and
depths that do not describe one surface fail closed instead of smearing.

The splat is a depth-fused draft without photometric optimization. Positions
stay in Atlas world units on RUB axes (Y up, gravity-levelled), the frame SPZ
stores natively, and remain development_only until a published size calibrates
the scale.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .decision_evidence_contracts import canonical_digest
from .gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from .local_reconstruction_adapters import _sha256_file

SCHEMA_VERSION = "atlas_depth_splat.v1"
POSE_RESULT = "atlas_posed_rgbd_result.json"
SH_C0 = 0.28209479177387814
# A 5 % depth jump between neighbouring pixels is a discontinuity: a continuous
# surface only reaches it beyond ~88 degrees of obliquity at Atlas's focal length.
EDGE_RELATIVE_JUMP = 0.05
MAX_MEDIAN_CROSS_VIEW_RESIDUAL = 0.05
CROSS_VIEW_SAMPLES = 20_000
CROSS_VIEW_NEIGHBOURS = 8
_KEY_BITS = 21
# Tangent sigma per voxel edge keeps neighbouring discs overlapping without gaps.
DISC_SIGMA_PER_VOXEL = 0.7
DISC_THICKNESS_RATIO = 0.1
SPLATS2MESH_MAX_RENDER = 1024
SPLATS2MESH_MIN_RENDER = 64
SPLATS2MESH_MAX_CAMERAS = 512
EXR_MAGIC = b"\x76\x2f\x31\x01"
_EXR_LINES_PER_BLOCK = {0: 1, 2: 1, 3: 16}  # none, ZIPS, ZIP
_EXR_PIXEL_TYPES = {1: "<f2", 2: "<f4"}
SPZ_MAGIC = 0x5053474E
SPZ_VERSION = 2
SPZ_FRACTIONAL_BITS = 12
SPZ_COLOR_SCALE = 0.15
PLY_PROPERTIES = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
                  "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")


def _rotation_matrix(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-6:
        raise ValueError("atlas_depth_splat_camera_rotation_invalid")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_exr_depth(path: Path) -> np.ndarray:
    """One-channel scanline OpenEXR (uncompressed, ZIPS or ZIP) as float32, NaN preserved.

    Atlas returns depth in exactly this shape. Production's long-lived virtualenv
    carries an OpenCV build that cannot decode EXR, so this stays dependency-free
    and refuses every other layout rather than guessing.
    """
    data = path.read_bytes()
    if data[:4] != EXR_MAGIC or struct.unpack_from("<I", data, 4)[0] != 2:
        raise ValueError("atlas_depth_exr_layout_unsupported")
    position, header = 8, {}
    while data[position] != 0:
        name_end = data.index(b"\0", position)
        kind_end = data.index(b"\0", name_end + 1)
        size = struct.unpack_from("<I", data, kind_end + 1)[0]
        header[data[position:name_end].decode()] = data[kind_end + 5:kind_end + 5 + size]
        position = kind_end + 5 + size
    position += 1
    channels = header.get("channels", b"")
    name_end = channels.find(b"\0")
    if name_end < 0 or len(channels) != name_end + 18 or channels[-1] != 0:
        raise ValueError("atlas_depth_exr_channels_unsupported")
    pixel_type, _, x_sampling, y_sampling = struct.unpack_from("<iB3xii", channels, name_end + 1)
    compression = header.get("compression", b"\xff")[0]
    if (pixel_type not in _EXR_PIXEL_TYPES or (x_sampling, y_sampling) != (1, 1)
            or compression not in _EXR_LINES_PER_BLOCK or header.get("lineOrder", b"\0") != b"\0"):
        raise ValueError("atlas_depth_exr_encoding_unsupported")
    x_min, y_min, x_max, y_max = struct.unpack("<iiii", header["dataWindow"])
    width, height = x_max - x_min + 1, y_max - y_min + 1
    dtype = np.dtype(_EXR_PIXEL_TYPES[pixel_type])
    lines = _EXR_LINES_PER_BLOCK[compression]
    blocks = -(-height // lines)
    offsets = struct.unpack_from(f"<{blocks}Q", data, position)
    rows = []
    for block, offset in enumerate(offsets):
        y, size = struct.unpack_from("<ii", data, offset)
        count = min(lines, height - block * lines)
        expected = count * width * dtype.itemsize
        if y != y_min + block * lines or offset + 8 + size > len(data):
            raise ValueError("atlas_depth_exr_block_invalid")
        payload = data[offset + 8:offset + 8 + size]
        if size != expected:
            delta = np.frombuffer(zlib.decompress(payload), dtype=np.uint8).astype(np.int64)
            if len(delta) != expected:
                raise ValueError("atlas_depth_exr_block_invalid")
            delta[1:] -= 128
            interleaved = (np.cumsum(delta) & 0xFF).astype(np.uint8)
            raw = np.empty(expected, np.uint8)
            raw[0::2], raw[1::2] = interleaved[:(expected + 1) // 2], interleaved[(expected + 1) // 2:]
            payload = raw.tobytes()
        rows.append(np.frombuffer(payload, dtype=dtype).reshape(count, width))
    return np.concatenate(rows).astype(np.float32)


def posed_frame(*, frame_id: str, rgb: np.ndarray, depth: np.ndarray, confidence: np.ndarray,
                camera: Mapping[str, Any]) -> dict[str, Any]:
    """Bind one returned image/depth/confidence bundle to the camera Atlas returned with it."""
    extrinsics, intrinsics = camera["extrinsics"], camera["intrinsics"]
    if extrinsics.get("coordinateSystem", "rub") != "rub":
        raise ValueError("atlas_depth_splat_camera_axes_unsupported")
    grid = (int(intrinsics["height"]), int(intrinsics["width"]))
    confidence = np.asarray(confidence)
    if np.shape(depth) != grid or confidence.shape != grid or np.shape(rgb) != (*grid, 3):
        raise ValueError("atlas_depth_splat_grid_mismatch")
    scale = float(np.iinfo(confidence.dtype).max) if confidence.dtype.kind == "u" else 1.0
    return {"frame_id": frame_id, "rgb": np.asarray(rgb, dtype=np.uint8),
            "depth": np.asarray(depth, dtype=np.float32), "confidence": confidence.astype(np.float32) / scale,
            "intrinsics": dict(intrinsics), "extrinsics": dict(extrinsics),
            "rotation": _rotation_matrix(extrinsics["quaternion"]),
            "position": np.asarray(extrinsics["position"], dtype=np.float64)}


def load_posed_rgbd(pose_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a retained images2PosedRGBD result and its digest-bound downloads."""
    result = json.loads((pose_root / POSE_RESULT).read_text())
    if result.get("digest") != canonical_digest(result, digest_field="digest"):
        raise ValueError("atlas_depth_splat_pose_result_changed")
    returned = result["operation"]["response"]["frames"]
    usage = result["operation_receipt"]["input_usage"]["frames"]
    if not returned or len(returned) != len(usage):
        raise ValueError("atlas_depth_splat_frame_count_mismatch")
    import cv2

    frames = []
    for frame, row in zip(returned, usage):
        paths = {}
        for output in row["outputs"]:
            path = pose_root / output["path"]
            if path.is_symlink() or not path.is_file() or _sha256_file(path) != output["sha256"]:
                raise ValueError("atlas_depth_splat_output_changed")
            paths[output["role"]] = path
        if not {"image", "depth", "confidence"} <= set(paths):
            raise ValueError("atlas_depth_splat_output_missing")
        image = cv2.imread(str(paths["image"]), cv2.IMREAD_COLOR)
        depth = read_exr_depth(paths["depth"])
        confidence = cv2.imread(str(paths["confidence"]), cv2.IMREAD_UNCHANGED)
        if image is None or depth is None or confidence is None or depth.ndim != 2 or confidence.ndim != 2:
            raise ValueError("atlas_depth_splat_output_undecodable")
        frames.append(posed_frame(frame_id=row["frame_id"], rgb=image[..., ::-1], depth=depth,
                                  confidence=confidence, camera=frame["camera"]))
    return result, frames


def surface_samples(frame: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """World points, colours, confidences and camera-facing normals of usable pixels."""
    k = frame["intrinsics"]
    depth = np.where((frame["confidence"] > 0) & (frame["depth"] > 0), frame["depth"], np.nan).astype(np.float64)
    center = depth[1:-1, 1:-1]
    with np.errstate(invalid="ignore"):
        keep = np.isfinite(center)
        for neighbour in (depth[1:-1, 2:], depth[1:-1, :-2], depth[2:, 1:-1], depth[:-2, 1:-1]):
            keep &= np.abs(neighbour - center) <= EDGE_RELATIVE_JUMP * center
    u = (np.arange(depth.shape[1]) + 0.5 - k["cx"]) / k["fx"]
    v = (np.arange(depth.shape[0]) + 0.5 - k["cy"]) / k["fy"]
    world = np.stack([u[None, :] * depth, -v[:, None] * depth, -depth], axis=-1) @ frame["rotation"].T
    world += frame["position"]
    normal = np.cross(world[1:-1, 2:] - world[1:-1, :-2], world[2:, 1:-1] - world[:-2, 1:-1])
    length = np.linalg.norm(normal, axis=-1)
    with np.errstate(invalid="ignore"):
        keep &= length > 0
    points = world[1:-1, 1:-1][keep]
    normal = normal[keep] / length[keep][:, None]
    normal[np.einsum("ij,ij->i", normal, frame["position"] - points) < 0] *= -1.0
    return {"points": points, "normals": normal,
            "colors": frame["rgb"][1:-1, 1:-1][keep].astype(np.float64) / 255.0,
            "weights": frame["confidence"][1:-1, 1:-1][keep].astype(np.float64),
            "footprint": center[keep] / math.sqrt(k["fx"] * k["fy"])}


def _subsample(sample: Mapping[str, np.ndarray]) -> dict[str, Any]:
    count = len(sample["points"])
    pick = np.linspace(0, count - 1, min(CROSS_VIEW_SAMPLES, count)).astype(np.int64)
    return {"points": sample["points"][pick], "footprint": sample["footprint"][pick], "usable": count}


def cross_view_agreement(frames: Sequence[Mapping[str, Any]],
                         samples: Sequence[Mapping[str, np.ndarray]]) -> dict[str, Any]:
    """Disagreement between each frame's points and the depth its nearest views returned.

    A target that sees a nearer surface at the projected pixel is occluding the point,
    which is not a disagreement; a target that sees past the point still counts.
    """
    centres = np.array([frame["position"] for frame in frames])
    pairs, residuals = [], []
    for index, (source, sample) in enumerate(zip(frames, samples)):
        picked = sample["points"]
        if not len(picked):
            continue
        order = np.argsort(np.linalg.norm(centres - centres[index], axis=1), kind="stable")
        for target in (frames[i] for i in order[order != index][:CROSS_VIEW_NEIGHBOURS]):
            k = target["intrinsics"]
            local = (picked - target["position"]) @ target["rotation"]
            planar = -local[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                column = np.floor(local[:, 0] / planar * k["fx"] + k["cx"])
                row = np.floor(-local[:, 1] / planar * k["fy"] + k["cy"])
                inside = (planar > 0) & (column >= 0) & (column < k["width"]) & (row >= 0) & (row < k["height"])
            observed = np.full(len(picked), np.nan)
            index_pair = (row[inside].astype(np.int64), column[inside].astype(np.int64))
            observed[inside] = np.where(target["confidence"][index_pair] > 0, target["depth"][index_pair], np.nan)
            seen = np.isfinite(observed) & (observed > 0)
            visible = seen & (observed >= planar * (1.0 - MAX_MEDIAN_CROSS_VIEW_RESIDUAL))
            if visible.sum() < 0.01 * len(picked):
                continue
            relative = np.abs(observed[visible] - planar[visible]) / planar[visible]
            residuals.append(relative)
            pairs.append({"source": source["frame_id"], "target": target["frame_id"],
                          "overlap_samples": int(visible.sum()), "occluded_samples": int((seen & ~visible).sum()),
                          "median_relative_residual": float(np.median(relative))})
    pooled = float(np.median(np.concatenate(residuals))) if residuals else None
    return {"pairs": pairs, "median_relative_residual": pooled, "limit": MAX_MEDIAN_CROSS_VIEW_RESIDUAL}


def _voxel_keys(points: np.ndarray, voxel: float) -> np.ndarray:
    cells = np.floor(points / voxel).astype(np.int64) + (1 << (_KEY_BITS - 1))
    if cells.size and (cells.min() < 0 or cells.max() >= 1 << _KEY_BITS):
        raise ValueError("atlas_depth_splat_scene_exceeds_voxel_grid")
    return (cells[:, 0] << (2 * _KEY_BITS)) | (cells[:, 1] << _KEY_BITS) | cells[:, 2]


class VoxelSums:
    """Running confidence-weighted sums per voxel, so memory follows voxels rather than pixels."""

    def __init__(self, voxel: float):
        self.voxel = voxel
        self.keys = np.empty(0, np.int64)
        self.sums = np.empty((0, 11))  # weight, weighted xyz, rgb and normal, sample count

    def add(self, sample: Mapping[str, np.ndarray]) -> None:
        weights = sample["weights"][:, None]
        values = np.concatenate([self.sums, np.concatenate([
            weights, sample["points"] * weights, sample["colors"] * weights, sample["normals"] * weights,
            np.ones_like(weights)], axis=1)])
        keys, inverse = np.unique(np.concatenate([self.keys, _voxel_keys(sample["points"], self.voxel)]),
                                  return_inverse=True)
        inverse = inverse.reshape(-1)
        self.sums = np.stack([np.bincount(inverse, weights=values[:, column], minlength=len(keys))
                              for column in range(values.shape[1])], axis=1)
        self.keys = keys

    def fused(self) -> dict[str, np.ndarray]:
        total = self.sums[:, :1]
        normal = self.sums[:, 7:10] / total
        agreement = np.linalg.norm(normal, axis=1)
        return {"xyz": self.sums[:, 1:4] / total, "rgb": self.sums[:, 4:7] / total,
                "normal": normal / np.maximum(agreement, 1e-12)[:, None], "normal_agreement": agreement,
                "confidence": total[:, 0] / self.sums[:, 10], "observations": self.sums[:, 10]}


def depth_fused_splat(fused: Mapping[str, np.ndarray], voxel: float) -> SplatData:
    """One Gaussian per voxel: a thin disc on the fused normal, isotropic where normals disagree."""
    count = len(fused["xyz"])
    normal = fused["normal"] * np.where(fused["normal"][:, 2:3] < 0, -1.0, 1.0)
    quats = np.stack([1.0 + normal[:, 2], -normal[:, 1], normal[:, 0], np.zeros(count)], axis=1)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    sigma = DISC_SIGMA_PER_VOXEL * voxel
    scales = np.full((count, 3), math.log(sigma))
    scales[fused["normal_agreement"] >= 0.5, 2] = math.log(sigma * DISC_THICKNESS_RATIO)
    opacity = 0.5 + 0.49 * np.clip(fused["confidence"], 0.0, 1.0)
    return SplatData(count=count, xyz=fused["xyz"].astype(np.float32),
                     opacity=np.log(opacity / (1.0 - opacity)).astype(np.float32),
                     f_dc=((fused["rgb"] - 0.5) / SH_C0).astype(np.float32), scales=scales.astype(np.float32),
                     quats=quats.astype(np.float32), properties=PLY_PROPERTIES)


def fuse_posed_frames(frames: Sequence[Mapping[str, Any]], *,
                      voxel_footprint_multiple: float = 2.0) -> tuple[SplatData, dict[str, Any]]:
    """Gate cross-view agreement, then stream every usable pixel into a draft splat.

    Full-resolution samples exist for one frame at a time; only bounded subsamples and
    per-voxel sums persist, so memory follows the scene's voxels, not its frame count.
    """
    if not math.isfinite(voxel_footprint_multiple) or voxel_footprint_multiple <= 0:
        raise ValueError("atlas_depth_splat_voxel_multiple_invalid")
    samples = [_subsample(surface_samples(frame)) for frame in frames]
    agreement = cross_view_agreement(frames, samples)
    if (agreement["median_relative_residual"] is not None
            and agreement["median_relative_residual"] > MAX_MEDIAN_CROSS_VIEW_RESIDUAL):
        raise ValueError("atlas_depth_splat_cross_view_inconsistent")
    footprints = np.concatenate([sample["footprint"] for sample in samples])
    if not len(footprints):
        raise ValueError("atlas_depth_splat_no_usable_depth")
    voxel = float(voxel_footprint_multiple * np.median(footprints))
    sums = VoxelSums(voxel)
    for frame in frames:
        sums.add(surface_samples(frame))
    fused = sums.fused()
    report = {"frames": [{"frame_id": frame["frame_id"], "usable_pixel_fraction": sample["usable"] / frame["depth"].size}
                         for frame, sample in zip(frames, samples)],
              "cross_view_agreement": agreement, "voxel_size": voxel,
              "voxel_footprint_multiple": voxel_footprint_multiple,
              "median_observations_per_gaussian": float(np.median(fused["observations"]))}
    return depth_fused_splat(fused, voxel), report


def _uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.round(values), 0, 255).astype(np.uint8)


def encode_spz(splat: SplatData) -> bytes:
    """Niantic SPZ v2 with degree-0 colour, gzip-wrapped and byte-reproducible."""
    fixed = np.round(splat.xyz.astype(np.float64) * (1 << SPZ_FRACTIONAL_BITS)).astype(np.int64)
    if np.any(np.abs(fixed) >= 1 << 23):
        raise ValueError("atlas_depth_splat_spz_position_out_of_range")
    positions = ((fixed[..., None] & 0xFFFFFF) >> np.array([0, 8, 16])) & 0xFF
    quats = splat.quats.astype(np.float64)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    quats *= np.where(quats[:, :1] < 0, -1.0, 1.0)
    payload = b"".join([
        struct.pack("<IIIBBBB", SPZ_MAGIC, SPZ_VERSION, splat.count, 0, SPZ_FRACTIONAL_BITS, 0, 0),
        positions.astype(np.uint8).tobytes(),
        _uint8(255.0 / (1.0 + np.exp(-splat.opacity.astype(np.float64)))).tobytes(),
        _uint8(splat.f_dc.astype(np.float64) * (SPZ_COLOR_SCALE * 255.0) + 127.5).tobytes(),
        _uint8((splat.scales.astype(np.float64) + 10.0) * 16.0).tobytes(),
        _uint8(quats[:, 1:] * 127.5 + 127.5).tobytes(),
    ])
    return gzip.compress(payload, compresslevel=9, mtime=0)


def decode_spz(data: bytes) -> SplatData:
    raw = gzip.decompress(data)
    magic, version, count, degree, fractional, _flags, _reserved = struct.unpack_from("<IIIBBBB", raw)
    if magic != SPZ_MAGIC or version != SPZ_VERSION or degree != 0 or len(raw) != 16 + count * 19:
        raise ValueError("atlas_depth_splat_spz_invalid")
    body = np.frombuffer(raw, dtype=np.uint8, offset=16).astype(np.int64)
    position, rest = body[:count * 9].reshape(count, 3, 3), body[count * 9:]
    fixed = position[..., 0] | (position[..., 1] << 8) | (position[..., 2] << 16)
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
    alpha = np.clip(rest[:count] / 255.0, 1e-6, 1 - 1e-6)
    colors, scales, rotation = (rest[count + 3 * count * index:count + 3 * count * (index + 1)].reshape(count, 3)
                                for index in range(3))
    xyz = rotation / 127.5 - 1.0
    w = np.sqrt(np.clip(1.0 - np.sum(xyz * xyz, axis=1), 0.0, None))
    return SplatData(count=count, xyz=(fixed / float(1 << fractional)).astype(np.float32),
                     opacity=np.log(alpha / (1 - alpha)).astype(np.float32),
                     f_dc=((colors / 255.0 - 0.5) / SPZ_COLOR_SCALE).astype(np.float32),
                     scales=(scales / 16.0 - 10.0).astype(np.float32),
                     quats=np.column_stack([w, xyz]).astype(np.float32), properties=PLY_PROPERTIES)


def splats2mesh_task_spec(*, frames: Sequence[Mapping[str, Any]], spz: Path,
                          source_video_digest: str) -> dict[str, Any]:
    """Mesh the splat from the capture's own cameras, resized to Atlas's render limits."""
    grids = {(int(frame["intrinsics"]["width"]), int(frame["intrinsics"]["height"])) for frame in frames}
    if len(grids) != 1:
        raise ValueError("atlas_depth_splat_camera_grids_differ")
    ((width, height),) = grids
    factor = max(min(1.0, SPLATS2MESH_MAX_RENDER / max(width, height)), SPLATS2MESH_MIN_RENDER / min(width, height))
    render = (round(width * factor), round(height * factor))
    if max(render) > SPLATS2MESH_MAX_RENDER:
        raise ValueError("atlas_depth_splat_render_aspect_unsupported")
    sx, sy = render[0] / width, render[1] / height
    chosen = list(frames)
    if len(chosen) > SPLATS2MESH_MAX_CAMERAS:
        chosen = [chosen[i] for i in np.linspace(0, len(chosen) - 1, SPLATS2MESH_MAX_CAMERAS).round().astype(int)]
    cameras = [{"intrinsics": {"fx": k["fx"] * sx, "fy": k["fy"] * sy, "cx": k["cx"] * sx, "cy": k["cy"] * sy,
                               "width": render[0], "height": render[1]},
                "extrinsics": {key: frame["extrinsics"][key] for key in ("position", "quaternion", "coordinateSystem")}}
               for frame in chosen for k in (frame["intrinsics"],)]
    return {"task": "splats2Mesh", "source_video_digest": source_video_digest,
            "request": {"splatAsset": {"assetId": "upload:splat"}, "cameras": cameras, "textureMode": "image_texture"},
            "uploads": {"splat": {"path": str(spz.resolve()), "sha256": _sha256_file(spz)}}}


def build_depth_splat(pose_root: Path, output_root: Path, *, voxel_footprint_multiple: float = 2.0) -> dict[str, Any]:
    """Write the draft splat (PLY + SPZ), its manifest and the splats2Mesh task spec."""
    result, frames = load_posed_rgbd(pose_root)
    return write_depth_splat(frames=frames, result=result, output_root=output_root,
                             voxel_footprint_multiple=voxel_footprint_multiple)


def write_depth_splat(*, frames, result, output_root: Path, voxel_footprint_multiple=2.0,
                      generated_result_digests=(), visual_review_digest=None):
    """Shared writer; edited RGBD remains explicitly generated and separately bound."""
    splat, report = fuse_posed_frames(frames, voxel_footprint_multiple=voxel_footprint_multiple)
    data = encode_spz(splat)
    decoded = decode_spz(data)
    if decoded.count != splat.count or np.max(np.abs(decoded.xyz - splat.xyz)) > 1.0 / (1 << SPZ_FRACTIONAL_BITS):
        raise ValueError("atlas_depth_splat_spz_round_trip_failed")
    output_root.mkdir(parents=True, exist_ok=True)
    ply = write_standard_3dgs_ply(splat, output_root / "atlas_depth_splat.ply")
    spz = output_root / "atlas_depth_splat.spz"
    spz.write_bytes(data)
    source_video_digest = result["binding"]["source_video_digest"]
    manifest = {"schema_version": SCHEMA_VERSION, "pose_result_digest": result["digest"],
                "pose_operation_id": result["operation"]["id"], "source_video_digest": source_video_digest,
                **report, "gaussian_count": splat.count,
                "outputs": [{"role": role, "path": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
                            for role, path in (("splat_ply", ply), ("splat_spz", spz))],
                "world_frame": "atlas_world_rub_y_up", "units": result.get("depth_unit"),
                "scale_status": result.get("scale_status"), "splat_kind": "depth_fused_draft",
                "photometric_optimization": False, "generated_content": bool(generated_result_digests),
                **({"generated_result_digests": list(generated_result_digests),
                    "visual_review_digest": visual_review_digest,
                    "appearance_basis": "generated_object_absent_background"} if generated_result_digests else {}),
                "metric_measurement_proven": False, "claim_scope": "development_only"}
    manifest["digest"] = canonical_digest(manifest, digest_field="digest")
    (output_root / "atlas_depth_splat_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    spec = splats2mesh_task_spec(frames=frames, spz=spz, source_video_digest=source_video_digest)
    (output_root / "splats2mesh_task_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fuse Atlas posed RGBD into a draft splat for splats2Mesh")
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--voxel-footprint-multiple", type=float, default=2.0)
    args = parser.parse_args(argv)
    manifest = build_depth_splat(args.pose_root, args.output_root,
                                 voxel_footprint_multiple=args.voxel_footprint_multiple)
    print(json.dumps({"gaussian_count": manifest["gaussian_count"], "voxel_size": manifest["voxel_size"],
                      "median_relative_residual": manifest["cross_view_agreement"]["median_relative_residual"],
                      "digest": manifest["digest"]}))


if __name__ == "__main__":
    main()
