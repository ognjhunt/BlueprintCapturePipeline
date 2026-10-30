"""Fuse Atlas posed RGBD into a draft Gaussian splat that splats2Mesh can mesh.

The public Atlas beta API poses images and returns per-frame depth, but it has no
task that creates a splat, and ``splats2Mesh`` (its only mesh export) accepts
nothing else. This stage back-projects every returned depth grid with its
returned camera, keeps finite, confident pixels away from depth discontinuities,
merges them on a voxel grid sized from the capture's own pixel footprint, and
writes one surface-aligned Gaussian per voxel as a standard 3DGS PLY and an SPZ
v2 upload. Cross-view depth agreement is measured before fusion, so poses and
depths that do not describe one surface fail closed instead of smearing; views
that do agree within a few percent are then rescaled onto their shared surface
before their pixels merge, so one surface fuses as one sheet.

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
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .decision_evidence_contracts import canonical_digest
from .gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from .local_reconstruction_adapters import _sha256_file

SCHEMA_VERSION = "atlas_depth_splat.v1"
FUSION_REVISION = "local-plane-slant-aware-consensus-v4"
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
# A camera's pixel rows land footprint / cos(incidence) apart on a slanted surface;
# beyond about 80 degrees the depth is too oblique to widen a disc further.
MIN_INCIDENCE_COSINE = 0.17
# Atlas depth arrives in plateaus of 5-10 pixels with ~9 mm steps: about 140
# distinct values per frame in a half-float EXR (measured 2026-10-24). Fused at
# pixel resolution, those terraces layer surfaces and tilt per-pixel normals.
DEPTH_SMOOTHING_SIGMA_PIXELS = 4.0
# A local plane replaces a pixel's depth only where it explains the window
# (RMS residual) and stays near the pixel; elsewhere, as across a real
# discontinuity, the pixel keeps its own depth. Terrace steps are ~0.3% RMS.
DEPTH_SMOOTHING_RESIDUAL = 0.015
DEPTH_SMOOTHING_TOLERANCE = 0.03
# Each view's depth carries its own low-frequency error: overlapping Atlas views
# disagreed by a median 1.9 % (measured 2026-10-24), which fused into layered
# sheets about five times thicker than one view's. Each pass rescales every
# frame towards the median of what the views sharing its surface see along its
# pixels; another view's surface votes only within the gate, so occluders never do.
DEPTH_CONSENSUS_PASSES = 3
DEPTH_CONSENSUS_GATE = 0.05
DEPTH_CONSENSUS_STRIDE = 4
DEPTH_CONSENSUS_SIGMA = 2.0  # in consensus-grid pixels
DEPTH_CONSENSUS_MIN_WEIGHT = 0.25
# Voters are the views with the most samples inside the gate, not the nearest
# cameras: 27 of the production export's 37 views were repair views keeping
# 0-5 % of their pixels beside a full view (2026-10-25), and they took most
# full views' nearest slots. Ranking reads this many samples per candidate.
DEPTH_CONSENSUS_RANKING_SAMPLES = 2_000
DISC_THICKNESS_RATIO = 0.1
SPLATS2MESH_MAX_RENDER = 1024
SPLATS2MESH_MIN_RENDER = 64
SPLATS2MESH_MAX_CAMERAS = 512
# splats2Mesh documents no splat limit. It meshed a 2.45M-Gaussian draft
# (2026-10-23) and failed with EXECUTION_FAILED on a 3.6M one (2026-10-24).
SPLATS2MESH_MAX_GAUSSIANS = 2_000_000
_MESH_INPUT_GRID_MULTIPLES = (1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0)
MESH_INPUT_REVISION = "merged-v1"
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


def write_exr_depth(path: Path, depth: np.ndarray) -> Path:
    """Top-left planar FLOAT scanlines; no codec or row-order reinterpretation."""
    pixels = np.ascontiguousarray(depth, dtype='<f4')
    if pixels.ndim != 2 or not all(pixels.shape):
        raise ValueError('atlas_depth_exr_grid_invalid')
    height, width = pixels.shape

    def attribute(name, kind, value):
        return name.encode() + b'\0' + kind.encode() + b'\0' + struct.pack('<I', len(value)) + value

    window = struct.pack('<iiii', 0, 0, width - 1, height - 1)
    header = b''.join([EXR_MAGIC, struct.pack('<I', 2),
        attribute('channels', 'chlist', b'Y\0' + struct.pack('<iB3xii', 2, 0, 1, 1) + b'\0'),
        attribute('compression', 'compression', b'\0'),
        attribute('dataWindow', 'box2i', window), attribute('displayWindow', 'box2i', window),
        attribute('lineOrder', 'lineOrder', b'\0'), attribute('pixelAspectRatio', 'float', struct.pack('<f', 1)),
        attribute('screenWindowCenter', 'v2f', struct.pack('<ff', 0, 0)),
        attribute('screenWindowWidth', 'float', struct.pack('<f', 1)), b'\0'])
    row_size = 8 + width * 4
    offsets = [len(header) + height * 8 + y * row_size for y in range(height)]
    data = header + struct.pack(f'<{height}Q', *offsets) + b''.join(
        struct.pack('<ii', y, width * 4) + pixels[y].tobytes() for y in range(height))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and path.read_bytes() != data):
        raise ValueError('atlas_depth_exr_output_changed')
    path.write_bytes(data)
    return path


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
    return result, posed_rgbd_frames(result, lambda output: pose_root / output["path"])


def posed_rgbd_frames(result: Mapping[str, Any],
                      locate: Callable[[Mapping[str, Any]], Path | None]) -> list[dict[str, Any]]:
    """Decode a pose result's frames; ``locate`` names where each digest-bound output was kept.

    A retained handoff may keep only the result beside its pose and the outputs in
    the download bundle: the receipt's digests, not their paths, bind the bytes.
    """
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
            path = locate(output)
            if path is None or path.is_symlink() or not path.is_file() or _sha256_file(path) != output["sha256"]:
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
    return frames


def smoothed_depth(depth: np.ndarray) -> np.ndarray:
    """Replace terraced depth with a local Gaussian-weighted plane fit within surfaces.

    A 3D plane is linear in inverse depth across a pinhole image, so the fit keeps
    flat and slanted surfaces exact, including at image borders and holes.
    Missing pixels stay missing; a window spanning a real edge keeps raw depth.
    """
    from scipy.ndimage import gaussian_filter

    valid = np.isfinite(depth) & (depth > 0)
    m = valid.astype(np.float64)
    with np.errstate(divide="ignore"):
        d = np.where(valid, 1.0 / depth.astype(np.float64), 0.0)  # inverse depth
    y, x = (axis.astype(np.float64) for axis in np.indices(depth.shape))

    def window(values):
        return gaussian_filter(values, DEPTH_SMOOTHING_SIGMA_PIXELS, mode="constant", truncate=3.0)

    s0, sx, sy, sd = window(m), window(m * x), window(m * y), window(d)
    sxx, sxy, syy = window(m * x * x), window(m * x * y), window(m * y * y)
    sxd, syd, sdd = window(d * x), window(d * y), window(d * d)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Moments about each pixel, so the fitted plane's intercept is its depth.
        mx, my, md = sx / s0, sy / s0, sd / s0
        cxx, cxy, cyy = sxx / s0 - mx * mx, sxy / s0 - mx * my, syy / s0 - my * my
        cxd, cyd, cdd = sxd / s0 - mx * md, syd / s0 - my * md, sdd / s0 - md * md
        determinant = cxx * cyy - cxy * cxy
        slope_x = (cxd * cyy - cyd * cxy) / determinant
        slope_y = (cyd * cxx - cxd * cxy) / determinant
        fit = md + slope_x * (x - mx) + slope_y * (y - my)
        residual = np.sqrt(np.maximum(cdd - slope_x * cxd - slope_y * cyd, 0.0))
        # Relative changes in inverse depth equal relative changes in depth.
        keep = (valid & (s0 > 0.25) & (determinant > 1e-6) & np.isfinite(fit) & (fit > 0)
                & (residual <= DEPTH_SMOOTHING_RESIDUAL * d)
                & (np.abs(fit - d) <= DEPTH_SMOOTHING_TOLERANCE * d))
        return np.where(keep, 1.0 / fit, depth).astype(depth.dtype)


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
    rays = frame["position"] - points
    facing = np.einsum("ij,ij->i", normal, rays)
    normal[facing < 0] *= -1.0
    footprint = center[keep] / math.sqrt(k["fx"] * k["fy"])
    incidence = np.abs(facing) / np.maximum(np.linalg.norm(rays, axis=1), 1e-12)
    return {"points": points, "normals": normal,
            "colors": frame["rgb"][1:-1, 1:-1][keep].astype(np.float64) / 255.0,
            "weights": frame["confidence"][1:-1, 1:-1][keep].astype(np.float64),
            "footprint": footprint, "slant": footprint / np.maximum(incidence, MIN_INCIDENCE_COSINE)}


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


def _inverse_depth_at(frame: Mapping[str, Any], column: np.ndarray, row: np.ndarray) -> np.ndarray:
    """Bilinear inverse depth at continuous pixel coordinates; NaN unless four usable pixels share a surface.

    Inverse depth is linear across a plane in a pinhole image, so a plane samples exactly;
    four pixels spanning a depth discontinuity would blend an occluder into a vote.
    """
    depth, confidence = frame["depth"], frame["confidence"]
    height, width = depth.shape
    with np.errstate(invalid="ignore"):
        x, y = column - 0.5, row - 0.5
        inside = (x >= 0) & (y >= 0) & (x < width - 1) & (y < height - 1)
    x, y = x[inside], y[inside]
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    total, usable = np.zeros(len(x)), np.ones(len(x), bool)
    nearest, farthest = np.full(len(x), np.inf), np.zeros(len(x))
    for dy, dx, weight in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)), (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
        d = depth[y0 + dy, x0 + dx].astype(np.float64)
        ok = np.isfinite(d) & (d > 0) & (confidence[y0 + dy, x0 + dx] > 0)
        usable &= ok
        total += weight / np.where(ok, d, np.inf)
        nearest, farthest = np.minimum(nearest, np.where(ok, d, np.inf)), np.maximum(farthest, np.where(ok, d, 0.0))
    usable &= farthest <= (1.0 + EDGE_RELATIVE_JUMP) * nearest
    out = np.full(len(column), np.nan)
    out[np.flatnonzero(inside)[usable]] = total[usable]
    return out


def _seen_ratio(view: Mapping[str, Any], points: np.ndarray) -> np.ndarray:
    """Depth the view sees along each point's ray over the point's own distance; NaN where it sees nothing usable."""
    k = view["intrinsics"]
    local = (points - view["position"]) @ view["rotation"]
    distance = np.where(local[:, 2] < 0, -local[:, 2], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / (_inverse_depth_at(view, local[:, 0] / distance * k["fx"] + k["cx"],
                                        -local[:, 1] / distance * k["fy"] + k["cy"]) * distance)


def _voting_views(frames: Sequence[Mapping[str, Any]], index: int, points: np.ndarray,
                  centres: np.ndarray) -> list[int]:
    """The views with the most of these points inside the gate, nearest first among equals."""
    if not len(points):
        return []
    pick = points[np.linspace(0, len(points) - 1, min(DEPTH_CONSENSUS_RANKING_SAMPLES, len(points))).astype(np.int64)]
    order = np.argsort(np.linalg.norm(centres - centres[index], axis=1), kind="stable")
    counted = []
    with np.errstate(invalid="ignore"):
        for other in order[order != index]:
            counted.append((int((np.abs(_seen_ratio(frames[other], pick) - 1.0) <= DEPTH_CONSENSUS_GATE).sum()), int(other)))
    ranked = sorted(counted, key=lambda row: -row[0])
    return [other for count, other in ranked[:CROSS_VIEW_NEIGHBOURS] if count]


def _surface_smoothed(correction: np.ndarray, voted: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Gaussian-weighted mean of the voted corrections over grid pixels on each pixel's own surface.

    A grid pixel whose depth differs by more than the gate is another surface, so a
    correction never crosses an occlusion edge. Away from votes the ratio fades to 1.
    """
    radius = int(3.0 * DEPTH_CONSENSUS_SIGMA + 0.5)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets[:, None] ** 2 + offsets[None, :] ** 2) / (2.0 * DEPTH_CONSENSUS_SIGMA ** 2))
    kernel /= kernel.sum()
    height, width = depth.shape
    padded_correction = np.pad(np.where(voted, correction, 0.0), radius)
    padded_voted = np.pad(voted.astype(np.float64), radius)
    padded_depth = np.pad(depth, radius, constant_values=np.nan)
    total, spread = np.zeros(depth.shape), np.zeros(depth.shape)
    with np.errstate(invalid="ignore"):
        for i, dy in enumerate(offsets):
            for j, dx in enumerate(offsets):
                window = (slice(radius + dy, radius + dy + height), slice(radius + dx, radius + dx + width))
                same = np.abs(padded_depth[window] - depth) <= DEPTH_CONSENSUS_GATE * depth
                total += kernel[i, j] * np.where(same, padded_correction[window], 0.0)
                spread += kernel[i, j] * np.where(same, padded_voted[window], 0.0)
    return 1.0 + total / np.maximum(spread, DEPTH_CONSENSUS_MIN_WEIGHT)


def _upsampled_on_surface(ratio: np.ndarray, grid_depth: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Bilinear full-resolution ratio from the grid samples on each pixel's own surface; 1 where none is."""
    start, stride = DEPTH_CONSENSUS_STRIDE // 2, DEPTH_CONSENSUS_STRIDE
    rows, columns = ratio.shape
    gy = np.clip((np.arange(depth.shape[0]) - start) / stride, 0, rows - 1)
    gx = np.clip((np.arange(depth.shape[1]) - start) / stride, 0, columns - 1)
    y0, x0 = np.floor(gy).astype(np.int64), np.floor(gx).astype(np.int64)
    total, weight = np.zeros(depth.shape), np.zeros(depth.shape)
    with np.errstate(invalid="ignore"):
        for ys, yw in ((y0, 1.0 - (gy - y0)), (np.minimum(y0 + 1, rows - 1), gy - y0)):
            for xs, xw in ((x0, 1.0 - (gx - x0)), (np.minimum(x0 + 1, columns - 1), gx - x0)):
                corner = np.ix_(ys, xs)
                same = np.abs(grid_depth[corner] - depth) <= DEPTH_CONSENSUS_GATE * depth
                total += np.where(same, yw[:, None] * xw[None, :] * ratio[corner], 0.0)
                weight += np.where(same, yw[:, None] * xw[None, :], 0.0)
    return np.where(weight > 0, total / np.where(weight > 0, weight, 1.0), 1.0)


def consensus_depths(frames: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rescale each frame's depth towards the median of the depth its overlapping views see along its pixels.

    A pixel's world point is projected into each voting view, the views with the
    most of the frame's samples inside the gate; that view's depth there over the
    point's own distance from it is one vote, kept only within the gate, so an
    occluder or a surface seen past an edge never votes. The median with the frame's
    own vote of 1 is smoothed over its own surface and applied at full resolution on
    that surface only: the correction is low-frequency, leaves surface detail alone and
    never crosses an occlusion edge. Views that already agree vote exactly 1 and stay
    unchanged.
    """
    frames = [dict(frame) for frame in frames]
    centres = np.array([frame["position"] for frame in frames])
    start, stride = DEPTH_CONSENSUS_STRIDE // 2, DEPTH_CONSENSUS_STRIDE
    voted_by_pass, voters_by_pass = [], []
    for _ in range(DEPTH_CONSENSUS_PASSES):
        ratios, voted, voters = [], [], []
        for index, frame in enumerate(frames):
            k = frame["intrinsics"]
            depth = frame["depth"][start::stride, start::stride].astype(np.float64)
            valid = np.isfinite(depth) & (depth > 0) & (frame["confidence"][start::stride, start::stride] > 0)
            rows, columns = np.nonzero(valid)
            z = depth[rows, columns]
            u = (columns * stride + start + 0.5 - k["cx"]) / k["fx"]
            v = (rows * stride + start + 0.5 - k["cy"]) / k["fy"]
            points = np.stack([u * z, -v * z, -z], axis=1) @ frame["rotation"].T + frame["position"]
            votes = [np.ones(len(z))]
            chosen = _voting_views(frames, index, points, centres)
            for other in chosen:
                seen = _seen_ratio(frames[other], points)
                with np.errstate(invalid="ignore"):
                    votes.append(np.where(np.abs(seen - 1.0) <= DEPTH_CONSENSUS_GATE, seen, np.nan))
            voters.append(len(chosen))
            stack = np.stack(votes)
            counted = np.isfinite(stack).sum(axis=0) >= 2
            correction, voted_grid = np.zeros(depth.shape), np.zeros(depth.shape, bool)
            correction[rows[counted], columns[counted]] = np.nanmedian(stack[:, counted], axis=0) - 1.0
            voted_grid[rows[counted], columns[counted]] = True
            grid_depth = np.where(valid, depth, np.nan)
            ratios.append((_surface_smoothed(correction, voted_grid, grid_depth), grid_depth))
            voted.append(float(counted.sum()) / depth.size)
        for frame, (ratio, grid_depth) in zip(frames, ratios):
            full = _upsampled_on_surface(ratio, grid_depth, frame["depth"])
            frame["depth"] = (frame["depth"] * full).astype(frame["depth"].dtype)
        voted_by_pass.append(round(float(np.median(voted)), 4))
        voters_by_pass.append(float(np.median(voters)))
    return frames, {"basis": "median_of_most_agreeing_views", "passes": DEPTH_CONSENSUS_PASSES,
                    "neighbours": CROSS_VIEW_NEIGHBOURS, "relative_gate": DEPTH_CONSENSUS_GATE,
                    "stride_pixels": stride, "sigma_grid_pixels": DEPTH_CONSENSUS_SIGMA,
                    "ranking_samples": DEPTH_CONSENSUS_RANKING_SAMPLES,
                    "median_voted_grid_share_by_pass": voted_by_pass,
                    "median_voting_views_by_pass": voters_by_pass}


def _voxel_keys(points: np.ndarray, voxel: float) -> np.ndarray:
    cells = np.floor(points / voxel).astype(np.int64) + (1 << (_KEY_BITS - 1))
    if cells.size and (cells.min() < 0 or cells.max() >= 1 << _KEY_BITS):
        raise ValueError("atlas_depth_splat_scene_exceeds_voxel_grid")
    return (cells[:, 0] << (2 * _KEY_BITS)) | (cells[:, 1] << _KEY_BITS) | cells[:, 2]


def _sum_by_key(keys: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum each row of ``columns`` (one row per quantity) over equal keys."""
    unique, inverse = np.unique(keys, return_inverse=True)
    inverse = inverse.reshape(-1)
    sums = np.empty((len(columns), len(unique)))
    for row in range(len(columns)):
        sums[row] = np.bincount(inverse, weights=columns[row], minlength=len(unique))
    return unique, sums


class VoxelSums:
    """Running confidence-weighted sums per voxel, so memory follows voxels rather than pixels.

    Each frame is reduced to its own voxels first. Reduced frames merge into the
    table once they reach half its size, so the table is not re-sorted per frame.
    """

    MIN_MERGE_ROWS = 1_000_000
    MERGE_CHUNK_ROWS = 65_536

    def __init__(self, voxel: float):
        self.voxel = voxel
        self._keys = np.empty(0, np.int64)
        # weight, weighted xyz, rgb and normal, sample count, weighted slanted footprint
        self._columns = np.empty((12, 0))
        self._pending: list[tuple[np.ndarray, np.ndarray]] = []
        self._pending_rows = 0

    def add(self, sample: Mapping[str, np.ndarray]) -> None:
        weights = sample["weights"]
        slant = sample.get("slant", np.zeros_like(weights))
        columns = np.concatenate([weights[None], (sample["points"] * weights[:, None]).T,
                                  (sample["colors"] * weights[:, None]).T, (sample["normals"] * weights[:, None]).T,
                                  np.ones_like(weights)[None], (slant * weights)[None]])
        self._pending.append(_sum_by_key(_voxel_keys(sample["points"], self.voxel), columns))
        self._pending_rows += len(self._pending[-1][0])
        if self._pending_rows >= max(self.MIN_MERGE_ROWS, len(self._keys) // 2):
            self._merge()

    def _merge(self) -> None:
        if self._pending:
            # Each table already has sorted, unique keys. Avoid copying all
            # twelve columns into a second pixel-sized table and constructing
            # a full inverse index before allocating the merged sums.
            tables = [(self._keys, self._columns), *self._pending]
            keys = np.unique(np.concatenate([part_keys for part_keys, _ in tables]))
            columns = np.zeros((12, len(keys)))
            for part_keys, part_columns in tables:
                for start in range(0, len(part_keys), self.MERGE_CHUNK_ROWS):
                    end = start + self.MERGE_CHUNK_ROWS
                    slots = np.searchsorted(keys, part_keys[start:end])
                    for row in range(12):
                        columns[row, slots] += part_columns[row, start:end]
            self._keys, self._columns = keys, columns
            self._pending, self._pending_rows = [], 0

    @property
    def keys(self) -> np.ndarray:
        self._merge()
        return self._keys

    @property
    def sums(self) -> np.ndarray:
        self._merge()
        return self._columns.T

    def fused(self) -> dict[str, np.ndarray]:
        total = self.sums[:, :1]
        normal = self.sums[:, 7:10] / total
        agreement = np.linalg.norm(normal, axis=1)
        return {"xyz": self.sums[:, 1:4] / total, "rgb": self.sums[:, 4:7] / total,
                "normal": normal / np.maximum(agreement, 1e-12)[:, None], "normal_agreement": agreement,
                "confidence": total[:, 0] / self.sums[:, 10], "observations": self.sums[:, 10],
                "slant": self.sums[:, 11] / total[:, 0]}


def depth_fused_splat(fused: Mapping[str, np.ndarray], voxel: float) -> SplatData:
    """One Gaussian per voxel: a thin disc on the fused normal, isotropic where normals disagree.

    A disc spans its voxel or, on a slanted surface, the spacing of the pixel rows
    that sampled it, so oblique views do not leave stripes between those rows.
    """
    count = len(fused["xyz"])
    normal = fused["normal"] * np.where(fused["normal"][:, 2:3] < 0, -1.0, 1.0)
    quats = np.stack([1.0 + normal[:, 2], -normal[:, 1], normal[:, 0], np.zeros(count)], axis=1)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    sigma = DISC_SIGMA_PER_VOXEL * np.maximum(voxel, fused.get("slant", np.zeros(count)))
    scales = np.repeat(np.log(sigma)[:, None], 3, axis=1)
    thin = fused["normal_agreement"] >= 0.5
    scales[thin, 2] = np.log(sigma[thin] * DISC_THICKNESS_RATIO)
    opacity = 0.5 + 0.49 * np.clip(fused["confidence"], 0.0, 1.0)
    return SplatData(count=count, xyz=fused["xyz"].astype(np.float32),
                     opacity=np.log(opacity / (1.0 - opacity)).astype(np.float32),
                     f_dc=((fused["rgb"] - 0.5) / SH_C0).astype(np.float32), scales=scales.astype(np.float32),
                     quats=quats.astype(np.float32), properties=PLY_PROPERTIES)


def _disc_normals(splat: SplatData) -> np.ndarray:
    """Each disc's rotated local z axis: the normal depth_fused_splat turned it onto."""
    w, x, y, z = (splat.quats[:, index].astype(np.float64) for index in range(4))
    normal = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1)
    return normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)


def splats2mesh_input(splat: SplatData, voxel: float, *, limit: int | None = None) -> tuple[SplatData, float]:
    """Merge discs on the finest coarser grid within the size splats2Mesh has meshed.

    Only the mesh input is merged; the full draft stays the viewing and download splat.
    """
    limit = SPLATS2MESH_MAX_GAUSSIANS if limit is None else limit
    if splat.count <= limit:
        return splat, voxel
    points = splat.xyz.astype(np.float64)
    for multiple in _MESH_INPUT_GRID_MULTIPLES:
        size = voxel * multiple
        keys, inverse = np.unique(_voxel_keys(points, size), return_inverse=True)
        if len(keys) <= limit:
            break
    else:
        raise ValueError("atlas_depth_splat_mesh_input_exceeds_limit")
    inverse = inverse.reshape(-1)
    weight = np.clip((splat.opacity_sigmoid.astype(np.float64) - 0.5) / 0.49, 1e-3, 1.0)
    total = np.bincount(inverse, weights=weight, minlength=len(keys))

    def mean(values):
        return np.stack([np.bincount(inverse, weights=weight * values[:, column], minlength=len(keys))
                         for column in range(values.shape[1])], axis=1) / total[:, None]

    # Disc normals are flipped towards +z, so a wall's neighbours can point either
    # way: merge their sign-free principal axis, not their vector mean. Agreement
    # is that axis's anisotropy: 1 for parallel normals, 0 for an even spread.
    normals = _disc_normals(splat)
    pairs = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    second = mean(np.stack([normals[:, i] * normals[:, j] for i, j in pairs], axis=1))
    values, vectors = np.linalg.eigh(second[:, [0, 1, 2, 1, 3, 4, 2, 4, 5]].reshape(-1, 3, 3))
    # Each disc's tangent sigma records the row spacing it had to bridge.
    spans = np.exp(splat.scales[:, :1].astype(np.float64)) / DISC_SIGMA_PER_VOXEL
    fused = {"xyz": mean(points), "rgb": mean(splat.f_dc.astype(np.float64) * SH_C0 + 0.5),
             "normal": vectors[:, :, 2], "normal_agreement": np.clip((3 * values[:, 2] - 1) / 2, 0.0, 1.0),
             "confidence": total / np.bincount(inverse, minlength=len(keys)), "slant": mean(spans)[:, 0]}
    return depth_fused_splat(fused, size), size


def fuse_posed_frames(frames: Sequence[Mapping[str, Any]], *,
                      voxel_footprint_multiple: float = 2.0) -> tuple[SplatData, dict[str, Any]]:
    """Gate cross-view agreement, then stream every usable pixel into a draft splat.

    Full-resolution samples exist for one frame at a time; only bounded subsamples and
    per-voxel sums persist, so memory follows the scene's voxels, not its frame count.
    """
    if not math.isfinite(voxel_footprint_multiple) or voxel_footprint_multiple <= 0:
        raise ValueError("atlas_depth_splat_voxel_multiple_invalid")
    frames = [{**frame, "depth": smoothed_depth(frame["depth"])} for frame in frames]
    samples = [_subsample(surface_samples(frame)) for frame in frames]
    agreement = cross_view_agreement(frames, samples)
    if (agreement["median_relative_residual"] is not None
            and agreement["median_relative_residual"] > MAX_MEDIAN_CROSS_VIEW_RESIDUAL):
        raise ValueError("atlas_depth_splat_cross_view_inconsistent")
    frames, consensus = consensus_depths(frames)
    samples = [_subsample(surface_samples(frame)) for frame in frames]
    consensus["median_relative_residual_after"] = cross_view_agreement(frames, samples)["median_relative_residual"]
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
              "depth_smoothing": {"basis": "local_plane_fit", "sigma_pixels": DEPTH_SMOOTHING_SIGMA_PIXELS,
                                  "relative_residual": DEPTH_SMOOTHING_RESIDUAL,
                                  "relative_tolerance": DEPTH_SMOOTHING_TOLERANCE},
              "depth_consensus": consensus,
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
    # Source portrait RGBD and generated landscape RGBD share a world frame,
    # not a pixel grid. Mesh rendering needs one grid; scale each camera's own
    # intrinsics independently so its original rays/FOV are preserved.
    width, height = max(grids, key=lambda grid: grid[0] / grid[1])
    factor = max(min(1.0, SPLATS2MESH_MAX_RENDER / max(width, height)), SPLATS2MESH_MIN_RENDER / min(width, height))
    render = (round(width * factor), round(height * factor))
    if max(render) > SPLATS2MESH_MAX_RENDER:
        raise ValueError("atlas_depth_splat_render_aspect_unsupported")
    chosen = list(frames)
    if len(chosen) > SPLATS2MESH_MAX_CAMERAS:
        chosen = [chosen[i] for i in np.linspace(0, len(chosen) - 1, SPLATS2MESH_MAX_CAMERAS).round().astype(int)]
    cameras = [{"intrinsics": {"fx": k["fx"] * sx, "fy": k["fy"] * sy, "cx": k["cx"] * sx, "cy": k["cy"] * sy,
                               "width": render[0], "height": render[1]},
                "extrinsics": {key: frame["extrinsics"][key] for key in ("position", "quaternion", "coordinateSystem")}}
               for frame in chosen for k in (frame["intrinsics"],)
               for sx, sy in ((render[0] / k["width"], render[1] / k["height"]),)]
    return {"task": "splats2Mesh", "source_video_digest": source_video_digest,
            "request": {"splatAsset": {"assetId": "upload:splat"}, "cameras": cameras, "textureMode": "image_texture"},
            "uploads": {"splat": {"path": str(spz.resolve()), "sha256": _sha256_file(spz)}}}


def _failed_attempt(root: Path) -> bool:
    """Whether every operation in a task attempt ended in a confirmed provider failure.

    Only then can nothing more be bought there. A completed, pending, rejected
    or unreconciled attempt is not failed; nor is one whose retry is still open.
    """
    state_path = root / "atlas_pose" / "submission.json"
    if (root / "result.json").exists() or not state_path.is_file():
        return False
    state = json.loads(state_path.read_text())
    operation = state.get("terminal_operation")
    if (state.get("status") != "failed" or not isinstance(operation, dict) or operation.get("done") is not True
            or operation.get("error") is None or operation.get("id") != state.get("operation_id")):
        return False
    return not (root / "retry").exists() or _failed_attempt(root / "retry")


def splats2mesh_attempt(splat_root: Path, mesh_root: Path) -> tuple[dict[str, Any], Path]:
    """The splats2Mesh spec and attempt root, with an oversized draft merged for meshing only.

    The retained export is never rewritten: its manifest digest binds coverage and
    bundle state. A root already attempted keeps its input, so a pending or
    completed mesh is never bought twice, unless every operation there failed;
    a merged input gets its own attempt root.
    """
    spec = json.loads((splat_root / "splats2mesh_task_spec.json").read_text())
    from .atlas_native_splat import load_splat_manifest
    manifest = load_splat_manifest(splat_root)
    if (manifest.get("task") == "posedImages2Splats"
            or manifest.get("gaussian_count", 0) <= SPLATS2MESH_MAX_GAUSSIANS
            or (mesh_root.exists() and not _failed_attempt(mesh_root))):
        return spec, mesh_root
    source = spec["uploads"]["splat"]
    if _sha256_file(Path(source["path"])) != source["sha256"]:
        raise ValueError("atlas_depth_splat_mesh_source_changed")
    input_root = mesh_root.with_name(f"{mesh_root.name}-input-{MESH_INPUT_REVISION}")
    receipt_path = input_root / "mesh_input.json"
    if not receipt_path.is_file():
        merged, voxel = splats2mesh_input(decode_spz(Path(source["path"]).read_bytes()), manifest["voxel_size"])
        input_root.mkdir(parents=True, exist_ok=True)
        output = input_root / "atlas_depth_splat_mesh_input.spz"
        output.write_bytes(encode_spz(merged))
        write_receipt = {"schema_version": "atlas_splats2mesh_input.v1", "basis": "merged_discs_for_splats2mesh_size",
                         "source_sha256": source["sha256"], "source_gaussian_count": manifest["gaussian_count"],
                         "source_voxel_size": manifest["voxel_size"], "gaussian_count": merged.count,
                         "voxel_size": voxel, "limit": SPLATS2MESH_MAX_GAUSSIANS,
                         "path": str(output.resolve()), "sha256": _sha256_file(output)}
        receipt_path.write_text(json.dumps(write_receipt, indent=2) + "\n")
    receipt = json.loads(receipt_path.read_text())
    if receipt["source_sha256"] != source["sha256"] or _sha256_file(Path(receipt["path"])) != receipt["sha256"]:
        raise ValueError("atlas_depth_splat_mesh_input_changed")
    merged_input = {"path": receipt["path"], "sha256": receipt["sha256"]}
    return ({**spec, "uploads": {**spec["uploads"], "splat": merged_input}},
            mesh_root.with_name(f"{mesh_root.name}-{MESH_INPUT_REVISION}"))


def fusion_profile() -> dict[str, Any]:
    """Bind derived exports to fusion semantics, independently of paid RGBD stages."""
    return {"revision": FUSION_REVISION, "edge_relative_jump": EDGE_RELATIVE_JUMP,
            "disc_sigma_per_voxel": DISC_SIGMA_PER_VOXEL,
            "minimum_incidence_cosine": MIN_INCIDENCE_COSINE,
            "disc_thickness_ratio": DISC_THICKNESS_RATIO,
            "depth_smoothing_sigma_pixels": DEPTH_SMOOTHING_SIGMA_PIXELS,
            "depth_smoothing_residual": DEPTH_SMOOTHING_RESIDUAL,
            "depth_smoothing_tolerance": DEPTH_SMOOTHING_TOLERANCE,
            "depth_consensus_passes": DEPTH_CONSENSUS_PASSES,
            "depth_consensus_neighbours": CROSS_VIEW_NEIGHBOURS,
            "depth_consensus_gate": DEPTH_CONSENSUS_GATE,
            "depth_consensus_stride": DEPTH_CONSENSUS_STRIDE,
            "depth_consensus_sigma": DEPTH_CONSENSUS_SIGMA,
            "depth_consensus_min_weight": DEPTH_CONSENSUS_MIN_WEIGHT,
            "depth_consensus_ranking_samples": DEPTH_CONSENSUS_RANKING_SAMPLES}


def build_depth_splat(pose_root: Path, output_root: Path, *, voxel_footprint_multiple: float = 2.0) -> dict[str, Any]:
    """Write the draft splat (PLY + SPZ), its manifest and the splats2Mesh task spec."""
    result, frames = load_posed_rgbd(pose_root)
    return write_depth_splat(frames=frames, result=result, output_root=output_root,
                             voxel_footprint_multiple=voxel_footprint_multiple)


def write_depth_splat(*, frames, result, output_root: Path, voxel_footprint_multiple=2.0,
                      generated_result_digests=(), visual_review_digest=None,
                      rebuild_source_manifest_digest=None,
                      generated_appearance_basis='generated_object_absent_background'):
    """Shared writer; edited RGBD remains explicitly generated and separately bound."""
    if (generated_appearance_basis not in ('generated_object_absent_background',
                                         'preserved_capture_with_bounded_completion')
            or generated_appearance_basis == 'preserved_capture_with_bounded_completion'
            and (not generated_result_digests or not visual_review_digest)):
        raise ValueError('atlas_depth_splat_generated_context_invalid')
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
    manifest = {"schema_version": SCHEMA_VERSION, "fusion_profile": fusion_profile(),
                "pose_result_digest": result["digest"],
                "pose_operation_id": result["operation"]["id"], "source_video_digest": source_video_digest,
                **report, "gaussian_count": splat.count,
                "outputs": [{"role": role, "path": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
                            for role, path in (("splat_ply", ply), ("splat_spz", spz))],
                "world_frame": "atlas_world_rub_y_up", "units": result.get("depth_unit"),
                "scale_status": result.get("scale_status"), "splat_kind": "depth_fused_draft",
                "photometric_optimization": False, "generated_content": bool(generated_result_digests),
                **({"generated_result_digests": list(generated_result_digests),
                    "visual_review_digest": visual_review_digest,
                    "appearance_basis": generated_appearance_basis} if generated_result_digests else {}),
                "metric_measurement_proven": False, "claim_scope": "development_only"}
    if rebuild_source_manifest_digest is not None:
        manifest['rebuild_source_manifest_digest'] = rebuild_source_manifest_digest
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
