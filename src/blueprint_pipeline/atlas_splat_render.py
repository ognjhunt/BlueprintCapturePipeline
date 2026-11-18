"""A CPU alpha-blended render of a Gaussian splat at one posed camera, in bounded memory.

The held-out look check (``atlas_splat_input_quality.heldout_quality``) used to
give each pixel the colour of the nearest opaque Gaussian centre. That is no
render: it ignores each Gaussian's footprint and opacity, so a blurry or
needle-ridden splat could score like a clean one. This is the standard 3D
Gaussian splatting forward pass (EWA projection with the 0.3 px low-pass, depth
sorted per 16 px tile, front-to-back compositing, alpha capped at 0.99 and
dropped under 1/255, a pixel done once its transmittance is under 1e-4), in
NumPy, so it runs on the control-plane host with no GPU and no new dependency.

Memory stays bounded whatever the splat size: Gaussians are projected
``PROJECT_CHUNK`` at a time and only the visible ones are kept (11 float32 each),
each band of tile rows is binned on its own, and each tile composites at most
``COMPOSITE_CHUNK`` Gaussians at a time. On the 2026-11-18 container (4 CPUs) a
2M-Gaussian splat at 640x360 renders in about 20-60 s at well under 1 GB.

Colour is the splat's view-dependent spherical harmonics when it carries them
(``sh_rest``, degree up to 3), otherwise its base colour.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

import numpy as np

REVISION = "numpy-tile-alpha-blend-v1"
TILE = 16
PROJECT_CHUNK = 1 << 18
COMPOSITE_CHUNK = 256
LOW_PASS = 0.3
MAX_ALPHA = 0.99
MIN_ALPHA = 1.0 / 255.0
DONE_TRANSMITTANCE = 1e-4
NEAR = 0.05
#: Footprints are cut at this many standard deviations, and never wider than the image.
SIGMA_CUT = 3.0
SH_C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199
_SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
_SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
          1.445305721320277, -0.5900435899266435)


def camera_of(frame: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Centre, camera-to-world rotation (RUB columns) and intrinsics of a posed view or a camera record."""
    if "position" in frame and "rotation" in frame:
        return np.asarray(frame["position"], float), np.asarray(frame["rotation"], float), frame["intrinsics"]
    from .atlas_depth_splat import _rotation_matrix
    extrinsics = frame["extrinsics"]
    return (np.asarray(extrinsics["position"], float), _rotation_matrix(extrinsics["quaternion"]),
            frame["intrinsics"])


def _quaternion_matrices(quats: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotations of (w, x, y, z) quaternions, the splat's ``rot_0..3`` order."""
    q = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1).reshape(-1, 3, 3)


def project_gaussians(xyz, scales_log, quats, position, rotation, intrinsics, width, height):
    """Each Gaussian's view depth, pixel centre and 2D covariance (a, b, c) at a ``width`` x ``height`` render.

    ``rotation`` maps camera axes to the world (RUB: the camera looks down -Z).
    The covariance includes the low-pass dilation; behind-camera Gaussians get z <= 0.
    """
    factor = width / float(intrinsics["width"])
    fx, fy = float(intrinsics["fx"]) * factor, float(intrinsics["fy"]) * factor
    cx, cy = float(intrinsics["cx"]) * factor, float(intrinsics["cy"]) * factor
    local = (np.asarray(xyz, np.float64) - position) @ rotation
    z = -local[:, 2]
    safe = np.where(z > NEAR, z, NEAR)
    # The Jacobian is taken at most 1.3 half-fields of view off axis, as 3D Gaussian splatting does.
    limit_x, limit_y = 1.3 * (width / 2.0) / fx, 1.3 * (height / 2.0) / fy
    tx = np.clip(local[:, 0] / safe, -limit_x, limit_x) * safe
    ty = np.clip(local[:, 1] / safe, -limit_y, limit_y) * safe
    u = fx * local[:, 0] / safe + cx
    v = -fy * local[:, 1] / safe + cy
    scale = np.exp(np.asarray(scales_log, np.float64))
    world = _quaternion_matrices(np.asarray(quats, np.float64)) * scale[:, None, :]
    # Camera-frame covariance: R^T (M M^T) R with M = rotation * scale.
    m = np.einsum("ji,njk->nik", rotation, world)
    sigma = m @ m.transpose(0, 2, 1)
    jac = np.zeros((len(z), 2, 3))
    jac[:, 0, 0] = fx / safe
    jac[:, 0, 2] = fx * tx / safe ** 2
    jac[:, 1, 1] = -fy / safe
    jac[:, 1, 2] = -fy * ty / safe ** 2
    cov = jac @ sigma @ jac.transpose(0, 2, 1)
    return z, u, v, cov[:, 0, 0] + LOW_PASS, cov[:, 0, 1], cov[:, 1, 1] + LOW_PASS


def _sh_colour(splat, index: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """RGB in [0, 1] of the Gaussians ``index`` seen along unit ``directions`` (camera to Gaussian)."""
    colour = SH_C0 * np.asarray(splat.f_dc, np.float64)[index]
    rest = getattr(splat, "sh_rest", None)
    if rest is not None and np.shape(rest)[1] >= 9:
        coefficients = np.shape(rest)[1] // 3
        r = np.asarray(rest, np.float64)[index].reshape(len(index), 3, coefficients)
        x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
        basis = [-_SH_C1 * y, _SH_C1 * z, -_SH_C1 * x]
        if coefficients >= 8:
            xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
            basis += [_SH_C2[0] * xy, _SH_C2[1] * yz, _SH_C2[2] * (2 * zz - xx - yy), _SH_C2[3] * xz,
                      _SH_C2[4] * (xx - yy)]
            if coefficients >= 15:
                basis += [_SH_C3[0] * y * (3 * xx - yy), _SH_C3[1] * xy * z, _SH_C3[2] * y * (4 * zz - xx - yy),
                          _SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy), _SH_C3[4] * x * (4 * zz - xx - yy),
                          _SH_C3[5] * z * (xx - yy), _SH_C3[6] * x * (xx - 3 * yy)]
        for k, value in enumerate(basis[:coefficients]):
            colour += r[:, :, k] * value[:, None]
    return np.clip(colour + 0.5, 0.0, 1.0)


def _visible(splat, position, rotation, intrinsics, width, height):
    """The Gaussians that land on the image: float32 rows (u, v, conic a, b, c, opacity, r, g, b, z, radius)."""
    count = len(splat.xyz)
    rows, indices = [], []
    opacity_logit = np.asarray(splat.opacity, np.float64)
    for start in range(0, count, PROJECT_CHUNK):
        part = slice(start, min(start + PROJECT_CHUNK, count))
        z, u, v, a, b, c = project_gaussians(splat.xyz[part], splat.scales[part], splat.quats[part],
                                             position, rotation, intrinsics, width, height)
        det = a * c - b * b
        mid = 0.5 * (a + c)
        major = mid + np.sqrt(np.maximum(mid * mid - det, 0.0))
        radius = np.minimum(np.ceil(SIGMA_CUT * np.sqrt(major)), float(max(width, height)))
        opacity = 1.0 / (1.0 + np.exp(-np.clip(opacity_logit[part], -30, 30)))
        keep = ((z > NEAR) & (det > 0) & (opacity >= MIN_ALPHA) & np.isfinite(u) & np.isfinite(v)
                & (u + radius > 0) & (u - radius < width) & (v + radius > 0) & (v - radius < height))
        if not keep.any():
            continue
        index = np.flatnonzero(keep) + start
        directions = np.asarray(splat.xyz[index], np.float64) - position
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        k = keep
        rows.append(np.column_stack([u[k], v[k], c[k] / det[k], -b[k] / det[k], a[k] / det[k], opacity[k],
                                     _sh_colour(splat, index, directions), z[k], radius[k]]).astype(np.float32))
        indices.append(index)
    if not rows:
        return np.zeros((0, 11), np.float32), np.zeros(0, np.int64)
    return np.concatenate(rows), np.concatenate(indices)


def _composite_tile(g, x0, y0, x1, y1):
    """Colour, alpha and expected depth of one tile from its depth-ordered Gaussians ``g``."""
    py, px = np.mgrid[y0:y1, x0:x1]
    px = px.reshape(-1).astype(np.float32) + 0.5
    py = py.reshape(-1).astype(np.float32) + 0.5
    pixels = len(px)
    colour = np.zeros((pixels, 3), np.float32)
    depth = np.zeros(pixels, np.float32)
    transmittance = np.ones(pixels, np.float32)
    for start in range(0, len(g), COMPOSITE_CHUNK):
        chunk = g[start:start + COMPOSITE_CHUNK]
        dx = px[:, None] - chunk[None, :, 0]
        dy = py[:, None] - chunk[None, :, 1]
        power = -0.5 * (chunk[None, :, 2] * dx * dx + chunk[None, :, 4] * dy * dy) - chunk[None, :, 3] * dx * dy
        alpha = np.minimum(MAX_ALPHA, chunk[None, :, 5] * np.exp(np.minimum(power, 0.0)))
        alpha[(alpha < MIN_ALPHA) | (power > 0)] = 0.0
        # Transmittance before each Gaussian; a pixel stops once it falls under the threshold.
        before = transmittance[:, None] * np.cumprod(
            np.concatenate([np.ones((pixels, 1), np.float32), 1.0 - alpha[:, :-1]], axis=1), axis=1)
        weight = np.where(before >= DONE_TRANSMITTANCE, before * alpha, 0.0).astype(np.float32)
        colour += weight @ chunk[:, 6:9]
        depth += weight @ chunk[:, 9]
        transmittance = (before[:, -1] * (1.0 - alpha[:, -1])).astype(np.float32)
        if transmittance.max() < DONE_TRANSMITTANCE:
            break
    shape = (y1 - y0, x1 - x0)
    return colour.reshape(*shape, 3), (1.0 - transmittance).reshape(shape), depth.reshape(shape)


def render(splat, camera: Mapping[str, Any], width: int, height: int | None = None, *,
           workers: int | None = None) -> dict[str, Any]:
    """Alpha-blended RGB (H, W, 3) in [0, 1] over black, accumulated alpha and expected depth at ``camera``.

    ``camera`` is a posed view (``position``, ``rotation``, ``intrinsics``) or a
    camera record (``extrinsics``, ``intrinsics``). ``height`` follows the
    camera's aspect when omitted. Expected depth is the alpha-weighted view
    depth divided by the alpha, NaN where nothing landed.
    """
    position, rotation, intrinsics = camera_of(camera)
    if height is None:
        height = max(1, round(float(intrinsics["height"]) * width / float(intrinsics["width"])))
    gaussians, _ = _visible(splat, position, rotation, intrinsics, width, height)
    rgb = np.zeros((height, width, 3), np.float32)
    alpha = np.zeros((height, width), np.float32)
    depth = np.zeros((height, width), np.float32)
    columns = (width + TILE - 1) // TILE
    bands = (height + TILE - 1) // TILE
    order = np.argsort(gaussians[:, 9], kind="stable")
    gaussians = gaussians[order]
    top, bottom = gaussians[:, 1] - gaussians[:, 10], gaussians[:, 1] + gaussians[:, 10]
    left, right = gaussians[:, 0] - gaussians[:, 10], gaussians[:, 0] + gaussians[:, 10]

    def band(row):
        y0, y1 = row * TILE, min(height, (row + 1) * TILE)
        inside = np.flatnonzero((bottom >= y0) & (top < y1))
        g, lo, hi = gaussians[inside], left[inside], right[inside]
        for column in range(columns):
            x0, x1 = column * TILE, min(width, (column + 1) * TILE)
            here = g[(hi >= x0) & (lo < x1)]
            if len(here):
                rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1], depth[y0:y1, x0:x1] = _composite_tile(here, x0, y0, x1, y1)
    workers = workers or min(4, os.cpu_count() or 1)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(band, range(bands)))
    else:
        for row in range(bands):
            band(row)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.where(alpha > MIN_ALPHA, depth / alpha, np.nan)
    return {"rgb": rgb, "alpha": alpha, "depth": expected, "visible_gaussians": int(len(gaussians)),
            "width": width, "height": height, "revision": REVISION,
            "colour_basis": ("spherical_harmonics" if getattr(splat, "sh_rest", None) is not None
                             and np.shape(splat.sh_rest)[1] >= 9 else "base_colour")}

