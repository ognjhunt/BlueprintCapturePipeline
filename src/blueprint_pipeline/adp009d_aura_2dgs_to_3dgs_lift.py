"""Lift the sealed Aura 2D-Gaussian-surfel PLY to a standard 3DGS PLY.

Isaac RTX has a native splat lane (NuRec volumes), and the ADP-009D goal prompt
prefers it: "If an existing native Gaussian/Omniverse renderer can render all
required layers directly and is already admitted, use it."  The repository
converter ``isaac_nurec_export.convert_ply_to_isaac_usd`` reaches that lane by
transcoding a *standard 3DGS* PLY, and the sealed Aura appearance is not one:
AuraFusion360 emits 2D Gaussian surfels with two scale axes rather than three.

Rather than feed a malformed file to the transcoder and hope, this module
produces a well-formed standard 3DGS PLY from the sealed bytes.  Three exact
differences are reconciled, all measured from the sealed file:

* **Missing third scale.**  A surfel is a flat disc: two in-plane log-scales and
  an implied normal.  The lift appends a third log-scale a fixed ratio below the
  smaller in-plane axis, keeping each Gaussian flat relative to its own size.
  A surfel is mathematically zero-thickness, so any thickness is an
  approximation -- this one is recorded, never asserted to be free.
* **Non-finite opacity.**  261,450 of 414,322 sealed opacities are ``+inf``
  (they are stored pre-sigmoid).  Infinity propagates through any downstream
  arithmetic, so it is clamped to a logit whose sigmoid is exactly 1.0 in
  float32 -- the same rendered opacity, finite bytes.
* **Extra ``is_masked`` channels.**  AuraFusion360 records which Gaussians it
  inpainted.  Standard 3DGS has no such property and the renderer does not read
  it, so it is dropped and the drop is recorded.

Nothing here changes the sealed source.  The lift is a derivative with both
digests bound in its receipt, and it never claims the lifted appearance matches
the sealed one -- that must be established by rendering both and comparing.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

LIFT_SCHEMA_VERSION = "adp009d_aura_2dgs_to_3dgs_lift.v1"

# A surfel is flat: the implied third axis is this fraction of the smaller
# in-plane axis.  Recorded in the receipt so a render comparison can sweep it.
DEFAULT_THICKNESS_RATIO = 0.01
# sigmoid(20) == 1.0 exactly in float32 (the residual 2.06e-9 is far below the
# 1.19e-7 float32 epsilon), so clamping here preserves rendered opacity.
OPACITY_LOGIT_CEILING = 20.0
# Guard against a lifted scale underflowing to zero: exp(-60) is 8.8e-27, which
# is comfortably normal in float32, while exp(-90) would flush to zero.
MIN_LOG_SCALE = -60.0

_SH_REST_COUNT = 45
_EXPECTED_2DGS_PROPERTIES = (
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    *[f"f_rest_{index}" for index in range(_SH_REST_COUNT)],
    "opacity",
    "scale_0",
    "scale_1",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "is_masked_0",
    "is_masked_1",
    "is_masked_2",
)
_STANDARD_3DGS_PROPERTIES = (
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    *[f"f_rest_{index}" for index in range(_SH_REST_COUNT)],
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


class AuraLiftError(ValueError):
    """Fail-closed lift contract errors."""

    def __init__(self, errors: list[str] | tuple[str, ...]) -> None:
        self.errors = tuple(sorted({str(error) for error in errors if str(error)}))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_binary_ply(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Read a little-endian binary PLY whose vertex properties are all float32."""

    source = Path(path)
    with source.open("rb") as handle:
        header = b""
        while b"end_header\n" not in header:
            chunk = handle.read(4096)
            if not chunk:
                raise AuraLiftError(["ply_header_unterminated"])
            header += chunk
        head, remainder = header.split(b"end_header\n", 1)
        lines = head.decode("ascii", errors="replace").splitlines()
        if not lines or lines[0].strip() != "ply":
            raise AuraLiftError(["ply_magic_missing"])
        if "format binary_little_endian 1.0" not in lines:
            raise AuraLiftError(["ply_format_not_binary_little_endian"])
        vertex_lines = [line for line in lines if line.startswith("element vertex")]
        if len(vertex_lines) != 1:
            raise AuraLiftError(["ply_vertex_element_ambiguous"])
        count = int(vertex_lines[0].split()[-1])
        properties: list[str] = []
        for line in lines:
            if not line.startswith("property "):
                continue
            parts = line.split()
            if parts[1] != "float":
                raise AuraLiftError([f"ply_property_not_float32:{parts[-1]}"])
            properties.append(parts[-1])
        dtype = np.dtype([(name, "<f4") for name in properties])
        payload = remainder + handle.read(count * dtype.itemsize - len(remainder))
    if len(payload) < count * dtype.itemsize:
        raise AuraLiftError(["ply_payload_truncated"])
    vertices = np.frombuffer(payload, dtype=dtype, count=count)
    return properties, vertices


def write_binary_ply(path: str | Path, properties: list[str], vertices: np.ndarray) -> None:
    """Write a little-endian binary float32 PLY with a deterministic header."""

    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {len(vertices)}\n"
    for name in properties:
        header += f"property float {name}\n"
    header += "end_header\n"
    with Path(path).open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertices.tobytes(order="C"))


def lift_aura_2dgs_ply_to_3dgs(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    thickness_ratio: float = DEFAULT_THICKNESS_RATIO,
) -> dict[str, Any]:
    """Produce a standard 3DGS PLY from the sealed Aura 2DGS PLY.

    Returns a digest-bound receipt.  Raises :class:`AuraLiftError` when the
    source is not the expected 2DGS layout, so an unrelated PLY can never be
    silently reinterpreted as the sealed appearance.
    """

    if not 0.0 < float(thickness_ratio) < 1.0:
        raise AuraLiftError(["thickness_ratio_out_of_range"])

    source = Path(source_path)
    destination = Path(destination_path)
    properties, vertices = read_binary_ply(source)

    if tuple(properties) != _EXPECTED_2DGS_PROPERTIES:
        missing = [p for p in _EXPECTED_2DGS_PROPERTIES if p not in properties]
        unexpected = [p for p in properties if p not in _EXPECTED_2DGS_PROPERTIES]
        errors = ["source_is_not_the_expected_aura_2dgs_layout"]
        if missing:
            errors.append("missing:" + ",".join(missing[:6]))
        if unexpected:
            errors.append("unexpected:" + ",".join(unexpected[:6]))
        if "scale_2" in properties:
            errors.append("source_already_has_three_scale_axes")
        raise AuraLiftError(errors)

    count = len(vertices)
    scale_0 = vertices["scale_0"].astype(np.float64)
    scale_1 = vertices["scale_1"].astype(np.float64)
    opacity = vertices["opacity"].astype(np.float64)

    # Third axis: a fixed ratio below the smaller in-plane axis, floored so no
    # Gaussian collapses to zero volume.
    scale_2 = np.minimum(scale_0, scale_1) + math.log(float(thickness_ratio))
    underflowed = int((scale_2 < MIN_LOG_SCALE).sum())
    scale_2 = np.maximum(scale_2, MIN_LOG_SCALE)
    # Flooring must never make the implied thickness exceed the in-plane extent.
    scale_2 = np.minimum(scale_2, np.minimum(scale_0, scale_1))

    non_finite_opacity = int((~np.isfinite(opacity)).sum())
    clamped_opacity = np.clip(
        np.nan_to_num(
            opacity, nan=0.0, posinf=OPACITY_LOGIT_CEILING, neginf=-OPACITY_LOGIT_CEILING
        ),
        -OPACITY_LOGIT_CEILING,
        OPACITY_LOGIT_CEILING,
    )

    lifted = np.empty(count, dtype=np.dtype([(n, "<f4") for n in _STANDARD_3DGS_PROPERTIES]))
    for name in _STANDARD_3DGS_PROPERTIES:
        if name == "scale_2":
            lifted[name] = scale_2.astype(np.float32)
        elif name == "opacity":
            lifted[name] = clamped_opacity.astype(np.float32)
        else:
            lifted[name] = vertices[name]

    if not np.isfinite(
        np.stack([lifted[name] for name in _STANDARD_3DGS_PROPERTIES], axis=0)
    ).all():
        raise AuraLiftError(["lifted_ply_contains_non_finite_values"])

    destination.parent.mkdir(parents=True, exist_ok=True)
    write_binary_ply(destination, list(_STANDARD_3DGS_PROPERTIES), lifted)

    receipt: dict[str, Any] = {
        "schema_version": LIFT_SCHEMA_VERSION,
        "status": "lifted",
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "source_mutated": False,
        "destination_path": str(destination),
        "destination_sha256": _sha256(destination),
        "vertex_count": count,
        "source_properties": list(properties),
        "destination_properties": list(_STANDARD_3DGS_PROPERTIES),
        "thickness_ratio": float(thickness_ratio),
        "third_axis_rule": "log_scale_2 = min(log_scale_0, log_scale_1) + ln(ratio)",
        "min_log_scale_floor": MIN_LOG_SCALE,
        "third_axis_floored_count": underflowed,
        "opacity_logit_ceiling": OPACITY_LOGIT_CEILING,
        "non_finite_opacity_count": non_finite_opacity,
        "dropped_properties": ["is_masked_0", "is_masked_1", "is_masked_2"],
        "appearance_equivalence_established": False,
        "appearance_equivalence_note": (
            "A surfel is zero-thickness; any lifted thickness is an approximation. "
            "Equivalence must be established by rendering the lifted and sealed "
            "appearances from identical camera poses and comparing, never assumed."
        ),
    }
    return receipt


__all__ = [
    "DEFAULT_THICKNESS_RATIO",
    "LIFT_SCHEMA_VERSION",
    "MIN_LOG_SCALE",
    "OPACITY_LOGIT_CEILING",
    "AuraLiftError",
    "lift_aura_2dgs_ply_to_3dgs",
    "read_binary_ply",
    "write_binary_ply",
]
