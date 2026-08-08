"""Read and write NVIDIA NuRec volumes, the splat format Isaac renders natively.

Isaac RTX renders NuRec volumes: an InteriorGS scene in this format has already
been rendered with a full-size robot composited inside it.  The same scene's
appearance authored as an Omniverse ``ParticleField`` has never rendered
correctly across a dozen attempts, so being able to *write* this format is what
lets a ghost-removed appearance use the renderer that works, instead of trading
a proven renderer away to remove a cosmetic artifact.

The container is plain: gzip around MessagePack.  Bulk arrays live in
``state_dict`` as raw little-endian buffers, each paired with a ``<key>.shape``
entry, at the precision named in the layer config -- 16 for float16.  Values are
stored **pre-activation**: scales are logs, densities are logits, and the
renderer applies ``scale_activation``/``density_activation`` itself.  Writing
raw learned parameters straight through is the safer direction, because every
activation applied on the authoring side is a place to be wrong about units.

Nothing here renders, and nothing here decides whether a volume is admissible.
"""

from __future__ import annotations

import gzip
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

NUREC_CODEC_SCHEMA_VERSION = "nurec_volume_codec.v1"

# The state_dict key prefix every bulk array shares.
GAUSSIAN_KEY_PREFIX = ".gaussians_nodes.gaussians."

# Arrays a gaussian layer carries, and the trailing dimension each must have.
# features_specular is 45 because a degree-3 spherical-harmonic radiance keeps
# 15 non-DC coefficients across 3 channels; features_albedo is the DC term.
GAUSSIAN_ARRAY_WIDTHS: dict[str, int] = {
    "positions": 3,
    "rotations": 4,
    "scales": 3,
    "densities": 1,
    "features_albedo": 3,
    "features_specular": 45,
}

PRECISION_DTYPES = {16: np.float16, 32: np.float32}

# gzip embeds an mtime, so a byte-comparable re-encode has to pin it.  Without
# this a round-trip differs in four header bytes and proves nothing.
DETERMINISTIC_GZIP_MTIME = 0


class NuRecCodecError(ValueError):
    """Fail-closed NuRec container errors."""

    def __init__(self, errors):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


def _require_msgpack():
    try:
        import msgpack
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise NuRecCodecError(["nurec_msgpack_unavailable"]) from exc
    return msgpack


def decode_nurec_bytes(raw: bytes) -> dict[str, Any]:
    """Decode a ``.nurec`` payload into its container document.

    Returns the ``nre_data`` mapping as stored, with bulk arrays left as raw
    bytes.  Use :func:`gaussian_arrays` to view them as numpy.
    """

    msgpack = _require_msgpack()
    if not raw[:2] == b"\x1f\x8b":
        raise NuRecCodecError(["nurec_payload_not_gzip"])
    try:
        document = msgpack.unpackb(
            gzip.decompress(raw), raw=False, strict_map_key=False
        )
    except Exception as exc:  # noqa: BLE001 - malformed container
        raise NuRecCodecError([f"nurec_payload_unreadable:{type(exc).__name__}"]) from exc
    if not isinstance(document, Mapping) or "nre_data" not in document:
        raise NuRecCodecError(["nurec_payload_missing_nre_data"])
    return dict(document["nre_data"])


def decode_nurec_file(path: str | Path) -> dict[str, Any]:
    return decode_nurec_bytes(Path(path).read_bytes())


def layer_precision(nre_data: Mapping[str, Any]) -> int:
    """The float precision the bulk arrays are stored at."""

    layers = ((nre_data.get("config") or {}).get("layers") or {})
    gaussians = layers.get("gaussians") or {}
    precision = gaussians.get("precision")
    if precision not in PRECISION_DTYPES:
        raise NuRecCodecError([f"nurec_unsupported_precision:{precision}"])
    return int(precision)


def gaussian_arrays(nre_data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """View the gaussian bulk arrays as numpy, still pre-activation."""

    dtype = PRECISION_DTYPES[layer_precision(nre_data)]
    state = nre_data.get("state_dict") or {}
    arrays: dict[str, np.ndarray] = {}
    errors: list[str] = []
    for name, width in GAUSSIAN_ARRAY_WIDTHS.items():
        key = f"{GAUSSIAN_KEY_PREFIX}{name}"
        if key not in state or f"{key}.shape" not in state:
            errors.append(f"nurec_array_missing:{name}")
            continue
        shape = tuple(int(v) for v in state[f"{key}.shape"])
        buffer = state[key]
        value = np.frombuffer(buffer, dtype=dtype)
        if value.size != int(np.prod(shape)) if shape else value.size != 0:
            errors.append(f"nurec_array_shape_mismatch:{name}")
            continue
        value = value.reshape(shape)
        if len(shape) != 2 or shape[1] != width:
            errors.append(f"nurec_array_width_unexpected:{name}:{shape}")
            continue
        arrays[name] = value
    if errors:
        raise NuRecCodecError(errors)
    counts = {name: int(value.shape[0]) for name, value in arrays.items()}
    if len(set(counts.values())) != 1:
        raise NuRecCodecError([f"nurec_array_count_disagreement:{counts}"])
    return arrays


def encode_nurec_bytes(nre_data: Mapping[str, Any]) -> bytes:
    """Encode a container document back to a ``.nurec`` payload.

    Deterministic: the gzip mtime is pinned, so encoding an unmodified document
    reproduces the original bytes and a round-trip is a real proof rather than
    an approximate one.
    """

    msgpack = _require_msgpack()
    packed = msgpack.packb({"nre_data": dict(nre_data)}, use_bin_type=True)
    buffer = io.BytesIO()
    with gzip.GzipFile(
        fileobj=buffer, mode="wb", compresslevel=9, mtime=DETERMINISTIC_GZIP_MTIME
    ) as handle:
        handle.write(packed)
    return buffer.getvalue()


def build_state_dict(arrays: Mapping[str, np.ndarray], *, precision: int) -> dict[str, Any]:
    """Lay gaussian arrays out the way the container stores them."""

    if precision not in PRECISION_DTYPES:
        raise NuRecCodecError([f"nurec_unsupported_precision:{precision}"])
    dtype = PRECISION_DTYPES[precision]
    errors: list[str] = []
    counts = set()
    state: dict[str, Any] = {}
    for name, width in GAUSSIAN_ARRAY_WIDTHS.items():
        if name not in arrays:
            errors.append(f"nurec_array_missing:{name}")
            continue
        value = np.ascontiguousarray(np.asarray(arrays[name]), dtype=dtype)
        if value.ndim != 2 or value.shape[1] != width:
            errors.append(f"nurec_array_width_unexpected:{name}:{value.shape}")
            continue
        if not np.isfinite(value.astype(np.float32)).all():
            # A non-finite parameter is not a renderable gaussian, and the
            # renderer would have no way to report which array carried it.
            errors.append(f"nurec_array_nonfinite:{name}")
            continue
        counts.add(int(value.shape[0]))
        key = f"{GAUSSIAN_KEY_PREFIX}{name}"
        state[key] = value.tobytes()
        state[f"{key}.shape"] = [int(value.shape[0]), int(value.shape[1])]
    if errors:
        raise NuRecCodecError(errors)
    if len(counts) != 1:
        raise NuRecCodecError([f"nurec_array_count_disagreement:{sorted(counts)}"])
    return state


def describe_volume(nre_data: Mapping[str, Any]) -> dict[str, Any]:
    """Report what a volume is, for a receipt, without rendering it."""

    arrays = gaussian_arrays(nre_data)
    layers = ((nre_data.get("config") or {}).get("layers") or {})
    gaussians = layers.get("gaussians") or {}
    particle = gaussians.get("particle") or {}
    return {
        "schema_version": NUREC_CODEC_SCHEMA_VERSION,
        "container_version": nre_data.get("version"),
        "model": nre_data.get("model"),
        "gaussian_count": int(arrays["positions"].shape[0]),
        "precision": layer_precision(nre_data),
        "density_activation": gaussians.get("density_activation"),
        "scale_activation": gaussians.get("scale_activation"),
        "rotation_activation": gaussians.get("rotation_activation"),
        "density_kernel_planar": particle.get("density_kernel_planar"),
        "radiance_sph_degree": particle.get("radiance_sph_degree"),
        "renderer": ((nre_data.get("config") or {}).get("renderer") or {}).get("name"),
    }


__all__ = [
    "GAUSSIAN_ARRAY_WIDTHS",
    "GAUSSIAN_KEY_PREFIX",
    "NUREC_CODEC_SCHEMA_VERSION",
    "NuRecCodecError",
    "build_state_dict",
    "decode_nurec_bytes",
    "decode_nurec_file",
    "describe_volume",
    "encode_nurec_bytes",
    "gaussian_arrays",
    "layer_precision",
]
