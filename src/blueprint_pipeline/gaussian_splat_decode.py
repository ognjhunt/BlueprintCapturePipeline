"""Decode Blueprint Gaussian-splat capture assets into standard 3DGS data.

The pipeline's native splat format is PlayCanvas ``splat-transform`` *compressed* PLY
(packed-uint ``chunk``/``vertex``/``sh`` elements) plus gzip-wrapped ``.spz``. Neither
is directly consumable by simulator renderers (Isaac NuRec) or by numpy scene analysis,
so this module provides two layers:

* :func:`read_standard_3dgs_ply` — pure-numpy reader for a *standard* INRIA float 3DGS
  PLY (``x,y,z,f_dc_*,opacity,scale_*,rot_*[,f_rest_*]``), mapping columns by property
  name. Used for scene analysis (camera + robot-start-pose derivation).
* :func:`convert_to_standard_ply` — wraps the canonical ``@playcanvas/splat-transform``
  node CLI (the exact tool that generated these assets) to decode compressed PLY / SPZ
  into a standard PLY. Correct-by-construction; no hand-rolled bit unpacking.

This module never claims rendering or physics; it only decodes/inspects splat geometry.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

SPLAT_TRANSFORM_CLI_REL = (
    "tools/splat_render/node_modules/@playcanvas/splat-transform/bin/cli.mjs"
)

# Standard 3DGS float-PLY vertex properties required for geometry analysis.
_REQUIRED_3DGS_PROPS = (
    "x", "y", "z",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
    "f_dc_0", "f_dc_1", "f_dc_2",
)
_FLOAT_PLY_TYPES = {"float", "float32"}


@dataclass
class SplatData:
    """Standard 3DGS splat arrays (geometry + base color/opacity)."""

    count: int
    xyz: np.ndarray        # (N, 3) float32 — splat centers
    opacity: np.ndarray    # (N,)   float32 — raw logit (apply sigmoid for [0, 1])
    f_dc: np.ndarray       # (N, 3) float32 — SH band-0 base color
    scales: np.ndarray     # (N, 3) float32 — log-scale
    quats: np.ndarray      # (N, 4) float32 — rotation quaternion (rot_0..3)
    properties: tuple[str, ...]
    sh_rest: np.ndarray | None = None  # (N, 3*((degree+1)^2-1)) channel-major

    @property
    def opacity_sigmoid(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.opacity, -30.0, 30.0)))

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        return self.xyz.min(axis=0), self.xyz.max(axis=0)


@dataclass
class CompressedSplatChunkBounds:
    """Per-chunk quantization bounds read straight out of a compressed PLY header.

    The PlayCanvas compressed format stores, per 256-splat chunk, the float
    min/max of the splat CENTER positions (plus scale/color ranges) as plain
    ``float`` properties of the ``chunk`` element — no bit unpacking required to
    read them. That is enough for CPU scene analysis (world AABB, floor-height
    estimate, label-alignment checks) without decoding 600k packed vertices or
    shelling out to node.
    """

    chunk_count: int
    vertex_count: int
    min_xyz: np.ndarray  # (C, 3) float32 — per-chunk position minima
    max_xyz: np.ndarray  # (C, 3) float32 — per-chunk position maxima

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        return self.min_xyz.min(axis=0), self.max_xyz.max(axis=0)

    def floor_z_estimate(
        self,
        *,
        bin_m: float = 0.05,
        band_m: float = 2.0,
        prominence: float = 0.35,
    ) -> float:
        """Robust floor height: the LOWEST prominent mode of per-chunk z-minima.

        Neither the straight minimum (one floater away from nonsense) nor a low
        percentile (under-floor reconstruction fuzz spans tens of centimeters)
        is reliable; the floor plane is where chunk minima PILE UP. We histogram
        the minima above a floater-trimmed lower bound and return the center of
        the first (lowest) bin whose count reaches ``prominence`` of the tallest
        bin — sparse under-floor fuzz is skipped, dense wall/furniture bands
        higher up never shadow the floor because we scan bottom-up.
        """
        z = self.min_xyz[:, 2].astype(np.float64)
        lo = float(np.percentile(z, 0.5))
        band = z[(z >= lo) & (z <= lo + band_m)]
        if band.size == 0:
            return lo
        edges = np.arange(lo, lo + band_m + bin_m, bin_m)
        hist, edges = np.histogram(band, bins=edges)
        if hist.size == 0 or hist.max() == 0:
            return lo
        threshold = max(1.0, prominence * float(hist.max()))
        for i, count in enumerate(hist):
            if count >= threshold:
                return float(0.5 * (edges[i] + edges[i + 1]))
        return float(np.percentile(band, 5.0))


_CHUNK_BOUND_PROPS = ("min_x", "min_y", "min_z", "max_x", "max_y", "max_z")


def read_compressed_ply_chunk_bounds(path: str | Path) -> CompressedSplatChunkBounds:
    """Read the ``chunk`` element bounds of a PlayCanvas compressed 3DGS PLY.

    Raises ``ValueError`` when the file is not the compressed multi-element
    layout (use :func:`read_standard_3dgs_ply` for standard PLYs).
    """
    path = Path(path)
    with open(path, "rb") as handle:
        magic = handle.readline().strip()
        if magic != b"ply":
            raise ValueError("not a PLY file (missing 'ply' magic)")
        fmt: str | None = None
        elements: list[tuple[str, int, list[tuple[str, str]]]] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("unexpected EOF in PLY header")
            text = line.decode("latin-1").strip()
            if text.startswith("format"):
                fmt = text.split()[1]
            elif text.startswith("element"):
                parts = text.split()
                elements.append((parts[1], int(parts[2]), []))
            elif text.startswith("property") and elements:
                parts = text.split()
                elements[-1][2].append((parts[1], parts[-1]))
            elif text == "end_header":
                break
        offset = handle.tell()
    if fmt != "binary_little_endian":
        raise ValueError(f"unsupported PLY format '{fmt}' (need binary_little_endian)")
    names = [name for name, _, _ in elements]
    if "chunk" not in names:
        raise ValueError(
            "not_a_compressed_splat_ply: no 'chunk' element "
            "(use read_standard_3dgs_ply for standard 3DGS PLYs)"
        )
    if names[0] != "chunk":
        raise ValueError("unsupported compressed PLY layout: 'chunk' is not the first element")
    _, chunk_count, chunk_props = elements[0]
    if any(ptype not in _FLOAT_PLY_TYPES for ptype, _ in chunk_props):
        raise ValueError("non-float chunk property; unsupported compressed PLY variant")
    prop_names = [name for _, name in chunk_props]
    index = {name: i for i, name in enumerate(prop_names)}
    missing = [key for key in _CHUNK_BOUND_PROPS if key not in index]
    if missing:
        raise ValueError(f"missing chunk bound properties: {missing}")
    vertex_count = next((count for name, count, _ in elements if name == "vertex"), 0)
    ncol = len(chunk_props)
    flat = np.fromfile(path, dtype="<f4", count=chunk_count * ncol, offset=offset)
    if flat.size != chunk_count * ncol:
        raise ValueError(
            f"truncated chunk element: expected {chunk_count * ncol} floats, got {flat.size}"
        )
    arr = flat.reshape(chunk_count, ncol)
    min_xyz = arr[:, [index["min_x"], index["min_y"], index["min_z"]]].astype(np.float32, copy=True)
    max_xyz = arr[:, [index["max_x"], index["max_y"], index["max_z"]]].astype(np.float32, copy=True)
    return CompressedSplatChunkBounds(
        chunk_count=chunk_count,
        vertex_count=vertex_count,
        min_xyz=min_xyz,
        max_xyz=max_xyz,
    )


def _parse_ply_header(handle) -> tuple[str, int, list[tuple[str, str]], int]:
    """Return (format, vertex_count, [(type, name)], data_offset). Rejects multi-element
    (compressed) PLYs — those must go through :func:`convert_to_standard_ply` first."""
    magic = handle.readline().strip()
    if magic != b"ply":
        raise ValueError("not a PLY file (missing 'ply' magic)")
    fmt: str | None = None
    count = 0
    props: list[tuple[str, str]] = []
    seen_vertex_element = False
    while True:
        line = handle.readline()
        if not line:
            raise ValueError("unexpected EOF in PLY header")
        text = line.decode("latin-1").strip()
        if text.startswith("format"):
            fmt = text.split()[1]
        elif text.startswith("element vertex"):
            count = int(text.split()[-1])
            seen_vertex_element = True
        elif text.startswith("element"):
            # e.g. 'element chunk' / 'element sh' => compressed/multi-element layout
            raise ValueError(
                f"not_a_standard_3dgs_ply: unexpected element '{text}' "
                "(use convert_to_standard_ply to decode compressed/SPZ first)"
            )
        elif text.startswith("property") and seen_vertex_element:
            parts = text.split()
            props.append((parts[1], parts[-1]))
        elif text == "end_header":
            break
    if fmt is None:
        raise ValueError("PLY header missing 'format'")
    return fmt, count, props, handle.tell()


def read_standard_3dgs_ply(path: str | Path) -> SplatData:
    """Read a standard binary-little-endian float 3DGS PLY into :class:`SplatData`."""
    path = Path(path)
    with open(path, "rb") as handle:
        fmt, count, props, offset = _parse_ply_header(handle)
    if fmt != "binary_little_endian":
        raise ValueError(f"unsupported PLY format '{fmt}' (need binary_little_endian)")
    if any(ptype not in _FLOAT_PLY_TYPES for ptype, _ in props):
        raise ValueError("non-float vertex property; not a standard 3DGS float PLY")
    names = [name for _, name in props]
    index = {name: i for i, name in enumerate(names)}
    missing = [key for key in _REQUIRED_3DGS_PROPS if key not in index]
    if missing:
        raise ValueError(f"missing 3dgs properties: {missing}")
    ncol = len(props)
    flat = np.fromfile(path, dtype="<f4", count=count * ncol, offset=offset)
    if flat.size != count * ncol:
        raise ValueError(
            f"truncated PLY body: expected {count * ncol} floats, got {flat.size}"
        )
    arr = flat.reshape(count, ncol)

    def cols(keys: Sequence[str]) -> np.ndarray:
        return arr[:, [index[k] for k in keys]].astype(np.float32, copy=True)

    rest_names = sorted(
        (name for name in names if re.fullmatch(r"f_rest_[0-9]+", name)),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    sh_rest = cols(rest_names) if rest_names else None
    if sh_rest is not None:
        coefficient_count = 1 + (sh_rest.shape[1] // 3)
        degree = int(round(coefficient_count**0.5)) - 1
        if sh_rest.shape[1] % 3 or (degree + 1) ** 2 != coefficient_count:
            raise ValueError("invalid f_rest property count for spherical harmonics")

    return SplatData(
        count=count,
        xyz=cols(["x", "y", "z"]),
        opacity=arr[:, index["opacity"]].astype(np.float32, copy=True),
        f_dc=cols(["f_dc_0", "f_dc_1", "f_dc_2"]),
        scales=cols(["scale_0", "scale_1", "scale_2"]),
        quats=cols(["rot_0", "rot_1", "rot_2", "rot_3"]),
        properties=tuple(names),
        sh_rest=sh_rest,
    )


def write_standard_3dgs_ply(splat: SplatData, path: str | Path) -> Path:
    """Write a minimal standard 3DGS PLY (geometry + DC + opacity + scale + rot).

    Round-trippable with :func:`read_standard_3dgs_ply`. Used for tests and as a
    deterministic interchange artifact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    order = [
        ("x", splat.xyz[:, 0]), ("y", splat.xyz[:, 1]), ("z", splat.xyz[:, 2]),
        ("f_dc_0", splat.f_dc[:, 0]), ("f_dc_1", splat.f_dc[:, 1]), ("f_dc_2", splat.f_dc[:, 2]),
        ("opacity", splat.opacity),
        ("scale_0", splat.scales[:, 0]), ("scale_1", splat.scales[:, 1]), ("scale_2", splat.scales[:, 2]),
        ("rot_0", splat.quats[:, 0]), ("rot_1", splat.quats[:, 1]),
        ("rot_2", splat.quats[:, 2]), ("rot_3", splat.quats[:, 3]),
    ]
    if splat.sh_rest is not None:
        rest = np.asarray(splat.sh_rest, dtype=np.float32)
        if rest.ndim != 2 or rest.shape[0] != splat.count or rest.shape[1] % 3:
            raise ValueError("invalid sh_rest shape")
        coefficient_count = 1 + rest.shape[1] // 3
        degree = int(round(coefficient_count**0.5)) - 1
        if (degree + 1) ** 2 != coefficient_count:
            raise ValueError("invalid sh_rest coefficient count")
        order.extend((f"f_rest_{index}", rest[:, index]) for index in range(rest.shape[1]))
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {splat.count}"]
    header += [f"property float {name}" for name, _ in order]
    header.append("end_header\n")
    table = np.empty((splat.count, len(order)), dtype="<f4")
    for col, (_, values) in enumerate(order):
        table[:, col] = np.asarray(values, dtype="<f4")
    with open(path, "wb") as handle:
        handle.write(("\n".join(header)).encode("ascii"))
        handle.write(table.tobytes(order="C"))
    return path


def find_splat_transform_cli(repo_root: str | Path | None = None) -> Path | None:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    cli = root / SPLAT_TRANSFORM_CLI_REL
    return cli if cli.is_file() else None


def convert_to_standard_ply(
    src: str | Path,
    dst: str | Path,
    *,
    repo_root: str | Path | None = None,
    node: str = "node",
    timeout_seconds: int = 900,
) -> dict:
    """Decode a compressed PlayCanvas PLY / SPZ into a standard 3DGS PLY via the
    canonical ``splat-transform`` CLI. Returns a status dict (never raises on a clean
    tool failure); ``status == 'completed'`` means ``dst`` is a standard 3DGS PLY."""
    src = Path(src)
    dst = Path(dst)
    cli = find_splat_transform_cli(repo_root)
    if cli is None:
        return {
            "status": "blocked",
            "blockers": ["splat_transform_cli_unavailable"],
            "expected_cli_relpath": SPLAT_TRANSFORM_CLI_REL,
            "input": str(src),
        }
    if not src.is_file():
        return {"status": "blocked", "blockers": ["splat_source_missing"], "input": str(src)}
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            [node, str(cli), "-w", "-q", str(src), str(dst)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return {"status": "blocked", "blockers": ["node_runtime_unavailable"], "input": str(src)}
    except subprocess.TimeoutExpired:
        return {"status": "blocked", "blockers": ["splat_transform_timeout"], "input": str(src)}
    if proc.returncode != 0 or not dst.is_file():
        return {
            "status": "blocked",
            "blockers": ["splat_transform_decode_failed"],
            "returncode": proc.returncode,
            "stderr_tail": (proc.stderr or "")[-2000:],
            "input": str(src),
        }
    return {
        "status": "completed",
        "input": str(src),
        "output": str(dst),
        "output_bytes": dst.stat().st_size,
        "decoder": "playcanvas_splat_transform",
    }
