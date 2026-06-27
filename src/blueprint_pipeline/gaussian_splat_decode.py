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

    @property
    def opacity_sigmoid(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.opacity, -30.0, 30.0)))

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        return self.xyz.min(axis=0), self.xyz.max(axis=0)


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

    return SplatData(
        count=count,
        xyz=cols(["x", "y", "z"]),
        opacity=arr[:, index["opacity"]].astype(np.float32, copy=True),
        f_dc=cols(["f_dc_0", "f_dc_1", "f_dc_2"]),
        scales=cols(["scale_0", "scale_1", "scale_2"]),
        quats=cols(["rot_0", "rot_1", "rot_2", "rot_3"]),
        properties=tuple(names),
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
