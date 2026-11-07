"""A room written as an Atlas provider world, for levelling tests.

Surfaces in metres, gravity-up with the floor at y = 0: a floor seen across the
room, a counter top at 0.9 m, a small table at 0.75 m and two walls. The world
is turned by ``rotation`` (a tilt, as Atlas returns it), moved by ``offset_m``
and written in provider units as a supported mesh (GLB) and a depth-fused
splat (standard 3DGS PLY) of thin discs lying in the tilted level surfaces.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh

from blueprint_pipeline.atlas_depth_splat import PLY_PROPERTIES
from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply


def tilt(degrees: float, axis=(1.0, 0.0, 0.6)) -> np.ndarray:
    """Rotation by ``degrees`` about a horizontal axis."""
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    angle = math.radians(degrees)
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * k @ k


def _grid(origin, u, v, nu, nv):
    i, j = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    vertices = (np.asarray(origin, float) + i.reshape(-1, 1) * np.asarray(u, float)
                + j.reshape(-1, 1) * np.asarray(v, float))
    k = (np.arange(nu - 1)[:, None] * nv + np.arange(nv - 1)[None, :]).reshape(-1)
    return vertices, np.concatenate([np.c_[k, k + nv, k + 1], np.c_[k + 1, k + nv, k + nv + 1]])


def room_parts(*, floor=True, counter=True, table=False, shelves=False, walls=True):
    parts = {}
    if floor:
        parts["floor"] = _grid((-2.0, 0.0, -1.5), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05), 81, 61)
    if counter:
        parts["counter"] = _grid((-2.0, 0.9, -1.5), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05), 41, 13)
    if table:
        parts["table"] = _grid((-0.5, 0.75, -0.3), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05), 21, 13)
    if shelves:
        # Warehouse rack levels: parallel to the floor, above it.
        for height in (1.5, 3.0):
            parts[f"shelf_{height}"] = _grid((-2.0, height, -1.5), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05), 41, 21)
    if walls:
        parts["back_wall"] = _grid((-2.0, 0.0, -1.55), (0.05, 0.0, 0.0), (0.0, 0.05, 0.0), 81, 51)
        parts["side_wall"] = _grid((2.05, 0.0, -1.5), (0.0, 0.0, 0.05), (0.0, 0.05, 0.0), 61, 51)
    return parts


def aisle_parts(*, seed=0, step=0.03, floor_width=1.3, hall_rows=1, floor_tilt_degrees=0.0, floor_wave_m=0.0):
    """A floor strip between two bookcases, in a hall of other shelving.

    The aisle floor is the largest level plane but holds only about 5% of the
    surface: shelf boards on both sides, book spines at uneven depths, back
    panels, an end bookcase and the far shelving around the hall make the rest.
    A grid ``step`` near an Atlas mesh's edge length (3 cm) gives each part its
    share of vertices. ``floor_width`` narrows the observed floor strip down the
    aisle's middle (0: none survived), and ``hall_rows`` adds rows of far shelving.
    ``floor_tilt_degrees`` tilts the floor along the aisle against everything else, as a
    reconstructed world's floor and shelves can disagree by a degree or two. ``floor_wave_m``
    makes the floor rise and fall that much along the aisle, as a reconstructed floor can.
    """
    rng = np.random.default_rng(seed)
    n = lambda length: int(round(length / step)) + 1  # noqa: E731
    parts = {}
    if floor_width:
        floor, triangles = _grid((-3.0, 0.0, -floor_width / 2), (step, 0.0, 0.0), (0.0, 0.0, step),
                                 n(6.0), n(floor_width))
        floor[:, 1] += floor[:, 0] * math.tan(math.radians(floor_tilt_degrees))
        floor[:, 1] += floor_wave_m * np.sin(2.0 * math.pi * floor[:, 0] / 1.7)
        parts["floor"] = (floor, triangles)
    boards = (0.10, 0.45, 0.80, 1.15, 1.50, 1.85, 2.20)
    for side in (-1.0, 1.0):
        front, back = side * 0.65, side * 0.95
        for height in boards:
            parts[f"board_{side:+.0f}_{height}"] = _grid((-3.0, height, front), (step, 0.0, 0.0),
                                                          (0.0, 0.0, side * step), n(6.0), n(0.3))
        for low in boards[:-1]:
            spines, triangles = _grid((-3.0, low + 0.01, front + side * 0.02), (step, 0.0, 0.0),
                                      (0.0, step, 0.0), n(6.0), n(0.27))
            # Books stand at uneven depths: each column of spines sits back by its own amount.
            depth = rng.uniform(0.0, 0.06, n(6.0)).repeat(n(0.27))
            spines[:, 2] += side * depth
            parts[f"spines_{side:+.0f}_{low}"] = (spines, triangles)
        parts[f"back_{side:+.0f}"] = _grid((-3.0, 0.0, back), (step, 0.0, 0.0), (0.0, step, 0.0), n(6.0), n(2.2))
    parts["end_bookcase"] = _grid((-3.3, 0.0, -0.65), (0.0, 0.0, step), (0.0, step, 0.0), n(1.3), n(2.2))
    for row in range(hall_rows):
        for z in (-6.0 - row, 6.0 + row):
            parts[f"hall_shelving_{z:+.0f}"] = _grid((-6.0, 0.0, z), (step, 0.0, 0.0), (0.0, step, 0.0),
                                                     n(12.0), n(2.4))
    return parts


def terrain_parts(*, seed=0, step=0.05):
    """Open rough ground: rolling, with bumps everywhere and no level surface."""
    rng = np.random.default_rng(seed)
    vertices, triangles = _grid((-4.0, 0.0, -4.0), (step, 0.0, 0.0), (0.0, 0.0, step), 161, 161)
    x, z = vertices[:, 0], vertices[:, 2]
    vertices[:, 1] = (0.8 * np.sin(0.9 * x) * np.cos(0.7 * z) + 0.48 * np.sin(2.1 * x + 1.0)
                      + 0.32 * np.cos(1.7 * z + 0.4) + rng.normal(0.0, 0.05, len(x)))
    return {"ground": (vertices, triangles)}


def write_room(splat_path: Path, mesh_path: Path, *, rotation=np.eye(3), offset_m=(0.0, 0.0, 0.0),
               meters_per_unit=1.0, noise_m=0.003, seed=0, sh_rest=False, aisle=False, terrain=False,
               **surfaces) -> dict:
    """Write the room so that provider units are ``(rotation @ metres + offset_m) / meters_per_unit``.

    ``aisle`` writes ``aisle_parts`` (``surfaces`` are its options) and ``terrain``
    writes ``terrain_parts``, instead of the room.
    """
    rng = np.random.default_rng(seed)
    rotation = np.asarray(rotation, dtype=float)
    parts = (aisle_parts(seed=seed, **surfaces) if aisle else terrain_parts(seed=seed) if terrain
             else room_parts(**surfaces))
    vertices, faces, spans, start = [], [], {}, 0
    for name, (points, triangles) in parts.items():
        vertices.append(points + (rng.normal(0.0, noise_m, points.shape) if noise_m else 0.0))
        faces.append(triangles + start)
        spans[name] = (start, start + len(points))
        start += len(points)
    world = np.concatenate(vertices)
    units = (world @ rotation.T + np.asarray(offset_m, dtype=float)) / meters_per_unit
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=units, faces=np.concatenate(faces), process=False).export(mesh_path)
    up, count = rotation @ [0.0, 1.0, 0.0], len(units)
    # The rotation taking each disc's thin local +Z onto the tilted up (wxyz).
    quats = np.tile([1.0 + up[2], -up[1], up[0], 0.0], (count, 1))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    splat = SplatData(count=count, xyz=units.astype(np.float32), opacity=np.zeros(count, np.float32),
                      f_dc=np.zeros((count, 3), np.float32),
                      scales=np.tile(np.log([0.02, 0.02, 0.002]), (count, 1)).astype(np.float32),
                      quats=quats.astype(np.float32), properties=PLY_PROPERTIES,
                      sh_rest=np.zeros((count, 9), np.float32) if sh_rest else None)
    write_standard_3dgs_ply(splat, splat_path)
    return {"world_m": world, "parts": spans}


def provider_units(points_m, *, rotation=np.eye(3), offset_m=(0.0, 0.0, 0.0), meters_per_unit=1.0) -> np.ndarray:
    return (np.asarray(points_m, dtype=float) @ np.asarray(rotation, dtype=float).T
            + np.asarray(offset_m, dtype=float)) / meters_per_unit
