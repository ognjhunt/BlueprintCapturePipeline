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


def room_parts(*, floor=True, counter=True, table=False, walls=True):
    parts = {}
    if floor:
        parts["floor"] = _grid((-2.0, 0.0, -1.5), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05), 81, 61)
    if counter:
        parts["counter"] = _grid((-2.0, 0.9, -1.5), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05), 41, 13)
    if table:
        parts["table"] = _grid((-0.5, 0.75, -0.3), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05), 21, 13)
    if walls:
        parts["back_wall"] = _grid((-2.0, 0.0, -1.55), (0.05, 0.0, 0.0), (0.0, 0.05, 0.0), 81, 51)
        parts["side_wall"] = _grid((2.05, 0.0, -1.5), (0.0, 0.0, 0.05), (0.0, 0.05, 0.0), 61, 51)
    return parts


def write_room(splat_path: Path, mesh_path: Path, *, rotation=np.eye(3), offset_m=(0.0, 0.0, 0.0),
               meters_per_unit=1.0, noise_m=0.003, seed=0, sh_rest=False, **surfaces) -> dict:
    """Write the room so that provider units are ``(rotation @ metres + offset_m) / meters_per_unit``."""
    rng = np.random.default_rng(seed)
    rotation = np.asarray(rotation, dtype=float)
    parts, vertices, faces, spans, start = room_parts(**surfaces), [], [], {}, 0
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
