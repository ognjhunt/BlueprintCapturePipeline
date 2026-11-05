"""Support lookups read the collider in bounded memory and return exactly what whole-mesh reads did.

A website preparation (the compile inside blueprint-pubsub-handoff-listener, on an
8.3 GB host with no swap) passed 4.8 GB at production density: every support_under
copied the collider, merged it and took its whole face adjacency (#3339), and each
copy then waited for the cyclic collector. Lookups now read a block of faces at a
time (``website_support_geometry.MergedCollider``), and container, designed-start and
floor-grounding reads do too. Here they are held byte for byte to the whole-mesh
reads (``tests/support_geometry_reference.py``, verbatim from before) on synthetic
scanned rooms: small ones in the fast lane, and 1x and 2x capture-447's 9.2 m room
(6.6 M and 13 M triangles) when ``BLUEPRINT_SUPPORT_MEMORY_SCALES`` names them.
"""
from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import trimesh

from blueprint_pipeline import website_container_geometry as container
from blueprint_pipeline import website_support_geometry as geometry
from tests import support_geometry_reference as reference

REPO_ROOT = Path(__file__).resolve().parents[1]
#: capture-447's room: 9.2 m across, 3.25 M points, 6.5 M triangles ("1x").
SIDE, HEIGHT, STEP_1X = 9.2, 2.6, 0.0075
TABLE = ((-1.4, -0.3), (0.2, 0.6), 0.75)
COUNTER = ((2.6, -3.0), (4.4, -2.4), 0.90)
BIN = ((-3.0, 2.0), (-2.6, 2.3), 0.02, 0.27)


def _grid(u, v, point):
    U, V = np.meshgrid(u, v, indexing="ij")
    points = point(U, V).reshape(-1, 3)
    n, m = len(u), len(v)
    a = (np.arange(n - 1)[:, None] * m + np.arange(m - 1)[None, :]).ravel()
    return points, np.concatenate([np.c_[a, a + 1, a + m], np.c_[a + 1, a + m + 1, a + m]])


def _box(low, high):
    points = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    return points, np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                             [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])


def scanned_room(scale: float = 1.0, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic scanned room, Y up, metres; ``scale`` multiplies capture-447's triangle density.

    Scan jitter on the floor and dining table leaves few faces within 5 degrees of
    level, as on an Atlas collider, so they are fitted; the counter is exactly level.
    The table's two halves share exact-duplicate seam vertices; the counter's seam is
    copied 3e-10 m off (merged to the first copy), with a fin on one of its edges
    (three faces) and degenerate faces; an open bin stands on the floor.
    """
    rng = np.random.default_rng(seed)
    step, half = STEP_1X / math.sqrt(scale), SIDE / 2
    parts = []

    def axis(low, high):
        return np.linspace(low, high, max(2, int(round((high - low) / step)) + 1))

    def bumpy(base, amplitude, jitter):
        return lambda U, V: np.stack([U, base + amplitude * np.sin(U * 1.7) * np.cos(V * 1.3)
                                      + rng.uniform(-jitter, jitter, U.shape), V], axis=-1)

    parts.append(_grid(axis(-half, half), axis(-half, half), bumpy(0.0, 0.004, 0.2 * step)))
    for x in (-half, half):
        parts.append(_grid(axis(-half, half), axis(0.0, HEIGHT), lambda U, V, x=x: np.stack(
            [x + rng.uniform(-0.001, 0.001, U.shape), V, U], axis=-1)))
    for z in (-half, half):
        parts.append(_grid(axis(-half, half), axis(0.0, HEIGHT), lambda U, V, z=z: np.stack(
            [U, V, z + rng.uniform(-0.001, 0.001, U.shape)], axis=-1)))
    (tx0, tz0), (tx1, tz1), top = TABLE
    points, faces = _grid(axis(tx0, tx1), axis(tz0, tz1), bumpy(top, 0.002, 0.16 * step))
    columns = len(axis(tz0, tz1))
    seam = (len(axis(tx0, tx1)) // 2) * columns
    right, remap = faces.min(axis=1) >= seam, np.arange(len(points))
    remap[seam:seam + columns] = len(points) + np.arange(columns)
    faces[right] = remap[faces[right]]
    parts.append((np.concatenate([points, points[seam:seam + columns]]), faces))
    for x in (tx0, tx1):
        parts.append(_grid(axis(tz0, tz1), axis(top - 0.03, top), lambda U, V, x=x: np.stack(
            [np.full_like(U, x), V, U], axis=-1)))
    for z in (tz0, tz1):
        parts.append(_grid(axis(tx0, tx1), axis(top - 0.03, top), lambda U, V, z=z: np.stack(
            [U, V, np.full_like(U, z)], axis=-1)))
    for x in (tx0 + 0.05, tx1 - 0.1):
        for z in (tz0 + 0.05, tz1 - 0.1):
            parts.append(_box((x, 0.0, z), (x + 0.05, top - 0.03, z + 0.05)))
    (cx0, cz0), (cx1, cz1), counter = COUNTER
    level = lambda U, V: np.stack([U, np.full_like(U, counter), V], axis=-1)  # noqa: E731
    middle = (cz0 + cz1) / 2
    parts.append(_grid(axis(cx0, cx1), axis(cz0, middle), level))
    points, faces = _grid(axis(cx0, cx1), axis(middle, cz1), level)
    points[np.isclose(points[:, 2], middle), 0] += 3e-10
    parts.append((points, faces))
    parts.append(_grid(axis(cx0, cx1), axis(0.0, counter), lambda U, V: np.stack(
        [U, V, np.full_like(U, cz0)], axis=-1)))
    # A fin on one of the counter's own edges (copies of its two corners): three faces use that edge.
    along, across = axis(cx0, cx1), axis(cz0, middle)
    i, j = len(along) // 3, len(across) // 2
    parts.append((np.array([[along[i], counter, across[j]], [along[i], counter, across[j + 1]],
                            [along[i], counter + 0.1, (across[j] + across[j + 1]) / 2],
                            [along[i + 1], counter, across[j]], [along[i + 1], counter, across[j]]]),
                  np.array([[0, 1, 2], [3, 4, 3]])))
    (bx0, bz0), (bx1, bz1), floor, rim = BIN
    parts.append(_grid(axis(bx0, bx1), axis(bz0, bz1), lambda U, V: np.stack([U, np.full_like(U, floor), V], axis=-1)))
    for x in (bx0, bx1):
        parts.append(_grid(axis(bz0, bz1), axis(floor, rim), lambda U, V, x=x: np.stack(
            [np.full_like(U, x), V, U], axis=-1)))
    for z in (bz0, bz1):
        parts.append(_grid(axis(bx0, bx1), axis(floor, rim), lambda U, V, z=z: np.stack(
            [U, V, np.full_like(U, z)], axis=-1)))
    offsets = np.cumsum([0] + [len(points) for points, _ in parts[:-1]])
    return (np.concatenate([points for points, _ in parts]).astype(np.float64),
            np.concatenate([faces + offset for (_, faces), offset in zip(parts, offsets, strict=True)]).astype(np.int64))


(_TX0, _TZ0), (_TX1, _TZ1), _TOP = TABLE
(_CX0, _CZ0), (_CX1, _CZ1), _COUNTER_TOP = COUNTER
#: Footprints (Y up) as compile looks them up: (lower, upper, placement residual).
QUERIES = {
    "table_start": ([_TX0 + 0.2, _TOP + 0.004, _TZ0 + 0.2], [_TX0 + 0.51, _TOP + 0.06, _TZ0 + 0.72], None),
    "table_start_residual": ([_TX0 + 0.2, _TOP + 0.004, _TZ0 + 0.2], [_TX0 + 0.51, _TOP + 0.06, _TZ0 + 0.72], 0.072),
    "table_other_end": ([_TX1 - 0.5, _TOP + 0.004, _TZ0 + 0.2], [_TX1 - 0.19, _TOP + 0.06, _TZ0 + 0.72], 0.072),
    "table_edge": ([_TX1 - 0.1, _TOP + 0.004, _TZ0 + 0.2], [_TX1 + 0.1, _TOP + 0.06, _TZ0 + 0.5], None),
    "counter": ([_CX0 + 0.3, _COUNTER_TOP + 0.001, -2.85], [_CX0 + 0.6, _COUNTER_TOP + 0.05, -2.55], None),
    "counter_fin": ([_CX0 + 0.45, _COUNTER_TOP + 0.001, _CZ0 + 0.05], [_CX0 + 0.75, _COUNTER_TOP + 0.05, _CZ0 + 0.25],
                    0.05),
    "floor": ([0.5, 0.004, 1.0], [0.9, 0.3, 1.4], None),
    "bin_floor": ([-2.85, 0.021, 2.1], [-2.75, 0.1, 2.2], None),
    "mid_air": ([1.0, 1.5, 1.0], [1.2, 1.6, 1.2], None),
}


def _frame(vertices: np.ndarray, frame: str):
    """The room in another frame: ``z_up`` swaps Y and Z; ``minus_y`` reflects Y (up_sign -1)."""
    if frame == "z_up":
        return vertices[:, [0, 2, 1]], 2, 1
    if frame == "minus_y":
        moved = vertices.copy()
        moved[:, 1] *= -1
        return moved, 1, -1
    return vertices, 1, 1


def _bounds(lower, upper, frame: str):
    lower, upper = np.array(lower, dtype=float), np.array(upper, dtype=float)
    if frame == "z_up":
        return lower[[0, 2, 1]], upper[[0, 2, 1]]
    if frame == "minus_y":
        lower[1], upper[1] = -upper[1], -lower[1]
    return lower, upper


@pytest.fixture(scope="module")
def small_room():
    return scanned_room(0.02)


def _mesh(vertices, faces, **kwargs):
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False, **kwargs)


def _same(old, new) -> None:
    assert json.dumps(new) == json.dumps(old)


@pytest.mark.parametrize("frame", ["y_up", "z_up", "minus_y"])
def test_every_lookup_is_byte_identical_to_the_whole_mesh_read(small_room, frame):
    vertices, up, up_sign = _frame(small_room[0], frame)
    mesh = _mesh(vertices, small_room[1])
    before = (mesh.vertices.copy(), mesh.faces.copy())
    found = {}
    for name, (lower, upper, residual) in QUERIES.items():
        low, high = _bounds(lower, upper, frame)
        kwargs = {"up": up, "meters_per_unit": 1.0, "up_sign": up_sign, "placement_residual_m": residual}
        old = reference.support_under(mesh, low, high, **kwargs)
        new = geometry.support_under(mesh, low, high, **kwargs)
        _same(old, new)
        found[name] = None if new is None else new["basis"]
    # Both lookups ran here: a level counter, fitted bumpy surfaces, and refusals.
    assert found["counter"] == found["bin_floor"] == "reconstructed_triangle_footprint_contacts"
    assert found["table_start"] == found["floor"] == "fitted_surface_under_footprint"
    assert found["mid_air"] is None
    # Nothing is copied into, merged in or cached on the collider itself.
    assert np.array_equal(mesh.vertices, before[0]) and np.array_equal(mesh.faces, before[1])
    assert not {"triangles", "face_normals", "face_adjacency", "edges_sorted"} & set(mesh._cache.cache)


def test_blocks_never_leave_a_lone_row():
    """numpy's ``@`` on one (1, 3) row can differ in its last bit from the same row in a block."""
    for size in (2, 3, 7):
        for count in range(0, 40):
            geometry.SUPPORT_FACE_BLOCK = size
            try:
                blocks = list(geometry._row_blocks(count))
            finally:
                geometry.SUPPORT_FACE_BLOCK = 1 << 18
            assert [index for block in blocks for index in range(block.start, block.stop)] == list(range(count))
            assert all(block.stop - block.start >= 2 for block in blocks) or count == 1
            assert all(block.stop - block.start <= size + 1 for block in blocks)


@pytest.mark.parametrize("block,bucket", [(7, 64), (1000, 5)])
def test_small_blocks_and_merge_buckets_change_no_byte(monkeypatch, block, bucket):
    """Every block boundary and every merge bucket gives the whole read's answer, in both up senses."""
    vertices, faces = scanned_room(0.006)
    monkeypatch.setattr(geometry, "SUPPORT_FACE_BLOCK", block)
    monkeypatch.setattr(geometry, "MERGE_BUCKET_VERTICES", bucket)
    found = set()
    for frame in ("y_up", "minus_y"):
        moved, up, up_sign = _frame(vertices, frame)
        mesh = _mesh(moved, faces)
        for name in ("table_start_residual", "counter_fin", "floor", "bin_floor"):
            lower, upper, residual = QUERIES[name]
            low, high = _bounds(lower, upper, frame)
            kwargs = {"up": up, "meters_per_unit": 1.0, "up_sign": up_sign, "placement_residual_m": residual}
            old = reference.support_under(mesh, low, high, **kwargs)
            _same(old, geometry.support_under(mesh, low, high, **kwargs))
            found.add(None if old is None else old["basis"])
    assert found == {"fitted_surface_under_footprint", "reconstructed_triangle_footprint_contacts"}


def test_merge_keeps_texture_seams_apart_and_ignores_given_normals(small_room):
    """As ``merge_vertices`` on a copy: UVs split a seam, cached vertex normals (dropped by the copy) do not."""
    vertices, faces = small_room
    uv = np.c_[vertices[:, 0], vertices[:, 2]] / SIDE + 0.5
    (tx0, tz0), (tx1, tz1), _ = TABLE
    seam = np.flatnonzero((np.abs(vertices[:, 0] - (tx0 + tx1) / 2) < 1e-6) & (np.abs(vertices[:, 1] - _TOP) < 0.02)
                          & (vertices[:, 2] > tz0 + 1e-6) & (vertices[:, 2] < tz1 - 1e-6))
    uv[seam[len(seam) // 2:]] += 0.25  # the right half's copies of the seam get other UVs
    textured = _mesh(vertices, faces, visual=trimesh.visual.TextureVisuals(uv=uv))
    normals = np.random.default_rng(3).normal(size=vertices.shape)
    with_normals = _mesh(vertices, faces, vertex_normals=normals / np.linalg.norm(normals, axis=1)[:, None])
    plain = _mesh(vertices, faces)
    for mesh in (textured, with_normals):
        for name in ("table_start", "table_start_residual", "table_other_end", "counter"):
            lower, upper, residual = QUERIES[name]
            kwargs = {"up": 1, "meters_per_unit": 1.0, "placement_residual_m": residual}
            _same(reference.support_under(mesh, lower, upper, **kwargs),
                  geometry.support_under(mesh, lower, upper, **kwargs))
    lower, upper, residual = QUERIES["table_start"]
    split, whole = (len(geometry.support_under(mesh, lower, upper, up=1, meters_per_unit=1.0)["face_indices"])
                    for mesh in (textured, plain))
    assert split < whole  # the textured seam is two surfaces, as merge_vertices leaves it


def test_signed_zero_bounds_are_the_whole_reads(monkeypatch):
    """A support whose extreme is 0.0 on some corners and -0.0 on others keeps the whole read's sign."""
    grid = _grid(np.linspace(0.0, 1.0, 41), np.linspace(-0.5, 0.5, 41),
                 lambda U, V: np.stack([U, np.full_like(U, 0.7), V], axis=-1))
    vertices, faces = grid
    vertices[(vertices[:, 0] == 0) & (vertices[:, 2] > 0), 0] = -0.0
    mesh = _mesh(vertices, faces)
    monkeypatch.setattr(geometry, "SUPPORT_FACE_BLOCK", 16)
    lower, upper = [0.02, 0.701, -0.05], [0.08, 0.75, 0.05]
    old = reference.support_under(mesh, lower, upper, up=1, meters_per_unit=1.0)
    _same(old, geometry.support_under(mesh, lower, upper, up=1, meters_per_unit=1.0))
    assert old["aabb_min"][0] == 0


@pytest.mark.parametrize("frame", ["y_up", "minus_y"])
def test_floor_grounding_and_container_interior_are_byte_identical(small_room, frame):
    vertices, up, up_sign = _frame(small_room[0], frame)
    mesh = _mesh(vertices, small_room[1])
    for lower, upper, floor in (([1.5, 0.12, -1.5], [2.1, 0.95, -0.9], 0.0),
                                ([_CX0 + 0.2, _COUNTER_TOP + 0.05, -2.9], [_CX0 + 0.8, 1.6, -2.5], _COUNTER_TOP)):
        low, high = _bounds(lower, upper, frame)
        height = floor if up_sign == 1 else -floor
        kwargs = {"up": up, "meters_per_unit": 1.0, "floor_height": height, "up_sign": up_sign}
        old = reference.ground_on_observed_floor(mesh, low, high, **kwargs)
        _same(old, geometry.ground_on_observed_floor(mesh, low, high, **kwargs))
    (bx0, bz0), (bx1, bz1), floor, rim = BIN
    low, high = _bounds([bx0 - 0.01, floor - 0.01, bz0 - 0.01], [bx1 + 0.01, rim, bz1 + 0.01], frame)
    kwargs = {"subject_dimensions": [0.1, 0.1, 0.1], "up": up, "meters_per_unit": 1.0, "up_sign": up_sign}
    old = reference.container_interior(mesh, low, high, **kwargs)
    assert old is not None
    _same(old, container.container_interior(mesh, low, high, **kwargs))


def test_a_repeat_lookup_holds_blocks_not_the_collider(small_room, monkeypatch):
    """After the first lookup's merged read, a lookup holds a few blocks and one byte per face.

    The whole-mesh read held about 390 bytes per face (a copy, its merge, its
    triangles and face adjacency); the blocked read holds a few blocks of faces and
    boolean masks over them.
    """
    monkeypatch.setattr(geometry, "SUPPORT_FACE_BLOCK", 4096)
    mesh = _mesh(*small_room)
    lower, upper, residual = QUERIES["table_start_residual"]
    kwargs = {"up": 1, "meters_per_unit": 1.0, "placement_residual_m": residual}
    peaks = {}
    for name, lookup in (("whole", reference.support_under), ("blocked", geometry.support_under)):
        lookup(mesh, lower, upper, **kwargs)
        gc.collect()
        tracemalloc.start()
        try:
            lookup(mesh, [lower[0] + 0.01, *lower[1:]], [upper[0] + 0.01, *upper[1:]], **kwargs)
            peaks[name] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    faces = len(mesh.faces)
    assert peaks["whole"] > 200 * faces
    assert peaks["blocked"] < 16 * faces + 8 * 4096 * 9 * 8, peaks


_PROGRAM = r"""
import gc, hashlib, json, resource, sys, trimesh
from pathlib import Path
implementation, collider, names = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
if implementation == "whole":
    # Each whole-mesh copy waits for the cyclic collector; collect often so one lookup fits on this host.
    gc.set_threshold(50, 5, 5)
    from tests import support_geometry_reference as support
    container = support
else:
    from blueprint_pipeline import website_support_geometry as support
    from blueprint_pipeline import website_container_geometry as container
from tests.test_website_support_memory import BIN, QUERIES

def peak_mb():
    rows = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    return int(rows["VmHWM"].split()[0]) // 1024

mesh = trimesh.load(collider, force="mesh", process=False)
loaded, digests = peak_mb(), {}
for name in names:
    if name == "container":
        (bx0, bz0), (bx1, bz1), floor, rim = BIN
        value = container.container_interior(mesh, [bx0 - 0.01, floor - 0.01, bz0 - 0.01], [bx1 + 0.01, rim, bz1 + 0.01],
                                             subject_dimensions=[0.1, 0.1, 0.1], up=1, meters_per_unit=1.0)
    elif name == "grounding":
        value = support.ground_on_observed_floor(mesh, [1.5, 0.12, -1.5], [2.1, 0.95, -0.9], up=1,
                                                 meters_per_unit=1.0, floor_height=0.0)
    else:
        lower, upper, residual = QUERIES[name]
        value = support.support_under(mesh, lower, upper, up=1, meters_per_unit=1.0, placement_residual_m=residual)
    digests[name] = hashlib.sha256(json.dumps(value).encode()).hexdigest()
print(json.dumps({"faces": len(mesh.faces), "loaded_mb": loaded, "peak_mb": peak_mb(), "digests": digests}))
"""
#: What the production-density rooms look up, as compile does: start and destination on the
#: fitted table, the level counter, the whole fitted floor, a container and floor grounding.
LARGE_LOOKUPS = ("table_start_residual", "table_other_end", "counter", "floor", "container", "grounding")


def _scales() -> list[float]:
    return [float(value) for value in os.environ.get("BLUEPRINT_SUPPORT_MEMORY_SCALES", "").split(",") if value]


@pytest.mark.slow
# An operator's opt-in measurement (several GB per process at 2x): the hermetic full lane, which
# refuses any skip, deselects it rather than skipping it on every cross-cutting run.
@pytest.mark.external_runtime
@pytest.mark.skipif(not _scales(), reason="set BLUEPRINT_SUPPORT_MEMORY_SCALES (for example 1,2) to run")
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads VmHWM from /proc")
@pytest.mark.parametrize("scale", _scales())
def test_production_density_rooms_are_byte_identical_in_a_fraction_of_the_memory(tmp_path, scale):
    """At 1x and 2x capture-447's density, as compile loads it: a GLB read with trimesh.load.

    The blocked lookups run in one process, as a preparation runs them. The whole-mesh
    reads need several GB each at 2x: each runs in its own process, one after another.
    """
    collider = tmp_path / "room.glb"
    _mesh(*scanned_room(scale)).export(collider, file_type="glb")
    gc.collect()
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "src")])}

    def run(implementation: str, names) -> dict:
        done = subprocess.run([sys.executable, "-c", _PROGRAM, implementation, str(collider), ",".join(names)],
                              capture_output=True, text=True, env=environment, cwd=REPO_ROOT, check=True)
        return json.loads(done.stdout.strip().splitlines()[-1])

    try:
        blocked = run("blocked", LARGE_LOOKUPS)
        whole = {name: run("whole", [name]) for name in LARGE_LOOKUPS}
    finally:
        collider.unlink(missing_ok=True)
    print(json.dumps({"scale": scale, "faces": blocked["faces"], "blocked_peak_mb": blocked["peak_mb"],
                      "whole_peak_mb": {name: row["peak_mb"] for name, row in whole.items()},
                      "loaded_mb": blocked["loaded_mb"]}))
    assert blocked["digests"] == {name: row["digests"][name] for name, row in whole.items()}
    # All six blocked lookups together add less than a quarter of what the largest whole-mesh one adds alone.
    added = max(row["peak_mb"] - row["loaded_mb"] for row in whole.values())
    assert (blocked["peak_mb"] - blocked["loaded_mb"]) * 4 < added, (blocked, whole)
