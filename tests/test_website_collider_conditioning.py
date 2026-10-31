"""Website colliders are flattened where objects are placed, before they are sealed.

capture-447ae4f4: the Atlas dining table stood 6 mm RMS off its fit with a 7.7 degree
median face tilt, so the 31 x 52 cm tray rested on its highest bumps and could rock.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from blueprint_pipeline import website_collider_conditioning as conditioning
from blueprint_pipeline import website_task_preparation as preparation
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.website_support_geometry import same_support_surface, support_under

TOP = 0.75
#: Z-up source points into a Y-up runtime frame (Atlas is Y-up): a proper rotation.
Y_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def _table(up: int = 2):
    """A bumpy 1.2 x 0.8 m table top at 0.75 m, its steep skirt, a spike, a bump and a box on it.

    Every face has its own three vertices, as in an Atlas collider. Returns the mesh
    and the named vertex positions (in the mesh's frame) the tests follow.
    """
    rng = np.random.default_rng(3)
    xs, ys = np.linspace(-0.6, 0.6, 49), np.linspace(-0.4, 0.4, 33)
    grid = np.array([[x, y, TOP + rng.uniform(-0.003, 0.003)] for y in ys for x in xs])
    index = {(i, j): j * len(xs) + i for j in range(len(ys)) for i in range(len(xs))}
    spike, bump = index[(42, 26)], index[(30, 8)]
    grid[spike, 2] = TOP + 0.05   # a spike: its faces leave the fit, so it is never the surface
    grid[bump, 2] = TOP + 0.008   # a bump: still the surface, but outside its fit band
    faces = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            a, b, c, d = index[(i, j)], index[(i + 1, j)], index[(i, j + 1)], index[(i + 1, j + 1)]
            faces += [[a, b, d], [a, d, c]]
    ring = ([index[(i, 0)] for i in range(len(xs))] + [index[(len(xs) - 1, j)] for j in range(1, len(ys))]
            + [index[(i, len(ys) - 1)] for i in range(len(xs) - 2, -1, -1)]
            + [index[(0, j)] for j in range(len(ys) - 2, 0, -1)])
    below = len(grid) + np.arange(len(ring))
    vertices = np.vstack([grid, grid[ring] - [0, 0, 0.05]])
    for k in range(len(ring)):
        a, b = ring[k], ring[(k + 1) % len(ring)]
        faces += [[a, b, below[(k + 1) % len(ring)]], [a, below[(k + 1) % len(ring)], below[k]]]
    box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    box.apply_translation((0.4, -0.25, TOP + 0.05))
    faces = np.vstack([faces, box.faces + len(vertices)])
    vertices = np.vstack([vertices, box.vertices])
    named = {"spike": vertices[spike], "bump": vertices[bump], "skirt": vertices[below],
             "box": np.asarray(box.vertices)}
    exploded = vertices[faces].reshape(-1, 3)
    if up == 1:
        exploded = exploded @ Y_UP.T
        named = {key: value @ Y_UP.T for key, value in named.items()}
    mesh = trimesh.Trimesh(vertices=exploded, faces=np.arange(len(exploded)).reshape(-1, 3), process=False)
    return mesh, named


def _footprint(centre, up: int = 2, size=(0.31, 0.2)):
    """A tray's footprint resting on the table, in the mesh's frame."""
    lower = np.array([centre[0] - size[0] / 2, centre[1] - size[1] / 2, TOP])
    upper = np.array([centre[0] + size[0] / 2, centre[1] + size[1] / 2, TOP + 0.045])
    if up == 1:
        corners = np.array([lower, upper]) @ Y_UP.T
        return corners.min(axis=0), corners.max(axis=0)
    return lower, upper


def _rows(vertices, points):
    keys = {row.tobytes() for row in np.asarray(points, dtype=np.float64)}
    return np.array([row.tobytes() in keys for row in vertices])


@pytest.mark.parametrize("up", [2, 1])
def test_flattening_moves_only_in_band_vertices_of_the_support_faces(up):
    mesh, named = _table(up)
    start = support_under(mesh, *_footprint((-0.3, -0.05), up), up=up, meters_per_unit=1.0)
    destination = support_under(mesh, *_footprint((0.2, 0.05), up), up=up, meters_per_unit=1.0)
    assert start["basis"] == destination["basis"] == "fitted_surface_under_footprint"
    assert same_support_surface(start, destination)
    placements = [{"role": "start_support", "support": start}, {"role": "destination_support", "support": destination}]
    before = np.asarray(mesh.vertices)
    after, record = conditioning.flatten_placement_surfaces(before, mesh.faces, placements, up=up, up_sign=1,
                                                            meters_per_unit=1.0, placement_uncertainty_m=None)
    [surface] = record["surfaces"]
    assert surface["status"] == "flattened" and surface["roles"] == ["start_support", "destination_support"]
    faces = sorted(set(start["face_indices"]) | set(destination["face_indices"]))
    assert surface["face_indices"] == faces
    top = max(start["top_runtime_units"], destination["top_runtime_units"])
    assert surface["plane"] == {"basis": "level_at_fitted_top", "up_axis_index": up, "up_sign": 1,
                                "top_runtime_units": top}
    # Only heights change, and only of positions every face of which is the surface's own.
    changed = np.any(after != before, axis=1)
    assert changed.any()
    assert np.array_equal(np.delete(after, up, axis=1), np.delete(before, up, axis=1))
    member = np.zeros(len(mesh.faces), dtype=bool)
    member[faces] = True
    assert not changed[mesh.faces[~member]].any()
    # By position too: no other face shares a moved vertex's position, so nothing tears.
    at_moved_position = _rows(before, before[changed])
    assert member[at_moved_position[mesh.faces].any(axis=1)].all()
    # Every moved vertex sat within the band of the fit and moved no further: it now lies at the top.
    normal = np.asarray(surface["fitted_plane"]["normal_upright"])
    point = np.asarray(surface["fitted_plane"]["point_upright_runtime_units"])
    band = surface["band_m"]
    assert surface["band_basis"] == "3x_fit_rms" and band == pytest.approx(3 * surface["fitted_plane"]["rms_m"])
    assert np.all(np.abs((before[changed] - point) @ normal) <= band + 1e-9)
    assert surface["level_to_fit_max_m"] <= band
    assert np.allclose(after[changed, up], top, atol=0)
    assert surface["max_displacement_m"] == pytest.approx(np.abs(after[changed, up] - before[changed, up]).max())
    # The spike, the out-of-band bump, the skirt (the table's edge) and the box stand as captured.
    for name in ("spike", "bump", "skirt", "box"):
        assert not changed[_rows(before, np.atleast_2d(named[name]))].any(), name
    assert surface["out_of_band_vertex_count"] >= 1 and surface["rim_vertex_count"] > 0
    assert surface["after"]["face_tilt_median_degrees"] == 0.0 and surface["after"]["level_face_fraction"] > 0.8
    assert surface["before"]["face_tilt_median_degrees"] > 0.5
    # Read back as every later reader reads it: one surface at the patch's top.
    flattened = trimesh.Trimesh(vertices=after, faces=mesh.faces, process=False)
    found = [support_under(flattened, *_footprint(centre, up), up=up, meters_per_unit=1.0)
             for centre in ((-0.3, -0.05), (0.2, 0.05), (0.0, 0.1))]
    assert all(row is not None and row["top_runtime_units"] == pytest.approx(top, abs=1e-9) for row in found)
    # The kept bump still carries a footprint over it, on the same surface.
    over_bump = support_under(flattened, *_footprint((0.0, -0.2), up), up=up, meters_per_unit=1.0)
    assert over_bump["top_runtime_units"] > top
    assert all(same_support_surface(found[0], row) for row in [*found[1:], over_bump])


def _base_scene(tmp_path: Path, mesh) -> dict:
    path = tmp_path / "atlas_mesh_levelled.glb"
    mesh.export(path)
    splat = tmp_path / "atlas_depth_splat_levelled.ply"
    splat.write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n")
    return {"splat_path": str(splat), "splat_digest": _sha256_file(splat), "collision_mesh_path": str(path),
            "collision_mesh_digest": _sha256_file(path), "collision_binding_id": "website-collider-raw",
            "up_axis": "Z", "meters_per_unit": 1.0}


def _condition(tmp_path, base, placements, **options):
    collider = trimesh.load(base["collision_mesh_path"], force="mesh", process=False)
    return conditioning.condition_placement_collider(base_scene=base, collider=collider, placements=placements,
        up=2, up_sign=1, meters_per_unit=1.0, placement_uncertainty_m=None, output_root=tmp_path / "out", **options)


def test_measured_or_authored_supports_are_untouched(tmp_path):
    # The development test's authored surface (website_development_test): exact level faces.
    surface = trimesh.creation.box(extents=(1.2, 0.8, 0.05))
    surface.apply_translation((0.0, 0.0, TOP - 0.025))
    authored = support_under(surface, *_footprint((0.0, 0.0)), up=2, meters_per_unit=1.0)
    assert authored["basis"] == "reconstructed_triangle_footprint_contacts"
    base = _base_scene(tmp_path, surface)
    footprint = _footprint((0.0, 0.0))
    assert _condition(tmp_path, base, [{"role": "start_support", "support": authored,
                                        "footprint": footprint}]) == {"status": "unchanged"}
    after, record = conditioning.flatten_placement_surfaces(surface.vertices, surface.faces,
        [{"role": "start_support", "support": authored}], up=2, up_sign=1, meters_per_unit=1.0,
        placement_uncertainty_m=None)
    assert np.array_equal(after, surface.vertices) and record["surfaces"] == []
    assert record["untouched_supports"] == [{"role": "start_support", "basis": authored["basis"],
                                             "reason": "not_a_fitted_surface"}]
    # A surface someone measured is never reshaped to a fit.
    mesh, _ = _table()
    base = _base_scene(tmp_path, mesh)
    fitted = support_under(mesh, *footprint, up=2, meters_per_unit=1.0)
    measured = {**fitted, "physical_measurement": True}
    result = _condition(tmp_path, base, [{"role": "start_support", "support": measured, "footprint": footprint}])
    assert result["status"] == "kept_raw" and result["record"]["reasons"] == ["measured_surface"]
    assert not (tmp_path / "out" / "collider").exists()
    assert _sha256_file(Path(base["collision_mesh_path"])) == base["collision_mesh_digest"]


def test_a_scene_without_supports_keeps_its_collider_digest_unless_smoothing_applies(tmp_path):
    mesh, named = _table()
    base = _base_scene(tmp_path, mesh)
    assert _condition(tmp_path, base, []) == {"status": "unchanged"}
    assert not (tmp_path / "out").exists()
    # Smoothing, when a caller asks for it, is the only change to a scene with no support.
    loose = {**conditioning.TAUBIN_DEFAULTS, "max_displacement_m": 1.0, "max_aabb_change_m": 1.0}
    smoothed = _condition(tmp_path, base, [], smoothing=loose)
    assert smoothed["status"] == "conditioned"
    sealed = smoothed["base_scene"]
    assert sealed["collision_mesh_digest"] != base["collision_mesh_digest"]
    assert sealed["collision_mesh_digest"] == _sha256_file(Path(sealed["collision_mesh_path"]))
    assert (sealed["splat_path"], sealed["splat_digest"]) == (base["splat_path"], base["splat_digest"])
    record = smoothed["record"]["smoothing"]
    assert record["status"] == "smoothed" and record["method"] == "taubin"
    assert {key: record[key] for key in ("lamb", "nu", "iterations")} == {"lamb": 0.5, "nu": 0.53, "iterations": 10}
    assert record["measured_max_displacement_m"] > 0
    raw = trimesh.load(base["collision_mesh_path"], force="mesh", process=False).vertices
    after = trimesh.load(sealed["collision_mesh_path"], force="mesh", process=False).vertices
    assert not np.array_equal(after, raw)
    # Walls and edges (the skirt) are pinned, and the box's sides with them.
    for name in ("skirt", "box"):
        rows = _rows(np.asarray(mesh.vertices), np.atleast_2d(named[name]))
        assert np.array_equal(after[rows], raw[rows]), name
    # Bounded: a smoothing that would move any position further keeps the raw collider and says why.
    tight = _condition(tmp_path / "tight", base, [], smoothing={"max_displacement_m": 1e-9})
    assert tight["status"] == "kept_raw" and tight["record"]["reasons"] == ["smoothing_bound_exceeded"]
    assert conditioning.DEFAULT_SMOOTHING is None


def test_a_surface_that_cannot_be_flattened_safely_keeps_the_raw_collider(tmp_path, monkeypatch):
    mesh, _ = _table()
    base = _base_scene(tmp_path, mesh)
    footprint = _footprint((-0.3, -0.05))
    support = support_under(mesh, *footprint, up=2, meters_per_unit=1.0)

    def broken(*_args, **_kwargs):
        raise MemoryError("collider too large")

    monkeypatch.setattr(conditioning, "flatten_placement_surfaces", broken)
    result = _condition(tmp_path, base, [{"role": "start_support", "support": support, "footprint": footprint}])
    assert result["status"] == "kept_raw"
    assert result["record"]["reasons"] == ["conditioning_failed:MemoryError:collider too large"]
    assert "base_scene" not in result and not (tmp_path / "out" / "collider").exists()


def _bumpy_arguments(tmp_path, monkeypatch):
    """The compile fixture with its table top made bumpy like an Atlas collider."""
    from tests.test_website_task_preparation import _arguments

    args = _arguments(tmp_path)
    path = Path(args["base_scene"]["collision_mesh_path"])
    mesh = trimesh.load(path, force="mesh", process=False)
    for _ in range(2):
        level = np.flatnonzero(np.all(np.abs(mesh.vertices[mesh.faces][:, :, 1] - 1.75) < 1e-6, axis=1))
        mesh = mesh.subdivide(face_index=level)
    vertices = mesh.vertices.copy()
    table = np.abs(vertices[:, 1] - 1.75) < 1e-6
    vertices[table, 1] += np.random.default_rng(5).uniform(-0.002, 0.002, int(table.sum()))
    trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False).export(path)
    args["base_scene"]["collision_mesh_digest"] = _sha256_file(path)
    real = preparation.register_source_to_runtime
    monkeypatch.setattr(preparation, "register_source_to_runtime",
                        lambda **kwargs: real(**{**kwargs, "sample_cap": 2400}))
    return args


def test_the_conditioned_collider_is_sealed_before_its_digest_is_bound(tmp_path, monkeypatch):
    args = _bumpy_arguments(tmp_path, monkeypatch)
    raw = dict(args["base_scene"])
    raw_vertices = trimesh.load(raw["collision_mesh_path"], force="mesh", process=False).vertices
    events, real_condition, real_support = [], conditioning.condition_placement_collider, preparation.support_under

    def condition(**kwargs):
        events.append("condition")
        return real_condition(**kwargs)

    def lookup(mesh, *a, **k):
        events.append("raw" if np.array_equal(mesh.vertices, raw_vertices) else "sealed")
        return real_support(mesh, *a, **k)

    monkeypatch.setattr(conditioning, "condition_placement_collider", condition)
    monkeypatch.setattr(preparation, "support_under", lookup)
    value = preparation.compile_website_scene_preparation(**args)
    assert value["status"] == "intake_ready", value["blockers"]
    record = value["collider_conditioning"]
    assert record["status"] == "conditioned"
    # The supports are found on the raw collider, flattened, and every later lookup reads the sealed one.
    assert events[:events.index("condition")] and set(events[:events.index("condition")]) == {"raw"}
    assert events[events.index("condition") + 1:] and set(events[events.index("condition") + 1:]) == {"sealed"}
    sealed = json.loads((args["output_root"] / "base_scene.json").read_text())
    digest = _sha256_file(Path(sealed["collision_mesh_path"]))
    assert value["binding"]["collision_mesh_digest"] == sealed["collision_mesh_digest"] == digest
    assert record["collision_mesh_digest"] == digest and record["source_collision_mesh_digest"] == raw["collision_mesh_digest"]
    assert sealed["collider_conditioning"]["source_collision_mesh_digest"] == raw["collision_mesh_digest"]
    assert sealed["collision_binding_id"] == "website-collider-" + digest[7:39]
    # The registration was fitted against the raw collider, which is retained unchanged; the splat never changes.
    assert value["registration"]["collision_mesh_digest"] == raw["collision_mesh_digest"]
    assert _sha256_file(Path(raw["collision_mesh_path"])) == raw["collision_mesh_digest"]
    assert (sealed["splat_path"], sealed["splat_digest"]) == (raw["splat_path"], raw["splat_digest"])
    assert value["binding"]["splat_digest"] == raw["splat_digest"]
    receipt = json.loads(Path(record["receipt_path"]).read_text())
    assert receipt["digest"] == record["receipt_digest"] and receipt["output"]["sha256"] == digest
    flattened = [row for row in record["flattening"]["surfaces"] if row["status"] == "flattened"]
    assert flattened and all(row["face_indices"] and row["max_displacement_m"] > 0 for row in flattened)
    # The recorded support is what support_under finds on the sealed collider.
    mpu = value["coordinate_frame"]["declared_meters_per_unit"]
    to_sim = np.asarray(value["coordinate_frame"]["runtime_to_simulator"])
    lower, upper = preparation._runtime_bounds({"minimum": value["subject"]["aabb_min_xyz"],
                                               "maximum": value["subject"]["aabb_max_xyz"]}, np.linalg.inv(to_sim))
    snap = value["support"]["subject_snap_m"] / mpu
    lower[1] -= snap
    upper[1] -= snap
    collider = trimesh.load(sealed["collision_mesh_path"], force="mesh", process=False)
    again = real_support(collider, lower, upper, up=1, meters_per_unit=mpu, up_sign=1,
                         placement_residual_m=value["coordinate_frame"]["placement_uncertainty_m"])
    assert again["face_indices"] == value["support"]["face_indices"]
    assert again["top_runtime_units"] == pytest.approx(value["support"]["top_runtime_units"], abs=1e-9)
    # A later reader rereads the sealed collider by digest; the raw base scene no longer binds.
    (args["output_root"] / "task_masks.geometry.json").write_text(json.dumps(args["task_masks"]))
    preparation.relocated_destination(value, args["output_root"] / "preparation.json")
    (args["output_root"] / "base_scene.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="website_destination_relocation_binding_changed"):
        preparation.relocated_destination(value, args["output_root"] / "preparation.json")


def test_a_level_scene_compiles_with_its_collider_unchanged(tmp_path):
    from tests.test_website_task_preparation import _arguments

    args = _arguments(tmp_path)
    value = preparation.compile_website_scene_preparation(**args)
    assert value["status"] == "intake_ready", value["blockers"]
    assert value["support"]["basis"] == "reconstructed_triangle_footprint_contacts"
    assert "collider_conditioning" not in value
    assert value["binding"]["collision_mesh_digest"] == args["base_scene"]["collision_mesh_digest"]
    assert not (args["output_root"] / "base_scene.json").exists()
    assert not (args["output_root"] / "collider").exists()
