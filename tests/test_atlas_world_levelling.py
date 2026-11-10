"""Levelling an Atlas world on its observed floor, against rooms with known truth.

The dishwasher capture of 2026-10-23 came back from Atlas with its floor 18
degrees off +Y. These rooms are tilted the same way; levelling must bring the
floor to y = 0 and leave counters, splat discs and cameras where gravity puts
them, or refuse with a reason.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from blueprint_pipeline import atlas_world_levelling as levelling
from blueprint_pipeline.gaussian_splat_decode import read_standard_3dgs_ply
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from tests.atlas_level_room_fixture import provider_units, tilt, write_room

MPU, OFFSET = 0.9, (0.3, -1.3, 0.2)
STANDING = [(0.0, 1.2, 0.0), (0.5, 1.35, 0.3), (-0.5, 1.5, -0.2)]


def level(tmp_path: Path, *, degrees=18.0, cameras=STANDING, **room):
    rotation = tilt(degrees)
    splat, mesh = tmp_path / "world/atlas_depth_splat.ply", tmp_path / "world/atlas_mesh_supported.glb"
    truth = write_room(splat, mesh, rotation=rotation, offset_m=OFFSET, meters_per_unit=MPU, **room)
    value = levelling.level_atlas_world(
        splat_path=splat, splat_digest=_sha256_file(splat), mesh_path=mesh, mesh_digest=_sha256_file(mesh),
        meters_per_unit=MPU, camera_centres=provider_units(cameras, rotation=rotation, offset_m=OFFSET,
                                                           meters_per_unit=MPU),
        output_root=tmp_path / "level")
    return value, truth


def outputs(value):
    return {row["role"]: Path(row["path"]) for row in value["outputs"]}


def test_a_tilted_world_is_levelled_on_its_floor_at_zero(tmp_path):
    value, truth = level(tmp_path)
    assert value["status"] == "levelled" and value["up_basis"] == "observed_floor"
    assert value["tilt_degrees"] == pytest.approx(18.0, abs=0.3)
    assert value["ground_plane_offset_m"] == 0.0 and value["physical_measurement"] is False
    assert value["floor"]["camera_heights_m"] == pytest.approx([1.2, 1.35, 1.5], abs=0.02)
    vertices = trimesh.load(str(outputs(value)["supported_mesh_glb"]), force="mesh", process=False).vertices * MPU
    (floor_start, floor_end), (counter_start, counter_end) = truth["parts"]["floor"], truth["parts"]["counter"]
    assert np.abs(vertices[floor_start:floor_end, 1]).max() < 0.015
    assert vertices[counter_start:counter_end, 1] == pytest.approx(0.9, abs=0.015)
    # The splat moves with the mesh, and each disc's thin axis now stands vertical.
    splat = read_standard_3dgs_ply(outputs(value)["splat_ply"])
    # (The floor filled under the filmed path is appended after the captured vertices.)
    assert np.allclose(splat.xyz * MPU, vertices[:len(splat.xyz)], atol=1e-4)
    thin = Rotation.from_quat(splat.quats[:, [1, 2, 3, 0]]).apply([0.0, 0.0, 1.0])
    assert np.abs(thin[:, 1]).min() > np.cos(np.radians(0.5))
    assert np.allclose(np.linalg.norm(splat.quats, axis=1), 1.0, atol=1e-5)


def test_levelling_turns_nothing_about_the_vertical(tmp_path):
    value, truth = level(tmp_path, noise_m=0.0)
    lift = np.asarray(value["provider_to_levelled"])
    # Tilted about a horizontal axis, the world comes back unturned: only the offset remains.
    assert np.allclose(lift[:3, :3] @ tilt(18.0), np.eye(3), atol=1e-6)


def test_a_retry_reuses_the_receipt_and_a_changed_copy_is_refused(tmp_path):
    first, _ = level(tmp_path)
    second, _ = level(tmp_path)
    assert second == first
    outputs(first)["splat_ply"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="atlas_world_levelling_cache_changed"):
        level(tmp_path)


def test_a_world_with_no_level_surface_is_refused(tmp_path):
    value, _ = level(tmp_path, floor=False, counter=False)
    assert value["status"] == "needs_input" and value["reason"] == "website_atlas_scene_level_unobserved"
    assert value["outputs"] == []
    # The refusal says what the search found, so it can be read without the mesh.
    search = value["level_search"]
    assert search["largest_level_plane_share"] < search["minimum_level_plane_share"] == levelling.MIN_LEVEL_SHARE
    assert search["parallel_level_share"] == 0.0


def test_a_refusal_keeps_the_height_profiles_around_the_cameras(tmp_path):
    value, _ = level(tmp_path, floor=False, counter=False)
    profiles = value["level_search"]["height_profiles"]
    assert value["status"] == "needs_input" and profiles[0]["source"] == "plus_y"
    for profile in profiles:
        assert profile["band_m"] == levelling.PROFILE_BAND_M
        assert len(profile["shares"]) == round((profile["to_m"] - profile["from_m"]) / profile["band_m"])
        assert sum(profile["shares"]) + profile["below_share"] + profile["above_share"] == pytest.approx(1.0, abs=1e-3)


def test_a_candidate_refitted_past_the_tilt_bound_never_levels(tmp_path):
    # Parallel planes whose normal is 47 degrees from +Y; the only candidate starts at 44 degrees.
    rng = np.random.default_rng(0)
    steep, start = (np.array([np.sin(np.radians(a)), np.cos(np.radians(a)), 0.0]) for a in (47.0, 44.0))
    across = np.cross(steep, [0.0, 0.0, 1.0])
    points = np.concatenate([offset * steep + rng.uniform(-3, 3, (4000, 1)) * across
                             + rng.uniform(-3, 3, (4000, 1)) * np.array([0.0, 0.0, 1.0])
                             for offset in (0.0, 0.4, 0.8, 1.2)])
    search = {}
    found = levelling._parallel_level_direction(np.zeros((0, 3)), np.zeros((0, 3), np.int64), points, search,
                                                directions=(start,))
    assert found is None
    assert search["parallel_candidates"][0]["tilt_degrees"] > levelling.MAX_TILT_DEGREES


def test_rough_open_ground_is_still_refused(tmp_path):
    value, _ = level(tmp_path, terrain=True, cameras=[(0.0, 1.8, 0.0), (1.0, 1.9, 0.5), (-1.0, 1.7, -0.5)])
    assert value["status"] == "needs_input" and value["reason"] == "website_atlas_scene_level_unobserved"
    search = value["level_search"]
    assert search["largest_level_plane_share"] < levelling.MIN_LEVEL_SHARE
    assert search["parallel_level_share"] < levelling.MIN_LEVEL_SHARE / 2


AISLE = [(2.5, 1.45, 0.0), (0.0, 1.5, 0.05), (-2.0, 1.4, -0.05)]


def test_an_aisle_floor_is_found_among_its_shelf_boards(tmp_path):
    # The floor strip holds 3% of the surface: no single plane passes, but the
    # floor and the shelf boards on both sides are parallel.
    value, _ = level(tmp_path, aisle=True, hall_rows=3, cameras=AISLE)
    assert value["level_search"]["largest_level_plane_share"] < levelling.MIN_LEVEL_SHARE
    assert value["status"] == "levelled" and value["level_direction_basis"] == "parallel_level_surfaces"
    assert value["tilt_degrees"] == pytest.approx(18.0, abs=0.3)
    assert value["up_basis"] == "observed_floor" and value["ground_plane_offset_m"] == 0.0
    assert value["floor"]["camera_heights_m"] == pytest.approx([1.45, 1.5, 1.4], abs=0.02)
    # The floor and the board heights are the layers: together far above the single-plane share.
    search = value["level_search"]
    assert search["parallel_level_layers"] >= 5 and search["parallel_level_share"] > 2 * levelling.MIN_LEVEL_SHARE
    # Both candidate directions were tried; the kept one holds the most, and its layers are reported
    # by height from the cameras: the floor 1.45 m below them.
    sources = {row["source"]: row for row in search["parallel_candidates"]}
    assert set(sources) == {"face_normal_mode", "largest_level_plane"}
    assert search["parallel_level_share"] == max(row["share"] for row in sources.values())
    assert any(abs(row["height_from_cameras_m"] + 1.45) <= 0.03 for row in search["densest_layers"])
    # Along every candidate, the profile shows the floor as the lowest band with anything in it.
    profiles = {row["source"]: row for row in search["height_profiles"]}
    assert set(profiles) == {"plus_y", "face_normal_mode", "largest_level_plane"}
    profile = profiles["face_normal_mode"]
    floor_band = int(round((-1.45 - profile["from_m"]) / profile["band_m"]))
    peak = max(range(floor_band - 1, floor_band + 2), key=lambda band: profile["shares"][band])
    assert profile["shares"][peak] >= 0.02 and not any(profile["shares"][:floor_band - 2])


def test_an_aisle_with_no_floor_left_is_levelled_but_declares_no_floor(tmp_path):
    value, _ = level(tmp_path, aisle=True, floor_width=0.0, cameras=AISLE)
    assert value["status"] == "levelled" and value["level_direction_basis"] == "parallel_level_surfaces"
    assert value["tilt_degrees"] == pytest.approx(18.0, abs=0.3)
    assert value["up_basis"] == "dominant_level_surface" and value["ground_plane_offset_m"] is None


def test_a_floor_a_few_degrees_off_the_start_direction_is_found_as_its_own_plane():
    from tests.atlas_level_room_fixture import aisle_parts
    points = np.concatenate([part for part, _ in aisle_parts(floor_tilt_degrees=2.5).values()])
    points = points + np.random.default_rng(0).normal(0.0, 0.003, points.shape)
    search = {}
    # Along +Y the floor spreads over 26 cm; the band narrows onto its own plane.
    floor = levelling._band_floor(points, np.array([0.0, 1.0, 0.0]), search)
    assert floor is not None and floor["inlier_share"] >= levelling.MIN_LEVEL_SHARE
    assert levelling._tilt(floor["normal"]) == pytest.approx(2.5, abs=0.15)
    assert floor["rms_m"] < 0.01 and search["floor_band"]["tilt_degrees"] == pytest.approx(2.5, abs=0.15)


def test_a_floor_off_the_shelves_levels_the_world_when_no_single_plane_is_found(tmp_path, monkeypatch):
    # The three-point search can miss a floor holding a few percent; the floor band still finds it.
    monkeypatch.setattr(levelling, "_level_direction", lambda *args, **kwargs: None)
    value, _ = level(tmp_path, degrees=13.0, aisle=True, floor_tilt_degrees=2.0, cameras=AISLE)
    assert value["status"] == "levelled" and value["up_basis"] == "observed_floor"
    assert value["ground_plane_offset_m"] == 0.0 and value["level_search"]["floor_band"]["inlier_share"] >= 0.04
    # Levelled on the floor's own plane: the floor is at y = 0 and the cameras stand on it.
    heights = value["floor"]["camera_heights_m"]
    assert all(1.3 < height < 1.6 for height in heights)


def _no_plane_found(monkeypatch):
    # As on the library aisle: neither the single-plane nor the parallel-layer search finds anything.
    monkeypatch.setattr(levelling, "_level_direction", lambda *args, **kwargs: None)
    monkeypatch.setattr(levelling, "_parallel_level_direction", lambda *args, **kwargs: None)


def test_an_uneven_floor_agreeing_with_the_surfaces_levels_the_world_but_is_not_declared(tmp_path, monkeypatch):
    _no_plane_found(monkeypatch)
    value, _ = level(tmp_path, degrees=13.0, aisle=True, floor_wave_m=0.08, cameras=AISLE)
    search = value["level_search"]
    assert search["floor_band"]["inlier_share"] < levelling.MIN_LEVEL_SHARE  # too uneven to declare
    assert search["direction_agreement_degrees"] <= levelling.DIRECTION_AGREEMENT_DEGREES
    assert value["status"] == "levelled" and value["level_direction_basis"] == "agreeing_normal_mode_and_floor_band"
    assert value["tilt_degrees"] == pytest.approx(13.0, abs=0.3)
    assert value["up_basis"] == "dominant_level_surface" and value["ground_plane_offset_m"] is None
    assert value["floor"] is None


def test_an_uneven_floor_that_disagrees_with_the_surfaces_is_refused(tmp_path, monkeypatch):
    _no_plane_found(monkeypatch)
    value, _ = level(tmp_path, degrees=13.0, aisle=True, floor_wave_m=0.08, floor_tilt_degrees=3.0, cameras=AISLE)
    assert value["status"] == "needs_input" and value["reason"] == "website_atlas_scene_level_unobserved"
    assert value["level_search"]["direction_agreement_degrees"] > levelling.DIRECTION_AGREEMENT_DEGREES
    # The provider's +Y is 13 degrees off as well, so the fallback floor does not stand in either.
    estimates = value["level_search"]["fallback_estimates"]
    assert max(estimates["floor_band_tilt_degrees"], estimates["face_normal_mode_tilt_degrees"]) \
        > levelling.FALLBACK_MAX_TILT_DEGREES


def test_the_flagged_fallback_floor_stands_in_for_an_uneven_one_near_the_providers_up(tmp_path, monkeypatch):
    _no_plane_found(monkeypatch)
    value, _ = level(tmp_path, degrees=0.0, aisle=True, floor_wave_m=0.08, floor_tilt_degrees=2.0, cameras=AISLE)
    assert value["level_search"]["direction_agreement_degrees"] > levelling.DIRECTION_AGREEMENT_DEGREES
    assert value["status"] == "levelled" and value["up_basis"] == "fallback_fitted_floor"
    assert value["level_direction_basis"] == "provider_up_fallback" and value["tilt_degrees"] == 0.0
    assert value["ground_plane_offset_m"] == 0.0 and value["physical_measurement"] is False
    fallback = value["fallback_floor"]
    assert fallback["claim"] == "development_only_estimate" and fallback["replaced_faces"] > 0
    assert fallback["band_spread_m"][0] < -0.05 < 0.05 < fallback["band_spread_m"][1]  # the floor was uneven
    assert all(1.3 < height < 1.6 for height in value["floor"]["camera_heights_m"])
    # The collider's floor is flat at y = 0; nothing level is left within the replaced band except it.
    mesh = trimesh.load(str(outputs(value)["supported_mesh_glb"]), force="mesh", process=False)
    tri = np.asarray(mesh.vertices)[np.asarray(mesh.faces)] * MPU
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    level_faces = np.abs(cross[:, 1]) >= np.cos(np.radians(30.0)) * np.linalg.norm(cross, axis=1)
    near = np.all(np.abs(tri[:, :, 1]) <= levelling.FALLBACK_REPLACE_HALF_M, axis=1)
    assert (level_faces & near).sum() == fallback["flat_floor_faces"] + value["walked_floor"]["faces_added"]
    assert np.abs(tri[level_faces & near][:, :, 1]).max() < 1e-6
    # The splat is only moved, never turned.
    assert np.allclose(np.asarray(value["provider_to_levelled"])[:3, :3], np.eye(3))


def test_the_fallback_floor_covers_the_band_when_no_face_passes_the_replacement_test():
    # Only vertical faces reach the floor band: nothing is replaced, and the flat floor is still built.
    from tests.atlas_level_room_fixture import _grid
    vertices, faces = _grid((-2.0, 0.0, -1.0), (0.05, 0.0, 0.0), (0.0, 0.05, 0.0), 81, 41)
    swapped = levelling._replace_uneven_floor(trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
                                              meters_per_unit=1.0)
    assert swapped["replaced_faces"] == 0 and swapped["flat_floor_faces"] > 0
    assert swapped["flat_floor_extent_m"][0] >= 4.0


def test_a_single_dominant_plane_keeps_its_own_basis(tmp_path):
    value, _ = level(tmp_path)
    assert value["level_direction_basis"] == "single_level_plane"


def test_a_warehouse_floor_seen_from_a_mezzanine_is_declared(tmp_path):
    # Not only kitchens: rack levels are parallel level layers, and the cameras stand 5 m up.
    mezzanine = [(0.0, 4.6, 0.0), (0.8, 5.0, -0.4), (-0.6, 5.2, 0.3)]
    value, _ = level(tmp_path, counter=False, shelves=True, cameras=mezzanine)
    assert value["status"] == "levelled" and value["up_basis"] == "observed_floor"
    assert value["tilt_degrees"] == pytest.approx(18.0, abs=0.3)
    assert value["floor"]["camera_heights_m"] == pytest.approx([4.6, 5.0, 5.2], abs=0.03)


def test_a_table_seen_alone_levels_the_world_but_is_not_declared_a_floor(tmp_path):
    close_up = [(0.0, 1.1, 0.0), (0.2, 1.15, 0.1), (-0.2, 1.05, 0.0)]
    value, _ = level(tmp_path, floor=False, counter=False, walls=False, table=True, cameras=close_up)
    assert value["status"] == "levelled" and value["up_basis"] == "dominant_level_surface"
    assert value["tilt_degrees"] == pytest.approx(18.0, abs=0.5)
    assert value["ground_plane_offset_m"] is None and value["floor"] is None


def test_a_floor_the_cameras_stand_below_is_not_declared(tmp_path):
    value, _ = level(tmp_path, cameras=[(0.0, -0.4, 0.0), (0.4, 1.3, 0.0), (0.2, 1.2, 0.3)])
    assert value["status"] == "levelled" and value["ground_plane_offset_m"] is None
    assert value["up_basis"] == "dominant_level_surface"


def test_a_view_dependent_splat_is_refused_rather_than_tilted(tmp_path):
    value, _ = level(tmp_path, sh_rest=True)
    assert value["status"] == "needs_input"
    assert value["reason"] == "website_atlas_scene_level_splat_view_dependent"


def _faces_over(mesh_path: Path, xz) -> int:
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    tri = np.asarray(mesh.vertices)[np.asarray(mesh.faces)] * MPU
    low, high = tri[:, :, [0, 2]].min(axis=1), tri[:, :, [0, 2]].max(axis=1)
    level_at_floor = np.all(np.abs(tri[:, :, 1]) < 1e-6, axis=1)
    return int((level_at_floor & np.all(low <= xz, axis=1) & np.all(xz <= high, axis=1)).sum())


def test_a_floor_is_filled_under_the_filmed_path_and_disclosed(tmp_path):
    # capture-5b067498 (2026-11-08): the scan kept no floor where the walk began, and the G1 fell through.
    value, _ = level(tmp_path)
    walked = value["walked_floor"]
    assert walked["claim"] == "development_only_estimate" and walked["basis"] == "floor_under_the_filmed_path"
    assert walked["faces_added"] == 2 * walked["cells"] > 0 and walked["cameras"] == len(STANDING)
    levelled = (np.asarray(provider_units(STANDING, rotation=tilt(18.0), offset_m=OFFSET, meters_per_unit=MPU))
                @ np.asarray(value["provider_to_levelled"])[:3, :3].T
                + np.asarray(value["provider_to_levelled"])[:3, 3]) * MPU
    for camera in levelled:
        assert _faces_over(outputs(value)["supported_mesh_glb"], camera[[0, 2]]) >= 2


def test_the_walked_floor_covers_each_footprint_within_its_margin_and_nothing_far_from_it():
    from tests.atlas_level_room_fixture import _grid
    # A wall and no floor at all: only the filled cells are level at y = 0.
    vertices, faces = _grid((-2.0, 0.0, 3.0), (0.05, 0.0, 0.0), (0.0, 0.05, 0.0), 81, 41)
    footprints = np.array([[0.0, 0.0], [0.3, 0.05], [0.6, 0.1]])
    filled = levelling._fill_walked_floor(trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
                                          footprints_m=footprints, meters_per_unit=1.0)
    mesh = filled.pop("mesh")
    tri = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    floor = tri[np.all(np.abs(tri[:, :, 1]) < 1e-9, axis=1)]
    assert len(floor) == filled["faces_added"] == 2 * filled["cells"]
    low, high = floor[:, :, [0, 2]].min(axis=1), floor[:, :, [0, 2]].max(axis=1)
    def covered(point):
        return bool(np.any(np.all(low <= point, axis=1) & np.all(point <= high, axis=1)))
    # Each footprint and a G1 start offset 15 cm from it stand on the filled floor; 2 m away nothing does.
    assert all(covered(point + offset) for point in footprints
               for offset in ([0, 0], [0.15, 0], [-0.15, 0], [0, 0.15], [0, -0.15]))
    assert not covered(np.array([0.0, 2.0])) and not covered(np.array([3.0, 0.0]))
    # The captured faces are kept as they were.
    assert np.array_equal(np.asarray(mesh.faces)[:len(faces)], faces)
    assert levelling._fill_walked_floor(mesh, footprints_m=np.zeros((0, 2)), meters_per_unit=1.0) is None


def _warped_aisle(mpu=1.0):
    """A floor rising along +x (0 at x = 0, 9 cm at x = 3, slightly curved, as a forward-only dolly warps it),
    a shelf's bottom board 15 cm above it inside the corridor, and the same floor beyond the corridor."""
    from tests.atlas_level_room_fixture import _grid
    vertices, faces = _grid((-1.0, 0.0, -2.0), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05), 101, 81)
    vertices[:, 1] = 0.03 * vertices[:, 0] + 0.002 * vertices[:, 0] ** 2
    board, board_faces = _grid((0.5, 0.0, 0.35), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05), 41, 6)
    board[:, 1] = 0.03 * board[:, 0] + 0.002 * board[:, 0] ** 2 + 0.15
    mesh = trimesh.Trimesh(vertices=np.vstack([vertices, board]) / mpu,
                           faces=np.vstack([faces, board_faces + len(vertices)]), process=False)
    return mesh, len(vertices), len(board_faces)


def test_the_warped_scanned_floor_in_the_walked_corridor_is_planarized_and_disclosed():
    # capture-5b067498 (2026-11-09): the rebuilt splat kept a warped floor, 6.5-9.4 cm above the levelled plane at
    # the start, so a G1 stood inside it. Owner: "flatten the corridor floor".
    mesh, floor_vertices, board_faces = _warped_aisle()
    footprints = np.column_stack([np.linspace(0.0, 3.0, 13), np.zeros(13)])
    planarized = levelling._planarize_walked_floor(mesh, footprints_m=footprints, meters_per_unit=1.0)
    rebuilt = planarized.pop("mesh")
    assert planarized["claim"] == "development_only_estimate"
    assert planarized["basis"] == "scanned_floor_in_the_walked_corridor_planarized"
    assert planarized["surface_curved_along"] == ["walk"] and planarized["replaced_faces"] > 0
    assert 0.09 <= planarized["surface_offset_m"]["max"] <= 0.16 and planarized["inlier_rms_m"] < 0.01
    planarized.pop("extra_cells")
    tri = np.asarray(rebuilt.vertices)[np.asarray(rebuilt.faces)]
    centroid = tri.mean(axis=1)
    near_path = (np.abs(centroid[:, 2]) <= 0.25) & (centroid[:, 0] >= 0.0) & (centroid[:, 0] <= 3.0)
    # No scanned floor face is left on the walk; the shelf board above it and the floor beyond the corridor stay.
    assert not np.any(near_path & (centroid[:, 1] < 0.13))
    assert int((np.asarray(rebuilt.faces).min(axis=1) >= floor_vertices).sum()) == board_faces
    assert np.any((np.abs(centroid[:, 2]) >= 1.2) & (centroid[:, 0] > 2.0) & (centroid[:, 1] > 0.06))
    # The walked floor then stands at y = 0 under every footprint and its 15 cm start offsets.
    filled = levelling._fill_walked_floor(rebuilt, footprints_m=footprints, meters_per_unit=1.0)
    floor = np.asarray(filled.pop("mesh").vertices)[np.asarray(rebuilt.faces).max() + 1:]
    assert np.all(np.abs(floor[:, 1]) < 1e-9)


def test_a_board_over_the_floor_is_never_taken_for_the_floor():
    # Codex on #3508: equally tessellated layers at 0 and 16 cm must not fit between them and lose the board.
    from tests.atlas_level_room_fixture import _grid
    floor, floor_faces = _grid((-1.0, 0.0, -1.0), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05), 81, 41)
    floor[:, 1] = 0.003 * np.sin(floor[:, 0] * 7.0)  # a scanned floor is never exactly flat
    board, board_faces = _grid((-1.0, 0.0, -1.0), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05), 81, 41)
    board[:, 1] = 0.16
    mesh = trimesh.Trimesh(vertices=np.vstack([floor, board]),
                           faces=np.vstack([floor_faces, board_faces + len(floor)]), process=False)
    footprints = np.column_stack([np.linspace(0.0, 2.0, 9), np.zeros(9)])
    record = levelling._planarize_walked_floor(mesh, footprints_m=footprints, meters_per_unit=1.0)
    rebuilt = record.pop("mesh")
    assert abs(record["surface_offset_m"]["median"]) < 0.01 and record["layer_m"] <= 0.05
    kept = np.asarray(rebuilt.faces)
    assert int((kept.min(axis=1) >= len(floor)).sum()) == len(board_faces)
    assert record["replaced_faces"] > 0 and int((kept.max(axis=1) < len(floor)).sum()) < len(floor_faces)


def test_a_floor_face_crossing_the_corridor_edge_leaves_no_hole_beyond_it():
    # Codex on #3508: a large floor triangle reaching into the corridor is dropped whole; its cells are filled.
    from tests.atlas_level_room_fixture import _grid
    fine, fine_faces = _grid((-0.5, 0.06, -0.5), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05), 41, 21)
    big = np.array([[0.2, 0.06, 0.3], [1.5, 0.06, 1.6], [-0.3, 0.06, 1.6]])
    mesh = trimesh.Trimesh(vertices=np.vstack([fine, big]),
                           faces=np.vstack([fine_faces, [[len(fine), len(fine) + 2, len(fine) + 1]]]), process=False)
    footprints = np.column_stack([np.linspace(0.0, 1.5, 7), np.zeros(7)])
    record = levelling._planarize_walked_floor(mesh, footprints_m=footprints, meters_per_unit=1.0)
    rebuilt = record.pop("mesh")
    assert not np.any(np.asarray(rebuilt.faces).max(axis=1) >= len(fine))  # the big triangle is dropped
    assert record["cells_beyond_corridor"] > 0
    filled = levelling._fill_walked_floor(rebuilt, footprints_m=footprints, meters_per_unit=1.0,
                                          extra_cells=record["extra_cells"])
    tri = np.asarray(filled["mesh"].vertices)[np.asarray(filled["mesh"].faces)]
    flat = tri[np.all(np.abs(tri[:, :, 1]) < 1e-9, axis=1)]
    low, high = flat[:, :, [0, 2]].min(axis=1), flat[:, :, [0, 2]].max(axis=1)
    # A point under the dropped triangle, a metre beyond the corridor, stands on the flat floor.
    point = np.array([0.4, 1.4])
    assert np.any(np.all(low <= point, axis=1) & np.all(point <= high, axis=1))


def test_a_floor_that_also_slopes_across_the_walk_is_followed_cell_by_cell():
    # capture-5b067498, levelling v9 (2026-11-09): the floor under the start stood 6.5-9.4 cm up and sloped
    # across the walk, outside the one fitted surface's layer, so it stayed and the G1 still overlapped it.
    from tests.atlas_level_room_fixture import _grid
    floor, floor_faces = _grid((-1.0, 0.0, -1.0), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05), 101, 41)
    x, z = floor[:, 0], floor[:, 2]
    # Level until x = 2, then climbing to 8 cm at x = 3 and tilting 10 cm/m across the walk there.
    rise = np.clip(x - 2.0, 0.0, 1.0)
    floor[:, 1] = 0.08 * rise + 0.10 * rise * z
    floor_faces = floor_faces[:, ::-1]  # wound downward, as the scan's faces under the start were
    board, board_faces = _grid((2.4, 0.0, 0.3), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05), 9, 5)
    board[:, 1] = 0.25
    mesh = trimesh.Trimesh(vertices=np.vstack([floor, board]),
                           faces=np.vstack([floor_faces, board_faces + len(floor)]), process=False)
    footprints = np.column_stack([np.linspace(0.0, 3.0, 13), np.zeros(13)])
    record = levelling._planarize_walked_floor(mesh, footprints_m=footprints, meters_per_unit=1.0)
    rebuilt = record.pop("mesh")
    tri = np.asarray(rebuilt.vertices)[np.asarray(rebuilt.faces)]
    centroid = tri.mean(axis=1)
    stance = (np.abs(centroid[:, 0] - 2.95) <= 0.3) & (np.abs(centroid[:, 2]) <= 0.3) & (centroid[:, 1] < 0.2)
    assert not np.any(stance)  # nothing of the scanned floor is left under a stance at the walk's end
    assert int((np.asarray(rebuilt.faces).min(axis=1) >= len(floor)).sum()) == len(board_faces)  # the board stays
    assert record["floor_cells"] > record["inlier_cells"] and record["surface_offset_m"]["max"] > 0.06
    end = [row for row in record["cells"] if abs(row["x_m"] - 2.875) < 0.2 and abs(row["z_m"]) < 0.3]
    assert end and all(row["floor"] and row["dropped_faces"] > 0 and row["left_faces"] == 0 for row in end)


def test_a_large_floor_face_on_a_slope_is_judged_corner_by_corner():
    # Codex on #3511: a floor triangle spanning cells on an 8 cm/m slope has corners more than the layer apart.
    from tests.atlas_level_room_fixture import _grid
    fine, fine_faces = _grid((-0.5, 0.0, -0.5), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05), 51, 21)
    fine[:, 1] = 0.08 * fine[:, 0]
    big = np.array([[0.0, 0.0, 0.1], [2.0, 0.16, 0.1], [1.0, 0.08, 0.4]])
    mesh = trimesh.Trimesh(vertices=np.vstack([fine, big]),
                           faces=np.vstack([fine_faces, [[len(fine), len(fine) + 2, len(fine) + 1]]]), process=False)
    footprints = np.column_stack([np.linspace(0.0, 2.0, 9), np.zeros(9)])
    record = levelling._planarize_walked_floor(mesh, footprints_m=footprints, meters_per_unit=1.0)
    assert not np.any(np.asarray(record["mesh"].faces).max(axis=1) >= len(fine))  # the large face went too


def test_a_second_floor_sheet_a_few_centimetres_up_goes_with_the_floor():
    # capture-5b067498 at v10 (2026-11-09): the lowest layer was replaced and a second near-level sheet 6-9 cm
    # above it stayed along the whole walk, so the G1 still stood inside it.
    from tests.atlas_level_room_fixture import _grid
    lower, lower_faces = _grid((-1.0, 0.0, -1.0), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05), 81, 41)
    lower[:, 1] = 0.02 + 0.003 * np.sin(lower[:, 0] * 7.0)
    upper, upper_faces = _grid((-1.0, 0.0, -1.0), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05), 81, 41)
    upper[:, 1] = 0.085 + 0.01 * np.sin(upper[:, 2] * 5.0)
    step, step_faces = _grid((0.5, 0.0, 0.2), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05), 9, 5)
    step[:, 1] = 0.18
    mesh = trimesh.Trimesh(vertices=np.vstack([lower, upper, step]), faces=np.vstack([
        lower_faces, upper_faces[:, ::-1] + len(lower), step_faces + len(lower) + len(upper)]), process=False)
    footprints = np.column_stack([np.linspace(0.0, 2.0, 9), np.zeros(9)])
    record = levelling._planarize_walked_floor(mesh, footprints_m=footprints, meters_per_unit=1.0)
    tri = np.asarray(record["mesh"].vertices)[np.asarray(record["mesh"].faces)]
    centroid = tri.mean(axis=1)
    on_walk = (centroid[:, 0] >= 0.0) & (centroid[:, 0] <= 2.0) & (np.abs(centroid[:, 2]) <= 0.25)
    assert not np.any(on_walk & (centroid[:, 1] < 0.15))  # both sheets are gone along the walk
    assert int((np.asarray(record["mesh"].faces).min(axis=1) >= len(lower) + len(upper)).sum()) == len(step_faces)
    assert record["clear_m"] == levelling.WALKED_FLOOR_CLEAR_M and record["left_near_level_faces"] >= len(step_faces)
    # The faces taken above the layer are counted per cell with their top, so a sheet reads apart from an object.
    high = [row for row in record["cells"] if "dropped_above_layer_faces" in row]
    assert record["dropped_above_layer_faces"] > 0 and len(high) == record["cells_with_faces_dropped_above_layer"] > 10
    assert all(0.07 <= row["dropped_above_layer_top_m"] <= 0.10 for row in high)


def test_the_low_pieces_a_floor_sheet_leaves_standing_go_and_a_wall_stays_whole():
    # capture-5b067498 at v11 (2026-11-09): the soles stood on the flat floor, and the knees still met steep
    # faces 4-10 cm up: the skirts of the second sheet, left standing once its top faces went.
    from tests.atlas_level_room_fixture import _grid
    floor, floor_faces = _grid((-1.0, 0.0, -1.0), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05), 81, 41)
    floor[:, 1] = 0.03
    # A fin across the walk: steep, from the scanned floor to 9 cm. Stored one triangle at a time, and joined
    # to the wall below only through the floor beside the corridor (shared positions).
    fin, fin_faces = _grid((1.0, 0.03, -0.2), (0.0, 0.02, 0.0), (0.0, 0.0, 0.05), 4, 9)
    fin = fin[fin_faces].reshape(-1, 3)
    fin_faces = np.arange(len(fin)).reshape(-1, 3)
    # A wall inside the corridor, from the scanned floor to 1 m: its lowest faces lie in the band and stay with it.
    wall, wall_faces = _grid((0.2, 0.03, 0.4), (0.05, 0.0, 0.0), (0.0, 0.05, 0.0), 21, 21)
    mesh = trimesh.Trimesh(vertices=np.vstack([floor, fin, wall]), faces=np.vstack([
        floor_faces, fin_faces + len(floor), wall_faces + len(floor) + len(fin)]), process=False)
    footprints = np.column_stack([np.linspace(0.0, 2.0, 9), np.zeros(9)])
    record = levelling._planarize_walked_floor(mesh, footprints_m=footprints, meters_per_unit=1.0)
    kept = np.asarray(record["mesh"].faces)
    assert not np.any((kept.min(axis=1) >= len(floor)) & (kept.max(axis=1) < len(floor) + len(fin)))  # the fin went
    assert int((kept.min(axis=1) >= len(floor) + len(fin)).sum()) == len(wall_faces)  # every wall face stays
    assert record["dropped_low_piece_faces"] == len(fin_faces)
    pieces = [row for row in record["cells"] if "dropped_low_piece_faces" in row]
    assert pieces and all(0.08 <= row["dropped_low_piece_top_m"] <= 0.10 for row in pieces)


def test_a_wall_with_a_fine_base_and_tall_upper_faces_keeps_its_base():
    # Codex on #3521: the base strip's tall neighbours must count, wherever their centroids lie.
    from tests.atlas_level_room_fixture import _grid
    floor, floor_faces = _grid((-1.0, 0.0, -1.0), (0.05, 0.0, 0.0), (0.0, 0.0, 0.05), 81, 41)
    floor[:, 1] = 0.03
    base, base_faces = _grid((0.2, 0.03, 0.4), (0.05, 0.0, 0.0), (0.0, 0.05, 0.0), 21, 3)  # 3 to 13 cm
    upper, upper_faces = _grid((0.2, 0.13, 0.4), (0.05, 0.0, 0.0), (0.0, 2.87, 0.0), 21, 2)  # 13 cm to 3 m
    mesh = trimesh.Trimesh(vertices=np.vstack([floor, base, upper]), faces=np.vstack([
        floor_faces, base_faces + len(floor), upper_faces + len(floor) + len(base)]), process=False)
    footprints = np.column_stack([np.linspace(0.0, 2.0, 9), np.zeros(9)])
    record = levelling._planarize_walked_floor(mesh, footprints_m=footprints, meters_per_unit=1.0)
    kept = np.asarray(record["mesh"].faces)
    assert int((kept.min(axis=1) >= len(floor)).sum()) == len(base_faces) + len(upper_faces)
    assert record["dropped_low_piece_faces"] == 0


def test_planarizing_keeps_a_flat_floor_and_needs_a_floor_to_fit():
    from tests.atlas_level_room_fixture import _grid
    vertices, faces = _grid((-1.0, 0.0, -1.0), (0.1, 0.0, 0.0), (0.0, 0.0, 0.1), 21, 21)
    flat = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    footprints = np.array([[0.0, 0.0], [0.5, 0.0]])
    record = levelling._planarize_walked_floor(flat, footprints_m=footprints, meters_per_unit=1.0)
    # A floor already flat at y = 0 is the walked floor already: nothing is replaced.
    assert record["replaced_faces"] == 0 and len(record.pop("mesh").faces) == len(faces)
    wall, wall_faces = _grid((-2.0, 0.0, 3.0), (0.05, 0.0, 0.0), (0.0, 0.05, 0.0), 81, 41)
    assert levelling._planarize_walked_floor(trimesh.Trimesh(vertices=wall, faces=wall_faces, process=False),
                                             footprints_m=footprints, meters_per_unit=1.0) is None
    assert levelling._planarize_walked_floor(flat, footprints_m=np.zeros((0, 2)), meters_per_unit=1.0) is None


def test_a_levelled_world_with_a_warped_floor_records_its_planarized_corridor(tmp_path):
    value, _ = level(tmp_path)
    assert value["binding"]["revision"] == levelling.REVISION == "atlas_world_levelling.v13"
    planarized = value["walked_floor"].get("planarized")
    # The fixture's floor is flat: the corridor's floor is fitted, and nothing off the plane needed replacing.
    assert planarized is not None and planarized["surface_offset_m"]["max"] < 0.02
