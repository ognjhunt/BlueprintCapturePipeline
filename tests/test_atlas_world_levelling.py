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
    assert np.allclose(splat.xyz * MPU, vertices, atol=1e-4)
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
