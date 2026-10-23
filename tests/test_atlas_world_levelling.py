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
