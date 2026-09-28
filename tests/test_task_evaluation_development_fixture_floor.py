from __future__ import annotations

import hashlib

from pxr import Usd

from blueprint_pipeline.task_evaluation_development_fixture_floor import (
    materialize_development_fixture_episode_floor,
    materialize_development_fixture_placement_floor,
)
from blueprint_pipeline.task_evaluation_robot_placement_geometry import (
    build_robot_placement_geometry_index,
)
from tests.test_task_evaluation_robot_placement_geometry import _assets


def test_floor_is_additive_and_bound_to_qualified_asset(tmp_path) -> None:
    scene, asset = _assets(tmp_path)
    original = scene.read_bytes()
    digest = "sha256:" + hashlib.sha256(asset.read_bytes()).hexdigest()
    derived = materialize_development_fixture_placement_floor(
        source_collision=scene, qualified_asset=asset,
        qualified_asset_digest=digest, target_world_m=[0.8, 0.0, 0.5],
        output_root=tmp_path,
    )
    assert scene.read_bytes() == original
    assert materialize_development_fixture_placement_floor(
        source_collision=scene, qualified_asset=asset,
        qualified_asset_digest=digest, target_world_m=[0.8, 0.0, 0.5],
        output_root=tmp_path,
    ) == derived
    assert Usd.Stage.Open(str(derived)).GetPrimAtPath(
        "/Scene/development_fixture_floor"
    ).IsValid()
    index = build_robot_placement_geometry_index(
        scene_collision_usd_path=derived, robot_asset_usd_path=asset,
    )
    assert {row.prim_path for row in index.support_surfaces} >= {
        "/Scene/Floor", "/Scene/development_fixture_floor"
    }


def test_episode_floor_preserves_sources_and_extends_visual_and_physical_bounds(tmp_path) -> None:
    scene, asset = _assets(tmp_path)
    original = scene.read_bytes()
    appearance_source = tmp_path / "appearance.usda"
    appearance_source.write_bytes(original)
    digest = "sha256:" + hashlib.sha256(asset.read_bytes()).hexdigest()
    collision, appearance = materialize_development_fixture_episode_floor(
        source_collision=scene,
        source_appearance=appearance_source,
        qualified_asset=asset,
        qualified_asset_digest=digest,
        target_world_m=[0.8, 0.0, 0.0],
        output_root=tmp_path / "episode",
    )
    assert scene.read_bytes() == original
    for path in (collision, appearance):
        stage = Usd.Stage.Open(str(path))
        assert stage.GetPrimAtPath("/Scene/development_fixture_floor").IsValid()
        assert stage.GetRootLayer().customLayerData["blueprintCapturedRoomQualified"] is False
