from __future__ import annotations

from blueprint_pipeline.isaac_live_geometry_validation import build_live_geometry_results


def test_live_geometry_requires_reach_facing_and_collision_clearance() -> None:
    clear = build_live_geometry_results(
        robot_xyz=(0.0, 0.0, 0.8),
        robot_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        target_xyz=(1.0, 0.0, 0.9),
        overlapping_prim_paths=("/World/G1/torso", "/World/GroundPlane"),
        robot_prim_path="/World/G1",
        max_reach_distance_m=1.5,
    )
    assert clear["stance"]["stance_valid"] is True
    assert clear["collision"]["collision_free"] is True

    blocked = build_live_geometry_results(
        robot_xyz=(0.0, 0.0, 0.8),
        robot_quaternion_xyzw=(0.0, 0.0, 1.0, 0.0),
        target_xyz=(2.0, 0.0, 0.9),
        overlapping_prim_paths=("/World/Kitchen/Counter",),
        robot_prim_path="/World/G1",
        max_reach_distance_m=1.5,
    )
    assert blocked["stance"]["reach_valid"] is False
    assert blocked["stance"]["facing_valid"] is False
    assert blocked["collision"]["collision_free"] is False
