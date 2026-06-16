#!/usr/bin/env python3
"""Render official Unitree G1 policy motion and handoff artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from blueprint_pipeline.official_g1_policy_handoff import build_official_g1_policy_handoff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--policy-manifest")
    parser.add_argument("--unitree-rl-gym-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--render-fps", type=int, default=24)
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument(
        "--camera-set",
        default="overview,side,follow,robot_pov",
        help="Comma-separated cameras: overview,side,follow,robot_pov,robot_pov_head,robot_pov_torso.",
    )
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--target-displacement-m", type=float)
    parser.add_argument("--fall-height-threshold-m", type=float, default=0.45)
    parser.add_argument("--command-x", type=float)
    parser.add_argument("--command-y", type=float)
    parser.add_argument("--command-yaw", type=float)
    parser.add_argument("--collision-proxy-limit", type=int, default=512)
    parser.add_argument("--base-path-clearance-m", type=float, default=0.38)
    parser.add_argument("--start-x", type=float)
    parser.add_argument("--start-y", type=float)
    parser.add_argument("--start-yaw", type=float, default=0.0)
    parser.add_argument("--goal-x", type=float)
    parser.add_argument("--goal-y", type=float)
    parser.add_argument("--goal-z", type=float, default=0.793)
    parser.add_argument("--navigation-grid-resolution-m", type=float, default=0.35)
    parser.add_argument("--navigation-max-speed-mps", type=float, default=0.55)
    parser.add_argument("--navigation-waypoint-tolerance-m", type=float, default=0.35)
    parser.add_argument("--navigation-yaw-gain", type=float, default=1.2)
    parser.add_argument("--navigation-max-yaw-rate", type=float, default=0.9)
    parser.add_argument("--disable-navigation-planner", action="store_true")
    parser.add_argument("--no-policy-source-snapshot", action="store_true")
    parser.add_argument(
        "--wrapper-xml",
        help=(
            "Deprecated compatibility flag. The handoff renderer generates a camera-enabled "
            "official Unitree MJCF instead."
        ),
    )
    args = parser.parse_args()
    command_xyz = None
    if (
        args.command_x is not None
        or args.command_y is not None
        or args.command_yaw is not None
    ):
        command_xyz = [
            0.5 if args.command_x is None else args.command_x,
            0.0 if args.command_y is None else args.command_y,
            0.0 if args.command_yaw is None else args.command_yaw,
        ]
    navigation_goal_xyz = (
        [args.goal_x, args.goal_y, args.goal_z]
        if args.goal_x is not None and args.goal_y is not None
        else None
    )
    result = build_official_g1_policy_handoff(
        capture_root=Path(args.capture_root),
        policy_manifest_path=args.policy_manifest,
        unitree_rl_gym_root=args.unitree_rl_gym_root,
        output_dir=args.output_dir,
        render_width=args.render_width,
        render_height=args.render_height,
        render_fps=args.render_fps,
        video_crf=args.video_crf,
        max_frames=args.max_frames,
        camera_set=args.camera_set,
        duration_seconds=args.duration_seconds,
        target_displacement_m=args.target_displacement_m,
        fall_height_threshold_m=args.fall_height_threshold_m,
        command_xyz=command_xyz,
        collision_proxy_limit=args.collision_proxy_limit,
        base_path_clearance_m=args.base_path_clearance_m,
        initial_root_xy=[args.start_x, args.start_y]
        if args.start_x is not None and args.start_y is not None
        else None,
        initial_root_yaw=args.start_yaw,
        navigation_goal_xyz=navigation_goal_xyz,
        navigation_grid_resolution_m=args.navigation_grid_resolution_m,
        navigation_max_speed_mps=args.navigation_max_speed_mps,
        navigation_waypoint_tolerance_m=args.navigation_waypoint_tolerance_m,
        navigation_yaw_gain=args.navigation_yaw_gain,
        navigation_max_yaw_rate=args.navigation_max_yaw_rate,
        enable_navigation_planner=not args.disable_navigation_planner,
        copy_policy_source_snapshot=not args.no_policy_source_snapshot,
    )
    print(result["artifacts"]["rendered_motion_manifest"])
    print(result["manifest_path"])
    print(result["status"])
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
