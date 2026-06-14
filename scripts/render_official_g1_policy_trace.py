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
    parser.add_argument("--no-policy-source-snapshot", action="store_true")
    parser.add_argument(
        "--wrapper-xml",
        help=(
            "Deprecated compatibility flag. The handoff renderer generates a camera-enabled "
            "official Unitree MJCF instead."
        ),
    )
    args = parser.parse_args()
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
        copy_policy_source_snapshot=not args.no_policy_source_snapshot,
    )
    print(result["artifacts"]["rendered_motion_manifest"])
    print(result["manifest_path"])
    print(result["status"])
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
