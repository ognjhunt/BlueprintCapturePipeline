"""Command-line interface for 3DGS rendering.

This module provides the CLI entry point for the render-static-scene command.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Render static scene video from ZeroScene bundle for DWM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "zeroscene_path",
        type=Path,
        nargs="?",
        help="Path to zeroscene/ directory",
    )

    # Output options
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-o", "--output",
        type=Path,
        help="Output video path (e.g., static_scene.mp4)",
    )
    output_group.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for image sequence",
    )

    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Image format for sequence output (default: png)",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video frame rate (default: 30)",
    )

    # Rendering options
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["diff-gaussian-rasterization", "gsplat", "cpu-numpy"],
        help="Rasterizer backend (default: auto-select best)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)",
    )

    parser.add_argument(
        "--background",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("R", "G", "B"),
        help="Background color (default: 0 0 0 = black)",
    )

    parser.add_argument(
        "--sh-degree",
        type=int,
        default=-1,
        help="SH degree to use (-1 = auto, 0-3 for manual)",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Gaussian scale modifier (default: 1.0)",
    )

    parser.add_argument(
        "--min-opacity",
        type=float,
        default=0.0,
        help="Minimum opacity filter (default: 0.0)",
    )

    # Info options
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print scene info and exit",
    )

    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List available rasterizer backends and exit",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def print_scene_info(zeroscene_path: Path) -> None:
    """Print information about the ZeroScene bundle."""
    print(f"\nZeroScene Bundle: {zeroscene_path}")
    print("=" * 60)

    # Scene info
    scene_info_path = zeroscene_path / "scene_info.json"
    if scene_info_path.exists():
        scene_info = json.loads(scene_info_path.read_text())
        print("\nScene Info:")
        print(f"  Capture ID: {scene_info.get('capture_id', 'N/A')}")
        print(f"  DWM Compatible: {scene_info.get('dwm_compatible', False)}")
        print(f"  Has Gaussians: {scene_info.get('has_gaussians', False)}")
        print(f"  Gaussians Format: {scene_info.get('gaussians_format', 'N/A')}")
        print(f"  Scale Factor: {scene_info.get('scale_factor', 1.0)}")
    else:
        print("\nWarning: scene_info.json not found")

    # Gaussians
    gaussians_path = zeroscene_path / "background" / "gaussians.ply"
    if gaussians_path.exists():
        size_mb = gaussians_path.stat().st_size / (1024 * 1024)
        print(f"\nGaussians: {gaussians_path}")
        print(f"  Size: {size_mb:.1f} MB")
    else:
        print("\nWarning: gaussians.ply not found")

    # Camera
    intrinsics_path = zeroscene_path / "camera" / "intrinsics.json"
    trajectory_path = zeroscene_path / "camera" / "trajectory.json"

    if intrinsics_path.exists():
        intrinsics = json.loads(intrinsics_path.read_text())
        print(f"\nCamera Intrinsics:")
        print(f"  Resolution: {intrinsics.get('width', 'N/A')}x{intrinsics.get('height', 'N/A')}")
        print(f"  Focal Length: fx={intrinsics.get('fx', 'N/A'):.1f}, fy={intrinsics.get('fy', 'N/A'):.1f}")
        print(f"  Principal Point: cx={intrinsics.get('cx', 'N/A'):.1f}, cy={intrinsics.get('cy', 'N/A'):.1f}")

    if trajectory_path.exists():
        trajectory = json.loads(trajectory_path.read_text())
        print(f"\nCamera Trajectory:")
        print(f"  Frames: {len(trajectory)}")
        if trajectory:
            print(f"  First frame: {trajectory[0].get('frame_id', 'N/A')}")
            print(f"  Last frame: {trajectory[-1].get('frame_id', 'N/A')}")


def list_backends() -> None:
    """Print available rasterizer backends."""
    from .rasterizer.backend_selector import get_rasterizer_info
    print(get_rasterizer_info())


def render(args: argparse.Namespace) -> int:
    """Run the rendering pipeline."""
    from .renderer import GaussianRenderer, RenderSettings

    zeroscene_path = args.zeroscene_path

    # Validate input
    if not zeroscene_path.exists():
        print(f"Error: ZeroScene path not found: {zeroscene_path}")
        return 1

    if not (zeroscene_path / "background" / "gaussians.ply").exists():
        print(f"Error: Gaussians not found in {zeroscene_path}")
        print("Make sure the pipeline was run with DWM-compatible export.")
        return 1

    # Create settings
    settings = RenderSettings(
        background_color=tuple(args.background),
        scaling_modifier=args.scale,
        sh_degree=args.sh_degree,
        min_opacity=args.min_opacity,
    )

    # Create renderer
    print(f"\nLoading scene from {zeroscene_path}...")
    start_time = time.time()

    try:
        renderer = GaussianRenderer.from_zeroscene(
            zeroscene_path,
            backend=args.backend,
            device=args.device,
            settings=settings,
        )
    except Exception as e:
        print(f"Error loading scene: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    load_time = time.time() - start_time
    print(f"Loaded in {load_time:.1f}s")

    if args.verbose:
        print(f"\nScene info: {json.dumps(renderer.get_scene_info(), indent=2)}")

    # Render frames
    print(f"\nRendering {len(renderer.trajectory)} frames...")
    start_time = time.time()

    def progress_callback(current: int, total: int) -> None:
        if current % 10 == 0 or current == total:
            elapsed = time.time() - start_time
            fps = current / elapsed if elapsed > 0 else 0
            eta = (total - current) / fps if fps > 0 else 0
            print(f"  Frame {current}/{total} ({fps:.1f} fps, ETA: {eta:.0f}s)")

    try:
        frames = renderer.render_trajectory(progress_callback=progress_callback)
    except Exception as e:
        print(f"Error rendering: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    render_time = time.time() - start_time
    avg_fps = len(frames) / render_time if render_time > 0 else 0
    print(f"Rendered {len(frames)} frames in {render_time:.1f}s ({avg_fps:.1f} fps)")

    # Save output
    if args.output:
        print(f"\nSaving video to {args.output}...")
        try:
            renderer.save_video(frames, args.output, fps=args.fps)
        except Exception as e:
            print(f"Error saving video: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    else:
        print(f"\nSaving frames to {args.output_dir}...")
        try:
            renderer.save_frames(frames, args.output_dir, format=args.format)
        except Exception as e:
            print(f"Error saving frames: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    print("\nDone!")
    print(f"\nThis video is the 'static-scene conditioning video' for DWM.")
    print("Next steps:")
    print("  1. Extract hand meshes using HaMeR (when available)")
    print("  2. Use DWM to generate interaction videos")

    return 0


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Handle info commands
    if args.list_backends:
        list_backends()
        return 0

    if args.info and args.zeroscene_path:
        print_scene_info(args.zeroscene_path)
        return 0

    # Validate required arguments
    if not args.zeroscene_path:
        print("Error: zeroscene_path is required")
        print("Usage: render-static-scene <zeroscene_path> -o <output.mp4>")
        return 1

    if not args.output and not args.output_dir:
        print("Error: either --output or --output-dir is required")
        return 1

    # Run rendering
    return render(args)


if __name__ == "__main__":
    sys.exit(main())
