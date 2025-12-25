#!/usr/bin/env python3
"""Setup script for BlueprintCapturePipeline environment.

This script helps install and verify all dependencies for the capture pipeline,
including system dependencies like COLMAP and optional GPU components.

Usage:
    python scripts/setup_environment.py [--check] [--install-colmap] [--install-sam]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_status(message: str, status: str = "info") -> None:
    """Print colored status message."""
    colors = {
        "ok": Colors.GREEN + "✓" + Colors.RESET,
        "warn": Colors.YELLOW + "⚠" + Colors.RESET,
        "error": Colors.RED + "✗" + Colors.RESET,
        "info": Colors.BLUE + "→" + Colors.RESET,
    }
    symbol = colors.get(status, colors["info"])
    print(f"  {symbol} {message}")


def print_header(title: str) -> None:
    """Print section header."""
    print(f"\n{Colors.BOLD}{title}{Colors.RESET}")
    print("=" * len(title))


def run_command(
    cmd: List[str],
    capture: bool = True,
    check: bool = True,
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    """Run a command and return result."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout,
            check=check,
        )
        return result
    except subprocess.CalledProcessError as e:
        return e
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return None


def check_python_version() -> Tuple[bool, str]:
    """Check Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor} (requires 3.10+)"


def check_pip() -> Tuple[bool, str]:
    """Check pip installation."""
    result = run_command([sys.executable, "-m", "pip", "--version"], check=False)
    if result and hasattr(result, 'returncode') and result.returncode == 0:
        version = result.stdout.split()[1] if result.stdout else "unknown"
        return True, f"pip {version}"
    return False, "pip not found"


def check_cuda() -> Tuple[bool, str]:
    """Check CUDA availability."""
    # Check via PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"CUDA {torch.version.cuda} ({torch.cuda.get_device_name(0)})"
        return False, "CUDA not available (PyTorch CPU mode)"
    except ImportError:
        pass

    # Check nvidia-smi
    result = run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=False)
    if result and hasattr(result, 'returncode') and result.returncode == 0:
        gpu_name = result.stdout.strip().split('\n')[0]
        return True, f"NVIDIA GPU: {gpu_name}"

    return False, "No CUDA GPU detected"


def check_colmap() -> Tuple[bool, str]:
    """Check COLMAP installation."""
    result = run_command(["colmap", "--version"], check=False)
    if result and hasattr(result, 'returncode') and result.returncode == 0:
        # Parse version from output
        version = "installed"
        if result.stdout:
            for line in result.stdout.split('\n'):
                if 'COLMAP' in line:
                    version = line.strip()
                    break
        return True, version
    return False, "COLMAP not installed"


def check_opencv() -> Tuple[bool, str]:
    """Check OpenCV installation."""
    try:
        import cv2
        return True, f"OpenCV {cv2.__version__}"
    except ImportError:
        return False, "OpenCV not installed"


def check_pytorch() -> Tuple[bool, str]:
    """Check PyTorch installation."""
    try:
        import torch
        cuda_str = f" (CUDA {torch.version.cuda})" if torch.cuda.is_available() else " (CPU)"
        return True, f"PyTorch {torch.__version__}{cuda_str}"
    except ImportError:
        return False, "PyTorch not installed"


def check_segment_anything() -> Tuple[bool, str]:
    """Check Segment Anything installation."""
    try:
        from segment_anything import sam_model_registry
        return True, "segment-anything installed"
    except ImportError:
        return False, "segment-anything not installed"


def check_ultralytics() -> Tuple[bool, str]:
    """Check Ultralytics YOLO installation."""
    try:
        import ultralytics
        return True, f"ultralytics {ultralytics.__version__}"
    except ImportError:
        return False, "ultralytics not installed"


def check_sam_weights() -> Tuple[bool, str]:
    """Check for SAM model weights."""
    sam_dir = Path.home() / ".cache" / "sam"
    models_dir = Path("models")

    for model_type in ["vit_h", "vit_l", "vit_b"]:
        for search_dir in [sam_dir, models_dir]:
            weight_path = search_dir / f"sam_{model_type}.pth"
            if weight_path.exists():
                size_mb = weight_path.stat().st_size / (1024 * 1024)
                return True, f"SAM {model_type} ({size_mb:.0f}MB) at {weight_path}"

    return False, "SAM weights not found"


def install_colmap_linux() -> bool:
    """Install COLMAP on Linux."""
    print_status("Attempting to install COLMAP via apt...", "info")

    # Try apt install
    result = run_command(
        ["sudo", "apt-get", "install", "-y", "colmap"],
        capture=False,
        check=False,
        timeout=600,
    )

    if result and hasattr(result, 'returncode') and result.returncode == 0:
        print_status("COLMAP installed successfully", "ok")
        return True

    print_status("apt install failed, trying snap...", "warn")

    # Try snap
    result = run_command(
        ["sudo", "snap", "install", "colmap"],
        capture=False,
        check=False,
        timeout=600,
    )

    if result and hasattr(result, 'returncode') and result.returncode == 0:
        print_status("COLMAP installed via snap", "ok")
        return True

    print_status("Could not install COLMAP automatically", "error")
    print("  Please install manually: https://colmap.github.io/install.html")
    return False


def install_colmap_macos() -> bool:
    """Install COLMAP on macOS."""
    print_status("Attempting to install COLMAP via Homebrew...", "info")

    # Check if brew is available
    if not shutil.which("brew"):
        print_status("Homebrew not found, please install it first", "error")
        print("  Visit: https://brew.sh")
        return False

    result = run_command(
        ["brew", "install", "colmap"],
        capture=False,
        check=False,
        timeout=1200,
    )

    if result and hasattr(result, 'returncode') and result.returncode == 0:
        print_status("COLMAP installed successfully", "ok")
        return True

    print_status("Could not install COLMAP", "error")
    return False


def install_colmap() -> bool:
    """Install COLMAP based on platform."""
    system = platform.system().lower()

    if system == "linux":
        return install_colmap_linux()
    elif system == "darwin":
        return install_colmap_macos()
    else:
        print_status(f"Automatic installation not supported on {system}", "error")
        print("  Please install COLMAP manually: https://colmap.github.io/install.html")
        return False


def download_sam_weights(model_type: str = "vit_b") -> bool:
    """Download SAM model weights."""
    urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }

    if model_type not in urls:
        print_status(f"Unknown model type: {model_type}", "error")
        return False

    url = urls[model_type]
    output_dir = Path.home() / ".cache" / "sam"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sam_{model_type}.pth"

    if output_path.exists():
        print_status(f"SAM {model_type} weights already exist", "ok")
        return True

    print_status(f"Downloading SAM {model_type} weights...", "info")
    print(f"  This may take a while (~375MB for vit_b, ~1.2GB for vit_l, ~2.4GB for vit_h)")

    try:
        # Download with progress
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\r  Progress: {percent}%", end="", flush=True)

        urllib.request.urlretrieve(url, output_path, reporthook)
        print()  # New line after progress
        print_status(f"Downloaded to {output_path}", "ok")
        return True

    except Exception as e:
        print()
        print_status(f"Download failed: {e}", "error")
        if output_path.exists():
            output_path.unlink()
        return False


def install_package(package: str, pip_args: List[str] = None) -> bool:
    """Install a Python package."""
    cmd = [sys.executable, "-m", "pip", "install"]
    if pip_args:
        cmd.extend(pip_args)
    cmd.append(package)

    print_status(f"Installing {package}...", "info")

    result = run_command(cmd, capture=True, check=False, timeout=600)

    if result and hasattr(result, 'returncode') and result.returncode == 0:
        print_status(f"Installed {package}", "ok")
        return True

    print_status(f"Failed to install {package}", "error")
    if result and hasattr(result, 'stderr') and result.stderr:
        print(f"  Error: {result.stderr[:200]}")
    return False


def run_checks() -> Dict[str, Tuple[bool, str]]:
    """Run all environment checks."""
    checks = {}

    print_header("Python Environment")
    checks["python"] = check_python_version()
    print_status(checks["python"][1], "ok" if checks["python"][0] else "error")

    checks["pip"] = check_pip()
    print_status(checks["pip"][1], "ok" if checks["pip"][0] else "error")

    print_header("GPU / CUDA")
    checks["cuda"] = check_cuda()
    print_status(checks["cuda"][1], "ok" if checks["cuda"][0] else "warn")

    print_header("Core Dependencies")
    checks["pytorch"] = check_pytorch()
    print_status(checks["pytorch"][1], "ok" if checks["pytorch"][0] else "warn")

    checks["opencv"] = check_opencv()
    print_status(checks["opencv"][1], "ok" if checks["opencv"][0] else "warn")

    print_header("3D Reconstruction")
    checks["colmap"] = check_colmap()
    print_status(checks["colmap"][1], "ok" if checks["colmap"][0] else "warn")

    print_header("Dynamic Masking (Optional)")
    checks["sam"] = check_segment_anything()
    print_status(checks["sam"][1], "ok" if checks["sam"][0] else "info")

    checks["sam_weights"] = check_sam_weights()
    print_status(checks["sam_weights"][1], "ok" if checks["sam_weights"][0] else "info")

    checks["yolo"] = check_ultralytics()
    print_status(checks["yolo"][1], "ok" if checks["yolo"][0] else "info")

    return checks


def print_summary(checks: Dict[str, Tuple[bool, str]]) -> None:
    """Print summary of checks."""
    print_header("Summary")

    required = ["python", "pip"]
    recommended = ["pytorch", "opencv", "colmap", "cuda"]
    optional = ["sam", "sam_weights", "yolo"]

    all_required = all(checks.get(k, (False,))[0] for k in required)
    all_recommended = all(checks.get(k, (False,))[0] for k in recommended)

    if all_required and all_recommended:
        print_status("All core components ready!", "ok")
    elif all_required:
        print_status("Basic requirements met, some recommended components missing", "warn")
    else:
        print_status("Some required components missing", "error")

    # Print recommendations
    if not checks.get("colmap", (False,))[0]:
        print("\n  To install COLMAP:")
        print("    - Ubuntu: sudo apt install colmap")
        print("    - macOS: brew install colmap")
        print("    - Or run: python scripts/setup_environment.py --install-colmap")

    if not checks.get("pytorch", (False,))[0]:
        print("\n  To install PyTorch:")
        print("    pip install torch torchvision")
        print("    (Visit https://pytorch.org for GPU-specific installation)")

    if not checks.get("opencv", (False,))[0]:
        print("\n  To install OpenCV:")
        print("    pip install opencv-python")

    if not checks.get("sam", (False,))[0]:
        print("\n  For dynamic masking (optional):")
        print("    pip install git+https://github.com/facebookresearch/segment-anything.git")
        print("    Or run: python scripts/setup_environment.py --install-sam")


def main():
    parser = argparse.ArgumentParser(
        description="Setup BlueprintCapturePipeline environment"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only run checks, don't install anything",
    )
    parser.add_argument(
        "--install-colmap",
        action="store_true",
        help="Install COLMAP",
    )
    parser.add_argument(
        "--install-sam",
        action="store_true",
        help="Install Segment Anything and download weights",
    )
    parser.add_argument(
        "--sam-model",
        choices=["vit_b", "vit_l", "vit_h"],
        default="vit_b",
        help="SAM model variant to download (default: vit_b)",
    )
    parser.add_argument(
        "--install-all",
        action="store_true",
        help="Install all optional components",
    )

    args = parser.parse_args()

    print(f"\n{Colors.BOLD}BlueprintCapturePipeline Environment Setup{Colors.RESET}")
    print("=" * 44)

    # Run checks first
    checks = run_checks()

    # Install components if requested
    if args.install_colmap or args.install_all:
        print_header("Installing COLMAP")
        install_colmap()

    if args.install_sam or args.install_all:
        print_header("Installing Segment Anything")
        if not checks.get("sam", (False,))[0]:
            install_package(
                "git+https://github.com/facebookresearch/segment-anything.git"
            )
        download_sam_weights(args.sam_model)

    # Re-run checks if we installed anything
    if args.install_colmap or args.install_sam or args.install_all:
        print("\n" + "=" * 44)
        print("Re-checking environment after installation...")
        checks = run_checks()

    print_summary(checks)

    # Return success if all required checks pass
    required = ["python", "pip"]
    return 0 if all(checks.get(k, (False,))[0] for k in required) else 1


if __name__ == "__main__":
    sys.exit(main())
