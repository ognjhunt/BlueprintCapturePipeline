#!/usr/bin/env python3
"""Check and report status of all pipeline dependencies.

Run this script to verify your environment is properly configured:
    python scripts/check_dependencies.py
"""

import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"  Python: {sys.version.split()[0]}")
        return True
    else:
        print(f"  Python: {sys.version.split()[0]} (requires >= 3.10)")
        return False


def check_package(name: str, import_name: str = None, version_attr: str = "__version__"):
    """Check if a package is installed and return version."""
    import_name = import_name or name
    try:
        module = __import__(import_name)
        version = getattr(module, version_attr, "available")
        return True, version
    except ImportError:
        return False, None


def check_colmap():
    """Check COLMAP availability (pycolmap or CLI)."""
    # Check pycolmap first
    try:
        import pycolmap
        return True, f"pycolmap {pycolmap.__version__} (Python bindings)"
    except ImportError:
        pass

    # Check CLI
    try:
        result = subprocess.run(
            ["colmap", "--help"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, "CLI"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False, None


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"CUDA {torch.version.cuda}"
        return False, "Not available (CPU mode)"
    except ImportError:
        return False, "PyTorch not installed"


def check_cuda_rasterizer():
    """Check CUDA-accelerated 3DGS rasterizer."""
    try:
        from diff_gaussian_rasterization import GaussianRasterizer
        return True, "available"
    except ImportError:
        return False, None


def main():
    print("=" * 60)
    print("Blueprint Capture Pipeline - Dependency Check")
    print("=" * 60)
    print()

    all_ok = True

    # Critical dependencies
    print("CRITICAL DEPENDENCIES (pipeline will fail without these):")
    print("-" * 60)

    if not check_python_version():
        all_ok = False

    packages = [
        ("numpy", "numpy", "NumPy"),
        ("opencv-python", "cv2", "OpenCV"),
        ("Pillow", "PIL", "Pillow"),
        ("scipy", "scipy", "SciPy"),
        ("torch", "torch", "PyTorch"),
    ]

    for pip_name, import_name, display_name in packages:
        ok, version = check_package(import_name)
        if ok:
            print(f"  {display_name}: {version}")
        else:
            print(f"  {display_name}: NOT INSTALLED")
            print(f"    Install with: pip install {pip_name}")
            all_ok = False

    # COLMAP (critical for non-ARKit paths)
    print()
    ok, version = check_colmap()
    if ok:
        print(f"  COLMAP: {version}")
    else:
        print(f"  COLMAP: NOT INSTALLED")
        print(f"    Install with: pip install pycolmap (recommended)")
        print(f"    Or: apt install colmap (Ubuntu) / brew install colmap (macOS)")
        all_ok = False

    print()
    print("OPTIONAL DEPENDENCIES (for better performance/features):")
    print("-" * 60)

    # CUDA
    ok, version = check_cuda()
    if ok:
        print(f"  CUDA: {version}")
    else:
        print(f"  CUDA: {version} (training will be slower)")

    # CUDA Rasterizer
    ok, version = check_cuda_rasterizer()
    if ok:
        print(f"  CUDA 3DGS Rasterizer: {version}")
    else:
        print(f"  CUDA 3DGS Rasterizer: NOT INSTALLED (10-100x slower rendering)")
        print(f"    Install with: pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git")

    # Other optional packages
    optional_packages = [
        ("plyfile", "plyfile", "PLYFile"),
        ("tqdm", "tqdm", "TQDM"),
        ("google-cloud-storage", "google.cloud.storage", "GCS"),
    ]

    for pip_name, import_name, display_name in optional_packages:
        ok, version = check_package(import_name)
        if ok:
            print(f"  {display_name}: {version}")
        else:
            print(f"  {display_name}: NOT INSTALLED (optional)")

    print()
    print("=" * 60)

    if all_ok:
        print("PIPELINE STATUS: READY")
        print("All critical dependencies are installed.")
    else:
        print("PIPELINE STATUS: NOT READY")
        print("Install missing critical dependencies before running the pipeline.")

    print("=" * 60)

    # Return exit code
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
