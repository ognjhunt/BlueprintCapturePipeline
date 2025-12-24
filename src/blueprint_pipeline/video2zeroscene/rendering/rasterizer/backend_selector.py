"""Backend selection for 3DGS rasterization.

This module provides automatic selection of the best available rasterization
backend based on installed dependencies and hardware capabilities.

Priority Order:
    1. CUDARasterizer (diff-gaussian-rasterization) - Fastest, most accurate
    2. GsplatRasterizer (gsplat) - Good CUDA alternative, easier install
    3. CPURasterizer (numpy) - Always available fallback
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from .base import RasterizerBackend
from .cpu_rasterizer import CPURasterizer, is_available as cpu_available
from .cuda_rasterizer import CUDARasterizer, is_available as cuda_available
from .gsplat_rasterizer import GsplatRasterizer, is_available as gsplat_available


# Registry of available backends
BACKEND_REGISTRY: Dict[str, Type[RasterizerBackend]] = {
    "diff-gaussian-rasterization": CUDARasterizer,
    "gsplat": GsplatRasterizer,
    "cpu-numpy": CPURasterizer,
}

# Priority order for backend selection
BACKEND_PRIORITY = [
    "diff-gaussian-rasterization",
    "gsplat",
    "cpu-numpy",
]


def available_backends() -> List[str]:
    """Get list of available rasterization backends.

    Returns:
        List of backend names that can be instantiated
    """
    available = []

    if cuda_available():
        available.append("diff-gaussian-rasterization")
    if gsplat_available():
        available.append("gsplat")
    if cpu_available():
        available.append("cpu-numpy")

    return available


def get_best_rasterizer(
    preferred: Optional[str] = None,
    device: str = "cuda",
) -> RasterizerBackend:
    """Get the best available rasterizer backend.

    Args:
        preferred: Preferred backend name (if available)
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Instantiated rasterizer backend

    Raises:
        RuntimeError: If no rasterizer backend is available
    """
    # Check for preferred backend
    if preferred is not None:
        if preferred in BACKEND_REGISTRY:
            backend_cls = BACKEND_REGISTRY[preferred]
            try:
                if preferred == "cpu-numpy":
                    return backend_cls()
                else:
                    return backend_cls(device=device)
            except (ImportError, RuntimeError) as e:
                print(f"Warning: Preferred backend '{preferred}' not available: {e}")
        else:
            print(f"Warning: Unknown backend '{preferred}', falling back to auto-select")

    # If CPU requested, skip GPU backends
    if device == "cpu":
        return CPURasterizer()

    # Try backends in priority order
    for backend_name in BACKEND_PRIORITY:
        if backend_name == "diff-gaussian-rasterization" and cuda_available():
            try:
                return CUDARasterizer(device=device)
            except Exception as e:
                print(f"Failed to initialize CUDARasterizer: {e}")
                continue

        if backend_name == "gsplat" and gsplat_available():
            try:
                return GsplatRasterizer(device=device)
            except Exception as e:
                print(f"Failed to initialize GsplatRasterizer: {e}")
                continue

        if backend_name == "cpu-numpy":
            return CPURasterizer()

    # Should never reach here since CPURasterizer always works
    raise RuntimeError("No rasterizer backend available")


def get_rasterizer_info() -> str:
    """Get detailed information about available rasterizers.

    Returns:
        Multi-line string with availability information
    """
    lines = ["3DGS Rasterizer Backends:"]
    lines.append("-" * 40)

    backends = [
        ("diff-gaussian-rasterization", cuda_available(),
         "GPU (CUDA) - Official INRIA implementation"),
        ("gsplat", gsplat_available(),
         "GPU (CUDA) - PyTorch-native implementation"),
        ("cpu-numpy", cpu_available(),
         "CPU - Pure Python/NumPy fallback"),
    ]

    for name, available, description in backends:
        status = "✓" if available else "✗"
        lines.append(f"  [{status}] {name}")
        lines.append(f"      {description}")

    lines.append("")
    lines.append("Best available: " + (available_backends()[0] if available_backends() else "None"))

    return "\n".join(lines)
