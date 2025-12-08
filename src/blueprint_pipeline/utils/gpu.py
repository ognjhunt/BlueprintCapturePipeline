"""GPU detection and management utilities."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Optional, List
from contextlib import contextmanager


@dataclass
class GPUInfo:
    """Information about an available GPU."""
    index: int
    name: str
    memory_total_mb: int
    memory_free_mb: int
    memory_used_mb: int
    utilization_percent: float
    temperature_c: Optional[int] = None

    @property
    def memory_available_gb(self) -> float:
        return self.memory_free_mb / 1024

    @property
    def memory_total_gb(self) -> float:
        return self.memory_total_mb / 1024


def get_available_gpu() -> Optional[GPUInfo]:
    """Get information about the first available GPU.

    Returns:
        GPUInfo if GPU is available, None otherwise.
    """
    gpus = list_gpus()
    if gpus:
        return gpus[0]
    return None


def list_gpus() -> List[GPUInfo]:
    """List all available GPUs with their status.

    Returns:
        List of GPUInfo objects for each detected GPU.
    """
    try:
        # Try nvidia-smi first
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append(
                    GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        memory_total_mb=int(parts[2]),
                        memory_free_mb=int(parts[3]),
                        memory_used_mb=int(parts[4]),
                        utilization_percent=float(parts[5]),
                        temperature_c=int(parts[6]) if len(parts) > 6 else None,
                    )
                )
        return gpus
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return []


def check_gpu_memory(required_gb: float = 20.0) -> bool:
    """Check if sufficient GPU memory is available.

    Args:
        required_gb: Minimum required free memory in GB.

    Returns:
        True if sufficient memory is available.
    """
    gpu = get_available_gpu()
    if gpu is None:
        return False
    return gpu.memory_available_gb >= required_gb


def check_cuda_available() -> bool:
    """Check if CUDA is available via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_torch_device() -> str:
    """Get the appropriate torch device string.

    Returns:
        'cuda' if available, 'mps' for Apple Silicon, else 'cpu'
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class GPUContext:
    """Context manager for GPU resource management.

    Provides utilities for:
    - Setting CUDA device visibility
    - Memory management
    - Resource cleanup
    """

    def __init__(
        self,
        device_index: int = 0,
        memory_fraction: Optional[float] = None,
        allow_growth: bool = True,
    ):
        """Initialize GPU context.

        Args:
            device_index: GPU device index to use.
            memory_fraction: Fraction of GPU memory to allocate (0-1).
            allow_growth: Whether to allow memory growth (PyTorch default).
        """
        self.device_index = device_index
        self.memory_fraction = memory_fraction
        self.allow_growth = allow_growth
        self._original_cuda_visible = None

    def __enter__(self) -> "GPUContext":
        """Set up GPU environment."""
        # Store original CUDA_VISIBLE_DEVICES
        self._original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

        # Set device visibility
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_index)

        # Configure PyTorch if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(0)  # After filtering, device 0 is our target

                # Set memory fraction if specified
                if self.memory_fraction is not None:
                    torch.cuda.set_per_process_memory_fraction(
                        self.memory_fraction, device=0
                    )
        except ImportError:
            pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up GPU resources."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        # Restore original CUDA_VISIBLE_DEVICES
        if self._original_cuda_visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._original_cuda_visible
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]

        return False

    def clear_cache(self):
        """Clear GPU memory cache."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def memory_stats(self) -> dict:
        """Get current memory statistics.

        Returns:
            Dict with allocated, reserved, and free memory in MB.
        """
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                    "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
                    "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
                }
        except ImportError:
            pass
        return {"allocated_mb": 0, "reserved_mb": 0, "max_allocated_mb": 0}


@contextmanager
def gpu_memory_guard(threshold_mb: int = 1000):
    """Context manager that clears cache if memory exceeds threshold.

    Args:
        threshold_mb: Memory threshold in MB to trigger cleanup.
    """
    try:
        import torch
        yield
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            if allocated > threshold_mb:
                torch.cuda.empty_cache()
    except ImportError:
        yield
