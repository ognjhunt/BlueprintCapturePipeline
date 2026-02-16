"""NuRec-first swappable asset orchestration package."""

from .capture_bridge import (
    CaptureDescriptor,
    build_capture_bundle_constraints,
    build_scene_manifest_seed,
    build_scene_request_from_descriptor,
)
from .swap_orchestrator import OrchestratorConfig, run_swap_pipeline

__version__ = "2.0.0"

__all__ = [
    "__version__",
    "CaptureDescriptor",
    "build_capture_bundle_constraints",
    "build_scene_manifest_seed",
    "build_scene_request_from_descriptor",
    "OrchestratorConfig",
    "run_swap_pipeline",
]
