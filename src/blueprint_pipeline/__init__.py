"""Qualification-first capture pipeline package."""

from .capture_bridge import (
    CaptureDescriptor,
    build_capture_bundle_constraints,
    build_scene_manifest_seed,
    build_scene_request_from_descriptor,
)
from .capture_orchestrator import run_capture_pipeline
from .materialization import materialize_capture_bundle, preview_capture_bundle
from .swap_orchestrator import OrchestratorConfig, run_swap_pipeline

__version__ = "2.0.0"

__all__ = [
    "__version__",
    "CaptureDescriptor",
    "build_capture_bundle_constraints",
    "build_scene_manifest_seed",
    "build_scene_request_from_descriptor",
    "run_capture_pipeline",
    "materialize_capture_bundle",
    "preview_capture_bundle",
    "OrchestratorConfig",
    "run_swap_pipeline",
]
