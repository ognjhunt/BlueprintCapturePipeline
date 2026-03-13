"""Site-world-first capture pipeline package."""

from .capture_bridge import (
    CaptureDescriptor,
    build_capture_bundle_constraints,
    build_scene_manifest_seed,
    build_scene_request_from_descriptor,
)
from .capture_orchestrator import PipelineConfig, run_capture_pipeline
from .evaluation_prep_stage import run_evaluation_prep_stage
from .materialization import materialize_capture_bundle, preview_capture_bundle
from .object_geometry_stage import run_object_geometry_stage

__version__ = "2.0.0"

__all__ = [
    "__version__",
    "CaptureDescriptor",
    "build_capture_bundle_constraints",
    "build_scene_manifest_seed",
    "build_scene_request_from_descriptor",
    "PipelineConfig",
    "run_capture_pipeline",
    "run_evaluation_prep_stage",
    "materialize_capture_bundle",
    "run_object_geometry_stage",
    "preview_capture_bundle",
]
