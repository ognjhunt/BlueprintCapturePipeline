"""Qualification-first capture pipeline package with lazy public imports.

Keeping ``__init__`` side-effect free is a runtime contract: console/module
entrypoints must not be imported before ``python -m`` executes them.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__version__ = "2.0.0"

_LAZY_PUBLIC_ATTRS = {
    "CaptureDescriptor": (".capture_bridge", "CaptureDescriptor"),
    "build_capture_bundle_constraints": (
        ".capture_bridge",
        "build_capture_bundle_constraints",
    ),
    "build_scene_manifest_seed": (".capture_bridge", "build_scene_manifest_seed"),
    "build_scene_request_from_descriptor": (
        ".capture_bridge",
        "build_scene_request_from_descriptor",
    ),
    "materialize_capture_bundle": (".materialization", "materialize_capture_bundle"),
    "preview_capture_bundle": (".materialization", "preview_capture_bundle"),
    "PipelineConfig": (".capture_orchestrator", "PipelineConfig"),
    "run_capture_pipeline": (".capture_orchestrator", "run_capture_pipeline"),
    "run_evaluation_prep_stage": (
        ".evaluation_prep_stage",
        "run_evaluation_prep_stage",
    ),
    "run_object_geometry_stage": (
        ".object_geometry_stage",
        "run_object_geometry_stage",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_PUBLIC_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_PUBLIC_ATTRS})


__all__ = ["__version__", *_LAZY_PUBLIC_ATTRS]
