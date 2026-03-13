"""Compatibility wrappers for shared runtime-layer grounding contracts."""

from __future__ import annotations

from blueprint_contracts.runtime_layer_contract import (
    DEGRADED_EDITABLE_RATIO_THRESHOLD,
    EDITABLE_LOW_CONFIDENCE_THRESHOLD,
    LOCK_VIOLATION_RETRY_BUDGET,
    PROTECTED_OBSERVED_THRESHOLD,
    PROTECTED_RECONSTRUCTED_THRESHOLD,
    TASK_CRITICAL_DILATION_PX,
    TASK_CRITICAL_OVERRIDE_THRESHOLD,
    build_canonical_render_policy,
    build_presentation_variance_policy,
    build_protected_regions_manifest,
    classify_region,
    grounding_fields_from_provenance,
    task_critical_object_ids,
    with_grounding_fields,
)
from blueprint_contracts.canonical_package import compute_canonical_package_version

__all__ = [
    "DEGRADED_EDITABLE_RATIO_THRESHOLD",
    "EDITABLE_LOW_CONFIDENCE_THRESHOLD",
    "LOCK_VIOLATION_RETRY_BUDGET",
    "PROTECTED_OBSERVED_THRESHOLD",
    "PROTECTED_RECONSTRUCTED_THRESHOLD",
    "TASK_CRITICAL_DILATION_PX",
    "TASK_CRITICAL_OVERRIDE_THRESHOLD",
    "build_canonical_render_policy",
    "build_presentation_variance_policy",
    "build_protected_regions_manifest",
    "classify_region",
    "compute_canonical_package_version",
    "grounding_fields_from_provenance",
    "task_critical_object_ids",
    "with_grounding_fields",
]
