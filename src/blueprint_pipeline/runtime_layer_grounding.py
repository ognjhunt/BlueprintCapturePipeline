"""Compatibility wrappers for shared runtime-layer grounding contracts."""

from __future__ import annotations

import sys
from pathlib import Path

_CONTRACTS_SRC = Path(__file__).resolve().parents[3] / "BlueprintContracts" / "src"
if str(_CONTRACTS_SRC) not in sys.path and _CONTRACTS_SRC.is_dir():
    sys.path.insert(0, str(_CONTRACTS_SRC))

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
