"""Pinned runtime identity for the DLO-Lab CUDA development canary."""

from __future__ import annotations

from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .measurement_dlo_lab_cable_adapter import (
    EXPECTED_DISTRIBUTION_VERSION,
    EXPECTED_SOURCE_COMMIT,
)


SCHEMA_VERSION = "measurement_dlo_lab_runtime_release.v1"
PYTORCH_VERSION = "2.9.1"
# Quadrants 0.8.0's release workflow exercises its CUDA wheel against the
# PyTorch cu128 index.  Keep the DLO development image on that upstream-tested
# CUDA line instead of the CUDA 13.0 image used by the first three diagnostics.
CUDA_VERSION = "12.8"
RUNTIME_IMAGE = (
    "pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime"
    "@sha256:7b324d212a4450795b49edba9949b7cdc72429148a64e974334bfe5774d51385"
)


class MeasurementDloLabRuntimeReleaseError(ValueError):
    pass


def build_measurement_dlo_lab_runtime_release() -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "runtime_image_digest": RUNTIME_IMAGE,
        "pytorch_version": PYTORCH_VERSION,
        "cuda_version": CUDA_VERSION,
        "dlo_lab_source_repository": "https://github.com/UMass-Embodied-AGI/DLO-Lab.git",
        "dlo_lab_source_commit": EXPECTED_SOURCE_COMMIT,
        "dlo_lab_distribution_name": "genesis-world",
        "dlo_lab_distribution_version": EXPECTED_DISTRIBUTION_VERSION,
        "required_backend": "cuda",
        "cpu_fallback_allowed": False,
        "source_install_requires_network": True,
        "benchmark_assets_required": False,
        "development_only": True,
        "qualification_created": False,
        "r7_admission_created": False,
        "production_route_eligible": False,
        "physical_success_established": False,
    }
    value["runtime_release_digest"] = canonical_digest(value, digest_field="runtime_release_digest")
    return value


def validate_measurement_dlo_lab_runtime_release(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    supplied = dict(value)
    expected = build_measurement_dlo_lab_runtime_release()
    if supplied != expected:
        raise MeasurementDloLabRuntimeReleaseError("measurement_dlo_lab_runtime_release_mismatch")
    return supplied


__all__ = [
    "CUDA_VERSION",
    "MeasurementDloLabRuntimeReleaseError",
    "PYTORCH_VERSION",
    "RUNTIME_IMAGE",
    "SCHEMA_VERSION",
    "build_measurement_dlo_lab_runtime_release",
    "validate_measurement_dlo_lab_runtime_release",
]
