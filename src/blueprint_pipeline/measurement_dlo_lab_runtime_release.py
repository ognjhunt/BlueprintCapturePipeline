"""Pinned runtime identity for the DLO-Lab CUDA development canary."""

from __future__ import annotations

from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .measurement_dlo_lab_cable_adapter import (
    EXPECTED_DISTRIBUTION_VERSION,
    EXPECTED_SOURCE_COMMIT,
    HEADLESS_DISPLAY_MODE,
)


SCHEMA_VERSION = "measurement_dlo_lab_runtime_release.v2"
PYTHON_VERSION = "3.12.11"
PYTHON_CONDA_SPEC = "python=3.12.11=h9e4cc4f_0_cpython"
PYTHON_CONDA_PACKAGE_SHA256 = (
    "sha256:6cca004806ceceea9585d4d655059e951152fc774a471593d4f5138e6a54c81d"
)
PYTORCH_VERSION = "2.9.1+cu128"
PYTORCH_WHEEL_URL = (
    "https://download-r2.pytorch.org/whl/cu128/"
    "torch-2.9.1%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl"
)
PYTORCH_WHEEL_SHA256 = "sha256:7cb4018f4ce68b61fd3ef87dc1c4ca520731c7b5b200e360ad47b612d7844063"
QUADRANTS_VERSION = "0.8.0"
QUADRANTS_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/05/c1/"
    "f3a8de448bfdef42507d82b97264b2f5c0383067ba10ec438d13a84243fe/"
    "quadrants-0.8.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
)
QUADRANTS_WHEEL_SHA256 = "sha256:6ee51b7299882dd78bdc04ce2385566dbcbb59bcf0b2ce0951af7a1ddfe51a40"
# Quadrants 0.8.0's release workflow exercises its CUDA wheel against the
# PyTorch cu128 index.  Keep the DLO development image on that upstream-tested
# CUDA line instead of the CUDA 13.0 image used by the first three diagnostics.
CUDA_VERSION = "12.8"
RUNTIME_IMAGE = (
    "pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime"
    "@sha256:7b324d212a4450795b49edba9949b7cdc72429148a64e974334bfe5774d51385"
)
DISPLAY_MODE = HEADLESS_DISPLAY_MODE
REQUIRED_DEBIAN_PACKAGES = [
    "git",
    "ca-certificates",
    "gdb",
    "libegl1",
    "libgl1",
    "libglib2.0-0",
]


class MeasurementDloLabRuntimeReleaseError(ValueError):
    pass


def build_measurement_dlo_lab_runtime_release() -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "runtime_image_digest": RUNTIME_IMAGE,
        "python_version": PYTHON_VERSION,
        "python_conda_spec": PYTHON_CONDA_SPEC,
        "python_conda_package_sha256": PYTHON_CONDA_PACKAGE_SHA256,
        "pytorch_version": PYTORCH_VERSION,
        "pytorch_wheel_url": PYTORCH_WHEEL_URL,
        "pytorch_wheel_sha256": PYTORCH_WHEEL_SHA256,
        "quadrants_version": QUADRANTS_VERSION,
        "quadrants_wheel_url": QUADRANTS_WHEEL_URL,
        "quadrants_wheel_sha256": QUADRANTS_WHEEL_SHA256,
        "cuda_version": CUDA_VERSION,
        "display_mode": DISPLAY_MODE,
        "required_debian_packages": list(REQUIRED_DEBIAN_PACKAGES),
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
    "DISPLAY_MODE",
    "MeasurementDloLabRuntimeReleaseError",
    "PYTHON_CONDA_PACKAGE_SHA256",
    "PYTHON_CONDA_SPEC",
    "PYTHON_VERSION",
    "PYTORCH_VERSION",
    "PYTORCH_WHEEL_SHA256",
    "PYTORCH_WHEEL_URL",
    "QUADRANTS_VERSION",
    "QUADRANTS_WHEEL_SHA256",
    "QUADRANTS_WHEEL_URL",
    "REQUIRED_DEBIAN_PACKAGES",
    "RUNTIME_IMAGE",
    "SCHEMA_VERSION",
    "build_measurement_dlo_lab_runtime_release",
    "validate_measurement_dlo_lab_runtime_release",
]
