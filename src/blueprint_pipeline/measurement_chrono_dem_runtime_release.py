"""Pinned source-build runtime identity for the Chrono::DEM CUDA canary."""

from __future__ import annotations

from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .measurement_chrono_dem_cuda_adapter import (
    EXPECTED_ENGINE_VERSION,
    EXPECTED_SOURCE_COMMIT,
)


SCHEMA_VERSION = "measurement_chrono_dem_cuda_runtime_release.v1"
CUDA_VERSION = "12.8.1"
UBUNTU_VERSION = "24.04"
RUNTIME_IMAGE = (
    "nvidia/cuda:12.8.1-devel-ubuntu24.04"
    "@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66"
)
AMD64_IMAGE_MANIFEST_DIGEST = (
    "sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43"
)
SOURCE_REPOSITORY = "https://github.com/projectchrono/chrono.git"
SOURCE_TAG = "10.0.0"
SOURCE_TAG_OBJECT = "94fc98ae6f7f2bdcaf4ea8d34ee0892409ac9810"
REQUIRED_DEBIAN_PACKAGES = [
    "ca-certificates",
    "cmake",
    "g++",
    "git",
    "libeigen3-dev",
    "ninja-build",
    "python3",
]
BUILD_CONFIGURATION = {
    "BUILD_BENCHMARKING": "OFF",
    "BUILD_DEMOS": "OFF",
    "BUILD_TESTING": "OFF",
    "CHRONO_CUDA_ARCHITECTURES": "native",
    "CH_ENABLE_MODULE_DEM": "ON",
    "CMAKE_BUILD_TYPE": "Release",
}


class MeasurementChronoDemRuntimeReleaseError(ValueError):
    pass


def build_measurement_chrono_dem_runtime_release() -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "runtime_image_digest": RUNTIME_IMAGE,
        "amd64_image_manifest_digest": AMD64_IMAGE_MANIFEST_DIGEST,
        "ubuntu_version": UBUNTU_VERSION,
        "cuda_version": CUDA_VERSION,
        "chrono_version": EXPECTED_ENGINE_VERSION,
        "chrono_source_repository": SOURCE_REPOSITORY,
        "chrono_source_tag": SOURCE_TAG,
        "chrono_source_tag_object": SOURCE_TAG_OBJECT,
        "chrono_source_commit": EXPECTED_SOURCE_COMMIT,
        "chrono_module": "chrono_dem",
        "required_debian_packages": list(REQUIRED_DEBIAN_PACKAGES),
        "build_configuration": dict(BUILD_CONFIGURATION),
        "required_backend": "cuda",
        "cpu_fallback_allowed": False,
        "source_build_requires_network": True,
        "benchmark_assets_required": False,
        "development_only": True,
        "qualification_created": False,
        "r7_admission_created": False,
        "production_route_eligible": False,
        "physical_success_established": False,
    }
    value["runtime_release_digest"] = canonical_digest(value, digest_field="runtime_release_digest")
    return value


def validate_measurement_chrono_dem_runtime_release(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    supplied = dict(value)
    if supplied != build_measurement_chrono_dem_runtime_release():
        raise MeasurementChronoDemRuntimeReleaseError(
            "measurement_chrono_dem_runtime_release_mismatch"
        )
    return supplied


__all__ = [
    "AMD64_IMAGE_MANIFEST_DIGEST",
    "BUILD_CONFIGURATION",
    "CUDA_VERSION",
    "MeasurementChronoDemRuntimeReleaseError",
    "REQUIRED_DEBIAN_PACKAGES",
    "RUNTIME_IMAGE",
    "SCHEMA_VERSION",
    "SOURCE_REPOSITORY",
    "SOURCE_TAG",
    "SOURCE_TAG_OBJECT",
    "UBUNTU_VERSION",
    "build_measurement_chrono_dem_runtime_release",
    "validate_measurement_chrono_dem_runtime_release",
]
