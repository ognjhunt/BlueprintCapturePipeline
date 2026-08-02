"""Immutable official Isaac runtime identity for paid measurement canaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "measurement_isaac_runtime_release.v1"
ISAAC_VERSION = "6.0.1"
RUNTIME_IMAGE = (
    "nvcr.io/nvidia/isaac-sim:6.0.1@"
    "sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9"
)
RUNTIME_PLATFORM = "linux/amd64"


class MeasurementIsaacRuntimeReleaseError(ValueError):
    pass


def build_measurement_isaac_runtime_release() -> dict[str, Any]:
    """Bind the official multi-arch release digest without runtime claims."""

    release = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "isaac_sim_version": ISAAC_VERSION,
        "resolved_image_digest": RUNTIME_IMAGE,
        "required_platform": RUNTIME_PLATFORM,
        "source_kind": "official_nvidia_ngc_release",
        "runtime_execution_completed": False,
        "provider_startup_proven": False,
        "measurement_suite_completed": False,
        "qualification_created": False,
        "r7_admission_created": False,
        "physical_success_established": False,
        "raw_secret_values_recorded": False,
        "proof_effect": "none",
        "claim_ceiling": "immutable_official_runtime_identity_only",
    }
    release["runtime_release_digest"] = canonical_digest(
        release, digest_field="runtime_release_digest"
    )
    return release


def validate_measurement_isaac_runtime_release(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    supplied = json.loads(json.dumps(dict(value)))
    expected = build_measurement_isaac_runtime_release()
    if supplied != expected:
        raise MeasurementIsaacRuntimeReleaseError(
            "measurement_isaac_runtime_release_mismatch"
        )
    return expected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    release = build_measurement_isaac_runtime_release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ISAAC_VERSION",
    "MeasurementIsaacRuntimeReleaseError",
    "RUNTIME_IMAGE",
    "RUNTIME_PLATFORM",
    "SCHEMA_VERSION",
    "build_measurement_isaac_runtime_release",
    "main",
    "validate_measurement_isaac_runtime_release",
]
