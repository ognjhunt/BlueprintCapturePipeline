from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.measurement_isaac_runtime_release import (
    MeasurementIsaacRuntimeReleaseError,
    build_measurement_isaac_runtime_release,
    validate_measurement_isaac_runtime_release,
)


ROOT = Path(__file__).parents[1]


def test_measurement_isaac_runtime_release_is_exact_non_authorizing_and_schema_valid() -> None:
    release = build_measurement_isaac_runtime_release()
    assert release["isaac_sim_version"] == "6.0.1"
    assert release["resolved_image_digest"].endswith(
        "sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9"
    )
    assert release["runtime_execution_completed"] is False
    assert release["r7_admission_created"] is False
    assert validate_measurement_isaac_runtime_release(release) == release
    schema = json.loads(
        (ROOT / "docs/schemas/measurement_isaac_runtime_release.v1.schema.json").read_text()
    )
    jsonschema.validate(release, schema)


def test_measurement_isaac_runtime_release_rejects_image_drift() -> None:
    release = copy.deepcopy(build_measurement_isaac_runtime_release())
    release["resolved_image_digest"] = "nvcr.io/nvidia/isaac-sim@sha256:" + "0" * 64
    with pytest.raises(MeasurementIsaacRuntimeReleaseError, match="release_mismatch"):
        validate_measurement_isaac_runtime_release(release)
