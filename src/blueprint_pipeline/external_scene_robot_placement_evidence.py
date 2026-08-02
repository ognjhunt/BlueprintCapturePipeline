"""Independently rehash exact external-scene Isaac robot visibility evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .external_scene_isaac_verification import (
    build_external_scene_isaac_verification_request,
)
from .isaac_reconstruction_verification import build_isaac_runtime_result_v3
from .provider_robot_placement_evidence import (
    ProviderRobotPlacementEvidenceError,
    _build_signed_isaac_visual_placement_evidence,
)


SCHEMA_VERSION = "external_scene_robot_placement_evidence.v1"


def _runtime_builder(
    value: Mapping[str, Any], *, verification_request: Mapping[str, Any]
) -> dict[str, Any]:
    del verification_request
    return build_isaac_runtime_result_v3(value)


def build_external_scene_robot_placement_evidence(
    *,
    verification_request: Mapping[str, Any],
    runtime_result: Mapping[str, Any],
    runtime_artifact_root: str | Path,
) -> dict[str, Any]:
    """Qualify visibility only; never infer clearance, reach, or task success."""

    return _build_signed_isaac_visual_placement_evidence(
        verification_request=verification_request,
        runtime_result=runtime_result,
        runtime_artifact_root=runtime_artifact_root,
        request_builder=build_external_scene_isaac_verification_request,
        runtime_builder=_runtime_builder,
        schema_version=SCHEMA_VERSION,
        digest_field="visual_placement_evidence_digest",
    )


ExternalSceneRobotPlacementEvidenceError = ProviderRobotPlacementEvidenceError


__all__ = [
    "ExternalSceneRobotPlacementEvidenceError",
    "SCHEMA_VERSION",
    "build_external_scene_robot_placement_evidence",
]
