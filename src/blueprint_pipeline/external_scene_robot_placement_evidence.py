"""Independently rehash exact external-scene Isaac robot visibility evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import write_json
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification-request", required=True)
    parser.add_argument("--runtime-result", required=True)
    parser.add_argument("--runtime-artifact-root", required=True)
    parser.add_argument("--result-out", required=True)
    args = parser.parse_args(argv)
    request = json.loads(Path(args.verification_request).read_text(encoding="utf-8"))
    runtime = json.loads(Path(args.runtime_result).read_text(encoding="utf-8"))
    result = build_external_scene_robot_placement_evidence(
        verification_request=request,
        runtime_result=runtime,
        runtime_artifact_root=args.runtime_artifact_root,
    )
    write_json(Path(args.result_out), result)
    return 0


__all__ = [
    "ExternalSceneRobotPlacementEvidenceError",
    "SCHEMA_VERSION",
    "build_external_scene_robot_placement_evidence",
]


if __name__ == "__main__":
    raise SystemExit(main())
