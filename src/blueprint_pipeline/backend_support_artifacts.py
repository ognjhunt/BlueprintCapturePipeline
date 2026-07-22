"""Registry-backed discovery of optional model-backend support artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from .core.common import read_json_any


@dataclass(frozen=True)
class BackendSupportArtifactSpec:
    artifact_id: str
    backend_id: str
    relative_path: str
    missing_payload: Mapping[str, Any] | None = None


class ResolvedBackendSupportArtifact(NamedTuple):
    path: Path
    payload: dict[str, Any]
    relative_path: str


BACKEND_SUPPORT_ARTIFACTS = (
    BackendSupportArtifactSpec(
        artifact_id="cosmos_zero_shot_benchmark",
        backend_id="cosmos_predict2_5",
        relative_path=(
            "cosmos_zero_shot_validation/cosmos_zero_shot_benchmark.json"
        ),
    ),
    BackendSupportArtifactSpec(
        artifact_id="cosmos_training_export",
        backend_id="cosmos_predict2_5",
        relative_path="cosmos_training_export/manifest.json",
        missing_payload={
            "schema_version": "cosmos_training_export_result.v1",
            "status": "not_requested",
            "reason": "legacy_cosmos_support_artifact_not_supplied",
            "claim_boundary": {
                "evaluation_prep_executes_model_specific_exporters": False,
                "explicit_external_support_artifact_required": True,
            },
        },
    ),
    BackendSupportArtifactSpec(
        artifact_id="cosmos_lora_training",
        backend_id="cosmos_predict2_5",
        relative_path="cosmos_training_export/training_run_manifest.json",
    ),
)


def resolve_backend_support_artifacts(
    pipeline_dir: Path,
    *,
    backend_id: str | None = None,
) -> dict[str, ResolvedBackendSupportArtifact]:
    """Resolve registered artifacts without invoking a model-specific exporter."""

    resolved: dict[str, ResolvedBackendSupportArtifact] = {}
    for spec in BACKEND_SUPPORT_ARTIFACTS:
        if backend_id is not None and spec.backend_id != backend_id:
            continue
        path = pipeline_dir / spec.relative_path
        if path.is_file():
            raw = read_json_any(path)
            payload = dict(raw) if isinstance(raw, Mapping) else {}
        else:
            payload = dict(spec.missing_payload or {})
        resolved[spec.artifact_id] = ResolvedBackendSupportArtifact(
            path=path,
            payload=payload,
            relative_path=spec.relative_path,
        )
    return resolved
