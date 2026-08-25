"""Closed production registry for GPU-backed scene-configuration stages.

Stage adapters validate and normalize scientific outputs.  Producers are the
separate execution side: they run the admitted tool inside the already-owned
parent allocation and return exact artifacts to its adapter.  A caller cannot
inject artifact paths or select Python/shell entrypoints.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_adapters import (
    ADMITTED_STAGE_ADAPTER_IDENTITIES,
)


PRODUCTION_RESULT_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_stage_production.v1"
)
StageProducer = Callable[..., Mapping[str, Any]]


class TaskEvaluationSceneConfigurationStageProducerError(RuntimeError):
    """A GPU stage did not resolve to one installed production producer."""


@dataclass(frozen=True, slots=True)
class SceneConfigurationStageProducerIdentity:
    capability: str
    adapter_id: str
    version: str


ADMITTED_PRODUCER_IDENTITIES = tuple(
    SceneConfigurationStageProducerIdentity(
        identity.capability, identity.adapter_id, identity.version
    )
    for identity in ADMITTED_STAGE_ADAPTER_IDENTITIES
    if identity.execution_class == "gpu_canary"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _identity(stage: Mapping[str, Any]) -> SceneConfigurationStageProducerIdentity:
    adapter = stage.get("adapter")
    if not isinstance(adapter, Mapping):
        raise TaskEvaluationSceneConfigurationStageProducerError(
            "scene_configuration_stage_producer_identity_invalid"
        )
    return SceneConfigurationStageProducerIdentity(
        str(stage.get("capability") or ""),
        str(adapter.get("id") or ""),
        str(adapter.get("version") or ""),
    )


class SceneConfigurationStageProducerRegistry:
    """Resolve only fixed producer identities and verify every returned byte."""

    def __init__(
        self,
        handlers: Mapping[SceneConfigurationStageProducerIdentity, StageProducer],
        *,
        admitted: Sequence[SceneConfigurationStageProducerIdentity] = (
            ADMITTED_PRODUCER_IDENTITIES
        ),
    ) -> None:
        admitted_set = frozenset(admitted)
        if len(admitted_set) != len(admitted) or any(
            identity not in admitted_set or not callable(handler)
            for identity, handler in handlers.items()
        ):
            raise TaskEvaluationSceneConfigurationStageProducerError(
                "scene_configuration_stage_producer_registry_invalid"
            )
        self._admitted = admitted_set
        self._handlers = dict(handlers)

    def execute(
        self,
        *,
        stage: Mapping[str, Any],
        output_root: Path,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], ...]:
        identity = _identity(stage)
        if identity not in self._admitted:
            raise TaskEvaluationSceneConfigurationStageProducerError(
                "scene_configuration_stage_producer_not_admitted:"
                f"{identity.adapter_id}:{identity.version}"
            )
        handler = self._handlers.get(identity)
        if handler is None:
            raise TaskEvaluationSceneConfigurationStageProducerError(
                "scene_configuration_stage_producer_not_installed:"
                f"{identity.adapter_id}:{identity.version}"
            )
        value = handler(stage=stage, output_root=output_root, **kwargs)
        if not isinstance(value, Mapping):
            raise TaskEvaluationSceneConfigurationStageProducerError(
                "scene_configuration_stage_production_result_invalid"
            )
        artifacts = value.get("artifacts")
        if (
            value.get("schema_version") != PRODUCTION_RESULT_SCHEMA_VERSION
            or value.get("status") != "completed"
            or value.get("stage_id") != stage.get("stage_id")
            or value.get("capability") != stage.get("capability")
            or value.get("provider_mutations_performed") != 0
            or value.get("paid_execution_requested") is not False
            or value.get("executed_inside_parent_configuration_run") is not True
            or value.get("production_result_digest")
            != canonical_digest(value, digest_field="production_result_digest")
            or not isinstance(artifacts, list)
            or not artifacts
        ):
            raise TaskEvaluationSceneConfigurationStageProducerError(
                "scene_configuration_stage_production_result_invalid"
            )
        root = output_root.resolve(strict=True)
        verified: list[dict[str, Any]] = []
        roles: set[str] = set()
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise TaskEvaluationSceneConfigurationStageProducerError(
                    "scene_configuration_stage_production_artifact_invalid"
                )
            role = str(artifact.get("role") or "")
            path = Path(str(artifact.get("path") or "")).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise TaskEvaluationSceneConfigurationStageProducerError(
                    "scene_configuration_stage_production_artifact_outside_root"
                ) from exc
            if (
                not role
                or role in roles
                or path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != artifact.get("size_bytes")
                or _sha256(path) != artifact.get("digest")
            ):
                raise TaskEvaluationSceneConfigurationStageProducerError(
                    "scene_configuration_stage_production_artifact_invalid"
                )
            roles.add(role)
            verified.append(dict(artifact))
        return tuple(verified)


__all__ = [
    "ADMITTED_PRODUCER_IDENTITIES",
    "PRODUCTION_RESULT_SCHEMA_VERSION",
    "SceneConfigurationStageProducerIdentity",
    "SceneConfigurationStageProducerRegistry",
    "StageProducer",
    "TaskEvaluationSceneConfigurationStageProducerError",
]
