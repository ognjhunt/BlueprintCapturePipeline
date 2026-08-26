from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_stage_producers import (
    ADMITTED_PRODUCER_IDENTITIES,
    PRODUCTION_RESULT_SCHEMA_VERSION,
    SceneConfigurationStageProducerRegistry,
    TaskEvaluationSceneConfigurationStageProducerError,
)


def _registry(*, escape: bool = False) -> SceneConfigurationStageProducerRegistry:
    identity = ADMITTED_PRODUCER_IDENTITIES[0]

    def produce(*, stage, output_root, **_kwargs):
        artifact = (output_root.parent.parent / "escaped.bin") if escape else output_root / "result.bin"
        artifact.write_bytes(b"generated-after-website-intake")
        result = {
            "schema_version": PRODUCTION_RESULT_SCHEMA_VERSION,
            "status": "completed",
            "stage_id": stage["stage_id"],
            "capability": stage["capability"],
            "provider_mutations_performed": 0,
            "paid_execution_requested": False,
            "executed_inside_parent_configuration_run": True,
            "artifacts": [
                {
                    "role": "generated_candidate",
                    "path": str(artifact),
                    "digest": "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "size_bytes": artifact.stat().st_size,
                }
            ],
            "production_result_digest": "",
        }
        result["production_result_digest"] = canonical_digest(
            result, digest_field="production_result_digest"
        )
        return result

    return SceneConfigurationStageProducerRegistry({identity: produce})


def _stage() -> dict:
    identity = ADMITTED_PRODUCER_IDENTITIES[0]
    return {
        "stage_id": "stage-1",
        "capability": identity.capability,
        "adapter": {"id": identity.adapter_id, "version": identity.version},
        "execution_class": "gpu_canary",
    }


def test_closed_producer_generates_and_rehashes_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "producer"
    output.mkdir()
    artifacts = _registry().execute(stage=_stage(), output_root=output)
    assert len(artifacts) == 1
    assert artifacts[0]["role"] == "generated_candidate"


def test_producer_cannot_return_operator_injected_path(tmp_path: Path) -> None:
    output = tmp_path / "producer"
    output.mkdir()
    with pytest.raises(
        TaskEvaluationSceneConfigurationStageProducerError,
        match="scene_configuration_stage_production_artifact_outside_root",
    ):
        _registry(escape=True).execute(stage=_stage(), output_root=output)
