from __future__ import annotations

import json

from blueprint_pipeline.backend_support_artifacts import (
    resolve_backend_support_artifacts,
)


def test_backend_support_registry_defaults_to_not_requested_without_execution(
    tmp_path,
) -> None:
    support = resolve_backend_support_artifacts(
        tmp_path,
        backend_id="cosmos_predict2_5",
    )

    assert set(support) == {
        "cosmos_zero_shot_benchmark",
        "cosmos_training_export",
        "cosmos_lora_training",
    }
    assert support["cosmos_training_export"].payload["status"] == "not_requested"
    assert support["cosmos_training_export"].payload["claim_boundary"] == {
        "evaluation_prep_executes_model_specific_exporters": False,
        "explicit_external_support_artifact_required": True,
    }
    assert not support["cosmos_training_export"].path.exists()


def test_backend_support_registry_reads_explicit_artifact(tmp_path) -> None:
    path = tmp_path / "cosmos_training_export" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status": "ready"}), encoding="utf-8")

    support = resolve_backend_support_artifacts(tmp_path)

    assert support["cosmos_training_export"].payload == {"status": "ready"}
    assert support["cosmos_training_export"].relative_path == (
        "cosmos_training_export/manifest.json"
    )
