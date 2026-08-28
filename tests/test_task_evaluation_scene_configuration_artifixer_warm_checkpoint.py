from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_artifixer_failure_evidence import (
    ARTIFIXER_RUNTIME_ACCEPTED_STATUS,
)
from blueprint_pipeline.task_evaluation_scene_configuration_artifixer_warm_checkpoint import (
    ArtifixerPostTrainingCheckpointError,
    artifixer_post_training_binding_digest,
    hydrate_artifixer_post_training_checkpoint,
    materialize_artifixer_post_training_checkpoint,
    validate_artifixer_post_training_checkpoint,
)


def _fixture(tmp_path: Path) -> dict:
    runtime = {
        "schema_version": "public_scene_artifixer3d_runtime_result.v1",
        "status": ARTIFIXER_RUNTIME_ACCEPTED_STATUS,
        "tasks": [{"task_id": "remove-source-object-104"}],
        "result_digest": "",
    }
    runtime["result_digest"] = canonical_digest(runtime, digest_field="result_digest")
    runtime_path = tmp_path / "runtime-result.json"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    frames = []
    for index in range(8):
        path = tmp_path / f"review-{index}.png"
        path.write_bytes(f"review-frame-{index}".encode())
        frames.append(
            {
                "frame_index": index,
                "camera_id": f"camera-{index}",
                "final_frame": {"path": str(path)},
            }
        )
    native = tmp_path / "configured.usdz"
    native.write_bytes(b"native-usdz")
    source = {
        "checkpoint_digest": "sha256:" + "1" * 64,
        "scientific_bindings": {"binding_digest": "sha256:" + "2" * 64},
        "completed_stage_prefix_count": 0,
        "completed_stage_results": [],
    }
    bindings = {
        "source_commit": "a" * 40,
        "run_id": "run-839873",
        "configuration_sha256": "sha256:" + "3" * 64,
        "dual_target_receipt_digest": "sha256:" + "4" * 64,
        "artifixer_bundle_manifest_digest": "sha256:" + "5" * 64,
        "artifixer_runtime_request_digest": "sha256:" + "6" * 64,
    }
    return {
        "runtime_path": runtime_path,
        "frames": frames,
        "native": native,
        "source": source,
        "bindings": bindings,
    }


def _materialize(tmp_path: Path) -> tuple[Path, dict, dict]:
    fixture = _fixture(tmp_path)
    root = tmp_path / "checkpoint"
    checkpoint = materialize_artifixer_post_training_checkpoint(
        source_diagnostic_checkpoint=fixture["source"],
        bindings=fixture["bindings"],
        runtime_result_path=fixture["runtime_path"],
        review_frames=fixture["frames"],
        native_appearance_path=fixture["native"],
        output_root=root,
    )
    return root, checkpoint, fixture


def test_post_training_checkpoint_seals_only_visual_review_inputs(
    tmp_path: Path,
) -> None:
    root, checkpoint, fixture = _materialize(tmp_path)
    reopened = validate_artifixer_post_training_checkpoint(
        checkpoint_root=root,
        expected_binding_digest=artifixer_post_training_binding_digest(
            fixture["bindings"]
        ),
        expected_source_checkpoint_digest=fixture["source"]["checkpoint_digest"],
    )
    hydrated = hydrate_artifixer_post_training_checkpoint(
        checkpoint_root=root,
        expected_binding_digest=checkpoint["binding_digest"],
    )

    assert reopened["rerun_paid_model_stages"] == ["artifixer_visual_review"]
    assert reopened["visual_review_provider_call_started"] is False
    assert len(reopened["inventory"]) == 10
    assert len(hydrated["review_frames"]) == 8
    assert Path(hydrated["native_appearance_path"]).read_bytes() == b"native-usdz"
    assert {path.name for path in root.rglob("*") if path.is_file()} == {
        "task_evaluation_scene_configuration_artifixer_post_training_checkpoint.v1.json",
        "runtime_result.json",
        "configured_appearance.usdz",
        *(f"{index:05d}.png" for index in range(8)),
    }


def test_post_training_checkpoint_refuses_changed_bytes(tmp_path: Path) -> None:
    root, _checkpoint, _fixture = _materialize(tmp_path)
    frame = root / "review/frames/00003.png"
    frame.chmod(0o640)
    frame.write_bytes(frame.read_bytes() + b"tampered")

    with pytest.raises(
        ArtifixerPostTrainingCheckpointError,
        match="scene_configuration_artifixer_warm_checkpoint_inventory_invalid",
    ):
        validate_artifixer_post_training_checkpoint(checkpoint_root=root)


def test_post_training_checkpoint_refuses_generic_completed_runtime_status(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runtime = json.loads(fixture["runtime_path"].read_text(encoding="utf-8"))
    runtime["status"] = "completed"
    runtime["result_digest"] = canonical_digest(runtime, digest_field="result_digest")
    fixture["runtime_path"].write_text(json.dumps(runtime), encoding="utf-8")

    with pytest.raises(
        ArtifixerPostTrainingCheckpointError,
        match="scene_configuration_artifixer_warm_runtime_result_invalid",
    ):
        materialize_artifixer_post_training_checkpoint(
            source_diagnostic_checkpoint=fixture["source"],
            bindings=fixture["bindings"],
            runtime_result_path=fixture["runtime_path"],
            review_frames=fixture["frames"],
            native_appearance_path=fixture["native"],
            output_root=tmp_path / "checkpoint",
        )


def test_post_training_checkpoint_refuses_secret_shaped_bindings(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["bindings"]["api_key"] = "sk-secret-must-not-survive"

    with pytest.raises(
        ArtifixerPostTrainingCheckpointError,
        match="scene_configuration_artifixer_warm_binding_invalid",
    ):
        materialize_artifixer_post_training_checkpoint(
            source_diagnostic_checkpoint=fixture["source"],
            bindings=fixture["bindings"],
            runtime_result_path=fixture["runtime_path"],
            review_frames=fixture["frames"],
            native_appearance_path=fixture["native"],
            output_root=tmp_path / "checkpoint",
        )
