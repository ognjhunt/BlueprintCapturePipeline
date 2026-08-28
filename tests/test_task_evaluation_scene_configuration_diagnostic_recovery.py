from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.task_evaluation_scene_configuration_adapters import (
    ADMITTED_STAGE_ADAPTER_IDENTITIES,
)
from blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_checkpoint import (
    hydrate_scene_configuration_diagnostic_completed_stages,
    materialize_scene_configuration_diagnostic_checkpoint,
    validate_scene_configuration_diagnostic_checkpoint,
)
from blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_recovery import (
    TaskEvaluationSceneConfigurationDiagnosticRecoveryError,
    recover_scene_configuration_diagnostic_stage_one_checkpoint,
)
from blueprint_pipeline.task_evaluation_scene_configuration_disclosure import (
    resolve_scene_configuration_disclosure,
)
from blueprint_pipeline.task_evaluation_scene_configuration_render_handoff import (
    validate_provider_render_handoff,
    materialize_provider_render_handoff,
)
from blueprint_pipeline.task_evaluation_scene_configuration_bundle import (
    _resolve_diagnostic_checkpoint_reference,
)
from tests.test_task_evaluation_scene_configuration_diagnostic_checkpoint import (
    _fixture,
)


def _write(path: Path, value: dict) -> Path:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(role: str, path: Path) -> dict:
    return {
        "role": role,
        "path": str(path),
        "digest": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _recovery_fixture(tmp_path: Path) -> dict[str, Path]:
    fixture = _fixture(tmp_path)
    source = Path(fixture["stage_path"]).parent
    configuration = {
        "schema_version": "observed_appearance_object_removal_configuration.v1",
        "source_object": {"publisher_instance_id": "104"},
        "production_render_required": True,
        "required_views": {"minimum": 8},
        "provider_disclosure": {
            "raw_interiorgs_bytes": True,
            "provider_training": False,
            "public_redistribution": False,
        },
        "human_authority": {
            "provider_retention_terms_accepted": True,
            "provider_training_authorized": False,
            "authority_reference": "human-authority-1",
        },
        "output_requirements": {"generated_pixels_labeled": True},
    }
    configuration_path = _write(source / "configuration.json", configuration)
    stages = [
        {
            "stage_id": f"stage-{index}",
            "capability": identity.capability,
            "adapter": {"id": identity.adapter_id, "version": identity.version},
            "execution_class": identity.execution_class,
            "depends_on": [] if index == 1 else [f"stage-{index - 1}"],
        }
        for index, identity in enumerate(
            ADMITTED_STAGE_ADAPTER_IDENTITIES, start=1
        )
    ]
    render = dict(fixture["render_result"])
    decision = resolve_scene_configuration_disclosure(
        stage_one_configuration=configuration,
        rights_admission={
            "provider_disclosure": {
                "raw_interiorgs_downloaded_bytes_may_be_uploaded": True,
                "provider_training_allowed": False,
                "public_redistribution_allowed": False,
                "provider_retention_rule": "ephemeral-provider-run-only",
            }
        },
    )
    render.update(
        {
            "disclosure_decision": decision,
            "control_plane_result_digest": "sha256:" + "c" * 64,
            "render_completed_on_provider": True,
            "result_digest": "",
        }
    )
    render["result_digest"] = canonical_digest(render, digest_field="result_digest")
    _write(Path(fixture["render_path"]), render)
    stage_input = dict(fixture["stage_input"])
    envelope = dict(stage_input["construction_envelope"])
    envelope["recipe"] = {
        **dict(envelope["recipe"]),
        "stage_sequence": stages,
    }
    envelope["stage_configuration_references"] = [
        {
            "stage_id": stage["stage_id"],
            "materialized_path": str(configuration_path),
            "digest": sha256(configuration_path),
            "size_bytes": configuration_path.stat().st_size,
        }
        for stage in stages
    ]
    envelope["render_inputs_result"] = render
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    stage_input.update(
        {
            "stage": stages[0],
            "configuration": configuration,
            "configuration_sha256": sha256(configuration_path),
            "execution_mode": "diagnostic_only",
            "construction_envelope": envelope,
        }
    )
    _write(Path(fixture["stage_path"]), stage_input)
    source_checkpoint = tmp_path / "source-checkpoint"
    materialize_scene_configuration_diagnostic_checkpoint(
        stage_production_input_path=fixture["stage_path"],
        render_inputs_result_path=fixture["render_path"],
        semantic_runtime_request_path=fixture["request_path"],
        semantic_runtime_result_path=fixture["result_path"],
        semantic_teacher_receipt_path=fixture["receipt_path"],
        output_root=source_checkpoint,
    )
    # The retained producer receipt came back from the provider, so its stage
    # input carries provider-local materialized paths. Recovery must rebind only
    # Stage 1 to the separately digest-verified control-plane configuration.
    provider_stage_input = json.loads(json.dumps(stage_input))
    provider_envelope = provider_stage_input["construction_envelope"]
    for index, row in enumerate(
        provider_envelope["stage_configuration_references"], start=1
    ):
        row["materialized_path"] = (
            "/workspace/task_evaluation_scene_configuration_provider_bundle/"
            f"provider_runtime/input/configurations/stage-{index}.json"
        )
    provider_envelope["envelope_digest"] = canonical_digest(
        provider_envelope, digest_field="envelope_digest"
    )
    _write(Path(fixture["stage_path"]), provider_stage_input)
    stage_input = provider_stage_input

    producer = tmp_path / "producer"
    producer.mkdir()
    appearance = producer / "configured_appearance_without_source_object.usdz"
    appearance.write_bytes(b"qualified-generated-appearance")
    thumbnail = producer / "configured_task_thumbnail.png"
    thumbnail.write_bytes(b"exact-selected-review-frame")
    review = {
        "schema_version": "task_evaluation_artifixer_ai_visual_review.v1",
        "status": "accepted",
        "publisher_instance_id": "104",
        "decision": "accepted",
        "semantic_object_absence_review_passed": True,
        "multiview_consistency_review_passed": True,
        "review_frame_count": 8,
        "all_review_frames_digest_bound": True,
        "ai_visual_review_completed": True,
        "human_review_completed": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "task_thumbnail_is_exact_review_frame": True,
        "task_thumbnail_selection": {
            "camera_id": "camera-3",
            "frame_sha256": sha256(thumbnail),
        },
        "reviewer": {
            "identity": "independent-vlm-reviewer-v1",
            "runtime": "openai_agents_sdk",
            "model": "gpt-5.6-terra",
        },
        "receipt_digest": "",
    }
    review["receipt_digest"] = canonical_digest(
        review, digest_field="receipt_digest"
    )
    review_path = _write(producer / "appearance_visual_review_receipt.v1.json", review)
    receipt = {
        "schema_version": "task_evaluation_artifixer_object_removal_result.v1",
        "status": "qualified_generated_appearance_edit",
        "publisher_instance_id": "104",
        "raw_interiorgs_bytes_sent_to_external_provider": False,
        "visual_review_receipt_digest": review["receipt_digest"],
        "visual_review_receipt_sha256": sha256(review_path),
        "semantic_object_free_visual_review_passed": True,
        "multiview_consistency_review_passed": True,
        "generated_pixels_labeled": True,
        "result_digest": "",
    }
    receipt["result_digest"] = canonical_digest(
        receipt, digest_field="result_digest"
    )
    receipt_path = _write(producer / "appearance_removal_receipt.v1.json", receipt)
    handoff = materialize_provider_render_handoff(
        render_inputs=render, output_root=producer
    )
    rows = [
        artifact("configured_appearance_without_source_object", appearance),
        artifact("appearance_removal_receipt", receipt_path),
        artifact("appearance_visual_review_receipt", review_path),
        artifact("configured_task_thumbnail", thumbnail),
        handoff,
    ]
    for row in rows:
        row["path"] = (
            "/workspace/task_evaluation_scene_configuration_provider_bundle/"
            f"runtime_output/stages/stage-1/producer/{Path(row['path']).name}"
        )
    production = {
        "schema_version": "task_evaluation_scene_configuration_stage_production.v1",
        "status": "completed",
        "stage_id": "stage-1",
        "capability": "observed_appearance_object_removal",
        "adapter_id": "artifixer3d_observed_object_removal",
        "source_commit": stage_input["source_commit"],
        "toolchain_digest": stage_input["toolchain_digest"],
        "provider_mutations_performed": 0,
        "paid_execution_requested": False,
        "executed_inside_parent_configuration_run": True,
        "artifacts": rows,
        "production_result_digest": "",
    }
    production["production_result_digest"] = canonical_digest(
        production, digest_field="production_result_digest"
    )
    production_path = _write(
        producer / "task_evaluation_scene_configuration_stage_production.v1.json",
        production,
    )
    return {
        "source_checkpoint": source_checkpoint,
        "stage_input": Path(fixture["stage_path"]),
        "configuration": configuration_path,
        "producer": producer,
        "production": production_path,
    }


def _recover(tmp_path: Path, fixture: dict[str, Path]) -> dict:
    return recover_scene_configuration_diagnostic_stage_one_checkpoint(
        source_checkpoint_root=fixture["source_checkpoint"],
        stage_production_input_path=fixture["stage_input"],
        stage_production_result_path=fixture["production"],
        stage_configuration_path=fixture["configuration"],
        producer_root=fixture["producer"],
        adapter_output_root=tmp_path / "adapter-output",
        checkpoint_output_root=tmp_path / "stage-one-checkpoint",
        reference_output_path=tmp_path / "checkpoint-reference.v1.json",
    )


def test_recovery_advances_exact_stage_one_and_carries_render_handoff(
    tmp_path: Path,
) -> None:
    fixture = _recovery_fixture(tmp_path)
    reference = _recover(tmp_path, fixture)
    checkpoint_root = Path(reference["checkpoint_root"])
    checkpoint = validate_scene_configuration_diagnostic_checkpoint(
        checkpoint_root=checkpoint_root
    )
    stage_input = json.loads(fixture["stage_input"].read_text(encoding="utf-8"))
    carried = hydrate_scene_configuration_diagnostic_completed_stages(
        checkpoint_root=checkpoint_root,
        stage_sequence=stage_input["construction_envelope"]["recipe"][
            "stage_sequence"
        ],
        configurations={
            "stage-1": (stage_input["configuration"], fixture["configuration"])
        },
    )
    handoff = next(
        row
        for row in carried[0]["output_artifacts"]
        if row["role"] == "provider_render_reference_manifest"
    )
    manifest, frames = validate_provider_render_handoff(handoff["path"])

    assert checkpoint["completed_stage_prefix_count"] == 1
    assert len(checkpoint["completed_stage_results"]) == 1
    assert manifest["frame_count"] == 8
    assert len(frames) == 8
    assert reference["diagnostic_only"] is True
    assert reference["qualification_eligible"] is False
    assert reference["reference_digest"] == canonical_digest(
        reference, digest_field="reference_digest"
    )
    assert _resolve_diagnostic_checkpoint_reference(
        tmp_path / "checkpoint-reference.v1.json"
    ) == checkpoint_root


def test_recovery_refuses_changed_provider_artifact(tmp_path: Path) -> None:
    fixture = _recovery_fixture(tmp_path)
    (fixture["producer"] / "configured_task_thumbnail.png").write_bytes(b"changed")

    with pytest.raises(
        TaskEvaluationSceneConfigurationDiagnosticRecoveryError,
        match="scene_configuration_diagnostic_stage_one_artifacts_invalid",
    ):
        _recover(tmp_path, fixture)
