from __future__ import annotations

import json
import hashlib
import zipfile
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import task_evaluation_scene_configuration_provider_cleanup as provider_cleanup
from blueprint_pipeline import task_evaluation_scene_configuration_vast as vast
from blueprint_pipeline.task_evaluation_scene_configuration_artifixer_failure_evidence import (
    ARTIFIXER_RUNTIME_ACCEPTED_STATUS,
)
from blueprint_pipeline.task_evaluation_scene_configuration_artifixer_warm_checkpoint import (
    SCHEMA_VERSION as ARTIFIXER_CHECKPOINT_SCHEMA_VERSION,
    materialize_artifixer_post_training_checkpoint,
    validate_artifixer_post_training_checkpoint,
)


CHECKPOINT_BODY = b'{}\n'
CHECKPOINT_DIGEST = "sha256:" + "a" * 64


def test_terminal_result_explicitly_records_that_no_raw_secrets_were_retained(
    tmp_path: Path,
) -> None:
    result = vast._seal_terminal_result(
        tmp_path,
        {
            "schema_version": vast.RESULT_SCHEMA_VERSION,
            "status": "refused",
            "blockers": ["test_refusal"],
        },
    )

    assert result["raw_secret_values_recorded"] is False
    assert result["result_digest"] == canonical_digest(
        result, digest_field="result_digest"
    )


def test_live_diagnostic_terminal_result_does_not_finalize_production_queue(
    tmp_path: Path, monkeypatch
) -> None:
    def unexpected_finalization(**_kwargs):
        raise AssertionError("diagnostic must not mutate the production queue")

    monkeypatch.setattr(vast, "finalize_scene_construction", unexpected_finalization)

    result = vast._seal_live_terminal_result(
        tmp_path,
        {
            "schema_version": vast.DIAGNOSTIC_RESULT_SCHEMA_VERSION,
            "status": "blocked_diagnostic_only",
            "blockers": ["diagnostic_refusal"],
        },
        receipt={},
        scene_construction_queue_root=None,
        diagnostic_only=True,
    )

    assert result["status"] == "blocked_diagnostic_only"
    assert result["blockers"] == ["diagnostic_refusal"]
    assert "scene_construction_queue_finalization" not in result


def _diagnostic_provider_result() -> dict:
    stage_results = []
    for index in range(1, 7):
        stage = {
            "schema_version": "task_evaluation_scene_configuration_stage_result.v1",
            "status": "completed",
            "stage_id": f"stage-{index}",
            "canonical_allocator": None,
            "provider_mutations_performed": 0,
            "paid_execution_requested": False,
            "executed_inside_parent_configuration_run": True,
            "raw_secret_values_recorded": False,
            "output_artifacts": [],
            "diagnostic_only": True,
            "qualification_eligible": False,
            "executed_inside_one_parent_provider_run": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
            "stage_result_digest": "",
        }
        stage["stage_result_digest"] = canonical_digest(
            stage, digest_field="stage_result_digest"
        )
        stage_results.append(stage)
    chain = {
        "schema_version": (
            "task_evaluation_scene_configuration_diagnostic_stage_chain.v1"
        ),
        "status": "completed_diagnostic_only_not_qualification_eligible",
        "run_id": "diagnostic-run-1",
        "stage_count": 6,
        "stage_results": stage_results,
        "stage_result_digests": [
            stage["stage_result_digest"] for stage in stage_results
        ],
        "executed_inside_one_parent_provider_run": False,
        "nested_provider_mutations_performed": 0,
        "nested_paid_execution_requested": False,
        "evaluation_episode_executed": False,
        "retry_cap": 0,
        "result_digest": "",
    }
    chain["result_digest"] = canonical_digest(chain, digest_field="result_digest")
    result = {
        "schema_version": (
            "task_evaluation_scene_configuration_diagnostic_provider_result.v1"
        ),
        "status": "completed_diagnostic_only_not_qualification_eligible",
        "diagnostic_stage_chain": chain,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "executed_inside_one_parent_provider_run": False,
        "configured_revision_publication_permitted": False,
        "offering_publication_permitted": False,
        "terminal_e2e_completion_permitted": False,
        "provider_zero_required_after_return": True,
        "raw_secret_values_recorded": False,
        "blockers": [],
        "advanced_checkpoint": {
            "provider_output_relative_root": "diagnostic_checkpoints/after-stage-6",
            "manifest_relative_path": (
                "diagnostic_checkpoints/after-stage-6/"
                "task_evaluation_scene_configuration_diagnostic_checkpoint.v1.json"
            ),
            "manifest_sha256": "sha256:"
            + hashlib.sha256(CHECKPOINT_BODY).hexdigest(),
            "checkpoint_digest": CHECKPOINT_DIGEST,
            "completed_stage_prefix_count": 6,
            "file_count": 1,
            "total_bytes": len(CHECKPOINT_BODY),
        },
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def test_diagnostic_provider_output_is_accepted_only_by_diagnostic_extractor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        vast,
        "validate_scene_configuration_diagnostic_checkpoint",
        lambda **_kwargs: {
            "checkpoint_digest": CHECKPOINT_DIGEST,
            "completed_stage_prefix_count": 6,
        },
    )
    archive = tmp_path / "output.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "task_evaluation_scene_configuration_provider_result.v1.json",
            json.dumps(_diagnostic_provider_result(), sort_keys=True),
        )
        bundle.writestr(
            "diagnostic_checkpoints/after-stage-6/"
            "task_evaluation_scene_configuration_diagnostic_checkpoint.v1.json",
            CHECKPOINT_BODY,
        )

    result, blockers = vast._extract_provider_output(
        archive,
        tmp_path / "diagnostic",
        maximum_archive_bytes=1_000_000,
        diagnostic_only=True,
    )
    assert blockers == []
    assert result["diagnostic_only"] is True

    _result, production_blockers = vast._extract_provider_output(
        archive,
        tmp_path / "production",
        maximum_archive_bytes=1_000_000,
    )
    assert "scene_configuration_provider_result_contract_invalid" in (
        production_blockers
    )


def test_diagnostic_provider_output_refuses_any_publication_permission(
    tmp_path: Path,
) -> None:
    result = _diagnostic_provider_result()
    result["offering_publication_permitted"] = True
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    archive = tmp_path / "output.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "task_evaluation_scene_configuration_provider_result.v1.json",
            json.dumps(result, sort_keys=True),
        )

    _value, blockers = vast._extract_provider_output(
        archive,
        tmp_path / "diagnostic",
        maximum_archive_bytes=1_000_000,
        diagnostic_only=True,
    )
    assert "scene_configuration_diagnostic_claim_boundary_invalid" in blockers


def test_blocked_diagnostic_output_preserves_validated_advanced_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        vast,
        "validate_scene_configuration_diagnostic_checkpoint",
        lambda **_kwargs: {
            "checkpoint_digest": CHECKPOINT_DIGEST,
            "completed_stage_prefix_count": 6,
        },
    )
    result = _diagnostic_provider_result()
    result["status"] = "blocked_diagnostic_only"
    result["blockers"] = ["stage_4_refused_after_stage_3_checkpoint"]
    result.pop("diagnostic_stage_chain")
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    archive = tmp_path / "blocked-output.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "task_evaluation_scene_configuration_provider_result.v1.json",
            json.dumps(result, sort_keys=True),
        )
        bundle.writestr(
            "diagnostic_checkpoints/after-stage-6/"
            "task_evaluation_scene_configuration_diagnostic_checkpoint.v1.json",
            CHECKPOINT_BODY,
        )

    observed, blockers = vast._extract_provider_output(
        archive,
        tmp_path / "blocked-diagnostic",
        maximum_archive_bytes=1_000_000,
        diagnostic_only=True,
    )

    assert "provider_result_blocker:stage_4_refused_after_stage_3_checkpoint" in blockers
    assert observed["_validated_advanced_checkpoint"]["checkpoint_digest"] == CHECKPOINT_DIGEST


def test_blocked_diagnostic_output_restores_authenticated_artifixer_checkpoint_modes(
    tmp_path: Path,
) -> None:
    runtime_result = tmp_path / "runtime-result.json"
    runtime_result.write_text(
        json.dumps(
            {
                "schema_version": "public_scene_artifixer3d_runtime_result.v1",
                "status": ARTIFIXER_RUNTIME_ACCEPTED_STATUS,
            }
        ),
        encoding="utf-8",
    )
    review_frames = []
    for index in range(8):
        frame = tmp_path / f"review-{index}.png"
        frame.write_bytes(f"review-frame-{index}".encode())
        review_frames.append(
            {
                "frame_index": index,
                "camera_id": f"camera-{index}",
                "final_frame": {"path": str(frame)},
            }
        )
    native = tmp_path / "configured.usdz"
    native.write_bytes(b"configured-appearance")
    checkpoint_root = tmp_path / "provider-checkpoint"
    checkpoint = materialize_artifixer_post_training_checkpoint(
        source_diagnostic_checkpoint={
            "checkpoint_digest": "sha256:" + "1" * 64,
            "scientific_bindings": {"binding_digest": "sha256:" + "2" * 64},
            "completed_stage_prefix_count": 0,
            "completed_stage_results": [],
        },
        bindings={
            "source_commit": "a" * 40,
            "run_id": "diagnostic-run-1",
            "configuration_sha256": "sha256:" + "3" * 64,
        },
        runtime_result_path=runtime_result,
        review_frames=review_frames,
        native_appearance_path=native,
        output_root=checkpoint_root,
    )
    relative_root = Path("stages/stage-1/producer/artifixer_post_training_checkpoint")
    manifest = checkpoint_root / f"{ARTIFIXER_CHECKPOINT_SCHEMA_VERSION}.json"
    result = _diagnostic_provider_result()
    result["status"] = "blocked_diagnostic_only"
    result["blockers"] = ["stage_2_refused_after_artifixer"]
    result.pop("diagnostic_stage_chain")
    result.pop("advanced_checkpoint")
    result["artifixer_post_training_checkpoint"] = {
        "provider_output_relative_root": relative_root.as_posix(),
        "manifest_relative_path": (relative_root / manifest.name).as_posix(),
        "manifest_sha256": "sha256:"
        + hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "checkpoint_digest": checkpoint["checkpoint_digest"],
        "source_diagnostic_checkpoint_digest": checkpoint[
            "source_diagnostic_checkpoint_digest"
        ],
        "binding_digest": checkpoint["binding_digest"],
        "file_count": 1 + len(checkpoint["inventory"]),
        "total_bytes": sum(
            path.stat().st_size for path in checkpoint_root.rglob("*") if path.is_file()
        ),
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    archive = tmp_path / "blocked-artifixer-output.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(vast.RESULT_FILENAME, json.dumps(result, sort_keys=True))
        for path in checkpoint_root.rglob("*"):
            if path.is_file():
                bundle.write(
                    path,
                    (relative_root / path.relative_to(checkpoint_root)).as_posix(),
                )

    observed, blockers = vast._extract_provider_output(
        archive,
        tmp_path / "extracted",
        maximum_archive_bytes=1_000_000,
        diagnostic_only=True,
    )

    assert blockers == [
        "provider_result_blocker:stage_2_refused_after_artifixer"
    ]
    reference = observed["_validated_artifixer_post_training_checkpoint"]
    extracted_root = Path(reference["checkpoint_root"])
    reopened = validate_artifixer_post_training_checkpoint(
        checkpoint_root=extracted_root
    )
    assert reopened["checkpoint_digest"] == checkpoint["checkpoint_digest"]
    assert {
        path.stat().st_mode & 0o777
        for path in extracted_root.rglob("*")
        if path.is_file() and path.name != manifest.name
    } == {0o440}


def test_escaped_adapter_finally_defers_cleanup_when_allocation_may_exist(
    tmp_path: Path,
) -> None:
    provider_run = tmp_path / "provider-run"
    provider_run.mkdir()
    started = tmp_path / "started-instance-id"
    started.write_text("456\n")

    adapter, may_have_allocated = vast._recover_escaped_adapter_failure(
        provider_run=provider_run,
        started_instance_id_path=started,
        failure_detail="fresh_ssh_probe_failed",
    )
    cleanup_called = False

    def cleanup(_path: Path) -> dict:
        nonlocal cleanup_called
        cleanup_called = True
        return {"status": "completed", "all_objects_absent": True}

    cleanup_result = provider_cleanup.cleanup_scene_staging(
        adapter=adapter,
        staging_dir=tmp_path / "staging",
        cleanup=cleanup,
    )

    assert may_have_allocated is True
    assert adapter["vast_instance_ids"] == [456]
    assert adapter["continuing_spend_from_this_run"] is True
    assert cleanup_result["status"] == "deferred_until_provider_absent"
    assert cleanup_called is False
