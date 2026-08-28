from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_checkpoint import (
    SCHEMA_VERSION,
    TaskEvaluationSceneConfigurationDiagnosticCheckpointError,
    advance_scene_configuration_diagnostic_checkpoint,
    diagnostic_checkpoint_scientific_binding_digest,
    hydrate_scene_configuration_diagnostic_completed_stages,
    hydrate_scene_configuration_diagnostic_render_inputs,
    hydrate_scene_configuration_diagnostic_semantic_outputs,
    materialize_scene_configuration_diagnostic_checkpoint,
    validate_scene_configuration_diagnostic_checkpoint,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return path


def _bound(path: Path, *, digest_key: str = "digest") -> dict:
    return {
        "path": str(path),
        digest_key: _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _fixture(tmp_path: Path) -> dict[str, Path | dict]:
    source = tmp_path / "source"
    source.mkdir()
    camera_ids = [f"camera-{index}" for index in range(8)]
    calibration_rows = [
        {
            "id": camera_id,
            "spec": {
                "pose": {
                    "T_world_camera_opencv": [
                        [1.0, 0.0, 0.0, float(index)],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                },
                "intrinsics": {
                    "w": 8,
                    "h": 8,
                    "fl_x": 6.0,
                    "fl_y": 6.0,
                    "cx": 4.0,
                    "cy": 4.0,
                },
            },
        }
        for index, camera_id in enumerate(camera_ids)
    ]
    calibration = source / "calibration.json"
    calibration.write_text(canonical_json(calibration_rows) + "\n", encoding="utf-8")
    render_manifest = _write(
        source / "render_manifest.json",
        {"schema_version": "sealed_camera_render_manifest.v1", "camera_ids": camera_ids},
    )
    retained = source / "retained.ply"
    retained.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 1\nend_header\n")
    frames = []
    teacher_frames = []
    request_frames = []
    result_frames = []
    for index, camera_id in enumerate(camera_ids):
        frame = source / f"frame-{index}.png"
        mask = source / f"mask-{index}.png"
        edit = source / f"edit-{index}.png"
        Image.new("RGB", (8, 8), color=(index, 20, 30)).save(frame)
        Image.new("L", (8, 8), color=255).save(mask)
        Image.new("RGB", (8, 8), color=(40, index, 60)).save(edit)
        frames.append(
            {
                "camera_id": camera_id,
                **_bound(frame),
                "source_object_mask": _bound(mask),
            }
        )
        request_frames.append(
            {
                "frame_index": index,
                "camera_id": camera_id,
                "input_rgb": _bound(frame, digest_key="sha256"),
                "edit_mask": _bound(mask, digest_key="sha256"),
            }
        )
        result_frames.append(
            {
                "frame_index": index,
                "camera_id": camera_id,
                "terminal_state": "completed_unreviewed_candidate",
                "semantic_teacher_frame": _bound(edit, digest_key="sha256"),
            }
        )
        teacher_frames.append(
            {
                "frame_index": index,
                "camera_id": camera_id,
                "source_original_frame": _bound(frame, digest_key="sha256"),
                "exact_repair_mask": _bound(mask, digest_key="sha256"),
                "whole_frame_semantic_teacher": _bound(edit, digest_key="sha256"),
            }
        )
    disclosure = {
        "render_execution_site": "provider",
        "source_appearance_bytes_disclosed": True,
        "derived_rendered_views_disclosed": True,
    }
    renderer_runtime = {
        "runtime_kind": "bundled_splat_renderer",
        "renderer_digest": "sha256:" + "8" * 64,
        "browser_digest": "sha256:" + "9" * 64,
    }
    render_result = {
        "schema_version": "task_evaluation_scene_configuration_render_inputs.v1",
        "status": "derived_method_inputs_materialized",
        "run_id": "run-1",
        "source_splat_digest": "sha256:" + "1" * 64,
        "source_appearance": {
            "path": "input/source.ply",
            "digest": "sha256:" + "1" * 64,
            "size_bytes": 123456,
        },
        "disclosure_decision": disclosure,
        "renderer_runtime": renderer_runtime,
        "camera_calibration": _bound(calibration),
        "render_manifest": _bound(render_manifest),
        "derived_frames": frames,
        "derived_frame_count": 8,
        "derived_gaussian_cutout": {
            "retained_count": 100,
            "retained_scene_without_source_object": _bound(retained),
        },
        "render_completed_on_provider": True,
        "result_digest": "",
    }
    render_result["result_digest"] = canonical_digest(
        render_result, digest_field="result_digest"
    )
    render_result_path = _write(source / "render_result.json", render_result)

    configuration = {
        "schema_version": "observed_appearance_object_removal_configuration.v1",
        "source_object": {"publisher_instance_id": "104"},
        "random_seed": 839873,
    }
    # Production configuration files retain publisher formatting.  The exact
    # byte digest therefore need not equal a digest of canonical reserialization.
    configuration_path = source / "configuration.json"
    configuration_path.write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    recipe_digest = "sha256:" + "2" * 64
    envelope = {
        "run_id": "run-1",
        "expected_production_commit": "a" * 40,
        "portable_envelope_digest": "sha256:" + "3" * 64,
        "control_plane_envelope_digest": "sha256:" + "4" * 64,
        "recipe_digest": recipe_digest,
        "recipe": {"recipe_digest": recipe_digest},
        "stage_configuration_references": [
            {
                "stage_id": "stage-1",
                "materialized_path": str(configuration_path),
                "digest": _sha256(configuration_path),
                "size_bytes": configuration_path.stat().st_size,
            }
        ],
        "envelope_digest": "",
    }
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    stage_input = {
        "schema_version": "task_evaluation_scene_configuration_stage_production_input.v1",
        "run_id": "run-1",
        "stage": {"stage_id": "stage-1"},
        "configuration": configuration,
        "configuration_sha256": _sha256(configuration_path),
        "source_commit": "a" * 40,
        "toolchain_digest": "sha256:" + "5" * 64,
        "construction_envelope": envelope,
    }
    stage_path = _write(source / "stage_input.json", stage_input)
    semantic_request = {
        "schema_version": "semantic_teacher_image_edit_runtime_request.v1",
        "source_commit_sha": "a" * 40,
        "source_packet_digest": "sha256:" + "6" * 64,
        "backend": {
            "registry_entry": {"backend_id": "openai-gpt-image-2"},
            "backend_entry_digest": "sha256:" + "7" * 64,
            "execution": {"model_snapshot": "gpt-image-2-2026-04-21"},
        },
        "prompt_policy": "remove bounded source object",
        "prompt": "Remove the object.",
        "tasks": [{"task_id": "remove-source-object-104", "frames": request_frames}],
        "max_parallel_requests": 2,
        "maximum_cost_usd": 2.0,
        "expected_request_cost_usd": 0.22,
        "retry_count": 0,
        "request_digest": "",
    }
    semantic_request["request_digest"] = canonical_digest(
        semantic_request, digest_field="request_digest"
    )
    request_path = _write(source / "semantic_request.json", semantic_request)
    semantic_result = {
        "schema_version": "semantic_teacher_image_edit_runtime_result.v1",
        "status": "completed_unreviewed_semantic_teacher_candidates",
        "source_runtime_request_digest": semantic_request["request_digest"],
        "backend_id": "openai-gpt-image-2",
        "backend_entry_digest": "sha256:" + "7" * 64,
        "adapter_id": "openai_images_edits_v1",
        "model_snapshot": "gpt-image-2-2026-04-21",
        "request_count": 8,
        "successful_request_count": 8,
        "failed_request_count": 0,
        "tasks": [{"task_id": "remove-source-object-104", "frames": result_frames}],
        "raw_secret_values_recorded": False,
        "result_digest": "",
    }
    semantic_result["result_digest"] = canonical_digest(
        semantic_result, digest_field="result_digest"
    )
    result_path = _write(source / "semantic_result.json", semantic_result)
    teacher_receipt = {
        "schema_version": "public_scene_whole_frame_semantic_teacher_candidates.v1",
        "status": "whole_frame_semantic_teacher_candidates_unreviewed",
        "task_id": "remove-source-object-104",
        "editor_identity": {
            "backend_id": semantic_result["backend_id"],
            "model_snapshot": semantic_result["model_snapshot"],
            "result_digest": semantic_result["result_digest"],
        },
        "frame_count": 8,
        "frames": teacher_frames,
        "receipt_digest": "",
    }
    teacher_receipt["receipt_digest"] = canonical_digest(
        teacher_receipt, digest_field="receipt_digest"
    )
    receipt_path = _write(source / "teacher_receipt.json", teacher_receipt)
    return {
        "stage_input": stage_input,
        "stage_path": stage_path,
        "render_result": render_result,
        "render_path": render_result_path,
        "request_path": request_path,
        "result_path": result_path,
        "receipt_path": receipt_path,
    }


def _materialize(tmp_path: Path) -> tuple[Path, dict, dict[str, Path | dict]]:
    fixture = _fixture(tmp_path)
    root = tmp_path / "checkpoint"
    result = materialize_scene_configuration_diagnostic_checkpoint(
        stage_production_input_path=fixture["stage_path"],
        render_inputs_result_path=fixture["render_path"],
        semantic_runtime_request_path=fixture["request_path"],
        semantic_runtime_result_path=fixture["result_path"],
        semantic_teacher_receipt_path=fixture["receipt_path"],
        output_root=root,
    )
    return root, result, fixture


def test_checkpoint_seals_exact_eight_frame_prefix_as_diagnostic_only(
    tmp_path: Path,
) -> None:
    root, result, _fixture_rows = _materialize(tmp_path)
    reopened = validate_scene_configuration_diagnostic_checkpoint(
        checkpoint_root=root,
        expected_scientific_binding_digest=result["scientific_bindings"][
            "binding_digest"
        ],
    )

    assert reopened["schema_version"] == SCHEMA_VERSION
    assert reopened["diagnostic_only"] is True
    assert reopened["qualification_eligible"] is False
    assert reopened["executed_inside_one_parent_provider_run"] is False
    assert reopened["configured_revision_publication_permitted"] is False
    assert reopened["offering_publication_permitted"] is False
    assert reopened["terminal_e2e_completion_permitted"] is False
    assert reopened["camera_count"] == 8
    assert reopened["semantic_teacher"]["completed_frame_count"] == 8
    assert len(reopened["inventory"]) == 30


def test_checkpoint_refuses_partial_semantic_prefix(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result_path = fixture["result_path"]
    value = json.loads(result_path.read_text(encoding="utf-8"))
    value["successful_request_count"] = 7
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    _write(result_path, value)

    with pytest.raises(
        TaskEvaluationSceneConfigurationDiagnosticCheckpointError,
        match="scene_configuration_diagnostic_checkpoint_prefix_incomplete",
    ):
        materialize_scene_configuration_diagnostic_checkpoint(
            stage_production_input_path=fixture["stage_path"],
            render_inputs_result_path=fixture["render_path"],
            semantic_runtime_request_path=fixture["request_path"],
            semantic_runtime_result_path=result_path,
            semantic_teacher_receipt_path=fixture["receipt_path"],
            output_root=tmp_path / "checkpoint",
        )


def test_checkpoint_refuses_secret_material_in_retained_inputs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    request_path = fixture["request_path"]
    value = json.loads(request_path.read_text(encoding="utf-8"))
    value["openai_api_key"] = "sk-secret-must-not-be-retained"
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    _write(request_path, value)

    with pytest.raises(
        TaskEvaluationSceneConfigurationDiagnosticCheckpointError,
        match="scene_configuration_diagnostic_checkpoint_secret_material_forbidden",
    ):
        materialize_scene_configuration_diagnostic_checkpoint(
            stage_production_input_path=fixture["stage_path"],
            render_inputs_result_path=fixture["render_path"],
            semantic_runtime_request_path=request_path,
            semantic_runtime_result_path=fixture["result_path"],
            semantic_teacher_receipt_path=fixture["receipt_path"],
            output_root=tmp_path / "checkpoint",
        )


def test_checkpoint_reopen_refuses_changed_frame_bytes(tmp_path: Path) -> None:
    root, _result, _fixture_rows = _materialize(tmp_path)
    frame = root / "render/frames/00000.png"
    frame.write_bytes(frame.read_bytes() + b"tampered")

    with pytest.raises(
        TaskEvaluationSceneConfigurationDiagnosticCheckpointError,
        match="scene_configuration_diagnostic_checkpoint_inventory_invalid",
    ):
        validate_scene_configuration_diagnostic_checkpoint(checkpoint_root=root)


def test_cross_commit_reuse_key_depends_on_scientific_bytes_not_commit_label(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    original = diagnostic_checkpoint_scientific_binding_digest(
        stage_input=fixture["stage_input"],
        render_inputs=fixture["render_result"],
    )
    next_stage = json.loads(json.dumps(fixture["stage_input"]))
    next_stage["source_commit"] = "b" * 40
    next_stage["toolchain_digest"] = "sha256:" + "e" * 64
    next_stage["construction_envelope"]["expected_production_commit"] = "b" * 40
    next_stage["construction_envelope"]["envelope_digest"] = canonical_digest(
        next_stage["construction_envelope"], digest_field="envelope_digest"
    )

    assert diagnostic_checkpoint_scientific_binding_digest(
        stage_input=next_stage,
        render_inputs=fixture["render_result"],
    ) == original


def test_hydration_reuses_exact_frames_and_skips_semantic_provider(
    tmp_path: Path,
) -> None:
    root, checkpoint, fixture = _materialize(tmp_path)
    hydrated = hydrate_scene_configuration_diagnostic_render_inputs(
        checkpoint_root=root,
        expected_scientific_binding_digest=checkpoint["scientific_bindings"][
            "binding_digest"
        ],
    )
    semantic_request = json.loads(
        fixture["request_path"].read_text(encoding="utf-8")
    )
    semantic = hydrate_scene_configuration_diagnostic_semantic_outputs(
        checkpoint_root=root,
        current_semantic_runtime_request=semantic_request,
        output_root=tmp_path / "hydrated-semantic",
    )

    assert hydrated["derived_frame_count"] == 8
    assert all(Path(row["path"]).is_file() for row in hydrated["derived_frames"])
    assert semantic["successful_request_count"] == 8
    assert semantic["provider_calls_performed"] == 0
    assert semantic["diagnostic_checkpoint_reused"] is True
    assert len(list((tmp_path / "hydrated-semantic").rglob("*.png"))) == 8


def test_semantic_checkpoint_reuse_binds_interleaved_frames_by_camera(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    interleaved = [0, 4, 5, 1, 6, 2, 7, 3]

    request_path = fixture["request_path"]
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request_frames = request["tasks"][0]["frames"]
    request["tasks"][0]["frames"] = [request_frames[index] for index in interleaved]
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    _write(request_path, request)

    result_path = fixture["result_path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result_frames = result["tasks"][0]["frames"]
    result["tasks"][0]["frames"] = [result_frames[index] for index in interleaved]
    result["source_runtime_request_digest"] = request["request_digest"]
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    _write(result_path, result)

    receipt_path = fixture["receipt_path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_frames = receipt["frames"]
    receipt["frames"] = [receipt_frames[index] for index in interleaved]
    receipt["editor_identity"]["result_digest"] = result["result_digest"]
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write(receipt_path, receipt)

    root = tmp_path / "checkpoint-interleaved"
    checkpoint = materialize_scene_configuration_diagnostic_checkpoint(
        stage_production_input_path=fixture["stage_path"],
        render_inputs_result_path=fixture["render_path"],
        semantic_runtime_request_path=request_path,
        semantic_runtime_result_path=result_path,
        semantic_teacher_receipt_path=receipt_path,
        output_root=root,
    )
    receipt_by_camera = {row["camera_id"]: row for row in receipt["frames"]}
    inventory_by_role = {row["role"]: row for row in checkpoint["inventory"]}
    for camera_id, row in receipt_by_camera.items():
        assert inventory_by_role[f"semantic_teacher_frame:{camera_id}"]["digest"] == row[
            "whole_frame_semantic_teacher"
        ]["sha256"]

    # Reproduce the already-paid legacy checkpoint defect: its semantic bytes
    # were complete and digest-bound, but seven role labels followed calibration
    # order rather than the semantic packet's deliberately interleaved order.
    manifest_path = root / f"{SCHEMA_VERSION}.json"
    legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
    semantic_rows = sorted(
        (
            row
            for row in legacy["inventory"]
            if str(row["role"]).startswith("semantic_teacher_frame:")
        ),
        key=lambda row: row["relative_path"],
    )
    legacy_roles = [row["role"] for row in semantic_rows]
    for row, role in zip(semantic_rows, reversed(legacy_roles), strict=True):
        row["role"] = role
    legacy["checkpoint_digest"] = canonical_digest(
        legacy, digest_field="checkpoint_digest"
    )
    _write(manifest_path, legacy)

    semantic = hydrate_scene_configuration_diagnostic_semantic_outputs(
        checkpoint_root=root,
        current_semantic_runtime_request=request,
        output_root=tmp_path / "hydrated-semantic-interleaved",
    )

    assert semantic["provider_calls_performed"] == 0
    for index, frame in enumerate(request["tasks"][0]["frames"]):
        expected = receipt_by_camera[frame["camera_id"]][
            "whole_frame_semantic_teacher"
        ]["sha256"]
        assert _sha256(
            tmp_path
            / "hydrated-semantic-interleaved"
            / "tasks"
            / request["tasks"][0]["task_id"]
            / f"{index:05d}.png"
        ) == expected


def test_semantic_checkpoint_reuse_ignores_execution_budget_and_scheduling(
    tmp_path: Path,
) -> None:
    root, _checkpoint, fixture = _materialize(tmp_path)
    request = json.loads(fixture["request_path"].read_text(encoding="utf-8"))

    # Checkpoints emitted before this repair bound their stored scientific
    # digest to budget/scheduling fields. Preserve that exact legacy shape to
    # prove a new retry compares against the checkpointed request bytes rather
    # than silently requiring the old normalization algorithm forever.
    legacy_omitted = {
        "path",
        "relative_path",
        "request_digest",
        "source_commit_sha",
        "source_packet_digest",
    }

    def legacy_normalize(value: object) -> object:
        if isinstance(value, dict):
            return {
                str(key): legacy_normalize(child)
                for key, child in value.items()
                if str(key) not in legacy_omitted
            }
        if isinstance(value, list):
            return [legacy_normalize(child) for child in value]
        return value

    manifest_path = (
        root / "task_evaluation_scene_configuration_diagnostic_checkpoint.v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["semantic_teacher"]["scientific_request_digest"] = canonical_digest(
        legacy_normalize(request)
    )
    manifest["checkpoint_digest"] = canonical_digest(
        manifest, digest_field="checkpoint_digest"
    )
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_path.chmod(0o444)

    request.update(
        {
            "max_parallel_requests": 4,
            "maximum_cost_usd": 0.0,
            "expected_request_cost_usd": 0.31,
            "retry_count": 3,
        }
    )
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )

    semantic = hydrate_scene_configuration_diagnostic_semantic_outputs(
        checkpoint_root=root,
        current_semantic_runtime_request=request,
        output_root=tmp_path / "hydrated-semantic-operational-drift",
    )

    assert semantic["successful_request_count"] == 8
    assert semantic["provider_calls_performed"] == 0
    assert semantic["diagnostic_checkpoint_reused"] is True


def test_semantic_model_or_prompt_change_refuses_reuse(tmp_path: Path) -> None:
    root, _checkpoint, fixture = _materialize(tmp_path)
    request = json.loads(fixture["request_path"].read_text(encoding="utf-8"))
    request["prompt"] = "A different scientific instruction."
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")

    with pytest.raises(
        TaskEvaluationSceneConfigurationDiagnosticCheckpointError,
        match="scene_configuration_diagnostic_checkpoint_semantic_request_mismatch",
    ):
        hydrate_scene_configuration_diagnostic_semantic_outputs(
            checkpoint_root=root,
            current_semantic_runtime_request=request,
            output_root=tmp_path / "hydrated-semantic",
        )


def test_renderer_identity_change_refuses_reuse(tmp_path: Path) -> None:
    root, checkpoint, fixture = _materialize(tmp_path)
    changed_render = json.loads(json.dumps(fixture["render_result"]))
    changed_render["renderer_runtime"]["renderer_digest"] = "sha256:" + "f" * 64
    changed_stage = fixture["stage_input"]
    changed = diagnostic_checkpoint_scientific_binding_digest(
        stage_input=changed_stage,
        render_inputs=changed_render,
    )

    assert changed != checkpoint["scientific_bindings"]["binding_digest"]
    with pytest.raises(
        TaskEvaluationSceneConfigurationDiagnosticCheckpointError,
        match="scene_configuration_diagnostic_checkpoint_invalid",
    ):
        validate_scene_configuration_diagnostic_checkpoint(
            checkpoint_root=root,
            expected_scientific_binding_digest=changed,
        )


def test_checkpoint_refuses_a_different_scientific_binding(tmp_path: Path) -> None:
    root, _result, _fixture_rows = _materialize(tmp_path)

    with pytest.raises(
        TaskEvaluationSceneConfigurationDiagnosticCheckpointError,
        match="scene_configuration_diagnostic_checkpoint_invalid",
    ):
        validate_scene_configuration_diagnostic_checkpoint(
            checkpoint_root=root,
            expected_scientific_binding_digest="sha256:" + "f" * 64,
        )


def _stage_prefix(tmp_path: Path, count: int):
    stages = []
    configurations = {}
    results = []
    for index in range(1, 7):
        stage_id = f"stage-{index}"
        stages.append(
            {
                "stage_id": stage_id,
                "depends_on": [] if index == 1 else [f"stage-{index - 1}"],
            }
        )
        configuration_path = tmp_path / f"stage-{index}.json"
        configuration_path.write_text(
            canonical_json({"stage": index}) + "\n", encoding="utf-8"
        )
        configurations[stage_id] = ({"stage": index}, configuration_path)
        if index <= count:
            artifact = tmp_path / f"artifact-{index}.bin"
            artifact.write_bytes(f"artifact-{index}".encode())
            result = {
                "schema_version": "task_evaluation_scene_configuration_stage_result.v1",
                "status": "completed",
                "stage_id": stage_id,
                "configuration_digest": _sha256(configuration_path),
                "output_artifacts": [
                    {
                        "role": f"stage-{index}-output",
                        **_bound(artifact),
                    }
                ],
                "diagnostic_only": True,
                "qualification_eligible": False,
                "executed_inside_one_parent_provider_run": False,
                "configured_revision_publication_permitted": False,
                "offering_publication_permitted": False,
                "terminal_e2e_completion_permitted": False,
                "stage_result_digest": "",
            }
            result["stage_result_digest"] = canonical_digest(
                result, digest_field="stage_result_digest"
            )
            results.append(result)
    return stages, configurations, results


def test_progressive_checkpoint_carries_valid_completed_stage_prefix(
    tmp_path: Path,
) -> None:
    root, _checkpoint, _fixture_rows = _materialize(tmp_path)
    stages, configurations, results = _stage_prefix(tmp_path, 3)
    advanced_root = tmp_path / "advanced"
    advanced = advance_scene_configuration_diagnostic_checkpoint(
        checkpoint_root=root,
        stage_results=results,
        stage_sequence=stages,
        configurations=configurations,
        output_root=advanced_root,
    )
    hydrated = hydrate_scene_configuration_diagnostic_completed_stages(
        checkpoint_root=advanced_root,
        stage_sequence=stages,
        configurations=configurations,
    )

    assert advanced["completed_stage_prefix_count"] == 3
    assert [row["stage_id"] for row in hydrated] == [
        "stage-1",
        "stage-2",
        "stage-3",
    ]
    assert all(
        Path(row["output_artifacts"][0]["path"]).is_file() for row in hydrated
    )


def test_progressive_checkpoint_refuses_changed_carried_configuration(
    tmp_path: Path,
) -> None:
    root, _checkpoint, _fixture_rows = _materialize(tmp_path)
    stages, configurations, results = _stage_prefix(tmp_path, 1)
    advanced_root = tmp_path / "advanced"
    advance_scene_configuration_diagnostic_checkpoint(
        checkpoint_root=root,
        stage_results=results,
        stage_sequence=stages,
        configurations=configurations,
        output_root=advanced_root,
    )
    configurations["stage-1"][1].write_text('{"stage":"changed"}\n', encoding="utf-8")

    with pytest.raises(
        TaskEvaluationSceneConfigurationDiagnosticCheckpointError,
        match="scene_configuration_diagnostic_completed_stage_invalid:stage-1",
    ):
        hydrate_scene_configuration_diagnostic_completed_stages(
            checkpoint_root=advanced_root,
            stage_sequence=stages,
            configurations=configurations,
        )


def _permute_semantic_orderings(fixture: dict, order: list[int]) -> None:
    """Reorder the semantic request/result/teacher frame rows in place."""

    request_digest: str | None = None
    for key, rows_path in (
        ("request_path", ("tasks", 0, "frames")),
        ("result_path", ("tasks", 0, "frames")),
        ("receipt_path", ("frames",)),
    ):
        path = fixture[key]
        value = json.loads(path.read_text(encoding="utf-8"))
        holder = value
        for step in rows_path[:-1]:
            holder = holder[step]
        rows = holder[rows_path[-1]]
        holder[rows_path[-1]] = [rows[index] for index in order]
        if key == "result_path" and request_digest is not None:
            value["source_runtime_request_digest"] = request_digest
        digest_field = {
            "request_path": "request_digest",
            "result_path": "result_digest",
            "receipt_path": "receipt_digest",
        }[key]
        value.pop(digest_field, None)
        value[digest_field] = canonical_digest(value, digest_field=digest_field)
        if key == "request_path":
            request_digest = value[digest_field]
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_interleaved_semantic_camera_order_is_accepted(tmp_path: Path) -> None:
    """The packet interleaves elevations; the calibration is elevation-major.

    Run ...-15c1ade8-...-191412Z carried request order
    [e0-a0, e1-a0, e1-a1, e0-a1, ...] against calibration
    [e0-a0..e0-a3, e1-a0..e1-a3] -- the same eight cameras -- and the
    checkpoint refused a correct pass with
    scene_configuration_diagnostic_checkpoint_semantic_camera_mismatch.
    Camera identity is the set, not each producer's ordering.
    """

    fixture = _fixture(tmp_path)
    _permute_semantic_orderings(fixture, [0, 4, 5, 1, 6, 2, 7, 3])

    root = tmp_path / "checkpoint"
    result = materialize_scene_configuration_diagnostic_checkpoint(
        stage_production_input_path=fixture["stage_path"],
        render_inputs_result_path=fixture["render_path"],
        semantic_runtime_request_path=fixture["request_path"],
        semantic_runtime_result_path=fixture["result_path"],
        semantic_teacher_receipt_path=fixture["receipt_path"],
        output_root=root,
    )

    assert result["camera_count"] == 8
    assert result["semantic_teacher"]["completed_frame_count"] == 8


def test_duplicated_semantic_camera_is_still_refused(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _permute_semantic_orderings(fixture, [0, 0, 2, 3, 4, 5, 6, 7])

    with pytest.raises(
        TaskEvaluationSceneConfigurationDiagnosticCheckpointError,
        match="semantic_camera_mismatch",
    ):
        materialize_scene_configuration_diagnostic_checkpoint(
            stage_production_input_path=fixture["stage_path"],
            render_inputs_result_path=fixture["render_path"],
            semantic_runtime_request_path=fixture["request_path"],
            semantic_runtime_result_path=fixture["result_path"],
            semantic_teacher_receipt_path=fixture["receipt_path"],
            output_root=tmp_path / "checkpoint-dup",
        )
