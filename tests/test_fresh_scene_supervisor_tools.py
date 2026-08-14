from __future__ import annotations

from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_supervisor.capabilities import SupervisorContext
from blueprint_pipeline.task_evaluation_supervisor.contracts import AutonomyMode
from blueprint_pipeline.task_evaluation_supervisor.supervisor import (
    default_authority_envelope,
)
from blueprint_pipeline.task_evaluation_supervisor.tools import (
    ToolRegistry,
    non_spend_tool_bindings,
)


def _authority(registry: ToolRegistry, *digests: str) -> dict:
    return default_authority_envelope(
        run_id="fresh-scene-tools-test",
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=list(digests),
    ).to_mapping()


def test_agents_sdk_can_inspect_but_not_advance_fresh_scene(tmp_path: Path) -> None:
    status = {
        "schema_version": "fresh_scene_paired_target_preparation.v1",
        "status": "blocked",
        "first_blocker": "fresh_scene_sam31_source_tracks_missing",
        "next_required_stage": "sam31_source_tracks",
        "status_digest": "",
    }
    status["status_digest"] = canonical_digest(status, digest_field="status_digest")
    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="fresh-scene-tools-test",
        customer_question="Prepare one fresh scene.",
        supervisor_output_dir=str(tmp_path),
        fresh_scene_preparation_status=status,
    )
    bindings = {
        binding.tool_id: binding
        for binding in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=_authority(registry, status["status_digest"]),
        )
    }

    assert "inspect_fresh_scene_preparation" in bindings
    assert "materialize_calibrated_object_masks" not in bindings
    observation = bindings["inspect_fresh_scene_preparation"].invoke(
        {"status_digest": status["status_digest"]}
    )
    assert observation["status"] == "completed"
    assert observation["typed_result"]["next_required_stage"] == "sam31_source_tracks"
    assert observation["typed_result"]["agent_advanced_stage"] is False
    assert observation["proof_effect"] == "none"


def test_agents_sdk_invokes_digest_bound_mask_materializer(tmp_path: Path) -> None:
    request = {
        "schema_version": "fresh_scene_calibrated_mask_tool_request.v1",
        "reviewed_track_selection_receipt_digest": "sha256:" + "a" * 64,
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    calls: list[dict] = []

    def materializer(*, request: dict, output_root: Path) -> dict:
        calls.append({"request": request, "output_root": output_root})
        result = {
            "schema_version": "public_scene_calibrated_object_mask_set.v1",
            "status": "calibrated_inferred_object_masks_materialized_pending_review",
            "task_count": 2,
            "camera_count_total": 16,
            "claim_boundary": {"masks_are_model_inferred_candidates": True},
            "receipt_digest": "",
        }
        result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
        return result

    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="fresh-scene-tools-test",
        customer_question="Prepare one fresh scene.",
        supervisor_output_dir=str(tmp_path),
        fresh_scene_calibrated_mask_request=request,
        fresh_scene_calibrated_mask_materializer=materializer,
    )
    bindings = {
        binding.tool_id: binding
        for binding in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=_authority(registry, request["request_digest"]),
        )
    }

    observation = bindings["materialize_calibrated_object_masks"].invoke(
        {"request_digest": request["request_digest"]}
    )
    assert observation["status"] == "completed"
    assert observation["mutability"] == "reversible_mutation"
    assert observation["cost_usd"] == 0.0
    assert observation["typed_result"]["agent_selected_unreviewed_tracks"] is False
    assert observation["typed_result"]["masks_are_model_inferred_candidates"] is True
    assert len(calls) == 1
    assert calls[0]["output_root"] == tmp_path / "generated/calibrated_object_masks"
    assert (tmp_path / "generated/calibrated_object_masks/tool_receipt.json").is_file()


def test_agents_sdk_invokes_digest_bound_sam_packet_builder(tmp_path: Path) -> None:
    request = {
        "schema_version": "fresh_scene_sam31_task_input_tool_request.v1",
        "calibrated_view_receipt_digest": "sha256:" + "b" * 64,
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    calls: list[dict] = []

    def materializer(*, request: dict, output_root: Path) -> dict:
        calls.append({"request": request, "output_root": output_root})
        result = {
            "schema_version": "public_scene_sam31_task_input_packet.v1",
            "status": "prepared_no_upload_no_execution",
            "task_id": "task_a",
            "camera_count": 8,
            "paid_execution_started": False,
            "provider_mutations_performed": 0,
            "receipt_digest": "",
        }
        result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
        return result

    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="fresh-scene-tools-test",
        customer_question="Prepare one fresh scene.",
        supervisor_output_dir=str(tmp_path),
        fresh_scene_sam31_task_input_request=request,
        fresh_scene_sam31_task_input_materializer=materializer,
    )
    bindings = {
        binding.tool_id: binding
        for binding in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=_authority(registry, request["request_digest"]),
        )
    }

    observation = bindings["materialize_sam31_task_inputs"].invoke(
        {"request_digest": request["request_digest"]}
    )
    assert observation["status"] == "completed"
    assert observation["typed_result"]["agent_authored_frame_registry"] is False
    assert observation["typed_result"]["paid_execution_started"] is False
    assert observation["typed_result"]["provider_mutations_performed"] == 0
    assert len(calls) == 1
    assert calls[0]["output_root"] == tmp_path / "generated/sam31_task_inputs"


def test_agents_sdk_invokes_digest_bound_removal_freeze_builder(tmp_path: Path) -> None:
    request = {
        "schema_version": "fresh_scene_removal_freeze_tool_request.v1",
        "tasks": {"task_a": {}},
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    calls: list[dict] = []

    def materializer(*, request: dict, output_root: Path) -> dict:
        calls.append({"request": request, "output_root": output_root})
        result = {
            "schema_version": "fresh_scene_removal_freeze_set.v1",
            "status": "excision_and_segment_sweep_freezes_materialized_no_execution",
            "task_count": 1,
            "paid_execution_started": False,
            "provider_mutations_performed": 0,
            "agent_selected_gaussian_indices": False,
            "canonical_source_altered": False,
            "receipt_digest": "",
        }
        result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
        return result

    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="fresh-scene-tools-test",
        customer_question="Prepare one fresh scene.",
        supervisor_output_dir=str(tmp_path),
        fresh_scene_removal_freeze_request=request,
        fresh_scene_removal_freeze_materializer=materializer,
    )
    bindings = {
        binding.tool_id: binding
        for binding in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=_authority(registry, request["request_digest"]),
        )
    }

    observation = bindings["materialize_fresh_scene_removal_freezes"].invoke(
        {"request_digest": request["request_digest"]}
    )
    assert observation["status"] == "completed"
    assert observation["typed_result"]["paid_execution_started"] is False
    assert observation["typed_result"]["provider_mutations_performed"] == 0
    assert observation["typed_result"]["agent_selected_gaussian_indices"] is False
    assert observation["typed_result"]["canonical_source_altered"] is False
    assert calls[0]["output_root"] == tmp_path / "generated/removal_freezes"


def test_agents_sdk_invokes_digest_bound_segment_cutout_builder(tmp_path: Path) -> None:
    request = {
        "schema_version": "fresh_scene_segment_cutout_tool_request.v1",
        "task_freeze_paths": ["task-a.json", "task-b.json"],
        "sweep_freeze_paths_by_task": {"task_a": "a.json", "task_b": "b.json"},
        "contribution_manifest_paths_by_task": {
            "task_a": "a-manifest.json",
            "task_b": "b-manifest.json",
        },
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    calls: list[dict] = []

    def materializer(*, request: dict, output_root: Path) -> dict:
        calls.append({"request": request, "output_root": output_root})
        result = {
            "schema_version": "adp009d_segment_contribution_cutout_set.v1",
            "status": (
                "repair_supported_segment_contribution_cutout_materialized_"
                "pending_full_deleted_layer_projection"
            ),
            "task_candidates": [{"task_id": "task_a"}, {"task_id": "task_b"}],
            "claim_boundary": {"canonical_source_altered": False},
            "receipt_digest": "",
        }
        result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
        return result

    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="fresh-scene-tools-test",
        customer_question="Prepare one fresh scene.",
        supervisor_output_dir=str(tmp_path),
        fresh_scene_segment_cutout_request=request,
        fresh_scene_segment_cutout_materializer=materializer,
    )
    bindings = {
        binding.tool_id: binding
        for binding in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=_authority(registry, request["request_digest"]),
        )
    }
    observation = bindings["materialize_fresh_scene_segment_cutout"].invoke(
        {"request_digest": request["request_digest"]}
    )
    assert observation["status"] == "completed"
    assert observation["typed_result"]["task_count"] == 2
    assert observation["typed_result"]["canonical_source_altered"] is False
    assert observation["typed_result"]["agent_selected_gaussian_indices"] is False
    assert observation["typed_result"]["paid_execution_started"] is False
    assert calls[0]["output_root"] == tmp_path / "generated/segment_cutout_set"


def test_agents_sdk_invokes_digest_bound_artifixer_candidate_builder(
    tmp_path: Path,
) -> None:
    request = {
        "schema_version": "fresh_scene_artifixer_candidate_preparation_request.v1",
        "segment_cutout_set_path": "cutout.json",
        "execution_authority_path": "authority.json",
        "selected_task_ids": ["task_a", "task_b"],
        "object_absent_reference_receipt_paths": [],
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    calls: list[dict] = []

    def materializer(*, request: dict, output_root: Path) -> dict:
        calls.append({"request": request, "output_root": output_root})
        result = {
            "schema_version": "fresh_scene_artifixer_candidate_preparation.v1",
            "status": "artifixer_candidate_inputs_prepared_no_model_no_execution",
            "task_count": 2,
            "semantic_teacher_execution_started": False,
            "artifixer3d_execution_started": False,
            "provider_mutations_performed": 0,
            "canonical_source_altered": False,
            "next_required_stage": "semantic_teacher_receipts",
            "receipt_digest": "",
        }
        result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
        return result

    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="fresh-scene-tools-test",
        customer_question="Prepare one fresh scene.",
        supervisor_output_dir=str(tmp_path),
        fresh_scene_artifixer_candidate_request=request,
        fresh_scene_artifixer_candidate_materializer=materializer,
    )
    bindings = {
        binding.tool_id: binding
        for binding in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=_authority(registry, request["request_digest"]),
        )
    }
    observation = bindings["materialize_fresh_scene_artifixer_candidate"].invoke(
        {"request_digest": request["request_digest"]}
    )
    assert observation["status"] == "completed"
    assert observation["typed_result"]["task_count"] == 2
    assert observation["typed_result"]["next_required_stage"] == (
        "semantic_teacher_receipts"
    )
    assert observation["typed_result"]["semantic_teacher_execution_started"] is False
    assert observation["typed_result"]["artifixer3d_execution_started"] is False
    assert observation["typed_result"]["provider_mutations_performed"] == 0
    assert observation["typed_result"]["canonical_source_altered"] is False
    assert calls[0]["output_root"] == tmp_path / "generated/artifixer_candidate"
