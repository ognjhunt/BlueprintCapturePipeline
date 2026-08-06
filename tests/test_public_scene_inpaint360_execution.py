from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_inpaint360_adapter import (
    SCHEMA_VERSION as ADAPTER_SCHEMA,
)
from blueprint_pipeline.public_scene_inpaint360_execution import (
    QUALITY_BLOCKER,
    REQUIRED_STAGES,
    Inpaint360ExecutionReceiptError,
    materialize_inpaint360_execution_receipt,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    evidence = tmp_path / "evidence"
    immutable = evidence / "v3/immutable_execution"
    render_root = evidence / "v3/same_camera_after_v3_metal"
    repo.mkdir()
    immutable.mkdir(parents=True)
    camera_ids = ["approach_wide", "approach_close"]
    source = {
        "repository": "https://github.com/dfki-av/Inpaint360GS",
        "commit": "a" * 40,
        "tree": "b" * 40,
        "tracked_files_clean": True,
        "source_modified_for_adapter": False,
    }
    adapter = {
        "schema_version": ADAPTER_SCHEMA,
        "status": "prepared_unexecuted",
        "scene": {
            "publisher_scene_id": "840313",
            "target_instance_id": "160",
            "target_semantic_label": "canned_beverage",
        },
        "source": source,
        "adapter": {
            "staged_artifacts": [
                {
                    "relative_path": f"source/images/{camera_id}.png",
                    "size_bytes": 1,
                    "sha256": "sha256:" + "c" * 64,
                }
                for camera_id in camera_ids
            ]
        },
        "receipt_digest": "",
    }
    adapter["receipt_digest"] = canonical_digest(adapter, digest_field="receipt_digest")
    adapter_path = repo / "adapter.json"
    _write(adapter_path, adapter)

    final_ply = immutable / "artifacts/point_cloud.ply"
    final_ply.parent.mkdir(parents=True)
    final_ply.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 12\nend_header\n"
    )
    selection = {
        "schema_version": "inpaint360_supplemental_fusion_view_selection.v1",
        "status": "accepted",
        "selection_timing": "before_lama_color_depth_inpainting",
        "selected_view": {"view_id": "00017", "foreground_pixels": 100},
        "blockers": [],
    }
    selection_path = immutable / "supplemental_fusion_view_selection.json"
    _write(selection_path, selection)
    budget = {
        "schema_version": "inpaint360_added_gaussian_budget_validation.v1",
        "status": "accepted",
        "baseline_vertex_count": 10,
        "post_removal_vertex_count": 9,
        "final_vertex_count": 12,
        "removed_vertex_count": 1,
        "added_vertex_count": 3,
        "maximum_added_vertex_count": 20,
        "blockers": [],
    }
    budget_path = immutable / "added_gaussian_budget_validation.json"
    _write(budget_path, budget)
    workflow = [
        {"stage": stage, "returncode": 0, "timed_out": False}
        for stage in REQUIRED_STAGES
    ]
    nested = {
        "repository": "https://github.com/advimman/lama",
        "commit": "d" * 40,
        "tree": "e" * 40,
        "matches": True,
        "changed_files": [],
    }
    runtime = {
        "schema_version": "adp_inpaint360_interiorgs_result.v1",
        "status": "completed",
        "scene_id": "840313",
        "target_instance_id": "160",
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "source_identity_before": {"matches": True, "changed_files": []},
        "source_identity_after": {"matches": True, "changed_files": []},
        "nested_dependency_identity_before": nested,
        "nested_dependency_identity_after": nested,
        "adapter_identity_before": {"matches": True, "changed_files": []},
        "source_modified": False,
        "workflow": workflow,
        "inpaint_3d_executed": True,
        "retry_cap": 0,
        "execution_source_class": (
            "released_source_with_digest_bound_blueprint_obb_and_input_validity_adapters"
        ),
        "unchanged_source_execution_claimed": False,
        "final_point_cloud": {
            "relative_path": final_ply.relative_to(immutable).as_posix(),
            "size_bytes": final_ply.stat().st_size,
            "sha256": _sha256(final_ply),
        },
        "supplemental_fusion_view_selection": selection,
        "added_gaussian_budget_validation": budget,
        "blockers": [],
    }
    runtime_path = immutable / "adp_inpaint360_interiorgs_result.json"
    _write(runtime_path, runtime)

    teardown = {
        "schema_version": "vast_teardown_manifest.v1",
        "status": "completed",
        "vast_instance_ids": [123],
        "teardown_actions_performed": [
            {
                "instance_id": 123,
                "action": "destroy_instance",
                "http_status_code": 200,
                "status": "completed",
            }
        ],
        "runner_gpu_teardown_completed": True,
        "continuing_spend_from_this_run": False,
    }
    teardown_path = evidence / "v3/vast_provider_run/vast_teardown_manifest.json"
    _write(teardown_path, teardown)
    run = {
        "schema_version": "adp_inpaint360_interiorgs_vast_run.v1",
        "status": "completed",
        "execution_result_path": str(runtime_path),
        "teardown_manifest_path": str(teardown_path),
        "estimated_cost_usd": 0.2,
        "hard_cap_usd": 6.0,
        "hard_ttl_seconds": 14_400,
        "retry_cap": 0,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "blockers": [],
    }
    run_path = evidence / "v3/adp_inpaint360_interiorgs_vast_result.json"
    _write(run_path, run)

    render_rows = []
    for index, camera_id in enumerate(camera_ids):
        frame = render_root / "frames" / f"{camera_id}.png"
        frame.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), color=(40 + index, 50, 60)).save(frame)
        render_rows.append(
            {
                "camera_id": camera_id,
                "relative_path": f"frames/{camera_id}.png",
                "digest": _sha256(frame),
                "width": 8,
                "height": 6,
                "pixel_std": 1.0,
            }
        )
    render = {
        "schema_version": "sealed_camera_render_manifest.v1",
        "status": "rendered_exact_cameras",
        "rendered_by": "reference_spark_renderer_exact_camera",
        "camera_set_label": "frozen_non_orbiting_2",
        "provider_splat_import_receipt_digest": _sha256(final_ply),
        "provider_reconstruction_alignment_digest": "sha256:" + "f" * 64,
        "splat_digest": _sha256(final_ply),
        "projection_pixel_convention": "colmap_pixel_center_half_offset",
        "renderer_identity": {
            "graphics_backend": "metal",
            "harness_digest": "sha256:" + "1" * 64,
            "render_entry_digest": "sha256:" + "2" * 64,
            "node_version": "v22",
            "warmup_ms": 3500,
            "settle_frames": 12,
            "settle_ms": 120,
        },
        "renders": render_rows,
        "render_count": len(render_rows),
        "rendered_by_isaac_rtx": False,
        "hidden_pixels_read_by_renderer": False,
        "proof_effect": "reference_render_for_independent_evaluation_only",
        "claim_ceiling": "appearance_reconstruction_candidate",
    }
    render["sealed_camera_render_manifest_digest"] = canonical_digest(
        render, digest_field="sealed_camera_render_manifest_digest"
    )
    render_path = render_root / "sealed_camera_render_manifest.v1.json"
    _write(render_path, render)
    locality = {
        "schema_version": "public_scene_inpainting_locality_measurement.v1",
        "status": "measured_no_admission_effect",
        "after_render_manifest_sha256": _sha256(render_path),
        "after_render_manifest_digest": render["sealed_camera_render_manifest_digest"],
        "dilation_pixels": 16,
        "rows": [
            {
                "camera_id": row["camera_id"],
                "after_sha256": row["digest"],
                "outside_mask_mean_absolute_error": 0.1,
            }
            for row in render_rows
        ],
        "aggregate": {"view_count": len(render_rows)},
        "thresholds_frozen_before_evaluation": False,
        "quality_pass_claimed": False,
        "admission_effect": "none",
        "claim_ceiling": "outside_mask_edit_locality_measurement_only",
        "raw_secret_values_recorded": False,
    }
    locality["locality_measurement_digest"] = canonical_digest(
        locality, digest_field="locality_measurement_digest"
    )
    locality_path = render_root / "outside_mask_locality_measurement.v1.json"
    _write(locality_path, locality)
    return {
        "repo": repo,
        "evidence": evidence,
        "adapter": adapter_path,
        "runtime": runtime_path,
        "run": run_path,
        "selection": selection_path,
        "budget": budget_path,
        "render": render_path,
        "locality": locality_path,
        "ply": final_ply,
        "teardown": teardown_path,
    }


def _materialize(paths: dict[str, Path]) -> dict:
    return materialize_inpaint360_execution_receipt(
        adapter_receipt_path=paths["adapter"],
        runtime_result_path=paths["runtime"],
        run_result_path=paths["run"],
        selected_view_receipt_path=paths["selection"],
        gaussian_budget_receipt_path=paths["budget"],
        metal_render_manifest_path=paths["render"],
        locality_measurement_path=paths["locality"],
        evidence_root=paths["evidence"],
        repo_root=paths["repo"],
        receipt_output=paths["repo"] / "receipt.json",
    )


def test_inpaint360_execution_seals_observed_v3_as_rejected_not_admitted(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)

    receipt = _materialize(paths)

    assert receipt["status"] == "executed_rejected_quality"
    assert receipt["blockers"] == [QUALITY_BLOCKER]
    assert receipt["claim_boundary"]["released_method_execution_completed"] is True
    assert receipt["claim_boundary"]["successful_inpainting_admitted"] is False
    assert receipt["claim_boundary"]["public_scene_suite_component_admitted"] is False
    assert receipt["execution"]["final_point_cloud"]["sha256"] == _sha256(paths["ply"])
    assert receipt["independent_render"]["manifest"]["graphics_backend"] == "metal"
    assert receipt["quality"]["admission_effect"] == "none"
    assert canonical_digest(receipt, digest_field="receipt_digest") == receipt["receipt_digest"]


def test_inpaint360_execution_rejects_changed_final_ply(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["ply"].write_bytes(paths["ply"].read_bytes() + b"changed")

    with pytest.raises(
        Inpaint360ExecutionReceiptError, match="inpaint360_final_point_cloud_changed"
    ):
        _materialize(paths)


def test_inpaint360_execution_rejects_caller_asserted_provider_zero(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    teardown = json.loads(paths["teardown"].read_text())
    teardown["teardown_actions_performed"] = []
    _write(paths["teardown"], teardown)

    with pytest.raises(
        Inpaint360ExecutionReceiptError, match="inpaint360_provider_zero_not_proven"
    ):
        _materialize(paths)


def test_inpaint360_execution_rejects_nonaccepted_selected_view(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    selection = json.loads(paths["selection"].read_text())
    selection["status"] = "blocked"
    _write(paths["selection"], selection)

    with pytest.raises(
        Inpaint360ExecutionReceiptError,
        match="inpaint360_selected_view_receipt_not_accepted",
    ):
        _materialize(paths)


def test_inpaint360_execution_rejects_non_metal_render(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    render = json.loads(paths["render"].read_text())
    render["renderer_identity"]["graphics_backend"] = "swiftshader"
    render["sealed_camera_render_manifest_digest"] = canonical_digest(
        render, digest_field="sealed_camera_render_manifest_digest"
    )
    _write(paths["render"], render)

    with pytest.raises(
        Inpaint360ExecutionReceiptError,
        match="inpaint360_exact_metal_render_manifest_invalid",
    ):
        _materialize(paths)


def test_inpaint360_execution_rejects_locality_manifest_mismatch(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    locality = json.loads(paths["locality"].read_text())
    locality["after_render_manifest_digest"] = "sha256:" + "0" * 64
    locality["locality_measurement_digest"] = canonical_digest(
        locality, digest_field="locality_measurement_digest"
    )
    _write(paths["locality"], locality)

    with pytest.raises(
        Inpaint360ExecutionReceiptError,
        match="inpaint360_outside_mask_locality_measurement_invalid",
    ):
        _materialize(paths)
