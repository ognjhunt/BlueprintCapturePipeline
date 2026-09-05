from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_calibrated_object_masks import materialize_calibrated_object_mask_set
from blueprint_pipeline.public_scene_sam31_track_selection_review import (
    materialize_sam31_track_selection_inputs, load_validated_sam31_track_selection_inputs,
    materialize_sam31_track_selection_review_candidate,
)
from blueprint_pipeline.public_scene_segment_contribution_cutout import (
    materialize_segment_contribution_cutout_set, materialize_segment_contribution_sweep_freeze,
)
from blueprint_pipeline.task_evaluation_scene_configuration_bundle import _portable_render_inputs
from blueprint_pipeline.task_evaluation_scene_configuration_render_inputs import (
    TaskEvaluationSceneConfigurationRenderInputsError, materialize_scene_configuration_render_inputs,
)
from blueprint_pipeline.task_evaluation_scene_configuration_stage_configuration import (
    SAM31_MASK_SOURCE, SAM31_SELECTION_RULE, stage_one_gaussian_inputs_refusal,
)
from tests.test_public_scene_calibrated_object_masks import (
    _fixture, _review, _task_packets_for_fixture, _run_production_ai_review,
)
from tests.test_adp009d_replacement_occlusion import _source_splat
from tests.test_adp_retained_scene_render_packet import _absolute_record, _relative_record, _write_json


def _seal(path, value, field="receipt_digest"):
    value[field] = canonical_digest(value, digest_field=field)
    _write_json(path, value)
    return path


def _sam_case(tmp_path, *, review_kind="human", monkeypatch=None, removal_only=False):
    fixture = _fixture(tmp_path / "sam-source", camera_count=16 if review_kind == "ai" else 2)
    fixture["tasks"] = fixture["tasks"][1:]
    fixture["task_inputs"] = {"task_b": fixture["task_inputs"]["task_b"]}
    removal_fixture = None
    if removal_only:
        from tests.test_public_scene_removal_selection import _selections
        removal_fixture, selection_result, selection_docs = _selections(tmp_path / "source-selection")
        fixture["tasks"] = [Path(selection_result["task_selection"]["path"])]
        fixture["task_inputs"] = {
            selection_docs["task"]["task_id"]: fixture["task_inputs"]["task_b"],
        }
    task = json.loads(fixture["tasks"][0].read_text())
    task_id = task["task_id"]
    scene_id = "841757" if removal_only else "fixture-scene"
    selected = {task_id: ["laptop-track"]}
    source = _source_splat(tmp_path / "standard.ply", count=10)
    raw = tmp_path / "original.compressed.ply"
    if removal_fixture is None:
        raw.write_bytes(b"test-only-compressed-source")
    else:
        installation = json.loads(removal_fixture["installation_receipt"].read_text())
        raw = Path(installation["destination_root"]) / installation["files"][0]["relative_path"]
    conversion_path = _seal(tmp_path / "conversion.json", {
        "schema_version": "standard_splat_conversion_receipt.v1",
        "status": "standard_splat_conversion_materialized",
        "source": _absolute_record(raw),
        "output": {**_absolute_record(source), "standard_3dgs_schema_validated": True,
                   "gaussian_count_preserved": True},
        "raw_source_uploaded": False, "gaussian_ownership_claimed": False,
    })
    conversion = json.loads(conversion_path.read_text())
    packets = _task_packets_for_fixture(fixture["source"].parent, fixture)
    packet = json.loads(packets[0].read_text())
    calibrated_path = Path(packet["calibrated_view_receipt"]["path"])
    calibrated = json.loads(calibrated_path.read_text())
    calibrated["scene"].update(target_instance_id=task["source_object"]["instance_id"],
                               publisher_scene_id=scene_id)
    calibrated["source_admission"] = {
        "standard_splat_conversion_receipt_digest": conversion["receipt_digest"],
    }
    calibrated["renderer"] = {
        "authorization_class": "method_input", "purpose_bound": True,
        "render_manifest_digests": {"images": "sha256:" + "a" * 64},
        "renderer_identity": {"name": "fixture-reference-renderer"},
    }
    sealed_path = fixture["images"] / "sealed_camera_render_manifest.v1.json"
    sealed = {
        "schema_version": "sealed_camera_render_manifest.v1", "status": "rendered_exact_cameras",
        "authorization_class": "method_input", "splat_digest": _absolute_record(source)["sha256"],
        "calibrated_camera_file": {"digest": _absolute_record(fixture["cameras"])["sha256"],
                                   "binding": "caller_file_exact_match"},
        "renders": [{"camera_id": row["camera_id"], "digest": row["sha256"]}
                    for row in calibrated["derived_artifacts"]["images"]],
        "render_count": len(calibrated["derived_artifacts"]["images"]),
    }
    _seal(sealed_path, sealed, "sealed_camera_render_manifest_digest")
    calibrated["renderer"]["render_manifest_digests"]["images"] = sealed["sealed_camera_render_manifest_digest"]
    _seal(calibrated_path, calibrated)
    packet["calibrated_view_receipt"] = {
        **_absolute_record(calibrated_path), "receipt_digest": calibrated["receipt_digest"],
    }
    _seal(packets[0], packet)
    prepared_root = tmp_path / "selection"
    materialize_sam31_track_selection_inputs(
        task_input_packet_paths=packets,
        source_track_result_paths_by_task={task_id: fixture["source"]},
        selected_track_ids_by_task=selected, output_root=prepared_root,
    )
    _, fixture["task_inputs"], _ = load_validated_sam31_track_selection_inputs(
        prepared_root / "public_scene_sam31_track_selection_inputs.v1.json")
    if review_kind == "ai":
        candidate_root = tmp_path / "review-candidate"
        materialize_sam31_track_selection_review_candidate(
            task_freeze_paths=fixture["tasks"], task_inputs=fixture["task_inputs"],
            selected_track_ids_by_task=selected, output_root=candidate_root,
        )
        _run_production_ai_review(
            monkeypatch=monkeypatch,
            candidate_path=candidate_root / "public_scene_sam31_track_selection_review_candidate.v1.json",
            output_root=tmp_path / "ai-review", decision="accepted",
        )
        review = tmp_path / "ai-review" / "public_scene_sam31_track_selection_ai_visual_review.v1.json"
    else:
        review = _review(tmp_path, fixture, selected)
    mask_root = tmp_path / "calibrated-masks"
    masks = materialize_calibrated_object_mask_set(
        task_freeze_paths=fixture["tasks"], task_inputs=fixture["task_inputs"],
        selected_track_ids_by_task=selected,
        reviewed_track_selection_receipt_path=review, output_root=mask_root,
    )
    mask_set_path = mask_root / "public_scene_calibrated_object_mask_set.v1.json"
    mask_task = masks["tasks"][0]
    camera_ids = sorted(row["camera_id"] for row in mask_task["masks"])
    rows = []
    for row in mask_task["masks"]:
        mask_path = mask_root / row["mask"]["relative_path"]
        rows.append({
            "camera_id": row["camera_id"], "historical_outer_mask": _absolute_record(mask_path),
            "zones": {zone: _absolute_record(mask_path) for zone in ("protected", "target_core", "uncertain")},
        })
    original_root = tmp_path / "excision"
    original_root.mkdir()
    # Sweep producer requires root-relative zone records. These are explicitly
    # synthetic measurement masks; no geometry/physics qualification is claimed.
    for row in rows:
        for zone, record in row["zones"].items():
            destination = original_root / f"{row['camera_id']}.{zone}.png"
            destination.write_bytes(Path(record["path"]).read_bytes())
            row["zones"][zone] = _relative_record(original_root, destination)
    contribution_method = {
        "name": "FlashSplat", "repository": "https://github.com/florinshen/FlashSplat",
        "commit": "3e3b14786333bf0163ba1b8541e86a3765112d7d",
        "rasterizer_repository": "https://github.com/florinshen/flashsplat-rasterization",
        "rasterizer_commit": "189c483ffa33dd6d5661343ce496df0c6eb80a0c",
        "contribution_semantics": "front_to_back_transmittance_times_alpha",
        "source_modified": False, "depth_anything_3_used": False,
    }
    original_path = _seal(original_root / "freeze.json", {
        "schema_version": "adp009b_gaussian_excision_audit_freeze.v1",
        "status": "frozen_before_excision_execution",
        "source_standard_splat": _absolute_record(source),
        "scene": {"task_id": task_id, "publisher_scene_id": scene_id,
                  "target_instance_id": task["source_object"]["instance_id"],
                  "removal_id": task["removal_plan"]["removal_id"],
                  "mask_set_id": task["removal_plan"]["mask_set_id"]},
        "target_collision_prim_path": task["removal_plan"]["source_collider_prim_path"],
        "camera_contract": _absolute_record(fixture["cameras"]),
        "camera_split": {"camera_count": len(camera_ids)},
        "masks": rows, "source_images": mask_task["source_images"],
        "render_input_packet": {"receipt": _absolute_record(calibrated_path)},
        "contribution_method": contribution_method,
        "policy": {"contribution_quantization_decimals": 6, "minimum_per_view_contribution": .01},
        "learned_policy_outcomes_observed": False, "replacement_usd_inserted": False,
    }, "freeze_digest")
    sweep_root = tmp_path / "sweep"
    sweep = materialize_segment_contribution_sweep_freeze(
        excision_freeze_path=original_path, output_root=sweep_root,
    )
    sweep_path = sweep_root / "adp009b_gaussian_excision_audit_freeze.v1.json"
    measurements = tmp_path / "measurements"
    measurements.mkdir()
    array = np.zeros((len(camera_ids), 3, 10), dtype=np.float32)
    array[0, 1, 2] = .2
    array[1, 2, 3] = .3
    repetitions = []
    for index in range(2):
        path = measurements / f"repetition-{index}.npz"
        np.savez_compressed(path, per_view_class_contribution=array)
        repetitions.append(_relative_record(measurements, path))
    contribution = _seal(measurements / "manifest.json", {
        "schema_version": "adp009b_gaussian_excision_contribution_evidence.v1",
        "freeze_digest": sweep["freeze_digest"],
        "class_order": ["protected", "target_core", "uncertain"],
        "camera_ids": camera_ids, "method": {**contribution_method, "released_code_executed": True},
        "repetitions": repetitions, "heldout_cameras_accessed_for_classification": False,
    }, "manifest_digest")
    cutout_root = tmp_path / "cutout"
    materialize_segment_contribution_cutout_set(
        source_standard_splat_path=source, task_freeze_paths=fixture["tasks"],
        sweep_freeze_paths_by_task={task_id: sweep_path},
        contribution_manifest_paths_by_task={task_id: contribution}, output_root=cutout_root,
    )
    evidence_paths = {
        "selection_inputs": prepared_root / "public_scene_sam31_track_selection_inputs.v1.json",
        "track_selection_review": review, "calibrated_mask_set": mask_set_path,
        "segment_cutout_set": cutout_root / "adp009d_segment_contribution_cutout_set.v1.json",
        "standard_splat_conversion": conversion_path,
    }
    authority = {"accepted_by": "synthetic-fixture-owner", "accepted_on": "2026-09-04",
                 "authority_reference": "synthetic-only", "private_derived_frame_disclosure_authorized": True,
                 "provider_retention_terms_accepted": True, "provider_training_terms_accepted": True,
                 "provider_training_authorized": False}
    config = {
        "schema_version": "observed_appearance_object_removal_configuration.v1",
        "production_render_required": True,
        "sam31_review_kind": review_kind,
        "source_object": {"scene_id": scene_id,
                          "publisher_instance_id": task["source_object"]["instance_id"],
                          "collision_candidate_prim": task["removal_plan"]["source_collider_prim_path"]},
        "gaussian_cutout": {"selection_rule": SAM31_SELECTION_RULE,
                           "retained_rows_must_remain_byte_exact": True},
        "required_views": {"minimum": 2, "lossless_inputs": True, "mask_source": SAM31_MASK_SOURCE},
        "provider_disclosure": {"raw_interiorgs_bytes": False, "derived_rendered_views": True},
        "human_authority": authority,
        "sam31_exact_mask_evidence": {
            key + "_digest": json.loads(path.read_text())["receipt_digest"]
            for key, path in evidence_paths.items()
        },
    }
    rights_path = tmp_path / "rights.json"
    _write_json(rights_path, {"private_provider_processing_allowed": True,
                             "provider_training_allowed": False, "public_redistribution_allowed": False})
    envelope = {
        "request": {"run_id": "sam31-fixture-construction"},
        "sam31_exact_mask_inputs": {key: _absolute_record(path) for key, path in evidence_paths.items()},
        "materialized_references": [
            {"contract_path": contract, "materialized_path": str(path),
             "digest": _absolute_record(path)["sha256"], "size_bytes": path.stat().st_size}
            for contract, path in (("scene.appearance.representation", raw), ("scene.rights.admission", rights_path))
        ],
    }
    return envelope, config, masks, mask_root, raw


def _consume(envelope, config, output):
    def forbidden(**kwargs):
        pytest.fail("SAM branch must not rerender, decode or synthesize projected masks")
    return materialize_scene_configuration_render_inputs(
        envelope=envelope, stage_one_configuration=config, output_root=output,
        renderer=forbidden, runtime_resolver=forbidden, splat_decoder=forbidden,
    )


def test_accepted_sam_masks_and_measured_global_cutout_are_consumed_exactly(tmp_path):
    envelope, config, masks, mask_root, raw = _sam_case(tmp_path)
    raw_before = raw.read_bytes()
    result = _consume(envelope, config, tmp_path / "result")
    assert stage_one_gaussian_inputs_refusal(config) is None
    assert result["source_object_masks"]["source"] == SAM31_MASK_SOURCE
    assert result["source_object_masks"]["observed_segmentation_truth"] is False
    assert result["derived_gaussian_cutout"]["removed_count"] == 2
    assert result["derived_gaussian_cutout"]["retained_count"] == 8
    assert result["raw_interiorgs_bytes_in_provider_packet"] is False
    assert result["provider_mutation_performed"] is False
    assert raw.read_bytes() == raw_before
    runtime = tmp_path / "runtime"
    portable = _portable_render_inputs(runtime=runtime, render=result)
    assert set(envelope["sam31_exact_mask_inputs"]) < set(portable["sam31_evidence_records"])
    for key, row in portable["sam31_evidence_records"].items():
        assert not Path(row["path"]).is_absolute()
        assert (runtime / row["path"]).is_file()
        if key in envelope["sam31_exact_mask_inputs"]:
            assert (runtime / row["path"]).read_bytes() == Path(
                envelope["sam31_exact_mask_inputs"][key]["path"]
            ).read_bytes()
    assert not any(path.name == raw.name for path in runtime.rglob("*"))
    originals = {row["camera_id"]: row for row in masks["tasks"][0]["masks"]}
    for row in result["derived_frames"]:
        assert Path(row["source_object_mask"]["path"]).read_bytes() == (
            mask_root / originals[row["camera_id"]]["mask"]["relative_path"]
        ).read_bytes()


@pytest.mark.parametrize("mutation,error", [
    ("mask_changed", "file_bytes_changed"),
    ("unbound_digest", "configuration_evidence_mismatch"),
    ("missing_review", "evidence_missing"),
    ("conversion_changed", "file_bytes_changed"),
    ("aabb_substitution", "stage_configuration_invalid"),
])
def test_sam_evidence_cannot_drift_or_fall_back_to_bounds(tmp_path, mutation, error):
    envelope, config, masks, mask_root, _ = _sam_case(tmp_path)
    if mutation == "mask_changed":
        path = mask_root / masks["tasks"][0]["masks"][0]["mask"]["relative_path"]
        Image.new("L", (4, 3), 255).save(path)
    elif mutation == "unbound_digest":
        config["sam31_exact_mask_evidence"]["calibrated_mask_set_digest"] = "sha256:" + "0" * 64
    elif mutation == "missing_review":
        envelope["sam31_exact_mask_inputs"].pop("track_selection_review")
    elif mutation == "conversion_changed":
        Path(envelope["sam31_exact_mask_inputs"]["standard_splat_conversion"]["path"]).write_text("{}")
    elif mutation == "aabb_substitution":
        config["gaussian_cutout"]["selection_rule"] = "gaussian_center_inside_registered_source_object_aabb"
        config["gaussian_cutout"]["aabb_padding_m"] = 0.0
    with pytest.raises(TaskEvaluationSceneConfigurationRenderInputsError, match=error):
        _consume(envelope, config, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_pending_immutable_plan_is_fulfilled_only_by_bound_producer_result(tmp_path):
    envelope, config, _, _, _ = _sam_case(tmp_path)
    config.pop("sam31_exact_mask_evidence")
    config["sam31_preparation_plan"] = {
        "uri": "s3://private/immutable/sam-plan.json", "digest": "sha256:" + "e" * 64,
        "size_bytes": 100,
    }
    envelope["request"]["expected_production_commit"] = "a" * 40
    result = {
        "schema_version": "task_evaluation_sam31_preparation_result.v1",
        "status": "exact_mask_inputs_ready", "source_commit": "a" * 40,
        "plan_digest": config["sam31_preparation_plan"]["digest"],
        "evidence": envelope["sam31_exact_mask_inputs"], "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    envelope["sam31_preparation_result"] = result
    assert stage_one_gaussian_inputs_refusal(config) is None
    consumed = _consume(envelope, config, tmp_path / "result")
    assert consumed["derived_gaussian_cutout"]["removed_count"] == 2
    result["source_commit"] = "b" * 40
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    with pytest.raises(TaskEvaluationSceneConfigurationRenderInputsError, match="preparation_result_invalid"):
        _consume(envelope, config, tmp_path / "must-not-exist")


def test_cutout_indices_must_match_measured_contributions_even_after_resigning(tmp_path):
    envelope, config, _, _, _ = _sam_case(tmp_path)
    record = envelope["sam31_exact_mask_inputs"]["segment_cutout_set"]
    path = Path(record["path"])
    receipt = json.loads(path.read_text())
    indices = receipt["shared_scene_union"]["outputs"]["deleted_source_indices"]
    indices_path = path.parent / indices["relative_path"]
    np.save(indices_path, np.array([1, 2], dtype=np.int64), allow_pickle=False)
    receipt["shared_scene_union"]["outputs"]["deleted_source_indices"] = _relative_record(path.parent, indices_path)
    _seal(path, receipt)
    envelope["sam31_exact_mask_inputs"]["segment_cutout_set"] = _absolute_record(path)
    config["sam31_exact_mask_evidence"]["segment_cutout_set_digest"] = receipt["receipt_digest"]
    with pytest.raises(TaskEvaluationSceneConfigurationRenderInputsError, match="cutout_indices_mismatch"):
        _consume(envelope, config, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_ai_review_uses_retained_agents_sdk_receipt_without_faking_human_acceptance(tmp_path, monkeypatch):
    envelope, config, _, _, _ = _sam_case(tmp_path, review_kind="ai", monkeypatch=monkeypatch)
    result = _consume(envelope, config, tmp_path / "result")
    assert result["source_object_masks"]["reviewer_kind"] == "ai"
    assert result["source_object_masks"]["human_review_completed"] is False
    config["sam31_review_kind"] = "human"
    with pytest.raises(TaskEvaluationSceneConfigurationRenderInputsError, match="reviewer_kind_mismatch"):
        _consume(envelope, config, tmp_path / "must-not-exist")


def test_removal_only_selection_needs_no_future_robot_or_policy_receipts(tmp_path):
    envelope, config, _, _, _ = _sam_case(tmp_path, removal_only=True)
    result = _consume(envelope, config, tmp_path / "result")
    assert result["publisher_instance_id"] == "115"
    assert result["derived_gaussian_cutout"]["retained_rows_byte_exact"] is True
    selection_path = Path(envelope["sam31_exact_mask_inputs"]["selection_inputs"]["path"])
    selections = json.loads(selection_path.read_text())
    task_path = Path(selections["tasks"][0]["task_freeze"]["path"])
    task = json.loads(task_path.read_text())
    assert task["schema_version"] == "public_scene_removal_task_selection.v1"
    assert "franka_placement_packet_digest" not in task["source_object"]
    assert "visibility_receipt_digest" not in task["source_object"]
