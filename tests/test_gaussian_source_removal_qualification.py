from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.dual_task_rehearsal_contract import REQUIRED_SELECTION_CRITERIA
from blueprint_pipeline.gaussian_source_removal_qualification import (
    GaussianSourceRemovalQualificationError,
    materialize_gaussian_source_removal_qualification,
)
from blueprint_pipeline.replacement_construction_bindings import (
    GAUSSIAN_REMOVAL_QUALIFICATION_SCHEMA_VERSION,
    MASK_SET_QUALIFICATION_SCHEMA_VERSION,
    materialize_replacement_construction_bindings,
)
from blueprint_pipeline.simready_graph_asset_static_qualification import (
    SCHEMA_VERSION as STATIC_GRAPH_ASSET_QUALIFICATION_SCHEMA_VERSION,
)
from blueprint_pipeline.simready_replacement_native_qualification import (
    NATIVE_IMPORT_PROBE_RESULT_SCHEMA_VERSION,
    materialize_simready_replacement_native_import_execution,
    materialize_simready_replacement_native_import_receipt,
    materialize_simready_replacement_native_qualification,
)


def _sha(value: int) -> str:
    return "sha256:" + f"{value:064x}"


def _seal(payload: dict, *, field: str) -> dict:
    payload = json.loads(json.dumps(payload))
    payload[field] = ""
    payload[field] = canonical_digest(payload, digest_field=field)
    return payload


def _write(path: Path, payload: dict, *, field: str) -> Path:
    sealed = _seal(payload, field=field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sealed, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_result(path: Path, payload: dict) -> Path:
    return _write(path, payload, field="result_digest")


def _scene_freeze(tmp_path: Path) -> Path:
    restrictions = {
        "redistribution": "none",
        "raw_private_upload": "forbidden",
        "derived_private_upload": "requires_authority",
        "retention": "bounded",
        "training": "forbidden",
        "publication": "derived_only",
    }
    return _write(
        tmp_path / "scene.json",
        {
            "schema_version": "dual_task_scene_freeze.v1",
            "selection_preregistration_digest": _sha(1),
            "learned_policy_outcomes_accessed": False,
            "selected_scene_id": "fixture_scene",
            "candidate_ledger": [
                {
                    "publisher_scene_id": "fixture_scene",
                    "decision": "selected",
                    "reason": "fixture selected without outcomes",
                    "method_outcomes_consulted": False,
                    "previously_used": False,
                }
            ],
            "source_components": {
                "interiorgs": {
                    "repository": "InteriorGS",
                    "revision": "fixture-revision",
                    "sha256": _sha(2),
                    "size_bytes": 100,
                    "license": "fixture-license",
                    "rights_admitted": True,
                    "restrictions": restrictions,
                },
                "sage_collision": {
                    "repository": "SAGE",
                    "revision": "fixture-revision",
                    "sha256": _sha(3),
                    "size_bytes": 200,
                    "license": "fixture-license",
                    "rights_admitted": True,
                    "restrictions": restrictions,
                },
            },
            "criterion_results": {
                criterion: {
                    "status": "observed_pass",
                    "evidence_digest": _sha(index + 10),
                    "remaining_gate": "none",
                }
                for index, criterion in enumerate(sorted(REQUIRED_SELECTION_CRITERIA))
            },
            "topology_survey_digest": _sha(30),
            "reconnaissance_render_digest": _sha(31),
            "scene_freeze_digest": "",
        },
        field="scene_freeze_digest",
    )


def _task_freeze(tmp_path: Path, scene_digest: str) -> Path:
    return _write(
        tmp_path / "task.json",
        {
            "schema_version": "dual_task_task_freeze.v1",
            "task_id": "fixture_rigid_task",
            "prompt": "relocate the fixture object",
            "task_kind": "rigid_object_manipulation",
            "scene_freeze_digest": scene_digest,
            "candidate_ids": ["pi05_droid", "groot_n17_droid"],
            "frozen_before_learned_policy_execution": True,
            "learned_policy_outcomes_accessed": False,
            "source_object": {
                "instance_id": "source_object_1",
                "semantic_label": "fixture object",
                "observed_bounds_world_m": {
                    "minimum": [0.0, 0.0, 0.0],
                    "maximum": [0.1, 0.1, 0.1],
                },
                "support_or_attachment_id": "support_1",
                "observed_pose_world": {
                    "position_world_m": [0.05, 0.05, 0.05],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "collision_identity_receipt_digest": _sha(40),
                "support_receipt_digest": _sha(41),
                "franka_placement_packet_digest": _sha(42),
                "visibility_receipt_digest": _sha(43),
            },
            "removal_plan": {
                "removal_id": "removal_1",
                "mask_set_id": "mask_set_1",
                "source_collider_prim_path": "/Root/source_object_1",
                "collider_deletion_id": "collider_delete_1",
                "replacement_asset_id": "replacement_asset_1",
                "replacement_qualification_id": "replacement_qualification_1",
            },
            "cameras": {
                "external": "external",
                "wrist": "wrist",
                "overview": "overview",
            },
            "overview_camera_policy_input": False,
            "overview_camera_deterministic_scoring_input": False,
            "execution_contract": {
                "control_frequency_hz": 20,
                "maximum_steps": 200,
                "settle_window_steps": 10,
                "seeds": [1],
                "canonical_scenario_cell_id": "canonical",
                "reset_state": {"robot": "reset"},
            },
            "deterministic_success_predicates": ["object_in_destination_after_release"],
            "failure_rungs": ["no_contact"],
            "target_configuration": {
                "kind": "pose_volume",
                "position_bounds_world_m": {
                    "minimum": [0.1, 0.1, 0.0],
                    "maximum": [0.2, 0.2, 0.1],
                },
                "orientation_reference_xyzw": [0.0, 0.0, 0.0, 1.0],
                "maximum_orientation_error_rad": 0.1,
                "support_id": "support_2",
                "release_required": True,
            },
            "articulation_graph": None,
            "task_freeze_digest": "",
        },
        field="task_freeze_digest",
    )


def _fixture_receipts(tmp_path: Path) -> dict[str, Path]:
    scene_path = _scene_freeze(tmp_path)
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    task_path = _task_freeze(tmp_path, scene["scene_freeze_digest"])
    task = json.loads(task_path.read_text(encoding="utf-8"))
    common = {
        "scene_id": scene["selected_scene_id"],
        "scene_freeze_digest": scene["scene_freeze_digest"],
        "task_id": task["task_id"],
        "task_freeze_digest": task["task_freeze_digest"],
        "source_object_instance_id": task["source_object"]["instance_id"],
        "removal_id": task["removal_plan"]["removal_id"],
        "mask_set_id": task["removal_plan"]["mask_set_id"],
    }
    mask_path = _write(
        tmp_path / "mask.json",
        {
            "schema_version": MASK_SET_QUALIFICATION_SCHEMA_VERSION,
            "status": "calibrated_mask_set_qualified",
            **common,
            "source_scene_sha256": scene["source_components"]["interiorgs"]["sha256"],
            "calibrated_masks_qualified": True,
            "receipt_digest": "",
        },
        field="receipt_digest",
    )
    ownership_path = _write(
        tmp_path / "ownership.json",
        {
            "schema_version": "adp009b_gaussian_excision_ownership_receipt.v1",
            "status": "three_way_ownership_materialized_heldout_not_evaluated",
            "freeze_digest": _sha(50),
            "ownership": {
                "source_gaussian_count": 12,
                "owned_count": 4,
                "retained_count": 8,
                "ambiguous_count": 0,
                "historical_obb_count": 4,
                "exhaustive": True,
                "pairwise_disjoint": True,
            },
            "heldout_cameras_accessed_for_classification": False,
            "replacement_usd_inserted": False,
            "receipt_digest": "",
        },
        field="receipt_digest",
    )
    ownership = json.loads(ownership_path.read_text(encoding="utf-8"))
    heldout_path = _write(
        tmp_path / "heldout.json",
        {
            "schema_version": "adp009b_gaussian_excision_heldout_audit.v1",
            "status": "heldout_gaussian_ownership_gate_passed",
            "freeze_digest": ownership["freeze_digest"],
            "ownership_receipt_digest": ownership["receipt_digest"],
            "heldout_gate_passed": True,
            "replacement_coverage_sweep_authorized": True,
            "receipt_digest": "",
        },
        field="receipt_digest",
    )
    join_path = _write(
        tmp_path / "join.json",
        {
            "schema_version": "articulated_excision_join.v1",
            "status": "join_admitted",
            "inpainting_policy": "inpainting_not_required",
            "suppression": {
                "mode": "deletion",
                "canonical_scan_modified": True,
                "reversible": False,
                "task_ids": [],
            },
            "bindings": {
                "ownership_receipt_digest": ownership["receipt_digest"],
                "owned_index_set_sha256": _sha(60),
                "retained_scene_ply_sha256": _sha(61),
            },
            "claim_boundary": {
                "gaussian_ownership_established": True,
            },
            "receipt_digest": "",
        },
        field="receipt_digest",
    )
    return {
        "scene": scene_path,
        "task": task_path,
        "mask": mask_path,
        "ownership": ownership_path,
        "heldout": heldout_path,
        "join": join_path,
    }


def test_materializer_derives_source_removal_qualification_from_upstream_receipts(
    tmp_path: Path,
) -> None:
    paths = _fixture_receipts(tmp_path)

    receipt = materialize_gaussian_source_removal_qualification(
        scene_freeze_path=paths["scene"],
        task_freeze_path=paths["task"],
        mask_set_receipt_path=paths["mask"],
        ownership_receipt_path=paths["ownership"],
        heldout_audit_receipt_path=paths["heldout"],
        excision_join_receipt_path=paths["join"],
        output_path=tmp_path / "qualified.json",
    )

    assert receipt["schema_version"] == GAUSSIAN_REMOVAL_QUALIFICATION_SCHEMA_VERSION
    assert receipt["status"] == "source_gaussian_removal_qualified"
    assert receipt["source_removal_qualified"] is True
    assert receipt["retained_records_byte_exact"] is True
    assert receipt["protected_geometry_deleted"] is False
    assert receipt["inpainting_policy"] == "inpainting_not_required"
    assert Path(receipt["upstream_evidence"]["heldout_audit"]["path"]).is_file()


def test_derived_source_removal_receipt_feeds_construction_bindings(
    tmp_path: Path,
) -> None:
    paths = _fixture_receipts(tmp_path)
    scene = json.loads(paths["scene"].read_text(encoding="utf-8"))
    task = json.loads(paths["task"].read_text(encoding="utf-8"))
    removal = task["removal_plan"]
    qualified_removal_path = tmp_path / "qualified_removal.json"
    materialize_gaussian_source_removal_qualification(
        scene_freeze_path=paths["scene"],
        task_freeze_path=paths["task"],
        mask_set_receipt_path=paths["mask"],
        ownership_receipt_path=paths["ownership"],
        heldout_audit_receipt_path=paths["heldout"],
        excision_join_receipt_path=paths["join"],
        output_path=qualified_removal_path,
    )
    collider_path = _write(
        tmp_path / "collider.json",
        {
            "schema_version": "source_collider_subtree_removal.v1",
            "status": "exact_source_collider_subtree_removed",
            "removal_id": removal["collider_deletion_id"],
            "sage_collision_usd_sha256": scene["source_components"]["sage_collision"][
                "sha256"
            ],
            "removed_scene_usd_sha256": _sha(70),
            "removed_prim_path": removal["source_collider_prim_path"],
            "source_bytes_unchanged": True,
            "unrelated_prim_inventory_unchanged": True,
            "remaining_target_collision_prim_count": 0,
            "removed_prim_count": 1,
            "replacement_inserted": False,
            "receipt_digest": "",
        },
        field="receipt_digest",
    )
    asset_sha256 = _sha(71)
    static_path = _write(
        tmp_path / "static_qualification.json",
        {
            "schema_version": STATIC_GRAPH_ASSET_QUALIFICATION_SCHEMA_VERSION,
            "status": "authored_structure_statically_qualified",
            "task_id": task["task_id"],
            "task_freeze_digest": task["task_freeze_digest"],
            "asset_id": removal["replacement_asset_id"],
            "replacement_usd": {
                "path": "/fixture/replacement.usda",
                "size_bytes": 123,
                "sha256": asset_sha256,
            },
            "authored_structure_statically_qualified": True,
            "structural_findings": [],
            "contract_blockers": ["native_simulator_import_unexecuted"],
            "receipt_digest": "",
        },
        field="receipt_digest",
    )
    native_probe_path = _write_result(
        tmp_path / "native_import_probe_result.json",
        {
            "schema_version": NATIVE_IMPORT_PROBE_RESULT_SCHEMA_VERSION,
            "status": "completed",
            "asset_id": removal["replacement_asset_id"],
            "replacement_asset_sha256": asset_sha256,
            "registered_static_qualification_digest": json.loads(
                static_path.read_text(encoding="utf-8")
            )["receipt_digest"],
            "native_isaac_executed": True,
            "native_simulator_import_qualified": True,
            "physical_equivalence_claimed": False,
            "candidate_policy_queried": False,
            "blockers": [],
            "simulator_import_identity": {
                "runtime": "fixture_native_import",
                "imported_prim_path": "/World/replacement_fixture",
            },
            "native_readback": {
                "asset_imported": True,
                "imported_prim_path": "/World/replacement_fixture",
            },
            "result_digest": "",
        },
    )
    native_execution_path = tmp_path / "native_import_execution.json"
    materialize_simready_replacement_native_import_execution(
        static_qualification_receipt_path=static_path,
        native_import_probe_result_path=native_probe_path,
        output_path=native_execution_path,
    )
    native_import_path = tmp_path / "native_import.json"
    materialize_simready_replacement_native_import_receipt(
        scene_freeze_receipt_path=paths["scene"],
        task_freeze_receipt_path=paths["task"],
        static_qualification_receipt_path=static_path,
        native_import_execution_receipt_path=native_execution_path,
        output_path=native_import_path,
    )
    replacement_path = tmp_path / "replacement.json"
    materialize_simready_replacement_native_qualification(
        scene_freeze_receipt_path=paths["scene"],
        task_freeze_receipt_path=paths["task"],
        static_qualification_receipt_path=static_path,
        native_import_receipt_path=native_import_path,
        output_path=replacement_path,
    )

    construction = materialize_replacement_construction_bindings(
        scene_freeze_receipt_path=paths["scene"],
        evidence_lanes=[
            {
                "task_freeze_receipt_path": str(paths["task"]),
                "mask_set_receipt_path": str(paths["mask"]),
                "gaussian_removal_receipt_path": str(qualified_removal_path),
                "source_collider_deletion_receipt_path": str(collider_path),
                "replacement_qualification_receipt_path": str(replacement_path),
            }
        ],
        output_path=tmp_path / "construction.json",
    )

    assert len(construction["bindings"]) == 1
    binding = construction["bindings"][0]
    assert binding["source_removal_qualified"] is True
    assert binding["source_removal_receipt_digest"] == json.loads(
        qualified_removal_path.read_text(encoding="utf-8")
    )["receipt_digest"]
    assert binding["evidence_receipts"]["gaussian_removal"]["path"] == str(
        qualified_removal_path.resolve()
    )


def test_materializer_rejects_failed_heldout_audit_before_construction_input(
    tmp_path: Path,
) -> None:
    paths = _fixture_receipts(tmp_path)
    heldout = json.loads(paths["heldout"].read_text(encoding="utf-8"))
    heldout["heldout_gate_passed"] = False
    heldout["receipt_digest"] = canonical_digest(heldout, digest_field="receipt_digest")
    paths["heldout"].write_text(
        json.dumps(heldout, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(GaussianSourceRemovalQualificationError) as excinfo:
        materialize_gaussian_source_removal_qualification(
            scene_freeze_path=paths["scene"],
            task_freeze_path=paths["task"],
            mask_set_receipt_path=paths["mask"],
            ownership_receipt_path=paths["ownership"],
            heldout_audit_receipt_path=paths["heldout"],
            excision_join_receipt_path=paths["join"],
            output_path=tmp_path / "qualified.json",
        )

    assert "gaussian_source_removal_heldout_not_passed" in excinfo.value.codes


def test_materializer_accepts_only_zero_residue_coverage_conditioned_deletion(
    tmp_path: Path,
) -> None:
    paths = _fixture_receipts(tmp_path)
    ownership = json.loads(paths["ownership"].read_text(encoding="utf-8"))
    ownership["source_standard_splat"] = {"sha256": _sha(90)}
    ownership["receipt_digest"] = canonical_digest(
        ownership, digest_field="receipt_digest"
    )
    paths["ownership"].write_text(
        json.dumps(ownership, sort_keys=True) + "\n", encoding="utf-8"
    )
    candidate_root = tmp_path / "coverage-candidate"
    candidate_root.mkdir()
    deleted_artifact = candidate_root / "deleted_source_indices.npy"
    retained_artifact = candidate_root / "retained_scene_gaussians.ply"
    deleted_artifact.write_bytes(b"fixture-deleted-indices")
    retained_artifact.write_bytes(b"fixture-retained-scene")

    def candidate_record(path: Path) -> dict[str, object]:
        return {
            "relative_path": path.relative_to(candidate_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    candidate_path = _write(
        candidate_root / "candidate.json",
        {
            "schema_version": "adp009b_ownership_coverage_cutout_candidate.v1",
            "status": "ownership_coverage_cutout_materialized_pending_actual_usd_source_layer_coverage",
            "source_standard_splat": {"sha256": _sha(90)},
            "source_ownership_receipt": {"receipt_digest": ownership["receipt_digest"]},
            "outputs": {
                "deleted_source_indices": candidate_record(deleted_artifact),
                "retained_scene_gaussians": candidate_record(retained_artifact),
            },
            "receipt_digest": "",
        },
        field="receipt_digest",
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    audit_path = tmp_path / "source-layer-audit.json"
    audit = {
        "schema_version": "adp009b_source_layer_replacement_coverage_audit.v1",
        "status": "source_layer_coverage_measured",
        "source_layer_splat_digest": _sha(90),
        "camera_ids": ["external"],
        "significant_alpha_threshold": 1.0 / 255.0,
        "coverage_margin_pixels": 1,
        "cells": [
            {
                "camera_id": "external",
                "uncovered_significant_pixel_count": 0,
                "largest_uncovered_component_pixels": 0,
                "uncovered_alpha_sum": 0.0,
                "uncovered_alpha_fraction": 0.0,
            }
        ],
        "manifest_digest": "",
    }
    audit["manifest_digest"] = canonical_digest(audit, digest_field="manifest_digest")
    audit_path.write_text(json.dumps(audit, sort_keys=True) + "\n", encoding="utf-8")
    coverage_path = _write(
        tmp_path / "source-layer-coverage.json",
        {
            "schema_version": "articulated_excision_coverage.v1",
            "status": "deleted_source_layer_replacement_coverage_qualified",
            "coverage_scope": "deleted_source_layer",
            "coverage_qualified": True,
            "all_deleted_source_contribution_occluded": True,
            "source_layer_splat_digest": _sha(90),
            "source_layer_coverage_audit": {
                "path": str(audit_path),
                "size_bytes": audit_path.stat().st_size,
                "sha256": "sha256:" + hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                "manifest_digest": audit["manifest_digest"],
            },
            "camera_ids": ["external"],
            "cells": [
                {
                    "camera_id": "external",
                    "residual_significant_pixels": 0,
                    "residual_max_connected_component_pixels": 0,
                    "outside_mask_changed_pixels": 0,
                }
            ],
            "receipt_digest": "",
        },
        field="receipt_digest",
    )
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    cutout_path = _write(
        tmp_path / "coverage-cutout.json",
        {
            "schema_version": "adp009b_coverage_conditioned_cutout.v1",
            "status": "coverage_conditioned_cutout_admitted",
            "bound_cutout_candidate_digest": candidate["receipt_digest"],
            "source_ownership_receipt_digest": ownership["receipt_digest"],
            "coverage_receipt_digest": coverage["receipt_digest"],
            "deleted_index_set_sha256": candidate["outputs"]["deleted_source_indices"][
                "sha256"
            ],
            "retained_scene_ply_sha256": candidate["outputs"][
                "retained_scene_gaussians"
            ]["sha256"],
            "coverage_qualified": True,
            "retained_rows_byte_exact": True,
            "learned_policy_outcomes_used": False,
            "factual_gaussian_ownership_claimed": False,
            "receipt_digest": "",
        },
        field="receipt_digest",
    )
    cutout = json.loads(cutout_path.read_text(encoding="utf-8"))
    join_path = _write(
        tmp_path / "coverage-join.json",
        {
            "schema_version": "articulated_excision_join.v1",
            "status": "join_admitted",
            "inpainting_policy": "inpainting_not_required",
            "suppression": {
                "mode": "deletion",
                "canonical_scan_modified": True,
                "reversible": False,
                "task_ids": [],
            },
            "bindings": {
                "ownership_receipt_digest": cutout["receipt_digest"],
                "source_ownership_receipt_digest": ownership["receipt_digest"],
                "coverage_receipt_digest": coverage["receipt_digest"],
                "deleted_index_set_sha256": cutout["deleted_index_set_sha256"],
                "retained_scene_ply_sha256": cutout["retained_scene_ply_sha256"],
            },
            "claim_boundary": {
                "gaussian_ownership_established": False,
                "visibility_after_replacement_is_the_criterion": True,
            },
            "receipt_digest": "",
        },
        field="receipt_digest",
    )

    receipt = materialize_gaussian_source_removal_qualification(
        scene_freeze_path=paths["scene"],
        task_freeze_path=paths["task"],
        mask_set_receipt_path=paths["mask"],
        ownership_receipt_path=paths["ownership"],
        heldout_audit_receipt_path=None,
        excision_join_receipt_path=join_path,
        coverage_conditioned_cutout_receipt_path=cutout_path,
        coverage_cutout_candidate_path=candidate_path,
        source_layer_coverage_receipt_path=coverage_path,
        output_path=tmp_path / "coverage-qualified.json",
    )

    assert receipt["source_removal_qualified"] is True
    assert receipt["ownership_proof_mode"] == "coverage_conditioned_visibility"
    assert receipt["claim_boundary"] == {
        "source_gaussian_removal_qualified": True,
        "protected_scene_geometry_deleted": False,
        "factual_gaussian_ownership_established": False,
        "coverage_conditioned_visibility_qualified": True,
        "replacement_native_import_qualified": False,
        "learned_policy_outcomes_used": False,
        "physical_equivalence": False,
    }
    assert receipt["coverage_conditioned_cutout_receipt_digest"] == cutout[
        "receipt_digest"
    ]

    coverage["all_deleted_source_contribution_occluded"] = False
    coverage["receipt_digest"] = canonical_digest(
        coverage, digest_field="receipt_digest"
    )
    coverage_path.write_text(json.dumps(coverage, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(GaussianSourceRemovalQualificationError) as excinfo:
        materialize_gaussian_source_removal_qualification(
            scene_freeze_path=paths["scene"],
            task_freeze_path=paths["task"],
            mask_set_receipt_path=paths["mask"],
            ownership_receipt_path=paths["ownership"],
            heldout_audit_receipt_path=None,
            excision_join_receipt_path=join_path,
            coverage_conditioned_cutout_receipt_path=cutout_path,
            coverage_cutout_candidate_path=candidate_path,
            source_layer_coverage_receipt_path=coverage_path,
            output_path=tmp_path / "blocked-coverage-qualified.json",
        )
    assert "gaussian_source_removal_coverage_conditioned_coverage_mismatch" in excinfo.value.codes
