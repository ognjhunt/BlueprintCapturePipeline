from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.fresh_scene_paired_target_preparation import (
    STAGE_CONTRACTS,
    materialize_fresh_scene_preparation_status,
)

from tests.test_public_scene_calibrated_object_masks import _task


def _tasks(root: Path) -> list[Path]:
    paths = []
    for slot, task_id in enumerate(("task_a", "task_b"), start=1):
        path = root / f"{task_id}.json"
        write_json(path, _task(task_id, slot))
        paths.append(path)
    return paths


def _artifact(root: Path, contract: dict, suffix: str) -> Path:
    value = {
        "schema_version": contract["schemas"][0],
        "status": contract["accepted_statuses"][0] if contract["accepted_statuses"] else None,
    }
    digest_fields = contract["digest_fields"]
    if digest_fields:
        value[digest_fields[0]] = canonical_digest(value, digest_field=digest_fields[0])
    path = root / f"{contract['stage_id']}-{suffix}.json"
    write_json(path, value)
    return path


def test_fresh_scene_reports_calibrated_views_before_sam(tmp_path: Path) -> None:
    result = materialize_fresh_scene_preparation_status(
        task_freeze_paths=_tasks(tmp_path),
        artifacts={},
        output_path=tmp_path / "status.json",
    )

    assert result["first_blocker"] == "fresh_scene_calibrated_scene_views_missing"
    assert result["next_required_stage"] == "calibrated_scene_views"
    assert result["stages"][0]["producer"] == "public_scene_inpainting_inputs"
    assert result["stages"][1]["status"] == "waiting_on_upstream"
    assert "artifixer3d_calibrated_preflight_missing" not in json.dumps(result)


def test_fresh_scene_advances_in_order_and_supports_per_task_rows(tmp_path: Path) -> None:
    task_paths = _tasks(tmp_path)
    artifacts: dict[str, object] = {}
    contribution_index = next(
        index
        for index, contract in enumerate(STAGE_CONTRACTS)
        if contract["stage_id"] == "gaussian_contribution_evidence"
    )
    for contract in STAGE_CONTRACTS[:contribution_index]:
        if contract["cardinality"] == "per_task":
            artifacts[contract["stage_id"]] = {
                task_id: str(_artifact(tmp_path, contract, task_id))
                for task_id in ("task_a", "task_b")
            }
        else:
            artifacts[contract["stage_id"]] = str(
                _artifact(tmp_path, contract, "shared")
            )
    result = materialize_fresh_scene_preparation_status(
        task_freeze_paths=task_paths,
        artifacts=artifacts,
        output_path=tmp_path / "status.json",
    )

    assert all(
        row["status"] == "completed"
        for row in result["stages"][:contribution_index]
    )
    assert result["first_blocker"] == "fresh_scene_gaussian_contribution_evidence_missing"
    assert result["next_required_stage"] == "gaussian_contribution_evidence"


def test_fresh_scene_names_visual_track_review_before_masks(tmp_path: Path) -> None:
    task_paths = _tasks(tmp_path)
    artifacts: dict[str, object] = {}
    for contract in STAGE_CONTRACTS:
        if contract["stage_id"] == "sam31_track_selection_review":
            break
        if contract["cardinality"] == "per_task":
            artifacts[contract["stage_id"]] = {
                task_id: str(_artifact(tmp_path, contract, task_id))
                for task_id in ("task_a", "task_b")
            }
        else:
            artifacts[contract["stage_id"]] = str(
                _artifact(tmp_path, contract, "shared")
            )
    result = materialize_fresh_scene_preparation_status(
        task_freeze_paths=task_paths,
        artifacts=artifacts,
        output_path=tmp_path / "status.json",
    )

    assert result["first_blocker"] == "fresh_scene_sam31_track_selection_review_missing"
    assert result["next_required_stage"] == "sam31_track_selection_review"


def test_fresh_scene_accepts_ai_review_schema_only_with_ai_status(tmp_path: Path) -> None:
    task_paths = _tasks(tmp_path)
    artifacts: dict[str, object] = {}
    review_contract = next(
        row for row in STAGE_CONTRACTS if row["stage_id"] == "sam31_track_selection_review"
    )
    for contract in STAGE_CONTRACTS:
        if contract["stage_id"] == "calibrated_object_masks":
            break
        if contract["cardinality"] == "per_task":
            artifacts[contract["stage_id"]] = {
                task_id: str(_artifact(tmp_path, contract, task_id))
                for task_id in ("task_a", "task_b")
            }
        elif contract["stage_id"] == "sam31_track_selection_review":
            value = {
                "schema_version": review_contract["schemas"][1],
                "status": review_contract["accepted_statuses"][1],
                "receipt_digest": "",
            }
            value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
            path = tmp_path / "ai-review.json"
            write_json(path, value)
            artifacts[contract["stage_id"]] = str(path)
        else:
            artifacts[contract["stage_id"]] = str(
                _artifact(tmp_path, contract, "shared")
            )

    accepted = materialize_fresh_scene_preparation_status(
        task_freeze_paths=task_paths,
        artifacts=artifacts,
        output_path=tmp_path / "accepted-status.json",
    )
    review_row = next(
        row for row in accepted["stages"] if row["stage_id"] == "sam31_track_selection_review"
    )
    assert review_row["status"] == "completed"

    review_path = Path(str(artifacts["sam31_track_selection_review"]))
    mismatched = json.loads(review_path.read_text(encoding="utf-8"))
    mismatched["status"] = "selected_tracks_human_review_accepted"
    mismatched["receipt_digest"] = canonical_digest(mismatched, digest_field="receipt_digest")
    write_json(review_path, mismatched)
    rejected = materialize_fresh_scene_preparation_status(
        task_freeze_paths=task_paths,
        artifacts=artifacts,
        output_path=tmp_path / "rejected-status.json",
    )
    assert (
        rejected["first_blocker"]
        == "fresh_scene_stage_artifact_invalid:sam31_track_selection_review"
    )


def test_fresh_scene_requires_registry_selected_edit_packet_before_paid_edit(
    tmp_path: Path,
) -> None:
    task_paths = _tasks(tmp_path)
    artifacts: dict[str, object] = {}
    for contract in STAGE_CONTRACTS:
        if contract["stage_id"] == "semantic_teacher_edit_packet":
            break
        if contract["cardinality"] == "per_task":
            artifacts[contract["stage_id"]] = {
                task_id: str(_artifact(tmp_path, contract, task_id))
                for task_id in ("task_a", "task_b")
            }
        else:
            artifacts[contract["stage_id"]] = str(
                _artifact(tmp_path, contract, "shared")
            )
    result = materialize_fresh_scene_preparation_status(
        task_freeze_paths=task_paths,
        artifacts=artifacts,
        output_path=tmp_path / "status.json",
    )

    assert result["first_blocker"] == "fresh_scene_semantic_teacher_edit_packet_missing"
    assert result["next_required_stage"] == "semantic_teacher_edit_packet"
    packet_stage = next(
        row for row in result["stages"] if row["stage_id"] == "semantic_teacher_edit_packet"
    )
    assert packet_stage["backend"] == "registry-selected rights-admitted image editor"


def test_complete_inventory_reaches_visual_review_without_claim_upgrade(tmp_path: Path) -> None:
    task_paths = _tasks(tmp_path)
    artifacts: dict[str, object] = {}
    for contract in STAGE_CONTRACTS:
        if contract["cardinality"] == "per_task":
            artifacts[contract["stage_id"]] = {
                task_id: str(_artifact(tmp_path, contract, task_id))
                for task_id in ("task_a", "task_b")
            }
        else:
            artifacts[contract["stage_id"]] = str(
                _artifact(tmp_path, contract, "shared")
            )
    result = materialize_fresh_scene_preparation_status(
        task_freeze_paths=task_paths,
        artifacts=artifacts,
        output_path=tmp_path / "status.json",
    )

    assert result["status"] == "ready_for_visual_qualification"
    assert result["first_blocker"] is None
    assert result["production_contract"]["simulator_outputs_are_physical_evidence"] is False
    assert result["production_contract"]["automatic_paid_retry_authorized"] is False
