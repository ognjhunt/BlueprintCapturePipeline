from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_aura_visual_abstention import (
    AuraVisualAbstentionError,
    materialize_aura_visual_abstention,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    execution = {
        "schema_version": "adp009b_aurafusion360_execution_receipt.v1",
        "status": "executed_candidate",
        "scene": {
            "publisher_scene_id": "840796",
            "target_instance_id": "ins123",
            "target_semantic_label": "refrigerator",
            "camera_count": 2,
        },
        "execution": {
            "final_frames": [
                {"camera_id": "a", "sha256": "sha256:" + "a" * 64},
                {"camera_id": "b", "sha256": "sha256:" + "b" * 64},
            ]
        },
        "quality": {"intermediate_stage_artifacts_retained": False},
        "claim_boundary": {"successful_inpainting_admitted": False},
        "receipt_digest": "",
    }
    execution["receipt_digest"] = canonical_digest(
        execution, digest_field="receipt_digest"
    )
    locality = {
        "schema_version": "public_scene_inpainting_locality_measurement.v1",
        "status": "measured_no_admission_effect",
        "scene": {"publisher_scene_id": "840796", "target_instance_id": "ins123"},
        "rows": [
            {"camera_id": "a", "after_sha256": "sha256:" + "a" * 64},
            {"camera_id": "b", "after_sha256": "sha256:" + "b" * 64},
        ],
        "aggregate": {"view_count": 2, "mean_outside_mask_psnr_db": 20.0},
        "quality_pass_claimed": False,
        "thresholds_frozen_before_evaluation": False,
        "locality_measurement_digest": "",
    }
    locality["locality_measurement_digest"] = canonical_digest(
        locality, digest_field="locality_measurement_digest"
    )
    request = {
        "schema_version": "adp009b_aura_visual_abstention_request.v1",
        "decision": "reject_visual_candidate",
        "reviewer_role": "evidence_operator",
        "aura_execution_receipt_path": "execution.json",
        "locality_measurement_path": "locality.json",
        "observed_artifact_codes": [
            "outside_mask_scene_damage",
            "semantic_hallucination_in_removed_volume",
        ],
    }
    _write(repo / "execution.json", execution)
    _write(repo / "request.json", request)
    _write(data / "locality.json", locality)
    return repo, data, repo / "request.json"


def test_visual_abstention_binds_frames_and_cannot_admit_result(tmp_path: Path) -> None:
    repo, data, request = _fixture(tmp_path)

    receipt = materialize_aura_visual_abstention(
        request_path=request,
        repo_root=repo,
        data_root=data,
        output_path=repo / "receipt.json",
    )

    assert receipt["status"] == "abstained_visual_artifact_rejection"
    assert receipt["quality_pass_claimed"] is False
    assert receipt["successful_inpainting_admitted"] is False
    assert receipt["claim_ceiling"] == "rejected_visual_candidate_only"
    assert receipt["failure_localization"] == (
        "stage_localization_missing_intermediate_inpaint_evidence"
    )


def test_visual_abstention_rejects_acceptance_and_broken_frame_join(
    tmp_path: Path,
) -> None:
    repo, data, request = _fixture(tmp_path)
    value = json.loads(request.read_text())
    value["decision"] = "accept_visual_candidate"
    _write(request, value)
    with pytest.raises(AuraVisualAbstentionError, match="rejection_decision_required"):
        materialize_aura_visual_abstention(
            request_path=request,
            repo_root=repo,
            data_root=data,
            output_path=repo / "receipt.json",
        )

    value["decision"] = "reject_visual_candidate"
    _write(request, value)
    locality = json.loads((data / "locality.json").read_text())
    locality["rows"][0]["after_sha256"] = "sha256:" + "c" * 64
    locality["locality_measurement_digest"] = canonical_digest(
        locality, digest_field="locality_measurement_digest"
    )
    _write(data / "locality.json", locality)
    with pytest.raises(AuraVisualAbstentionError, match="locality_frame_join_invalid"):
        materialize_aura_visual_abstention(
            request_path=request,
            repo_root=repo,
            data_root=data,
            output_path=repo / "receipt.json",
        )
