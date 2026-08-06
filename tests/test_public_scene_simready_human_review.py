from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_simready_human_review import (
    SimReadyHumanReviewError,
    materialize_human_review,
)


def _receipt(path: Path, value: dict[str, object]) -> str:
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(value["receipt_digest"])


def _fixture(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    evidence = tmp_path / "evidence"
    repo.mkdir()
    evidence.mkdir()
    replacement_path = repo / "replacement.json"
    replacement_digest = _receipt(
        replacement_path, {"status": "composed_static_candidate"}
    )
    visual_path = evidence / "visual.json"
    visual_digest = _receipt(
        visual_path,
        {
            "status": "rendered_visual_review_candidate",
            "replacement_receipt_digest": replacement_digest,
        },
    )
    match_path = evidence / "match.json"
    match_digest = _receipt(
        match_path,
        {
            "status": "diagnosed_match_candidate",
            "visual_review_receipt_digest": visual_digest,
            "aggregate": {
                "camera_count": 8,
                "median_silhouette_iou": 0.93,
                "median_delta_e76": 10.4,
                "projected_scale_and_pose_gate_passed": True,
                "colour_appearance_gate_passed": True,
            },
        },
    )
    request = {
        "schema_version": "adp009b_simready_human_visual_review_request.v1",
        "decision": "approve_for_native_validation",
        "reviewer_role": "project_owner",
        "approval_statement": "close enough; approved",
        "artifacts": {
            "replacement_receipt_path": "replacement.json",
            "replacement_receipt_digest": replacement_digest,
            "visual_review_relative_path": "visual.json",
            "visual_review_receipt_digest": visual_digest,
            "match_review_relative_path": "match.json",
            "match_review_receipt_digest": match_digest,
        },
    }
    request_path = repo / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    return {
        "repo": repo,
        "evidence": evidence,
        "request": request_path,
        "match": match_path,
        "output": repo / "receipt.json",
    }


def _run(paths: dict[str, Path]) -> dict[str, object]:
    return materialize_human_review(
        request_path=paths["request"],
        repo_root=paths["repo"],
        evidence_root=paths["evidence"],
        output_path=paths["output"],
    )


def test_human_review_binds_approval_without_technical_admission(tmp_path: Path) -> None:
    receipt = _run(_fixture(tmp_path))

    assert receipt["status"] == "human_accepted_for_native_validation"
    assert receipt["technical_admission"] is False
    assert receipt["dynamic_contact_proven"] is False
    assert "native_ovphysx_drop_contact_settle_missing" in receipt["blockers"]


def test_human_review_rejects_changed_review_bytes(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["match"].write_text("{}", encoding="utf-8")
    with pytest.raises(SimReadyHumanReviewError, match="match_review_receipt_digest_invalid"):
        _run(paths)


def test_human_review_cannot_assert_technical_admission(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    request = json.loads(paths["request"].read_text(encoding="utf-8"))
    request["qualified"] = True
    paths["request"].write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(
        SimReadyHumanReviewError, match="human_review_cannot_assert_technical_admission"
    ):
        _run(paths)
