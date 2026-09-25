from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from blueprint_pipeline.live_pipeline_result_artifact_resolution import (
    resolve_live_pipeline_result_artifact,
)
from blueprint_pipeline.native_g1_private_review_delivery import _stage_review_artifacts
from blueprint_pipeline.task_evaluation_result_delivery import TaskEvaluationResultDeliveryError


def _artifact(root: Path, relative: str, payload: bytes) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "relative_path": relative,
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _review(source: Path) -> dict[str, object]:
    episodes = []
    for index in range(4):
        pair = "manipulation_pair" if index < 2 else "movement_pair"
        prefix = f"{pair}/candidate-{index}/episode"
        episodes.append({
            "frame_manifest": _artifact(
                source, f"{prefix}/frames.json", f"frames-{index}".encode()
            ),
            "review_videos": {
                camera: _artifact(
                    source, f"{prefix}/{camera}.mp4", f"{camera}-{index}".encode()
                ) for camera in ("head", "overview")
            },
        })
    return {
        "status": "verified_private_development_review",
        "claim_ceiling": "development_only",
        "public_redistribution_authorized": False,
        "physical_outcome_claimed": False,
        "review_digest": "sha256:" + "a" * 64,
        "episodes": episodes,
    }


def test_g1_private_media_uses_live_authenticated_artifact_resolver(tmp_path: Path) -> None:
    source = tmp_path / "immutable_execution"
    source.mkdir()
    review = _review(source)
    result_root = tmp_path / "policy-results"
    result_root.mkdir()
    run_id = "g1-841757-test"
    registered = _stage_review_artifacts(
        review=review, source_root=source, result_root=result_root, run_id=run_id
    )
    assert registered["status"] == "registered_private_development_review"
    assert registered["artifact_count"] == 12
    assert registered["public_redistribution_authorized"] is False
    registry_root = result_root / f"{run_id}-activation"
    assert registry_root.is_dir()
    for episode in review["episodes"]:
        for role, artifact in (
            ("g1_frame_manifest", episode["frame_manifest"]),
            ("g1_head_video", episode["review_videos"]["head"]),
            ("g1_overview_video", episode["review_videos"]["overview"]),
        ):
            artifact_id = hashlib.sha256(
                f"{role}\0{artifact['relative_path']}\0{artifact['sha256']}".encode()
            ).hexdigest()[:32]
            path, record = resolve_live_pipeline_result_artifact(
                legacy_state_root=tmp_path / "legacy",
                policy_canary_result_root=result_root,
                run_id=run_id,
                artifact_id=artifact_id,
            )
            assert path.read_bytes() == (source / artifact["relative_path"]).read_bytes()
            assert record["sha256"] == artifact["sha256"]
            assert record["content_type"] == (
                "application/json" if role == "g1_frame_manifest" else "video/mp4"
            )
    with pytest.raises(ValueError, match="already_registered"):
        _stage_review_artifacts(
            review=review, source_root=source, result_root=result_root, run_id=run_id
        )


def test_g1_private_media_rejects_path_escape_and_changed_bytes(tmp_path: Path) -> None:
    source = tmp_path / "immutable_execution"
    source.mkdir()
    review = _review(source)
    result_root = tmp_path / "policy-results"
    result_root.mkdir()
    review["episodes"][0]["frame_manifest"]["relative_path"] = "../outside.json"
    with pytest.raises(ValueError, match="artifact_path_invalid"):
        _stage_review_artifacts(
            review=review, source_root=source, result_root=result_root, run_id="g1-escape"
        )
    assert not (result_root / "g1-escape-activation").exists()

    review = _review(source)
    first = source / review["episodes"][0]["review_videos"]["head"]["relative_path"]
    first.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact_identity_invalid"):
        _stage_review_artifacts(
            review=review, source_root=source, result_root=result_root, run_id="g1-tamper"
        )
    assert not (result_root / "g1-tamper-activation").exists()


def test_g1_private_media_resolver_detects_post_registration_tamper(tmp_path: Path) -> None:
    source = tmp_path / "immutable_execution"
    source.mkdir()
    review = _review(source)
    result_root = tmp_path / "policy-results"
    result_root.mkdir()
    run_id = "g1-tamper-read"
    _stage_review_artifacts(
        review=review, source_root=source, result_root=result_root, run_id=run_id
    )
    artifact = review["episodes"][0]["frame_manifest"]
    artifact_id = hashlib.sha256(
        f"g1_frame_manifest\0{artifact['relative_path']}\0{artifact['sha256']}".encode()
    ).hexdigest()[:32]
    path = result_root / f"{run_id}-activation/evidence" / artifact["relative_path"]
    path.write_bytes(b"changed")
    with pytest.raises(TaskEvaluationResultDeliveryError, match="reverification_failed"):
        resolve_live_pipeline_result_artifact(
            legacy_state_root=tmp_path / "legacy",
            policy_canary_result_root=result_root,
            run_id=run_id,
            artifact_id=artifact_id,
        )
