from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_result_delivery import (
    TaskEvaluationResultDeliveryError,
    materialize_task_evaluation_result_delivery,
    resolve_task_evaluation_result_artifact,
)


def _write(root: Path, relative_path: str, content: bytes) -> dict[str, object]:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "relative_path": relative_path,
        "sha256": "sha256:" + hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def _evidence_index(root: Path) -> Path:
    receipt = _write(
        root,
        "episode/receipt.json",
        (
            b'{"candidate_actions":[[0.1]],"commanded_actions":'
            b'[{"observed_after_rad":[0.1],"observed_before_rad":[0.0]}],'
            b'"terminal":true}\n'
        ),
    )
    manifest = _write(root, "episode/multicamera.json", b'{"frames":[]}\n')
    lossless = _write(root, "episode/external-000.png", b"lossless-rgb")
    exact = _write(root, "episode/policy-input-000.png", b"exact-policy-input")
    lossless["png_sha256"] = lossless.pop("sha256")
    exact["png_sha256"] = exact.pop("sha256")
    videos = {
        camera: _write(root, f"episode/{camera}.mp4", f"{camera}-video".encode())
        for camera in ("external", "wrist", "overview")
    }
    payload: dict[str, object] = {
        "schema_version": "adp_manipulation_episode_evidence_index.v1",
        "run_identity": {"run_id": "run-delivery-1"},
        "episodes": [
            {
                "episode_id": "episode-1",
                "episode_kind": "learned_candidate",
                "subject_id": "frozen-policy-a",
                "receipt": {**receipt, "receipt_digest": "sha256:" + "1" * 64},
                "score": {
                    "status": "complete",
                    "outcome": "task_complete",
                    "task_succeeded": True,
                    "grader_authority": "deterministic_simulator_state",
                },
                "frame_manifest": {**manifest, "frame_manifest_digest": "sha256:" + "2" * 64},
                "lossless_camera_frames": [lossless],
                "exact_policy_input_frames": [exact],
                "videos": videos,
            }
        ],
        "episode_count": 1,
        "required_camera_ids": ["external", "wrist", "overview"],
        "overview_is_review_only": True,
        "scores_are_deterministic_simulator_state": True,
        "review_videos_are_not_physical_truth": True,
        "typed_abstention": None,
        "supporting_evidence": [],
        "index_digest": "",
    }
    payload["index_digest"] = canonical_digest(payload, digest_field="index_digest")
    path = root / "episode_evidence_index.v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _envelope() -> dict[str, str]:
    return {"decision_envelope_digest": "sha256:" + "a" * 64}


def test_materializes_sealed_review_and_full_evidence_packages(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "run-delivery-1"
    evidence_root = run_root / "artifacts"
    index_path = _evidence_index(evidence_root)

    delivery = materialize_task_evaluation_result_delivery(
        run_root=run_root,
        run_id="run-delivery-1",
        state="decided",
        decision_envelope=_envelope(),
        episode_evidence_index_path=index_path,
    )

    assert delivery["status"] == "ready"
    assert [row["status"] for row in delivery["stages"]] == [
        "complete",
        "complete",
        "complete",
        "complete",
        "ready",
    ]
    assert delivery["summary"] == {
        "episode_count": 1,
        "learned_candidate_episode_count": 1,
        "control_episode_count": 0,
        "successful_episode_count": 1,
    }
    packages = {
        row["role"]: row
        for row in delivery["artifacts"]
        if row["content_type"] == "application/zip"
    }
    assert set(packages) == {"review_package", "full_evidence_package"}
    full_path, full_record = resolve_task_evaluation_result_artifact(
        run_root=run_root,
        run_id="run-delivery-1",
        artifact_id=packages["full_evidence_package"]["artifact_id"],
    )
    assert full_record["sha256"] == packages["full_evidence_package"]["sha256"]
    with zipfile.ZipFile(full_path) as archive:
        names = set(archive.namelist())
        packaged_receipt = json.loads(archive.read("data/episode/receipt.json"))
    assert "summary.json" in names
    assert "scenarios.csv" in names
    assert "manifest-sha256.txt" in names
    assert "data/episode/policy-input-000.png" in names
    assert "data/episode/overview.mp4" in names
    assert packaged_receipt["candidate_actions"] == [[0.1]]
    assert packaged_receipt["commanded_actions"][0]["observed_after_rad"] == [0.1]


def test_blocks_publication_when_index_has_not_arrived(tmp_path: Path) -> None:
    delivery = materialize_task_evaluation_result_delivery(
        run_root=tmp_path / "runs" / "run-delivery-1",
        run_id="run-delivery-1",
        state="abstained",
        decision_envelope=_envelope(),
    )
    assert delivery["status"] == "blocked"
    assert delivery["blockers"] == ["episode_evidence_index_missing"]
    assert delivery["stages"][0] == {"stage": "validate", "status": "blocked"}


def test_tampered_exact_policy_input_fails_closed(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "run-delivery-1"
    evidence_root = run_root / "artifacts"
    index_path = _evidence_index(evidence_root)
    (evidence_root / "episode" / "policy-input-000.png").write_bytes(b"tampered")

    with pytest.raises(
        TaskEvaluationResultDeliveryError,
        match="delivery_artifact_(digest|size)_mismatch",
    ):
        materialize_task_evaluation_result_delivery(
            run_root=run_root,
            run_id="run-delivery-1",
            state="decided",
            decision_envelope=_envelope(),
            episode_evidence_index_path=index_path,
        )
