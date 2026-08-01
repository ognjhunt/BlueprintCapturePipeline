from __future__ import annotations

import hashlib
import json
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_processed_task_testbed import (
    compile_public_processed_task_testbed_proxy,
)


def _file_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    source_digest = "sha256:" + "a" * 64
    root = tmp_path / "processed"
    frames = []
    observations = []
    for index, frame_id in enumerate(
        ("long:frame_00100", "long:frame_00120", "long:frame_00300"),
        start=1,
    ):
        filename = frame_id.replace("long:", "") + ".jpg"
        relative = f"candidate_dataset/training/{filename}"
        payload = f"fixture-image-{index}".encode()
        image_path = root / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(payload)
        digest = _file_digest(payload)
        frames.append(
            {
                "frame_id": frame_id,
                "candidate_relative_path": relative,
                "frame_digest": digest,
                "split": "training",
            }
        )
        observations.append(
            {
                "observation_id": frame_id,
                "image_digest": digest,
                "image_relative_path": relative,
                "split": "training",
                "camera": {"status": "fixture"},
            }
        )
    candidate = {
        "schema_version": "processed_candidate_dataset_manifest.v1",
        "capture_digest": source_digest,
        "frames": frames,
        "heldout_pixels_included": False,
    }
    candidate["candidate_dataset_digest"] = canonical_digest(
        candidate, digest_field="candidate_dataset_digest"
    )
    camera = {
        "schema_version": "processed_camera_observation_manifest.v1",
        "source_capture_digest": source_digest,
        "calibration_status": "dataset_provided_not_independently_verified",
        "observations": observations,
        "hidden_heldout_pixels_included": False,
    }
    camera["camera_observation_digest"] = canonical_digest(
        camera, digest_field="camera_observation_digest"
    )
    processed = {
        "schema_version": "processed_observation_dataset_manifest.v1",
        "source_capture_digest": source_digest,
        "capture_authority_profile": "public_processed_rgbd_pose_sequence",
        "coordinate_frame_declaration": {
            "source": "fixture",
            "world_up": "not_independently_verified",
            "metric_scale": "dataset_declared_not_independently_verified",
        },
    }
    processed["dataset_manifest_digest"] = canonical_digest(
        processed, digest_field="dataset_manifest_digest"
    )
    processed_path = root / "reconstruction_dataset_manifest.json"
    candidate_path = root / "candidate_dataset_manifest.json"
    camera_path = root / "candidate_camera_observation_manifest.json"
    _write_json(processed_path, processed)
    _write_json(candidate_path, candidate)
    _write_json(camera_path, camera)
    ply_payload = b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n"
    ply_path = tmp_path / "polycam_pointcloud.ply"
    ply_path.write_bytes(ply_payload)
    appearance = {
        "schema_version": "public_indoor_proxy_replay.v1",
        "source_bundle": {"digest": source_digest},
        "source_artifact": {"digest": _file_digest(ply_payload)},
    }
    appearance_path = tmp_path / "public_indoor_proxy_replay.json"
    _write_json(appearance_path, appearance)
    return {
        "processed": processed_path,
        "candidate": candidate_path,
        "camera": camera_path,
        "appearance": appearance_path,
        "ply": ply_path,
    }


def test_public_processed_walkthrough_compiles_partial_decision_without_raw_upgrade(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    kwargs = {
        "processed_dataset_manifest_path": fixture["processed"],
        "candidate_dataset_manifest_path": fixture["candidate"],
        "camera_observation_manifest_path": fixture["camera"],
        "appearance_proxy_summary_path": fixture["appearance"],
        "appearance_ply_path": fixture["ply"],
        "output_root": tmp_path / "output",
        "operator_identity": "operator:fixture",
        "source_commit_sha": "b" * 40,
        "timestamp": "2026-08-01T12:00:00-05:00",
    }

    first = compile_public_processed_task_testbed_proxy(**kwargs)
    second = compile_public_processed_task_testbed_proxy(**kwargs)

    assert first == second
    assert first["qa_status"] == "accepted"
    assert first["task_approval_state_before_decision"] == "task_approval_required"
    assert first["task_intent_source"] == "operator_approved_candidate"
    assert first["overall_outcome"] == "partial_decision"
    verdicts = {row["claim_id"]: row["verdict"] for row in first["per_claim_verdicts"]}
    assert verdicts == {
        "analytic-reach": "abstention",
        "comparative-policy-ranking": "abstention",
        "modeled-collision": "abstention",
        "physical-task-success": "abstention",
        "processed-visibility": "supported",
    }
    assert first["claim_flags"] == {
        "processed_captured_observation": True,
        "raw_capture_authority": False,
        "decoded_video_timing": False,
        "metric_scale_verified": False,
        "collision_geometry": False,
        "physics": False,
        "physical_task_success": False,
        "deployment_readiness": False,
        "safety_certification": False,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    assert first["cost_usd"] == 0.0
    assert first["physical_evidence_requests"][0]["robot_run_initiated"] is False

    artifact_root = kwargs["output_root"] / "public_processed_task_testbed_proxy"
    testbed = json.loads((artifact_root / "testbed.json").read_text())
    assert all(
        row["evidence_id"] != "raw_capture" for row in testbed["evidence_inventory"]
    )
    processed_source = next(
        row
        for row in testbed["evidence_inventory"]
        if row["evidence_id"] == "processed_capture_observations"
    )
    assert processed_source["raw_capture_authority"] is False
    decision = json.loads((artifact_root / "decision_envelope.json").read_text())
    assert decision["deployment_approval"] is False
    assert decision["uncertainty"]["ranking_science_boundary"] == (
        "thesis_not_supported"
    )
