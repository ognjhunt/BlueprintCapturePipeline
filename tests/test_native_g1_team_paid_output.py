"""A team score is deliverable only with its exact policy inputs and media."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)
from blueprint_pipeline.native_g1_team_paid_output import verify_g1_team_paid_output
from blueprint_pipeline.native_g1_team_policy_execution_packet import SCHEMA as PACKET_SCHEMA
from blueprint_pipeline.native_g1_team_policy_worker import (
    FILENAME as WORKER_FILENAME,
    PRECLOSE_FILENAME,
    PRECLOSE_SCHEMA,
    SCHEMA as WORKER_SCHEMA,
)
from blueprint_pipeline.native_g1_team_scored_scene_episode import RESULT_FILENAME
from tests.test_native_g1_team_supervised_episode import _run


def _evidence(tmp_path: Path, monkeypatch):
    supervised, produced, _ = _run(tmp_path, monkeypatch)
    assert supervised["status"] == "completed_development_only"
    root = tmp_path / "paid-output"
    root.mkdir()
    shutil.copytree(produced, root / "episode")
    episode = json.loads((root / "episode/episode" / RESULT_FILENAME).read_text())
    packet = {
        "schema_version": PACKET_SCHEMA,
        "request": {
            "run_id": "team-g1-book-1",
            "owner": episode["owner"],
            "objective_id": episode["objective_id"],
            "policy_profile": {
                "profile_digest": episode["profile_digest"],
                "delivery": {"mode": episode["delivery_mode"]},
            },
        },
        "operator_approval": {"approval_digest": supervised["operator_approval_digest"]},
        "trusted_setup": {
            "scene_id": episode["scene_id"],
            "task_id": episode["task_id"],
            "setup_digest": episode["source_setup_digest"],
            "source_packet_receipt_digest": episode["source_packet_receipt_digest"],
        },
    }
    packet["intent_id"] = "g1-team-policy-" + cross_runtime_canonical_digest({
        "owner": packet["request"]["owner"], "run_id": packet["request"]["run_id"],
    }).removeprefix("sha256:")
    packet["packet_digest"] = cross_runtime_canonical_digest(packet, digest_field="packet_digest")
    scene_receipt_digest = "sha256:" + "f" * 64
    core = {
        "claim_ceiling": "development_only",
        "execution_packet_digest": packet["packet_digest"],
        "operator_approval_digest": supervised["operator_approval_digest"],
        "scene_packet_receipt_digest": scene_receipt_digest,
        "profile_digest": episode["profile_digest"],
        "objective_id": episode["objective_id"],
        "delivery_mode": episode["delivery_mode"],
        "supervised_episode_result_digest": supervised["result_digest"],
        "policy_query_count": episode["policy_query_count"],
        "blocker_type": None,
        "ranking_eligible": False,
        "provider_teardown_verified": False,
        "official_billing_reconciled": False,
        "public_redistribution_authorized": False,
    }
    preclose = {
        **core,
        "schema_version": PRECLOSE_SCHEMA,
        "status": "awaiting_simulator_close",
        "teardown": {"environment": "closed", "simulator": "close_requested"},
    }
    preclose["preclose_digest"] = canonical_digest(preclose, digest_field="preclose_digest")
    (root / PRECLOSE_FILENAME).write_text(json.dumps(preclose))
    worker = {
        **core,
        "schema_version": WORKER_SCHEMA,
        "status": "completed_development_only",
        "teardown": {"environment": "closed", "simulator": "closed"},
    }
    worker["result_digest"] = canonical_digest(worker, digest_field="result_digest")
    (root / WORKER_FILENAME).write_text(json.dumps(worker))
    args = {
        "output_dir": root,
        "execution_packet": packet,
        "scene_plan_digest": episode["scene_plan_digest"],
        "scene_packet_receipt_digest": scene_receipt_digest,
    }
    return args, episode


def test_team_paid_output_verifies_score_queries_and_lossless_media(
    tmp_path: Path, monkeypatch
) -> None:
    args, episode = _evidence(tmp_path, monkeypatch)
    verified = verify_g1_team_paid_output(**args)
    assert verified["status"] == "verified_development_only"
    assert verified["policy_query_count"] == episode["policy_query_count"] == 2
    assert verified["score"] == episode["score"]
    assert set(verified["media"]["review_videos"]) == {"head", "overview"}
    assert verified["provider_teardown_verified"] is False
    assert verified["official_billing_reconciled"] is False


def test_team_paid_output_rejects_tampered_video_and_changed_policy_binding(
    tmp_path: Path, monkeypatch
) -> None:
    args, _ = _evidence(tmp_path, monkeypatch)
    verified = verify_g1_team_paid_output(**args)
    video = args["output_dir"] / verified["media"]["review_videos"]["head"]["relative_path"]
    video.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="g1_pair_media"):
        verify_g1_team_paid_output(**args)
    packet = dict(args["execution_packet"])
    packet["packet_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="packet_invalid"):
        verify_g1_team_paid_output(**{**args, "execution_packet": packet})


def test_team_paid_output_rejects_missing_policy_input_frame(
    tmp_path: Path, monkeypatch
) -> None:
    args, _ = _evidence(tmp_path, monkeypatch)
    verified = verify_g1_team_paid_output(**args)
    manifest = json.loads((
        args["output_dir"] / verified["media"]["frame_manifest"]["relative_path"]
    ).read_text())
    relative = manifest["policy_input_observations"][0]["views"]["head"]["relative_path"]
    (args["output_dir"] / "episode/episode" / relative).unlink()
    with pytest.raises(ValueError, match="multicamera_frame_manifest"):
        verify_g1_team_paid_output(**args)
