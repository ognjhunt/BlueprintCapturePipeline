"""Selected review uses actual scored worker evidence and the shared registry."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.native_g1_team_paid_output import verify_g1_team_paid_output
from blueprint_pipeline.native_g1_team_private_review import (
    project_g1_team_private_review,
    validate_g1_team_private_review,
)
from blueprint_pipeline.native_g1_private_review_delivery import _stage_review_artifacts
from tests.test_native_g1_team_paid_output import _evidence


def _inputs(tmp_path, monkeypatch):
    args, episode = _evidence(tmp_path, monkeypatch)
    packet = args["execution_packet"]
    profile = packet["request"]["policy_profile"]
    verification = verify_g1_team_paid_output(**args)
    bundle = {
        "execution_packet_digest": packet["packet_digest"],
        "intent_id": packet["intent_id"],
        "policy_profile_digest": profile["profile_digest"],
        "objective_id": packet["request"]["objective_id"],
        "delivery_mode": profile["delivery"]["mode"],
        "scene_id": episode["scene_id"], "task_id": episode["task_id"],
        "container_image": "fixture-isaac-image",
        "bundle_sha256": "sha256:" + "a" * 64,
    }
    return args, verification, bundle


def test_selected_review_retains_actual_failure_queries_and_reopens_media(tmp_path, monkeypatch):
    args, verification, bundle = _inputs(tmp_path, monkeypatch)
    review = project_g1_team_private_review(
        verification=verification, bundle=bundle, execution_packet=args["execution_packet"],
    )
    assert validate_g1_team_private_review(review) == review
    assert review["schema_version"] == "native_g1_team_private_review.v1"
    assert "campaign_plan_digest" not in review
    assert review["owner_user_id"] == "owner-1"
    assert review["organization_id"] == "team-1"
    assert review["policy_delivery_mode"] == "authenticated_endpoint"
    assert review["runtime_delivery_mode"] == "container"
    assert len(review["episodes"]) == 1
    assert review["episodes"][0]["score"]["outcome"] == "failure"
    assert review["episodes"][0]["policy_query_count"] == 2
    # Paths resolve below the same immutable_execution root used by the controller.
    source = tmp_path / "immutable_execution"
    worker = source / "selected-worker/worker"
    worker.parent.mkdir(parents=True)
    import shutil
    shutil.copytree(args["output_dir"], worker)
    result = tmp_path / "results"
    result.mkdir()
    registration = _stage_review_artifacts(
        review=review, source_root=source, result_root=result, run_id=review["run_id"],
    )
    assert registration["artifact_count"] == 3
    assert _stage_review_artifacts(
        review=review, source_root=source, result_root=result, run_id=review["run_id"],
    ) == registration
    artifact = review["episodes"][0]["review_videos"]["head"]
    (Path(registration["run_root"]) / "evidence" / artifact["relative_path"]).write_bytes(b"tamper")
    with pytest.raises(ValueError, match="reverification_failed"):
        _stage_review_artifacts(
            review=review, source_root=source, result_root=result, run_id=review["run_id"],
        )


@pytest.mark.parametrize("change", [
    "packet", "profile", "bundle_profile", "bundle_intent", "bundle_objective",
    "bundle_scene", "queries", "public", "missing_video", "duplicate_video", "path",
])
def test_selected_projection_rejects_unbound_evidence(tmp_path, monkeypatch, change):
    args, verification, bundle = _inputs(tmp_path, monkeypatch)
    if change == "packet":
        args["execution_packet"]["request"]["owner"]["user_id"] = "another-owner"
    elif change == "profile":
        verification["profile_digest"] = "sha256:" + "0" * 64
    elif change.startswith("bundle_"):
        key = {"bundle_profile": "policy_profile_digest", "bundle_intent": "intent_id",
               "bundle_objective": "objective_id", "bundle_scene": "scene_id"}[change]
        bundle[key] = "wrong"
    elif change == "queries":
        verification["policy_query_count"] = 0
    elif change == "public":
        verification["public_redistribution_authorized"] = True
    elif change == "missing_video":
        del verification["media"]["review_videos"]["head"]
    elif change == "duplicate_video":
        verification["media"]["review_videos"]["overview"] = deepcopy(
            verification["media"]["review_videos"]["head"]
        )
    else:
        verification["media"]["frame_manifest"]["relative_path"] = "episode/../secret"
    with pytest.raises(ValueError, match="g1_team_private_review"):
        project_g1_team_private_review(
            verification=verification, bundle=bundle, execution_packet=args["execution_packet"],
        )


@pytest.mark.parametrize("change", [
    "count", "physical", "delivery", "path", "profile", "score", "run_id", "owner",
    "organization", "intent", "delivery_mapping", "objective_mapping",
])
def test_selected_record_validation_refuses_resealed_invalid_contract(tmp_path, monkeypatch, change):
    args, verification, bundle = _inputs(tmp_path, monkeypatch)
    review = project_g1_team_private_review(
        verification=verification, bundle=bundle, execution_packet=args["execution_packet"],
    )
    if change == "count":
        review["episodes"] *= 4
    elif change == "physical":
        review["physical_outcome_claimed"] = True
    elif change == "delivery":
        review["policy_delivery_mode"] = "unknown"
    elif change == "path":
        review["episodes"][0]["frame_manifest"]["relative_path"] = "movement_pair/frames.json"
    elif change == "profile":
        review["policy_profile_digest"] = "sha256:" + "0" * 64
    elif change == "run_id":
        review["run_id"] = "different-run"
    elif change == "owner":
        review["owner_user_id"] = "different-owner"
    elif change == "organization":
        review["organization_id"] = "different-team"
    elif change == "intent":
        review["intent_id"] = "g1-team-policy-" + "0" * 64
    elif change == "delivery_mapping":
        review["policy_delivery_mode"] = {}
    elif change == "objective_mapping":
        review["episodes"][0]["objective_id"] = {}
    else:
        del review["episodes"][0]["score"]["score_digest"]
    review["review_digest"] = cross_runtime_canonical_digest(review, digest_field="review_digest")
    with pytest.raises(ValueError, match="g1_team_private_review"):
        validate_g1_team_private_review(review)


def test_selected_registry_refuses_different_request_id(tmp_path, monkeypatch):
    args, verification, bundle = _inputs(tmp_path, monkeypatch)
    review = project_g1_team_private_review(
        verification=verification, bundle=bundle, execution_packet=args["execution_packet"],
    )
    with pytest.raises(ValueError, match="selected_intent_mismatch"):
        _stage_review_artifacts(
            review=review, source_root=args["output_dir"], result_root=tmp_path, run_id="another-run",
        )
    assert not (tmp_path / "another-run-activation").exists()


def test_python_selected_fixture_is_cross_runtime_sealed():
    fixture = Path(__file__).parent / "fixtures/native_g1_team_private_review.v1.json"
    review = json.loads(fixture.read_text())
    assert validate_g1_team_private_review(review) == review
