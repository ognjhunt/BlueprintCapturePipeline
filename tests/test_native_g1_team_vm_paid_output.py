"""Canonical collection must retain both host and guest evidence (fake Isaac)."""

import json
import shutil
import zipfile

import pytest

from blueprint_pipeline import native_g1_team_paid_policy as lane
from blueprint_pipeline import native_g1_team_vm_host as host
from blueprint_pipeline.native_g1_team_vm_bundle_support import BOOTSTRAP_RESULT_FILENAME
from blueprint_pipeline.native_g1_team_private_review import project_g1_team_private_review
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from tests import test_native_g1_team_vm_host as vm_fixtures


COMMIT = "a" * 40


@pytest.fixture
def rehearsal(tmp_path, monkeypatch, request):
    yield from vm_fixtures.rehearsal.__wrapped__(tmp_path, monkeypatch, request)


def _collected(tmp_path, rehearsal):
    args, verification, _, _, _, _, _ = rehearsal
    packet = verification["execution_packet"]
    packet["request"]["run_id"] = "owner-review-fixture"
    packet["intent_id"] = "g1-team-policy-" + cross_runtime_canonical_digest({
        "owner": packet["request"]["owner"], "run_id": packet["request"]["run_id"],
    }).removeprefix("sha256:")
    packet["implementation_commit"] = COMMIT
    packet["packet_digest"] = cross_runtime_canonical_digest(packet, digest_field="packet_digest")
    # Rebind the fake archive device receipt before the actual socket/session
    # rehearsal. This never observes or claims physical device access.
    gpu = host.preflight_g1_vm_host(packet).get("archive_gpu_device_binding")
    if gpu is not None:
        gpu["execution_packet_digest"] = packet["packet_digest"]
        gpu["receipt_digest"] = canonical_digest(gpu, digest_field="receipt_digest")
    result = host.run_g1_team_vm_host(**args)
    assert result["status"] == "completed_development_only", result
    job = tmp_path / "canonical-job"
    attempt = job / "attempts/attempt_001"
    root = attempt / "immutable_execution"
    root.parent.mkdir(parents=True)
    shutil.copytree(verification["output_dir"], root)
    bootstrap = {
        "schema_version": "native_g1_team_vm_bootstrap_entrypoint.v1",
        "status": "host_exited", "stage_reached": "vm-host", "runner_exit_code": 0,
        "implementation_commit": COMMIT, "provider_mutation_performed": False,
        "gpu_runtime_qualified": False, "claim_ceiling": "development_only",
    }
    host._write(root / BOOTSTRAP_RESULT_FILENAME, bootstrap)
    archive = job / "input.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(lane.PACKET_RELATIVE_PATH, json.dumps(verification["execution_packet"]))
    bundle = {"bundle_path": str(archive), "bundle_sha256": lane._sha256(archive),
              "implementation_commit": COMMIT,
              "scene_plan_digest": verification["scene_plan_digest"],
              "scene_packet_receipt_digest": verification["scene_packet_receipt_digest"]}
    bundle.update({"intent_id": packet["intent_id"], "execution_packet_digest": packet["packet_digest"],
                   "policy_profile_digest": packet["request"]["policy_profile"]["profile_digest"],
                   "delivery_mode": packet["request"]["policy_profile"]["delivery"]["mode"],
                   "objective_id": packet["request"]["objective_id"],
                   "scene_id": packet["trusted_setup"]["scene_id"], "task_id": packet["trusted_setup"]["task_id"],
                   "container_image": host.NATIVE_TASK_ARENA_IMAGE})
    return {"attempt_root": str(attempt)}, bundle, job, root


@pytest.mark.slow
@pytest.mark.parametrize("rehearsal", ["container", "noncontainer_artifact"], indirect=True)
def test_canonical_collection_reopens_both_host_and_guest_closures(tmp_path, rehearsal):
    result, bundle, job, _ = _collected(tmp_path, rehearsal)
    verified = lane._verify_output(result, bundle, job=job)
    assert verified["schema_version"] == "native_g1_team_paid_output_verification.v1"
    assert verified["policy_query_count"] == 2
    assert set(verified["media"]["review_videos"]) == {"head", "overview"}
    paired = verified["isolated_policy_host_verification"]
    assert paired["schema_version"] == "native_g1_team_vm_output_verification.v1"
    assert paired["provider_teardown_verified"] is False
    assert paired["official_billing_reconciled"] is False
    assert paired["public_redistribution_authorized"] is False
    assert paired["bootstrap_entrypoint_digest"].startswith("sha256:")
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        packet = json.loads(archive.read(lane.PACKET_RELATIVE_PATH))
    review = project_g1_team_private_review(verification=verified, bundle=bundle, execution_packet=packet)
    assert review["policy_delivery_mode"] == bundle["delivery_mode"]
    assert review["episodes"][0]["policy_query_count"] == 2
    assert set(review["episodes"][0]["review_videos"]) == {"head", "overview"}


@pytest.mark.slow
@pytest.mark.parametrize("fault", ["missing", "commit", "exit", "boolean", "stage", "gpu", "provider",
                                  "host", "relay", "video"])
def test_canonical_collection_rejects_missing_or_resealed_foreign_host_evidence(tmp_path, rehearsal, fault):
    result, bundle, job, root = _collected(tmp_path, rehearsal)
    bootstrap_path = root / BOOTSTRAP_RESULT_FILENAME
    if fault == "missing":
        bootstrap_path.unlink()
    elif fault == "host":
        (root / host.HOST_FILENAME).unlink()
    elif fault == "relay":
        path = root / "policy-host" / host.RELAY_FILENAME
        relay = json.loads(path.read_text())
        relay["inference_query_count"] += 1
        path.unlink()
        host._write(path, relay)
    elif fault == "video":
        videos = list((root / "selected-worker/worker").rglob("*.mp4"))
        assert videos
        videos[0].write_bytes(b"changed retained video")
    else:
        bootstrap = json.loads(bootstrap_path.read_text())
        key, value = {
            "commit": ("implementation_commit", "b" * 40),
            "exit": ("runner_exit_code", 7), "boolean": ("runner_exit_code", False),
            "stage": ("stage_reached", "host-python-provisioning"),
            "gpu": ("gpu_runtime_qualified", True), "provider": ("provider_mutation_performed", True),
        }[fault]
        bootstrap[key] = value
        bootstrap_path.unlink()
        host._write(bootstrap_path, bootstrap)
    with pytest.raises((ValueError, OSError)):
        lane._verify_output(result, bundle, job=job)
