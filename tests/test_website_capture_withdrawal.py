"""Only synthetic local files are deleted by these lifecycle tests."""
import json
from pathlib import Path

import pytest

from blueprint_pipeline.consent_takedown import read_consent_state
from blueprint_pipeline.website_capture_withdrawal import (
    acknowledge_withdrawal,
    apply_cloud_cleanup,
    apply_local_cleanup,
    inspect_withdrawal,
    plan_cloud_cleanup,
    plan_local_cleanup,
)


def fixture(tmp_path):
    root = tmp_path / "scenes/site-req1/captures/walkthrough-req1"
    (root / "raw").mkdir(parents=True)
    (root / "raw/manifest.json").write_text(json.dumps({"site_submission_id": "req1", "scene_id": "site-req1",
        "capture_id": "walkthrough-req1", "consent_status": "granted"}))
    (root / "raw/video.mp4").write_bytes(b"synthetic-video")
    command = {"schema_version": "website_capture_withdrawal.v1", "request_id": "req1", "scene_id": "site-req1",
        "capture_id": "walkthrough-req1", "withdrawal_id": "withdrawal-req1", "requested_at_iso": "2026-10-06T00:00:00Z"}
    return root, command


class FakeStorage:
    bucket_name = "synthetic"
    def __init__(self):
        self.rows = [{"name": "scenes/site-req1/captures/walkthrough-req1/raw/video.mp4", "generation": "123",
                      "size": 15, "crc32c": "AAAAAA=="}]
        self.soft = []
        self.deleted = []
        self.crash = False
    def objects(self, prefix):
        return [dict(row) for row in self.rows]
    def soft_deleted_objects(self, prefix):
        return self.soft
    def delete(self, name, generation):
        self.deleted.append((name, generation))
        self.rows = [row for row in self.rows if (row["name"], row["generation"]) != (name, generation)]
        if self.crash:
            self.crash = False
            raise OSError("synthetic unknown deletion acknowledgement")


def test_cloud_unknown_acknowledgement_replays_exact_remaining_generations(tmp_path):
    root, command = fixture(tmp_path)
    acknowledge_withdrawal(capture_root=root, command=command)
    storage = FakeStorage()
    plan = plan_cloud_cleanup(capture_root=root, storage=storage)
    storage.crash = True
    with pytest.raises(OSError):
        apply_cloud_cleanup(capture_root=root, storage=storage, expected_plan_digest=plan["digest"], authorize_cloud_deletion=True)
    receipt = apply_cloud_cleanup(capture_root=root, storage=storage, expected_plan_digest=plan["digest"], authorize_cloud_deletion=True)
    assert len(storage.deleted) == 1
    assert receipt["cloud_object_absence_verified"] is True
    assert receipt["cloud_storage_cleanup_verified"] is False
    assert receipt["provider_acknowledgement"] == "unknown"
    assert receipt["deletion_confirmed"] is False
    assert apply_cloud_cleanup(capture_root=root, storage=storage, expected_plan_digest=plan["digest"], authorize_cloud_deletion=True) == receipt
    assert len(storage.deleted) == 1


@pytest.mark.parametrize("fault", ["unauthorized", "foreign", "generation", "hold", "soft_deleted"])
def test_cloud_cleanup_refuses_unknown_scope_new_versions_holds_and_retained_copies(tmp_path, fault):
    root, command = fixture(tmp_path)
    acknowledge_withdrawal(capture_root=root, command=command)
    storage = FakeStorage()
    plan = plan_cloud_cleanup(capture_root=root, storage=storage)
    if fault == "foreign":
        storage.rows[0]["name"] = "scenes/site-other/captures/walkthrough-other/raw/video.mp4"
    elif fault == "generation":
        storage.rows[0]["generation"] = "999"
    elif fault == "hold":
        (root.parent.parent / "website_withdrawal/legal_hold.json").write_text('{"legal_hold":true}')
    elif fault == "soft_deleted":
        storage.soft = [{"name": storage.rows[0]["name"], "generation": "100"}]
    with pytest.raises(ValueError):
        apply_cloud_cleanup(capture_root=root, storage=storage, expected_plan_digest=plan["digest"], authorize_cloud_deletion=fault != "unauthorized")
    assert inspect_withdrawal(capture_root=root)["cloud_object_absence_verified"] is False
    if fault != "soft_deleted":
        assert storage.deleted == []


def test_acknowledgement_is_durable_denial_not_deletion(tmp_path):
    root, command = fixture(tmp_path)
    before = (root / "raw/video.mp4").read_bytes()
    receipt = acknowledge_withdrawal(capture_root=root, command=command)
    assert receipt["pipeline_acknowledged"] is True
    assert receipt["local_cleanup_verified"] is False
    assert receipt["deletion_confirmed"] is False
    assert (root / "raw/video.mp4").read_bytes() == before
    assert acknowledge_withdrawal(capture_root=root, command=command) == receipt
    assert read_consent_state(root)["state"] == "revoked"
    late = root.parent / "supplement-late"
    assert read_consent_state(late)["state"] == "revoked"


def test_synthetic_cleanup_exact_plan_replay_and_provider_unknown(tmp_path):
    root, command = fixture(tmp_path)
    acknowledge_withdrawal(capture_root=root, command=command)
    plan = plan_local_cleanup(capture_root=root)
    receipt = apply_local_cleanup(capture_root=root, expected_plan_digest=plan["digest"], authorize_local_deletion=True)
    assert not (root / "raw/video.mp4").exists()
    assert receipt["local_cleanup_verified"] is True
    assert receipt["provider_acknowledgement"] == "unknown"
    assert receipt["deletion_confirmed"] is False
    assert receipt["state"] == "local_cleanup_verified_external_pending"
    assert apply_local_cleanup(capture_root=root, expected_plan_digest=plan["digest"], authorize_local_deletion=True) == receipt
    assert inspect_withdrawal(capture_root=root) == receipt
    assert read_consent_state(root)["state"] == "revoked"


@pytest.mark.parametrize("fault", ["foreign_root", "foreign_manifest", "changed_retry", "unauthorized", "changed_plan", "legal_hold", "symlink"])
def test_scope_identity_hold_and_deletion_authorization_fail_closed(tmp_path, fault):
    root, command = fixture(tmp_path)
    if fault == "foreign_root":
        root = root.parent / "walkthrough-other"
    elif fault == "foreign_manifest":
        (root / "raw/manifest.json").write_text(json.dumps({"site_submission_id": "other"}))
    if fault in {"foreign_root", "foreign_manifest"}:
        with pytest.raises(ValueError):
            acknowledge_withdrawal(capture_root=root, command=command)
        return
    acknowledge_withdrawal(capture_root=root, command=command)
    if fault == "changed_retry":
        command["withdrawal_id"] = "changed"
        with pytest.raises(ValueError):
            acknowledge_withdrawal(capture_root=root, command=command)
        return
    plan = plan_local_cleanup(capture_root=root)
    if fault == "changed_plan":
        (root / "raw/video.mp4").write_bytes(b"replacement")
    elif fault == "legal_hold":
        (root.parent.parent / "website_withdrawal/legal_hold.json").write_text('{"legal_hold":true}')
    elif fault == "symlink":
        (root / "raw/escape").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError):
        apply_local_cleanup(capture_root=root, expected_plan_digest=plan["digest"], authorize_local_deletion=fault != "unauthorized")
    assert (root / "raw/video.mp4").exists()


def test_crash_after_unlink_replays_intent_without_deleting_replacement(tmp_path, monkeypatch):
    root, command = fixture(tmp_path)
    acknowledge_withdrawal(capture_root=root, command=command)
    plan = plan_local_cleanup(capture_root=root)
    original = Path.unlink
    count = 0
    def interrupted(path, *args, **kwargs):
        nonlocal count
        result = original(path, *args, **kwargs)
        if path.name == "video.mp4":
            count += 1
            raise OSError("synthetic crash after unlink")
        return result
    monkeypatch.setattr(Path, "unlink", interrupted)
    with pytest.raises(OSError):
        apply_local_cleanup(capture_root=root, expected_plan_digest=plan["digest"], authorize_local_deletion=True)
    monkeypatch.setattr(Path, "unlink", original)
    assert apply_local_cleanup(capture_root=root, expected_plan_digest=plan["digest"], authorize_local_deletion=True)["local_cleanup_verified"] is True
    assert count == 1
    (root / "raw/video.mp4").write_bytes(b"new unplanned data")
    assert inspect_withdrawal(capture_root=root)["local_cleanup_verified"] is False
    with pytest.raises(ValueError):
        apply_local_cleanup(capture_root=root, expected_plan_digest=plan["digest"], authorize_local_deletion=True)
    assert (root / "raw/video.mp4").read_bytes() == b"new unplanned data"
