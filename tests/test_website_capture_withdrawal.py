"""Only synthetic local files are deleted by these lifecycle tests."""
import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline.consent_takedown import read_consent_state
from blueprint_pipeline import website_capture_withdrawal as module
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_capture_withdrawal import (
    acknowledge_withdrawal,
    apply_cloud_cleanup,
    apply_local_cleanup,
    inspect_withdrawal,
    plan_cloud_cleanup,
    plan_local_cleanup,
)


def fixture(tmp_path, request_id="req1"):
    root = tmp_path / f"scenes/site-{request_id}/captures/walkthrough-{request_id}"
    (root / "raw").mkdir(parents=True)
    (root / "raw/manifest.json").write_text(json.dumps({"site_submission_id": request_id, "scene_id": f"site-{request_id}",
        "capture_id": f"walkthrough-{request_id}", "consent_status": "granted"}))
    (root / "raw/video.mp4").write_bytes(b"synthetic-video")
    command = {"schema_version": "website_capture_withdrawal.v1", "request_id": request_id, "scene_id": f"site-{request_id}",
        "capture_id": f"walkthrough-{request_id}", "withdrawal_id": f"withdrawal-{request_id}", "requested_at_iso": "2026-10-06T00:00:00Z"}
    return root, command


def proof_path(root, cloud=False):
    return next((root.parent.parent / "website_withdrawal" / ("cloud_verifications" if cloud else "verifications")).glob("*.json"))


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
    original = os.unlink
    count = 0
    def interrupted(path, *args, **kwargs):
        nonlocal count
        result = original(path, *args, **kwargs)
        if Path(path).name == "video.mp4":
            count += 1
            raise OSError("synthetic crash after unlink")
        return result
    monkeypatch.setattr(os, "unlink", interrupted)
    with pytest.raises(OSError):
        apply_local_cleanup(capture_root=root, expected_plan_digest=plan["digest"], authorize_local_deletion=True)
    monkeypatch.setattr(os, "unlink", original)
    assert apply_local_cleanup(capture_root=root, expected_plan_digest=plan["digest"], authorize_local_deletion=True)["local_cleanup_verified"] is True
    assert count == 1
    (root / "raw/video.mp4").write_bytes(b"new unplanned data")
    assert inspect_withdrawal(capture_root=root)["local_cleanup_verified"] is False
    with pytest.raises(ValueError):
        apply_local_cleanup(capture_root=root, expected_plan_digest=plan["digest"], authorize_local_deletion=True)
    assert (root / "raw/video.mp4").read_bytes() == b"new unplanned data"


def test_replacement_at_final_deletion_boundary_is_preserved(tmp_path, monkeypatch):
    root, command = fixture(tmp_path)
    acknowledge_withdrawal(capture_root=root, command=command)
    plan = plan_local_cleanup(capture_root=root)
    source = root / "raw/video.mp4"
    original_hash, original_rename = module._sha256_file, os.rename
    hashes, replaced = 0, False
    def replace_source():
        nonlocal replaced
        if not replaced:
            replacement = root / "raw/replacement.tmp"
            replacement.write_bytes(b"unplanned replacement")
            replacement.replace(source)
            replaced = True
    def race_hash(path):
        nonlocal hashes
        result = original_hash(path)
        if path == source:
            hashes += 1
            if hashes == 2:  # The legacy final source hash/unlink boundary.
                replace_source()
        return result
    def race_move(src, dst, *args, **kwargs):
        if Path(src).name == "video.mp4":
            replace_source()
        return original_rename(src, dst, *args, **kwargs)
    monkeypatch.setattr(module, "_sha256_file", race_hash)
    monkeypatch.setattr(os, "rename", race_move)
    with pytest.raises(ValueError, match="source_changed"):
        apply_local_cleanup(capture_root=root, expected_plan_digest=plan["digest"], authorize_local_deletion=True)
    assert replaced
    assert source.read_bytes() == b"unplanned replacement"
    assert inspect_withdrawal(capture_root=root)["local_cleanup_verified"] is False


def test_crash_after_quarantine_keeps_pending_payload_visible_and_replays_once(tmp_path, monkeypatch):
    root, command = fixture(tmp_path)
    acknowledge_withdrawal(capture_root=root, command=command)
    plan = plan_local_cleanup(capture_root=root)
    original = os.rename
    moved = 0
    def interrupted(src, dst, *args, **kwargs):
        nonlocal moved
        original(src, dst, *args, **kwargs)
        if Path(src).name == "video.mp4":
            moved += 1
            raise OSError("synthetic crash after quarantine")
    monkeypatch.setattr(os, "rename", interrupted)
    with pytest.raises(OSError):
        apply_local_cleanup(capture_root=root, expected_plan_digest=plan["digest"], authorize_local_deletion=True)
    assert inspect_withdrawal(capture_root=root)["local_cleanup_verified"] is False
    monkeypatch.setattr(os, "rename", original)
    assert apply_local_cleanup(capture_root=root, expected_plan_digest=plan["digest"], authorize_local_deletion=True)["local_cleanup_verified"] is True
    assert moved == 1


def test_unplanned_quarantine_is_preserved_when_source_is_concurrently_recreated(tmp_path, monkeypatch):
    root, command = fixture(tmp_path)
    acknowledge_withdrawal(capture_root=root, command=command)
    plan = plan_local_cleanup(capture_root=root)
    original = os.rename
    source = root / "raw/video.mp4"
    def replace_at_move(src, dst, *args, **kwargs):
        if Path(src).name == "video.mp4":
            replacement = root / "raw/replacement.tmp"
            replacement.write_bytes(b"unplanned candidate")
            replacement.replace(source)
        original(src, dst, *args, **kwargs)
        if Path(src).name == "video.mp4":
            source.write_bytes(b"concurrently recreated source")
    monkeypatch.setattr(os, "rename", replace_at_move)
    with pytest.raises(ValueError, match="source_changed"):
        apply_local_cleanup(capture_root=root, expected_plan_digest=plan["digest"], authorize_local_deletion=True)
    assert source.read_bytes() == b"concurrently recreated source"
    candidates = list((root.parent.parent / "website_withdrawal/quarantine").rglob("video.mp4"))
    assert len(candidates) == 1
    assert candidates[0].read_bytes() == b"unplanned candidate"
    assert inspect_withdrawal(capture_root=root)["local_cleanup_verified"] is False
    monkeypatch.setattr(os, "rename", original)
    authorized_recovery = plan_local_cleanup(capture_root=root)
    assert any(row["path"].startswith("website_withdrawal/quarantine/") for row in authorized_recovery["files"])
    with pytest.raises(ValueError):
        apply_local_cleanup(capture_root=root, expected_plan_digest=authorized_recovery["digest"])
    assert apply_local_cleanup(capture_root=root, expected_plan_digest=authorized_recovery["digest"], authorize_local_deletion=True)["local_cleanup_verified"] is True
    assert not source.exists()
    assert not candidates[0].exists()


def test_directory_substitution_cannot_delete_outside_site(tmp_path, monkeypatch):
    root, command = fixture(tmp_path)
    acknowledge_withdrawal(capture_root=root, command=command)
    plan = plan_local_cleanup(capture_root=root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "manifest.json").write_bytes(b"outside customer data")
    original = os.rename
    switched = False
    def substitute_parent(src, dst, *args, **kwargs):
        nonlocal switched
        if not switched:
            original(root / "raw", root / "moved-raw")
            (root / "raw").symlink_to(outside, target_is_directory=True)
            switched = True
        return original(src, dst, *args, **kwargs)
    monkeypatch.setattr(os, "rename", substitute_parent)
    with pytest.raises((ValueError, OSError)):
        apply_local_cleanup(capture_root=root, expected_plan_digest=plan["digest"], authorize_local_deletion=True)
    assert switched
    assert (outside / "manifest.json").read_bytes() == b"outside customer data"


def test_cleanup_proof_cannot_be_replayed_from_another_site(tmp_path):
    donor, command = fixture(tmp_path / "donor")
    acknowledge_withdrawal(capture_root=donor, command=command)
    plan = plan_local_cleanup(capture_root=donor)
    apply_local_cleanup(capture_root=donor, expected_plan_digest=plan["digest"], authorize_local_deletion=True)
    receiver, command = fixture(tmp_path / "receiver", "req2")
    acknowledge_withdrawal(capture_root=receiver, command=command)
    for path in (receiver / "raw").iterdir():
        path.unlink()
    (receiver.parent.parent / "website_withdrawal/local_cleanup_verified.json").write_bytes(
        proof_path(donor).read_bytes())
    with pytest.raises(ValueError):
        inspect_withdrawal(capture_root=receiver)


@pytest.mark.parametrize("fault", ["schema", "tombstone", "bucket", "false_absence", "missing_plan"])
def test_cloud_proof_requires_bound_retained_plan_and_positive_absence(tmp_path, fault):
    root, command = fixture(tmp_path)
    acknowledge_withdrawal(capture_root=root, command=command)
    storage = FakeStorage()
    plan = plan_cloud_cleanup(capture_root=root, storage=storage)
    apply_cloud_cleanup(capture_root=root, storage=storage, expected_plan_digest=plan["digest"], authorize_cloud_deletion=True)
    path = proof_path(root, cloud=True)
    value = json.loads(path.read_text())
    if fault == "schema":
        value["schema_version"] = "unknown"
    elif fault == "tombstone":
        value["tombstone_digest"] = "sha256:" + "f" * 64
    elif fault == "bucket":
        value["bucket"] = "foreign"
    elif fault == "false_absence":
        value["all_object_versions_absence_observed"] = False
    else:
        value["plan_digest"] = "sha256:" + "f" * 64
    value["digest"] = canonical_digest(value, digest_field="digest")
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError, FileNotFoundError)):
        inspect_withdrawal(capture_root=root)


def test_retained_cloud_observation_does_not_assert_current_absence(tmp_path):
    root, command = fixture(tmp_path)
    acknowledge_withdrawal(capture_root=root, command=command)
    storage = FakeStorage()
    plan = plan_cloud_cleanup(capture_root=root, storage=storage)
    verified = apply_cloud_cleanup(capture_root=root, storage=storage, expected_plan_digest=plan["digest"], authorize_cloud_deletion=True)
    storage.rows = [{"name": "scenes/site-req1/late.jpg", "generation": "999", "size": 1, "crc32c": "AAAAAA=="}]
    observed = acknowledge_withdrawal(capture_root=root, command=command)
    assert observed["cloud_object_absence_verified"] is False
    assert observed["cloud_object_current_absence_status"] == "unknown"
    assert observed["cloud_object_cleanup_receipt_digest"] == verified["cloud_object_cleanup_receipt_digest"]
    assert observed["cloud_object_absence_observed_at_iso"]


@pytest.mark.parametrize("hold", [{}, {"legal_hold": "true"}, {"legal_hold": None}])
def test_malformed_hold_refuses_deletion_instead_of_inferencing_release(tmp_path, hold):
    root, command = fixture(tmp_path)
    acknowledge_withdrawal(capture_root=root, command=command)
    plan = plan_local_cleanup(capture_root=root)
    (root.parent.parent / "website_withdrawal/legal_hold.json").write_text(json.dumps(hold))
    with pytest.raises(ValueError):
        apply_local_cleanup(capture_root=root, expected_plan_digest=plan["digest"], authorize_local_deletion=True)
    assert (root / "raw/video.mp4").exists()


@pytest.mark.parametrize("cloud", [False, True])
def test_explicit_new_plan_handles_late_sources_without_rewriting_audit_history(tmp_path, cloud):
    root, command = fixture(tmp_path)
    acknowledge_withdrawal(capture_root=root, command=command)
    storage = FakeStorage()
    if cloud:
        first = plan_cloud_cleanup(capture_root=root, storage=storage)
        apply_cloud_cleanup(capture_root=root, storage=storage, expected_plan_digest=first["digest"], authorize_cloud_deletion=True)
        storage.rows = [{"name": "scenes/site-req1/late.jpg", "generation": "999", "size": 1, "crc32c": "AAAAAA=="}]
    else:
        first = plan_local_cleanup(capture_root=root)
        apply_local_cleanup(capture_root=root, expected_plan_digest=first["digest"], authorize_local_deletion=True)
        (root / "raw/late.jpg").write_bytes(b"late synthetic bytes")
    journal = root.parent.parent / "website_withdrawal"
    retained = {str(path.relative_to(journal)): path.read_bytes() for path in journal.rglob("*.json")}
    if cloud:
        second = plan_cloud_cleanup(capture_root=root, storage=storage)
        with pytest.raises(ValueError):
            apply_cloud_cleanup(capture_root=root, storage=storage, expected_plan_digest=second["digest"])
        result = apply_cloud_cleanup(capture_root=root, storage=storage, expected_plan_digest=second["digest"], authorize_cloud_deletion=True)
        assert result["cloud_object_absence_verified"] is True
        assert len(storage.deleted) == 2
    else:
        second = plan_local_cleanup(capture_root=root)
        with pytest.raises(ValueError):
            apply_local_cleanup(capture_root=root, expected_plan_digest=second["digest"])
        result = apply_local_cleanup(capture_root=root, expected_plan_digest=second["digest"], authorize_local_deletion=True)
        assert result["local_cleanup_verified"] is True
        assert not (root / "raw/late.jpg").exists()
    assert first["digest"] != second["digest"]
    for relative, data in retained.items():
        assert (journal / relative).read_bytes() == data
