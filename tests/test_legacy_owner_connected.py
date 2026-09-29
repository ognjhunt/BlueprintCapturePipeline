# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_legacy_owner.py
"""A reviewed old folder gains an owner label without payload action."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_legacy_owner_packet import _consent, _survey


@pytest.fixture
def installed(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_lane_legacy_owner as legacy
    from blueprint_pipeline import control_plane_lane_owner_consents as owners

    work = tmp_path / "work"
    target = work / "lanes" / "diagnostics" / "old-1"
    target.mkdir(parents=True)
    payload = target / "one.log"
    payload.write_bytes(b"old, preserved")
    registry = tmp_path / "requests" / "legacy-owner-registrations"
    registry.mkdir(parents=True, mode=0o700)
    registry.chmod(0o700)
    (registry / ".legacy-owner.lock").write_bytes(b"")
    (registry / ".legacy-owner.lock").chmod(0o600)
    policy = tmp_path / "policy.json"
    raw = (json.dumps(dict(schema_version="control_plane_lane_owner_policy.v1", enabled=True,
                           principals=[dict(principal="operator", owners=["owner"],
                                            allowed_actions=["register"], max_consent_seconds=500)]),
                      sort_keys=True, separators=(",", ":")) + "\n").encode()
    policy.write_bytes(raw)
    policy.chmod(0o600)
    config = SimpleNamespace(owner_consent_store=str(tmp_path / "requests" / "owner-consents"),
                             lane_scratch_work_root=str(work / "lanes"),
                             lane_scratch_inputs_root=str(tmp_path / "inputs" / "lanes"),
                             lane_owner_policy_file=str(policy))
    consent = _consent(str(target))
    consent["policy_sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(owners, "_installed_config", lambda files, path: config)
    monkeypatch.setattr(legacy, "_load_old_consent", lambda *args, **kwargs: consent)
    monkeypatch.setattr(legacy, "_fresh_census", lambda *args, **kwargs: _survey(str(target)))
    monkeypatch.setattr(legacy.os, "geteuid", lambda: 0)
    def protected(info, *, directory=False, mode=None):
        import stat
        if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
            raise owners.OwnerCensusConsentError("owner_consent_store_unsafe")
        if mode is not None and stat.S_IMODE(info.st_mode) != mode:
            raise owners.OwnerCensusConsentError("owner_consent_store_unsafe")
    monkeypatch.setattr(owners, "_protected", protected)
    return target, payload, registry, legacy


def _issue_and_approve(installed):
    target, _, _, legacy = installed
    packet = legacy.issue_version_packet(consent_id="a" * 32,
                                         consent_sha256="sha256:" + "b" * 64,
                                         consent_size_bytes=1, selected_path=str(target),
                                         installed_config_path="/fixture/door.json", now=1010,
                                         monotonic=lambda: 0)
    approval = legacy.issue_generation_approval(packet_id=packet["packet_id"],
                                                ack_packet_digest=packet["packet_digest"],
                                                principal="operator", owner="owner",
                                                installed_config_path="/fixture/door.json", now=1020,
                                                monotonic=lambda: 0)
    return packet, approval


def test_connected_owner_review_preserves_payload_and_never_mints_cleanup_lease(installed):
    target, payload, registry, legacy = installed
    packet, approval = _issue_and_approve(installed)
    assert approval["approval_digest"]
    result = legacy.apply_owner_review(packet_id=packet["packet_id"],
                                       installed_config_path="/fixture/door.json", now=1021,
                                       monotonic=lambda: 0)
    assert result["status"] == "legacy_owner_review_registered"
    assert result["gc_eligible"] is False and result["candidate_bytes"] is None
    assert payload.read_bytes() == b"old, preserved"
    assert not (target / ".lane-scratch.v1.json").exists()
    assert list(registry.glob("*.head.json"))
    observed = legacy.observe_owner_review(installed_config_path="/fixture/door.json", now=1022,
                                           monotonic=lambda: 0)
    row = next(item for item in observed["rows"] if item["path"] == str(target))
    assert row["owner"] == "owner" and row["classification"] == "legacy_owner_review"
    assert row["gc_eligible"] is False and row["references_clear"] is False
    assert row["candidate_bytes"] is None and row["eta_seconds"] is None


def test_connected_unknown_fd_label_requires_distinct_ack_and_stays_keep(installed, monkeypatch):
    target, payload, _, legacy = installed
    unknown = _survey(str(target))
    unknown.update(status="incomplete", scan_errors=["process_inventory_unreadable"],
                   candidate_count=1)
    monkeypatch.setattr(legacy, "_fresh_census", lambda *args, **kwargs: unknown)
    packet = legacy.issue_version_packet(consent_id="a" * 32,
                                         consent_sha256="sha256:" + "b" * 64,
                                         consent_size_bytes=1, selected_path=str(target),
                                         installed_config_path="/fixture/door.json", now=1010,
                                         monotonic=lambda: 0)
    with pytest.raises(legacy.LegacyOwnerError, match="legacy_owner_approval_ack_mismatch"):
        legacy.issue_generation_approval(packet_id=packet["packet_id"],
                                         ack_packet_digest=packet["packet_digest"],
                                         principal="operator", owner="owner",
                                         installed_config_path="/fixture/door.json", now=1020,
                                         monotonic=lambda: 0)
    legacy.issue_generation_approval(packet_id=packet["packet_id"],
                                     ack_packet_digest=packet["packet_digest"],
                                     ack_process_fd_unknown=True, principal="operator", owner="owner",
                                     installed_config_path="/fixture/door.json", now=1020,
                                     monotonic=lambda: 0)
    legacy.apply_owner_review(packet_id=packet["packet_id"],
                              installed_config_path="/fixture/door.json", now=1021,
                              monotonic=lambda: 0)
    observed = legacy.observe_owner_review(installed_config_path="/fixture/door.json", now=1022,
                                           monotonic=lambda: 0)
    row = next(item for item in observed["rows"] if item["path"] == str(target))
    assert observed["status"] == "incomplete"
    assert row["classification"] == "owner_review_reference_unknown"
    assert row["gc_eligible"] is False and row["references_clear"] is False
    assert payload.read_bytes() == b"old, preserved"
    assert not (target / ".lane-scratch.v1.json").exists()


def test_crash_before_head_is_recoverable_without_promoting_torn_record(installed, monkeypatch):
    target, payload, registry, legacy = installed
    packet, _ = _issue_and_approve(installed)
    real_publish = legacy.LegacyOwnerStore.publish_head
    monkeypatch.setattr(legacy.LegacyOwnerStore, "publish_head", lambda *a, **k: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError, match="crash"):
        legacy.apply_owner_review(packet_id=packet["packet_id"],
                                  installed_config_path="/fixture/door.json", now=1021,
                                  monotonic=lambda: 0)
    assert not list(registry.glob("*.head.json"))
    assert payload.exists()
    uncommitted = legacy.observe_owner_review(installed_config_path="/fixture/door.json", now=1021,
                                              monotonic=lambda: 0)
    assert all("owner" not in row for row in uncommitted["rows"])
    monkeypatch.setattr(legacy.LegacyOwnerStore, "publish_head", real_publish)
    result = legacy.apply_owner_review(packet_id=packet["packet_id"],
                                       installed_config_path="/fixture/door.json", now=1021,
                                       monotonic=lambda: 0)
    assert result["status"] == "legacy_owner_review_registered"
    assert Path(target / "one.log").read_bytes() == b"old, preserved"


def test_lane_scratch_is_attributed_and_expires(installed):
    target, payload, _, legacy = installed
    packet, _ = _issue_and_approve(installed)
    legacy.apply_owner_review(packet_id=packet["packet_id"],
                              installed_config_path="/fixture/door.json", now=1021,
                              monotonic=lambda: 0)
    payload.write_bytes(b"old, preserveD")
    changed = legacy.observe_owner_review(installed_config_path="/fixture/door.json", now=1022,
                                          monotonic=lambda: 0)
    assert all("owner" not in row for row in changed["rows"])
    payload.write_bytes(b"old, preserved")
    expired = legacy.observe_owner_review(installed_config_path="/fixture/door.json", now=1110,
                                          monotonic=lambda: 0)
    assert all("owner" not in row for row in expired["rows"])
    assert target.exists()


def test_survey_ignores_fake_in_folder_lease_and_keeps_target(installed):
    target, payload, _, legacy = installed
    packet, _ = _issue_and_approve(installed)
    legacy.apply_owner_review(packet_id=packet["packet_id"],
                              installed_config_path="/fixture/door.json", now=1021,
                              monotonic=lambda: 0)
    (target / ".lane-scratch.v1.json").write_bytes(b'{"cleanup":"delete"}')
    report = legacy.observe_owner_review(installed_config_path="/fixture/door.json", now=1022,
                                         monotonic=lambda: 0)
    assert all("owner" not in row for row in report["rows"])
    assert payload.exists()


def test_active_or_incomplete_reference_inventory_keeps_legacy_target(installed, monkeypatch):
    target, payload, registry, legacy = installed
    packet, _ = _issue_and_approve(installed)
    active = _survey(str(target))
    active["rows"][0]["references"] = ["queue"]
    monkeypatch.setattr(legacy, "_fresh_census", lambda *a, **k: active)
    with pytest.raises(legacy.LegacyOwnerError, match="legacy_owner_references_incomplete"):
        legacy.apply_owner_review(packet_id=packet["packet_id"],
                                  installed_config_path="/fixture/door.json", now=1021,
                                  monotonic=lambda: 0)
    assert not list(registry.glob("*.registration.json"))
    assert payload.exists()
    monkeypatch.setattr(legacy, "_fresh_census", lambda *a, **k: (_ for _ in ()).throw(
        legacy.LegacyOwnerError("legacy_owner_references_incomplete")))
    report = legacy.observe_owner_review(installed_config_path="/fixture/door.json", now=1022,
                                         monotonic=lambda: 0)
    assert report["status"] == "incomplete" and report["observed_owner_count"] == 0


def test_unregistered_top_level_folder_is_reported_not_deleted(installed, monkeypatch):
    target, _, _, legacy = installed
    top = target.parents[2] / "old-drawer-experiment"
    top.mkdir()
    payload = top / "evidence.bin"
    payload.write_bytes(b"keep old evidence")
    consent = _consent(str(top))
    consent["decisions"][0]["decision"].update(lane="diagnostics", name="old-drawer-experiment")
    policy = target.parents[3] / "policy.json"
    consent["policy_sha256"] = "sha256:" + hashlib.sha256(policy.read_bytes()).hexdigest()
    monkeypatch.setattr(legacy, "_load_old_consent", lambda *a, **k: consent)
    monkeypatch.setattr(legacy, "_fresh_census", lambda *a, **k: _survey(str(top)))
    packet = legacy.issue_version_packet(consent_id="a" * 32,
                                         consent_sha256="sha256:" + "b" * 64,
                                         consent_size_bytes=1, selected_path=str(top),
                                         installed_config_path="/fixture/door.json", now=1010,
                                         monotonic=lambda: 0)
    legacy.issue_generation_approval(packet_id=packet["packet_id"],
                                     ack_packet_digest=packet["packet_digest"],
                                     principal="operator", owner="owner",
                                     installed_config_path="/fixture/door.json", now=1020,
                                     monotonic=lambda: 0)
    legacy.apply_owner_review(packet_id=packet["packet_id"],
                              installed_config_path="/fixture/door.json", now=1021,
                              monotonic=lambda: 0)
    report = legacy.observe_owner_review(installed_config_path="/fixture/door.json", now=1022,
                                         monotonic=lambda: 0)
    [row] = report["rows"]
    assert row["path"] == str(top) and row["owner"] == "owner"
    assert row["gc_eligible"] is False and row["candidate_bytes"] is None
    assert payload.read_bytes() == b"keep old evidence" and top.is_dir()


def test_revoked_owner_policy_or_replaced_target_drops_label(installed):
    target, payload, _, legacy = installed
    packet, _ = _issue_and_approve(installed)
    legacy.apply_owner_review(packet_id=packet["packet_id"],
                              installed_config_path="/fixture/door.json", now=1021,
                              monotonic=lambda: 0)
    policy = target.parents[3] / "policy.json"
    original = policy.read_bytes()
    policy.write_bytes(original.replace(b'"enabled":true', b'"enabled":false'))
    revoked = legacy.observe_owner_review(installed_config_path="/fixture/door.json", now=1022,
                                          monotonic=lambda: 0)
    assert all("owner" not in row for row in revoked["rows"])
    policy.write_bytes(original)
    replacement = target.parent / "replacement"
    replacement.mkdir()
    (replacement / "one.log").write_bytes(payload.read_bytes())
    target.rename(target.parent / "saved-original")
    replacement.rename(target)
    changed = legacy.observe_owner_review(installed_config_path="/fixture/door.json", now=1022,
                                          monotonic=lambda: 0)
    assert all("owner" not in row for row in changed["rows"])
    assert (target / "one.log").exists()
