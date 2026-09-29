# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_legacy_owner.py
"""A path-only owner consent never silently becomes target authority."""

import pytest
import json


def _consent(path):
    return dict(schema_version="control_plane_lane_owner_consent.v1",
                consent_id="a" * 32, consent_digest="sha256:" + "b" * 64,
                principal="operator", policy_sha256="sha256:" + "c" * 64,
                issued_at_epoch=1000, expires_at_epoch=1300,
                census=dict(sha256="sha256:" + "d" * 64, size_bytes=123),
                annotations=dict(sha256="sha256:" + "e" * 64, size_bytes=456),
                execution_authorized=False, target_generation_bound=False,
                decisions=[dict(census_row=dict(path=path, references=[], unreadable=0),
                                decision=dict(path=path, action="register", owner="owner",
                                              lane="diagnostics", name="old-1",
                                              cleanup="owner_review", ttl_seconds=100))])


def _survey(path):
    return dict(schema_version="control_plane_lane_scratch_census.v1", status="complete",
                scan_errors=[], rows=[dict(path=path, references=[], unreadable=0,
                                           allocated_bytes=4096)])


def test_packet_requires_separate_approved_target_generation(tmp_path):
    from blueprint_pipeline.control_plane_lane_legacy_owner import (
        build_version_packet, snapshot_generation,
    )

    root = tmp_path / "work"
    target = root / "lanes" / "diagnostics" / "old-1"
    target.mkdir(parents=True)
    (target / "one.log").write_bytes(b"one")
    snapshot = snapshot_generation(target, allowed_roots=(root,))
    packet = build_version_packet(_consent(str(target)), selected_path=str(target),
                                  generation=snapshot, fresh_census=_survey(str(target)), now=1010)
    assert packet["owner"] == "owner"
    assert packet["target_generation"] == snapshot
    assert packet["old_consent"]["target_generation_bound"] is False
    assert packet["execution_authorized"] is False
    assert packet["approval_required"] is True


@pytest.mark.parametrize("change", [
    lambda c, s: c["decisions"][0]["decision"].update(action="delete"),
    lambda c, s: c["decisions"][0]["decision"].update(cleanup="delete"),
    lambda c, s: s.update(status="incomplete"),
    lambda c, s: s["rows"][0].update(references=["queue"]),
    lambda c, s: s["rows"][0].update(unreadable=1),
    lambda c, s: s["rows"][0].update(path="/other"),
])
def test_packet_refuses_destructive_or_incomplete_old_and_current_evidence(tmp_path, change):
    from blueprint_pipeline.control_plane_lane_legacy_owner import (
        LegacyOwnerError, build_version_packet, snapshot_generation,
    )

    root = tmp_path / "work"
    target = root / "lanes" / "diagnostics" / "old-1"
    target.mkdir(parents=True)
    consent, survey = _consent(str(target)), _survey(str(target))
    change(consent, survey)
    with pytest.raises(LegacyOwnerError):
        build_version_packet(consent, selected_path=str(target),
                             generation=snapshot_generation(target, allowed_roots=(root,)),
                             fresh_census=survey, now=1010)


def test_second_owner_approval_requires_exact_ack_and_policy(tmp_path):
    from blueprint_pipeline.control_plane_lane_legacy_owner import (
        LegacyOwnerError, approve_version_packet, build_version_packet, snapshot_generation,
    )

    root = tmp_path / "work"
    target = root / "lanes" / "diagnostics" / "old-1"
    target.mkdir(parents=True)
    policy = dict(schema_version="control_plane_lane_owner_policy.v1", enabled=True,
                  principals=[dict(principal="operator", owners=["owner"],
                                   allowed_actions=["register"], max_consent_seconds=500)])
    raw = (json.dumps(policy, sort_keys=True, separators=(",", ":")) + "\n").encode()
    import hashlib
    consent = _consent(str(target))
    consent["policy_sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    packet = build_version_packet(consent, selected_path=str(target),
                                  generation=snapshot_generation(target, allowed_roots=(root,)),
                                  fresh_census=_survey(str(target)), now=1010)
    with pytest.raises(LegacyOwnerError, match="legacy_owner_approval_ack_mismatch"):
        approve_version_packet(packet, ack_packet_digest="sha256:" + "0" * 64,
                               current_policy_bytes=raw, principal="operator", owner="owner", now=1020)
    approval = approve_version_packet(packet, ack_packet_digest=packet["packet_digest"],
                                      current_policy_bytes=raw, principal="operator", owner="owner", now=1020)
    assert approval["approved_action"] == "register_owner_review"
    assert approval["packet_digest"] == packet["packet_digest"]
    assert approval["execution_authorized"] is False
    changed = raw.replace(b'"enabled":true', b'"enabled":false')
    with pytest.raises(LegacyOwnerError, match="legacy_owner_policy_changed"):
        approve_version_packet(packet, ack_packet_digest=packet["packet_digest"],
                               current_policy_bytes=changed, principal="operator", owner="owner", now=1020)


def test_apply_validation_drops_attribution_on_mutation_or_expiry(tmp_path):
    from blueprint_pipeline.control_plane_lane_legacy_owner import (
        LegacyOwnerError, approve_version_packet, build_version_packet,
        validate_registration, snapshot_generation,
    )
    import hashlib

    root = tmp_path / "work"
    target = root / "lanes" / "diagnostics" / "old-1"
    target.mkdir(parents=True)
    payload = target / "one.log"
    payload.write_bytes(b"alpha")
    raw = b'{"enabled":true,"principals":[{"allowed_actions":["register"],"max_consent_seconds":500,"owners":["owner"],"principal":"operator"}],"schema_version":"control_plane_lane_owner_policy.v1"}\n'
    consent = _consent(str(target))
    consent["policy_sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    packet = build_version_packet(consent, selected_path=str(target),
                                  generation=snapshot_generation(target, allowed_roots=(root,)),
                                  fresh_census=_survey(str(target)), now=1010)
    approval = approve_version_packet(packet, ack_packet_digest=packet["packet_digest"],
                                      current_policy_bytes=raw, principal="operator", owner="owner", now=1020)
    result = validate_registration(packet, approval,
                                   current_generation=snapshot_generation(target, allowed_roots=(root,)),
                                   fresh_census=_survey(str(target)), current_policy_bytes=raw, now=1021)
    assert result["owner"] == "owner" and result["gc_eligible"] is False
    assert result["references_clear"] is False and result["candidate_bytes"] is None
    payload.write_bytes(b"bravo")
    with pytest.raises(LegacyOwnerError, match="legacy_target_changed"):
        validate_registration(packet, approval,
                              current_generation=snapshot_generation(target, allowed_roots=(root,)),
                              fresh_census=_survey(str(target)), current_policy_bytes=raw, now=1021)
    with pytest.raises(LegacyOwnerError, match="legacy_owner_approval_expired"):
        validate_registration(packet, approval, current_generation=packet["target_generation"],
                              fresh_census=_survey(str(target)), current_policy_bytes=raw, now=packet["expires_at_epoch"])
