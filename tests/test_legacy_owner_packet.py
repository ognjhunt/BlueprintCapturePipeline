# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_legacy_owner.py
"""A path-only owner consent never silently becomes target authority."""

import pytest


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
