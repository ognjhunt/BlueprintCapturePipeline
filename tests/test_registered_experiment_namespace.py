"""Registered names cannot fall back to legacy creation or lease mutation."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_scratch.py

import json
import os

import pytest

from blueprint_pipeline import control_plane_lane_scratch as scratch


@pytest.mark.parametrize("operation", ["create", "renew", "release"])
@pytest.mark.parametrize("lane", ["g1", "g1-checkpoint"])
def test_registered_native_entrypoint_refuses_before_any_acquisition(tmp_path, monkeypatch, operation, lane):
    monkeypatch.setattr(os, "open", lambda *a, **kw: pytest.fail("registered legacy open"))
    monkeypatch.setattr(os, "mkdir", lambda *a, **kw: pytest.fail("registered legacy mkdir"))
    name = "registered-" + "a" * 32
    with pytest.raises(scratch.LaneScratchError, match="lane_scratch_registered_authority_required"):
        if operation == "create":
            scratch.create_lane_scratch(lane, name, root=tmp_path, owner="owner", reason="scratch",
                class_intent="scratch", cleanup="delete", ttl_seconds=100, run_ref="run1")
        else:
            function = scratch.renew_lane_scratch if operation == "renew" else scratch.release_lane_scratch
            options = {"ttl_seconds": 100} if operation == "renew" else {}
            function(root=tmp_path, lane=lane, name=name, owner="owner",
                     expected_digest="sha256:" + "a" * 64, **options)


def test_native_nonreserved_defaults_still_create_renew_and_release(tmp_path):
    path = scratch.create_lane_scratch("g1", "legacy", root=tmp_path, owner="owner", reason="scratch",
        class_intent="scratch", cleanup="delete", ttl_seconds=100, run_ref="run1", now=lambda: 1000)
    initial = json.loads((path / scratch.LEASE_FILE).read_bytes())
    renewed = scratch.renew_lane_scratch(root=tmp_path, lane="g1", name="legacy", owner="owner",
        expected_digest=initial["lease_digest"], ttl_seconds=200, now=lambda: 1050)
    assert renewed["expires_at_epoch"] == 1250
    released = scratch.release_lane_scratch(root=tmp_path, lane="g1", name="legacy", owner="owner",
        expected_digest=renewed["lease_digest"], now=lambda: 1060)
    assert released["released_at_epoch"] == 1060
