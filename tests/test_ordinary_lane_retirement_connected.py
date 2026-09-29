# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_retirement.py
#   src/blueprint_pipeline/control_plane_lane_experiment_birth.py
#   src/blueprint_pipeline/control_plane_lane_experiment_actions.py
#   src/blueprint_pipeline/control_plane_storage_gc.py
"""ADP-009D/day-28: uninstrumented ordinary lanes have no retirement authority."""

import pytest


def test_generic_registered_name_is_reserved_for_authentic_reader_and_lease_protocol(tmp_path):
    from blueprint_pipeline import control_plane_lane_scratch as scratch
    from blueprint_pipeline import control_plane_lane_experiment_consumer as consumer

    root = tmp_path / "lanes"
    lane = root / "diagnostics"
    lane.mkdir(parents=True)
    name = "registered-" + "a" * 32
    target = lane / name
    assert consumer.registered_target(target / "payload", (root,)) == target
    with pytest.raises(scratch.LaneScratchError, match="registered_authority_required"):
        scratch.create_lane_scratch("diagnostics", name, root=root, owner="owner", reason="fixture",
                                    class_intent="scratch", cleanup="delete", ttl_seconds=100,
                                    run_ref="run-1", now=lambda: 10)
    assert not target.exists()


def test_ordinary_lane_without_enrolled_producer_cannot_claim_delete_authority():
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    assert "ordinary_local_disposable.v1" not in issuer._PROFILES


def test_any_registered_name_cannot_be_renewed_outside_current_authority(tmp_path):
    from blueprint_pipeline import control_plane_lane_scratch as scratch
    root = tmp_path / "lanes"
    (root / "diagnostics" / ("registered-" + "a" * 32)).mkdir(parents=True)
    with pytest.raises(scratch.LaneScratchError, match="registered_authority_required"):
        scratch.renew_lane_scratch(root=root, lane="diagnostics", name="registered-" + "a" * 32,
                                   owner="owner", expected_digest="sha256:" + "b" * 64, ttl_seconds=100,
                                   now=lambda: 1100)
