# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_retirement.py
#   src/blueprint_pipeline/control_plane_lane_experiment_birth.py
#   src/blueprint_pipeline/control_plane_lane_experiment_actions.py
#   src/blueprint_pipeline/control_plane_storage_gc.py
"""ADP-009D/day-28: a newly issued ordinary lane can retire only its own bytes."""

from pathlib import Path
import json

import pytest

from tests.test_registered_experiment_retirement_flow import (  # noqa: F401
    retirement_installation, _gc, _current_entry,
)
from tests.test_registered_experiment_issuer import installation  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401


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


def test_ordinary_lane_new_owned_delete_requires_birth_and_separate_owner_action(retirement_installation):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    from blueprint_pipeline import control_plane_lane_experiment_birth as birth

    config, settings, _, _ = retirement_installation
    issued = issuer.issue_experiment_creation_intent(
        installed_config_path=config, principal="operator", owner="owner", root="work",
        lane="diagnostics", reference_value="run-ordinary", lease_ttl_seconds=1800,
        participant_profile="ordinary_local_disposable.v1", request_records=(), now=lambda: 1000,
    )
    (Path(settings["lane_scratch_work_root"]) / "diagnostics").mkdir(mode=0o750)
    born = birth.create_registered_experiment(issued["intent_id"], expected_intent=issued["intent"],
                                              installed_config_path=config, now=lambda: 1001)
    target = Path(born["path"])
    assert target.parent.name == "diagnostics"
    payload = target / "one.log"
    payload.write_bytes(b"small disposable diagnostic")
    # The enabled timer and expired lease by themselves have no action grant.
    assert _gc(retirement_installation)["registered_experiments"]["outcomes"] == []
    assert payload.exists()
    approved = issuer.issue_experiment_action_intent(
        issued["intent_id"], principal="operator", owner="owner", action="delete",
        expires_at_epoch=3500, installed_config_path=config, now=lambda: 2900,
    )
    report = _gc(retirement_installation)
    rows = [row for row in report["registered_experiments"]["outcomes"]
            if row["action_id"] == approved["action_id"]]
    assert len(rows) == 1 and rows[0]["decision"] == "retired"
    assert not payload.exists()
    assert _current_entry(retirement_installation, issued["intent_id"])["state"] == "retired"


def test_ordinary_registered_name_cannot_be_renewed_outside_current_authority(retirement_installation):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    from blueprint_pipeline import control_plane_lane_experiment_birth as birth
    from blueprint_pipeline import control_plane_lane_scratch as scratch

    config, settings, _, _ = retirement_installation
    issued = issuer.issue_experiment_creation_intent(
        installed_config_path=config, principal="operator", owner="owner", root="work",
        lane="diagnostics", reference_value="run-ordinary", lease_ttl_seconds=1800,
        participant_profile="ordinary_local_disposable.v1", request_records=(), now=lambda: 1000,
    )
    (Path(settings["lane_scratch_work_root"]) / "diagnostics").mkdir(mode=0o750)
    born = birth.create_registered_experiment(issued["intent_id"], expected_intent=issued["intent"],
                                              installed_config_path=config, now=lambda: 1001)
    target = Path(born["path"])
    lease = json.loads((target / scratch.LEASE_FILE).read_bytes())
    with pytest.raises(scratch.LaneScratchError, match="registered_authority_required"):
        scratch.renew_lane_scratch(root=target.parents[1], lane="diagnostics", name=target.name,
                                   owner="owner", expected_digest=lease["lease_digest"], ttl_seconds=100,
                                   now=lambda: 1100)
    assert json.loads((target / scratch.LEASE_FILE).read_bytes()) == lease
