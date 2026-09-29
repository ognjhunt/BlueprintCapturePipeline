"""A zero-participant birth cannot itself prove that its writer has stopped."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_actions.py
#   src/blueprint_pipeline/control_plane_lane_experiment_birth.py

from tests.test_registered_experiment_retirement_flow import (
    _born_scratch, _gc, _issue_action, _payload_snapshot,
    retirement_installation,  # noqa: F401
)
from tests.test_registered_experiment_issuer import installation  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
import pytest


def test_unsealed_local_disposable_keeps_late_writer_payload(retirement_installation):  # noqa: F811
    state = retirement_installation
    grant, _, target = _born_scratch(state)
    with pytest.raises(ValueError, match="experiment_producer_completion_missing"):
        _issue_action(state, grant)
    # The birth has zero enrolled participants. A normal owner can still add
    # bytes after action issuance and before the timer's unlink interval.
    (target / "late-writer.bin").write_bytes(b"writer still alive")
    before = _payload_snapshot(target)

    assert _gc(state)["registered_experiments"]["outcomes"] == []
    assert _payload_snapshot(target) == before


def test_existing_unsealed_action_is_skipped_by_timer_and_direct_run(retirement_installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_actions as code
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root

    state = retirement_installation
    grant, _, target = _born_scratch(state)
    # Manufacture an action from the prior release at the single new eligibility
    # boundary. Production issue, timer and run must all reject this old action.
    with monkeypatch.context() as prior_release:
        prior_release.setattr(code, "_require_disposable_producer_completion", lambda: None)
        action = _issue_action(state, grant)
    (target / "late-writer.bin").write_bytes(b"writer still alive")
    before = _payload_snapshot(target)

    outcomes = _gc(state)["registered_experiments"]["outcomes"]
    assert len(outcomes) == 1 and outcomes[0]["decision"] == "kept"
    assert outcomes[0]["reason"] == "experiment_producer_completion_missing"
    assert outcomes[0]["removed_logical_bytes"] == outcomes[0]["removed_allocated_bytes"] == 0
    direct = root.run_registered_experiment_action(
        action["action_id"], expected_action_intent=action["action_intent"],
        installed_config_path=state[0], now=lambda: 2900)
    assert direct["decision"] == "kept" and direct["reason"] == "experiment_producer_completion_missing"
    assert _payload_snapshot(target) == before
