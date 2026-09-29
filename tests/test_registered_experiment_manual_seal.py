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


def test_unsealed_local_disposable_keeps_late_writer_payload(retirement_installation):  # noqa: F811
    installation = retirement_installation
    grant, _, target = _born_scratch(installation)
    _issue_action(installation, grant)
    # The birth has zero enrolled participants. A normal owner can still add
    # bytes after action issuance and before the timer's unlink interval.
    (target / "late-writer.bin").write_bytes(b"writer still alive")
    before = _payload_snapshot(target)

    outcomes = _gc(installation)["registered_experiments"]["outcomes"]
    assert len(outcomes) == 1
    assert outcomes[0]["decision"] == "kept"
    assert outcomes[0]["reason"] == "experiment_producer_completion_missing"
    assert _payload_snapshot(target) == before
