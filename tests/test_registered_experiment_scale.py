"""Actual full supported member domain; zero-sized payloads bound disk usage."""

# ruff: noqa: F811
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_actions.py
#   src/blueprint_pipeline/control_plane_lane_experiment_work.py
#   src/blueprint_pipeline/control_plane_storage_gc.py
from pathlib import Path

import pytest

from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_registered_experiment_birth import birth
from tests.test_registered_experiment_issuer import installation, issue  # noqa: F401
from tests.test_registered_experiment_offload import expired_completed_evidence  # noqa: F401
from tests.test_registered_experiment_retirement_flow import (  # noqa: F401
    _current_entry,
    _gc,
    _issue_action,
    retirement_installation,
)


@pytest.mark.slow
def test_actual_4096_member_unsealed_expiry_is_kept_without_widening_native_budget(
    retirement_installation, monkeypatch
):
    from blueprint_pipeline import control_plane_lane_experiment_work as work
    from tests.test_registered_experiment_retirement_flow import _payload_snapshot

    setup = retirement_installation
    grant = issue(setup)
    born = birth(setup, grant)
    target = Path(born["path"])
    for number in range(4096):
        (target / f"m-{number:04d}").write_bytes(b"")
    before = _payload_snapshot(target)
    entry = _current_entry(setup, grant["intent_id"])
    assert entry["state"] == "active" and entry["completion"] is None
    records = {path.relative_to(setup[2]): path.read_bytes()
               for path in setup[2].rglob("*") if path.is_file()}
    peaks = []
    original = work._ActionFiles.slot

    def observed(self):
        peaks.append(len(self.owned) + len(self.probe_owned))
        return original(self)

    monkeypatch.setattr(work._ActionFiles, "slot", observed)
    # Expiry and a large member inventory do not supply a producer seal. This
    # exercises refusal with native budgets, not positive deletion throughput.
    with pytest.raises(ValueError, match="^experiment_producer_completion_missing$"):
        _issue_action(setup, grant)
    assert {path.relative_to(setup[2]): path.read_bytes()
            for path in setup[2].rglob("*") if path.is_file()} == records
    phase = _gc(setup)["registered_experiments"]
    assert phase["enabled"] is True and phase["outcomes"] == []
    assert len(list(target.iterdir())) == 4098
    assert _payload_snapshot(target) == before
    assert _current_entry(setup, grant["intent_id"]) == entry
    assert entry["generation"] == born["generation"]
    assert peaks and max(peaks) < 128


@pytest.mark.slow
@pytest.mark.parametrize(
    "member_count", [pytest.param(32, id="small32"), pytest.param(4096, id="full4096")]
)
def test_actual_4096_evidence_offload_restores_under_same_store_and_fd_limits(
    expired_completed_evidence, monkeypatch, member_count
):
    from blueprint_pipeline import control_plane_lane_experiment_archive as archive
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root
    from blueprint_pipeline import control_plane_lane_experiment_work as work
    from blueprint_pipeline import control_plane_lane_scratch as scratch
    from tests.test_registered_experiment_offload import Cloud

    setup, target, born, _, intent_id = expired_completed_evidence
    control = {scratch.LEASE_FILE, ".registered-experiment.v1.json"}
    count = sum(path.relative_to(target).parts[0] not in control for path in target.rglob("*"))
    for number in range(member_count - count):
        (target / f"extra-{number:04d}").write_bytes(b"")
    before = {
        str(path.relative_to(target)): path.read_bytes()
        for path in target.rglob("*")
        if path.is_file() and path.name not in control
    }
    cloud = Cloud()
    monkeypatch.setattr(archive, "_client", lambda *args: (cloud, "development-only"))
    peaks = []
    original = work._ActionFiles.slot

    def observed(self):
        peaks.append(len(self.owned) + len(self.probe_owned))
        return original(self)

    monkeypatch.setattr(work._ActionFiles, "slot", observed)
    from functools import partial
    from types import SimpleNamespace

    from blueprint_pipeline import control_plane_disk_budget as disk
    from blueprint_pipeline import control_plane_lane_experiment_restore as restoration

    # The portable fixture cannot create /var/lib's production ledger, and this
    # Mac has less free space than the unchanged protected production floor.
    # Use actual native ledger/measurement/reservation/release with its explicit
    # finite disk model. The separate real Linux acceptance uses actual capacity.
    native_reserve = partial(
        disk.reserve_control_plane_disk,
        reservation_root=setup[0].parent / "actual-restore-ledger",
        disk_usage=lambda _: SimpleNamespace(
            total=64 * 1024**3, used=32 * 1024**3, free=32 * 1024**3
        ),
    )
    reservations = []

    def actual_reserve(*args, **kwargs):
        token = native_reserve(*args, **kwargs)
        reservations.append(token)
        return token

    monkeypatch.setattr(restoration, "reserve_control_plane_disk", actual_reserve)
    # The maximum member domain needs the authenticated one-hour policy grant,
    # not this tiny fixture's usual ten-minute grant; all native clocks remain.
    action = root.issue_experiment_action_intent(
        intent_id,
        principal="operator",
        owner="owner",
        action="offload",
        expires_at_epoch=6500,
        installed_config_path=setup[0],
        now=lambda: 2900,
    )
    outcome = _gc(setup)["registered_experiments"]["outcomes"][0]
    assert outcome["action_id"] == action["action_id"]
    assert outcome["decision"] == "retired", outcome
    selected = root.issue_experiment_restore_intent(
        intent_id,
        principal="operator",
        owner="owner",
        lease_ttl_seconds=600,
        expires_at_epoch=6500,
        installed_config_path=setup[0],
        now=lambda: 2901,
    )
    try:
        restored = root.restore_registered_experiment(
            selected["action_id"],
            expected_restore_intent=selected["restore_intent"],
            installed_config_path=setup[0],
            now=lambda: 2901,
            _pins_root=setup[0].parent / "pins",
        )
    except ValueError as error:
        native = error.__context__
        if isinstance(native, OSError):
            import traceback

            print(
                "LOCAL_FULL_DOMAIN_NATIVE_ERROR",
                native.errno,
                native.filename,
                [
                    (Path(row.filename).name, row.lineno, row.name)
                    for row in traceback.extract_tb(native.__traceback__)[-12:]
                ],
            )
        raise
    assert restored["decision"] == "restored", restored
    assert len(reservations) == 1 and reservations[0].released
    assert reservations[0].path.parent == setup[0].parent / "actual-restore-ledger"
    assert not reservations[0].path.exists()
    assert {
        str(path.relative_to(target)): path.read_bytes()
        for path in target.rglob("*")
        if path.is_file() and path.name not in control
    } == before
    assert _current_entry(setup, intent_id)["state"] == "active"
    assert _current_entry(setup, intent_id)["generation"] == born["generation"]
    assert peaks and max(peaks) < 128
