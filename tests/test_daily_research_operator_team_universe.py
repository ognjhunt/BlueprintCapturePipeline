"""Private team export publication/pin uses real bridge/Store with hermetic Firestore/GCS doubles."""
import base64
import importlib.util
from pathlib import Path

import pytest

from tests.daily_team_evidence_fixture import TODAY, export
from tests.test_daily_research_operator_site_universe import fixture as bridge_fixture
from tools.daily_research import team_universe as te
from tools.daily_research.runner import Refusal

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("team_evidence_operator", ROOT / "tools/daily_research/operators/team-universe-evidence.py")
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)


@pytest.fixture
def fixture(tmp_path):
    generator = bridge_fixture.__wrapped__(tmp_path)
    bridge, _, restart = next(generator)
    path = tmp_path / "team-evidence.json"
    path.write_bytes(export())
    yield bridge, path, restart
    try:
        next(generator)
    except StopIteration:
        pass


def publish(fixture):
    bridge, path, _ = fixture
    return operator.publish(bridge, path, apply=True, today=TODAY)


def setpin(fixture, stored, **options):
    bridge, _, _ = fixture
    return operator.pin(bridge, sha256=stored["sha256"], generation=stored["generation"],
                        approval_reference="synthetic-reviewed-owner", today=TODAY, **options)


def test_default_readonly_createonly_readback_pin_restart_and_control_preservation(fixture):
    bridge, path, restart = fixture
    before = bridge.call("control")
    plan = operator.publish(bridge, path, today=TODAY)
    assert plan["state"] == "planned" and plan["object_writes"] == 0
    assert bridge.call("team_universe_snapshot")["state"] == "unavailable"
    stored = publish(fixture)
    repeated = publish(fixture)
    assert stored["generation"] == repeated["generation"]
    assert setpin(fixture, stored)["state"] == "planned" and "team_universe" not in bridge.call("control")
    result = setpin(fixture, stored, apply=True)
    assert result["readback_verified"] and result["pin"]["version"] == 1
    control = bridge.call("control")
    assert all(control[k] == v for k, v in before.items() if k != "lease")
    snap = bridge.call("team_universe_snapshot")
    assert base64.b64decode(snap["data"]) == path.read_bytes()
    again = restart()
    try:
        assert operator.show(again, today=TODAY)["input"]["state"] == "attached"
    finally:
        again.close()
    assert setpin(fixture, stored, apply=True)["pin"]["version"] == 2
    with pytest.raises(Refusal, match="pin_conflict"):
        setpin(fixture, stored, expect="none", apply=True)


def test_wrong_generation_hash_approval_and_configure_cannot_change_pin(fixture):
    bridge, _, _ = fixture
    stored = publish(fixture)
    with pytest.raises(Refusal, match="object_unavailable"):
        setpin(fixture, {**stored, "generation": "999999"})
    with pytest.raises(te.TeamEvidenceError):
        operator.pin(bridge, sha256=stored["sha256"], generation=stored["generation"], approval_reference="PENDING", today=TODAY)
    pinned = setpin(fixture, stored, apply=True)["pin"]
    bridge.call("acquire", scope="research_release")
    try:
        with pytest.raises(Refusal, match="requires_pin_operation"):
            bridge.call("configure", value={**bridge.call("control"), "team_universe": {**pinned, "version": 100}})
        with pytest.raises(Refusal, match="version_conflict"):
            bridge.call("team_universe_set", expected_sha256=pinned["sha256"], value={**pinned, "version": 5})
    finally:
        bridge.call("release")


def test_unfinished_daily_row_stops_pin_before_acquire_and_under_lease(fixture):
    bridge, _, _ = fixture
    stored = publish(fixture)
    plan = setpin(fixture, stored)
    bridge.call("acquire")
    try:
        bridge.call("import_run", row={"date": "2026-10-04", "run_key": "blueprint-researcher:2026-10-04",
                    "metadata": {}, "state": "running", "cleanup_required": True})
        with pytest.raises(Refusal, match="active_research_qa_repair_or_publication"):
            bridge.call("team_universe_set", expected_sha256=None, value=plan["pin"])
    finally:
        bridge.call("release")
    with pytest.raises(Refusal, match="active_research_qa_repair_or_publication"):
        setpin(fixture, stored, apply=True)
    assert "team_universe" not in bridge.call("control")
