"""A full campaign needs its entire declared runtime before it can spend."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os

import pytest

from blueprint_pipeline import agent_run_executor as executor
from blueprint_pipeline import team_candidate_quick10_launch as launch
from blueprint_pipeline.team_candidate_quick10_vast import team_execution_deadline
from blueprint_pipeline.team_evaluation_plan import campaign_ttl_seconds
from tests.test_team_candidate_quick10_launch import _world, _Broker, _staged_run_reported


@pytest.mark.parametrize("window, accepted", [(57600, False), (61800, False), (62100, True), (70000, True)])
def test_a_deadline_cannot_silently_truncate_the_planned_campaign(window, accepted):
    authority = {"hard_ttl_seconds": campaign_ttl_seconds(), "hard_deadline_epoch": 1000000}
    deadline, blockers = team_execution_deadline(authority, 1000 + window, now=1000)
    assert deadline == 1000 + window
    assert (blockers == []) is accepted
    if not accepted:
        assert blockers == ["team_evaluation_execution_window_below_planned_runtime"]


def test_runner_timeout_covers_the_highest_supported_planner_rate():
    root = Path(__file__).resolve().parents[1]
    unit = (root / "deploy/systemd/blueprint-team-evaluation-runner.service").read_text()
    assert "TimeoutStartSec=7d" in unit
    assert 7 * 86400 > campaign_ttl_seconds(control_hz=120) + launch.STOP_TEARDOWN_GRACE_SECONDS


def test_short_funding_refuses_before_releasing_the_credential(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    original = launch._funding

    def funding(row, profile):
        return {**original(row, profile), "expires_at_iso":
                (datetime.now(timezone.utc) + timedelta(hours=16)).isoformat()}

    monkeypatch.setattr(launch, "_funding", funding)
    result = executor.poll_once(client=world["client"], capture_root=world["capture"],
                                team_launch_hooks=world["hooks"])
    assert result["blocked"] == 1
    assert _Broker.accessed == [] and world["allocator_calls"] == []
    assert world["client"].blocked == ["team_evaluation_funding_window_below_planned_runtime"]


@pytest.mark.parametrize("window", [57600, 61801])
def test_short_execution_window_refuses_before_the_only_allocation(tmp_path, monkeypatch, window):
    import time
    world = _world(tmp_path, monkeypatch)
    world["runner"]["execution_monitor"] = lambda control: {
        "ok": True, "run_id": control["run_id"], "execution_deadline_at_ms": (time.time() + window) * 1000}
    _staged, result = _staged_run_reported(world)
    assert result["blocked"] == 1 and world["allocator_calls"] == []
    assert world["client"].blocked == ["team_evaluation_execution_window_below_planned_runtime"]
    assert not (world["job_dir"] / ".team-credential").exists()


@pytest.mark.parametrize('hourly_rate, accepted', [(0.8, False), (0.4, True)])
def test_faster_robot_must_fit_the_cap_before_claiming_its_one_use_credential(tmp_path, monkeypatch, hourly_rate, accepted):
    import json
    from blueprint_pipeline.team_candidate_quick10_registry import REGISTRY_ENV
    from tests.test_team_embodiment import manifest, write_robot_usd
    from tests.test_team_candidate_quick10_launch import _sealed

    model = write_robot_usd(tmp_path / 'arm.usda')
    world = _world(tmp_path, monkeypatch, robot={'manifest': manifest(model, control_hz=30), 'model': model})
    registry_path = Path(os.environ[REGISTRY_ENV])
    registry = json.loads(registry_path.read_text())
    registry['profiles'][0]['max_hourly_rate_usd'] = hourly_rate
    registry_path.write_text(json.dumps(_sealed(registry, 'registry_digest')))
    result = executor.poll_once(client=world['client'], capture_root=world['capture'], team_launch_hooks=world['hooks'])
    if accepted:
        assert result['staged'] == 1 and len(_Broker.accessed) == 1
    else:
        assert result['blocked'] == 1 and _Broker.accessed == []
        assert world['client'].blocked == ['team_evaluation_budget_below_planned_runtime']
        assert not (world['job_dir'] / '.team-credential').exists()
    assert world['allocator_calls'] == []


@pytest.mark.parametrize("slack", [1, 1801])
def test_funding_without_enough_dispatch_and_allocator_slack_refuses_before_claim(tmp_path, monkeypatch, slack):
    world = _world(tmp_path, monkeypatch)
    original = launch._funding

    def funding(row, profile):
        return {**original(row, profile), 'expires_at_iso':
                (datetime.now(timezone.utc) + timedelta(seconds=campaign_ttl_seconds() + slack)).isoformat()}

    monkeypatch.setattr(launch, '_funding', funding)
    result = executor.poll_once(client=world['client'], capture_root=world['capture'], team_launch_hooks=world['hooks'])
    assert result['blocked'] == 1 and _Broker.accessed == [] and world['allocator_calls'] == []
    assert world['client'].blocked == ['team_evaluation_funding_window_below_planned_runtime']
