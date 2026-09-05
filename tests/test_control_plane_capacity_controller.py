"""Capacity is measured, forecast, alerted and grown before intake has to refuse anything."""

from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_capacity_controller as cap

Usage = namedtuple("Usage", "total used free")
GIB = 1024**3


def _usage(free_gib: float, total_gib: float = 154.0):
    total = int(total_gib * GIB)
    free = int(free_gib * GIB)
    return lambda _path: Usage(total, total - free, free)


def _reservation(root: Path, name: str, *, expected_bytes: int, expires_at: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.json").write_text(
        json.dumps({"expected_bytes": expected_bytes, "expires_at_epoch": expires_at}), encoding="utf-8"
    )


def test_measurement_projects_admission_exactly_as_intake_does(tmp_path: Path) -> None:
    """The host at 9.67 GiB free: an 8 GiB floor leaves 1.67 GiB, so every 2 GiB
    stage is refused.  That was the 503 nobody saw coming."""
    ledger = tmp_path / "reservations"
    _reservation(ledger, "live", expected_bytes=GIB, expires_at=2_000.0)
    _reservation(ledger, "expired", expected_bytes=5 * GIB, expires_at=500.0)

    row = cap.measure_mount("/var/lib/blueprint", reservation_root=ledger, disk_usage=_usage(9.67), now=1_000.0)

    assert row["status"] == "measured" and row["level"] == "critical"
    assert row["floor_bytes"] == 8 * GIB
    assert row["reserved_bytes"] == GIB and row["live_reservations"] == 1
    assert row["refused_roles"] == sorted(cap.CHAIN_ROLES)
    assert row["free_needed_for_one_role_bytes"] == 10 * GIB
    healthy = cap.measure_mount("/var/lib/blueprint", reservation_root=ledger, disk_usage=_usage(60.0), now=1_000.0)
    assert healthy["level"] == "ok" and healthy["refused_roles"] == []
    warning = cap.measure_mount("/var/lib/blueprint", reservation_root=ledger, disk_usage=_usage(40.0), now=1_000.0)
    assert warning["level"] == "warning" and warning["refused_roles"] == []


def test_forecast_uses_the_oldest_observation_inside_the_window() -> None:
    now = 10 * 86400.0
    history = [
        {"mount": "/m", "status": "measured", "observed_at_epoch": now - 2 * 86400, "free_bytes": 40 * GIB, "floor_bytes": 8 * GIB},
        {"mount": "/m", "status": "measured", "observed_at_epoch": now - 30 * 86400, "free_bytes": 100 * GIB, "floor_bytes": 8 * GIB},
        {"mount": "/other", "status": "measured", "observed_at_epoch": now - 86400, "free_bytes": 1, "floor_bytes": 0},
    ]
    current = {"mount": "/m", "status": "measured", "free_bytes": 30 * GIB, "floor_bytes": 8 * GIB}

    result = cap.forecast(history, current, now=now)

    assert result["status"] == "growing"
    assert result["growth_bytes_per_day"] == 5 * GIB
    assert result["days_until_floor"] == pytest.approx(4.4, abs=0.01)
    assert cap.forecast([], current, now=now) == {"status": "insufficient_history"}
    assert cap.forecast(history[:1], {**current, "free_bytes": 50 * GIB}, now=now)["status"] == "not_growing"


def test_controller_writes_evidence_alerts_on_escalation_and_repeats_hourly_while_critical(
    tmp_path: Path,
) -> None:
    posted: list[tuple[str, str]] = []

    def poster(url: str, report) -> None:
        posted.append((url, report["level"]))

    common = dict(
        mounts=["/var/lib/blueprint"],
        report_root=tmp_path / "capacity",
        reservation_root=tmp_path / "reservations",
        webhook_url="https://alerts.example/hook",
        volume=None,
        ack="",
        token="",
        poster=poster,
    )
    ok = cap.run_controller(**common, disk_usage=_usage(80.0), now=1_000.0)
    assert ok["level"] == "ok" and posted == []
    assert (tmp_path / "capacity" / "latest.json").is_file()

    critical = cap.run_controller(**common, disk_usage=_usage(9.0), now=2_000.0)
    assert critical["level"] == "critical" and critical["alert_posted"] is True
    assert posted == [("https://alerts.example/hook", "critical")]
    assert {a["code"] for a in critical["alerts"]} == {"admission_refused", "utilization_critical"}

    again = cap.run_controller(**common, disk_usage=_usage(9.0), now=2_600.0)
    assert again["alert_posted"] is False and len(posted) == 1
    later = cap.run_controller(**common, disk_usage=_usage(9.0), now=2_000.0 + 3_601)
    assert later["alert_posted"] is True and len(posted) == 2

    history = (tmp_path / "capacity" / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history) == 4
    latest = json.loads((tmp_path / "capacity" / "latest.json").read_text(encoding="utf-8"))
    assert latest["report_digest"] == later["report_digest"]


def test_volume_grows_one_step_only_when_critical_acknowledged_and_under_the_maximum(
    tmp_path: Path,
) -> None:
    report_ok = cap.build_capacity_report(
        mounts=["/mnt/work"], reservation_root=tmp_path / "r", disk_usage=_usage(60.0), now=1.0
    )
    assert cap.plan_volume_resize(report_ok, volume_id="vol-1", volume_mount="/mnt/work", current_size_gib=100, max_gib=500) is None
    report_full = cap.build_capacity_report(
        mounts=["/mnt/work"], reservation_root=tmp_path / "r", disk_usage=_usage(9.0), now=1.0
    )
    plan = cap.plan_volume_resize(report_full, volume_id="vol-1", volume_mount="/mnt/work", current_size_gib=100, max_gib=500)
    assert plan == {
        "status": "planned",
        "volume_id": "vol-1",
        "mount": "/mnt/work",
        "current_size_gib": 100,
        "target_size_gib": 150,
    }
    capped = cap.plan_volume_resize(report_full, volume_id="vol-1", volume_mount="/mnt/work", current_size_gib=500, max_gib=500)
    assert capped["status"] == "blocked" and capped["reason"] == "volume_at_maximum"

    calls: list = []

    def api(url, *, token, method, payload=None):
        calls.append((url, token, method, payload))
        return {"action": {"status": "in-progress"}}

    class Done:
        returncode = 0

    def runner(command, **_kwargs):
        calls.append(tuple(command))
        return Done()

    with pytest.raises(cap.ControlPlaneCapacityError, match="not_acknowledged"):
        cap.resize_volume(plan, ack="", token="t", device="/dev/sda", api=api, runner=runner)
    receipt = cap.resize_volume(plan, ack=cap.RESIZE_ACK, token="t", device="/dev/sda", api=api, runner=runner, now=5.0)
    assert receipt["status"] == "applied" and receipt["to_size_gib"] == 150
    assert receipt["provider_mutation_performed"] is True
    assert calls[0] == (
        "https://api.digitalocean.com/v2/volumes/vol-1/actions",
        "t",
        "POST",
        {"type": "resize", "size_gigabytes": 150},
    )
    assert calls[1] == ("resize2fs", "/dev/sda")

    def rejecting(url, *, token, method, payload=None):
        return {"action": {"status": "errored"}}

    with pytest.raises(cap.ControlPlaneCapacityError, match="resize_rejected"):
        cap.resize_volume(plan, ack=cap.RESIZE_ACK, token="t", device="/dev/sda", api=rejecting, runner=runner)


def test_controller_blocks_resize_without_acknowledgement_and_records_the_plan(tmp_path: Path) -> None:
    volume = {
        "id": "vol-1",
        "mount": "/var/lib/blueprint",
        "device": "/dev/sda",
        "current_size_gib": 100,
        "max_gib": 300,
        "step_gib": 50,
    }
    common = dict(
        mounts=["/var/lib/blueprint"],
        report_root=tmp_path / "capacity",
        reservation_root=tmp_path / "r",
        webhook_url="",
        volume=volume,
        poster=lambda *_args: None,
    )
    blocked = cap.run_controller(**common, ack="", token="", disk_usage=_usage(9.0), now=1.0)
    assert blocked["volume_resize"]["status"] == "blocked"
    assert blocked["volume_resize"]["reason"] == "resize_not_acknowledged"

    resized: list = []

    def resizer(plan, **_kwargs):
        resized.append(plan)
        return {"status": "applied", "to_size_gib": plan["target_size_gib"]}

    applied = cap.run_controller(
        **common, ack=cap.RESIZE_ACK, token="tok", resizer=resizer, disk_usage=_usage(9.0), now=2.0
    )
    assert applied["volume_resize"] == {"status": "applied", "to_size_gib": 150}
    assert len(resized) == 1
    healthy = cap.run_controller(**common, ack=cap.RESIZE_ACK, token="tok", resizer=resizer, disk_usage=_usage(80.0), now=3.0)
    assert healthy["volume_resize"] == {"status": "not_needed"} and len(resized) == 1
