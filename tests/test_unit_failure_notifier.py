"""A failed unit is recorded, and reported once and then hourly on the capacity alert route."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_capacity_controller as cap
from blueprint_pipeline import unit_failure_notifier as notifier

REPO_ROOT = Path(__file__).resolve().parents[1]
UNIT = "blueprint-gpu-spend-guard.service"
HOOK = "https://alerts.example/hook"
TIMEOUT = {"MONITOR_UNIT": UNIT, "MONITOR_SERVICE_RESULT": "timeout", "MONITOR_EXIT_CODE": "killed",
           "MONITOR_EXIT_STATUS": "TERM", "MONITOR_INVOCATION_ID": "0123456789abcdef0123456789abcdef"}


def _fail(root: Path, now: float, posted: list, *, environ=TIMEOUT, unit: str = UNIT, webhook: str = HOOK):
    return notifier.record_unit_failure(
        unit, record_root=root, webhook_url=webhook, environ=environ, now=now,
        poster=lambda url, report: posted.append((url, report)),
    )


def test_a_unit_failing_every_minute_posts_once_and_then_hourly(tmp_path: Path) -> None:
    posted: list = []
    start = 1_800_000_000.0
    for minute in range(121):
        record = _fail(tmp_path, start + 60 * minute, posted)
    assert [report["alerts"][0]["failure_count"] for _url, report in posted] == [1, 61, 121]
    assert all(url == HOOK for url, _report in posted)
    assert record["failure_count"] == 121 and record["alert_posted"] is True
    assert record["first_failure_epoch"] == start and record["last_alert_epoch"] == start + 7200
    saved = json.loads((tmp_path / f"{UNIT}.json").read_text(encoding="utf-8"))
    assert saved == record and saved["schema_version"] == notifier.SCHEMA_VERSION
    assert saved["result"] == "timeout" and saved["exit_status"] == "TERM"
    assert cap.ALERT_REPEAT_SECONDS == 3600  # the controller's own repeat interval is the dedupe


def test_each_unit_is_deduplicated_on_its_own(tmp_path: Path) -> None:
    posted: list = []
    other = "blueprint-pipeline-intake.service"
    _fail(tmp_path, 1000.0, posted)
    _fail(tmp_path, 1030.0, posted, unit=other, environ={})
    _fail(tmp_path, 1060.0, posted)
    assert [report["alerts"][0]["unit"] for _url, report in posted] == [UNIT, other]
    assert sorted(path.name for path in tmp_path.iterdir()) == [f"{UNIT}.json", f"{other}.json"]


def test_a_failed_post_is_recorded_and_retried_on_the_next_failure(tmp_path: Path) -> None:
    def refuse(_url, _report):
        raise OSError("https://alerts.example/hook?token=secret refused")

    first = notifier.record_unit_failure(UNIT, record_root=tmp_path, webhook_url=HOOK, environ=TIMEOUT,
                                         now=1000.0, poster=refuse)
    assert first["alert_posted"] is False and first["alert_error"] == "OSError"
    assert "secret" not in (tmp_path / f"{UNIT}.json").read_text(encoding="utf-8")
    posted: list = []
    second = _fail(tmp_path, 1060.0, posted)
    assert len(posted) == 1 and second["alert_posted"] is True and second["alert_error"] is None
    assert second["failure_count"] == 2
    _fail(tmp_path, 1120.0, posted)
    assert len(posted) == 1, "delivered: the hourly repeat applies again"


def test_without_a_webhook_the_failure_is_still_recorded(tmp_path: Path) -> None:
    posted: list = []
    record = _fail(tmp_path, 1000.0, posted, webhook="")
    assert posted == [] and record["alert_posted"] is False
    assert record["alert_error"] == "operator_alert_route_unconfigured"
    # Once the route exists the next failure reports at once.
    assert _fail(tmp_path, 1060.0, posted)["alert_posted"] is True and len(posted) == 1


def test_what_systemd_did_not_say_stays_unknown(tmp_path: Path) -> None:
    posted: list = []
    silent = _fail(tmp_path, 1000.0, posted, environ={})
    assert silent["result"] is None and silent["exit_status"] is None and silent["invocation_id"] is None
    assert posted[0][1]["alerts"][0]["code"] == f"unit_failed:{UNIT}:unknown"
    # Variables that describe another unit, or are not a plain word, are not taken.
    assert notifier.observed_failure(UNIT, {**TIMEOUT, "MONITOR_UNIT": "blueprint-other.service"}) == {
        "result": None, "exit_code": None, "exit_status": None, "invocation_id": None}
    assert notifier.observed_failure(UNIT, {**TIMEOUT, "MONITOR_SERVICE_RESULT": "a b\nc"})["result"] is None
    coded = notifier.alert_report({"unit": UNIT, "result": "exit-code", "exit_status": "2"})
    assert coded["alerts"][0]["code"] == f"unit_failed:{UNIT}:exit-code:2"


@pytest.mark.parametrize("unit", ["../etc/passwd.service", "ssh.service", "blueprint-x.socket", "blueprint-a/b.service", ""])
def test_only_a_blueprint_unit_name_becomes_a_record(tmp_path: Path, unit: str) -> None:
    with pytest.raises(ValueError, match="unit_failure_unit_name_invalid"):
        notifier.record_unit_failure(unit, record_root=tmp_path, webhook_url="", environ={}, now=1.0)
    assert list(tmp_path.iterdir()) == []


def test_it_posts_with_the_capacity_controllers_own_webhook_client(tmp_path: Path, monkeypatch) -> None:
    payloads = []

    class Response:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def urlopen(request, *, timeout):
        payloads.append((request.full_url, json.loads(request.data)))
        return Response()

    monkeypatch.setattr(cap.urllib.request, "urlopen", urlopen)
    record = notifier.record_unit_failure(UNIT, record_root=tmp_path, webhook_url=HOOK, environ=TIMEOUT, now=1000.0)
    assert record["alert_posted"] is True
    url, payload = payloads[0]
    assert url == HOOK and payload["schema_version"] == "control_plane_capacity_alert.v1"
    assert payload["page"] is True and payload["severity"] == "page"
    # What an operator reads in the channel names the unit and how it failed.
    assert f"systemd unit_failed:{UNIT}:timeout" in payload["text"]
    assert payload["summary"] == f"control plane: unit_failed:{UNIT}:timeout"
    assert payload["alerts"][0]["unit_runbook"] == notifier.RUNBOOK
    # The controller refuses a route that is not https; the notifier records that and does not fail.
    plain = notifier.record_unit_failure("blueprint-pipeline-intake.service", record_root=tmp_path,
                                         webhook_url="http://alerts.example/hook", environ={}, now=1000.0)
    assert plain["alert_posted"] is False and plain["alert_error"] == "ControlPlaneCapacityError"


def test_main_records_from_the_units_environment_and_never_fails_on_a_dead_route(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(notifier.RECORD_ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(cap.WEBHOOK_URL_ENV, HOOK)
    for key, value in TIMEOUT.items():
        monkeypatch.setenv(key, value)

    def unreachable(*_args, **_kwargs):
        raise OSError("no route")

    monkeypatch.setattr(cap.urllib.request, "urlopen", unreachable)
    assert notifier.main(["--unit", UNIT]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["unit"] == UNIT and printed["result"] == "timeout" and printed["alert_error"] == "OSError"
    assert json.loads((tmp_path / f"{UNIT}.json").read_text(encoding="utf-8")) == printed


def test_importing_the_notifier_loads_nothing_heavy() -> None:
    probe = ("import sys, blueprint_pipeline.unit_failure_notifier; "
             "print(sorted(m for m in sys.modules if m.startswith('blueprint_pipeline.')))")
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True,
                            env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert json.loads(result.stdout.replace("'", '"')) == [
        "blueprint_pipeline.core", "blueprint_pipeline.core.common", "blueprint_pipeline.unit_failure_notifier"]
