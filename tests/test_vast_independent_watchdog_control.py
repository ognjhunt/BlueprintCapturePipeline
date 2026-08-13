from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import vast_independent_watchdog_control as control


class _FakeProcess:
    pid = 4321

    def __init__(self, command: list[str], **kwargs: Any) -> None:
        self.command = command
        self.kwargs = kwargs
        self.returncode: int | None = None
        out_dir = Path(command[command.index("--out-dir") + 1])
        prefix = command[command.index("--pod-name-prefix") + 1]
        payload = {
            "status": "armed",
            "independent_process": True,
            "pid": self.pid,
            "pod_name_prefix": prefix,
            "provider": "vast",
        }
        (out_dir / control.EVIDENCE_NAME).write_text(json.dumps(payload), encoding="utf-8")

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test_watchdog_is_armed_detached_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control.subprocess, "Popen", _FakeProcess)

    handoff, handle = control.arm_independent_vast_watchdog(
        job_dir=tmp_path,
        max_live_minutes=3,
        generated_at="2026-07-27T00:00:00+00:00",
        allowed_active_instance_ids=[47226054],
    )

    assert handoff["status"] == "armed"
    assert handoff["watchdog_armed_before_allocation"] is True
    assert handle is not None
    assert handle.process.kwargs["start_new_session"] is True
    assert handle.process.kwargs["stdin"] is control.subprocess.DEVNULL
    assert handle.pod_name_prefix.startswith("blueprint-groot-oscar-canary-vast-wam-")
    assert handle.allowed_active_instance_ids == (47226054,)
    assert handoff["allowed_active_instance_ids"] == [47226054]
    index = handle.process.command.index("--allowed-active-instance-id")
    assert handle.process.command[index + 1] == "47226054"


def test_paired_native_import_prefix_is_accepted_by_real_watchdog_contract(
    tmp_path: Path,
) -> None:
    import time

    from blueprint_pipeline.groot_oscar_runpod_watchdog import arm_watchdog

    receipt = arm_watchdog(
        out_dir=tmp_path,
        pod_name_prefix="blueprint-adp-paired-native-import-bound-run-",
        deadline_epoch=time.time() + 120,
        provider_name="vast",
    )

    assert receipt["status"] == "armed"
    assert receipt["provider"] == "vast"
    assert receipt["pre_deadline_provider_mutation_allowed"] is False


def test_watchdog_exact_instance_handoff_is_atomic_and_private(tmp_path: Path) -> None:
    path = tmp_path / "watchdog" / "started_vast_instance_id.txt"

    control.write_started_vast_instance_id(path, 46031731)

    assert path.read_text(encoding="utf-8") == "46031731\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not path.with_suffix(".tmp").exists()


def test_watchdog_process_start_failure_blocks_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_start(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("synthetic start failure")

    monkeypatch.setattr(control.subprocess, "Popen", fail_start)

    handoff, handle = control.arm_independent_vast_watchdog(
        job_dir=tmp_path,
        max_live_minutes=3,
        generated_at="2026-07-27T00:00:00+00:00",
    )

    assert handle is None
    assert handoff["status"] == "blocked"
    assert handoff["watchdog_armed_before_allocation"] is False
    assert handoff["provider_mutations_performed"] == 0
    assert handoff["blockers"] == ["independent_vast_watchdog_process_start_failed"]


def test_watchdog_rejects_one_minute_ttl_before_process_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        control.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("short TTL must block before process start"),
    )

    handoff, handle = control.arm_independent_vast_watchdog(
        job_dir=tmp_path,
        max_live_minutes=1,
        generated_at="2026-07-27T00:00:00+00:00",
    )

    assert handle is None
    assert handoff["status"] == "blocked"
    assert handoff["blockers"] == ["independent_vast_watchdog_ttl_too_short"]


def test_watchdog_without_allocation_is_cancelled_without_provider_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control.subprocess, "Popen", _FakeProcess)
    _handoff, handle = control.arm_independent_vast_watchdog(
        job_dir=tmp_path,
        max_live_minutes=3,
        generated_at="2026-07-27T00:00:00+00:00",
    )
    assert handle is not None

    result = control.close_independent_vast_watchdog(
        job_dir=tmp_path,
        handle=handle,
        instance_ids=[],
        provider_teardown_completed=True,
        provider_allocation_impossible=True,
    )

    assert result["status"] == "cancelled_no_allocation"
    assert result["provider_mutations_performed"] == 0
    assert handle.process.poll() == -15


def test_watchdog_stays_armed_when_create_identity_is_ambiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control.subprocess, "Popen", _FakeProcess)
    _handoff, handle = control.arm_independent_vast_watchdog(
        job_dir=tmp_path,
        max_live_minutes=3,
        generated_at="2026-07-27T00:00:00+00:00",
    )
    assert handle is not None

    result = control.close_independent_vast_watchdog(
        job_dir=tmp_path,
        handle=handle,
        instance_ids=[],
        provider_teardown_completed=True,
    )

    assert result["status"] == "retained_until_hard_ttl"
    assert result["reason"] == "provider_allocation_identity_ambiguous"
    assert handle.process.poll() is None


def test_close_returns_terminal_as_soon_as_evidence_confirms_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow watchdog must not be misread as unclosed.

    v7/v9 wrote provider_terminal evidence ~2-3 minutes after the owner cancel
    request while close only waited 45 seconds for process exit and read the
    evidence once, so a correctly closing watchdog was recorded as
    joint_agent_independent_watchdog_not_closed. Close now polls the evidence
    during the wait and returns terminal as soon as absence is confirmed.
    """

    import threading
    import time as time_module

    monkeypatch.setattr(control.subprocess, "Popen", _FakeProcess)
    _handoff, handle = control.arm_independent_vast_watchdog(
        job_dir=tmp_path,
        max_live_minutes=3,
        generated_at="2026-08-09T00:00:00+00:00",
    )
    assert handle is not None

    def _write_terminal_evidence() -> None:
        (handle.out_dir / control.EVIDENCE_NAME).write_text(
            json.dumps(
                {
                    "status": "provider_terminal",
                    "provider_absence_confirmed": True,
                    "provider_mutations_performed": 0,
                }
            ),
            encoding="utf-8",
        )

    threading.Timer(0.5, _write_terminal_evidence).start()
    started = time_module.monotonic()
    result = control.close_independent_vast_watchdog(
        job_dir=tmp_path,
        handle=handle,
        instance_ids=[47283980],
        provider_teardown_completed=True,
        wait_seconds=30.0,
    )
    elapsed = time_module.monotonic() - started

    assert result["status"] == "provider_terminal"
    assert result["provider_absence_confirmed"] is True
    assert elapsed < 10.0


def test_close_default_wait_covers_slow_absence_confirmation() -> None:
    import inspect

    signature = inspect.signature(control.close_independent_vast_watchdog)
    assert signature.parameters["wait_seconds"].default >= 300.0
