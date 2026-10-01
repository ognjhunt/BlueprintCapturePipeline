"""Failures retain truthful reports and the deadline also interrupts parent work."""

import json
import socket
import time
from types import SimpleNamespace

import pytest


def test_whole_run_deadline_interrupts_synchronous_work_and_restores_alarm():
    import signal
    from scripts.control_plane_concurrency_reporting import whole_run_deadline

    previous = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="global_timeout"):
        with whole_run_deadline(time.monotonic() + 0.05):
            time.sleep(2)
    assert time.monotonic() - started < 1
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)


def test_unreadable_measurements_and_malformed_child_still_seal_failed_report(tmp_path):
    from scripts.control_plane_concurrency_reporting import finish_report

    roots = {name: tmp_path / name for name in ("control_plane", "objects", "workers")}
    for root in roots.values():
        root.mkdir()
    malformed = roots["workers"] / "child.json"
    malformed.write_text("{incomplete")
    args = SimpleNamespace(
        report=tmp_path / "report.json",
        expected_beta_concurrency=1,
        owner_confirmed_concurrency=False,
        maximum_retained_gib=0.1,
        maximum_delta_gib=0.1,
    )

    def unreadable(root):
        raise ValueError("allocation_measurement_incomplete")

    sampler = SimpleNamespace(
        peak_bytes=10,
        sample_count=1,
        incomplete_scan_count=1,
        samples=[],
        interval_seconds=0.1,
    )
    result = finish_report(
        args=args,
        source="1" * 40,
        release={"source_commit": "1" * 40},
        confinement={"contract_fixture": True},
        roots=roots,
        baseline=10,
        children=[{"result_path": malformed, "scene_key": "one"}],
        scenes=[],
        sampler=sampler,
        blockers=["child_failed:one:1"],
        retirement=None,
        peak=0,
        held=0,
        measure=unreadable,
    )
    report = json.loads(args.report.read_text())
    assert result == 1 and report["status"] == "failed"
    assert report["final_allocated_bytes"] is None
    assert report["control_plane_delta_bytes"] is None
    assert report["worker_allocated_bytes"] is None
    assert report["owner_sized_acceptance_complete"] is False
    assert any("allocation_measurement_incomplete" in value for value in report["blockers"])
    assert any("JSONDecodeError" in value for value in report["blockers"])
    assert report["host"]["hostname"] == socket.gethostname()
    assert report["release_binding"]["source_commit"] == "1" * 40
    assert malformed.read_text() == "{incomplete"


def test_retained_failed_child_receipt_must_have_original_digest(tmp_path):
    from scripts.control_plane_concurrency_reporting import failed_child_scenes

    child = tmp_path / "result.json"
    child.write_text(json.dumps({"scene_id": "one", "stages": [], "receipt_digest": "wrong"}))
    blockers = []
    assert failed_child_scenes([{"result_path": child, "scene_key": "one"}], blockers) == []
    assert any("child_receipt_changed" in value for value in blockers)
