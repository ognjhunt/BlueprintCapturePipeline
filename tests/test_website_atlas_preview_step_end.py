"""A step that outruns the unit's plan is ended by the tick, with its traceback, not two hours later by the unit.

2026-11-24, 11:41: one job's scale review, slowed at the batch slice's memory bound,
held the serial tick for two hours with 21 minutes of CPU; no other job's preview
advanced until the unit's ``TimeoutStartSec`` killed it, and nothing said where it stood.
"""
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from blueprint_pipeline import website_atlas_preview as preview

SRC = str(Path(preview.__file__).resolve().parents[1])


def run(code: str, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, check=False,
                          timeout=timeout, env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"})


def test_the_step_end_leaves_the_later_passes_their_time_inside_the_units_limit():
    assert preview.STEP_END_SECONDS == preview.UNIT_TIMEOUT_SECONDS - preview.LATER_PASSES_SECONDS == 65 * 60
    assert preview.STEP_START_BUDGET_SECONDS < preview.STEP_END_SECONDS < preview.UNIT_TIMEOUT_SECONDS


def test_a_step_stuck_in_native_code_is_ended_at_its_bound_naming_where_it_stood():
    """The process ends as the unit's limit ends it (status 1, nothing after the step runs), with the traceback."""
    started = time.monotonic()
    done = run("""
        import re
        from blueprint_pipeline import website_atlas_preview as preview

        def stuck_scale_review_measurement():
            re.match(r"(a+)+$", "a" * 60 + "b")  # holds the interpreter in native code for hours

        with preview._ended_if_still_running(1.0):
            stuck_scale_review_measurement()
        print("returned")
    """)
    assert done.returncode == 1 and "returned" not in done.stdout
    assert time.monotonic() - started < 30
    assert "Timeout" in done.stderr and "stuck_scale_review_measurement" in done.stderr


def test_a_step_that_returns_in_time_is_not_ended_after_it():
    done = run("""
        import time
        from blueprint_pipeline import website_atlas_preview as preview

        with preview._ended_if_still_running(1.0):
            pass
        try:
            with preview._ended_if_still_running(1.0):
                raise ValueError("a refusal")
        except ValueError:
            pass
        time.sleep(2.5)
        print("tick finished")
    """)
    assert done.returncode == 0 and "tick finished" in done.stdout and "Timeout" not in done.stderr


def _jobs(tmp_path, *names):
    paths = {}
    for name in names:
        path = tmp_path / f"bucket/scenes/site-req{name}/captures/walkthrough-req{name}/pipeline/atlas" / preview.JOB_FILE
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({}))
        paths[path] = name
    return paths


@pytest.fixture
def bounds(monkeypatch):
    calls = []
    monkeypatch.setattr(preview.faulthandler, "dump_traceback_later",
                        lambda seconds, **options: calls.append(("armed", seconds, options)))
    monkeypatch.setattr(preview.faulthandler, "cancel_dump_traceback_later", lambda: calls.append(("cancelled",)))
    return calls


def test_the_units_tick_bounds_each_step_at_the_plans_end_and_disarms_after_it(monkeypatch, tmp_path, bounds):
    jobs = _jobs(tmp_path, "a", "b")
    elapsed, stepped = [100.0], []

    def advance(path, *, now):
        stepped.append(jobs[path])
        assert bounds[-1][0] == "armed", "the bound is armed while the step runs"
        elapsed[0] += 600
        if jobs[path] == "a":
            raise RuntimeError("an unreadable job")
        return "processing"
    monkeypatch.setattr(preview, "advance_job", advance)
    monkeypatch.setattr(preview, "_read", lambda path: json.loads(path.read_text()))

    counts = preview.tick(tmp_path, now=lambda: 0, clock=lambda: elapsed[0], budget_started=40.0,
                          end_overrun_steps=True)
    assert counts == {"unreadable": 1, "processing": 1} and stepped == ["a", "b"]
    end = 40.0 + preview.STEP_END_SECONDS
    assert bounds == [("armed", end - 100.0, {"exit": True, "file": 2}), ("cancelled",),
                      ("armed", end - 700.0, {"exit": True, "file": 2}), ("cancelled",)]


def test_a_tick_run_without_the_bound_never_arms_it(monkeypatch, tmp_path, bounds):
    _jobs(tmp_path, "a")
    monkeypatch.setattr(preview, "advance_job", lambda path, *, now: "processing")
    monkeypatch.setattr(preview, "_read", lambda path: json.loads(path.read_text()))
    assert preview.tick(tmp_path, now=lambda: 0) == {"processing": 1}
    assert bounds == []


def test_the_tick_command_bounds_its_steps(monkeypatch, tmp_path):
    from blueprint_pipeline import owned_capture_atlas_worker
    seen = {}
    monkeypatch.setattr(preview, "trigger_stop_reason", lambda: lambda: None)
    for name in ("company_clip_tick", "company_simulation_tick", "job_thumbnail_tick"):
        monkeypatch.setattr(preview, name, lambda root, stop=None: {})
    monkeypatch.setattr(preview, "scene_preview_tick", lambda root: {})
    monkeypatch.setattr(owned_capture_atlas_worker, "tick_from_environment", dict)
    monkeypatch.setattr(preview, "tick", lambda root, **options: seen.update(options) or {})
    assert preview.main(["tick", "--storage-root", str(tmp_path)]) == 0
    assert seen["end_overrun_steps"] is True and isinstance(seen["budget_started"], float)
