from __future__ import annotations

from blueprint_pipeline import adp009d_provisioning_preflight as preflight


def test_every_missing_prerequisite_is_reported_not_just_the_first() -> None:
    """Three paid runs each learned one missing dependency; that is the bug."""

    report = preflight.collect_preflight()

    assert report["reports_all_failures_not_the_first"] is True
    # Whatever this host lacks, the report enumerates rather than short-circuits.
    assert isinstance(report["blockers"], list)
    assert report["status"] in {"ready", "incomplete"}
    if report["blockers"]:
        assert report["status"] == "incomplete"


def test_the_checks_are_the_ones_live_runs_actually_failed_on(monkeypatch) -> None:
    """Nothing speculative: each entry broke a run or is issued by the script."""

    assert "linux/input.h" in preflight.REQUIRED_HEADERS  # broke v37
    assert "/usr/bin/python3" in preflight.REQUIRED_INTERPRETERS  # broke v34
    for command in ("curl", "git", "apt-get"):
        assert command in preflight.REQUIRED_COMMANDS
    # Python.h broke v38 and is probed via sysconfig, not a guessed path.
    report = preflight.collect_preflight()
    assert "Python.h" in report["checked_headers"]


def test_a_fully_equipped_host_reports_ready(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "_command_present", lambda name: True)
    monkeypatch.setattr(preflight, "_header_present", lambda name: True)
    monkeypatch.setattr(preflight, "_python_header_present", lambda path: True)
    monkeypatch.setattr(preflight.Path, "is_file", lambda self: True)

    report = preflight.collect_preflight()

    assert report["status"] == "ready"
    assert report["blockers"] == []


def test_all_three_historical_failures_would_have_surfaced_together(
    monkeypatch,
) -> None:
    """The whole point: one run would have reported what took three."""

    monkeypatch.setattr(preflight, "_command_present", lambda name: name != "gcc")
    monkeypatch.setattr(preflight, "_header_present", lambda name: False)
    monkeypatch.setattr(preflight, "_python_header_present", lambda path: False)
    monkeypatch.setattr(preflight.Path, "is_file", lambda self: False)

    report = preflight.collect_preflight()

    assert "missing_header:linux/input.h" in report["blockers"]
    assert "missing_header:Python.h" in report["blockers"]
    assert "missing_interpreter:/usr/bin/python3" in report["blockers"]
    # Three separate paid discoveries, present in one report.
    assert len(report["blockers"]) >= 3


def test_either_compiler_name_satisfies_the_compiler_check(monkeypatch) -> None:
    """cc and gcc are interchangeable; only both absent is a real failure."""

    monkeypatch.setattr(preflight, "_header_present", lambda name: True)
    monkeypatch.setattr(preflight, "_python_header_present", lambda path: True)
    monkeypatch.setattr(preflight.Path, "is_file", lambda self: True)

    monkeypatch.setattr(preflight, "_command_present", lambda name: name != "gcc")
    assert "missing_command:c_compiler" not in preflight.collect_preflight()["blockers"]

    monkeypatch.setattr(
        preflight, "_command_present", lambda name: name not in ("gcc", "cc")
    )
    assert "missing_command:c_compiler" in preflight.collect_preflight()["blockers"]


def test_the_preflight_is_never_fatal_and_runs_before_and_after_apt() -> None:
    """It reports; the install that follows fixes most of what it finds."""

    from blueprint_pipeline.adp009d_policy_provisioning import (
        build_provisioning_script,
    )

    script = build_provisioning_script("pi05_droid")
    assert script.count("adp009d_provisioning_preflight.py") == 2
    assert "preflight_before.json" in script
    assert "preflight_after.json" in script
    # Before the apt step, and again after it, so the fix is proven.
    assert script.index("preflight_before") < script.index("apt-get install")
    assert script.index("apt-get install") < script.index("preflight_after")
    # Never fatal on its own.
    for marker in ("preflight_before.json", "preflight_after.json"):
        line = [ln for ln in script.splitlines() if marker in ln][0]
        assert line.rstrip().endswith("|| true")
