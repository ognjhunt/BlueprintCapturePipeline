from __future__ import annotations

import signal
import subprocess
from pathlib import Path

from blueprint_pipeline import impacted_test_selection as MODULE


ROOT = Path(__file__).resolve().parents[1]


def test_documentation_change_uses_sentinels_without_full_suite() -> None:
    plan = MODULE.build_plan(ROOT, ["docs/example.md"])

    assert plan["requires_full_suite"] is False
    assert set(MODULE.SENTINEL_TESTS).issubset(plan["selected_tests"])
    assert plan["selected_tests"] == sorted(plan["selected_tests"])


def test_workflow_change_adds_release_contract_tests_without_full_suite() -> None:
    plan = MODULE.build_plan(ROOT, [".github/workflows/ci.yml"])

    assert plan["requires_full_suite"] is False
    assert "tests/test_release_engineering_contracts.py" in plan["selected_tests"]


def test_source_change_maps_direct_and_importing_tests() -> None:
    plan = MODULE.build_plan(
        ROOT,
        ["src/blueprint_pipeline/paid_resource_admission.py"],
    )

    assert plan["requires_full_suite"] is False
    assert "tests/test_paid_resource_admission.py" in plan["selected_tests"]


def test_unmapped_executable_and_dependency_changes_request_full_suite() -> None:
    unmapped = MODULE.build_plan(ROOT, ["scripts/brand_new_uncovered_launcher.py"])
    dependency = MODULE.build_plan(ROOT, ["pyproject.toml"])

    assert unmapped["requires_full_suite"] is True
    assert "unmapped_executable_surface:scripts/brand_new_uncovered_launcher.py" in unmapped[
        "reasons"
    ]
    assert dependency["requires_full_suite"] is True
    assert "cross_cutting_file:pyproject.toml" in dependency["reasons"]


def test_changed_test_is_selected_directly() -> None:
    plan = MODULE.build_plan(ROOT, ["tests/test_capture_qa.py"])

    assert plan["requires_full_suite"] is False
    assert "tests/test_capture_qa.py" in plan["selected_tests"]


def test_build_loop_default_is_two_minutes() -> None:
    args = MODULE.parse_args([])

    assert MODULE.DEFAULT_TIMEOUT_SECONDS == 120
    assert args.timeout_seconds == 120


def test_timeout_terminates_the_entire_pytest_process_group(monkeypatch) -> None:
    signals: list[tuple[int, int]] = []

    class FakeProcess:
        pid = 12345
        waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired(["pytest"], timeout)
            return -signal.SIGTERM

    monkeypatch.setattr(MODULE.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(MODULE.os, "killpg", lambda pid, sent_signal: signals.append((pid, sent_signal)))

    status = MODULE.run_pytest(
        ROOT,
        selected_tests=["tests/test_impacted_test_selection.py"],
        timeout_seconds=1,
        junit_path=None,
    )

    assert status == 124
    assert signals == [(12345, signal.SIGTERM)]


def test_a_script_test_that_loads_its_subject_by_name_is_not_invisible(tmp_path: Path) -> None:
    """Otherwise every live-profile edit escalates to the full suite.

    Script tests in this repository import their subject with
    `_load("<stem>")` rather than by path, so the selector saw no link between
    `scripts/build_artifixer3d_live_profile.py` and the test that covers it,
    reported `unmapped_executable_surface`, and demanded the full suite for a
    change one file already tested.
    """

    root = tmp_path
    (root / "scripts").mkdir()
    (root / "tests").mkdir()
    (root / "scripts" / "build_widget_live_profile.py").write_text("x = 1\n", encoding="utf-8")
    (root / "tests" / "test_widget_live_lane.py").write_text(
        'builder = _load("build_widget_live_profile")\n', encoding="utf-8"
    )

    plan = MODULE.build_plan(root, ["scripts/build_widget_live_profile.py"])

    assert "tests/test_widget_live_lane.py" in plan["selected_tests"]
    assert plan["requires_full_suite"] is False


def test_a_bare_stem_inside_a_longer_name_does_not_count_as_coverage(tmp_path: Path) -> None:
    """Matching an unquoted stem would map a script to tests that never load it."""

    root = tmp_path
    (root / "scripts").mkdir()
    (root / "tests").mkdir()
    (root / "scripts" / "build_widget.py").write_text("x = 1\n", encoding="utf-8")
    (root / "tests" / "test_unrelated.py").write_text(
        'name = "build_widget_live_profile"\n', encoding="utf-8"
    )

    plan = MODULE.build_plan(root, ["scripts/build_widget.py"])

    assert "tests/test_unrelated.py" not in plan["selected_tests"]
    assert plan["requires_full_suite"] is True
