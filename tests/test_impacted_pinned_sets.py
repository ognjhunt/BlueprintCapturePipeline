"""Tests that assert a complete set are selected whenever a member can change."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from blueprint_pipeline import impacted_pinned_sets as pinned
from blueprint_pipeline import impacted_test_selection as selection

ROOT = Path(__file__).resolve().parents[1]


def _write(root: Path, relative: str, text: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _write(tmp_path, "deploy/systemd/a.timer")
    _write(tmp_path, "deploy/systemd/a.service")
    _write(tmp_path, "src/blueprint_pipeline/__init__.py")
    _write(tmp_path, "src/blueprint_pipeline/codes.py", """\
        from .extra import EXTRA_CODES
        BASE = {"one"}
        PAGE_CODES = frozenset({*BASE, *EXTRA_CODES})
        """)
    _write(tmp_path, "src/blueprint_pipeline/extra.py", 'EXTRA_CODES = {"two"}\n')
    _write(tmp_path, "src/blueprint_pipeline/unrelated.py", "X = 1\n")
    _write(tmp_path, "src/blueprint_pipeline/rules.py", "RULE = 1\n")
    _write(tmp_path, "tests/test_units.py", """\
        from pathlib import Path
        ROOT = Path(__file__).resolve().parents[1]
        SYSTEMD = ROOT / "deploy" / "systemd"

        def test_every_timer():
            assert sorted(SYSTEMD.glob("*.timer"))

        def test_every_unit():
            assert list(SYSTEMD.iterdir())

        def test_one_unit():
            assert (SYSTEMD / "a.service").exists()
        """)
    _write(tmp_path, "tests/test_codes.py", """\
        import pytest
        from blueprint_pipeline import codes as alarm

        def test_codes_are_pinned():
            assert alarm.PAGE_CODES == {"one", "two"}

        def test_one_code():
            assert "one" in alarm.PAGE_CODES

        @pytest.mark.pins("blueprint_pipeline.rules")
        def test_a_rule_another_module_owns():
            assert True

        @pytest.mark.pins("deploy/systemd/*.service")
        class TestDeclaredOnAClass:
            def test_member(self):
                assert True
        """)
    return tmp_path


def _pinned(repo: Path, *changed: str) -> dict[str, set[str]]:
    sources = {path.relative_to(repo).as_posix(): path.read_text() for path in (repo / "tests").glob("test_*.py")}
    return pinned.pinned_tests(repo, changed=list(changed), test_sources=sources,
                               import_roots=selection.IMPORT_ROOTS)


def test_a_new_file_in_an_iterated_directory_selects_the_tests_that_list_it(repo):
    assert set(_pinned(repo, "deploy/systemd/b.timer")) == {
        "tests/test_units.py::test_every_timer", "tests/test_units.py::test_every_unit"}
    # The glob decides: a service is not a timer. A test that names one file pins no set.
    assert set(_pinned(repo, "deploy/systemd/b.service")) == {
        "tests/test_units.py::test_every_unit", "tests/test_codes.py::TestDeclaredOnAClass::test_member"}
    assert _pinned(repo, "deploy/other/b.timer") == {}


def test_a_constant_compared_with_a_literal_set_follows_its_definition(repo):
    expected = {"tests/test_codes.py::test_codes_are_pinned"}
    assert set(_pinned(repo, "src/blueprint_pipeline/codes.py")) == expected
    # A module the definition takes a name from defines the set too.
    assert set(_pinned(repo, "src/blueprint_pipeline/extra.py")) == expected
    assert _pinned(repo, "src/blueprint_pipeline/unrelated.py") == {}


def test_a_declared_pin_selects_the_test_when_its_module_changes(repo):
    assert _pinned(repo, "src/blueprint_pipeline/rules.py") == {
        "tests/test_codes.py::test_a_rule_another_module_owns": {"src/blueprint_pipeline/rules.py"}}


def test_the_plan_runs_pinned_tests_in_the_bounded_lane(repo):
    # No test names the new timer, so the file mapping finds nothing and asks for the full lane;
    # the tests that list the directory still run here, in the lane that reports first.
    plan = selection.build_plan(repo, ["deploy/systemd/b.timer"])
    assert plan["requires_full_suite"] is True
    for node in ("tests/test_units.py::test_every_timer", "tests/test_units.py::test_every_unit"):
        assert node in plan["selected_tests"]
        assert f"pinned_set:{node}:1" in plan["reasons"]
    assert "tests/test_units.py" not in plan["selected_tests"]
    # A file the bounded lane already runs whole is not added again node by node.
    plan = selection.build_plan(repo, ["tests/test_codes.py", "src/blueprint_pipeline/extra.py"])
    assert "tests/test_codes.py" in plan["selected_tests"]
    assert not any(row.startswith("tests/test_codes.py::") for row in plan["selected_tests"])


# A change of the kind that broke each of these tests on 2026-11-10, when only the full suite ran them.
BROKEN_ON_2026_11_10 = [
    ("deploy/systemd/blueprint-a-new-holdable-trigger.timer",
     "tests/test_operator_door_units.py::test_every_holdable_trigger_waits_for_boot_hold_reconciliation"),
    ("src/blueprint_pipeline/control_plane_stall_alarm.py",
     "tests/test_control_plane_capacity_controller.py::test_money_credit_spend_and_stall_pages_are_unchanged"),
    ("src/blueprint_pipeline/account_monthly_spend.py",
     "tests/test_control_plane_capacity_controller.py::test_money_credit_spend_and_stall_pages_are_unchanged"),
    ("src/blueprint_pipeline/website_preparation_status.py",
     "tests/test_website_preparation_terminal_status.py::test_status_version_is_opt_in_and_capped_at_newest_known"),
    ("src/blueprint_pipeline/website_gemini_receipts.py",
     "tests/test_atlas_native_splat.py::test_an_interrupted_review_is_asked_again_only_under_one_new_reservation"),
    ("src/blueprint_pipeline/website_gemini_receipts.py",
     "tests/test_task_object_rigidity.py::test_an_analysis_bought_before_rigidity_is_never_bought_again"),
]


@pytest.mark.parametrize("changed, test", BROKEN_ON_2026_11_10)
def test_each_pinned_test_that_failed_only_in_the_full_suite_is_now_selected(changed, test):
    sources = selection._test_sources(ROOT)
    reached = pinned.pinned_tests(ROOT, changed=[changed], test_sources=sources,
                                  import_roots=selection.IMPORT_ROOTS)
    assert reached.get(test) == {changed}


def test_every_declared_pin_names_something_that_exists():
    declared = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        source = path.read_text(encoding="utf-8", errors="replace")
        if pinned.DECLARED not in source:
            continue
        for node, targets in pinned.declared_targets(source):
            for target in targets:
                declared += 1
                if "/" in target:
                    assert target.endswith("/") or any(ROOT.glob(target)), (path.name, node, target)
                else:
                    assert pinned._module_file(ROOT, target, selection.IMPORT_ROOTS), (path.name, node, target)
    assert declared >= 3
