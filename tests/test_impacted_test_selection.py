from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline import impacted_test_selection as MODULE


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("changed_path", [
    "src/blueprint_pipeline/widget.py",
    "src/blueprint_pipeline/owned/__init__.py",
])
def test_unrelated_source_does_not_require_an_import_regex_scan(monkeypatch, changed_path):
    """No import pattern can match when its required module literal is absent."""
    calls = []
    original_search = MODULE.re.search

    def observe(pattern, source):
        calls.append(pattern)
        return original_search(pattern, source)

    monkeypatch.setattr(MODULE.re, "search", observe)
    assert MODULE._matching_tests(
        changed_path=changed_path,
        test_sources={"tests/test_unrelated.py": "import blueprint_pipeline.unrelated\n"},
    ) == set()
    assert calls == []


def test_documentation_change_uses_sentinels_without_full_suite() -> None:
    plan = MODULE.build_plan(ROOT, ["docs/example.md"])

    assert plan["requires_full_suite"] is False
    assert set(MODULE.SENTINEL_TESTS).issubset(plan["selected_tests"])
    assert plan["selected_tests"] == sorted(plan["selected_tests"])


def test_workflow_change_adds_release_contract_tests_without_full_suite() -> None:
    plan = MODULE.build_plan(ROOT, [".github/workflows/ci.yml"])

    assert plan["requires_full_suite"] is False
    assert "tests/test_release_engineering_contracts.py" in plan["selected_tests"]


def test_full_lane_workflow_change_executes_the_changed_full_suite() -> None:
    plan = MODULE.build_plan(ROOT, [".github/workflows/full-test-lane.yml"])

    assert plan["requires_full_suite"] is True
    assert "cross_cutting_file:.github/workflows/full-test-lane.yml" in plan["reasons"]


def test_source_change_maps_direct_and_importing_tests(tmp_path: Path) -> None:
    source = tmp_path / "src/blueprint_pipeline"
    source.mkdir(parents=True)
    tests = tmp_path / "tests"
    tests.mkdir()
    (source / "widget.py").write_text("value = 1\n")
    (tests / "test_widget.py").write_text("from blueprint_pipeline import widget\n")
    (tests / "test_widget_consumer.py").write_text("import blueprint_pipeline.widget\n")
    plan = MODULE.build_plan(tmp_path, ["src/blueprint_pipeline/widget.py"])
    assert plan["requires_full_suite"] is False
    assert {"tests/test_widget.py", "tests/test_widget_consumer.py"} <= set(plan["selected_tests"])
    for index in range(MODULE.MAX_IMPACTED_TEST_FILES):
        (tests / f"test_consumer_{index}.py").write_text("import blueprint_pipeline.widget\n")
    # This many consumers in a tree this small is more than a quarter of its tests.
    broad = MODULE.build_plan(tmp_path, ["src/blueprint_pipeline/widget.py"])
    assert broad["requires_full_suite"] is True
    assert broad["requires_wide_impacted"] is False
    assert any(reason.startswith("impacted_test_count_exceeds_budget:") for reason in broad["reasons"])
    assert any(reason.startswith("impacted_test_count_exceeds_wide_ceiling:") for reason in broad["reasons"])


def _widget_tree(tmp_path: Path, *, consumers: int, unrelated: int) -> set[str]:
    source = tmp_path / "src/blueprint_pipeline"
    source.mkdir(parents=True)
    tests = tmp_path / "tests"
    tests.mkdir()
    (source / "widget.py").write_text("value = 1\n")
    for index in range(consumers):
        (tests / f"test_consumer_{index}.py").write_text("import blueprint_pipeline.widget\n")
    for index in range(unrelated):
        (tests / f"test_unrelated_{index}.py").write_text("def test_nothing(): pass\n")
    return {f"tests/test_consumer_{index}.py" for index in range(consumers)}


def test_selection_above_the_budget_keeps_every_file_in_the_wide_lane(tmp_path: Path) -> None:
    consumers = _widget_tree(tmp_path, consumers=MODULE.MAX_IMPACTED_TEST_FILES + 1, unrelated=200)

    plan = MODULE.build_plan(tmp_path, ["src/blueprint_pipeline/widget.py"])

    assert plan["requires_full_suite"] is False
    assert plan["requires_wide_impacted"] is True
    assert f"impacted_test_count_wide:{len(consumers)}" in plan["reasons"]
    # Every test that references the changed file still runs before merge.
    assert set(plan["wide_impacted_tests"]) == consumers
    assert sorted(path for shard in plan["wide_shards"] for path in shard["files"]) == sorted(consumers)
    # The bounded lane stays bounded: its 480-second budget holds only the sentinels.
    assert set(plan["selected_tests"]) == set(MODULE.SENTINEL_TESTS)


def test_narrow_selection_never_requests_the_wide_lane(tmp_path: Path) -> None:
    consumers = _widget_tree(tmp_path, consumers=MODULE.MAX_IMPACTED_TEST_FILES, unrelated=200)

    plan = MODULE.build_plan(tmp_path, ["src/blueprint_pipeline/widget.py"])

    assert plan["requires_full_suite"] is False
    assert plan["requires_wide_impacted"] is False
    assert plan["wide_impacted_tests"] == []
    assert plan["wide_shards"] == []
    assert consumers <= set(plan["selected_tests"])


def test_absolute_wide_ceiling_still_requests_the_full_suite(tmp_path: Path, monkeypatch) -> None:
    _widget_tree(tmp_path, consumers=MODULE.MAX_IMPACTED_TEST_FILES + 1, unrelated=1000)
    monkeypatch.setattr(MODULE, "MAX_WIDE_TEST_FILES", MODULE.MAX_IMPACTED_TEST_FILES)

    plan = MODULE.build_plan(tmp_path, ["src/blueprint_pipeline/widget.py"])

    assert plan["requires_full_suite"] is True
    assert plan["requires_wide_impacted"] is False


@pytest.mark.parametrize("other_change", [
    "pyproject.toml",
    "tests/conftest.py",
    "scripts/brand_new_uncovered_launcher.py",
    "deploy/brand_new_uncovered_tool.py",
])
def test_every_other_full_suite_reason_still_wins_over_the_wide_lane(tmp_path: Path, other_change: str) -> None:
    _widget_tree(tmp_path, consumers=MODULE.MAX_IMPACTED_TEST_FILES + 1, unrelated=200)

    plan = MODULE.build_plan(tmp_path, ["src/blueprint_pipeline/widget.py", other_change])

    assert plan["requires_full_suite"] is True
    assert plan["requires_wide_impacted"] is False
    assert any(reason.startswith("impacted_test_count_exceeds_budget:") for reason in plan["reasons"])
    assert plan["wide_shards"] == []


def test_too_many_changed_files_still_win_over_the_wide_lane(tmp_path: Path) -> None:
    _widget_tree(tmp_path, consumers=MODULE.MAX_IMPACTED_TEST_FILES + 1, unrelated=200)
    changed = ["src/blueprint_pipeline/widget.py"]
    changed += [f"docs/page_{index}.md" for index in range(MODULE.MAX_CHANGED_FILES)]

    plan = MODULE.build_plan(tmp_path, changed)

    assert plan["requires_full_suite"] is True
    assert plan["requires_wide_impacted"] is False


def _write_baseline(root: Path, seconds: dict[str, float]) -> None:
    baseline = root / MODULE.DURATION_BASELINE
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(json.dumps({
        "default_seconds_per_test": 0.5,
        "files": [{"path": path, "duration_seconds": value} for path, value in seconds.items()],
    }))


@pytest.mark.parametrize(("seconds", "shards"), [
    (100.0, 1),
    (600.0, 1),
    (601.0, 2),
    (1800.0, 3),
    (2400.0, 4),
    (50_000.0, 4),
])
def test_wide_shard_count_follows_the_estimate_and_stops_at_four(
    tmp_path: Path, seconds: float, shards: int
) -> None:
    files = {f"tests/test_file_{index}.py": seconds / 10 for index in range(10)}
    _write_baseline(tmp_path, files)

    assert len(MODULE._wide_shards(tmp_path, files=sorted(files), test_sources={})) == shards


def test_wide_shards_balance_longest_first_and_are_deterministic(tmp_path: Path) -> None:
    files = {"tests/test_a.py": 600.0, "tests/test_b.py": 300.0, "tests/test_c.py": 200.0, "tests/test_d.py": 100.0}
    _write_baseline(tmp_path, files)

    plan = MODULE._wide_shards(tmp_path, files=sorted(files), test_sources={})

    assert plan == [
        {"index": 0, "estimated_seconds": 600.0, "files": ["tests/test_a.py"]},
        {"index": 1, "estimated_seconds": 600.0, "files": ["tests/test_b.py", "tests/test_c.py", "tests/test_d.py"]},
    ]
    assert plan == MODULE._wide_shards(tmp_path, files=sorted(files), test_sources={})


def test_file_missing_from_the_baseline_is_estimated_from_its_tests(tmp_path: Path) -> None:
    _write_baseline(tmp_path, {"tests/test_known.py": 10.0})

    estimates = MODULE._estimated_seconds(
        tmp_path,
        files=["tests/test_known.py", "tests/test_new.py", "tests/test_empty.py"],
        test_sources={"tests/test_new.py": "def test_a(): pass\ndef test_b(): pass\n"},
    )

    assert estimates == {"tests/test_known.py": 10.0, "tests/test_new.py": 1.0, "tests/test_empty.py": 0.5}


@pytest.mark.parametrize("baseline_text", [None, "not json", "{}", '{"files": [{"path": "x"}]}'])
def test_unusable_baseline_only_loses_the_balance_hint(tmp_path: Path, baseline_text: str | None) -> None:
    if baseline_text is not None:
        baseline = tmp_path / MODULE.DURATION_BASELINE
        baseline.parent.mkdir(parents=True)
        baseline.write_text(baseline_text)

    shards = MODULE._wide_shards(tmp_path, files=["tests/test_a.py", "tests/test_b.py"], test_sources={})

    assert [shard["files"] for shard in shards] == [["tests/test_a.py", "tests/test_b.py"]]


def test_committed_baseline_is_readable_as_the_wide_lane_hint() -> None:
    estimates = MODULE._estimated_seconds(ROOT, files=["tests/test_impacted_test_selection.py"], test_sources={})

    baseline = json.loads((ROOT / MODULE.DURATION_BASELINE).read_text())
    measured = {row["path"]: row["duration_seconds"] for row in baseline["files"]}
    assert estimates == {"tests/test_impacted_test_selection.py": measured["tests/test_impacted_test_selection.py"]}


def test_a_change_to_the_wide_lane_runs_the_wide_lane() -> None:
    plan = MODULE.build_plan(ROOT, [MODULE.WIDE_LANE_WORKFLOW])

    assert plan["requires_full_suite"] is False
    assert plan["requires_wide_impacted"] is True
    assert f"wide_lane_changed:{MODULE.WIDE_LANE_WORKFLOW}" in plan["reasons"]
    assert "tests/test_release_engineering_contracts.py" in plan["wide_impacted_tests"]


@pytest.mark.parametrize(("consumers", "expected_wide", "expected_matrix"), [
    (MODULE.MAX_IMPACTED_TEST_FILES, "false", "[]"),
    (MODULE.MAX_IMPACTED_TEST_FILES + 1, "true", "[0]"),
])
def test_github_output_carries_the_wide_decision_and_its_shards(
    tmp_path: Path, consumers: int, expected_wide: str, expected_matrix: str
) -> None:
    _widget_tree(tmp_path, consumers=consumers, unrelated=200)
    plan = MODULE.build_plan(tmp_path, ["src/blueprint_pipeline/widget.py"])
    output = tmp_path / "github-output"

    MODULE.write_github_output(plan, output)

    lines = output.read_text().splitlines()
    assert f"requires_wide_impacted={expected_wide}" in lines
    assert f"wide_shard_matrix={expected_matrix}" in lines
    assert "requires_full_suite=false" in lines


def test_github_output_matrix_names_every_planned_shard(tmp_path: Path) -> None:
    files = {f"tests/test_file_{index}.py": 500.0 for index in range(12)}
    _write_baseline(tmp_path, files)
    plan = {
        "requires_full_suite": False,
        "selected_tests": [],
        "requires_wide_impacted": True,
        "wide_shards": MODULE._wide_shards(tmp_path, files=sorted(files), test_sources={}),
    }
    output = tmp_path / "github-output"

    MODULE.write_github_output(plan, output)

    assert "wide_shard_matrix=[0,1,2,3]" in output.read_text().splitlines()


def test_unmapped_executable_and_dependency_changes_request_full_suite() -> None:
    unmapped = MODULE.build_plan(ROOT, ["scripts/brand_new_uncovered_launcher.py"])
    dependency = MODULE.build_plan(ROOT, ["pyproject.toml"])

    assert unmapped["requires_full_suite"] is True
    assert "unmapped_executable_surface:scripts/brand_new_uncovered_launcher.py" in unmapped[
        "reasons"
    ]
    assert dependency["requires_full_suite"] is True
    assert "cross_cutting_file:pyproject.toml" in dependency["reasons"]


def test_package_initializer_maps_its_consumers_not_every_initializer(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_own_package.py").write_text("from blueprint_pipeline.owned import widget\n")
    (tests / "test_own_child.py").write_text("import blueprint_pipeline.owned.child\n")
    (tests / "test_parent_import.py").write_text("from blueprint_pipeline import owned\n")
    for index in range(MODULE.MAX_IMPACTED_TEST_FILES + 1):
        (tests / f"test_unrelated_{index}.py").write_text('filename = "__init__.py"\n')
    plan = MODULE.build_plan(tmp_path, ["src/blueprint_pipeline/owned/__init__.py"])
    assert plan["requires_full_suite"] is False
    assert set(plan["selected_tests"]) == set(MODULE.SENTINEL_TESTS) | {
        "tests/test_own_package.py", "tests/test_own_child.py", "tests/test_parent_import.py",
    }


def test_initializer_outside_src_maps_only_tests_naming_it(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_door.py").write_text("# deploy/operator-door/operator_door/__init__.py\n")
    for index in range(MODULE.MAX_IMPACTED_TEST_FILES + 1):
        (tests / f"test_unrelated_{index}.py").write_text('filename = "__init__.py"\n')
    plan = MODULE.build_plan(tmp_path, ["deploy/operator-door/operator_door/__init__.py"])
    assert plan["requires_full_suite"] is False
    assert set(plan["selected_tests"]) == set(MODULE.SENTINEL_TESTS) | {"tests/test_door.py"}


def test_source_basename_is_not_a_dependency_on_a_longer_filename(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_own_service.py").write_text("from blueprint_pipeline.owned.service import Service\n")
    (tests / "test_file_loader.py").write_text('source = root / "service.py"\n')
    (tests / "test_other_service.py").write_text('source = root / "live_pipeline_intake_service.py"\n')
    plan = MODULE.build_plan(tmp_path, ["src/blueprint_pipeline/owned/service.py"])
    assert plan["requires_full_suite"] is False
    assert {"tests/test_own_service.py", "tests/test_file_loader.py"} <= set(plan["selected_tests"])
    assert "tests/test_other_service.py" not in plan["selected_tests"]


def test_longer_module_and_package_names_do_not_count_as_coverage(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_longer.py").write_text("import blueprint_pipeline.owned.service_worker\n")
    plan = MODULE.build_plan(tmp_path, ["src/blueprint_pipeline/owned/service.py"])
    assert plan["requires_full_suite"] is True
    assert "tests/test_longer.py" not in plan["selected_tests"]
    (tests / "test_longer.py").write_text("from blueprint_pipeline import owned_else\n")
    plan = MODULE.build_plan(tmp_path, ["src/blueprint_pipeline/owned/__init__.py"])
    assert plan["requires_full_suite"] is True


def _source_tree(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "tests").mkdir(exist_ok=True)


CHANGED = "src/blueprint_pipeline/changed.py"


def test_a_module_no_test_names_maps_through_what_imports_it(tmp_path: Path) -> None:
    """atlas_preparation_job_control.py alone ran six full suites in one day."""
    _source_tree(tmp_path, {
        CHANGED: "value = 1\n",
        "src/blueprint_pipeline/first.py": "from blueprint_pipeline import changed\n",
        "src/blueprint_pipeline/second.py": "import blueprint_pipeline.first as first\n",
        "src/blueprint_pipeline/third.py": "from blueprint_pipeline.second import helper\n",
        "tests/test_of_first.py": "import blueprint_pipeline.first\n",
        "tests/test_of_second.py": "from blueprint_pipeline import second\n",
        "tests/test_of_third.py": "import blueprint_pipeline.third\n",
        "tests/test_unrelated.py": "import blueprint_pipeline.unrelated\n",
    })

    plan = MODULE.build_plan(tmp_path, [CHANGED])

    assert plan["requires_full_suite"] is False
    assert {"tests/test_of_first.py", "tests/test_of_second.py"} <= set(plan["selected_tests"])
    # Two steps only: three would select most of the repository for a leaf module.
    assert "tests/test_of_third.py" not in plan["selected_tests"]
    assert "tests/test_unrelated.py" not in plan["selected_tests"]
    assert f"mapped_through_importers:{CHANGED}:2" in plan["reasons"]


@pytest.mark.parametrize("importer", [
    "from blueprint_pipeline.changed import value\n",
    "from blueprint_pipeline import (\n    other,  # a comment\n    changed as renamed,\n)\n",
    "def later():\n    from blueprint_pipeline import changed\n    return changed\n",
    "import os, blueprint_pipeline.changed as renamed\n",
    "from .changed import value\n",
    "from . import changed\n",
])
def test_every_way_of_importing_counts(tmp_path: Path, importer: str) -> None:
    _source_tree(tmp_path, {
        CHANGED: "value = 1\n",
        "src/blueprint_pipeline/first.py": importer,
        "tests/test_of_first.py": "import blueprint_pipeline.first\n",
    })

    assert "tests/test_of_first.py" in MODULE.build_plan(tmp_path, [CHANGED])["selected_tests"]


def test_a_relative_import_resolves_from_its_own_package(tmp_path: Path) -> None:
    _source_tree(tmp_path, {
        "src/blueprint_pipeline/pkg/__init__.py": "from .inner import helper\n",
        "src/blueprint_pipeline/pkg/inner.py": "from ..changed import value\nhelper = value\n",
        "src/blueprint_pipeline/pkg/deep/leaf.py": "from ...changed import value\n",
        CHANGED: "value = 1\n",
        "tests/test_of_inner.py": "import blueprint_pipeline.pkg.inner\n",
        "tests/test_of_leaf.py": "import blueprint_pipeline.pkg.deep.leaf\n",
        "tests/test_of_pkg.py": "import blueprint_pipeline.pkg\n",
    })

    plan = MODULE.build_plan(tmp_path, [CHANGED])

    assert {"tests/test_of_inner.py", "tests/test_of_leaf.py", "tests/test_of_pkg.py"} <= set(plan["selected_tests"])


@pytest.mark.parametrize("importer", [
    "import blueprint_pipeline.changed_more\n",
    "from blueprint_pipeline import changed_more\n",
    "changed = 1\n",
    "# from blueprint_pipeline import changed\n",
])
def test_a_longer_name_or_a_comment_is_not_an_importer(tmp_path: Path, importer: str) -> None:
    _source_tree(tmp_path, {
        CHANGED: "value = 1\n",
        "src/blueprint_pipeline/first.py": importer,
        "tests/test_of_first.py": "import blueprint_pipeline.first\n",
    })

    plan = MODULE.build_plan(tmp_path, [CHANGED])

    assert plan["requires_full_suite"] is True
    assert f"unmapped_executable_surface:{CHANGED}" in plan["reasons"]


def test_a_script_counts_as_an_importer(tmp_path: Path) -> None:
    _source_tree(tmp_path, {
        CHANGED: "value = 1\n",
        "scripts/run_changed.py": "from blueprint_pipeline import changed\n",
        "tests/test_run_changed_cli.py": 'tool = _load("run_changed")\n',
    })

    assert "tests/test_run_changed_cli.py" in MODULE.build_plan(tmp_path, [CHANGED])["selected_tests"]


@pytest.mark.parametrize("reference", [
    "from operator_door.contract import value\n",
    "from operator_door import contract\n",
    "import operator_door.contract\n",
])
def test_a_door_module_maps_to_tests_importing_it(tmp_path: Path, reference: str) -> None:
    """scripted_policy_registration_contract.py alone ran eight full suites in one day."""
    door = "deploy/operator-door/operator_door/contract.py"
    _source_tree(tmp_path, {door: "value = 1\n", "tests/test_door_contract.py": reference})

    plan = MODULE.build_plan(tmp_path, [door])

    assert plan["requires_full_suite"] is False
    assert "tests/test_door_contract.py" in plan["selected_tests"]
    assert f"mapped:{door}:1" in plan["reasons"]


def test_a_door_module_maps_through_its_relative_importers(tmp_path: Path) -> None:
    door = "deploy/operator-door/operator_door/contract.py"
    _source_tree(tmp_path, {
        door: "value = 1\n",
        "deploy/operator-door/operator_door/registration.py": "from .contract import value\n",
        "tests/test_door_registration.py": "from operator_door import registration\n",
    })

    plan = MODULE.build_plan(tmp_path, [door])

    assert plan["requires_full_suite"] is False
    assert "tests/test_door_registration.py" in plan["selected_tests"]


@pytest.mark.parametrize("config", [
    "src/blueprint_pipeline/config/runtime_pin.json",
    "src/blueprint_pipeline/config/runtime_pin/inner/lock.txt",
])
def test_a_config_file_maps_to_the_tests_of_its_loader(tmp_path: Path, config: str) -> None:
    _source_tree(tmp_path, {
        config: "{}\n",
        "src/blueprint_pipeline/loader.py": 'PIN = Path(__file__).parent / "config" / "runtime_pin"\n'
        if "inner" in config else 'PIN = Path(__file__).parent / "config" / "runtime_pin.json"\n',
        "tests/test_of_loader.py": "import blueprint_pipeline.loader\n",
        "tests/test_unrelated.py": "import blueprint_pipeline.unrelated\n",
    })

    plan = MODULE.build_plan(tmp_path, [config])

    assert plan["requires_full_suite"] is False
    assert "tests/test_of_loader.py" in plan["selected_tests"]
    assert "tests/test_unrelated.py" not in plan["selected_tests"]


def test_a_config_file_nothing_loads_still_requests_the_full_suite(tmp_path: Path) -> None:
    config = "src/blueprint_pipeline/config/orphan.json"
    _source_tree(tmp_path, {config: "{}\n", "tests/test_unrelated.py": "x = 1\n"})

    plan = MODULE.build_plan(tmp_path, [config])

    assert plan["requires_full_suite"] is True
    assert f"unmapped_executable_surface:{config}" in plan["reasons"]


def test_a_new_module_nothing_references_still_requests_the_full_suite(tmp_path: Path) -> None:
    _source_tree(tmp_path, {CHANGED: "value = 1\n", "tests/test_unrelated.py": "x = 1\n"})

    plan = MODULE.build_plan(tmp_path, [CHANGED])

    assert plan["requires_full_suite"] is True
    assert f"unmapped_executable_surface:{CHANGED}" in plan["reasons"]


def test_an_importer_with_no_tests_of_its_own_still_requests_the_full_suite(tmp_path: Path) -> None:
    _source_tree(tmp_path, {
        CHANGED: "value = 1\n",
        "src/blueprint_pipeline/first.py": "from blueprint_pipeline import changed\n",
        "tests/test_unrelated.py": "x = 1\n",
    })

    assert MODULE.build_plan(tmp_path, [CHANGED])["requires_full_suite"] is True


def test_a_module_a_test_names_directly_is_not_widened_through_its_importers(tmp_path: Path) -> None:
    """The closure only replaces a full suite: it never grows a selection that already works."""
    _source_tree(tmp_path, {
        CHANGED: "value = 1\n",
        "src/blueprint_pipeline/first.py": "from blueprint_pipeline import changed\n",
        "tests/test_changed.py": "from blueprint_pipeline import changed\n",
        "tests/test_of_first.py": "import blueprint_pipeline.first\n",
    })

    plan = MODULE.build_plan(tmp_path, [CHANGED])

    assert "tests/test_changed.py" in plan["selected_tests"]
    assert "tests/test_of_first.py" not in plan["selected_tests"]


def test_the_recurring_unmapped_surfaces_of_2026_10_31_now_map() -> None:
    for path in (
        "deploy/operator-door/operator_door/scripted_policy_registration_contract.py",
        "deploy/operator-door/operator_door/embodiment_preparation.py",
        "deploy/operator-door/operator_door/public_policy_registration_contract.py",
        "src/blueprint_pipeline/atlas_preparation_job_control.py",
        "src/blueprint_pipeline/deployment_embodiment_simulation_models.py",
        "src/blueprint_pipeline/config/isaacsim_pink_601.json",
    ):
        plan = MODULE.build_plan(ROOT, [path])
        assert not any(reason.startswith("unmapped_executable_surface") for reason in plan["reasons"]), path


def test_changed_test_is_selected_directly() -> None:
    plan = MODULE.build_plan(ROOT, ["tests/test_capture_qa.py"])

    assert plan["requires_full_suite"] is False
    assert "tests/test_capture_qa.py" in plan["selected_tests"]


def test_build_loop_default_is_two_minutes() -> None:
    args = MODULE.parse_args([])

    assert MODULE.DEFAULT_TIMEOUT_SECONDS == 120
    assert args.timeout_seconds == 120
    assert MODULE.DEFAULT_WORKER_COUNT == 4


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


def test_policy_canary_hermetic_rehearsals_follow_their_production_modules() -> None:
    """A worker or bundle change must run the tests that stand in for a paid GPU attempt.

    Five paid Quick-10 attempts failed on defects these suites catch in seconds.
    The selector maps a changed module to the tests whose source names it, so
    the suites import their subjects by dotted module path; this pin fails if
    that import is ever rewritten into a form the selector cannot see.
    """

    rehearsal = "tests/test_native_task_arena_policy_canary_lifecycle_rehearsal.py"
    closure = "tests/test_provider_runtime_import_closure.py"
    expectations = {
        "src/blueprint_pipeline/native_task_arena_policy_canary_worker.py": {rehearsal, closure},
        "src/blueprint_pipeline/team_evaluation_worker_retention.py": {rehearsal, closure},
        "src/blueprint_pipeline/native_task_arena_policy_canary_bundle.py": {closure},
        "src/blueprint_pipeline/native_task_arena_policy_canary_session.py": {rehearsal},
        "src/blueprint_pipeline/adp009d_policy_episode.py": {rehearsal},
        "src/blueprint_pipeline/native_task_arena_execution_contract.py": {closure},
    }
    for changed, expected in expectations.items():
        plan = MODULE.build_plan(ROOT, [changed])
        assert plan["requires_full_suite"] is False, (changed, plan["reasons"])
        assert expected <= set(plan["selected_tests"]), (changed, plan["selected_tests"])
