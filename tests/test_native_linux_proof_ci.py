"""Offline scheduling contracts; real root/UID acceptance remains hosted Linux."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from blueprint_pipeline import impacted_test_selection

ROOT = Path(__file__).resolve().parents[1]


def _jobs(name: str) -> dict:
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())["jobs"]


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        ([".github/workflows/full-test-lane.yml"], False),
        (["pyproject.toml"], False),
        (["tests/test_registered_feature_linux.py"], True),
        (["tests/test_legacy_owner_linux.py"], True),
        (["docs/example.md"], False),
    ],
)
def test_native_scheduling_executes_the_real_impact_plan_decision(
    tmp_path: Path, changed: list[str], expected: bool
) -> None:
    plan = impacted_test_selection.build_plan(ROOT, changed)
    if changed == [".github/workflows/full-test-lane.yml"]:
        assert plan["requires_full_suite"] is True
        assert not any("_linux.py" in row for row in plan["selected_tests"])
        full = _jobs("ci.yml")["cross-cutting-full-suite"]
        assert full["if"] == "needs.impact.outputs.requires_full_suite == 'true'"
        assert full["uses"] == "./.github/workflows/full-test-lane.yml"
    impact = _jobs("ci.yml")["impact"]
    script = next(step["run"] for step in impact["steps"] if step.get("id") == "native-plan")
    match = re.search(r"python3 - <<'PYTHON'\n(.*?)\nPYTHON", script, re.DOTALL)
    assert match is not None
    manifest, output = tmp_path / "impact.json", tmp_path / "github-output"
    manifest.write_text(json.dumps(plan))
    result = subprocess.run(
        [sys.executable, "-c", match.group(1)],
        env=os.environ | {"IMPACT_PLAN": str(manifest), "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text() == f"required={str(expected).lower()}\n"


def test_full_shards_reuse_the_protected_native_environment_without_reducing_collection() -> None:
    shard = _jobs("full-test-lane.yml")["full-pytest-shard"]
    native = _jobs("ci.yml")["native-feature-linux"]
    assert shard["runs-on"] == native["runs-on"] == "ubuntu-24.04"
    for name in ("BLUEPRINT_DISPOSABLE_LINUX_TEST", "PYTHONDONTWRITEBYTECODE", "UV_PROJECT_ENVIRONMENT"):
        assert shard["env"][name] == native["env"][name]
    install = next(
        step["run"] for step in shard["steps"] if "uv sync --frozen" in step.get("run", "")
    )
    preflight = (
        "sudo --non-interactive true",
        'test "$manager" = /usr/lib/systemd/systemd',
        "test -f /sys/fs/cgroup/cgroup.controllers",
        "uv venv --python /usr/bin/python3 /var/lib/blueprint-native-test-venv",
    )
    for command in preflight:
        assert command in install
        assert install.index(command) < install.index("uv sync --frozen")
    sync = install.split("uv sync --frozen", 1)[1].split("uv pip install", 1)[0]
    assert "--python /usr/bin/python3" in sync
    assert "--no-editable" in sync
    # The protected installed entry requires single-link imported source;
    # uv's cache hardlinks fail that native safety check even after chown.
    assert "--link-mode copy" in sync
    assert "uv pip install --python /var/lib/blueprint-native-test-venv/bin/python --link-mode copy ./BlueprintContracts" in install
    seal = "sudo chown -hR root:root /var/lib/blueprint-native-test-venv"
    assert install.index(seal) > install.index("uv pip install")
    for step in shard["steps"]:
        for line in step.get("run", "").splitlines():
            if line.strip().startswith("uv run "):
                assert line.strip().startswith("uv run --no-sync "), step["name"]
    collect = next(step["run"] for step in shard["steps"] if step["name"] == "Collect full lane")
    execute = next(step["run"] for step in shard["steps"] if step["name"] == "Run full lane shard")
    for script in (collect, execute):
        assert '-m "not external_runtime and not external_data"' in script
        assert "--ignore" not in script
        assert "--deselect" not in script
        assert "-k " not in script
        assert "blueprint_pipeline.pytest_full_lane_evidence" in script
    assert '"${shard_files[@]}"' in execute


def test_required_native_proof_still_gates_the_pr_on_success() -> None:
    jobs = _jobs("ci.yml")
    assert jobs["native-feature-linux"]["if"] == "needs.impact.outputs.native_feature_required == 'true'"
    gate = jobs["impacted-gate"]
    assert "native-feature-linux" in gate["needs"]
    step = gate["steps"][0]
    assert step["env"]["NATIVE_REQUIRED"] == "${{ needs.impact.outputs.native_feature_required }}"
    assert step["env"]["NATIVE_RESULT"] == "${{ needs.native-feature-linux.result }}"
    assert 'if test "${NATIVE_REQUIRED}" = "true"; then\n  test "${NATIVE_RESULT}" = "success"' in step["run"]


def test_retained_failed_restore_paths_execute_before_remaining_native_cases():
    from tests.historical_generation_native_acceptance import CONNECTED_CASES
    assert [row[0] for row in CONNECTED_CASES[:4]] == [
        'before_restore_final', 'unwritten_stage', 'unlogged_directory', 'reconcile_resume_remove']
    assert len(CONNECTED_CASES) == len(dict(CONNECTED_CASES)) == 38
    run = next(step['run'] for step in _jobs('ci.yml')['native-feature-linux']['steps']
               if step['name'] == 'Prove actual root and ordinary-UID lifecycle')
    first = run.split('python -m pytest', 1)[1].splitlines()[1].strip()
    assert first.startswith('tests/test_historical_generation_linux.py::test_actual_connected_historical_interruption')
    assert '--maxfail=1' in run and '--deselect' not in run and '-k ' not in run


def test_expired_pending_absence_requires_its_own_native_acceptance_case() -> None:
    from tests.historical_generation_native_acceptance import CONNECTED_CASES
    assert dict(CONNECTED_CASES)['reconcile_absent_expiry'] == dict(action='offload',
        restore_interruption='unlogged_member', reconciliation_interruption='reconcile_remove', absent_expiry=True)
    assert dict(CONNECTED_CASES)['reconcile_observation_expiry'] == dict(action='offload',
        restore_interruption='unlogged_member', reconciliation_interruption='reconcile_remove',
        absent_expiry=True, observation_expiry=True)
    run = next(step['run'] for step in _jobs('ci.yml')['native-feature-linux']['steps']
               if step['name'] == 'Prove actual root and ordinary-UID lifecycle')
    assert 'assert len(cases) == 45' in run
    assert 'case.find(tag)' in run and 'for tag in ("skipped", "error", "failure")' in run


def test_admitted_path_boundaries_need_real_native_restore_cases():
    from tests.historical_generation_native_acceptance import CONNECTED_CASES
    cases = dict(CONNECTED_CASES)
    assert cases['escaped_path_restore'] == dict(action='offload', controlled_names=True)
    assert cases['maximum_depth_restore'] == dict(action='offload', deep_tree=True,
                                                  restore_interruption='stage_complete')


def test_present_delete_expiry_requires_each_actual_original_operation_boundary() -> None:
    from tests.historical_generation_native_acceptance import CONNECTED_CASES
    cases = dict(CONNECTED_CASES)
    for name, phase, resume in (('reconcile_delete_expiry', None, False),
        ('reconcile_pending_delete_expiry', 'reconcile_intent', False),
        ('reconcile_resume_delete_expiry', 'reconcile_intent', True)):
        assert cases[name] == dict(action='offload', restore_interruption='unlogged_member',
            reconciliation_interruption=phase, delete_expiry=True, resume_delete_expiry=resume)


def test_resumed_unlink_has_actual_current_and_expired_absence_cases() -> None:
    from tests.historical_generation_native_acceptance import CONNECTED_CASES
    for case, expired in [('reconcile_resume_remove', False), ('reconcile_resume_absent_expiry', True)]:
        options = dict(action='offload', restore_interruption='unlogged_member',
            reconciliation_interruption='reconcile_intent', delete_expiry=True, resume_remove=True)
        if expired:
            options['resume_absent_expiry'] = True
        assert dict(CONNECTED_CASES)[case] == options
