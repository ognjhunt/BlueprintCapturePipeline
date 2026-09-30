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
        "uv venv --python /usr/bin/python3 /opt/blueprint-native-test-venv",
    )
    for command in preflight:
        assert command in install
        assert install.index(command) < install.index("uv sync --frozen")
    sync = install.split("uv sync --frozen", 1)[1].split("uv pip install", 1)[0]
    assert "--python /usr/bin/python3" in sync
    assert "--no-editable" in sync
    assert "uv pip install --python /opt/blueprint-native-test-venv/bin/python ./BlueprintContracts" in install
    seal = "sudo chown -hR root:root /opt/blueprint-native-test-venv"
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
