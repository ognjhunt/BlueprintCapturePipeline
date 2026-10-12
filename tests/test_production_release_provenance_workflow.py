from pathlib import Path
import json
import os
import re
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/production-release-provenance.yml"
PROMOTED = "a" * 40
REPOSITORY = "ognjhunt/BlueprintCapturePipeline"


def _steps() -> list[dict]:
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    return workflow["jobs"]["verify"]["steps"]


def test_provenance_is_verified_in_ci_with_read_only_ephemeral_authority() -> None:
    source = WORKFLOW.read_text()
    workflow = yaml.load(source, Loader=yaml.BaseLoader)
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    assert workflow["on"]["workflow_run"]["workflows"] == ["Full Test Lane"]
    assert workflow["on"]["workflow_run"]["branches"] == ["main"]
    job = workflow["jobs"]["verify"]
    assert job["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert "secrets." not in source
    steps = job["steps"]
    guard, checkout = steps[:2]
    assert guard["id"] == "promotion"
    assert '.head_branch == "main"' in guard["run"]
    assert '.status == "completed" and .conclusion == "success"' in guard["run"]
    assert '.display_title == "Full Test Lane / production_deployment_promotion"' in guard["run"]
    # Provenance binds the exact tested commit while main still contains it,
    # never "is still HEAD": a later merge must not void a green promotion.
    assert 'test "${promoted_sha}" = "${current_main}"' not in source
    assert "git/ref/heads/main" not in source
    assert checkout["with"]["persist-credentials"] == "false"
    assert checkout["with"]["ref"] == "${{ steps.promotion.outputs.sha }}"
    for step in steps[1:]:
        assert step["if"] == "steps.promotion.outputs.eligible == 'true'"
        if "uses" in step:
            assert re.search(r"@[0-9a-f]{40}$", step["uses"])
    verification = steps[3]["run"]
    assert "scripts/verify_deploy_release_provenance.py" in verification
    assert "--expected-sha" in verification and "--run-url" in verification
    assert steps[-1]["with"]["if-no-files-found"] == "error"
    assert "paid_resource_allocator" not in source


def _stub_gh(tmp_path: Path, *, compare_status: str, behind_by: int) -> dict[str, str]:
    """A ``gh`` that serves the promotion run and main's comparison to it."""
    run = {
        "event": "workflow_dispatch",
        "display_title": "Full Test Lane / production_deployment_promotion",
        "repository": {"full_name": REPOSITORY},
        "head_branch": "main",
        "status": "completed",
        "conclusion": "success",
        "path": ".github/workflows/full-test-lane.yml@refs/heads/main",
        "head_sha": PROMOTED,
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    (tmp_path / "compare.json").write_text(
        json.dumps({"status": compare_status, "behind_by": behind_by, "ahead_by": 3})
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'case "$2" in\n'
        f'  "repos/{REPOSITORY}/actions/runs/7") cat "{tmp_path}/run.json" ;;\n'
        f'  "repos/{REPOSITORY}/compare/{PROMOTED}...main")\n'
        '    [ "$3" = "--jq" ] || exit 64\n'
        f'    jq -r "$4" "{tmp_path}/compare.json" ;;\n'
        '  *) echo "unexpected gh call: $*" >&2; exit 65 ;;\n'
        "esac\n"
    )
    gh.chmod(0o755)
    uv = bin_dir / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    output = tmp_path / "github-output"
    output.write_text("")
    return {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_EVENT_NAME": "workflow_run",
        "GITHUB_OUTPUT": str(output),
        "GITHUB_WORKSPACE": str(tmp_path),
        "RUNNER_TEMP": str(tmp_path),
        "PROMOTION_RUN_ID": "7",
        "EXPECTED_SHA": PROMOTED,
        "PROMOTION_URL": f"https://github.com/{REPOSITORY}/actions/runs/7",
    }


def _run(step: dict, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", step["run"]], env=env, capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize(
    ("compare_status", "behind_by", "admitted"),
    [
        ("identical", 0, True),
        # main moved on after the lane started: the tested commit is still on main.
        ("ahead", 0, True),
        # main no longer contains the tested commit (rewritten history).
        ("behind", 2, False),
        ("diverged", 1, False),
    ],
)
def test_promotion_binds_a_tested_commit_main_still_contains(
    tmp_path: Path, compare_status: str, behind_by: int, admitted: bool
) -> None:
    env = _stub_gh(tmp_path, compare_status=compare_status, behind_by=behind_by)
    guard = _steps()[0]

    result = _run(guard, env)

    output = Path(env["GITHUB_OUTPUT"]).read_text()
    if admitted:
        assert result.returncode == 0, result.stderr
        assert "eligible=true\n" in output
        assert f"sha={PROMOTED}\n" in output
    else:
        assert result.returncode != 0
        assert "eligible=true" not in output


@pytest.mark.parametrize(
    ("compare_status", "behind_by", "admitted"),
    [("identical", 0, True), ("ahead", 0, True), ("behind", 2, False)],
)
def test_verified_provenance_is_retained_only_while_main_contains_the_commit(
    tmp_path: Path, compare_status: str, behind_by: int, admitted: bool
) -> None:
    env = _stub_gh(tmp_path, compare_status=compare_status, behind_by=behind_by)
    verification = next(
        step for step in _steps() if "scripts/verify_deploy_release_provenance.py" in step.get("run", "")
    )

    result = _run(verification, env)

    assert (result.returncode == 0) is admitted, result.stderr
