"""The door's transient scripts, run for real against stub git/systemctl/python."""

# Covers (for impacted-test selection):
#   deploy/operator-door/door-common.sh
#   deploy/operator-door/door-deploy.sh
#   deploy/operator-door/door-replay.sh
#   deploy/operator-door/door-upgrade.sh
#   deploy/operator-door/install.sh

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

DOOR = Path(__file__).resolve().parents[1] / "deploy" / "operator-door"
SHA = "0123456789abcdef0123456789abcdef01234567"
DEPLOY_ID = "20260923T120000Z-deploy-0000abcd"
REPLAY_ID = "20260923T120000Z-stage-replay-0000abcd"

GIT_STUB = r"""#!/bin/bash
echo "git $*" >> "$STUB_LOG"
args=("$@"); [ "${args[0]}" = "-C" ] && args=("${args[@]:2}")
last="${args[${#args[@]}-1]}"
case "${args[0]}" in
  clone) mkdir -p "$last/.git" ;;
  merge-base) exit "${FAKE_ON_MAIN_RC:-0}" ;;
  branch) [ -n "${FAKE_PUSHED:-}" ] && echo "  origin/feature" ;;
  worktree)
    if [ "${args[1]}" = "add" ]; then mkdir -p "${args[4]}"; else rm -rf "$last"; fi ;;
esac
exit 0
"""

SYSTEMCTL_STUB = r"""#!/bin/bash
echo "systemctl $*" >> "$STUB_LOG"
if [ "$1" = "is-active" ] && [ "$2" = "${FAKE_BUSY_UNIT:-}" ]; then
  count=$(cat "$STUB_DIR/busy" 2>/dev/null || echo 0)
  if [ "$count" -lt "${FAKE_BUSY_POLLS:-0}" ]; then echo $((count + 1)) > "$STUB_DIR/busy"; echo active; exit 0; fi
fi
echo inactive
exit 3
"""

PYTHON_STUB = r"""#!/bin/bash
echo "venv-python $*" >> "$STUB_LOG"
while [ $# -gt 0 ]; do
  case "$1" in
    --receipt-out) mkdir -p "$(dirname "$2")"; echo '{"status": "deployed"}' > "$2"; shift ;;
    --json-out) echo '{"schema": "task_evaluation_stage_replay_report.v1"}' > "$2"; shift ;;
  esac
  shift
done
exit "${FAKE_TOOL_RC:-0}"
"""


def _write_stub(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture()
def env(tmp_path: Path) -> dict[str, str]:
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    _write_stub(stubs / "git", GIT_STUB)
    _write_stub(stubs / "systemctl", SYSTEMCTL_STUB)
    _write_stub(tmp_path / "venv-python", PYTHON_STUB)
    results = tmp_path / "results"
    results.mkdir()
    return {
        **os.environ,
        "PATH": f"{stubs}:{os.environ['PATH']}",
        "STUB_LOG": str(tmp_path / "calls.log"),
        "STUB_DIR": str(tmp_path),
        "DOOR_RESULTS_DIR": str(results),
        "DOOR_COMMIT": SHA,
        "DOOR_SOURCE_CLONE": str(tmp_path / "tools" / "operator-door-source"),
        "DOOR_REFERENCE_REPO": str(tmp_path / "no-reference"),
        "DOOR_UPSTREAM_URL": "https://github.com/example/BlueprintCapturePipeline.git",
        "DOOR_VENV_PYTHON": str(tmp_path / "venv-python"),
        "DOOR_STATE_ROOT": str(tmp_path / "state"),
        "DOOR_IDLE_UNITS": "blueprint-a.service,blueprint-b.service",
        "DOOR_IDLE_WAIT_SECONDS": "30",
        "DOOR_IDLE_POLL_SECONDS": "0",
    }


def _run(script: str, env: dict[str, str], **extra: str) -> tuple[int, dict, list[str]]:
    done = subprocess.run(["/bin/bash", str(DOOR / script)], env={**env, **extra},
                          capture_output=True, text=True, check=False, timeout=60)
    request_id = extra.get("DOOR_REQUEST_ID", "")
    outcome_path = Path(env["DOOR_RESULTS_DIR"]) / f"{request_id}.outcome.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8")) if outcome_path.exists() else {}
    calls = Path(env["STUB_LOG"]).read_text(encoding="utf-8").splitlines() if Path(env["STUB_LOG"]).exists() else []
    return done.returncode, outcome, calls


def _tool_call(calls: list[str]) -> str:
    return next(call for call in calls if call.startswith("venv-python "))


def test_main_deploy_runs_the_target_commits_deploy_tool(env: dict[str, str]) -> None:
    rc, outcome, calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_MODE="main",
                              DOOR_WAIT_FOR_IDLE="1")
    assert rc == 0 and outcome["status"] == "deployed" and outcome["exit_code"] == 0
    tool = _tool_call(calls)
    assert "/scripts/deploy_control_plane_commit.py" in tool
    assert f"--source-repo {env['DOOR_SOURCE_CLONE']} --source-commit {SHA}" in tool
    assert "--iteration --preserve-configured-controls-state" in tool and "--canary" not in tool
    assert tool.endswith(f"--receipt-out {env['DOOR_STATE_ROOT']}/deploy-receipts/iteration_{SHA[:12]}_door.json")
    assert any(call.startswith("git clone --quiet --no-checkout") for call in calls)
    assert any("merge-base --is-ancestor " + SHA + " origin/main" in call for call in calls)
    assert any("worktree remove --force" in call for call in calls)
    assert outcome["receipt"].endswith(f"iteration_{SHA[:12]}_door.json")
    assert (Path(env["DOOR_RESULTS_DIR"]) / f"{DEPLOY_ID}.log").exists()


def test_canary_deploy_passes_canary_and_requires_a_pushed_ref(env: dict[str, str]) -> None:
    rc, outcome, calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_MODE="canary",
                              DOOR_WAIT_FOR_IDLE="0", FAKE_PUSHED="1")
    assert rc == 0 and "--iteration --canary --preserve-configured-controls-state" in _tool_call(calls)
    rc, outcome, _ = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_MODE="canary",
                          DOOR_WAIT_FOR_IDLE="0")
    assert rc == 2 and outcome["code"] == "commit_not_pushed"


def test_main_deploy_of_an_unmerged_commit_is_refused(env: dict[str, str]) -> None:
    rc, outcome, calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_MODE="main",
                              DOOR_WAIT_FOR_IDLE="0", FAKE_ON_MAIN_RC="1")
    assert rc == 2 and outcome == {**outcome, "status": "refused", "code": "commit_not_on_main"}
    assert not any(call.startswith("venv-python") for call in calls)


def test_deploy_waits_for_busy_controller_units(env: dict[str, str]) -> None:
    rc, outcome, calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_MODE="main",
                              DOOR_WAIT_FOR_IDLE="1", FAKE_BUSY_UNIT="blueprint-b.service", FAKE_BUSY_POLLS="2")
    assert rc == 0 and outcome["status"] == "deployed"
    assert sum("is-active blueprint-b.service" in call for call in calls) == 3


def test_deploy_gives_up_when_units_stay_busy(env: dict[str, str]) -> None:
    rc, outcome, calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_MODE="main",
                              DOOR_WAIT_FOR_IDLE="1", DOOR_IDLE_WAIT_SECONDS="0",
                              FAKE_BUSY_UNIT="blueprint-a.service", FAKE_BUSY_POLLS="99")
    assert rc == 2 and outcome["code"] == "idle_wait_timeout:blueprint-a.service"
    assert not any(call.startswith("git ") for call in calls)


def test_a_failing_deploy_tool_is_reported(env: dict[str, str]) -> None:
    rc, outcome, _ = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_MODE="main",
                          DOOR_WAIT_FOR_IDLE="0", FAKE_TOOL_RC="2")
    assert rc == 2 and outcome["status"] == "failed" and outcome["code"] == "deploy_tool_exit_2"


@pytest.mark.parametrize(
    ("overrides", "code"),
    [({"DOOR_COMMIT": "not-a-sha"}, "commit_invalid"), ({"DOOR_MODE": "promote"}, "mode_invalid")],
)
def test_deploy_rechecks_its_inputs(env: dict[str, str], overrides: dict[str, str], code: str) -> None:
    values = {"DOOR_REQUEST_ID": DEPLOY_ID, "DOOR_MODE": "main", "DOOR_WAIT_FOR_IDLE": "0", **overrides}
    rc, outcome, calls = _run("door-deploy.sh", env, **values)
    assert rc == 2 and outcome["code"] == code and not calls


def test_deploy_refuses_a_malformed_request_id_before_touching_any_path(env: dict[str, str]) -> None:
    rc, _, calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID="../../etc/x", DOOR_MODE="main")
    assert rc == 2 and not calls and not list(Path(env["DOOR_RESULTS_DIR"]).iterdir())


def test_replay_runs_isolated_from_the_candidate_tree(env: dict[str, str]) -> None:
    child = "sam31-" + "ab" * 16
    rc, outcome, calls = _run("door-replay.sh", env, DOOR_REQUEST_ID=REPLAY_ID, DOOR_CHILD=child,
                              DOOR_PARENT="", FAKE_PUSHED="1")
    tool = _tool_call(calls)
    assert rc == 0 and outcome["status"] == "replayed"
    assert f"-m blueprint_pipeline.task_evaluation_stage_replay --child {child} --isolate --json-out" in tool
    assert "--allow-paid" not in tool
    assert outcome["report"].endswith(f"{REPLAY_ID}.replay.json")


def test_replay_refuses_a_bad_target(env: dict[str, str]) -> None:
    rc, outcome, _ = _run("door-replay.sh", env, DOOR_REQUEST_ID=REPLAY_ID, DOOR_CHILD="sam31-..",
                          DOOR_PARENT="")
    assert rc == 2 and outcome["code"] == "replay_target_invalid"


@pytest.mark.parametrize("script", ["door-common.sh", "door-deploy.sh", "door-replay.sh", "door-upgrade.sh",
                                    "install.sh"])
def test_scripts_parse(script: str) -> None:
    assert subprocess.run(["/bin/bash", "-n", str(DOOR / script)], check=False).returncode == 0


def test_no_script_uses_the_wrapper_deploys_or_allow_paid() -> None:
    for script in DOOR.glob("*.sh"):
        text = script.read_text(encoding="utf-8")
        code = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
        for line in code:
            assert "deploy_control_plane_iteration.sh" not in line, script.name
            assert "deploy_control_plane_canary.sh" not in line, script.name
            assert "--allow-paid" not in line, script.name
        assert "set -euo pipefail" in text or script.name == "door-common.sh"


def test_installer_never_overwrites_tokens_and_validates_caddy_first() -> None:
    text = (DOOR / "install.sh").read_text(encoding="utf-8")
    assert 'if [ ! -e "$config_dir/tokens.json" ]; then' in text
    assert text.index("caddy validate") < text.index("systemctl reload caddy")
    assert "cp -p \"$backup\" \"$caddyfile\"" in text
