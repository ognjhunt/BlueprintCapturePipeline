"""The door's transient scripts, run for real against stub git/systemctl/python."""

# Covers (for impacted-test selection):
#   deploy/operator-door/door-common.sh
#   deploy/operator-door/door-deploy.sh
#   deploy/operator-door/door-upgrade.sh
#   deploy/operator-door/door-retire-scene-workspace.sh
#   deploy/operator-door/door-restore-scene-workspace.sh
#   deploy/operator-door/install.sh

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

DOOR = Path(__file__).resolve().parents[1] / "deploy" / "operator-door"
SHA = "0123456789abcdef0123456789abcdef01234567"
DEPLOY_ID = "20260923T120000Z-deploy-0000abcd"

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
    key_dir = tmp_path / "deploy-key"
    key_dir.mkdir()
    (key_dir / "github").write_text("not a real key\n", encoding="utf-8")
    (key_dir / "known_hosts").write_text("github.com ssh-ed25519 AAAA\n", encoding="utf-8")
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
        "DOOR_GITHUB_KEY": str(key_dir / "github"),
        "DOOR_GITHUB_KNOWN_HOSTS": str(key_dir / "known_hosts"),
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
    rc, outcome, calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_WAIT_FOR_IDLE="1")
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


def test_main_deploy_of_an_unmerged_commit_is_refused(env: dict[str, str]) -> None:
    rc, outcome, calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_WAIT_FOR_IDLE="0", FAKE_ON_MAIN_RC="1")
    assert rc == 2 and outcome == {**outcome, "status": "refused", "code": "commit_not_on_main"}
    assert not any(call.startswith("venv-python") for call in calls)


def test_deploy_waits_for_busy_controller_units(env: dict[str, str]) -> None:
    rc, outcome, calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_WAIT_FOR_IDLE="1", FAKE_BUSY_UNIT="blueprint-b.service", FAKE_BUSY_POLLS="2")
    assert rc == 0 and outcome["status"] == "deployed"
    assert sum("is-active blueprint-b.service" in call for call in calls) == 3


def test_deploy_gives_up_when_units_stay_busy(env: dict[str, str]) -> None:
    rc, outcome, calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_WAIT_FOR_IDLE="1", DOOR_IDLE_WAIT_SECONDS="0",
                              FAKE_BUSY_UNIT="blueprint-a.service", FAKE_BUSY_POLLS="99")
    assert rc == 2 and outcome["code"] == "idle_wait_timeout:blueprint-a.service"
    assert not any(call.startswith("git ") for call in calls)


def test_a_failing_deploy_tool_is_reported(env: dict[str, str]) -> None:
    rc, outcome, _ = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_WAIT_FOR_IDLE="0", FAKE_TOOL_RC="2")
    assert rc == 2 and outcome["status"] == "failed" and outcome["code"] == "deploy_tool_exit_2"


@pytest.mark.parametrize(
    ("overrides", "code"),
    [({"DOOR_COMMIT": "not-a-sha"}, "commit_invalid"), ({"DOOR_COMMIT": "0" * 39}, "commit_invalid")],
)
def test_deploy_rechecks_its_inputs(env: dict[str, str], overrides: dict[str, str], code: str) -> None:
    values = {"DOOR_REQUEST_ID": DEPLOY_ID, "DOOR_WAIT_FOR_IDLE": "0", **overrides}
    rc, outcome, calls = _run("door-deploy.sh", env, **values)
    assert rc == 2 and outcome["code"] == code and not calls


def test_deploy_refuses_a_malformed_request_id_before_touching_any_path(env: dict[str, str]) -> None:
    rc, _, calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID="../../etc/x")
    assert rc == 2 and not calls and not list(Path(env["DOOR_RESULTS_DIR"]).iterdir())


@pytest.mark.parametrize("script", ["door-common.sh", "door-deploy.sh", "door-upgrade.sh", "door-hold-expire.sh",
                                    "door-retire-scene-workspace.sh", "door-restore-scene-workspace.sh", "install.sh"])
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
        assert re.search(r"set -e[A-Za-z]*uo pipefail", text) or script.name == "door-common.sh"


def test_installer_never_overwrites_tokens_and_validates_caddy_first() -> None:
    text = (DOOR / "install.sh").read_text(encoding="utf-8")
    assert 'if [ ! -e "$config_dir/tokens.json" ]; then' in text
    assert text.index("caddy validate") < text.index("systemctl reload caddy")
    assert "cp -p \"$backup\" \"$caddyfile\"" in text


def test_installer_keeps_root_writes_out_of_service_account_reach() -> None:
    text = (DOOR / "install.sh").read_text(encoding="utf-8")
    assert 'install -d -o root -g blueprint-door -m 2770 "$state_root/requests/pending"' in text
    assert 'install -d -o root -g root -m 0755 "$state_root/requests/$sub"' in text
    assert 'chown root:blueprint "$config_dir/tokens.json"' in text


def test_installer_rolls_back_code_and_units_and_checks_as_the_service_account() -> None:
    text = (DOOR / "install.sh").read_text(encoding="utf-8")
    assert "trap 'rollback; exit 1' ERR" in text and "set -eEuo pipefail" in text
    assert "trap 'rollback; exit 143' TERM" in text
    assert "trap 'rollback; exit 130' INT" in text
    assert text.index("trap 'rollback; exit 143' TERM") < text.index('mv "$stage" "$install_root"')
    assert text.rindex("trap - ERR TERM INT") > text.index("self-test --allow-no-tokens")
    assert '"$units_backup/$unit"' in text and 'mv "$install_root.previous" "$install_root"' in text
    assert "runuser -u blueprint --" in text and "self-test --allow-no-tokens" in text


def test_deploy_script_only_deploys_main() -> None:
    text = (DOOR / "door-deploy.sh").read_text(encoding="utf-8")
    code = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    assert not any("--canary" in line for line in code)
    assert any("door_require_on_main" in line for line in code)


def test_github_upstream_fetches_with_the_door_deploy_key(env: dict[str, str]) -> None:
    """The repository is private and the host has no other GitHub credential."""

    rc, outcome, calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_WAIT_FOR_IDLE="0")
    assert rc == 0 and outcome["status"] == "deployed"
    ssh = (f"ssh -i {env['DOOR_GITHUB_KEY']} -o IdentitiesOnly=yes -o BatchMode=yes"
           f" -o StrictHostKeyChecking=yes -o UserKnownHostsFile={env['DOOR_GITHUB_KNOWN_HOSTS']}")
    clone = next(call for call in calls if call.startswith("git clone"))
    assert f"--config core.sshCommand={ssh}" in clone
    assert "--config url.git@github.com:.insteadOf=https://github.com/" in clone
    source = env["DOOR_SOURCE_CLONE"]
    configured = calls.index(f"git -C {source} config core.sshCommand {ssh}")
    rewritten = calls.index(f"git -C {source} config url.git@github.com:.insteadOf https://github.com/")
    fetch = next(index for index, call in enumerate(calls) if call.startswith(f"git -C {source} fetch"))
    assert configured < fetch and rewritten < fetch


def test_github_upstream_without_a_deploy_key_is_refused(env: dict[str, str]) -> None:
    Path(env["DOOR_GITHUB_KEY"]).unlink()
    rc, outcome, calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_WAIT_FOR_IDLE="0")
    assert rc == 2 and outcome["code"] == "github_deploy_key_missing"
    assert not any(call.startswith(("git clone", "venv-python")) or " fetch " in call for call in calls)


def test_non_github_upstream_needs_no_deploy_key(env: dict[str, str]) -> None:
    Path(env["DOOR_GITHUB_KEY"]).unlink()
    rc, outcome, calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_WAIT_FOR_IDLE="0",
                              DOOR_UPSTREAM_URL="/srv/mirror/BlueprintCapturePipeline.git")
    assert rc == 0 and outcome["status"] == "deployed"
    assert not any("sshCommand" in call for call in calls)


def test_installer_creates_the_deploy_key_once_and_root_only() -> None:
    text = (DOOR / "install.sh").read_text(encoding="utf-8")
    assert 'install -d -o root -g root -m 0700 "$key_dir"' in text
    assert 'if [ ! -e "$key_dir/github" ]; then' in text
    assert "ssh-keygen -q -t ed25519 -N ''" in text
    assert "https://api.github.com/meta" in text


# --- retire-scene-workspace ---------------------------------------------------------------------------

RETIRE_ID = "20260926T120000Z-retire-scene-workspace-0000abcd"
RETIRE_PYTHON_STUB = r"""#!/bin/bash
{ echo "retention $*"; echo "cwd $PWD"; echo "pythonpath ${PYTHONPATH:-}"
  echo "from-env-file ${BLUEPRINT_FAKE_FROM_ENV_FILE:-unset}"
  echo "artifact-bucket ${BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_EXPECTED_BUCKET:-unset}"
  echo "probe ${BLUEPRINT_PROBE-unset}"; echo "ticks ${BLUEPRINT_TICKS-unset}"
  echo "credentials ${GOOGLE_APPLICATION_CREDENTIALS-unset}"
  echo "indented ${BLUEPRINT_INDENTED-unset}"; echo "exported ${BLUEPRINT_EXPORTED-unset}"
  echo "unlisted ${UNLISTED_SETTING-unset}"; } >> "$STUB_LOG"
while [ $# -gt 0 ]; do
  case "$1" in
    --result-out) [ -n "${FAKE_RETIREMENT:-}" ] && printf '%s' "$FAKE_RETIREMENT" > "$2"; shift ;;
  esac
  shift
done
exit "${FAKE_TOOL_RC:-0}"
"""


@pytest.fixture()
def retire_env(env: dict[str, str], tmp_path: Path) -> dict[str, str]:
    _write_stub(tmp_path / "retention-python", RETIRE_PYTHON_STUB)
    release = tmp_path / "releases" / SHA
    release.mkdir(parents=True)
    (tmp_path / "active").symlink_to(release)
    env_file = tmp_path / "pipeline-control-plane.env"
    env_file.write_text("BLUEPRINT_FAKE_FROM_ENV_FILE=loaded\nGOOGLE_APPLICATION_CREDENTIALS=/etc/blueprint/sa.json\n",
                        encoding="utf-8")
    values = {**env, "DOOR_VENV_PYTHON": str(tmp_path / "retention-python"),
              "DOOR_CONTROL_PLANE_REPO": str(tmp_path / "active"), "DOOR_CONTROL_PLANE_ENV_FILE": str(env_file),
              "DOOR_SCENE_ID": "site-capture-1"}
    for name in ("DOOR_COMMIT", "DOOR_SOURCE_CLONE", "DOOR_UPSTREAM_URL"):
        values.pop(name)  # a retirement fetches no code
    return values


@pytest.mark.parametrize(("result", "status", "code", "rc"), [
    ({"status": "planned", "plan": {"status": "retirable"}}, "planned", None, 0),
    ({"status": "retained", "reasons": ["open_scene_intent:scene-abc", "pinned"]}, "retained",
     "open_scene_intent:scene-abc", 0),
    ({"status": "retired", "receipt": "/r.json"}, "retired", None, 0),
    ({"status": "failed", "code": "scene_workspace_not_found"}, "failed", "scene_workspace_not_found", 1),
    ({"status": "retained", "reasons": ["raw_not_verified_in_cloud:captures/c/raw/a b\nc.mov"]}, "retained",
     "raw_not_verified_in_cloud:captures/c/raw/a_b_c.mov", 0),
    (None, "failed", "retention_exit_3", 3),
])
def test_retirement_outcome_follows_the_module_status(retire_env: dict[str, str], result, status, code, rc) -> None:
    extra = {"FAKE_RETIREMENT": json.dumps(result)} if result is not None else {"FAKE_TOOL_RC": "3"}
    done_rc, outcome, calls = _run("door-retire-scene-workspace.sh", retire_env, DOOR_REQUEST_ID=RETIRE_ID, **extra)

    assert (done_rc, outcome["status"], outcome["code"], outcome["exit_code"]) == (rc, status, code, rc)
    assert outcome["scene_id"] == "site-capture-1"
    assert outcome["result"] == f"{retire_env['DOOR_RESULTS_DIR']}/{RETIRE_ID}.retirement.json"
    assert not any(call.startswith(("git ", "systemctl ")) for call in calls)


def test_restore_runs_the_active_release_and_requires_a_restored_result(retire_env: dict[str, str]) -> None:
    request_id = "20260926T120000Z-restore-scene-workspace-0000abcd"
    values = {**retire_env, "DOOR_BUCKET": "blueprint-8c1ca.appspot.com",
              "BLUEPRINT_PUBSUB_HANDOFF_STORAGE_ROOT": "/var/lib/blueprint/pubsub-handoffs"}
    rc, outcome, calls = _run("door-restore-scene-workspace.sh", values, DOOR_REQUEST_ID=request_id,
                              FAKE_RETIREMENT=json.dumps({"status": "restored"}))
    assert rc == 0 and outcome["status"] == "restored"
    assert any("restore --receipt /var/lib/blueprint/pubsub-handoffs/blueprint-8c1ca.appspot.com/scenes/"
               "site-capture-1.retired.v1.json --destination /var/lib/blueprint/pubsub-handoffs/"
               "blueprint-8c1ca.appspot.com/scenes/site-capture-1" in call for call in calls)
    rc, outcome, _ = _run("door-restore-scene-workspace.sh", values, DOOR_REQUEST_ID=request_id,
                          FAKE_RETIREMENT=json.dumps({"status": "failed"}))
    assert rc == 1 and outcome["status"] == "failed"


def test_retirement_runs_the_active_release_module_with_the_control_plane_environment(
    retire_env: dict[str, str], tmp_path: Path
) -> None:
    rc, outcome, calls = _run("door-retire-scene-workspace.sh", retire_env, DOOR_REQUEST_ID=RETIRE_ID,
                              DOOR_BUCKET="blueprint-8c1ca.appspot.com", DOOR_APPLY="1",
                              FAKE_RETIREMENT=json.dumps({"status": "retired"}))

    result = f"{retire_env['DOOR_RESULTS_DIR']}/{RETIRE_ID}.retirement.json"
    assert rc == 0 and outcome["status"] == "retired"
    assert calls[0] == ("retention -m blueprint_pipeline.website_scene_workspace_retention retire "
                        "--scene-id site-capture-1 --bucket blueprint-8c1ca.appspot.com "
                        f"--apply --ack retire-scene-workspace --result-out {result}")
    assert calls[1:6] == [f"cwd {tmp_path / 'releases' / SHA}", "pythonpath src", "from-env-file loaded",
                          "artifact-bucket blueprint-task-evaluation-artifacts-prod", "probe unset"]
    assert "credentials /etc/blueprint/sa.json" in calls
    log = (Path(retire_env["DOOR_RESULTS_DIR"]) / f"{RETIRE_ID}.log").read_text(encoding="utf-8")
    assert "sa.json" not in log and "FAKE_FROM_ENV_FILE" not in log, "the environment file is never echoed"


def test_the_environment_file_is_read_as_data_never_run(retire_env: dict[str, str], tmp_path: Path) -> None:
    """systemd's KEY=VALUE format, not shell: nothing in it is expanded, run, or allowed to steer the script."""

    ran = tmp_path / "command-substitution-ran"
    Path(retire_env["DOOR_CONTROL_PLANE_ENV_FILE"]).write_text(
        "# the operator environment\n"
        "\n"
        f"BLUEPRINT_PROBE=$(touch {ran})\n"
        f"BLUEPRINT_TICKS=`touch {ran}`\n"
        'BLUEPRINT_FAKE_FROM_ENV_FILE="double quoted"\n'
        "GOOGLE_APPLICATION_CREDENTIALS='/etc/blueprint/sa.json'\n"
        "PATH=/nonexistent\n"
        "DOOR_SCENE_ID=hijacked\n"
        "UNLISTED_SETTING=1\n"
        "  BLUEPRINT_INDENTED=skipped\n"
        "export BLUEPRINT_EXPORTED=skipped\n"
        "not an assignment\n",
        encoding="utf-8")

    rc, outcome, calls = _run("door-retire-scene-workspace.sh", retire_env, DOOR_REQUEST_ID=RETIRE_ID,
                              FAKE_RETIREMENT=json.dumps({"status": "planned"}))

    assert rc == 0 and outcome["status"] == "planned", "PATH from the file would have broken the script"
    assert not ran.exists()
    assert f"probe $(touch {ran})" in calls and f"ticks `touch {ran}`" in calls
    assert "from-env-file double quoted" in calls and "credentials /etc/blueprint/sa.json" in calls
    assert "--scene-id site-capture-1 " in calls[0] and outcome["scene_id"] == "site-capture-1"
    assert {"indented unset", "exported unset", "unlisted unset"} <= set(calls)


def test_a_dry_run_passes_neither_apply_nor_the_ack(retire_env: dict[str, str]) -> None:
    rc, _, calls = _run("door-retire-scene-workspace.sh", retire_env, DOOR_REQUEST_ID=RETIRE_ID,
                        FAKE_RETIREMENT=json.dumps({"status": "planned"}))
    assert rc == 0 and "--apply" not in calls[0] and "--ack" not in calls[0] and "--bucket" not in calls[0]


@pytest.mark.parametrize(("overrides", "code"), [
    ({"DOOR_SCENE_ID": "../etc"}, "scene_id_invalid"),
    ({"DOOR_SCENE_ID": ".."}, "scene_id_invalid"),
    ({"DOOR_BUCKET": "Bad_Bucket"}, "bucket_invalid"),
    ({"DOOR_APPLY": "yes"}, "apply_invalid"),
])
def test_retirement_rechecks_its_inputs(retire_env: dict[str, str], overrides: dict[str, str], code: str) -> None:
    rc, outcome, calls = _run("door-retire-scene-workspace.sh", retire_env, DOOR_REQUEST_ID=RETIRE_ID, **overrides)
    assert rc == 2 and outcome["code"] == code and not calls


def test_retirement_refuses_another_kinds_request_id(retire_env: dict[str, str]) -> None:
    rc, _, calls = _run("door-retire-scene-workspace.sh", retire_env, DOOR_REQUEST_ID=DEPLOY_ID)
    assert rc == 2 and not calls and not list(Path(retire_env["DOOR_RESULTS_DIR"]).iterdir())


def test_retirement_archives_to_the_same_artifact_store_as_the_reclaim_timer() -> None:
    gc = (DOOR.parents[1] / "deploy/systemd/blueprint-control-plane-storage-gc.service").read_text(encoding="utf-8")
    script = (DOOR / "door-retire-scene-workspace.sh").read_text(encoding="utf-8")
    bindings = [line.split("=", 1)[1] for line in gc.splitlines()
                if line.startswith("Environment=BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_")]
    assert bindings
    for binding in bindings:
        name, value = binding.split("=", 1)
        assert f': "${{{name}:={value}}}"' in script, name


def test_installer_copies_every_door_script() -> None:
    lines = (DOOR / "install.sh").read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith('cp "$source_dir"/door-common.sh'))
    copy = " ".join(lines[start:start + 2])  # the staging copy command and its continuation line
    for script in sorted(DOOR.glob("door-*.sh")):
        assert f'"$source_dir"/{script.name}' in copy, script.name


def test_hold_expiry_script_releases_only_matching_active_expired_generation(tmp_path: Path) -> None:
    holds = tmp_path / "holds"
    holds.mkdir()
    unit = "blueprint-scene-progression.timer"
    old_id = "20260926T120000Z-hold-0000abcd"
    new_id = "20260926T120001Z-hold-0000abce"
    record = {"schema": "blueprint_operator_door_hold.v1", "unit": unit, "owner": "alice",
              "reason": "inspect", "requested_by": "cloud", "request_id": new_id,
              "created_at": "2026-09-26T12:00:01+00:00", "expires_at": "2026-09-26T12:01:01+00:00",
              "expires_at_epoch": 1, "status": "active"}
    path = holds / f"{unit}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    log = tmp_path / "systemctl.log"
    _write_stub(stubs / "systemctl", '#!/bin/bash\necho "$*" >> "$STUB_LOG"\nexit 0\n')
    values = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "STUB_LOG": str(log),
              "DOOR_HOLDS_DIR": str(holds), "DOOR_HOLD_UNIT": unit, "DOOR_HOLD_REQUEST_ID": old_id}

    def run(request_id: str) -> int:
        return subprocess.run(["/bin/bash", str(DOOR / "door-hold-expire.sh")],
                              env={**values, "DOOR_HOLD_REQUEST_ID": request_id}, check=False).returncode

    assert run(old_id) == 0
    assert not log.exists(), "an old expiry cannot release a renewed hold"
    assert run(new_id) == 0
    assert log.read_text().splitlines() == [f"--no-block start -- {unit}"]
    assert json.loads(path.read_text())["status"] == "expired_released"
    assert run(new_id) == 0
    assert log.read_text().splitlines() == [f"--no-block start -- {unit}"], "expiry is idempotent"
