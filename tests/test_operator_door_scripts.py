"""The door's transient scripts, run for real against stub git/systemctl/python."""

# Covers (for impacted-test selection):
#   deploy/operator-door/door-common.sh
#   deploy/operator-door/door-deploy.sh
#   deploy/operator-door/door-upgrade.sh
#   deploy/operator-door/door-retire-scene-workspace.sh
#   deploy/operator-door/door-restore-scene-workspace.sh
#   deploy/operator-door/door-provider-output-resume.sh
#   deploy/operator-door/door-repair-notifier-binding.sh
#   deploy/operator-door/install.sh

from __future__ import annotations

import json
import hashlib
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


@pytest.mark.parametrize("valid_identity", [False, True])
def test_notifier_result_files_are_readable_by_the_door(env: dict[str, str], valid_identity: bool) -> None:
    install_root = Path(env["STUB_DIR"]) / "installed"
    module = install_root / "operator_door/notifier_repair.py"
    module.parent.mkdir(parents=True)
    module.write_text("import json, sys\nfrom pathlib import Path\n"
                      "path = Path(sys.argv[sys.argv.index('--receipt-out') + 1])\n"
                      "path.write_text(json.dumps({'status': 'already_correct'}))\n"
                      "print('safe fixed-target receipt')\n")
    request_id = "20261002T140000Z-unit-0000abcd"
    rc, outcome, _calls = _run("door-repair-notifier-binding.sh", env, DOOR_REQUEST_ID=request_id,
                               DOOR_INSTALL_ROOT=str(install_root),
                               DOOR_EXPECTED_POSTCHECK_SHA256="sha256:" + "a" * 64 if valid_identity else "invalid",
                               DOOR_EXPECTED_SOURCE_COMMIT=SHA)
    assert rc == (0 if valid_identity else 2)
    assert outcome["status"] == ("repaired" if valid_identity else "refused")
    for suffix in ("log", "outcome.json"):
        path = Path(env["DOOR_RESULTS_DIR"]) / f"{request_id}.{suffix}"
        assert stat.S_IMODE(path.stat().st_mode) == 0o644


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


@pytest.fixture()
def canonical_env(env: dict[str, str]) -> dict[str, str]:
    # Fixed names only: provider/auth environment must never reach subprocesses
    # or appear in a canonical test's assertion fixture repr.
    return {key: value for key, value in env.items()
            if key.startswith(("DOOR_", "STUB_")) or key in {"PATH", "HOME", "TMPDIR"}}


def test_canonical_deploy_uses_provenance_and_preserves_controls(canonical_env: dict[str, str]) -> None:
    env = canonical_env
    provenance = Path(env["DOOR_RESULTS_DIR"]) / f"{DEPLOY_ID}.release-provenance.json"
    payload = json.dumps({"schema_version": "blueprint.deploy_release_provenance.v1",
                          "git_sha": SHA, "status": "verified"}).encode()
    provenance.write_bytes(payload)
    provenance.chmod(0o600)
    rc, outcome, calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID,
                              DOOR_WAIT_FOR_IDLE="0", DOOR_RELEASE_PROVENANCE_FILE=str(provenance),
                              DOOR_RELEASE_PROVENANCE_SHA256=hashlib.sha256(payload).hexdigest())
    assert rc == 0 and outcome["status"] == "deployed"
    tool = _tool_call(calls)
    assert f"--release-provenance {provenance}" in tool
    assert "--iteration" not in tool and "--canary" not in tool
    assert "--preserve-configured-controls-state" in tool
    assert outcome["receipt"].endswith(f"production_{SHA[:12]}_door.json")
    assert any("merge-base --is-ancestor " + SHA + " origin/main" in call for call in calls)


@pytest.mark.parametrize("case", ["missing_file", "missing_hash", "wrong_path", "bad_hash", "hash_mismatch",
                                  "oversized", "empty", "symlink", "wrong_mode"])
def test_canonical_deploy_refuses_invalid_staged_bytes_before_git(canonical_env: dict[str, str], case: str) -> None:
    env = canonical_env
    payload = b'{"status":"verified"}'
    if case == "oversized":
        payload = b" " * (16 * 1024 + 1)
    if case == "empty":
        payload = b""
    provenance = Path(env["DOOR_RESULTS_DIR"]) / f"{DEPLOY_ID}.release-provenance.json"
    provenance.write_bytes(payload)
    provenance.chmod(0o600)
    digest = hashlib.sha256(payload).hexdigest()
    overrides = {"DOOR_RELEASE_PROVENANCE_FILE": str(provenance), "DOOR_RELEASE_PROVENANCE_SHA256": digest}
    if case == "missing_file":
        overrides.pop("DOOR_RELEASE_PROVENANCE_FILE")
    if case == "missing_hash":
        overrides.pop("DOOR_RELEASE_PROVENANCE_SHA256")
    if case == "wrong_path":
        overrides["DOOR_RELEASE_PROVENANCE_FILE"] = str(provenance.parent / "other.json")
    if case == "bad_hash":
        overrides["DOOR_RELEASE_PROVENANCE_SHA256"] = "invalid"
    if case == "hash_mismatch":
        overrides["DOOR_RELEASE_PROVENANCE_SHA256"] = "0" * 64
    if case == "wrong_mode":
        provenance.chmod(0o644)
    if case == "symlink":
        original = provenance.with_suffix(".original")
        provenance.rename(original)
        provenance.symlink_to(original)
    rc, outcome, calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID,
                              DOOR_WAIT_FOR_IDLE="0", **overrides)
    assert rc == 2 and outcome["code"] == "release_provenance_transport_invalid"
    assert not any(call.startswith(("git ", "venv-python ")) for call in calls)


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
                                    "door-retire-scene-workspace.sh", "door-restore-scene-workspace.sh",
                                    "door-lane-scratch.sh", "door-provider-output-resume.sh", "install.sh"])
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


def test_lane_scratch_script_uses_the_active_release_and_exact_lease_arguments(retire_env: dict[str, str]) -> None:
    request_id = "20260926T120000Z-lane-scratch-0000abcd"
    values = {**retire_env, "DOOR_SCRATCH_ROOT": "/mnt/blueprint-work/lanes", "DOOR_SCRATCH_ACTION": "release",
              "DOOR_SCRATCH_LANE": "g1", "DOOR_SCRATCH_NAME": "run-1", "DOOR_SCRATCH_OWNER": "agent-1",
              "DOOR_SCRATCH_EXPECTED_DIGEST": "sha256:" + "a" * 64,
              "FAKE_RETIREMENT": '{"status":"released"}'}
    rc, outcome, calls = _run("door-lane-scratch.sh", values, DOOR_REQUEST_ID=request_id)
    assert rc == 0 and outcome["status"] == "released"
    assert any("-m blueprint_pipeline.control_plane_lane_scratch_door release" in call for call in calls)
    assert any("--root /mnt/blueprint-work/lanes" in call for call in calls)
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
              "expires_at_epoch": 1, "enabled_before": True, "status": "active"}
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
    record["expires_at_epoch"] = 4070908800
    path.write_text(json.dumps(record), encoding="utf-8")
    assert run(new_id) == 0
    assert not log.exists(), "a current hold cannot be released before its expiry"
    assert json.loads(path.read_text())["status"] == "active"
    record["expires_at_epoch"] = 1
    path.write_text(json.dumps(record), encoding="utf-8")
    assert run(new_id) == 0
    assert log.read_text().splitlines() == [f"enable -- {unit}", f"--no-block start -- {unit}"]
    assert not path.exists()
    archived = holds / "history" / f"{unit}.{new_id}.json"
    assert json.loads(archived.read_text())["status"] == "expired_released"
    assert run(new_id) == 0
    assert log.read_text().splitlines() == [f"enable -- {unit}", f"--no-block start -- {unit}"], "expiry is idempotent"



# --- provider-output-resume ---------------------------------------------------------------------

RESUME_ID = "20260929T120000Z-provider-output-resume-0000abcd"
SETPRIV_STUB = r"""#!/bin/bash
options=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do options+=("$1"); shift; done
shift
{ echo "setpriv ${options[*]} --"; echo "umask $(umask)"; } >> "$STUB_LOG"
exec "$@"
"""
RESUME_PYTHON_STUB = r"""#!/bin/bash
{ echo "resume $*"; echo "cwd $PWD"; echo "pythonpath ${PYTHONPATH:-}"
  echo "artifact-bucket ${BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_EXPECTED_BUCKET:-unset}"
  echo "ledger ${BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT:-unset}"
  echo "from-env-file ${BLUEPRINT_FAKE_FROM_ENV_FILE:-unset}"; } >> "$STUB_LOG"
printf '%s\n' "${FAKE_RESUME:-}"
exit "${FAKE_TOOL_RC:-0}"
"""


@pytest.fixture()
def resume_env(retire_env: dict[str, str], tmp_path: Path) -> dict[str, str]:
    _write_stub(tmp_path / "stubs" / "setpriv", SETPRIV_STUB)
    _write_stub(tmp_path / "resume-python", RESUME_PYTHON_STUB)
    canaries = tmp_path / "canaries"
    (canaries / "activation-1" / "allocator" / "attempts" / "attempt_003").mkdir(parents=True)
    values = {**retire_env, "DOOR_VENV_PYTHON": str(tmp_path / "resume-python"), "DOOR_CANARY_ROOT": str(canaries),
              "DOOR_RUN": "activation-1", "DOOR_ATTEMPT": "3", "DOOR_SERVICE_USER": "blueprint"}
    values.pop("DOOR_SCENE_ID")
    return values


@pytest.mark.parametrize(("ingest", "resume", "status", "code", "rc"), [
    ("1", {"status": "completed", "blockers": []}, "completed", None, 0),
    (None, {"status": "blocked", "blockers": ["staged_output_promotion_receipt_missing"]}, "blocked",
     "staged_output_promotion_receipt_missing", 1),
])
def test_provider_output_resume_runs_the_release_module_as_the_service_user(
        resume_env: dict[str, str], tmp_path: Path, ingest, resume, status, code, rc) -> None:
    """Review I5: the attempt tree belongs to the service user, so the resume runs as ``blueprint``
    with a private umask; the root script only keeps the door's log, result and outcome."""
    extra = {"FAKE_RESUME": json.dumps(resume), **({"DOOR_INGEST": ingest} if ingest else {})}
    done_rc, outcome, calls = _run("door-provider-output-resume.sh", resume_env, DOOR_REQUEST_ID=RESUME_ID, **extra)

    attempt = tmp_path / "canaries/activation-1/allocator/attempts/attempt_003"
    assert (done_rc, outcome["status"], outcome["code"], outcome["exit_code"]) == (rc, status, code, rc)
    assert calls[0] == "setpriv --reuid=blueprint --regid=blueprint --init-groups --inh-caps=-all --"
    assert calls[1] == "umask 0077"
    assert calls[2] == ("resume -m blueprint_pipeline.provider_output_promotion resume --attempt-root "
                        f"{attempt}" + (" --ingest" if ingest else ""))
    assert calls[3:8] == [f"cwd {tmp_path / 'releases' / SHA}", "pythonpath src",
                          "artifact-bucket blueprint-task-evaluation-artifacts-prod",
                          "ledger /var/lib/blueprint/pipeline-control-plane/disk-reservations",
                          "from-env-file loaded"]
    result = Path(resume_env["DOOR_RESULTS_DIR"]) / f"{RESUME_ID}.provider-output-resume.json"
    assert outcome["result"] == str(result) and json.loads(result.read_text()) == resume
    assert oct(result.stat().st_mode & 0o777) == oct(0o644)  # the door reads it
    assert not any(call.startswith(("git ", "systemctl ")) for call in calls)


@pytest.mark.parametrize(("overrides", "code"), [
    ({"DOOR_RUN": "../activation-1"}, "provider_output_resume_run_invalid"),
    ({"DOOR_RUN": ".."}, "provider_output_resume_run_invalid"),
    ({"DOOR_ATTEMPT": "0"}, "provider_output_resume_attempt_invalid"),
    ({"DOOR_ATTEMPT": "3; rm -rf /"}, "provider_output_resume_attempt_invalid"),
    ({"DOOR_INGEST": "yes"}, "provider_output_resume_ingest_invalid"),
    ({"DOOR_SERVICE_USER": "root"}, "provider_output_resume_user_invalid"),
    ({"DOOR_ATTEMPT": "4"}, "provider_output_resume_attempt_missing"),
])
def test_provider_output_resume_rechecks_its_inputs(resume_env: dict[str, str], overrides, code) -> None:
    rc, outcome, calls = _run("door-provider-output-resume.sh", resume_env, DOOR_REQUEST_ID=RESUME_ID, **overrides)
    assert rc == 2 and outcome["code"] == code and not calls


def test_provider_output_resume_never_follows_a_link_out_of_the_canary_root(
        resume_env: dict[str, str], tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere" / "attempts" / "attempt_003"
    elsewhere.mkdir(parents=True)
    run = Path(resume_env["DOOR_CANARY_ROOT"]) / "activation-2"
    run.mkdir()
    (run / "allocator").symlink_to(elsewhere.parent.parent, target_is_directory=True)
    rc, outcome, calls = _run("door-provider-output-resume.sh", {**resume_env, "DOOR_RUN": "activation-2"},
                              DOOR_REQUEST_ID=RESUME_ID)
    assert rc == 2 and outcome["code"] == "provider_output_resume_attempt_outside_root" and not calls
