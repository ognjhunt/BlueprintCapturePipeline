"""The door's transient scripts, run for real against stub git/systemctl/python."""

# Covers (for impacted-test selection):
#   deploy/operator-door/door-common.sh
#   deploy/operator-door/door-deploy.sh
#   deploy/operator-door/door-upgrade.sh
#   deploy/operator-door/door-retire-scene-workspace.sh
#   deploy/operator-door/door-restore-scene-workspace.sh
#   deploy/operator-door/door-provider-output-resume.sh
#   deploy/operator-door/door-selected-handoff.sh
#   deploy/operator-door/door-extend-scene-execution-window.sh
#   deploy/operator-door/door-repair-notifier-binding.sh
#   deploy/operator-door/install.sh

from __future__ import annotations

import json
import hashlib
import os
import re
import stat
import subprocess
import sys
import time
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
    if [ "${args[1]}" = "add" ]; then
      mkdir -p "${args[4]}"
      if [ -n "${FAKE_TOOL_DRAINS:-}" ]; then
        mkdir -p "${args[4]}/scripts"
        echo 'parser.add_argument("--drain-unit", action="append")' > "${args[4]}/scripts/deploy_control_plane_commit.py"
      fi
    else rm -rf "$last"; fi ;;
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
echo "staging-tmp $TMPDIR" >> "$STUB_LOG"
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
    _write_stub(stubs / "python3", '#!/bin/bash\n'
                'if [[ "$*" == *"/operator_door/deploy_staging.py"* ]]; then\n'
                '  echo staging-helper >> "$STUB_LOG"\n'
                '  [ "${FAKE_STAGING_REFUSED:-0}" = 0 ] || exit 2\n'
                '  echo "$STUB_DIR/work-staging"; exit 0\nfi\n'
                f'exec "{sys.executable}" "$@"\n')
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


def test_deploy_binds_temp_admission_and_child_builders_to_fixed_parent(env: dict[str, str]) -> None:
    rc, outcome, calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID,
                              TMPDIR="/untrusted/caller/override")
    assert rc == 0 and outcome["status"] == "deployed"
    assert "staging-helper" in calls
    assert f"staging-tmp {env['STUB_DIR']}/work-staging" in calls


def test_deploy_refuses_bad_staging_parent_before_source_or_deploy_tool(env: dict[str, str]) -> None:
    rc, outcome, calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID,
                              FAKE_STAGING_REFUSED="1")
    assert rc == 2 and outcome["code"] == "deploy_staging_parent_unsafe"
    assert calls == ["staging-helper"]


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
    assert not any(call.startswith("venv-python") for call in calls)


def test_deploy_lets_a_draining_tool_stop_triggers_before_waiting(env: dict[str, str]) -> None:
    """A tool that drains after quiesce replaces the door's racy idle poll."""
    rc, outcome, calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_WAIT_FOR_IDLE="1",
                              FAKE_TOOL_DRAINS="1", FAKE_BUSY_UNIT="blueprint-a.service", FAKE_BUSY_POLLS="99")
    assert rc == 0 and outcome["status"] == "deployed"
    assert not any("is-active" in call for call in calls)
    assert ("--drain-unit blueprint-a.service --drain-unit blueprint-b.service --drain-seconds 30"
            in _tool_call(calls))


def test_no_wait_deploy_never_drains(env: dict[str, str]) -> None:
    rc, outcome, calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_WAIT_FOR_IDLE="0",
                              FAKE_TOOL_DRAINS="1")
    assert rc == 0 and outcome["status"] == "deployed"
    assert "--drain-unit" not in _tool_call(calls)
    assert not any("is-active" in call for call in calls)


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
                                    "door-lane-scratch.sh", "door-provider-output-resume.sh",
                                    "door-extend-scene-execution-window.sh", "door-extend-scene-execution-budget.sh",
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


def test_deploy_log_times_every_step_before_the_tool_starts(env: dict[str, str]) -> None:
    """About 8 of a deploy's 40 minutes passed before the tool's first stage, unrecorded."""
    rc, outcome, _calls = _run("door-deploy.sh", env, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_WAIT_FOR_IDLE="0")
    assert rc == 0 and outcome["status"] == "deployed"
    log = (Path(env["DOOR_RESULTS_DIR"]) / f"{DEPLOY_ID}.log").read_text(encoding="utf-8")
    phases = re.findall(r"^\[\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\] phase=(\w+)$", log, re.MULTILINE)
    assert phases == [
        "staging_guard_start", "staging_guard_done", "activation_hold_start", "activation_hold_done",
        "fetch_main_start", "fetch_main_done", "prepare_source_done", "require_on_main_done", "tool_worktree_done", "tool_start",
    ]


def _real_git(*args: str, cwd: Path) -> str:
    done = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                          cwd=cwd, capture_output=True, text=True, check=True)
    return done.stdout.strip()


@pytest.fixture()
def door_clone(tmp_path: Path) -> dict[str, object]:
    """A clone that an older door filled with every branch, and an upstream that moved on."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _real_git("init", "--quiet", "--initial-branch=main", cwd=upstream)
    _real_git("commit", "--quiet", "--allow-empty", "-m", "first", cwd=upstream)
    _real_git("branch", "old-feature", cwd=upstream)
    clone = tmp_path / "tools" / "operator-door-source"
    clone.parent.mkdir()
    _real_git("clone", "--quiet", "--no-checkout", str(upstream), str(clone), cwd=tmp_path)
    assert "refs/remotes/origin/old-feature" in _real_git("for-each-ref", cwd=clone)
    assert "refs/remotes/origin/HEAD" in _real_git("for-each-ref", cwd=clone)
    _real_git("commit", "--quiet", "--allow-empty", "-m", "second", cwd=upstream)
    merged = _real_git("rev-parse", "HEAD", cwd=upstream)
    _real_git("branch", "new-feature", cwd=upstream)
    _real_git("switch", "--quiet", "new-feature", cwd=upstream)
    _real_git("commit", "--quiet", "--allow-empty", "-m", "unmerged", cwd=upstream)
    unmerged = _real_git("rev-parse", "HEAD", cwd=upstream)
    _real_git("switch", "--quiet", "main", cwd=upstream)
    results = tmp_path / "results"
    results.mkdir()
    return {"upstream": upstream, "clone": clone, "merged": merged, "unmerged": unmerged, "results": results}


def _prepare_source(door_clone: dict[str, object], commit: str) -> dict:
    """Run the door's own helpers with real git against a real upstream."""
    script = (f'set -euo pipefail\n. "{DOOR}/door-common.sh"\n'
              'door_init_request deploy\ndoor_init_git\ndoor_prepare_source\ndoor_require_on_main\n'
              'door_outcome deployed "" 0\n')
    done = subprocess.run(["/bin/bash", "-c", script], capture_output=True, text=True, check=False, env={
        **os.environ, "DOOR_REQUEST_ID": DEPLOY_ID, "DOOR_RESULTS_DIR": str(door_clone["results"]),
        "DOOR_COMMIT": commit, "DOOR_SOURCE_CLONE": str(door_clone["clone"]),
        "DOOR_UPSTREAM_URL": str(door_clone["upstream"]), "DOOR_VENV_PYTHON": "/nonexistent",
    })
    outcome = json.loads((Path(door_clone["results"]) / f"{DEPLOY_ID}.outcome.json").read_text(encoding="utf-8"))
    return {"rc": done.returncode, **outcome}


def test_door_fetches_only_main_and_forgets_branches_an_older_door_fetched(door_clone: dict[str, object]) -> None:
    clone = door_clone["clone"]

    outcome = _prepare_source(door_clone, str(door_clone["merged"]))

    assert outcome["rc"] == 0 and outcome["status"] == "deployed"
    assert _real_git("rev-parse", "refs/remotes/origin/main", cwd=clone) == door_clone["merged"]
    refs = _real_git("for-each-ref", "--format=%(refname)", "refs/remotes/origin", cwd=clone).splitlines()
    # origin/HEAD is a symbolic ref to main: deleting it through update-ref would
    # follow it and delete origin/main, which is the door's whole deploy check.
    assert sorted(refs) == ["refs/remotes/origin/HEAD", "refs/remotes/origin/main"]
    assert _real_git("rev-parse", "origin/HEAD", cwd=clone) == door_clone["merged"]

    # The second deploy has nothing left to forget and still succeeds.
    assert _prepare_source(door_clone, str(door_clone["merged"]))["status"] == "deployed"


def test_door_still_refuses_a_pushed_commit_main_does_not_contain(door_clone: dict[str, object]) -> None:
    outcome = _prepare_source(door_clone, str(door_clone["unmerged"]))

    assert outcome["rc"] == 2 and outcome["code"] == "commit_not_on_main"


def test_door_still_refuses_a_commit_nobody_pushed(door_clone: dict[str, object]) -> None:
    outcome = _prepare_source(door_clone, "f" * 40)

    assert outcome["rc"] == 2 and outcome["code"] == "commit_not_found"


PARKED_OLDER_ID = "20260923T115900Z-deploy-0000aaaa"


@pytest.fixture()
def parked_env(door_clone: dict[str, object], tmp_path: Path) -> dict[str, str]:
    """The real door-deploy.sh, parked_settle.py and git. Only the deploy tool is a stub."""
    stubs = tmp_path / "parked-stubs"
    stubs.mkdir()
    _write_stub(stubs / "systemctl", SYSTEMCTL_STUB)
    _write_stub(tmp_path / "venv-python", PYTHON_STUB)
    _write_stub(stubs / "python3", '#!/bin/bash\n'
                'if [[ "$*" == *"/operator_door/deploy_staging.py"* ]]; then\n'
                '  [ -z "${FAKE_STAGING_REFUSED:-}" ] || exit 1\n'
                '  echo "$STUB_DIR/work-staging"; exit 0\n'
                'fi\n'
                f'exec "{sys.executable}" "$@"\n')
    (tmp_path / "parked").mkdir()
    (tmp_path / "state" / "deploy-receipts").mkdir(parents=True)
    return {
        **os.environ,
        "PATH": f"{stubs}:{os.environ['PATH']}",
        "STUB_LOG": str(tmp_path / "parked-calls.log"),
        "STUB_DIR": str(tmp_path),
        "DOOR_REQUEST_ID": DEPLOY_ID,
        "DOOR_RESULTS_DIR": str(door_clone["results"]),
        "DOOR_SOURCE_CLONE": str(door_clone["clone"]),
        "DOOR_REFERENCE_REPO": str(tmp_path / "no-reference"),
        "DOOR_UPSTREAM_URL": str(door_clone["upstream"]),
        "DOOR_VENV_PYTHON": str(tmp_path / "venv-python"),
        "DOOR_STATE_ROOT": str(tmp_path / "state"),
        "DOOR_WAIT_FOR_IDLE": "0",
        "DOOR_PARKED": "1",
        "DOOR_PARKED_DIR": str(tmp_path / "parked"),
        "DOOR_ACTIVE_RELEASE_LINK": str(tmp_path / "active"),
    }


def _park_for_script(values: dict[str, str], request_id: str, commit: str, *, active_at_park: str = "") -> None:
    path = Path(values["DOOR_PARKED_DIR"]) / f"{request_id}.json"
    path.write_text(json.dumps({"schema": "blueprint_operator_door_parked_deploy.v1", "id": request_id,
                                "commit": commit, "wait_for_idle": False, "parked_at_epoch": 1.0,
                                "active_release_at_park": active_at_park}), encoding="utf-8")
    path.chmod(0o644)


def _activate_for_script(values: dict[str, str], commit: str, *, proven: bool) -> None:
    link = Path(values["DOOR_ACTIVE_RELEASE_LINK"])
    link.unlink(missing_ok=True)
    link.symlink_to(Path(values["STUB_DIR"]) / "releases" / commit)
    if proven:
        receipt = Path(values["DOOR_STATE_ROOT"]) / "deploy-receipts" / f"iteration_{commit[:12]}_door.json"
        receipt.write_text(json.dumps({"status": "deployed", "source_commit": commit}), encoding="utf-8")


def _run_parked(values: dict[str, str], commit: str, request_id: str = DEPLOY_ID) -> tuple[int, dict, list[str]]:
    done = subprocess.run(["/bin/bash", str(DOOR / "door-deploy.sh")], capture_output=True, text=True, check=False,
                          timeout=120, env={**values, "DOOR_COMMIT": commit, "DOOR_REQUEST_ID": request_id})
    path = Path(values["DOOR_RESULTS_DIR"]) / f"{request_id}.outcome.json"
    outcome = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    log = Path(values["STUB_LOG"])
    return done.returncode, outcome, log.read_text(encoding="utf-8").splitlines() if log.exists() else []


def _first_commit(door_clone: dict[str, object]) -> str:
    return _real_git("rev-parse", f"{door_clone['merged']}~1", cwd=door_clone["upstream"])


def _older_parked_outcome(values: dict[str, str]) -> dict:
    path = Path(values["DOOR_RESULTS_DIR"]) / f"{PARKED_OLDER_ID}.outcome.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _parked_names(values: dict[str, str]) -> list[str]:
    return sorted(path.stem for path in Path(values["DOOR_PARKED_DIR"]).glob("*.json"))


def test_a_parked_deploy_runs_the_commit_containing_the_others_whichever_asked(parked_env, door_clone) -> None:
    first, newest = _first_commit(door_clone), str(door_clone["merged"])
    _activate_for_script(parked_env, first, proven=True)
    _park_for_script(parked_env, PARKED_OLDER_ID, newest, active_at_park=first)
    _park_for_script(parked_env, DEPLOY_ID, first, active_at_park=first)
    # The newest request names what already runs, so it settles alone and the next one deploys.
    rc, outcome, calls = _run_parked(parked_env, first)
    assert rc == 0 and outcome["status"] == "superseded"
    assert not any(call.startswith("venv-python") for call in calls)

    rc, outcome, calls = _run_parked(parked_env, newest, PARKED_OLDER_ID)

    assert rc == 0 and outcome["status"] == "deployed" and outcome["commit"] == newest
    tool = _tool_call(calls)
    assert f"--source-commit {newest}" in tool
    assert tool.endswith(f"iteration_{newest[:12]}_door.json")
    assert _parked_names(parked_env) == []


def _second_on_main(door_clone) -> str:
    """A commit that contains the fixture's merged commit, and one on main that does not."""
    upstream = door_clone["upstream"]
    _real_git("commit", "--quiet", "--allow-empty", "-m", "third", cwd=upstream)
    return _real_git("rev-parse", "HEAD", cwd=upstream)


def test_a_request_a_deploy_covers_ends_only_after_a_receipt_proves_that_deploy(parked_env, door_clone) -> None:
    first, second = _first_commit(door_clone), str(door_clone["merged"])
    third = _second_on_main(door_clone)
    _activate_for_script(parked_env, first, proven=True)
    _park_for_script(parked_env, PARKED_OLDER_ID, second, active_at_park=first)
    _park_for_script(parked_env, DEPLOY_ID, third, active_at_park=first)

    rc, outcome, calls = _run_parked(parked_env, third)

    assert rc == 0 and outcome["status"] == "deployed" and outcome["commit"] == third
    assert f"--source-commit {third}" in _tool_call(calls)
    assert _parked_names(parked_env) == [PARKED_OLDER_ID]
    older = Path(parked_env["DOOR_RESULTS_DIR"]) / f"{PARKED_OLDER_ID}.outcome.json"
    assert not older.exists()

    _activate_for_script(parked_env, third, proven=True)
    rc, outcome, calls = _run_parked(parked_env, second, PARKED_OLDER_ID)
    assert rc == 0 and outcome["status"] == "superseded" and outcome["superseded_by_commit"] == third
    assert _parked_names(parked_env) == []


def test_when_the_newest_commit_fails_the_older_ones_keep_their_own_try(parked_env, door_clone) -> None:
    first, second = _first_commit(door_clone), str(door_clone["merged"])
    third = _second_on_main(door_clone)
    _activate_for_script(parked_env, first, proven=True)
    _park_for_script(parked_env, PARKED_OLDER_ID, second, active_at_park=first)
    _park_for_script(parked_env, DEPLOY_ID, third, active_at_park=first)

    rc, outcome, _ = _run_parked({**parked_env, "FAKE_TOOL_RC": "2"}, third)

    assert rc == 2 and outcome["status"] == "failed" and outcome["code"] == "deploy_tool_exit_2"
    assert _parked_names(parked_env) == [PARKED_OLDER_ID]
    assert not (Path(parked_env["DOOR_RESULTS_DIR"]) / f"{PARKED_OLDER_ID}.outcome.json").exists()

    Path(parked_env["STUB_LOG"]).unlink()
    rc, outcome, calls = _run_parked(parked_env, second, PARKED_OLDER_ID)

    assert rc == 0 and outcome["status"] == "deployed" and outcome["commit"] == second
    assert f"--source-commit {second}" in _tool_call(calls)
    assert _parked_names(parked_env) == []


def _side_commit_merged_after(door_clone, base: str) -> tuple[str, str, str]:
    """base - second - third - merge(side), with side branching from base."""
    upstream = door_clone["upstream"]
    third = _second_on_main(door_clone)
    _real_git("switch", "--quiet", "-c", "side", base, cwd=upstream)
    _real_git("commit", "--quiet", "--allow-empty", "-m", "side", cwd=upstream)
    side = _real_git("rev-parse", "HEAD", cwd=upstream)
    _real_git("switch", "--quiet", "main", cwd=upstream)
    _real_git("merge", "--quiet", "--no-ff", "-m", "merge side", "side", cwd=upstream)
    return third, side, _real_git("rev-parse", "main", cwd=upstream)


def _outcome_of(values: dict[str, str], request_id: str) -> dict:
    path = Path(values["DOOR_RESULTS_DIR"]) / f"{request_id}.outcome.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def test_a_red_commit_someone_else_named_fails_that_request_and_the_starter_gets_its_try(parked_env, door_clone) -> None:
    """The request that started has the good, older commit, and the door still tries the newest first."""
    first, second = _first_commit(door_clone), str(door_clone["merged"])
    third = _second_on_main(door_clone)
    _activate_for_script(parked_env, first, proven=True)
    _park_for_script(parked_env, PARKED_OLDER_ID, third, active_at_park=first)
    _park_for_script(parked_env, DEPLOY_ID, second, active_at_park=first)

    rc, outcome, calls = _run_parked({**parked_env, "FAKE_TOOL_RC": "2"}, second)

    assert rc == 2 and outcome == {}, "this request is not the one that failed"
    assert f"--source-commit {third}" in _tool_call(calls)
    older = _outcome_of(parked_env, PARKED_OLDER_ID)
    assert older["status"] == "failed" and older["code"] == "deploy_tool_exit_2"
    assert _parked_names(parked_env) == [DEPLOY_ID]
    log = Path(parked_env["DOOR_RESULTS_DIR"])
    assert f"deploying {third} for request {PARKED_OLDER_ID}" in (log / f"{DEPLOY_ID}.log").read_text(encoding="utf-8")
    assert "phase=tool_start" in (log / f"{PARKED_OLDER_ID}.log").read_text(encoding="utf-8")

    Path(parked_env["STUB_LOG"]).unlink()
    rc, outcome, calls = _run_parked(parked_env, second)

    assert rc == 0 and outcome["status"] == "deployed" and outcome["commit"] == second
    assert f"--source-commit {second}" in _tool_call(calls)
    assert _parked_names(parked_env) == []


def test_a_good_commit_someone_else_named_succeeds_for_it_and_covers_the_starter(parked_env, door_clone) -> None:
    first, second = _first_commit(door_clone), str(door_clone["merged"])
    third = _second_on_main(door_clone)
    _activate_for_script(parked_env, first, proven=True)
    _park_for_script(parked_env, PARKED_OLDER_ID, third, active_at_park=first)
    _park_for_script(parked_env, DEPLOY_ID, second, active_at_park=first)

    rc, outcome, _ = _run_parked(parked_env, second)

    assert rc == 0 and outcome == {}
    assert _outcome_of(parked_env, PARKED_OLDER_ID)["status"] == "deployed"
    assert _outcome_of(parked_env, PARKED_OLDER_ID)["commit"] == third
    assert _parked_names(parked_env) == [DEPLOY_ID]

    _activate_for_script(parked_env, third, proven=True)
    rc, outcome, _ = _run_parked(parked_env, second)

    assert rc == 0 and outcome["status"] == "superseded" and outcome["superseded_by_commit"] == third


def _side_commit_merged_after(door_clone, base: str) -> tuple[str, str, str]:
    """base - second - third - merge(side), with side branching from base."""
    upstream = door_clone["upstream"]
    third = _second_on_main(door_clone)
    _real_git("switch", "--quiet", "-c", "side", base, cwd=upstream)
    _real_git("commit", "--quiet", "--allow-empty", "-m", "side", cwd=upstream)
    side = _real_git("rev-parse", "HEAD", cwd=upstream)
    _real_git("switch", "--quiet", "main", cwd=upstream)
    _real_git("merge", "--quiet", "--no-ff", "-m", "merge side", "side", cwd=upstream)
    return third, side, _real_git("rev-parse", "main", cwd=upstream)


def test_a_request_the_deploy_leaves_out_hears_so_on_the_next_wake_up(parked_env, door_clone) -> None:
    """Side is on main but not in what runs, and third is ahead of what runs without it."""
    first, second = _first_commit(door_clone), str(door_clone["merged"])
    third, side, _ = _side_commit_merged_after(door_clone, first)
    _activate_for_script(parked_env, second, proven=True)
    _park_for_script(parked_env, PARKED_OLDER_ID, third, active_at_park=first)
    _park_for_script(parked_env, DEPLOY_ID, side, active_at_park=first)

    rc, outcome, calls = _run_parked(parked_env, side)

    assert f"--source-commit {third}" in _tool_call(calls)
    assert rc == 0 and outcome == {}
    assert _outcome_of(parked_env, PARKED_OLDER_ID)["status"] == "deployed"

    _activate_for_script(parked_env, third, proven=True)
    rc, outcome, _ = _run_parked(parked_env, side)

    assert outcome["status"] == "failed" and outcome["code"] == "parked_commit_not_contained"
    assert outcome["active_release"] == third
    assert rc == 0  # the unit did its job; the request's own outcome carries the refusal


def test_a_merged_branch_tip_goes_out_inside_the_merge_that_contains_it(parked_env, door_clone) -> None:
    first, second = _first_commit(door_clone), str(door_clone["merged"])
    _, side, merge = _side_commit_merged_after(door_clone, first)
    _activate_for_script(parked_env, second, proven=True)
    _park_for_script(parked_env, PARKED_OLDER_ID, merge, active_at_park=first)
    _park_for_script(parked_env, DEPLOY_ID, side, active_at_park=first)

    rc, outcome, calls = _run_parked(parked_env, side)

    assert rc == 0 and f"--source-commit {merge}" in _tool_call(calls)
    assert _outcome_of(parked_env, PARKED_OLDER_ID)["commit"] == merge
    assert _parked_names(parked_env) == [DEPLOY_ID]


def test_a_refused_staging_guard_keeps_every_request_parked(parked_env, door_clone) -> None:
    newest = str(door_clone["merged"])
    _activate_for_script(parked_env, _first_commit(door_clone), proven=True)
    _park_for_script(parked_env, DEPLOY_ID, newest)

    rc, outcome, calls = _run_parked({**parked_env, "FAKE_STAGING_REFUSED": "1"}, newest)

    assert rc == 0 and outcome == {}
    assert not any(call.startswith("venv-python") for call in calls)
    assert _parked_names(parked_env) == [DEPLOY_ID]


def test_a_record_stays_until_its_outcome_exists(parked_env, door_clone) -> None:
    first, newest = _first_commit(door_clone), str(door_clone["merged"])
    _activate_for_script(parked_env, first, proven=True)
    _park_for_script(parked_env, DEPLOY_ID, newest, active_at_park=first)
    (Path(parked_env["DOOR_RESULTS_DIR"]) / f"{DEPLOY_ID}.outcome.json").mkdir()

    done = subprocess.run(["/bin/bash", str(DOOR / "door-deploy.sh")], capture_output=True, text=True, check=False,
                          timeout=120, env={**parked_env, "DOOR_COMMIT": newest})

    assert done.returncode != 0
    assert _parked_names(parked_env) == [DEPLOY_ID]


def test_a_parked_commit_a_proven_release_already_runs_starts_no_tool(parked_env, door_clone) -> None:
    newest = str(door_clone["merged"])
    _activate_for_script(parked_env, newest, proven=True)
    _park_for_script(parked_env, DEPLOY_ID, newest)

    rc, outcome, calls = _run_parked(parked_env, newest)

    assert rc == 0 and outcome["status"] == "superseded"
    assert not any(call.startswith("venv-python") for call in calls)
    assert _parked_names(parked_env) == []


def test_a_rollback_parked_during_a_deploy_exits_as_a_refusal_the_client_sees(parked_env, door_clone) -> None:
    first, newest = _first_commit(door_clone), str(door_clone["merged"])
    _activate_for_script(parked_env, newest, proven=True)
    _park_for_script(parked_env, DEPLOY_ID, first, active_at_park=newest)

    rc, outcome, calls = _run_parked(parked_env, first)

    assert outcome["status"] == "failed" and outcome["code"] == "parked_rollback_refused"
    assert rc == 0  # the unit did its job; the request's own outcome carries the refusal
    assert not any(call.startswith("venv-python") for call in calls)


def test_another_deploys_lock_keeps_everything_parked_with_no_outcome(parked_env, door_clone) -> None:
    import fcntl

    newest = str(door_clone["merged"])
    _activate_for_script(parked_env, _first_commit(door_clone), proven=True)
    _park_for_script(parked_env, DEPLOY_ID, newest)
    lock = os.open(Path(parked_env["DOOR_STATE_ROOT"]) / "control-plane-deploy.lock", os.O_RDWR | os.O_CREAT)
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        rc, outcome, calls = _run_parked(parked_env, newest)
    finally:
        os.close(lock)

    assert rc == 0 and outcome == {}
    assert not any(call.startswith("venv-python") for call in calls)
    assert _parked_names(parked_env) == [DEPLOY_ID]


def test_a_failed_fetch_keeps_every_request_parked_and_writes_no_outcome(parked_env, door_clone) -> None:
    """Nothing about a request made the fetch fail, so an outage must not use up the queue."""
    newest = str(door_clone["merged"])
    _activate_for_script(parked_env, _first_commit(door_clone), proven=True)
    _park_for_script(parked_env, PARKED_OLDER_ID, newest)
    _park_for_script(parked_env, DEPLOY_ID, newest)
    parked_env["DOOR_UPSTREAM_URL"] = str(Path(parked_env["STUB_DIR"]) / "no-such-upstream")

    rc, outcome, calls = _run_parked(parked_env, newest)

    assert rc == 0 and outcome == {}
    assert not any(call.startswith("venv-python") for call in calls)
    assert _parked_names(parked_env) == sorted([PARKED_OLDER_ID, DEPLOY_ID])


def test_a_direct_deploy_whose_fetch_fails_still_refuses(parked_env, door_clone) -> None:
    newest = str(door_clone["merged"])
    del parked_env["DOOR_PARKED"], parked_env["DOOR_PARKED_DIR"]
    parked_env["DOOR_UPSTREAM_URL"] = str(Path(parked_env["STUB_DIR"]) / "no-such-upstream")

    rc, outcome, _ = _run_parked(parked_env, newest)

    assert rc == 2 and outcome["code"] == "fetch_failed"


def test_a_settle_that_cannot_decide_is_reported_and_starts_no_tool(parked_env, door_clone) -> None:
    newest = str(door_clone["merged"])
    _park_for_script(parked_env, DEPLOY_ID, newest)
    del parked_env["DOOR_ACTIVE_RELEASE_LINK"]

    rc, outcome, calls = _run_parked(parked_env, newest)

    assert rc == 2 and outcome["code"] == "parked_settle_failed"
    assert not any(call.startswith("venv-python") for call in calls)


def test_a_deploy_requested_directly_keeps_the_power_to_roll_back(parked_env, door_clone) -> None:
    first, newest = _first_commit(door_clone), str(door_clone["merged"])
    _activate_for_script(parked_env, newest, proven=True)
    del parked_env["DOOR_PARKED"], parked_env["DOOR_PARKED_DIR"]

    rc, outcome, calls = _run_parked(parked_env, first)

    assert rc == 0 and outcome["status"] == "deployed" and outcome["commit"] == first
    assert f"--source-commit {first}" in _tool_call(calls)


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
    end = next(index for index in range(start, len(lines)) if not lines[index].endswith("\\"))
    copy = " ".join(lines[start:end + 1])
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


@pytest.mark.parametrize("helper_rc", [0, 2])
def test_public_asset_shell_only_enqueues_as_service_user_with_literal_grant(env, tmp_path, helper_rc):
    repository = tmp_path / "repository"
    repository.mkdir()
    _write_stub(tmp_path / "stubs/setpriv", SETPRIV_STUB)
    _write_stub(tmp_path / "stubs/chgrp", "#!/bin/bash\nexit 0\n")
    helper = tmp_path / "preparation-python"
    _write_stub(helper, '''#!/bin/bash
printf '%s\\n' "$@" > "$STUB_DIR/preparation-argv"
printf '%s\\n' "$GIT_CONFIG_COUNT" "$GIT_CONFIG_KEY_0" "$GIT_CONFIG_VALUE_0" > "$STUB_DIR/preparation-git-config"
: > "$STUB_DIR/preparation-private"
printf '%s\\n' '{"status":"grant_queued","claim_ceiling":"development_only"}'
exit "$FAKE_HELPER_RC"
''')
    grant = json.dumps({"authorization_reference": "development_only $(touch must-not-exist) `false` %n", "request_digest": "sealed"})
    request_id = "20261029T154718Z-public-embodiment-preparation-0000abcd"
    arguments = dict(DOOR_REQUEST_ID=request_id, DOOR_VENV_PYTHON=str(helper),
        DOOR_CONTROL_PLANE_REPO=str(repository), DOOR_SERVICE_USER="blueprint",
        DOOR_EMBODIMENT_PREPARATION_REQUEST=grant, FAKE_HELPER_RC=str(helper_rc))
    rc, outcome, calls = _run("door-public-embodiment-preparation.sh", env, **arguments)
    assert rc == helper_rc
    assert outcome["status"] == ("completed" if helper_rc == 0 else "failed")
    assert (tmp_path / "preparation-argv").read_text().splitlines() == ["-m",
        "blueprint_pipeline.deployment_embodiment_preparation_controller", "enqueue", "--request-json", grant]
    assert (tmp_path / "preparation-git-config").read_text().splitlines() == ["1", "safe.directory", str(repository.resolve())]
    assert any("--reuid=blueprint --regid=blueprint" in call for call in calls)
    assert not any("systemctl" in call or "git " in call for call in calls)
    assert not (repository / "must-not-exist").exists()
    assert stat.S_IMODE((tmp_path / "preparation-private").stat().st_mode) == 0o600
    result = Path(env["DOOR_RESULTS_DIR"]) / f"{request_id}.public-embodiment-preparation.json"
    assert stat.S_IMODE(result.stat().st_mode) == 0o640
    rc, outcome, _ = _run("door-public-embodiment-preparation.sh", env, **{**arguments, "DOOR_SERVICE_USER": "root"})
    assert rc == 2 and outcome["code"] == "embodiment_preparation_user_invalid"


@pytest.mark.parametrize("helper_rc", [0, 2])
@pytest.mark.parametrize("operation", ["install-public-piper-policy", "install-public-yam-policy", "install-public-franka-policy", "install-public-trlc-policy", "install-scripted-robot-baselines", "install-native-g1-policies"])
def test_public_policy_shell_builds_unprivileged_then_uses_only_installed_stdlib_installer(env, tmp_path, helper_rc, operation):
    repository = tmp_path / "repository"
    repository.mkdir()
    _write_stub(tmp_path / "stubs/setpriv", SETPRIV_STUB)
    _write_stub(tmp_path / "stubs/chgrp", "#!/bin/bash\nexit 0\n")
    helper = tmp_path / "policy-preparation-python"
    _write_stub(helper, '''#!/bin/bash
printf '%s\\n' "$@" > "$STUB_DIR/public-policy-prepare-argv"
printf '%s\\n' '{"status":"prepared","claim_ceiling":"development_only"}'
exit "$FAKE_HELPER_RC"
''')
    _write_stub(tmp_path / "stubs/python3", '''#!/bin/bash
if [[ "$*" == *"/operator_door/public_policy_registration.py"* || "$*" == *"/operator_door/native_g1_registration.py"* ]]; then
  printf '%s\\n' "$@" > "$STUB_DIR/public-policy-root-argv"
  printf '%s\\n' '{"status":"installed","claim_ceiling":"development_only"}'
  exit 0
fi
exec "''' + sys.executable + '''" "$@"
''')
    request = json.dumps({"authorization_reference": "development_only $(touch must-not-exist) `false` %n"})
    arguments = dict(DOOR_REQUEST_ID="20261029T154718Z-" + operation + "-0000abcd",
        DOOR_VENV_PYTHON=str(helper), DOOR_CONTROL_PLANE_REPO=str(repository), DOOR_SERVICE_USER="blueprint",
        DOOR_PUBLIC_POLICY_REQUEST=request, DOOR_PUBLIC_POLICY_OPERATION=operation, FAKE_HELPER_RC=str(helper_rc))
    rc, outcome, calls = _run("door-install-public-piper-policy.sh", env, **arguments)
    assert rc == helper_rc
    assert outcome["status"] == ("completed" if helper_rc == 0 else "failed")
    assert (tmp_path / "public-policy-prepare-argv").read_text().splitlines() == [
        "-m", ("blueprint_pipeline.native_g1_registration_install" if operation == "install-native-g1-policies" else
               "blueprint_pipeline.scripted_robot_registration_install" if operation == "install-scripted-robot-baselines"
               else "blueprint_pipeline.openwam_piper_registration"), "--request-json", request]
    root_argv = tmp_path / "public-policy-root-argv"
    if helper_rc == 0:
        installer = "native_g1_registration.py" if operation == "install-native-g1-policies" else "public_policy_registration.py"
        assert root_argv.read_text().splitlines() == ["-I", "-S", str(DOOR / "operator_door" / installer),
                                                   "--request-json", request]
    else:
        assert not root_argv.exists()
    assert any("--reuid=blueprint --regid=blueprint" in call for call in calls)
    assert not any("systemctl" in call or "git " in call for call in calls)
    assert not (repository / "must-not-exist").exists()
    rc, outcome, _ = _run("door-install-public-piper-policy.sh", env, **{**arguments, "DOOR_SERVICE_USER": "root"})
    assert rc == 2 and outcome["code"] == "public_policy_service_user_invalid"


@pytest.mark.parametrize(("guard_rc", "helper_rc"), [(0, 0), (0, 1), (1, 0)])
def test_selected_handoff_receipts_readable_and_service_artifacts_private(env, tmp_path, guard_rc, helper_rc):
    request_id = "20261009T154718Z-selected-handoff-0000abcd"
    repository = tmp_path / "repository"
    repository.mkdir()
    _write_stub(tmp_path / "stubs/setpriv", SETPRIV_STUB)
    _write_stub(tmp_path / "stubs/chgrp", "#!/bin/bash\nexit 0\n")
    python = tmp_path / "selected-python"
    _write_stub(python, '''#!/bin/bash
if [[ "$*" == *production_runtime_env_guard* ]]; then
  : > "$STUB_DIR/guard-private"
  exit "$FAKE_GUARD_RC"
fi
: > "$STUB_DIR/helper-private"
printf '%s\\n' '{"status":"blocked"}'
exit "$FAKE_HELPER_RC"
''')
    rc, outcome, calls = _run("door-selected-handoff.sh", env, DOOR_REQUEST_ID=request_id,
        DOOR_VENV_PYTHON=str(python), DOOR_CONTROL_PLANE_REPO=str(repository), DOOR_SERVICE_USER="blueprint",
        DOOR_SELECTED_HANDOFF_REQUEST='{"mode":"dispatch"}', FAKE_GUARD_RC=str(guard_rc),
        FAKE_HELPER_RC=str(helper_rc))
    assert rc == (2 if guard_rc else helper_rc)
    assert outcome["status"] == ("refused" if guard_rc else "failed" if helper_rc else "completed")
    for suffix in ("log", "outcome.json"):
        assert stat.S_IMODE((Path(env["DOOR_RESULTS_DIR"]) / f"{request_id}.{suffix}").stat().st_mode) == 0o644
    assert stat.S_IMODE((tmp_path / "guard-private").stat().st_mode) == 0o600
    if not guard_rc:
        result = Path(env["DOOR_RESULTS_DIR"]) / f"{request_id}.selected-handoff.json"
        assert stat.S_IMODE(result.stat().st_mode) == 0o640
        assert stat.S_IMODE((tmp_path / "helper-private").stat().st_mode) == 0o600
    else:
        assert not (tmp_path / "helper-private").exists()
    assert all(line == "umask 0077" for line in calls if line.startswith("umask "))
@pytest.mark.parametrize("mode", ["atlas-status", "atlas-retry", "discover-selector"])
def test_selected_atlas_shell_runs_only_service_helper_without_runtime_reconciliation(env, tmp_path, mode):
    repository = tmp_path / "repository"
    repository.mkdir()
    _write_stub(tmp_path / "stubs/setpriv", SETPRIV_STUB)
    _write_stub(tmp_path / "stubs/chgrp", "#!/bin/bash\nexit 0\n")
    python = tmp_path / "selected-python"
    _write_stub(python, '''#!/bin/bash
printf '%s\\n' "$*" >> "$STUB_LOG"
if [[ "$*" == *production_runtime_env_guard* ]]; then exit 19; fi
printf '%s\\n' '{"status":"atlas_status_returned"}'
''')
    arguments = dict(DOOR_REQUEST_ID="20261009T154718Z-selected-handoff-0000abcd",
        DOOR_VENV_PYTHON=str(python), DOOR_CONTROL_PLANE_REPO=str(repository), DOOR_SERVICE_USER="blueprint",
        DOOR_SELECTED_HANDOFF_REQUEST=json.dumps({"mode": mode}))
    rc, outcome, calls = _run("door-selected-handoff.sh", env, **arguments)
    assert rc == 0 and outcome["status"] == "completed"
    assert any("--reuid=blueprint --regid=blueprint" in call for call in calls)
    assert any("blueprint_pipeline.selected_handoff_recovery" in call for call in calls)
    assert not any("production_runtime_env_guard" in call for call in calls)
    rc, outcome, _ = _run("door-selected-handoff.sh", env, **{**arguments, "DOOR_SERVICE_USER": "root"})
    assert rc == 2 and outcome["code"] == "selected_handoff_user_invalid"


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


# --- extend-scene-execution-window ---------------------------------------------------------------

WINDOW_ID = "20261031T120000Z-extend-scene-execution-window-0000abcd"


@pytest.fixture()
def window_env(env: dict[str, str], tmp_path: Path) -> dict[str, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _write_stub(tmp_path / "stubs" / "setpriv", SETPRIV_STUB)
    _write_stub(tmp_path / "window-python", '''#!/bin/bash
{ printf 'window %s\\n' "$*"; echo "intake-root ${BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT:-unset}"; } >> "$STUB_LOG"
printf '%s\\n' "${FAKE_WINDOW:-}"
exit "${FAKE_TOOL_RC:-0}"
''')
    return {**env, "DOOR_VENV_PYTHON": str(tmp_path / "window-python"), "DOOR_CONTROL_PLANE_REPO": str(repository),
            "DOOR_SERVICE_USER": "blueprint", "DOOR_REQUESTED_BY": "cloud-scene-runs",
            "DOOR_SCENE_WINDOW_REQUEST": json.dumps({"kind": "extend-scene-execution-window",
                                                     "authorization_reference": "approval $(touch pwned)"})}


@pytest.mark.parametrize(("report", "rc", "status", "code"), [
    ({"door_outcome": "planned"}, 0, "planned", None),
    ({"door_outcome": "extended"}, 0, "extended", None),
    ({"door_outcome": "already_covered"}, 0, "already_covered", None),
    ({"door_outcome": "webapp_notice_failed"}, 1, "failed", "webapp_notice_failed"),
    ({"door_outcome": "refused", "code": "scene_execution_window_intent_revoked"}, 2, "refused",
     "scene_execution_window_intent_revoked"),
])
def test_scene_window_runs_the_release_module_as_the_service_user(window_env, tmp_path, report, rc, status, code):
    done, outcome, calls = _run("door-extend-scene-execution-window.sh", window_env, DOOR_REQUEST_ID=WINDOW_ID,
                                FAKE_WINDOW=json.dumps(report), FAKE_TOOL_RC=str(rc))
    assert (done, outcome["status"], outcome["code"], outcome["exit_code"]) == (rc, status, code, rc)
    assert calls[0] == "setpriv --reuid=blueprint --regid=blueprint --init-groups --inh-caps=-all --"
    assert calls[1] == "umask 0077"
    assert calls[2].startswith("window -m blueprint_pipeline.task_evaluation_scene_execution_window_door --request-json ")
    assert calls[2].endswith(f"--door-caller cloud-scene-runs --door-request-id {WINDOW_ID}")
    assert not (tmp_path / "pwned").exists()
    result = Path(window_env["DOOR_RESULTS_DIR"]) / f"{WINDOW_ID}.scene-window.json"
    assert json.loads(result.read_text()) == report and outcome["result"] == str(result)


@pytest.mark.parametrize(("overrides", "code"), [
    ({"DOOR_SERVICE_USER": "root"}, "scene_window_user_invalid"),
    ({"DOOR_REQUESTED_BY": "../root"}, "scene_window_caller_invalid"),
    ({"DOOR_REQUESTED_BY": "a b"}, "scene_window_caller_invalid"),
])
def test_scene_window_rechecks_its_inputs(window_env, overrides, code):
    rc, outcome, calls = _run("door-extend-scene-execution-window.sh", window_env, DOOR_REQUEST_ID=WINDOW_ID,
                              **overrides)
    assert rc == 2 and outcome["code"] == code and not calls


def test_scene_window_script_is_installed_with_the_door():
    installer = (DOOR / "install.sh").read_text(encoding="utf-8")
    assert '"$source_dir"/door-extend-scene-execution-window.sh' in installer


# ------------------------------------------------- scene-configuration activation hold

LIVE_RELEASE = "c" * 40


def _live_activation(values: dict[str, str], commit: str = LIVE_RELEASE) -> None:
    """A pending scene-configuration activation bound to ``commit``, and that release running."""
    root = Path(values["STUB_DIR"])
    pending = Path(values["DOOR_STATE_ROOT"]) / "task-evaluation-launch-activations" / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    (pending / "act-1-abc.json").write_text(json.dumps({"request": {
        "activation_id": "act-1", "lane": "task_evaluation_scene_configuration",
        "expected_production_commit": commit}}), encoding="utf-8")
    link = Path(values.get("DOOR_ACTIVE_RELEASE_LINK") or root / "active")
    if not link.is_symlink():
        link.symlink_to(root / "releases" / commit)


def _hold_env(env: dict[str, str]) -> dict[str, str]:
    root = Path(env["STUB_DIR"])
    (root / "parked").mkdir(exist_ok=True)
    return {**env, "DOOR_ACTIVATION_HOLD_PARKED_DIR": str(root / "parked"),
            "DOOR_ACTIVE_RELEASE_LINK": str(root / "active")}


def test_a_direct_deploy_during_a_live_activation_parks_itself_before_fetching(env: dict[str, str]) -> None:
    values = _hold_env(env)
    _live_activation(values)

    rc, outcome, calls = _run("door-deploy.sh", values, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_WAIT_FOR_IDLE="1")

    assert rc == 0 and outcome == {}  # the deploy that settles it writes its outcome
    assert not any(call.startswith(("venv-python", "git")) for call in calls)
    record = json.loads((Path(values["DOOR_ACTIVATION_HOLD_PARKED_DIR"]) / f"{DEPLOY_ID}.json").read_text(encoding="utf-8"))
    assert record["commit"] == SHA and record["wait_for_idle"] is True
    assert record["active_release_at_park"] == LIVE_RELEASE and "deploying_commit" not in record
    log = (Path(values["DOOR_RESULTS_DIR"]) / f"{DEPLOY_ID}.log").read_text(encoding="utf-8")
    assert "staying parked: scene_configuration_activation_live:pending:act-1" in log
    hold_file = json.loads((Path(values["DOOR_ACTIVATION_HOLD_PARKED_DIR"]) / ".activation-hold").read_text(encoding="utf-8"))
    assert hold_file["live"] == [{"activation_id": "act-1", "state": "pending"}]


def test_an_override_deploys_during_a_live_activation(env: dict[str, str]) -> None:
    values = _hold_env(env)
    _live_activation(values)

    rc, outcome, calls = _run("door-deploy.sh", values, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_WAIT_FOR_IDLE="0",
                              DOOR_ACTIVATION_HOLD_OVERRIDE="money")

    assert rc == 0 and outcome["status"] == "deployed"
    assert any(call.startswith("venv-python") for call in calls)
    assert not list(Path(values["DOOR_ACTIVATION_HOLD_PARKED_DIR"]).glob("*.json"))


def test_an_activation_bound_to_another_release_does_not_hold_the_deploy(env: dict[str, str]) -> None:
    values = _hold_env(env)
    _live_activation(values, commit="d" * 40)
    (Path(values["STUB_DIR"]) / "active").unlink()
    (Path(values["STUB_DIR"]) / "active").symlink_to(Path(values["STUB_DIR"]) / "releases" / LIVE_RELEASE)

    rc, outcome, _ = _run("door-deploy.sh", values, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_WAIT_FOR_IDLE="0")

    assert rc == 0 and outcome["status"] == "deployed"


def test_a_canonical_deploy_during_a_live_activation_is_refused_not_parked(env: dict[str, str]) -> None:
    values = _hold_env(env)
    _live_activation(values)
    provenance = Path(values["DOOR_RESULTS_DIR"]) / f"{DEPLOY_ID}.release-provenance.json"
    provenance.write_bytes(b'{"schema": "fixture"}')
    provenance.chmod(0o600)

    rc, outcome, calls = _run("door-deploy.sh", values, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_WAIT_FOR_IDLE="0",
                              DOOR_RELEASE_PROVENANCE_FILE=str(provenance),
                              DOOR_RELEASE_PROVENANCE_SHA256=hashlib.sha256(provenance.read_bytes()).hexdigest())

    assert rc == 2 and outcome["status"] == "refused" and outcome["code"] == "scene_configuration_activation_live"
    assert outcome["activation_hold"] == "scene_configuration_activation_live:pending:act-1"
    # When to try again (the next recheck), and when the cap lets the same release's deploy go.
    now = time.time()
    retry, cap = int(outcome["retry_after_epoch"]), int(outcome["deploys_by_epoch"])
    assert now < retry <= now + 5 * 60 + 5 and now + 284 * 60 < cap <= now + 285 * 60 + 5
    assert outcome["retry_after"].endswith("+00:00") and outcome["deploys_by"].endswith("+00:00")
    assert not any(call.startswith("venv-python") for call in calls)
    assert not list(Path(values["DOOR_ACTIVATION_HOLD_PARKED_DIR"]).glob("*.json"))


def test_a_parked_deploy_stays_parked_unmarked_while_an_activation_is_live(parked_env, door_clone) -> None:
    newest = str(door_clone["merged"])
    first = _first_commit(door_clone)
    _activate_for_script(parked_env, first, proven=True)
    _park_for_script(parked_env, DEPLOY_ID, newest, active_at_park=first)
    _live_activation(parked_env, commit=first)

    rc, outcome, calls = _run_parked(parked_env, newest)

    assert rc == 0 and outcome == {}
    assert not any(call.startswith("venv-python") for call in calls)
    assert _parked_names(parked_env) == [DEPLOY_ID]
    record = json.loads((Path(parked_env["DOOR_PARKED_DIR"]) / f"{DEPLOY_ID}.json").read_text(encoding="utf-8"))
    assert "deploying_commit" not in record  # so the next wake-up does not call it interrupted


def test_pending_work_behind_a_held_activation_path_unit_does_not_hold_the_deploy(env: dict[str, str]) -> None:
    values = _hold_env(env)
    _live_activation(values)
    holds = Path(values["STUB_DIR"]) / "holds"  # beside parked/, as in the door's spool
    holds.mkdir()
    (holds / "blueprint-task-evaluation-launch-activation.path.json").write_text(
        json.dumps({"status": "active"}), encoding="utf-8")

    rc, outcome, _ = _run("door-deploy.sh", values, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_WAIT_FOR_IDLE="0")

    assert rc == 0 and outcome["status"] == "deployed"


def test_a_free_form_override_reason_deploys_and_is_logged(env: dict[str, str]) -> None:
    values = _hold_env(env)
    _live_activation(values)

    rc, outcome, _ = _run("door-deploy.sh", values, DOOR_REQUEST_ID=DEPLOY_ID, DOOR_WAIT_FOR_IDLE="0",
                          DOOR_ACTIVATION_HOLD_OVERRIDE="fixes_live_activation")

    assert rc == 0 and outcome["status"] == "deployed"
    log = (Path(values["DOOR_RESULTS_DIR"]) / f"{DEPLOY_ID}.log").read_text(encoding="utf-8")
    assert "activation_hold_overridden:fixes_live_activation" in log


# --- extend-scene-execution-budget ---------------------------------------------------------------

BUDGET_ID = "20261101T150000Z-extend-scene-execution-budget-0000abcd"


@pytest.fixture()
def budget_env(env: dict[str, str], tmp_path: Path) -> dict[str, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _write_stub(tmp_path / "stubs" / "setpriv", SETPRIV_STUB)
    _write_stub(tmp_path / "budget-python", '''#!/bin/bash
{ printf 'budget %s\\n' "$*"; echo "intake-root ${BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT:-unset}"; } >> "$STUB_LOG"
printf '%s\\n' "${FAKE_BUDGET:-}"
exit "${FAKE_TOOL_RC:-0}"
''')
    return {**env, "DOOR_VENV_PYTHON": str(tmp_path / "budget-python"), "DOOR_CONTROL_PLANE_REPO": str(repository),
            "DOOR_SERVICE_USER": "blueprint", "DOOR_REQUESTED_BY": "cloud-scene-runs",
            "DOOR_SCENE_BUDGET_REQUEST": json.dumps({"kind": "extend-scene-execution-budget",
                                                     "approval_evidence": "approval $(touch pwned)"})}


@pytest.mark.parametrize(("report", "rc", "status", "code"), [
    ({"door_outcome": "planned"}, 0, "planned", None),
    ({"door_outcome": "extended"}, 0, "extended", None),
    ({"door_outcome": "refused", "code": "budget_apply_ack_mismatch"}, 2, "refused", "budget_apply_ack_mismatch"),
    ({"door_outcome": "already_covered"}, 0, "failed", "scene_budget_failed"),
])
def test_scene_budget_runs_the_release_module_as_the_service_user(budget_env, tmp_path, report, rc, status, code):
    done, outcome, calls = _run("door-extend-scene-execution-budget.sh", budget_env, DOOR_REQUEST_ID=BUDGET_ID,
                                FAKE_BUDGET=json.dumps(report), FAKE_TOOL_RC=str(rc))
    expected_rc = 1 if status == "failed" else rc
    assert (done, outcome["status"], outcome["code"], outcome["exit_code"]) == (expected_rc, status, code, expected_rc)
    assert calls[0] == "setpriv --reuid=blueprint --regid=blueprint --init-groups --inh-caps=-all --"
    assert calls[2].startswith("budget -m blueprint_pipeline.task_evaluation_scene_execution_budget_door --request-json ")
    assert calls[2].endswith(f"--door-caller cloud-scene-runs --door-request-id {BUDGET_ID}")
    assert not (tmp_path / "pwned").exists()
    result = Path(budget_env["DOOR_RESULTS_DIR"]) / f"{BUDGET_ID}.scene-budget.json"
    assert json.loads(result.read_text()) == report and outcome["result"] == str(result)


@pytest.mark.parametrize(("overrides", "code"), [
    ({"DOOR_SERVICE_USER": "root"}, "scene_budget_user_invalid"),
    ({"DOOR_REQUESTED_BY": "../root"}, "scene_budget_caller_invalid"),
])
def test_scene_budget_rechecks_its_inputs(budget_env, overrides, code):
    rc, outcome, calls = _run("door-extend-scene-execution-budget.sh", budget_env, DOOR_REQUEST_ID=BUDGET_ID,
                              **overrides)
    assert rc == 2 and outcome["code"] == code and not calls


def test_scene_budget_script_is_installed_with_the_door():
    installer = (DOOR / "install.sh").read_text(encoding="utf-8")
    assert '"$source_dir"/door-extend-scene-execution-budget.sh' in installer
