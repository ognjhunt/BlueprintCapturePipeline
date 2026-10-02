"""A fixed notifier migration keeps unrelated unit bytes and refuses drift."""

# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/notifier_repair.py
#   deploy/operator-door/operator_door/requests.py
#   deploy/operator-door/operator_door/spool_runner.py

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy/operator-door"))
from operator_door import notifier_repair as repair  # noqa: E402
from operator_door.hostinfo import CommandResult  # noqa: E402
from operator_door.requests import RequestRefused, validate_request  # noqa: E402

COMMIT = "a" * 40
BEFORE_COMMAND = "{ path=/bin/bash ; argv[]=legacy-postcheck ; }"
EXPECTED = "sha256:" + hashlib.sha256(BEFORE_COMMAND.encode()).hexdigest()


class Runner:
    def __init__(self, fail_reload: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.fail_reload = fail_reload
        self.reloaded = False

    def run(self, argv: list[str], timeout: float) -> CommandResult:
        self.calls.append(list(argv))
        if argv[1] == "show":
            value = repair.NEW_INVOCATION if self.reloaded else BEFORE_COMMAND
            return CommandResult(0, f"ExecStartPost={value}\n", "")
        if argv[1] == "is-enabled":
            return CommandResult(1, "disabled\n", "")
        if argv[1] == "is-active":
            return CommandResult(3, "inactive\n", "")
        if argv[1] == "daemon-reload":
            self.reloaded = not self.fail_reload
            return CommandResult(1 if self.fail_reload else 0, "", "private stderr")
        raise AssertionError(argv)


@pytest.fixture()
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, bytes]:
    directory = tmp_path.resolve() / "blueprint-pipeline-control-plane.service.d"
    directory.mkdir()
    dropin = directory / "95-blueprint-active-release.conf"
    original = (b"[Service]\nEnvironment=AUTH_TOKEN=synthetic-private\n"
                b"ExecStartPost=\nExecStartPost=/usr/bin/logger unrelated-hook\n"
                b"ExecStartPost=/bin/bash -lc 'printf retained; exec /bin/bash deploy/systemd/blueprint-control-plane-postchecks.sh'\n"
                b"CPUQuota=200%\n")
    dropin.write_bytes(original)
    release = tmp_path.resolve() / "releases" / COMMIT
    source = release / "deploy/systemd/blueprint-control-plane-postchecks.sh"
    source.parent.mkdir(parents=True)
    source.write_text('env "PYTHONPATH=${PWD}/src" "${BLUEPRINT_PIPELINE_PYTHON}"\n')
    active = tmp_path.resolve() / "active"
    active.symlink_to(release)
    monkeypatch.setattr(repair, "DROPIN", dropin)
    monkeypatch.setattr(repair, "ACTIVE", active)
    monkeypatch.setattr(repair, "RELEASES", release.parent)
    monkeypatch.setattr(repair, "ROOT_UID", os.getuid())
    monkeypatch.setattr(repair, "ROOT_GID", os.getgid())
    monkeypatch.setattr(repair.holds, "read", lambda *_: {"status": "active", "require_explicit_release": True})
    return dropin, original


def test_repair_changes_only_matching_invocation_and_retains_private_backup(installed: tuple[Path, bytes]) -> None:
    path, original = installed
    runner = Runner()
    receipt = repair.repair_binding(EXPECTED, COMMIT, runner=runner)
    changed = path.read_bytes()
    assert b"Environment=AUTH_TOKEN=synthetic-private\n" in changed
    assert b"ExecStartPost=/usr/bin/logger unrelated-hook\n" in changed
    assert b"printf retained; exec env BLUEPRINT_PIPELINE_REPO=/opt/blueprint/task-evaluation-control-plane" in changed
    assert b"/bin/bash /opt/blueprint/task-evaluation-control-plane/deploy/systemd/blueprint-control-plane-postchecks.sh" in changed
    assert changed.splitlines()[:4] == original.splitlines()[:4]
    assert changed.splitlines()[-1] == original.splitlines()[-1]
    backup = Path(receipt["backup_path"])
    assert backup.read_bytes() == original and backup.stat().st_mode & 0o777 == 0o600
    assert receipt["restart_performed"] is False and receipt["slack_send_performed"] is False
    assert "synthetic-private" not in json.dumps(receipt)
    assert [call[1] for call in runner.calls].count("daemon-reload") == 1
    assert not any(call[1] in {"start", "restart", "enable"} for call in runner.calls)


@pytest.mark.parametrize("failure", ["metadata", "source", "shape", "symlink", "hold"])
def test_preflight_refusals_preserve_original_bytes(installed: tuple[Path, bytes], monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    path, original = installed
    digest, commit = EXPECTED, COMMIT
    if failure == "metadata":
        digest = "sha256:" + "0" * 64
    elif failure == "source":
        commit = "b" * 40
    elif failure == "shape":
        original = original.replace(b"postchecks.sh'", b"postchecks.sh; other-command'")
        path.write_bytes(original)
    elif failure == "symlink":
        target = path.with_suffix(".retained")
        path.rename(target)
        path.symlink_to(target)
    elif failure == "hold":
        monkeypatch.setattr(repair.holds, "read", lambda *_: None)
    with pytest.raises(repair.RepairRefused):
        repair.repair_binding(digest, commit, runner=Runner())
    assert path.read_bytes() == original
    assert not list(path.parent.glob(".blueprint-postcheck-backup.*"))


def test_reload_failure_restores_original_and_retains_failed_replacement(installed: tuple[Path, bytes]) -> None:
    path, original = installed
    with pytest.raises(repair.RepairRefused, match="reload_failed_original_restored"):
        repair.repair_binding(EXPECTED, COMMIT, runner=Runner(fail_reload=True))
    assert path.read_bytes() == original
    backup = next(path.parent.glob(".blueprint-postcheck-backup.*"))
    assert (backup / "original.conf").read_bytes() == original
    assert (backup / "failed-replacement.conf").read_bytes() != original


def test_request_is_fixed_target_and_no_arbitrary_setter() -> None:
    request = {"kind": "unit", "unit": "blueprint-pipeline-control-plane.service", "action": "repair-notifier-binding",
               "expected_postcheck_sha256": EXPECTED, "expected_source_commit": COMMIT}
    assert validate_request(request) == request
    for changed in ({"unit": "blueprint-other.service"}, {"expected_source_commit": "main"},
                    {"expected_postcheck_sha256": "x"}, {"path": "/etc/other"}, {"command": "arbitrary"}):
        with pytest.raises(RequestRefused):
            validate_request({**request, **changed})
