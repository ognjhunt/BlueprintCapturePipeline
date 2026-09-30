"""Rehearse restricted upgrades locally with synthetic credentials and fake host tools."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

DOOR = Path(__file__).resolve().parents[1] / "deploy" / "operator-door"
BASELINE = "d78ee479368c2df99370a9dc4a61dd328d07d215"


def _stub(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/python3\n" + body)
    path.chmod(0o755)


def _credentials(paths: list[Path]) -> dict:
    return {
        path.name: (path.read_bytes(), path.stat().st_mode, path.stat().st_uid,
                    path.stat().st_gid, path.stat().st_ino)
        for path in paths
    }


@pytest.mark.parametrize("case", ["success", "self-test-failure", "swap-failure"])
@pytest.mark.parametrize("existing_fence", [False, True])
def test_restricted_installer_preserves_credentials_and_recovers_old_door(
    tmp_path: Path, case: str, existing_fence: bool,
) -> None:
    """Exercise the full installer; every host/network command is a local stub."""
    door = tmp_path / "door"
    door.mkdir()
    original_commit = "a" * 40 if existing_fence else BASELINE
    (door / "INSTALLED_COMMIT").write_text(original_commit + "\n")
    if existing_fence:
        (door / "operator_door").mkdir()
        (door / "operator_door" / "admitted_controls.py").write_text(
            "DISPATCHER_HOLD_ONLY = True\n")
    config = tmp_path / "config"
    (config / "deploy-key").mkdir(parents=True)
    credentials = [config / name for name in
                   ("tokens.json", "deploy-key/github", "deploy-key/known_hosts")]
    for index, path in enumerate(credentials):
        path.write_bytes(b"development-only synthetic credential\n")
        path.chmod((0o600, 0o600, 0o644)[index])
    before = _credentials(credentials)
    units = tmp_path / "systemd"
    units.mkdir()
    for name in ("blueprint-operator-door.service", "blueprint-operator-door-runner.service",
                 "blueprint-operator-door-runner.path"):
        (units / name).write_text("development-only old unit fixture\n")
    original_units = {path.name: path.read_bytes() for path in units.iterdir()}
    caddy = tmp_path / "Caddyfile"
    caddy.write_text("development-only Caddy fixture\n")
    binary = tmp_path / "bin"
    binary.mkdir()
    _stub(binary / "id", "import sys\nprint('0') if sys.argv[1:] == ['-u'] else None\n")
    _stub(binary / "getent", "pass\n")
    calls = tmp_path / "calls"
    log = ("import sys\nfrom pathlib import Path\np=Path(" + repr(str(calls)) + ")\n"
           "with p.open('a') as f: f.write(' '.join(sys.argv)+'\\n')\n")
    _stub(binary / "chown", log)
    _stub(binary / "systemctl", log)
    # Any GitHub/meta or Caddy call fails; only the local listener health probe is permitted.
    _stub(binary / "curl", "import sys\nassert sys.argv[-1].startswith('http://127.0.0.1:8767/')\n")
    _stub(binary / "runuser", "import sys\nsys.exit(" +
          ("1" if case == "self-test-failure" else "0") + ")\n")
    _stub(binary / "git", "print('b' * 40)\n")
    _stub(binary / "install", """import pathlib, shutil, sys
a = sys.argv[1:]
directory = '-d' in a
paths = []
i = 0
while i < len(a):
    if a[i] in ('-o', '-g', '-m'):
        i += 2
    elif a[i] == '-d':
        i += 1
    else:
        paths.append(a[i])
        i += 1
if directory:
    for path in paths:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
else:
    shutil.copyfile(paths[0], paths[1])
""")
    if case == "swap-failure":
        _stub(binary / "mv", "import subprocess, sys\na=sys.argv[1:]\n"
              "if a == " + repr([str(door) + ".new", str(door)]) + ": sys.exit(1)\n"
              "sys.exit(subprocess.run(['/bin/mv', *a]).returncode)\n")
    # Do not inherit credentials, provider settings, or remote endpoints from the session.
    environment = {
        "PATH": str(binary) + os.pathsep + "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "DOOR_INSTALL_ROOT": str(door), "DOOR_STATE_ROOT_DIR": str(tmp_path / "state"),
        "DOOR_CONFIG_DIR": str(config), "DOOR_SYSTEMD_DIR": str(units),
        "DOOR_CADDYFILE": str(caddy),
    }
    result = subprocess.run(["/bin/bash", str(DOOR / "install.sh"), "--upgrade"],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert _credentials(credentials) == before
    assert caddy.read_text() == "development-only Caddy fixture\n"
    # Stubbing ownership tools must not conceal any attempted credential ownership change.
    assert all(str(path) not in calls.read_text() for path in credentials)
    assert door.is_dir(), result.stderr
    if case == "success":
        assert result.returncode == 0, result.stderr
        assert "DISPATCHER_HOLD_ONLY = True" in (
            door / "operator_door" / "admitted_controls.py").read_text()
        assert (door / "INSTALLED_COMMIT").read_text() == "b" * 40 + "\n"
    else:
        assert result.returncode != 0
        assert (door / "INSTALLED_COMMIT").read_text() == original_commit + "\n"
        assert all((units / name).read_bytes() == content
                   for name, content in original_units.items())
        assert not (units / "blueprint-operator-door-hold-sweep.service").exists()
        assert not (units / "blueprint-operator-door-hold-sweep.timer").exists()
        if existing_fence:
            assert (door / "operator_door" / "admitted_controls.py").read_text() == (
                "DISPATCHER_HOLD_ONLY = True\n")
