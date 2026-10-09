"""ADP rollout gate adds no unrelated operate controls or credential bootstrap."""

from __future__ import annotations

import json
import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOOR = ROOT / "deploy" / "operator-door"
BASELINE = "d78ee479368c2df99370a9dc4a61dd328d07d215"


def _profile_package(tmp_path):
    package = tmp_path / "operator_door"
    shutil.copytree(DOOR / "operator_door", package)
    (package / "admitted_controls.py").write_text("DISPATCHER_HOLD_ONLY = True\n")
    return tmp_path


def test_profile_fences_api_schema_and_privileged_spool_reader(tmp_path):
    package_root = _profile_package(tmp_path)
    script = '''
from pathlib import Path
from operator_door import requests as r
from operator_door import status
assert set(r._SCOPES) == {"deploy", "unit", "door-upgrade", "hold", "release-hold", "selected-handoff"}
assert r.required_scope("selected-handoff") == "operate"
# No arbitrary new operation: incomplete selected bodies remain refused.
try: r.validate_request({"kind":"selected-handoff", "mode":"dispatch"})
except r.RequestRefused as e: assert e.code == "selected_handoff_fields_invalid"
else: raise AssertionError("unbound selected dispatch admitted")
assert status.DISPATCHER_HOLD_ONLY is True
assert status._SCOPES == r._SCOPES
assert r.required_scope("door-upgrade") == "deploy"
body = {"kind":"hold", "unit":"blueprint-agent-run-dispatcher.timer", "owner":"rollout", "reason":"fixture", "expires_in_seconds":3600}
assert r.validate_request(body) == body
assert r.required_scope("hold") == "operate"
assert r.validate_request({"kind":"release-hold", "unit":body["unit"]})
# The exact dispatcher gate changes; unrelated installed trigger controls do not.
for unit in ("blueprint-pubsub-handoff-listener.timer", "blueprint-control-plane-capacity.timer", "blueprint-control-plane-storage-gc.timer", "blueprint-task-evaluation-preflight.timer", "blueprint-task-evaluation-terminal-resource-release.path"):
    for action in ("stop", "restart"):
        assert r.validate_request({"kind":"unit","unit":unit,"action":action})
for action in ("stop", "restart"):
    try: r.validate_request({"kind":"unit","unit":"blueprint-gpu-spend-guard.timer","action":action})
    except r.RequestRefused as e: assert e.code == "unit_safety_critical"
    else: raise AssertionError("baseline safety restriction changed")
for kind in ("retire-scene-workspace", "restore-scene-workspace", "retire-scene", "restore-scene", "lane-scratch", "owner-census-decision", "legacy-owner-census", "provider-output-resume"):
    for call in (lambda: r.required_scope(kind), lambda: r.validate_request({"kind":kind})):
        try: call()
        except r.RequestRefused as e: assert e.code == "kind_unknown"
        else: raise AssertionError("unrelated operation admitted")
    try: r.validate_request_id("20260930T000000Z-"+kind+"-0123abcd")
    except r.RequestRefused: pass
    else: raise AssertionError("forged privileged spool id admitted")
for unit, code in (("blueprint-pubsub-handoff-listener.timer", "hold_unit_profile_refused"), ("blueprint-gpu-spend-guard.timer", "unit_safety_critical")):
    try: r.validate_request({**body,"unit":unit})
    except r.RequestRefused as e: assert e.code == code
    else: raise AssertionError("other hold target admitted")
try: r.validate_request({"kind":"unit","unit":body["unit"],"action":"stop"})
except r.RequestRefused as e: assert e.code == "unit_stop_requires_hold"
else: raise AssertionError("bare stop admitted")
'''
    completed = subprocess.run([sys.executable, "-c", script], cwd=package_root,
                               env={**os.environ, "PYTHONPATH": str(package_root),
                                    "PYTHONDONTWRITEBYTECODE": "1"},
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr


def _installer_preflight(tmp_path, *, marker=BASELINE, fenced=False, missing=False):
    install_root = tmp_path / "door"
    install_root.mkdir()
    (install_root / "INSTALLED_COMMIT").write_text(marker + "\n")
    if fenced:
        (install_root / "operator_door").mkdir()
        (install_root / "operator_door" / "admitted_controls.py").write_text(
            "DISPATCHER_HOLD_ONLY = True\n")
    config = tmp_path / "config"
    (config / "deploy-key").mkdir(parents=True)
    for name in ("tokens.json", "deploy-key/github", "deploy-key/known_hosts"):
        path = config / name
        path.write_text(json.dumps({"development_only": True}))
        path.chmod(0o600)
    if missing:
        (config / "deploy-key/github").unlink()
    before = {path:(path.read_bytes(),path.stat().st_mode) for path in config.rglob("*") if path.is_file()}
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    for name in ("id", "getent"):
        path = stub_dir / name
        path.write_text("#!/bin/sh\n[ \"$1\" = -u ] && echo 0\nexit 0\n")
        path.chmod(0o700)
    # Run the installer's real >=3.10 check with the actual test interpreter;
    # macOS's /usr/bin/python3 may be 3.9 even when this suite uses Python 3.12.
    interpreter = stub_dir / 'python3'
    interpreter.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
    interpreter.chmod(0o700)
    # Execute only the prerequisite section, before staging or any host write.
    prefix = (DOOR / "install.sh").read_text().split("# 1. Stage and check", 1)[0]
    prefix = prefix.replace('source_dir="$(cd "$(dirname "$0")" && pwd)"',
                            'source_dir=' + str(DOOR))
    prefix += '\nprintf "%s %s\\n" "$dispatcher_hold_only" "$caddy"\n'
    script = tmp_path / "preflight.sh"
    script.write_text(prefix)
    result = subprocess.run(["bash", str(script), "--upgrade"],
                            env={**os.environ, "PATH": str(stub_dir)+os.pathsep+os.environ["PATH"],
                                 "DOOR_INSTALL_ROOT": str(install_root), "DOOR_CONFIG_DIR": str(config)},
                            capture_output=True, text=True, timeout=20)
    assert {path:(path.read_bytes(),path.stat().st_mode) for path in before} == before
    return result


@pytest.mark.parametrize("fenced", [False, True])
def test_original_protocol_selects_or_preserves_fence_without_credential_changes(tmp_path, fenced):
    result = _installer_preflight(tmp_path, marker="a"*40 if fenced else BASELINE, fenced=fenced)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1 0"


def test_missing_existing_credential_refuses_before_bootstrap_or_staging(tmp_path):
    result = _installer_preflight(tmp_path, missing=True)
    assert result.returncode != 0
    assert "hold upgrade credential prerequisite missing" in result.stderr
    assert not (tmp_path / "door.new").exists()


def test_other_existing_installations_keep_their_default_profile(tmp_path):
    result = _installer_preflight(tmp_path, marker="a"*40)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0 1"
