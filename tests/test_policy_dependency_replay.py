from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def replay_script():
    path = Path(__file__).parents[1] / "scripts/replay_openpi_policy_dependencies.py"
    spec = importlib.util.spec_from_file_location("policy_dependency_replay", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_replay_consumes_the_actual_production_sync(replay_script, tmp_path):
    args = replay_script.production_sync_arguments(
        source=tmp_path / "source", venv=tmp_path / "policy", uv=tmp_path / "uv",
    )
    assert args[:2] == [str(tmp_path / "uv"), "sync"]
    assert all(flag in args for flag in ("--active", "--no-dev", "--frozen"))
    assert args[args.index("--python") + 1] == str(tmp_path / "policy/bin/python")
    assert args[args.index("--no-build-package") + 1] == "mujoco"
    assert not any("/opt/" in word for word in args)


def test_replay_retains_all_pinned_image_interpreter_selectors(replay_script, tmp_path):
    settings = replay_script.image_interpreter_environment(tmp_path / "sdk")
    assert settings["VIRTUAL_ENV"] == settings["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / "sdk")
    assert settings["UV_PYTHON"] == str(tmp_path / "sdk/bin/python")
    assert settings["UV_PYTHON_PREFERENCE"] == "only-system"


def test_replay_refuses_an_environment_snapshot_from_another_image(replay_script, tmp_path, monkeypatch):
    value = json.loads(replay_script.IMAGE_ENVIRONMENT_FIXTURE.read_text())
    value["image"] += "-different"
    path = tmp_path / "image.json"
    path.write_text(json.dumps(value))
    monkeypatch.setattr(replay_script, "IMAGE_ENVIRONMENT_FIXTURE", path)
    with pytest.raises(ValueError, match="image_environment_mismatch"):
        replay_script.image_interpreter_environment(tmp_path)


def test_production_setup_defeats_inherited_discovery_before_both_commands(replay_script, tmp_path):
    """Execute the real setup scope, recording each installer invocation."""
    uv = tmp_path / "recording-uv"
    log = tmp_path / "uv.jsonl"
    uv.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
                  "with open(os.environ['UV_TEST_LOG'],'a') as f:\n"
                  " f.write(json.dumps({'args':sys.argv[1:],'environment':{k:os.environ[k] "
                  "for k in ['UV_PYTHON_PREFERENCE','UV_PYTHON_DOWNLOADS','UV_PYTHON_INSTALL_DIR',"
                  "'UV_PROJECT_ENVIRONMENT','VIRTUAL_ENV']}})+'\\n')\n")
    uv.chmod(0o700)
    install_root = tmp_path / "managed python"
    setup = replay_script.production_environment_setup(
        source=tmp_path / "source", venv=tmp_path / "policy", uv=uv,
        python_install_root=install_root,
    )
    subprocess.run(["bash", "-c", setup], check=True, env={
        **os.environ, "UV_TEST_LOG": str(log), "UV_PYTHON_PREFERENCE": "only-system",
        "UV_PYTHON_DOWNLOADS": "never", "UV_PYTHON": "/opt/isaaclab-venv/bin/python",
        "VIRTUAL_ENV": str(tmp_path / "sdk"), "UV_PROJECT_ENVIRONMENT": str(tmp_path / "sdk"),
    })
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [call["args"][0] for call in calls] == ["venv", "sync"]
    expected = {
        "UV_PYTHON_PREFERENCE": "managed", "UV_PYTHON_DOWNLOADS": "automatic",
        "UV_PYTHON_INSTALL_DIR": str(install_root),
        "UV_PROJECT_ENVIRONMENT": str(tmp_path / "sdk"), "VIRTUAL_ENV": str(tmp_path / "sdk"),
    }
    assert calls[0]["environment"] == expected
    assert calls[1]["environment"] == {**expected, "VIRTUAL_ENV": str(tmp_path / "policy")}
    assert calls[0]["args"][2] == "3.11.9"
    assert calls[1]["args"][calls[1]["args"].index("--python") + 1] == str(tmp_path / "policy/bin/python")


@pytest.mark.parametrize("broken", ["missing", "duplicated", "late", "wrong_root", "two_venvs"])
def test_setup_replay_refuses_missing_late_or_unbound_environment(replay_script, monkeypatch, tmp_path, broken):
    script = replay_script.build_provisioning_script("pi05_droid")
    line = 'export UV_PYTHON_PREFERENCE="managed"'
    if broken == "missing":
        script = script.replace(line, "")
    elif broken == "duplicated":
        script = script.replace(line, line + "\n" + line)
    elif broken == "late":
        script = script.replace(line, "") + "\n" + line
    elif broken == "wrong_root":
        script = script.replace('export UV_PYTHON_INSTALL_DIR="/opt/adp009d-python"',
                                'export UV_PYTHON_INSTALL_DIR="/opt/isaaclab-venv"')
    else:
        script += "\n" + next(line for line in script.splitlines() if '"$UV" venv ' in line)
    monkeypatch.setattr(replay_script, "build_provisioning_script", lambda _: script)
    with pytest.raises(ValueError, match="policy_dependency_replay_"):
        replay_script.production_environment_setup(
            source=tmp_path, venv=tmp_path, uv=tmp_path, python_install_root=tmp_path,
        )


@pytest.mark.parametrize("broken", ["missing_python", "wrong_python", "two_syncs"])
def test_replay_refuses_an_unbound_or_ambiguous_installer(replay_script, monkeypatch, tmp_path, broken):
    script = replay_script.build_provisioning_script("pi05_droid")
    if broken == "missing_python":
        script = script.replace('--python "/opt/adp009d-policy-venv/pi05_droid/bin/python"', "")
    elif broken == "wrong_python":
        script = script.replace(
            '--python "/opt/adp009d-policy-venv/pi05_droid/bin/python"',
            '--python "/opt/isaaclab-venv/bin/python"',
        )
    else:
        script += "\n" + next(line for line in script.splitlines() if '"$UV" sync ' in line)
    monkeypatch.setattr(replay_script, "build_provisioning_script", lambda _: script)
    with pytest.raises(ValueError, match="policy_dependency_replay_"):
        replay_script.production_sync_arguments(source=tmp_path, venv=tmp_path, uv=tmp_path)
