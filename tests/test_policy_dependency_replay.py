from __future__ import annotations

import importlib.util
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
