"""Postchecks must import the selected ADP release through a shared virtualenv."""

import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


POSTCHECK = Path(__file__).resolve().parents[1] / "deploy/systemd/blueprint-control-plane-postchecks.sh"


@pytest.mark.parametrize("shared_venv", [True, False])
@pytest.mark.parametrize("audit_rc,alert_rc,expected_rc", [(0, 3, 3), (7, 0, 7)])
def test_postchecks_use_selected_release_and_preserve_failure_signals(
    tmp_path: Path, shared_venv: bool, audit_rc: int, alert_rc: int, expected_rc: int
) -> None:
    selected = tmp_path / "release"
    stale = tmp_path / "stale"
    for root, origin in ((selected, "release"), (stale, "stale")):
        package = root / "src/blueprint_pipeline"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        for role, code in (("proof_audit", audit_rc), ("manifest_alert", alert_rc)):
            (package / f"live_pipeline_{role}.py").write_text(
                "import os\n"
                "with open(os.environ['POSTCHECK_LOG'], 'a') as stream:\n"
                f"    stream.write({(origin + ':' + role + chr(10))!r})\n"
                f"raise SystemExit({code})\n"
            )
    # The shared interpreter carries an older editable checkout in production.
    # Give this subprocess an explicit stale import root to reproduce that risk.
    shim = "#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n'
    tools = tmp_path / "tools"
    tools.mkdir()
    fallback = tools / "python3"
    fallback.write_text(shim)
    fallback.chmod(0o755)
    if shared_venv:
        interpreter = selected / ".venv/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text(shim)
        interpreter.chmod(0o755)
    link = tmp_path / "active-release"
    link.symlink_to(selected, target_is_directory=True)
    log = tmp_path / "postchecks.log"
    env = dict(os.environ, BLUEPRINT_PIPELINE_REPO=str(link),
               PYTHONPATH=str(stale / "src"), POSTCHECK_LOG=str(log),
               PATH=str(tools) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(["bash", str(POSTCHECK)], env=env, capture_output=True,
                            text=True, timeout=10, check=False)
    assert result.returncode == expected_rc, result.stderr
    assert log.read_text().splitlines() == ["release:proof_audit", "release:manifest_alert"]


UNIT = POSTCHECK.parent / "blueprint-pipeline-control-plane.service"
ACTIVE_RELEASE = "/opt/blueprint/task-evaluation-control-plane"
SHARED_PYTHON = "/opt/blueprint/BlueprintCapturePipeline/.venv/bin/python"
UNIT_MODULES = ("production_runtime_env_guard", "live_pipeline_control_plane",
                "live_pipeline_proof_audit", "live_pipeline_manifest_alert")


def _logging_checkout(root: Path, origin: str) -> None:
    package = root / "src/blueprint_pipeline"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    for module in UNIT_MODULES:
        (package / f"{module}.py").write_text(
            "import os\n"
            "with open(os.environ['POSTCHECK_LOG'], 'a') as stream:\n"
            f"    stream.write(':'.join(({origin + ':' + module!r}, os.environ.get('INTERPRETER', '?'),\n"
            "                           os.path.realpath(os.environ['BLUEPRINT_PIPELINE_REPO']))) + '\\n')\n"
        )
    script = root / "deploy/systemd" / POSTCHECK.name
    script.parent.mkdir(parents=True)
    script.write_text(POSTCHECK.read_text())


def _interpreter(path: Path, tag: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\nINTERPRETER={tag} exec " + shlex.quote(sys.executable) + ' "$@"\n')
    path.chmod(0o755)
    return path


def test_control_plane_unit_runs_the_active_release_when_the_credential_env_names_an_archive(
    tmp_path: Path,
) -> None:
    """systemd lets ``EnvironmentFile=`` override ``Environment=`` (PR #519).

    Production's credential env file names an archived checkout in
    ``BLUEPRINT_PIPELINE_REPO``, so the pass, its proof audit and its blocked
    manifest alert all ran archived code: the deployed alert's setup blockers
    never reached the operator. A release tree has no ``.venv`` and the audit
    needs the shared interpreter's dependencies, so bare ``python3`` must not run it.
    """
    release, archived = tmp_path / "release", tmp_path / "archived"
    _logging_checkout(release, "release")
    _logging_checkout(archived, "archived")
    link = tmp_path / "active-release"
    link.symlink_to(release, target_is_directory=True)
    shared = _interpreter(tmp_path / "shared-venv/bin/python", "shared")
    _interpreter(tmp_path / "tools/python3", "bare")

    unit = UNIT.read_text()
    environment = {}
    for line in unit.splitlines():
        if line.startswith("Environment="):
            key, _, value = line.removeprefix("Environment=").partition("=")
            environment[key] = value.replace(ACTIVE_RELEASE, str(link)).replace(SHARED_PYTHON, str(shared))
    environment["BLUEPRINT_PIPELINE_REPO"] = str(archived)  # the credential EnvironmentFile wins
    log = tmp_path / "control-plane.log"
    environment.update(POSTCHECK_LOG=str(log),
                       PATH=str(tmp_path / "tools") + os.pathsep + os.environ["PATH"])

    for directive in ("ExecStartPre", "ExecStart", "ExecStartPost"):
        prefix = f"{directive}=/bin/bash -lc '"
        line = next(line for line in unit.splitlines() if line.startswith(prefix))
        script = line.removeprefix(prefix).removesuffix("'").replace("$$", "$")
        result = subprocess.run(["bash", "-c", script], env=environment, capture_output=True,
                                text=True, timeout=10, check=False)
        assert result.returncode == 0, f"{directive}: {result.stderr}"

    resolved = os.path.realpath(release)
    assert log.read_text().splitlines() == [
        f"release:{module}:shared:{resolved}" for module in UNIT_MODULES
    ]
