"""Postchecks must import the selected ADP release through a shared virtualenv."""

import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


POSTCHECK = Path(__file__).resolve().parents[1] / "deploy/systemd/blueprint-control-plane-postchecks.sh"


@pytest.mark.parametrize("shared_venv", [True, False])
@pytest.mark.parametrize("alert_rc", [0, 3])
def test_postchecks_use_selected_release_and_preserve_failure_signals(
    tmp_path: Path, shared_venv: bool, alert_rc: int
) -> None:
    selected = tmp_path / "release"
    stale = tmp_path / "stale"
    for root, origin in ((selected, "release"), (stale, "stale")):
        package = root / "src/blueprint_pipeline"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "live_pipeline_manifest_alert.py").write_text(
            "import os\n"
            "with open(os.environ['POSTCHECK_LOG'], 'a') as stream:\n"
            f"    stream.write({(origin + ':manifest_alert' + chr(10))!r})\n"
            f"raise SystemExit({alert_rc})\n"
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
    assert result.returncode == alert_rc, result.stderr
    assert log.read_text().splitlines() == ["release:manifest_alert"]


