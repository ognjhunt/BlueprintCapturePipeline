"""Exercise the sealed team's shell bootstrap before simulator provisioning."""
from pathlib import Path
import os
import shlex
import subprocess
import sys

import pytest

from blueprint_pipeline.team_candidate_quick10_bundle import _entrypoint


@pytest.mark.parametrize("missing", ["ffmpeg", "ffprobe"])
@pytest.mark.parametrize("install_succeeds", [True, False])
def test_missing_media_tools_are_repaired_or_block_before_isaac(
    tmp_path: Path, missing: str, install_succeeds: bool,
) -> None:
    # Execute the real generated bootstrap to its first simulator invocation.
    # Only package discovery/installation is replaced with shell test doubles.
    prefix = _entrypoint("BLUEPRINT_TEAM_POLICY_BEARER_FILE").split(
        "\n/isaac-sim/python.sh -m blueprint_pipeline.native_task_runtime_source_provision", 1
    )[0].replace("/isaac-sim/python.sh", shlex.quote(sys.executable))
    out = tmp_path / "output"
    installed = tmp_path / "installed"
    apt_log = tmp_path / "apt.log"
    doubles = f'''
command() {{
  if [ "$1" = -v ] && {{ [ "$2" = ffmpeg ] || [ "$2" = ffprobe ]; }}; then
    [ "$2" != {shlex.quote(missing)} ] || [ -f {shlex.quote(str(installed))} ]
  else
    builtin command "$@"
  fi
}}
apt-get() {{
  printf '%s\\n' "$*" >> {shlex.quote(str(apt_log))}
  if [ "$1" = install ]; then
    {"touch " + shlex.quote(str(installed)) if install_succeeds else "return 1"}
  fi
}}
'''
    script = tmp_path / "entrypoint.sh"
    script.write_text(doubles + prefix + '\ntrap - EXIT INT TERM HUP\necho SIMULATOR_BOUNDARY_REACHED\n')
    completed = subprocess.run(
        ["bash", str(script)], env={**os.environ, "BLUEPRINT_ADP_ARENA_OUTPUT_DIR": str(out)},
        capture_output=True, text=True, timeout=15,
    )
    assert apt_log.is_file(), "The team entrypoint omitted the required media bootstrap"
    assert "install -y -qq ffmpeg" in apt_log.read_text()
    if install_succeeds:
        assert completed.returncode == 0
        assert "SIMULATOR_BOUNDARY_REACHED" in completed.stdout
    else:
        assert completed.returncode != 0
        assert "SIMULATOR_BOUNDARY_REACHED" not in completed.stdout
        assert "policy_canary_media_toolchain_unavailable" in (
            out / "native_task_arena_policy_canary_session_result.v1.json"
        ).read_text()
