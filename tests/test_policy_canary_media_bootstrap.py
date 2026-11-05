"""The policy-canary entrypoint installs its own media tools before any checkpoint or cell."""
from pathlib import Path
import os
import shlex
import subprocess
import sys

import pytest

from blueprint_pipeline.native_task_arena_policy_canary_bundle import _entrypoint


@pytest.mark.parametrize("missing", ["ffmpeg", "ffprobe"])
@pytest.mark.parametrize("install_succeeds", [True, False])
@pytest.mark.parametrize("candidates", [("molmoact2_droid",), ("pi05_droid", "groot_n17_droid")])
def test_missing_media_tools_are_repaired_or_block_before_any_policy(
    tmp_path: Path, missing: str, install_succeeds: bool, candidates: tuple[str, ...],
) -> None:
    """2026-11-05: only the pi05/GR00T provisioning installed ffmpeg, so every molmoact2 cell
    refused at its dependency gate. The real generated bootstrap runs to its first simulator
    invocation; only package discovery and installation are shell test doubles."""
    prefix = _entrypoint(local_candidates=candidates).split(
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
        capture_output=True, text=True, timeout=30,
    )
    assert apt_log.is_file(), "The policy-canary entrypoint omitted the required media bootstrap"
    assert "install -y -qq ffmpeg" in apt_log.read_text()
    if install_succeeds:
        assert completed.returncode == 0, completed.stderr
        assert "SIMULATOR_BOUNDARY_REACHED" in completed.stdout
    else:
        assert completed.returncode == 86
        assert "SIMULATOR_BOUNDARY_REACHED" not in completed.stdout
        assert "policy_canary_media_toolchain_unavailable" in (
            out / "native_task_arena_policy_canary_session_result.v1.json"
        ).read_text()


def test_present_media_tools_install_nothing(tmp_path: Path) -> None:
    prefix = _entrypoint(local_candidates=("molmoact2_droid",)).split(
        "\n/isaac-sim/python.sh -m blueprint_pipeline.native_task_runtime_source_provision", 1
    )[0].replace("/isaac-sim/python.sh", shlex.quote(sys.executable))
    apt_log = tmp_path / "apt.log"
    doubles = f'''
command() {{
  if [ "$1" = -v ] && {{ [ "$2" = ffmpeg ] || [ "$2" = ffprobe ]; }}; then return 0; fi
  builtin command "$@"
}}
apt-get() {{ printf '%s\\n' "$*" >> {shlex.quote(str(apt_log))}; }}
'''
    script = tmp_path / "entrypoint.sh"
    script.write_text(doubles + prefix + '\ntrap - EXIT INT TERM HUP\necho SIMULATOR_BOUNDARY_REACHED\n')
    completed = subprocess.run(["bash", str(script)], env={**os.environ, "BLUEPRINT_ADP_ARENA_OUTPUT_DIR": str(tmp_path / "o")},
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0 and "SIMULATOR_BOUNDARY_REACHED" in completed.stdout
    assert not apt_log.exists()
