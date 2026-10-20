"""Exercise evaluator transport on images with Python but no unzip binary."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from blueprint_pipeline.vast_evaluator_probe_script import evaluator_provider_probe_script


@pytest.mark.slow
def test_evaluator_extracts_and_uploads_without_unzip_or_apt(tmp_path: Path) -> None:
    commands = tmp_path / "commands"
    commands.mkdir()
    (commands / "python3").symlink_to(sys.executable)
    for name, target in (("bash", "/bin/bash"), ("rm", "/bin/rm")):
        (commands / name).symlink_to(target)
    for name in ("curl", "ffmpeg"):
        path = commands / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    apt = commands / "apt-get"
    apt.write_text('#!/bin/sh\nprintf attempted > "$APT_ATTEMPT_MARKER"\nexit 99\n')
    apt.chmod(0o755)
    assert not (commands / "unzip").exists()

    bundle = tmp_path / "input.zip"
    entrypoint = '''#!/bin/bash
"$BLUEPRINT_EVALUATOR_PROVIDER_PYTHON" - <<'PY'
import json, os
from pathlib import Path
output = Path(os.environ["BLUEPRINT_EVALUATOR_PROVIDER_OUTPUT_DIR"])
output.mkdir(parents=True)
(output / "evaluator_runtime_result.json").write_text(json.dumps({"status":"fixture_transport_completed"}))
PY
'''
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("provider_runtime/run_evaluator_provider_runtime.sh", entrypoint)
    work = tmp_path / "work"
    work.mkdir()
    uploaded = tmp_path / "uploaded.zip"
    marker = tmp_path / "apt-attempted"
    common = '''WORK_DIR="$TEST_WORK_DIR"; export BLUEPRINT_VAST_WORK_DIR="$WORK_DIR";
BUNDLE_URL=fixture; OUTPUT_PUT_URL=fixture;
blueprint_download_url() { /bin/cp "$TEST_INPUT_BUNDLE" "$2"; }
blueprint_upload_put() { /bin/cp "$2" "$TEST_UPLOADED_OUTPUT"; }
blueprint_upload_read_response() { return 0; }
blueprint_upload_cleanup() { return 0; }
if [ -z "$BUNDLE_URL" ] || [ -z "$OUTPUT_PUT_URL" ]; then exit 98; else
'''
    result = subprocess.run(["/bin/bash", "-c", evaluator_provider_probe_script(common)],
        env={**os.environ, "PATH":str(commands), "TEST_WORK_DIR":str(work),
             "TEST_INPUT_BUNDLE":str(bundle), "TEST_UPLOADED_OUTPUT":str(uploaded),
             "APT_ATTEMPT_MARKER":str(marker)}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists(), "Python ZIP extraction must not trigger apt for missing unzip"
    assert "BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_STARTED" in result.stdout
    assert "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK" in result.stdout
    assert "BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED" not in result.stdout
    with zipfile.ZipFile(uploaded) as archive:
        assert json.loads(archive.read("evaluator_runtime_result.json")) == {
            "status":"fixture_transport_completed"}
