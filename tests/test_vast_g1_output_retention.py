"""The G1 review archive retains evidence without shipping runtime caches."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from blueprint_pipeline.vast_provider_adapter import _probe_shell_script


def test_g1_archive_keeps_diagnostics_and_media_but_excludes_runtime_caches(
    tmp_path: Path,
) -> None:
    shell = _probe_shell_script(
        "https://heartbeat.invalid",
        enable_blueprint_bundle=True,
        provider_bundle_kind="native_g1_development_campaign",
    )
    scripts = re.findall(r"\$RUNTIME_PY - <<'PY'\n(.*?)\nPY\n", shell, re.S)
    archive_script = next(
        script
        for script in scripts
        if "output_zip = work_dir / 'adp_arena_provider_runtime_output.zip'" in script
    )
    output = tmp_path / "runtime_output"
    (output / "policy-runtime-build/policy-runtime/lib").mkdir(parents=True)
    (output / "policy-runtime-build/policy-runtime/lib/installed.so").write_bytes(b"a" * 1024)
    (output / "policy-runtime-build/build.log").write_text("missing header\n")
    (output / "models/checkpoints/pi").mkdir(parents=True)
    (output / "models/checkpoints/pi/model.safetensors").write_bytes(b"weights")
    (output / "models/sonic").mkdir()
    (output / "models/sonic/model_encoder.onnx").write_bytes(b"encoder")
    (output / "models/candidate.json").write_text('{"sha256":"pinned"}\n')
    (output / "models/sonic.json").write_text('{"sha256":"pinned"}\n')
    (output / "media").mkdir()
    (output / "media/camera.mp4").write_bytes(b"episode-video")
    (output / "native_g1_provider_campaign_result.v1.json").write_text('{"status":"blocked"}\n')

    completed = subprocess.run(
        [sys.executable, "-I", "-c", archive_script],
        env={
            **os.environ,
            "BLUEPRINT_ADP_ARENA_OUTPUT_DIR": str(output),
            "BLUEPRINT_VAST_WORK_DIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    with zipfile.ZipFile(tmp_path / "adp_arena_provider_runtime_output.zip") as archive:
        assert "native_g1_provider_campaign_result.v1.json" in archive.namelist()
        assert "policy-runtime-build/build.log" in archive.namelist()
        assert "models/candidate.json" in archive.namelist()
        assert "models/sonic.json" in archive.namelist()
        assert "media/camera.mp4" in archive.namelist()
        assert not any(
            name.startswith(prefix)
            for name in archive.namelist()
            for prefix in (
                "policy-runtime-build/policy-runtime/",
                "models/checkpoints/",
                "models/sonic/",
            )
        )
