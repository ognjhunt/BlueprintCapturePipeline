"""Exercise the actual export script at the previously lost 100 MB boundary."""

import json
import re
import zipfile

import pytest

from blueprint_pipeline.adp_isaac_lab_arena_vast import _extract_provider_output
from blueprint_pipeline.vast_provider_adapter import _probe_shell_script


def export_script():
    shell = _probe_shell_script("https://heartbeat.invalid", enable_blueprint_bundle=True,
                               provider_bundle_kind="scene_configuration_native_continuation")
    return next(script for script in re.findall(r"\$RUNTIME_PY - <<'PY'\n(.*?)\nPY\n", shell, re.S)
                if "output_zip = work_dir / 'adp_arena_provider_runtime_output.zip'" in script)


def test_native_export_retains_large_scene_and_measurement_archive(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    (output / "adp009d_native_microcheck.json").write_text(json.dumps({"status": "completed"}))
    payload = output / "native_continuation_stage_output.zip"
    with payload.open("wb") as stream:
        stream.truncate(100_000_001)
    monkeypatch.setenv("BLUEPRINT_ADP_ARENA_OUTPUT_DIR", str(output))
    monkeypatch.setenv("BLUEPRINT_VAST_WORK_DIR", str(tmp_path))
    exec(compile(export_script(), "<actual-provider-export>", "exec"), {})
    with zipfile.ZipFile(tmp_path / "adp_arena_provider_runtime_output.zip") as archive:
        assert archive.getinfo(payload.name).file_size == 100_000_001
        assert archive.getinfo("adp009d_native_microcheck.json").file_size > 0


@pytest.mark.parametrize("fault", ["missing", "empty", "symlink"])
def test_native_export_refuses_missing_or_invalid_required_evidence(tmp_path, monkeypatch, fault):
    output = tmp_path / "output"
    output.mkdir()
    (output / "adp009d_native_microcheck.json").write_text('{"status":"completed"}')
    evidence = output / "native_continuation_stage_output.zip"
    if fault == "empty":
        evidence.touch()
    elif fault == "symlink":
        target = tmp_path / "target"
        target.write_bytes(b"evidence")
        evidence.symlink_to(target)
    monkeypatch.setenv("BLUEPRINT_ADP_ARENA_OUTPUT_DIR", str(output))
    monkeypatch.setenv("BLUEPRINT_VAST_WORK_DIR", str(tmp_path))
    with pytest.raises(SystemExit, match="provider_required_evidence_"):
        exec(compile(export_script(), "<actual-provider-export>", "exec"), {})
    assert not (tmp_path / "adp_arena_provider_runtime_output.zip").exists()


def test_native_ingestion_refuses_the_actual_summary_only_success_archive(tmp_path):
    source = tmp_path / "summary_only.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("adp009d_native_microcheck.json", '{"status":"completed"}')
    destination = tmp_path / "extracted"
    result = _extract_provider_output(source, destination, result_name="adp009d_native_microcheck.json",
                                     required_members=("native_continuation_stage_output.zip",))
    assert result["status"] == "blocked"
    assert "adp_arena_required_output_missing:native_continuation_stage_output.zip" in result["blockers"]
    assert not destination.exists()
