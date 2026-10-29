"""The asset import lane has its own result; no policy/task execution is implied."""

from __future__ import annotations

import json
import re
import zipfile

import pytest

from blueprint_pipeline import vast_provider_adapter as vast
from blueprint_pipeline import vast_provider_output_recovery as recovery
from blueprint_pipeline.deployment_embodiment_preparation_contract import (
    BUNDLE_KIND, MAX_OUTPUT_BYTES, RESULT_FILENAME,
)
from blueprint_pipeline.provider_runtime_bundle_contract import (
    PROVIDER_RUNTIME_BUNDLE_KINDS,
)


def _export_script():
    shell = vast._probe_shell_script(
        "https://heartbeat.invalid", enable_blueprint_bundle=True,
        provider_bundle_kind=BUNDLE_KIND,
        expected_provider_bundle_sha256="sha256:" + "1" * 64,
    )
    assert "BLUEPRINT_VAST_ARENA_BUNDLE_SHA256_VERIFIED" in shell
    assert "run_adp_arena_provider_runtime.sh" in shell
    return next(script for script in re.findall(r"\$RUNTIME_PY - <<'PY'\n(.*?)\nPY\n", shell, re.S)
                if "output_zip = work_dir / 'adp_arena_provider_runtime_output.zip'" in script)


def _output(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    (output / RESULT_FILENAME).write_text(json.dumps({"status": "completed"}))
    monkeypatch.setenv("BLUEPRINT_ADP_ARENA_OUTPUT_DIR", str(output))
    monkeypatch.setenv("BLUEPRINT_VAST_WORK_DIR", str(tmp_path))
    return output


def test_asset_preparation_uses_isaac_transport_without_task_or_video_requirement(tmp_path):
    assert BUNDLE_KIND in PROVIDER_RUNTIME_BUNDLE_KINDS
    assert vast._is_isaac_provider_bundle(BUNDLE_KIND)
    assert vast._provider_expected_video_count(BUNDLE_KIND) == 0
    assert vast._resolve_launch_mode(
        requested="auto", enable_isaac_smoke=True, enable_blueprint_bundle=True,
        provider_bundle_kind=BUNDLE_KIND) == "ssh_direct"
    assert vast._resolve_probe_image(
        public_image="public", isaac_image="isaac", enable_isaac_smoke=False,
        enable_blueprint_bundle=True, provider_bundle_kind=BUNDLE_KIND) == "isaac"
    env = vast._probe_env(
        job_dir=tmp_path, enable_isaac_smoke=True, provider_bundle_kind=BUNDLE_KIND,
        forward_hf_token=False, provider_hard_deadline_epoch=1234567890.0)
    assert env["BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH"] == "1234567890.0"
    assert env["ACCEPT_EULA"] == "Y"
    assert "HF_TOKEN" not in env
    assert recovery._REMOTE_ARCHIVE_BY_BUNDLE_KIND[BUNDLE_KIND] == (
        "/workspace/adp_arena_provider_runtime_output.zip")


def test_export_retains_the_large_model_and_the_distinct_result(tmp_path, monkeypatch):
    output = _output(tmp_path, monkeypatch)
    model = output / "bodies" / "yam" / "prepared" / "robot.usd"
    model.parent.mkdir(parents=True)
    with model.open("wb") as stream:
        stream.truncate(100_000_001)
    exec(compile(_export_script(), "<actual-provider-export>", "exec"), {})
    with zipfile.ZipFile(tmp_path / "adp_arena_provider_runtime_output.zip") as archive:
        assert archive.getinfo("bodies/yam/prepared/robot.usd").file_size == 100_000_001
        assert archive.getinfo(RESULT_FILENAME).file_size > 0
        assert "native_task_arena_policy_canary_session_result.v1.json" not in archive.namelist()


@pytest.mark.parametrize("fault", ["missing_result", "symlink", "symlink_directory", "over_budget"])
def test_export_refuses_incomplete_unbounded_or_nonregular_asset_outputs(tmp_path, monkeypatch, fault):
    output = _output(tmp_path, monkeypatch)
    if fault == "missing_result":
        (output / RESULT_FILENAME).unlink()
    elif fault in {"symlink", "symlink_directory"}:
        target = tmp_path / "external"
        if fault == "symlink_directory":
            target.mkdir()
        else:
            target.write_bytes(b"external bytes")
        (output / "linked").symlink_to(target)
    else:
        with (output / "oversize.usd").open("wb") as stream:
            stream.truncate(MAX_OUTPUT_BYTES)
    with pytest.raises(SystemExit):
        exec(compile(_export_script(), "<actual-provider-export>", "exec"), {})
    assert not (tmp_path / "adp_arena_provider_runtime_output.zip").exists()


def _transport_bundle(tmp_path, *, change_worker=False):
    """Tiny transport fixture; native sources/imports are covered by the lane tests."""
    import hashlib
    from pathlib import Path
    import time

    from blueprint_pipeline.deployment_embodiment_preparation_bundle import ENTRYPOINT, MANIFEST_PATH, SCHEMA
    from blueprint_pipeline.deployment_embodiment_preparation_contract import (
        INPUT_FILENAME, INPUT_SCHEMA, REQUEST_SCHEMA, sealed,
    )
    from blueprint_pipeline.native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE

    request = sealed({
        "schema_version": REQUEST_SCHEMA, "authorization_id": "transport-test",
        "release_commit": "1" * 40, "embodiment_ids": ["i2rt_yam_bimanual"],
        "max_spend_usd": 4.0, "max_hourly_rate_usd": .8, "ttl_seconds": 3600,
        "expires_at_epoch": time.time() + 86400, "authorization_reference": "test://transport",
    }, "request_digest")
    inputs = sealed({"schema_version": INPUT_SCHEMA, "request": request}, "inputs_digest")
    package = Path(vast.__file__).parent
    members = {
        "provider_runtime/run_adp_arena_provider_runtime.sh": ENTRYPOINT.encode(),
        "provider_runtime/" + INPUT_FILENAME: json.dumps(inputs).encode(),
        "provider_runtime/blueprint_pipeline/deployment_embodiment_preparation_worker.py": (
            package / "deployment_embodiment_preparation_worker.py").read_bytes(),
        "provider_runtime/blueprint_pipeline/deployment_embodiment_preparation_contract.py": (
            package / "deployment_embodiment_preparation_contract.py").read_bytes(),
        "provider_runtime/native_task_runtime_sources/native_task_runtime_source_packet.v1.json": b"{}",
        "provider_runtime/native_task_runtime_sources/native_task_runtime_sources.zip": b"test packet",
    }
    manifest = sealed({"schema_version": SCHEMA, "status": "ready", "inputs": inputs,
        "container_image": NATIVE_TASK_ARENA_IMAGE,
        "files": [{"path": name, "size_bytes": len(value),
                   "sha256": "sha256:" + hashlib.sha256(value).hexdigest()}
                  for name, value in members.items()]}, "manifest_digest")
    members[MANIFEST_PATH] = json.dumps(manifest).encode()
    if change_worker:
        members["provider_runtime/blueprint_pipeline/deployment_embodiment_preparation_worker.py"] += b"# changed\n"
    archive = tmp_path / "provider.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, value in members.items():
            zipped.writestr(name, value)
    # A mutable adjacent manifest is never an alternate source of authority.
    (tmp_path / "embodiment_asset_preparation_bundle.v1.json").write_text(
        '{"local_bundle_ready_for_remote_staging":false}')
    return archive


@pytest.mark.parametrize("change_worker", [False, True])
def test_preflight_binds_immutable_asset_bundle_not_sidecar(tmp_path, change_worker):
    result = vast._blueprint_bundle_preflight(
        job_dir=tmp_path, generated_at="2026-10-29T00:00:00Z",
        enable_blueprint_bundle=True, enable_isaac_smoke=True, provider_bundle_kind=BUNDLE_KIND,
        bundle_path=_transport_bundle(tmp_path, change_worker=change_worker),
        provider_bundle_url="https://storage.googleapis.com/bucket/input.zip",
        provider_output_put_url="https://storage.googleapis.com/bucket/output.zip",
        verify_staging_urls=False,
    )
    assert result["provider_bundle_readiness_source"] == "immutable_bundle_member"
    if change_worker:
        assert result["status"] == "blocked"
        assert "embodiment_asset_preparation_provider_manifest_invalid" in result["blockers"]
    else:
        assert result["status"] == "passed", result["blockers"]
        assert result["provider_bundle_local_ready_for_remote_staging"] is True


@pytest.mark.parametrize("fault", [None, "native_task_result", "wrong_inputs", "missing_model", "policy_query"])
def test_completion_checks_asset_identity_and_retained_bytes(tmp_path, fault):
    import hashlib

    from blueprint_pipeline.adp_isaac_lab_arena_vast import _provider_execution_transport_completed
    from blueprint_pipeline.deployment_embodiment_preparation_contract import RESULT_SCHEMA, sealed

    payload = b"retained import bytes"
    (tmp_path / "model.usd").write_bytes(payload)
    result = {
        "schema_version": RESULT_SCHEMA, "status": "completed", "claim_ceiling": "robot_asset_import_only",
        "candidate_policy_queried": False, "task_executed": False, "provider_teardown_verified": False,
        "inputs_digest": "sha256:" + "2" * 64,
        "results": [{"embodiment_id": "i2rt_yam_bimanual", "status": "import_failed", "reason": "test"}],
        "files": [{"path": "model.usd", "size_bytes": len(payload),
                   "sha256": "sha256:" + hashlib.sha256(payload).hexdigest()}],
    }
    bundle = {"inputs_digest": result["inputs_digest"],
              "inputs": {"request": {"embodiment_ids": ["i2rt_yam_bimanual"]}}}
    if fault == "native_task_result":
        result["schema_version"] = "native_task_arena_construction_result.v1"
    elif fault == "wrong_inputs":
        result["inputs_digest"] = "sha256:" + "3" * 64
    elif fault == "missing_model":
        (tmp_path / "model.usd").unlink()
    elif fault == "policy_query":
        result["candidate_policy_queried"] = True
    result = sealed(result, "result_digest")
    (tmp_path / RESULT_FILENAME).write_text(json.dumps(result))
    assert _provider_execution_transport_completed(
        result, provider_bundle_kind=BUNDLE_KIND, expected_output_filename=RESULT_FILENAME,
        bundle=bundle, result_path=str(tmp_path / RESULT_FILENAME)) is (fault is None)


def test_export_checks_final_archive_bytes_including_zip_overhead(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    _output(tmp_path, monkeypatch)
    original_stat = Path.stat

    def archive_overhead(path, *args, **kwargs):
        measured = original_stat(path, *args, **kwargs)
        if path == tmp_path / "adp_arena_provider_runtime_output.zip":
            return SimpleNamespace(st_size=MAX_OUTPUT_BYTES + 1)
        return measured

    monkeypatch.setattr(Path, "stat", archive_overhead)
    with pytest.raises(SystemExit, match="embodiment_preparation_archive_bound_exceeded"):
        exec(compile(_export_script(), "<actual-provider-export>", "exec"), {})
