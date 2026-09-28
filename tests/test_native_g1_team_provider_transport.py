"""The real selected bundle must survive canonical Isaac provider transport."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import re
import sys
import zipfile

import pytest

from blueprint_pipeline import native_g1_team_provider_bundle as builder
from blueprint_pipeline import vast_provider_adapter as vast
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_execution_contract import native_task_arena_execution_transport_completed
from blueprint_pipeline.provider_runtime_bundle_contract import provider_runtime_contract_blockers, provider_command_execute_fallback_allowed
from tests.test_native_g1_team_provider_bundle import _inputs


def test_actual_selected_bundle_passes_isaac_static_transport(tmp_path, monkeypatch):
    args, _ = _inputs(tmp_path, monkeypatch, endpoint=True)
    receipt = builder.build_g1_team_provider_bundle(**args)
    assert vast._is_isaac_provider_bundle(builder.PROVIDER_BUNDLE_KIND)
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        manifest = builder.validate_g1_team_provider_manifest(archive)
        assert manifest["manifest_digest"] == receipt["manifest_digest"]
        blockers = provider_runtime_contract_blockers(
            provider_bundle_kind=builder.PROVIDER_BUNDLE_KIND,
            entrypoint_text=archive.read(builder.ENTRYPOINT).decode(),
            runner_text=archive.read("provider_runtime/blueprint_pipeline/native_g1_team_provider_runtime.py").decode(),
        )
        assert blockers == []
    result = vast._blueprint_bundle_preflight(
        job_dir=tmp_path, generated_at="2026-09-28T02:00:00Z",
        enable_blueprint_bundle=True, enable_isaac_smoke=True,
        provider_bundle_kind=builder.PROVIDER_BUNDLE_KIND,
        bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://objects.example.org/private-bundle",
        provider_output_put_url="https://objects.example.org/private-output",
    )
    assert result["blockers"] == []


def test_provider_manifest_refuses_changed_shipped_bytes(tmp_path, monkeypatch):
    args, _ = _inputs(tmp_path, monkeypatch, endpoint=True)
    receipt = builder.build_g1_team_provider_bundle(**args)
    path = Path(receipt["bundle_path"])
    with zipfile.ZipFile(path) as archive:
        members = {info.filename: (info, archive.read(info)) for info in archive.infolist()}
    info, data = members[builder.PACKET_RELATIVE_PATH]
    members[builder.PACKET_RELATIVE_PATH] = (info, data + b" ")
    with zipfile.ZipFile(path, "w") as archive:
        for info, data in members.values():
            archive.writestr(info, data)
    with zipfile.ZipFile(path) as archive, pytest.raises(ValueError, match="artifact"):
        builder.validate_g1_team_provider_manifest(archive)


def test_selected_terminal_transport_requires_queried_bound_development_result():
    proof = {
        "schema_version": "native_g1_team_provider_result.v1", "status": "completed_development_only",
        "candidate_policy_queried": True, "claim_ceiling": "development_only",
        "verified_output": {"status": "verified_development_only", "policy_query_count": 1},
        "provider_teardown_verified": False, "official_billing_reconciled": False,
        "public_redistribution_authorized": False,
    }
    proof["result_digest"] = canonical_digest(proof, digest_field="result_digest")
    assert native_task_arena_execution_transport_completed(proof, expected_output_filename=builder.RESULT_FILENAME)
    for field, value in (("candidate_policy_queried", False), ("provider_teardown_verified", True), ("claim_ceiling", "physical_proof")):
        changed = {**proof, field: value}
        changed["result_digest"] = canonical_digest(changed, digest_field="result_digest")
        assert not native_task_arena_execution_transport_completed(changed, expected_output_filename=builder.RESULT_FILENAME)
    assert not native_task_arena_execution_transport_completed(proof, expected_output_filename="foreign-result.json")


def test_real_selected_bootstrap_shell_retains_large_lossless_files(tmp_path):
    script = vast._probe_shell_script(
        "https://heartbeat.example.org", enable_blueprint_bundle=True,
        enable_isaac_smoke=True, provider_bundle_kind=builder.PROVIDER_BUNDLE_KIND,
        expected_provider_bundle_sha256="sha256:" + "a" * 64,
    )
    shell = tmp_path / "bootstrap.sh"
    shell.write_text(script)
    assert subprocess.run(["bash", "-n", str(shell)], capture_output=True, check=False).returncode == 0
    blocks = re.findall(r"\$RUNTIME_PY - <<'PY'\n(.*?)\nPY\n", script, re.S)
    packaging = [block for block in blocks if "preserve_all_output = True" in block]
    assert len(packaging) == 1
    output = tmp_path / "output"
    output.mkdir()
    (output / builder.RESULT_FILENAME).write_text(json.dumps({"status": "blocked"}))
    frame = output / "policy-input.png"
    # Sparse bytes make this a >100 MB regression case without consuming that
    # much local disk. This is packaging proof, not valid observation evidence.
    with frame.open("wb") as stream:
        stream.truncate(100_000_001)
    environment = {**os.environ, "BLUEPRINT_ADP_ARENA_OUTPUT_DIR": str(output), "BLUEPRINT_VAST_WORK_DIR": str(tmp_path)}
    result = subprocess.run([sys.executable, "-c", packaging[0]], env=environment, capture_output=True, check=False)
    assert result.returncode == 0
    with zipfile.ZipFile(tmp_path / "adp_arena_provider_runtime_output.zip") as archive:
        assert archive.getinfo("policy-input.png").file_size == 100_000_001
        assert archive.getinfo(builder.RESULT_FILENAME).file_size > 0
    (output / builder.RESULT_FILENAME).unlink()
    result = subprocess.run([sys.executable, "-c", packaging[0]], env=environment, capture_output=True, check=False)
    assert result.returncode != 0


def test_selected_logs_cannot_trigger_command_restart_even_with_override():
    assert not provider_command_execute_fallback_allowed(builder.PROVIDER_BUNDLE_KIND, configured_override=True)
    assert not provider_command_execute_fallback_allowed(builder.PROVIDER_BUNDLE_KIND, configured_override=False)
    assert provider_command_execute_fallback_allowed("native_g1_development_campaign", configured_override=False)
    assert provider_command_execute_fallback_allowed("wam", configured_override=True)
