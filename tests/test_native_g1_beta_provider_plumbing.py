"""Offline provider transport checks; no allocation or native policy proof."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import zipfile

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import provider_runtime_bundle_contract as contract
from blueprint_pipeline import vast_provider_adapter as vast
from blueprint_pipeline import vast_provider_output_recovery as recovery
from blueprint_pipeline.vast_arena_output_packaging import arena_provider_output_packaging_script
from blueprint_pipeline import adp_isaac_lab_arena_vast as arena
from blueprint_pipeline import native_g1_beta_transport as transport
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_vast_provider_output_recovery import _install_identity

KIND = "native_g1_beta_screening"
RESULT = "native_g1_beta_session_result.v1.json"


def test_g1_beta_uses_native_image_and_one_shot_transport():
    assert KIND in contract.PROVIDER_RUNTIME_BUNDLE_KINDS
    assert vast._is_isaac_provider_bundle(KIND)
    assert vast._resolve_probe_image(public_image="unused", isaac_image="pinned-isaac",
        enable_isaac_smoke=False, enable_blueprint_bundle=True, provider_bundle_kind=KIND) == "pinned-isaac"
    assert vast._resolve_launch_mode(requested="auto", enable_isaac_smoke=True,
        enable_blueprint_bundle=True, provider_bundle_kind=KIND) == "ssh_direct"
    assert vast._provider_expected_video_count(KIND) == 0
    for override in (False, True):
        assert contract.provider_command_execute_fallback_allowed(KIND, configured_override=override) is False
    # The new beta kind does not alter existing campaign or selected-team behavior.
    assert contract.provider_command_execute_fallback_allowed("native_g1_development_campaign", configured_override=False)
    assert not contract.provider_command_execute_fallback_allowed("native_g1_team_policy", configured_override=True)


def test_static_contract_requires_the_dedicated_runner_and_crash_result():
    entry = "\n".join(("native_task_runtime_source_provision", "native_g1_beta_runtime", RESULT,
                        "g1_beta_worker_failed_without_result"))
    runner = "\n".join(("native_g1_development_worker", "BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH",
                         "candidate_policy_queried", "development_only"))
    assert contract.provider_runtime_contract_blockers(provider_bundle_kind=KIND,
        entrypoint_text=entry, runner_text=runner) == []
    assert contract.provider_runtime_contract_blockers(provider_bundle_kind=KIND,
        entrypoint_text=entry.replace(RESULT, "adp009d_native_microcheck.json"), runner_text=runner)
    assert contract.provider_runtime_contract_blockers(provider_bundle_kind=KIND,
        entrypoint_text=entry, runner_text="execute_paired_session") == [
            "provider_runner_missing_native_g1_beta_runtime_contract"]


def test_deadline_is_injected_and_cannot_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setattr(vast, "_read_hf_token_file", lambda: (None, {}))
    env = vast._probe_env(job_dir=tmp_path, enable_isaac_smoke=True,
        provider_bundle_kind=KIND, provider_hard_deadline_epoch=1_900_000_000.0,
        forward_hf_token=False)
    assert env["BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH"] == "1900000000.0"
    with pytest.raises(ValueError, match="deadline_override_forbidden"):
        vast._probe_env(job_dir=tmp_path, enable_isaac_smoke=True, provider_bundle_kind=KIND,
            provider_runtime_environment={"BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH": "9999999999"})


def _packaging(tmp_path, output, *, prefix=""):
    source = arena_provider_output_packaging_script(KIND)
    code = re.search(r"\$RUNTIME_PY - <<'PY'\n(.*?)\nPY\n", source, re.S).group(1)
    return subprocess.run([sys.executable, "-c", prefix + code], capture_output=True, check=False,
        env={**os.environ, "BLUEPRINT_ADP_ARENA_OUTPUT_DIR": str(output),
             "BLUEPRINT_VAST_WORK_DIR": str(tmp_path)})


def test_actual_bootstrap_and_packager_retain_evidence_not_execution_inputs(tmp_path):
    script = vast._probe_shell_script("https://heartbeat.example.org", enable_blueprint_bundle=True,
        enable_isaac_smoke=True, provider_bundle_kind=KIND,
        expected_provider_bundle_sha256="sha256:" + "a" * 64)
    assert "native_task_runtime_dependency_download" in script
    assert "BLUEPRINT_VAST_PROVIDER_BUNDLE_DOWNLOADED" in script
    assert subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False).returncode == 0
    output = tmp_path / "output"
    output.mkdir()
    (output / RESULT).write_text(json.dumps({"status": "blocked", "claim_ceiling": "development_only"}))
    excluded = ["policy-runtime-build/policy-runtime/model.bin", "policy-runtime-build/wheel-downloads/torch.whl",
        "models/checkpoints/weights.bin",
        "models/sonic/weights.bin", "models/torch/weights.bin", "cell_packets/cell/scene.usda"]
    retained = ["models/checkpoint-inventory.json", "policy-runtime-build/build.json",
        "cells/cell/worker/query_checkpoints/0.json", "cells/cell/request.json"]
    for name in excluded + retained:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("development-only fixture")
    # Preserve an entire observation even above the shared generic 100 MB cutoff.
    frame = output / "cells/cell/worker/policy-input.png"
    with frame.open("wb") as stream:
        stream.truncate(100_000_001)
    result = _packaging(tmp_path, output)
    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(tmp_path / "adp_arena_provider_runtime_output.zip") as archive:
        assert set(archive.namelist()) == {RESULT, *retained, "cells/cell/worker/policy-input.png"}
        assert archive.getinfo("cells/cell/worker/policy-input.png").file_size == 100_000_001


@pytest.mark.parametrize("defect", ["missing_g1_result", "symlink", "too_large", "terminal_limit"])
def test_packaging_refuses_incomplete_or_over_budget_archive(tmp_path, defect):
    output = tmp_path / "output"
    output.mkdir()
    (output / RESULT).write_text("{}")
    prefix = ""
    if defect == "missing_g1_result":
        (output / RESULT).unlink()
        (output / "adp009d_native_microcheck.json").write_text("{}")
    elif defect == "symlink":
        (output / "aliased-result.json").symlink_to(output / RESULT)
    elif defect == "terminal_limit":
        assert transport.MAX_TERMINAL_BYTES == 128 * 1024**2
        with (output / RESULT).open("wb") as stream:
            stream.truncate(transport.MAX_TERMINAL_BYTES + 1)
    else:
        # Exercise the actual post-compression budget branch without making a 4 GiB file.
        prefix = ("from pathlib import Path\nfrom types import SimpleNamespace\n"
            "original_stat = Path.stat\n"
            "Path.stat = lambda path, *args, **kwargs: SimpleNamespace(st_size=4*1024**3+1) "
            "if path.name == 'adp_arena_provider_runtime_output.zip' else original_stat(path,*args,**kwargs)\n")
    result = _packaging(tmp_path, output, prefix=prefix)
    assert result.returncode != 0
    assert b"BLUEPRINT_VAST_PROVIDER_OUTPUT_ZIP_WRITTEN" not in result.stdout
    if defect == "too_large":
        assert b"g1_beta_archive_bound_exceeded" in result.stderr
        assert not (tmp_path / "adp_arena_provider_runtime_output.zip").exists()


@pytest.mark.parametrize("valid_zip", [True, False])
def test_recovery_preserves_exact_archive_without_rerunning_policy(tmp_path, monkeypatch, valid_zip):
    _install_identity(monkeypatch, tmp_path)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(RESULT, "{}")
    payload = stream.getvalue() if valid_zip else b"not a native session archive"
    sha = hashlib.sha256(payload).hexdigest()
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if kwargs.get("text"):
            return SimpleNamespace(returncode=0, stdout=f"{len(payload)} {sha}\n")
        kwargs["stdout"].write(payload)
        return SimpleNamespace(returncode=0, stderr=b"")
    monkeypatch.setattr(recovery.subprocess, "run", run)
    path = tmp_path / "retained.zip"
    result = recovery.recover_provider_output_before_teardown(
        connection={"ssh_host": "example.invalid", "ssh_port": 2222}, provider_bundle_kind=KIND,
        output_path=path, attempt_dir=tmp_path / "attempt", expected_size_bytes=len(payload))
    assert result["status"] == ("completed" if valid_zip else "blocked")
    assert path.exists() is valid_zip
    assert len(calls) == 2
    assert all("adp_arena_provider_runtime_output.zip" in cmd[-1] for cmd in calls)
    if valid_zip:
        assert path.read_bytes() == payload


def test_recovery_cannot_exceed_session_output_quote(tmp_path, monkeypatch):
    _install_identity(monkeypatch, tmp_path)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert kwargs.get("text"), "oversized remote archive must not be downloaded"
        return SimpleNamespace(returncode=0, stdout=f"{4*1024**3+1} {'a'*64}\n")
    monkeypatch.setattr(recovery.subprocess, "run", run)
    result = recovery.recover_provider_output_before_teardown(
        connection={"ssh_host": "example.invalid", "ssh_port": 2222}, provider_bundle_kind=KIND,
        output_path=tmp_path / "retained.zip", attempt_dir=tmp_path / "attempt",
        expected_size_bytes=4*1024**3+1, maximum_size_bytes=8*1024**3)
    assert result["status"] == "blocked"
    assert len(calls) == 1
    assert not (tmp_path / "retained.zip").exists()


def test_canonical_allocator_cli_delegates_same_control_recheck(tmp_path, monkeypatch):
    module = ModuleType("blueprint_pipeline.native_g1_beta_paid")
    identity = {"orchestrator_source_commit": "a" * 40}
    def recheck():
        return ["development_only_test"], identity
    monkeypatch.setattr(allocator, "_control_plane_checkout_blockers", recheck)
    calls = []
    def dispatch(args, *, control_recheck):
        calls.append(args)
        assert control_recheck is recheck
        assert control_recheck() == (["development_only_test"], identity)
        assert args.g1_beta_session_authority == str(tmp_path / "authority.json")
        assert args.g1_beta_session_bundle_receipt == str(tmp_path / "bundle.json")
        assert args.execute is False
        return 2
    module.dispatch_g1_beta_allocator_cli = dispatch
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert allocator.main(["gpu-canary", "--provider", "vast", "--probe-kind", "native-g1-beta-screening",
        "--g1-beta-session-authority", str(tmp_path / "authority.json"),
        "--g1-beta-session-bundle-receipt", str(tmp_path / "bundle.json")]) == 2
    assert len(calls) == 1


def test_existing_campaign_arguments_keep_their_values(tmp_path, monkeypatch):
    flags = ("bundle-receipt", "manipulation-packet", "movement-packet", "book-handoff",
        "movement-handoff", "navigation-authority", "publisher-source", "runtime-source-receipt")
    monkeypatch.setattr(allocator, "_control_plane_checkout_blockers", lambda: ([], {}))
    def dispatch(args, **kwargs):
        for name in flags:
            assert getattr(args, "g1_campaign_" + name.replace("-", "_")) == str(tmp_path / name)
        assert args.g1_campaign_rights_review == ["review-one", "review-two"]
        return {"status": "dry_run_ready"}
    monkeypatch.setattr(allocator, "dispatch_g1_paid_campaign", dispatch)
    argv = ["gpu-canary", "--provider", "vast", "--probe-kind", allocator.NATIVE_G1_DEVELOPMENT_CAMPAIGN_PROBE_KIND]
    for name in flags:
        argv.extend(["--g1-campaign-" + name, str(tmp_path / name)])
    argv.extend(["--g1-campaign-rights-review", "review-one", "--g1-campaign-rights-review", "review-two"])
    assert allocator.main(argv) == 0


def _terminal(status="blocked", queried=False):
    result = {"schema_version": transport.RESULT_SCHEMA, "status": status,
        "candidate_id": "native_g1_gr00t_n16", "candidate_policy_queried": queried,
        "episodes": [{"cell_id": "cell-01", "status": status}],
        "blockers": ["bootstrap_failed"] if status == "blocked" else [],
        "claim_ceiling": "development_only", "ranking_eligible": False,
        "qualification_permitted": False, "public_media_publication_authorized": False,
        "model_redistribution_authorized": False, "physical_outcome_claimed": False,
        **{key: "sha256:" + "a" * 64 for key in ("authority_digest", "runtime_inputs_digest",
            "plan_digest", "configuration_digest", "admission_digest", "runtime_identity_digest",
            "scenario_plan_digest", "task_success_contract_digest")}}
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


@pytest.mark.parametrize("status,queried", [("blocked", False), ("blocked", True), ("completed", True)])
def test_typed_transport_retains_bootstrap_and_policy_failures(status, queried):
    result = _terminal(status, queried)
    manifest = {"candidate_ids": [result["candidate_id"]], "cell_ids": ["cell-01"],
        **{key: result[key] for key in ("authority_digest", "runtime_inputs_digest", "task_success_contract_digest")}}
    assert arena._provider_execution_transport_completed(result, provider_bundle_kind=KIND,
        expected_output_filename=RESULT, bundle=manifest, result_path="/development-only/result.json")
    assert not arena._provider_execution_transport_completed(result, provider_bundle_kind=KIND,
        expected_output_filename="adp_arena_native_canary.json", bundle=manifest, result_path=None)
    assert not arena._provider_execution_transport_completed(result, provider_bundle_kind="native_task_arena",
        expected_output_filename=RESULT, bundle=manifest, result_path="/development-only/result.json")
    for key in ("authority_digest", "runtime_inputs_digest", "task_success_contract_digest"):
        assert not arena._provider_execution_transport_completed(result, provider_bundle_kind=KIND,
            expected_output_filename=RESULT, bundle={**manifest, key: "sha256:" + "b" * 64}, result_path=None)
    assert not transport.transport_completed(result, manifest={**manifest, "cell_ids": ["other-cell"]})


@pytest.mark.parametrize("defect", ["scope", "digest", "query_type", "duplicate_cell", "empty", "incomplete"])
def test_typed_result_cannot_borrow_success_or_scope(defect):
    result = _terminal()
    if defect == "scope":
        result["physical_outcome_claimed"] = True
    elif defect == "digest":
        result["result_digest"] = "sha256:" + "f" * 64
    elif defect == "query_type":
        result["candidate_policy_queried"] = 0
    elif defect == "duplicate_cell":
        result["episodes"] *= 2
    elif defect == "empty":
        result["episodes"] = []
    else:
        result["status"] = "completed"
    if defect != "digest":
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    assert not transport.transport_completed(result)


def test_shared_output_scanner_prefers_blocked_g1_terminal_over_completed_child(tmp_path):
    result = _terminal()
    path = tmp_path / "output.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(RESULT, json.dumps(result))
        archive.writestr("cells/0/native_task_arena_policy_result.v1.json", json.dumps({"status": "completed"}))
    observed = vast._inspect_provider_runtime_output_zip(path, video_extract_dir=None, expected_video_count=0)
    assert observed["runtime_result_present"] is True
    assert observed["runtime_result_member"] == RESULT
    assert observed["runtime_result_status"] == "blocked"
    assert observed["runtime_result_blockers"] == ["bootstrap_failed"]


@pytest.mark.parametrize("defect", ["duplicate", "symlink", "compressed_limit", "expanded_limit", "terminal_limit"])
def test_extraction_refuses_invalid_or_over_budget_g1_archive(tmp_path, monkeypatch, defect):
    path = tmp_path / "output.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(RESULT, json.dumps(_terminal()))
        if defect == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr(RESULT, "{}")
        elif defect == "symlink":
            info = zipfile.ZipInfo("link")
            info.external_attr = 0o120777 << 16
            archive.writestr(info, "some-file")
        elif defect == "expanded_limit":
            archive.writestr("trace.json", b"x" * 5000, compress_type=zipfile.ZIP_DEFLATED)
    if defect == "compressed_limit":
        monkeypatch.setattr(transport, "MAX_OUTPUT_BYTES", path.stat().st_size - 1)
    elif defect == "expanded_limit":
        monkeypatch.setattr(transport, "MAX_OUTPUT_BYTES", path.stat().st_size + 1)
    elif defect == "terminal_limit":
        monkeypatch.setattr(transport, "MAX_TERMINAL_BYTES", 1)
    target = tmp_path / "immutable"
    observed = arena._extract_provider_output(path, target, result_name=RESULT, blocker_prefix="g1_beta")
    assert observed["status"] == "blocked"
    assert not target.exists()
    assert "g1_beta_output_" in " ".join(observed["blockers"])
