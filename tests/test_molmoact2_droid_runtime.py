"""Exact DROID runtime wiring with fake physics and no paid/provider calls."""
from copy import deepcopy
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline import molmoact2_droid_runtime as runtime
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_beta_policy_screening_launch import _plan
from tests.test_beta_robolab_runtime import stock_robolab  # noqa: F401


def test_automatic_droid_candidate_builds_exact_bundle_and_canonical_transfer_quote(tmp_path, monkeypatch):
    from blueprint_pipeline.molmoact2_droid_bundle import transfer_byte_budget
    from blueprint_pipeline.policy_canary_session_bundle_identity import validate_policy_canary_provider_manifest
    from blueprint_pipeline.team_candidate_policy_client import canary_policy_client
    from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight
    from blueprint_pipeline import native_task_arena_vast as transport
    plan, _ = _plan(tmp_path, monkeypatch, scene_half="team_scene")
    candidate = next(row for row in plan["candidates"] if row["candidate_id"] == runtime.CANDIDATE_ID)
    assert candidate["status"] == "selected", candidate["blockers"]
    spec = launch._read(launch._bound_path(candidate["execution_spec"]["policy_execution_spec"]))
    assert spec["weights_identity_verified"] is False
    assert spec["candidate_rights_binding"]["source_license"] == "Apache-2.0"
    assert spec["candidate_rights_binding"]["hf_license_metadata_present"] is False
    client = canary_policy_client(spec)
    assert client.action_chunk_rows == 15 and client.open_loop_horizon == 8
    assert client.evidence_summary()["execution_binding_digest"] == spec["execution_spec_digest"]
    prepared = launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt")
    bundle = launch._read(Path(prepared["bundle_receipt"]["path"]))
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        manifest = validate_policy_canary_provider_manifest(json.loads(archive.read("provider_runtime/adp_arena_provider_manifest.json")))
        assert manifest["team_policy_interface_id"] == runtime.INTERFACE_ID
        assert "team_credential_env" not in manifest and manifest["minimum_gpu_memory_gb"] == 48
        assert "provider_runtime/molmoact2-droid-dependencies.v1.json" in archive.namelist()
        assert "provider_runtime/blueprint_pipeline/_catalog_data/molmoact2_droid_release.v1.json" in archive.namelist()
        assert b"molmoact2_droid_provisioning" in archive.read("provider_runtime/adp009d_policy_provisioning.molmoact2_droid.sh")
        assert "provider_runtime/molmoact2-yam-dependencies.v1.json" not in archive.namelist()
    result = _blueprint_bundle_preflight(job_dir=tmp_path / "preflight", generated_at="fixture",
        enable_blueprint_bundle=True, enable_isaac_smoke=True, provider_bundle_kind="native_task_arena_policy_canary_session",
        bundle_path=Path(bundle["bundle_path"]), provider_bundle_url="https://storage.example.com/bundle.zip",
        provider_output_put_url="https://storage.example.com/output.zip")
    assert result["status"] == "passed", result.get("blockers")
    download, upload = transfer_byte_budget(bundle)
    assert download > 21_781_401_931 + 4_045_566_007 + 64 * 1024**3
    observed = []
    monkeypatch.setenv("BLUEPRINT_POLICY_CANARY_OUTPUT_DELIVERY", "download")
    monkeypatch.setattr(transport, "run_arena_native_control_vast", lambda **kwargs: observed.append(kwargs) or {"status": "dry_run_ready"})
    authority = launch._read(Path(prepared["authority"]["path"]))
    transport.run_native_task_arena_policy_canary_session_vast(job_dir=tmp_path / "transport", prepared_bundle=bundle,
        session_authority=authority, paid_resource_admission_grant=None, execute=False,
        hard_cap_usd=float(authority["hard_cap_usd"]), hard_ttl_seconds=authority["hard_ttl_seconds"])
    assert observed[0]["expected_provider_download_bytes"] == download
    assert observed[0]["expected_provider_upload_bytes"] == upload == 8_000_000_000
    assert observed[0]["min_gpu_ram_mb"] == 48_000 and observed[0]["forward_hf_token"] is False
    assert observed[0]["stale_offer_create_retry_limit"] == 0 and observed[0]["require_independent_watchdog"]
    assert not (tmp_path / "attempt/allocator_launch_intent.json").exists()


@pytest.mark.parametrize("backend", [None, "robolab.v1"])
@pytest.mark.usefixtures("stock_robolab")
def test_real_native_loop_and_robolab_consume_two_camera_droid_arrays(backend):
    from tests.test_adp009d_policy_episode import _Environment, _run
    from tests.test_molmoact2_droid_policy import client_runtime
    client, _, model, _, _ = client_runtime()
    model.action = np.zeros((15, 8), dtype=np.float32)
    model.action[:, 0], model.action[:, 7] = .25, .9
    environment = _Environment()
    result = _run(environment, client, candidate_id=runtime.CANDIDATE_ID, evaluation_backend=backend)
    assert len(model.calls) == 4 and len(environment.steps) == 38
    assert len(result["commanded_actions"]) == 32 and result["candidate_policy_queried"]
    assert result["action_space"] == "molmoact2_droid_absolute_joint_position_plus_absolute_gripper"
    for row in model.calls:
        assert row["external_cam"].shape == row["wrist_cam"].shape == (360, 640, 3)
        assert row["state"].shape == (8,)
    assert client.evidence_summary()["native_execution_validated"] is False


@pytest.mark.parametrize("fault", ["rights", "checkpoint", "endpoint", "configuration"])
def test_exact_droid_binding_rejects_resealed_mismatch(tmp_path, monkeypatch, fault):
    plan, _ = _plan(tmp_path, monkeypatch)
    candidate = next(row for row in plan["candidates"] if row["candidate_id"] == runtime.CANDIDATE_ID)
    spec = deepcopy(launch._read(launch._bound_path(candidate["execution_spec"]["policy_execution_spec"])))
    if fault == "rights":
        spec["candidate_rights_binding"]["source_license"] = "unverified"
        spec["candidate_rights_binding"]["rights_receipt_digest"] = canonical_digest(spec["candidate_rights_binding"], digest_field="rights_receipt_digest")
    elif fault == "checkpoint":
        spec["checkpoint_digest"] = "sha256:" + "e" * 64
    elif fault == "endpoint":
        spec["policy_endpoint"]["host"] = "remote.example"
    else:
        spec["policy_spec"]["binding"]["configuration_digest"] = "sha256:" + "e" * 64
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    with pytest.raises(ValueError, match="execution_binding_invalid"):
        runtime.validate_execution_spec(spec)
